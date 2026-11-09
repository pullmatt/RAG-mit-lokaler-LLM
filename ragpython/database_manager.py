from pathlib import Path

import chromadb
import ollama

from config import Config


class DatabaseManager:
    """Persistente Chroma-Datenbank für Text-RAG und multimodalen Vision-Cache."""

    def __init__(self, config=None):
        self.config = config or Config

        self.config.DATABASE_DIR.mkdir(parents=True, exist_ok=True)
        self.config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.config.CHROMA_DIR)
        )

        self.collection = self.client.get_or_create_collection(
            name=self.config.COLLECTION_NAME
        )

        # None bedeutet Legacy-/Globale Suche. Eine leere Menge bedeutet,
        # dass für den aktiven Chat keine Dokumente freigegeben sind.
        self.allowed_source_keys = None

        # Separater Vektorraum: Vision-Analysen haben dieselbe Embedding-
        # Dimension wie der Text-RAG, dürfen aber die Textsuche nicht stören.
        self.vision_collection = self.client.get_or_create_collection(
            name=self.config.VISION_COLLECTION_NAME
        )

    # =========================================================
    # Text-RAG
    # =========================================================

    def count(self):
        return self.collection.count()

    def add_documents_batch(self, documents):
        if not documents:
            return

        self.collection.upsert(
            ids=[str(item["document_id"]) for item in documents],
            documents=[str(item["text"]) for item in documents],
            metadatas=[
                {
                    "source": str(item["source"]),
                    "page": int(item["page"]),
                    "has_visual": bool(item.get("has_visual", False)),
                    "has_ocr": bool(item.get("has_ocr", False)),
                    "image_path": str(item.get("image_path", "")),
                    "is_cache": bool(item.get("is_cache", False)),
                    "content_type": "text",
                }
                for item in documents
            ],
            embeddings=[item["embedding"] for item in documents],
        )

    @staticmethod
    def source_key(source, is_cache=False):
        """Eindeutiger Dokumentbezeichner; trennt permanenten RAG und Cache."""
        prefix = "cache::" if bool(is_cache) else "document::"
        return prefix + str(source or "")

    @staticmethod
    def empty_results():
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    def _scope_where(self, is_cache=None):
        """Erzeugt einen Chroma-Filter, ohne bei aktivem Scope global auszuweichen."""
        scope = self.allowed_source_keys
        if scope is None:
            return {"is_cache": bool(is_cache)} if is_cache is not None else None
        if not scope:
            return {"$and": [{"source": {"$in": ["__NO_DOCUMENT_ALLOWED__"]}}]}

        groups = {False: [], True: []}
        for key in scope:
            key = str(key)
            if key.startswith("cache::"):
                groups[True].append(key[len("cache::"):])
            elif key.startswith("document::"):
                groups[False].append(key[len("document::"):])
            else:
                # Keine implizite Freigabe für untypisierte Keys.
                continue

        branches = []
        for cache_flag, sources in groups.items():
            if is_cache is not None and cache_flag != bool(is_cache):
                continue
            if not sources:
                continue
            conditions = [{"is_cache": cache_flag}, {"source": {"$in": sorted(set(sources))}}]
            branches.append({"$and": conditions})

        if not branches:
            return {"$and": [{"source": {"$in": ["__NO_DOCUMENT_ALLOWED__"]}}]}
        if len(branches) == 1:
            return branches[0]
        return {"$or": branches}

    def search(self, embedding, n_results, is_cache=None):
        kwargs = {
            "query_embeddings": [embedding],
            "n_results": max(1, int(n_results)),
            "include": ["documents", "metadatas", "distances"],
        }
        where = self._scope_where(is_cache=is_cache)
        if where is not None:
            kwargs["where"] = where

        # Für leere/unmögliche Scopes Chroma gar nicht erst aufrufen.
        if self.allowed_source_keys is not None and not self.allowed_source_keys:
            return self.empty_results()
        try:
            return self.collection.query(**kwargs)
        except Exception:
            # Ein fehlgeschlagener Projekt-/Chatfilter darf niemals zu einer
            # globalen Suche führen. Der Legacy-Fallback ist nur ohne aktiven
            # Scope zulässig.
            if "where" in kwargs and self.allowed_source_keys is None:
                kwargs.pop("where", None)
                return self.collection.query(**kwargs)
            raise

    def delete_source(self, source):
        try:
            self.collection.delete(where={"source": str(source)})
        except Exception:
            data = self.collection.get(include=["metadatas"])
            ids = []
            for doc_id, meta in zip(
                data.get("ids", []),
                data.get("metadatas", []) or [],
            ):
                if (meta or {}).get("source") == str(source):
                    ids.append(doc_id)
            if ids:
                self.collection.delete(ids=ids)

        self.delete_vision_source(source)

    def delete_collection(self):
        try:
            self.client.delete_collection(self.config.COLLECTION_NAME)
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name=self.config.COLLECTION_NAME
        )

        # Ein vollständiger Text-Neuaufbau macht auch alte Vision-Seitenpfade
        # ungültig. Daher den separaten Vision-Cache ebenfalls sauber neu anlegen.
        try:
            self.client.delete_collection(self.config.VISION_COLLECTION_NAME)
        except Exception:
            pass

        self.vision_collection = self.client.get_or_create_collection(
            name=self.config.VISION_COLLECTION_NAME
        )

    # =========================================================
    # Vision-Cache
    # =========================================================

    def migrate_legacy_vision_entries(self):
        """Migriert alte Vision-Einträge aus der Text-Collection.

        Frühere Versionen konnten Vision-Analysen in der normalen Collection
        speichern. Diese Einträge werden anhand von content_type=vision bzw.
        vision_hash erkannt, in die separate Vision-Collection übernommen,
        mit einem echten Embedding versehen und anschließend aus der
        Text-Collection entfernt. Ein normaler Text-Chunk wird dabei nicht
        verändert.
        """
        data = self.collection.get(
            include=["documents", "metadatas"]
        )

        ids = data.get("ids", []) or []
        docs = data.get("documents", []) or []
        metas = data.get("metadatas", []) or []

        legacy = []
        for idx, doc_id in enumerate(ids):
            meta = dict(metas[idx] or {}) if idx < len(metas) else {}
            content_type = str(meta.get("content_type", "")).lower()
            if content_type != "vision" and not meta.get("vision_hash"):
                continue
            if idx >= len(docs):
                continue
            if not meta.get("source") or meta.get("page") is None:
                continue
            legacy.append((doc_id, docs[idx], meta))

        if not legacy:
            return

        print(f"[VISION] Migriere {len(legacy)} alte Vision-Einträge...")

        for doc_id, doc, meta in legacy:
            source = str(meta["source"])
            page = int(meta["page"])
            vision_hash = str(meta.get("vision_hash", "legacy"))
            new_id = self._vision_id(source, page, vision_hash)

            # Nur übernehmen, wenn noch kein Eintrag existiert.
            existing = self.vision_collection.get(ids=[new_id], include=["documents"])
            if not existing.get("ids"):
                embedding = ollama.embed(
                    model=self.config.EMBEDDING_MODEL,
                    input=[f"{source} Seite {page}\n{doc}"],
                )["embeddings"][0]

                self.vision_collection.upsert(
                    ids=[new_id],
                    documents=[str(doc)],
                    metadatas=[{
                        "source": source,
                        "page": page,
                        "vision_hash": vision_hash,
                        "is_cache": bool(meta.get("is_cache", False)),
                        "content_type": "vision",
                    }],
                    embeddings=[embedding],
                )

            self.collection.delete(ids=[doc_id])

        print("[VISION] Legacy-Vision-Einträge erfolgreich migriert.")

    def get_vision_analysis(self, source, page, vision_hash):
        try:
            data = self.vision_collection.get(
                where={
                    "$and": [
                        {"source": str(source)},
                        {"page": int(page)},
                        {"vision_hash": str(vision_hash)},
                    ]
                },
                include=["documents", "metadatas"],
            )
        except Exception:
            return None

        docs = data.get("documents", []) or []
        return docs[0] if docs else None

    def add_vision_analysis(
        self,
        source,
        page,
        vision_hash,
        analysis,
        embedding=None,
        is_cache=False,
    ):
        doc_id = self._vision_id(source, page, vision_hash)

        metadata = {
            "source": str(source),
            "page": int(page),
            "vision_hash": str(vision_hash),
            "is_cache": bool(is_cache),
            "content_type": "vision",
        }

        kwargs = {
            "ids": [doc_id],
            "documents": [str(analysis)],
            "metadatas": [metadata],
        }

        if embedding is not None:
            kwargs["embeddings"] = [embedding]

        self.vision_collection.upsert(**kwargs)

    def search_vision(self, embedding, n_results, is_cache=None):
        kwargs = {
            "query_embeddings": [embedding],
            "n_results": max(1, int(n_results)),
            "include": ["documents", "metadatas", "distances"],
        }
        where = self._scope_where(is_cache=is_cache)
        if where is not None:
            kwargs["where"] = where
        if self.allowed_source_keys is not None and not self.allowed_source_keys:
            return self.empty_results()
        return self.vision_collection.query(**kwargs)

    def delete_vision_analysis(self, source, page, vision_hash):
        try:
            self.vision_collection.delete(
                ids=[self._vision_id(source, page, vision_hash)]
            )
        except Exception:
            pass

    def delete_vision_source(self, source):
        try:
            self.vision_collection.delete(
                where={"source": str(source)}
            )
        except Exception:
            try:
                data = self.vision_collection.get(include=["metadatas"])
                ids = []
                for doc_id, meta in zip(
                    data.get("ids", []),
                    data.get("metadatas", []) or [],
                ):
                    if (meta or {}).get("source") == str(source):
                        ids.append(doc_id)
                if ids:
                    self.vision_collection.delete(ids=ids)
            except Exception:
                pass

    @staticmethod
    def _vision_id(source, page, vision_hash):
        import hashlib

        raw = f"{source}|{page}|{vision_hash}"
        return "vision_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # =========================================================
    # Migration bestehender Vision-Einträge
    # =========================================================

    def ensure_vision_embeddings(self, embedding_model, batch_size=32):
        """Gibt alten Vision-Einträgen nachträglich Embeddings.

        Frühere Versionen speicherten die 17 Vision-Einträge ohne Vektor.
        Sie werden hier einmalig nachgerüstet. Bereits vorhandene Embeddings
        werden nicht neu berechnet.
        """
        try:
            data = self.vision_collection.get(
                include=["documents", "metadatas", "embeddings"]
            )
        except Exception:
            return

        ids = data.get("ids", []) or []
        docs = data.get("documents", []) or []
        metas = data.get("metadatas", []) or []
        embeddings = data.get("embeddings", None)

        if not ids or not docs:
            return

        missing = []
        for idx, doc_id in enumerate(ids):
            existing = None
            if embeddings is not None and idx < len(embeddings):
                existing = embeddings[idx]
            if existing is None or len(existing) == 0:
                missing.append(idx)

        if not missing:
            return

        print(f"[VISION] Nachrüstung von {len(missing)} Vision-Embeddings...")

        for start in range(0, len(missing), batch_size):
            batch_indices = missing[start:start + batch_size]
            batch_docs = [
                f"{metas[i].get('source', '')} Seite {metas[i].get('page', '')}\n{docs[i]}"
                for i in batch_indices
            ]

            response = ollama.embed(
                model=embedding_model,
                input=batch_docs,
            )
            batch_embeddings = response["embeddings"]

            for i, vector in zip(batch_indices, batch_embeddings):
                self.vision_collection.upsert(
                    ids=[ids[i]],
                    documents=[docs[i]],
                    metadatas=[metas[i]],
                    embeddings=[vector],
                )

        print("[VISION] Vision-Embeddings nachgerüstet.")
