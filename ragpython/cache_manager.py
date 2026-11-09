from pathlib import Path
import hashlib
import json
import shutil

from config import Config


class CacheManager:
    """Verwaltet temporäre PDFs getrennt vom permanenten Chemie-RAG.

    Cache-PDFs werden in derselben Chroma-Collection indexiert, aber immer
    über das Metadatenfeld is_cache=True getrennt. Die Klausur ist dadurch
    Aufgabenquelle, nicht Ersatz für die permanenten Fachquellen.
    """

    def __init__(self, rag):
        self.rag = rag
        self.config = rag.config
        self.root = self.config.CACHE_DOCUMENTS_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self.hash_file = self.config.DATABASE_DIR / "cache_document_hashes.json"
        self.hashes = self._load_hashes()

    def _load_hashes(self):
        try:
            data = json.loads(self.hash_file.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return {}
            # Ältere Versionen speicherten Windows-Backslashes. Vereinheitliche
            # Hash-Schlüssel, damit dieselbe PDF nicht als mehrere Quellen gilt.
            normalized = {}
            for source, digest in data.items():
                key = str(source).replace("\\", "/")
                normalized[key] = digest
            return normalized
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_hashes(self):
        self.hash_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.hash_file.with_suffix(self.hash_file.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.hashes, handle, ensure_ascii=False, indent=2)
            handle.flush()
        tmp.replace(self.hash_file)

    def _indexed_cache_sources(self):
        """Gibt die Cache-Quellen zurück, für die Chroma tatsächlich Einträge hat."""
        try:
            data = self.rag.database.collection.get(
                where={"is_cache": True}, include=["metadatas"]
            )
        except Exception as error:
            print(f"[CACHE] Vorhandene Indexeinträge konnten nicht geprüft werden: {error}")
            return set()
        return {
            str((meta or {}).get("source", "")).replace("\\", "/")
            for meta in data.get("metadatas", []) or []
            if (meta or {}).get("source")
        }

    @staticmethod
    def _file_hash(path):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _source_for(path):
        # Der gleiche relative Quellname wird auch in den Chroma-Metadaten
        # und für die Seitenbildpfade verwendet.
        return path.relative_to(self.config.PROJECT_ROOT).as_posix()

    def _cleanup_legacy_source_aliases(self, canonical_source):
        """Entfernt alte Cache-Quellen, die sich nur durch Slash-Schreibweise unterscheiden.

        Rückgabe True, wenn Aliase gefunden wurden und die kanonische Quelle deshalb
        vorsichtshalber neu indexiert werden sollte.
        """
        try:
            data = self.rag.database.collection.get(
                where={"is_cache": True}, include=["metadatas"]
            )
        except Exception:
            return False
        aliases = set()
        canonical_norm = str(canonical_source).replace("\\", "/").casefold()
        for meta in data.get("metadatas", []) or []:
            source = str((meta or {}).get("source", ""))
            if not source:
                continue
            if source.replace("\\", "/").casefold() == canonical_norm and source != canonical_source:
                aliases.add(source)
        for alias in aliases:
            print(f"[CACHE] Pfad-Alias bereinigen: {alias} -> {canonical_source}")
            self.rag.database.delete_source(alias)
            self._remove_page_images(alias)
        return bool(aliases)

    def _remove_page_images(self, source):
        source_id = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
        folder = self.config.PAGE_IMAGES_DIR / source_id
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)

    def sync(self):
        """Indexiert neue/geänderte PDFs und entfernt gelöschte Cache-Dateien."""
        pdfs = sorted(
            p for p in self.root.rglob("*")
            if p.is_file() and p.suffix.lower() == ".pdf" and not p.name.startswith("._")
        )
        current = {}
        indexed_sources = self._indexed_cache_sources()
        print(f"[CACHE] PDFs gefunden: {len(pdfs)}")

        for pdf in pdfs:
            try:
                source = self._source_for(pdf)
                digest = self._file_hash(pdf)
            except Exception as error:
                print(f"[CACHE] Datei übersprungen: {pdf} ({error})")
                continue

            current[source] = digest
            had_aliases = self._cleanup_legacy_source_aliases(source)
            old_digest = self.hashes.get(source)
            source_norm = source.replace("\\", "/")
            if old_digest == digest and source_norm in indexed_sources and not had_aliases:
                continue
            if old_digest == digest and source_norm not in indexed_sources:
                print(f"[CACHE] REPAIR: {source} hat einen Hash, aber keine Chroma-Einträge.")
            if had_aliases:
                # Bei Alias-Funden alle kanonischen Chunks ebenfalls ersetzen, damit
                # keine Mischung aus alten und neuen Metadaten/Seitenbildern bleibt.
                old_digest = None

            if old_digest is not None:
                print(f"[CACHE] UPDATE/REPAIR: {source}")
                self.rag.database.delete_source(source)
                self._remove_page_images(source)
            else:
                print(f"[CACHE] NEW/RECONCILE: {source}")
                # Falls eine ältere Cache-Version diese Datei bereits indexiert
                # hatte, aber noch keine Hash-Datei vorliegt, entfernen wir alte
                # Chunks vor dem Upsert. Sonst könnten z. B. gelöschte Seiten
                # als veraltete Treffer erhalten bleiben.
                self.rag.database.delete_source(source)
                self._remove_page_images(source)

            try:
                self.rag._index_pdf(pdf, source, is_cache=True)
                self.hashes[source] = digest
                indexed_sources.add(source.replace("\\", "/"))
                self._save_hashes()
            except Exception as error:
                print(f"[CACHE] Indexierung fehlgeschlagen: {source}\n        {error}")

        for stale_source in sorted(set(self.hashes) - set(current)):
            print(f"[CACHE] ENTFERNT: {stale_source}")
            self.rag.database.delete_source(stale_source)
            self._remove_page_images(stale_source)
            self.hashes.pop(stale_source, None)

        self._save_hashes()

    def has_documents(self):
        try:
            data = self.rag.database.collection.get(
                where={"is_cache": True}, include=["metadatas"]
            )
            return bool(data.get("ids"))
        except Exception:
            return bool(self.hashes)

    def get_all_results(self):
        """Liefert alle indexierten Cache-Chunks, nicht nur Top-k-Treffer."""
        try:
            data = self.rag.database.collection.get(
                where={"is_cache": True},
                include=["documents", "metadatas"],
            )
        except Exception as error:
            print(f"[CACHE] Vollständiges Lesen nicht möglich: {error}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        rows = []
        ids = data.get("ids", []) or []
        docs = data.get("documents", []) or []
        metas = data.get("metadatas", []) or []
        for i, doc_id in enumerate(ids):
            if i >= len(docs) or i >= len(metas):
                continue
            meta = dict(metas[i] or {})
            if not meta.get("is_cache", False):
                continue
            # Cache-Aufgaben und Klausurseiten bleiben innerhalb der Quellen,
            # die für den aktiven Chat freigegeben wurden.
            if not self.rag._is_source_allowed(meta):
                continue
            rows.append((str(meta.get("source", "")), int(meta.get("page", 0) or 0), str(doc_id), docs[i], meta))
        rows.sort(key=lambda row: (row[0].casefold(), row[1], row[2]))
        return {
            "documents": [[row[3] for row in rows]],
            "metadatas": [[row[4] for row in rows]],
            "distances": [[0.0 for _ in rows]],
        }

    def get_summary_results(self):
        return self.get_all_results()
