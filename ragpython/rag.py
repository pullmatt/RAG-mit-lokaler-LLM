from pathlib import Path
import hashlib
import json
import shutil
import re
import unicodedata
from datetime import datetime

import ollama
import pymupdf

from config import Config
from database_manager import DatabaseManager
from cache_manager import CacheManager


class ChemistryRAG:

    # =========================================================
    # Suchbegriffe / Stopwords
    # =========================================================
    # Zentral definiert, damit sowohl _search_terms() als auch
    # _lexical_search() dieselbe Liste verwenden können.
    SEARCH_STOPWORDS = {
        "der", "die", "das", "den", "dem", "des",
        "ein", "eine", "einer", "einem", "einen",
        "und", "oder", "aber", "wie", "was", "warum",
        "wieso", "weshalb", "ist", "sind", "sich", "zu",
        "von", "für", "mit", "im", "in", "auf", "an",
        "aus", "bei", "als", "auch", "noch", "über",
        "welche", "welcher", "welches",
        "wichtig", "wichtigsten", "wichtige", "wichtigen",
        "relevant", "relevante", "relevanten", "relevantesten",
        "funktioniert", "funktionieren",
        "kann", "können", "bitte",
        "schritt", "schritte",
        "inhalt", "inhalte",
        "erklären", "erklärt", "erklärung",
        "aufbau", "aufgebaut",
        "ablauf", "abläufe",
    }


    def __init__(self, config=None):

        self.config = config or Config

        # -----------------------------------------------------
        # Verzeichnisse
        # -----------------------------------------------------

        self.config.DOCUMENTS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        self.config.CACHE_DOCUMENTS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        self.config.PAGE_IMAGES_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        # -----------------------------------------------------
        # Datenbank
        # -----------------------------------------------------

        self.database = DatabaseManager(self.config)
        print(f"[DB] Persistenter Projektindex: {self.config.CHROMA_DIR}")
        try:
            print(f"[DB] Bereits vorhandene Text-Chunks: {self.database.count()}")
        except Exception as error:
            print(f"[DB] Bestehender Index kann nicht gezählt werden: {error}")

        # Alte Vision-Einträge aus früheren Versionen lagen teilweise noch
        # in der normalen Text-Collection. Sie werden einmalig in die
        # separate Vision-Collection migriert und dort eingebettet.
        try:
            self.database.migrate_legacy_vision_entries()
        except Exception as error:
            print(f"[VISION] Legacy-Migration übersprungen: {error}")

        try:
            self.database.ensure_vision_embeddings(
                self.config.EMBEDDING_MODEL,
                batch_size=32,
            )
        except Exception as error:
            print(f"[VISION] Embedding-Migration übersprungen: {error}")

        # -----------------------------------------------------
        # Hash-Dateien
        # -----------------------------------------------------

        self.hash_file = (
            self.config.DATABASE_DIR
            / "document_hashes.json"
        )

        self.version_file = (
            self.config.DATABASE_DIR
            / "index_version.json"
        )

        self.document_hashes = (
            self._load_json(
                self.hash_file,
                {}
            )
        )

        # -----------------------------------------------------
        # Chat
        # -----------------------------------------------------

        self.conversation_history = []
        self.active_source_keys = None  # None = globale Legacy-Suche; set() = keine freigegebenen Quellen
        self.active_memory_summary = ""
        self._last_chat_done_reason = None
        # Offene Rückfrage zur Dokumentauswahl; verhindert stilles Fallback auf falsche Datei.
        self._pending_target_clarification = None
        # Textseiten für wiederholte Schnellsuchen im Arbeitsspeicher cachen.
        # Der Schlüssel enthält Zeitstempel und Dateigröße; geänderte PDFs werden neu gelesen.
        self._quick_pdf_text_cache = {}

        # -----------------------------------------------------
        # OCR
        # -----------------------------------------------------

        self.ocr_available = (
            self._check_ocr()
        )

        if (
            self.config.OCR_ENABLED
            and not self.ocr_available
        ):

            print(
                "[OCR] Tesseract nicht gefunden – "
                "OCR wird deaktiviert."
            )

        # -----------------------------------------------------
        # Cache
        # -----------------------------------------------------

        self.cache = CacheManager(
            self
        )

    # =========================================================
    # JSON
    # =========================================================

    @staticmethod
    def _load_json(
        path,
        default
    ):

        if not path.exists():
            return default

        try:

            with open(
                path,
                "r",
                encoding="utf-8"
            ) as file:

                return json.load(file)

        except Exception:

            return default

    def _save_json(
        self,
        path,
        data
    ):
        """Schreibt Metadaten atomar, damit ein Abbruch keinen leeren Indexmarker erzeugt."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
            file.flush()
        temporary.replace(path)

    # =========================================================
    # OCR
    # =========================================================

    def _check_ocr(self):

        if not self.config.OCR_ENABLED:
            return False

        try:

            import pytesseract

            pytesseract.get_tesseract_version()

            return True

        except Exception:

            return False

    def _ocr(
        self,
        page
    ):

        if not self.ocr_available:
            return ""

        try:

            textpage = (
                page.get_textpage_ocr(
                    language=
                        self.config.OCR_LANGUAGE,

                    dpi=
                        self.config.OCR_DPI,

                    full=True,
                )
            )

            return (
                textpage
                .extractText()
                .strip()
            )

        except Exception:

            print(
                "[OCR] OCR fehlgeschlagen."
            )

            return ""

    # =========================================================
    # Hash
    # =========================================================

    def _hash_file(
        self,
        path
    ):

        sha = hashlib.sha256()

        with open(
            path,
            "rb"
        ) as file:

            for block in iter(
                lambda:
                    file.read(
                        1024 * 1024
                    ),
                b""
            ):

                sha.update(block)

        return sha.hexdigest()

    def _hash_bytes(
        self,
        data
    ):

        return hashlib.sha256(
            data
        ).hexdigest()

    # =========================================================
    # Seitenbild
    # =========================================================

    def _render_page(
        self,
        page
    ):

        return page.get_pixmap(
            dpi=self.config.VISION_DPI,
            alpha=False,
        ).tobytes(
            "png"
        )

    def _needs_visual_page(
        self,
        page,
        text
    ):
        """Decide whether a page image is actually needed.

        Text-rich PDF pages stay on the normal text/RAG path. In particular,
        vector drawings (common for equations, lines and typesetting) must not
        trigger Vision by themselves. Vision is reserved for pages with too
        little extractable text, where OCR/visual inspection can add value.
        """
        if not self.config.VISION_ENABLED:
            return False

        # A text-rich page is handled as text even if the PDF contains logos,
        # embedded decorative images, or many vector drawing primitives.
        if len((text or "").strip()) >= self.config.OCR_TEXT_THRESHOLD:
            return False

        # Low/no text is the only automatic trigger. Rendering and actual
        # Vision inference remain lazy and happen only if retrieval needs it.
        return True

    def _image_path(
        self,
        source,
        page
    ):

        source_id = hashlib.sha256(
            source.encode(
                "utf-8"
            )
        ).hexdigest()[:16]

        folder = (
            self.config.PAGE_IMAGES_DIR
            / source_id
        )

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

        return (
            folder
            / f"page_{page:04d}.png"
        )

    # =========================================================
    # Chunking
    # =========================================================

    def _chunks(
        self,
        text
    ):

        text = text.strip()

        if not text:
            return []

        chunks = []

        start = 0

        while start < len(text):

            end = (
                start
                + self.config.CHUNK_SIZE
            )

            chunk = (
                text[start:end]
                .strip()
            )

            if chunk:
                chunks.append(
                    chunk
                )

            if end >= len(text):
                break

            start = (
                end
                - self.config.CHUNK_OVERLAP
            )

        return chunks

    # =========================================================
    # Embeddings
    # =========================================================

    def _embedding(
        self,
        text
    ):

        response = ollama.embed(
            model=
                self.config.EMBEDDING_MODEL,

            input=text,
        )

        return response[
            "embeddings"
        ][0]

    def _embeddings_batch(
        self,
        texts
    ):

        if not texts:
            return []

        response = ollama.embed(
            model=
                self.config.EMBEDDING_MODEL,

            input=texts,
        )

        return response[
            "embeddings"
        ]

    # =========================================================
    # Embedding Batch
    # =========================================================

    def _flush_embedding_batch(
        self,
        pending
    ):

        if not pending:
            return

        texts = [
            item["text"]
            for item in pending
        ]

        print(
            f"    [EMBED] "
            f"{len(texts)} Chunks...",
            end="",
            flush=True
        )

        embeddings = (
            self._embeddings_batch(
                texts
            )
        )

        documents = []

        for item, embedding in zip(
            pending,
            embeddings
        ):

            documents.append({
                "document_id":
                    item[
                        "document_id"
                    ],

                "text":
                    item["text"],

                "embedding":
                    embedding,

                "source":
                    item["source"],

                "page":
                    item["page"],

                "has_visual":
                    item[
                        "has_visual"
                    ],

                "has_ocr":
                    item[
                        "has_ocr"
                    ],

                "image_path":
                    item[
                        "image_path"
                    ],

                "is_cache":
                    item.get(
                        "is_cache",
                        False
                    ),
            })

        self.database.add_documents_batch(
            documents
        )

        print(
            " OK"
        )

    # =========================================================
    # PDF Indexierung
    # =========================================================

    def _index_pdf(
        self,
        pdf_path,
        source,
        is_cache=False
    ):

        label = (
            "[CACHE]"
            if is_cache
            else "[INDEX]"
        )

        print(
            f"\n{label} {source}"
        )

        document = pymupdf.open(
            pdf_path
        )

        pending = []

        visual_count = 0

        ocr_count = 0

        chunk_count = 0

        total_pages = len(
            document
        )

        for page_no, page in enumerate(
            document,
            start=1
        ):

            print(
                f"  Seite "
                f"{page_no}/"
                f"{total_pages}",
                end="",
                flush=True
            )

            text = (
                page
                .get_text(
                    "text"
                )
                .strip()
            )

            ocr_text = ""

            if (
                len(text)
                <
                self.config
                    .OCR_TEXT_THRESHOLD
            ):

                ocr_text = (
                    self._ocr(
                        page
                    )
                )

                if ocr_text:
                    ocr_count += 1

            needs_visual = (
                self._needs_visual_page(
                    page,
                    text
                )
            )

            image_rel = ""

            if needs_visual:

                image_path = (
                    self._image_path(
                        source,
                        page_no
                    )
                )

                if not image_path.exists():

                    image_bytes = (
                        self._render_page(
                            page
                        )
                    )

                    image_path.write_bytes(
                        image_bytes
                    )

                image_rel = str(
                    image_path.relative_to(
                        self.config
                            .PROJECT_ROOT
                    )
                )

                visual_count += 1

            parts = []

            if text:

                parts.append(
                    "[PDF-TEXT]\n"
                    + text
                )

            if ocr_text:

                parts.append(
                    "[OCR-TEXT]\n"
                    + ocr_text
                )

            page_text = (
                "\n\n".join(
                    parts
                )
                .strip()
            )

            # -------------------------------------------------
            # Bild-only-Seite
            # -------------------------------------------------

            if (
                not page_text
                and needs_visual
            ):

                page_text = (
                    "[VISUELLE SEITE]\n"
                    "Diese Seite enthält "
                    "visuelle Inhalte, "
                    "die bei Bedarf anhand "
                    "des Originalbildes "
                    "analysiert werden."
                )

            if not page_text:

                print(
                    " → leer"
                )

                continue

            chunks = (
                self._chunks(
                    page_text
                )
            )

            for chunk_no, chunk in enumerate(
                chunks
            ):

                doc_id = (
                    f"{source}"
                    f"__page_{page_no}"
                    f"__chunk_{chunk_no}"
                )

                pending.append({
                    "document_id":
                        doc_id,

                    "text":
                        chunk,

                    "source":
                        source,

                    "page":
                        page_no,

                    "has_visual":
                        needs_visual,

                    "has_ocr":
                        bool(
                            ocr_text
                        ),

                    "image_path":
                        image_rel,

                    "is_cache":
                        is_cache,
                })

                chunk_count += 1

            print(
                f" → "
                f"{len(chunks)} "
                f"Chunks"
            )

            if (
                len(pending)
                >=
                self.config
                    .EMBEDDING_BATCH_SIZE
            ):

                self._flush_embedding_batch(
                    pending
                )

                pending = []

        if pending:

            self._flush_embedding_batch(
                pending
            )

        document.close()

        print(
            f"  Fertig | "
            f"Chunks: {chunk_count} | "
            f"visuelle Seiten: "
            f"{visual_count} | "
            f"OCR: {ocr_count}"
        )

    # =========================================================
    # Projekt-/Chat-Scope
    # =========================================================

    def set_document_scope(self, source_keys=None, memory_summary=""):
        """Setzt den verbindlichen Quelldokument-Scope für den aktiven Chat.

        source_keys nutzt document::<source> bzw. cache::<source>. None ist
        ausschließlich für Legacy-Aufrufer eine globale Suche; die GUI übergibt
        immer eine konkrete Menge, auch wenn diese leer ist.
        """
        self.active_source_keys = None if source_keys is None else set(source_keys)
        self.database.allowed_source_keys = self.active_source_keys
        self.active_memory_summary = str(memory_summary or "").strip()

    def _is_source_allowed(self, metadata):
        if self.active_source_keys is None:
            return True
        meta = dict(metadata or {})
        key = self.database.source_key(
            meta.get("source", ""), meta.get("is_cache", False)
        )
        return key in self.active_source_keys

    def get_indexed_sources(self):
        """Liefert die eindeutig indexierten Quelldateien aus Chroma."""
        data = self.database.collection.get(include=["metadatas"])
        rows = {}
        for meta in data.get("metadatas", []) or []:
            meta = dict(meta or {})
            source = str(meta.get("source", "")).strip()
            if not source:
                continue
            is_cache = bool(meta.get("is_cache", False))
            key = self.database.source_key(source, is_cache)
            item = rows.setdefault(key, {
                "key": key, "source": source, "is_cache": is_cache,
                "pages": set(), "chunks": 0,
            })
            item["chunks"] += 1
            if meta.get("page") is not None:
                item["pages"].add(int(meta.get("page") or 0))
        result = []
        for item in rows.values():
            item["page_count"] = len(item.pop("pages"))
            result.append(item)
        result.sort(key=lambda x: (not x["is_cache"], x["source"].casefold()))
        return result

    # =========================================================
    # Permanente Dokumente
    # =========================================================

    def index_documents(self, force_rebuild=False):
        """Indexiert neue/geänderte PDFs inkrementell.

        Der Start der Anwendung löscht niemals automatisch eine persistente
        Chroma-Collection. Ein vollständiger Neuaufbau muss ausdrücklich mit
        force_rebuild=True angefordert werden.
        """
        current_version = self._load_json(self.version_file, {}).get("version")

        if force_rebuild:
            print("[INDEX] Manueller vollständiger Neuaufbau angefordert.")
            self.database.delete_collection()
            self.document_hashes = {}
            if self.config.PAGE_IMAGES_DIR.exists():
                shutil.rmtree(self.config.PAGE_IMAGES_DIR, ignore_errors=True)
            self.config.PAGE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        elif current_version != self.config.INDEX_VERSION:
            # Ein Versionsmarker fehlt gelegentlich bei alten Installationen oder
            # nach einem abgebrochenen Shutdown. Die Collection bleibt erhalten;
            # identische Dateien werden weiterverwendet und fehlende Chunks repariert.
            if current_version is None:
                print("[INDEX] Kein gültiger Versionsmarker – vorhandene Datenbank bleibt erhalten.")
            else:
                print(
                    f"[INDEX] Index-Version {current_version!r} != "
                    f"{self.config.INDEX_VERSION!r}; kein automatisches Löschen. "
                    "Bestehende Einträge bleiben erhalten."
                )

        self._save_json(self.version_file, {"version": self.config.INDEX_VERSION})

        # Hash-Übereinstimmung allein ist nicht ausreichend: Chroma könnte nach
        # einem abgebrochenen Import oder einer manuellen Bereinigung leer sein.
        # Nur überspringen, wenn der persistente Index den Quellpfad wirklich enthält.
        try:
            indexed_permanent = {
                str(item.get("source", "")).replace("\\", "/")
                for item in self.get_indexed_sources()
                if not item.get("is_cache")
            }
        except Exception as error:
            print(f"[INDEX] Konnte vorhandene Quellen nicht prüfen; es wird sicher weitergearbeitet: {error}")
            indexed_permanent = set()

        pdfs = sorted(
            (pdf for pdf in self.config.DOCUMENTS_DIR.rglob("*")
             if pdf.is_file() and pdf.suffix.casefold() == ".pdf"
             and not pdf.name.startswith("._")),
            key=lambda pdf: str(pdf).casefold(),
        )

        current_sources = set()

        print(
            f"[INDEX] PDFs gefunden: "
            f"{len(pdfs)}"
        )

        for pdf in pdfs:

            source = pdf.relative_to(
                self.config.DOCUMENTS_DIR
            ).as_posix()

            current_sources.add(
                source
            )

            try:

                new_hash = (
                    self._hash_file(
                        pdf
                    )
                )

            except Exception as error:

                print(
                    f"[ERROR] Datei konnte "
                    f"nicht gelesen werden: "
                    f"{source}"
                )

                print(
                    f"        {error}"
                )

                continue

            old_hash = self.document_hashes.get(source)
            source_norm = source.replace("\\", "/")

            if old_hash == new_hash and source_norm in indexed_permanent:
                print(f"[SKIP] {source} (Datei und Chroma-Einträge unverändert)")
                continue

            if old_hash == new_hash and source_norm not in indexed_permanent:
                print(f"[REPAIR] {source}: Hash vorhanden, Chroma-Einträge fehlen – indexiere erneut.")

            if old_hash is not None:

                print(
                    f"[UPDATE] {source}"
                )

                self.database.delete_source(
                    source
                )

            else:

                print(
                    f"[NEW] {source}"
                )

            try:

                self._index_pdf(
                    pdf,
                    source,
                    is_cache=False
                )

            except Exception as error:

                print(
                    f"[ERROR] PDF konnte "
                    f"nicht verarbeitet "
                    f"werden: {source}"
                )

                print(
                    f"        {error}"
                )

                continue

            self.document_hashes[
                source
            ] = new_hash

            self._save_json(
                self.hash_file,
                self.document_hashes
            )

        deleted = (
            set(
                self.document_hashes
            )
            - current_sources
        )

        for source in deleted:

            print(
                f"[DELETE] {source}"
            )

            self.database.delete_source(
                source
            )

            del self.document_hashes[
                source
            ]

        self._save_json(
            self.hash_file,
            self.document_hashes
        )

        print(
            f"[INDEX] Fertig – "
            f"{self.database.count()} "
            f"Chroma-Einträge."
        )

    # =========================================================
    # Automatischer Cache
    # =========================================================

    def sync_cache(
        self
    ):

        self.cache.sync()

    # =========================================================
    # Retrieval
    # =========================================================

    # Retrieval-Konfiguration. Die Werte sind bewusst hier gebündelt,
    # damit die Retrieval-Engine unabhängig von der restlichen RAG-Logik
    # feinjustiert werden kann.
    RETRIEVAL_SEMANTIC_CANDIDATES = 80
    RETRIEVAL_LEXICAL_CANDIDATES = 120
    RETRIEVAL_FUSED_CANDIDATES = 40
    RETRIEVAL_PAGE_CANDIDATES = 24
    RETRIEVAL_FINAL_CANDIDATES = 16
    # Reciprocal Rank Fusion statt direkter Mischung der Chroma-Distanzen.
    RRF_K = 60.0

    # BM25-Parameter.
    BM25_K1 = 1.35
    BM25_B = 0.72

    SOURCE_QUALITY_BOOST = 0.10
    SOURCE_QUALITY_PENALTY = 0.10
    VISION_LEXICAL_CANDIDATES = 24
    VISION_MIN_SCORE = 0.30
    VISION_RANK_TOP_K = 6
    VISION_PROMPT_VERSION = "vision-v3-page-content-inventory"

    # Relevanz-Gates. Unterhalb dieser Werte soll eine Seite nicht bis
    # zur Vision/Qwen-Stufe weitergereicht werden.
    MIN_FINAL_SCORE = 0.19
    MIN_STRONG_SCORE = 0.27
    MIN_LEXICAL_COVERAGE = 0.18

    SEARCH_STOPWORDS = {
        "der", "die", "das", "den", "dem", "des",
        "ein", "eine", "einer", "einem", "einen",
        "und", "oder", "aber", "wie", "was", "warum",
        "wieso", "weshalb", "ist", "sind", "sich", "zu",
        "von", "für", "mit", "im", "in", "auf", "an",
        "aus", "bei", "als", "auch", "noch", "über",
        "unter", "durch", "nach", "vor", "zwischen", "gegen",
        "welche", "welcher", "welches", "welchem", "welchen",
        "wichtig", "wichtigsten", "wichtige", "wichtigen",
        "relevant", "relevante", "relevanten", "relevantesten",
        "funktioniert", "funktionieren", "funktioniere",
        "kann", "können", "bitte", "bitten",
        "schritt", "schritte", "schritten",
        "inhalt", "inhalte", "erklären", "erklärt", "erklärung",
        "aufbau", "aufgebaut", "ablauf", "abläufe",
        "zeigen", "zeige", "beschreiben", "beschreibung",
        "erläutern", "erläutert", "erläuterung",
        "macht", "machen", "geht", "gehen", "dabei",
    }

    # Nur wenige sehr belastbare fachliche Synonyme. Sie dienen dazu,
    # dass unterschiedliche Benennungen desselben Konzepts zusammenfinden.
    # Fachkonzepte mit positiven und negativen Signalen.
    # Ein Query kann mehrere Konzepte gleichzeitig enthalten. Positive
    # Begriffe erhöhen die Evidenz; negative Begriffe verhindern, dass
    # fachlich verwandte, aber falsche Themen (z.B. Nernst-Planck statt
    # Nernst-Gleichung oder Spin-Bahn-Kopplung statt Spin-Spin-Kopplung)
    # nach oben rutschen.
    QUERY_CONCEPTS = {
        "nernst_equation": {
            "triggers": {"nernst gleichung", "nernstgleichung"},
            "positive": {
                "nernst", "nernst gleichung", "nernstgleichung",
                "elektrodenpotential", "elektrodenpotenzial",
                "elektromotorische kraft", "emk", "zellpotential",
                "zellpotenzial", "reaktionsquotient"
            },
            "negative": {
                "nernst planck", "nernstplanck", "nernst sche diffusionsschicht",
                "nernst'sche diffusionsschicht", "diffusionsschicht",
                "diffusionsstrom", "diffusionsschichtdicke"
            },
            "anchor": {"nernst gleichung", "nernstgleichung"},
            "required_groups": [],
        },
        "nmr": {
            "triggers": {"nmr", "kernmagnetische resonanz"},
            "positive": {
                "nmr", "kernmagnetische resonanz", "chemische verschiebung",
                "chemical shift", "spin spin kopplung", "spin spinkopplung",
                "spinspin kopplung", "spinspinkopplung", "ppm",
                "spektrometer", "spektrum", "nmr spektrum", "multiplett",
                "singulett", "dublett", "triplet", "triplett", "quartett",
                "kopplungskonstante", "j wert", "j wert", "j kopplung",
                "protonen nmr", "1h nmr"
            },
            "negative": {
                "spin bahn kopplung", "spinbahnkoppelung", "spinbahnkopplung",
                "spin bahn", "photophysik", "intersystem crossing", "isc",
                "iv r", "vibronische", "vibronisch",
                "parahydrogen", "para hydrogen", "hyperpolarisation",
                "hyperpolarization", "phip", "signalverstärkung",
                "signal enhancement", "hyperpolarisiert", "hyperpolarized",
                "polarisation transfer", "polarization transfer",
                "hydrogenation", "hydrierung", "transient nmr",
                "transiente nmr", "nuclear alignment"
            },
            "anchor": {"nmr", "kernmagnetische resonanz"},
            "required_groups": [
                {"chemische verschiebung", "chemical shift", "ppm"},
                {"spin spin kopplung", "spin spinkopplung", "spinspin kopplung",
                 "spinspinkopplung", "kopplungskonstante", "j wert", "j kopplung"}
            ],
        },
        "sn1_sn2": {
            "triggers": {
                "sn1", "sn2", "sn 1", "sn 2",
                "nukleophile substitution", "nukleophiler substitutionsmechanismus",
                "nukleophile substitutionsreaktion"
            },
            "positive": {
                "sn1", "sn2", "nukleophile substitution", "nukleophil",
                "substrat", "abgangsgruppe", "leaving group", "karbokation",
                "carbokation", "inversion", "racemisierung", "stereochemie",
                "protic", "aprotic", "geschwindigkeitsgesetz", "reaktionsgeschwindigkeit",
                "bimolekular", "unimolekular", "zweite ordnung", "erste ordnung",
                "tertiaer", "tertiär", "sterische hinderung"
            },
            "negative": {
                "dna", "rna", "replikation", "replikation", "transkription",
                "translation", "polymerase", "primer", "phosphodiester",
                "base excision", "nukleinsäure", "nukleinsaure", "chromosom",
                "genexpression", "elongation", "ribosom"
            },
            "anchor": {"sn1", "sn2", "nukleophile substitution"},
            "required_groups": [],
        },
    }

    CHEMICAL_SYNONYMS = {
        "citratzyklus": {
            "citratzyklus", "citrat zyklus", "tricarbonsäurezyklus",
            "tricarbonsaeurezyklus", "krebszyklus", "tca zyklus", "tca",
        },
        "tricarbonsäurezyklus": {
            "citratzyklus", "citrat zyklus", "tricarbonsäurezyklus",
            "tricarbonsaeurezyklus", "krebszyklus", "tca zyklus", "tca",
        },
        "krebszyklus": {
            "citratzyklus", "citrat zyklus", "tricarbonsäurezyklus",
            "tricarbonsaeurezyklus", "krebszyklus", "tca zyklus", "tca",
        },
        "nadh": {"nadh", "nad h", "nicotinamidadenindinukleotid"},
        "fadh2": {"fadh2", "fad h2", "flavinadenindinukleotid"},
        "atp": {"atp", "adenosintriphosphat"},
        "dna": {"dna", "desoxyribonukleinsäure", "desoxyribonucleinsäure"},
        "rna": {"rna", "ribonukleinsäure", "ribonucleinsäure"},
    }

    @staticmethod
    def _empty_results():
        return {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    @staticmethod
    def _normalize_search_text(text):
        text = str(text or "").lower()
        text = unicodedata.normalize("NFKD", text)
        text = "".join(
            char for char in text if not unicodedata.combining(char)
        )

        replacements = {
            "β": "beta", "α": "alpha", "γ": "gamma", "δ": "delta",
            "μ": "mu", "µ": "mu", "κ": "kappa", "λ": "lambda",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        return re.sub(r"[^a-z0-9äöüß]+", " ", text).strip()

    @classmethod
    def _compact_search_text(cls, text):
        return re.sub(r"[^a-z0-9äöüß]+", "", cls._normalize_search_text(text))

    @classmethod
    def _tokenize_search_text(cls, text):
        normalized = cls._normalize_search_text(text)
        tokens = []
        for token in normalized.split():
            if len(token) < 2:
                continue
            tokens.append(token)
        return tokens

    @classmethod
    def _search_terms(cls, question):
        terms = []
        for token in cls._tokenize_search_text(question):
            if len(token) < 3 or token in cls.SEARCH_STOPWORDS:
                continue
            if token not in terms:
                terms.append(token)
        return terms

    @classmethod
    def _query_phrases(cls, question):
        normalized = cls._normalize_search_text(question)
        words = [
            w for w in normalized.split()
            if len(w) >= 3 and w not in cls.SEARCH_STOPWORDS
        ]
        phrases = set()

        # Benachbarte Fachbegriffe.
        for n in (2, 3):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i+n])
                if all(len(x) >= 3 for x in words[i:i+n]):
                    phrases.add(phrase)

        # Explizite chemische Synonyme als Phrase ergänzen.
        for canonical, variants in cls.CHEMICAL_SYNONYMS.items():
            if any(v in normalized for v in variants):
                phrases.add(canonical)
                phrases.update(variants)

        return phrases

    @classmethod
    def _expanded_query_terms(cls, question):
        terms = set(cls._search_terms(question))
        normalized = cls._normalize_search_text(question)

        for canonical, variants in cls.CHEMICAL_SYNONYMS.items():
            if canonical in normalized or any(v in normalized for v in variants):
                for variant in variants:
                    for token in cls._tokenize_search_text(variant):
                        if len(token) >= 3 and token not in cls.SEARCH_STOPWORDS:
                            terms.add(token)

        return terms

    @classmethod
    def _query_concept_profile(cls, question):
        normalized = cls._normalize_search_text(question)
        compact = cls._compact_search_text(question)
        active = []

        for name, concept in cls.QUERY_CONCEPTS.items():
            triggered = False
            for trigger in concept["triggers"]:
                t_norm = cls._normalize_search_text(trigger)
                t_compact = cls._compact_search_text(trigger)
                if t_norm in normalized or t_compact in compact:
                    triggered = True
                    break
            if triggered:
                active.append({
                    "name": name,
                    "positive": set(concept["positive"]),
                    "negative": set(concept["negative"]),
                    "anchor": set(concept["anchor"]),
                    "required_groups": [set(group) for group in concept.get("required_groups", [])],
                })

        return active

    @classmethod
    def _concept_evidence(cls, doc_norm, doc_compact, profile):
        positive_hits = set()
        negative_hits = set()
        anchor_hits = set()

        for term in profile["positive"]:
            norm = cls._normalize_search_text(term)
            compact = cls._compact_search_text(term)
            if norm in doc_norm or compact in doc_compact:
                positive_hits.add(term)

        for term in profile["negative"]:
            norm = cls._normalize_search_text(term)
            compact = cls._compact_search_text(term)
            if norm in doc_norm or compact in doc_compact:
                negative_hits.add(term)

        for term in profile["anchor"]:
            norm = cls._normalize_search_text(term)
            compact = cls._compact_search_text(term)
            if norm in doc_norm or compact in doc_compact:
                anchor_hits.add(term)

        required_hits = []
        for group in profile.get("required_groups", []):
            group_hit = False
            for term in group:
                norm = cls._normalize_search_text(term)
                compact = cls._compact_search_text(term)
                if norm in doc_norm or compact in doc_compact:
                    group_hit = True
                    break
            if group_hit:
                required_hits.append(True)

        return positive_hits, negative_hits, anchor_hits, required_hits

    def _ensure_lexical_index(self):
        """Baut einmalig pro Chroma-Bestand einen kleinen BM25-Index auf."""
        try:
            count = self.database.collection.count()
        except Exception:
            count = -1

        if getattr(self, "_lexical_index_count", None) == count:
            return

        stored = self.database.collection.get(
            include=["documents", "metadatas"]
        )

        documents = stored.get("documents", []) or []
        metadatas = stored.get("metadatas", []) or []

        postings = {}
        subword_postings = {}
        doc_lengths = []
        normalized_docs = []
        compact_docs = []

        for idx, doc in enumerate(documents):
            tokens = self._tokenize_search_text(doc)
            doc_lengths.append(len(tokens))
            normalized_docs.append(self._normalize_search_text(doc))
            compact_docs.append(self._compact_search_text(doc))

            unique_tokens = set(tokens)
            for token in unique_tokens:
                postings.setdefault(token, []).append(idx)

                # Komposita sind in deutschen Chemietexten sehr häufig:
                # Citratzyklus, Elektronentransportkette,
                # Oxidationsphosphorylierung usw. Ein kleiner 4-Gramm-Index
                # erlaubt, dass "citrat" auch "citratzyklus" findet, ohne
                # bei jeder Anfrage alle Dokumente zu scannen.
                if len(token) >= 6:
                    for pos in range(len(token) - 3):
                        gram = token[pos:pos + 4]
                        subword_postings.setdefault(gram, set()).add(idx)

        self._lexical_documents = documents
        self._lexical_metadatas = metadatas
        self._lexical_postings = postings
        self._lexical_subword_postings = subword_postings
        self._lexical_doc_lengths = doc_lengths
        self._lexical_normalized_docs = normalized_docs
        self._lexical_compact_docs = compact_docs
        self._lexical_index_count = count

    @classmethod
    def _source_quality_signal(cls, source, question):
        source_norm = cls._normalize_search_text(source)
        question_norm = cls._normalize_search_text(question)
        educational = (
            "vorlesung", "skript", "ubungsblatt", "musterlosung",
            "seminar", "praktikum", "protokoll", "wq", "lernshit",
            "lehr", "zusammenfassung", "klausur", "prufung",
        )
        research = (
            "journal", "doi", "review", "coordination chemistry reviews",
            "j am chem", "angewandte", "parahydrogen", "duckett",
            "acs article", "research article", "journal article",
            "metronidazole", "supporting information",
        )
        score = 0.0
        if any(term in source_norm for term in educational):
            score += cls.SOURCE_QUALITY_BOOST
        if "brauchen wir nicht" in source_norm:
            score -= cls.SOURCE_QUALITY_PENALTY
        didactic = any(token in question_norm for token in (
            "erklar", "unterschied", "wie funktioniert",
            "bedeutet", "prinzip", "warum", "aufgabe",
            "was ist", "was sind", "interpretier", "muster",
            "grundlage", "definition", "verstandlich",
        ))
        if didactic and any(term in source_norm for term in research):
            score -= min(0.18, cls.SOURCE_QUALITY_PENALTY * 1.8)
        return max(-0.20, min(0.20, score))

    @classmethod
    def _explicit_target_score(cls, question, doc_text, source=""):
        """Boost konkrete, vom Nutzer genannte Ziele wie Butanon/Spektrum.

        Das ist absichtlich konservativ: Nur klar erkennbare Fach-/Dokumentziele
        werden geboostet, damit allgemeine Fragen nicht vom Dateinamen dominiert werden.
        """
        q = cls._normalize_search_text(question)
        if not q:
            return 0.0

        doc = cls._normalize_search_text(doc_text)
        src = cls._normalize_search_text(source)
        score = 0.0

        explicit_targets = {
            "butanon": {"butanon", "butan 2 on", "butan 2 on nmr", "2 butanon"},
            "ethanol": {"ethanol"},
            "acetaldehyd": {"acetaldehyd", "ethanal"},
        }

        for variants in explicit_targets.values():
            if any(v in q for v in variants):
                if any(v in doc or v in src for v in variants):
                    score += 0.20

        # Explizite Spektrum-/Abbildungsfragen profitieren von Seiten,
        # deren Titel/Dateiname selbst auf Spektroskopie hinweist.
        spectrum_query = any(term in q for term in (
            "spektrum", "nmr spektrum", "spektrum auswerten",
            "spektrum interpretieren", "peak", "signal", "signale"
        ))
        if spectrum_query:
            if any(term in src or term in doc for term in (
                "spektro", "nmr", "spektrum", "spectrum", "spektroskop"
            )):
                score += 0.08

        return min(0.30, score)

    @classmethod
    def _vision_lexical_score(cls, question, analysis, source=""):
        if not analysis:
            return 0.0
        terms = cls._expanded_query_terms(question)
        if not terms:
            return 0.0
        doc_norm = cls._normalize_search_text(analysis)
        doc_compact = cls._compact_search_text(analysis)
        hits = 0
        for term in terms:
            norm = cls._normalize_search_text(term)
            compact = cls._compact_search_text(term)
            if norm in doc_norm or compact in doc_compact:
                hits += 1
        phrases = cls._query_phrases(question)
        phrase_hits = 0
        for phrase in phrases:
            norm = cls._normalize_search_text(phrase)
            compact = cls._compact_search_text(phrase)
            if norm in doc_norm or compact in doc_compact:
                phrase_hits += 1
        coverage = hits / max(len(terms), 1)
        score = min(0.75, coverage * 0.75)
        score += min(0.25, phrase_hits * 0.10)
        score += cls._source_quality_signal(source, question) * 0.5
        score += cls._explicit_target_score(question, analysis, source)
        return max(0.0, min(1.0, score))

    def _lexical_search(self, question, n_results, is_cache=None):
        """BM25 + Phrase/Exact-Match-Ranking auf Chunk-Ebene."""
        self._ensure_lexical_index()

        terms = self._expanded_query_terms(question)
        phrases = self._query_phrases(question)
        concept_profiles = self._query_concept_profile(question)
        if not terms:
            return self._empty_results()

        documents = self._lexical_documents
        metadatas = self._lexical_metadatas
        postings = self._lexical_postings
        lengths = self._lexical_doc_lengths
        normalized_docs = self._lexical_normalized_docs
        compact_docs = self._lexical_compact_docs

        N = max(len(documents), 1)
        avgdl = sum(lengths) / max(len(lengths), 1)
        avgdl = max(avgdl, 1.0)

        # Kandidaten nur aus Postings ziehen, statt jeden Chunk bei jeder
        # Frage komplett zu durchsuchen.
        candidate_ids = set()
        subword_postings = self._lexical_subword_postings
        for term in terms:
            candidate_ids.update(postings.get(term, []))
            if len(term) >= 4:
                candidate_ids.update(subword_postings.get(term[:4], set()))

        # Synonyme/Komposita können als Teilstring vorkommen. Dafür werden
        # nur die bereits relevanten Kandidaten weiter untersucht.
        candidates = []
        query_normalized = self._normalize_search_text(question)
        query_compact = self._compact_search_text(question)

        for idx in candidate_ids:
            meta = metadatas[idx] or {}
            if is_cache is not None:
                if bool(meta.get("is_cache", False)) != bool(is_cache):
                    continue
            if not self._is_source_allowed(meta):
                continue

            doc_norm = normalized_docs[idx]
            doc_compact = compact_docs[idx]
            dl = max(lengths[idx], 1)

            score = 0.0
            matched = set()
            exact_matches = 0
            concept_bonus = 0.0
            concept_penalty = 0.0
            concept_anchor_count = 0
            concept_negative_count = 0
            concept_required_groups = 0
            concept_required_total = 0

            # BM25: seltene Begriffe werden automatisch stärker gewichtet.
            for term in terms:
                posting = postings.get(term)
                if not posting or idx not in posting:
                    # Kompositum-Fallback: z.B. "citrat" in "citratzyklus".
                    # Der Kandidat wurde bereits über den 4-Gramm-Index
                    # gefunden; jetzt prüfen wir den tatsächlichen Treffer.
                    if term not in doc_compact:
                        continue
                    tf = 1
                else:
                    tf = doc_norm.split().count(term)

                df = len(posting) if posting else 1
                idf = max(0.0, __import__("math").log(
                    1.0 + (N - df + 0.5) / (df + 0.5)
                ))
                denom = tf + self.BM25_K1 * (
                    1.0 - self.BM25_B + self.BM25_B * dl / avgdl
                )
                score += idf * (tf * (self.BM25_K1 + 1.0)) / max(denom, 1e-9)
                matched.add(term)

            coverage = len(matched) / max(len(terms), 1)

            # Fachkonzept-Scoring: Ein aktives Konzept verlangt nicht, dass
            # alle Begriffe auf derselben Seite stehen, aber ein eindeutiger
            # Anker wie "NMR" oder "Nernst-Gleichung" ist deutlich stärker
            # als generische Wörter wie "Spin" oder "Kopplung".
            for profile in concept_profiles:
                positive_hits, negative_hits, anchor_hits, required_hits = self._concept_evidence(
                    doc_norm, doc_compact, profile
                )
                concept_anchor_count += len(anchor_hits)
                concept_negative_count += len(negative_hits)
                concept_required_groups += len(required_hits)
                concept_required_total += len(profile.get("required_groups", []))

                if positive_hits:
                    concept_bonus += min(3.0, 0.85 * len(positive_hits))
                if anchor_hits:
                    concept_bonus += 2.2 + min(1.5, 0.5 * max(len(anchor_hits) - 1, 0))
                if negative_hits:
                    # Ein negativer Fachbegriff ist ein starkes Gegenargument.
                    concept_penalty += min(5.0, 2.2 * len(negative_hits))

                # Bei einem expliziten Fachanker soll eine Seite ohne diesen
                # Anker nicht allein durch generische Wörter gewinnen.
                if profile["anchor"] and not anchor_hits:
                    concept_penalty += 1.4

            # Bei mehrteiligen Fachfragen muss der Treffer möglichst mehrere
            # der tatsächlich abgefragten Konzepte abdecken. Das verhindert
            # z.B. dass ein beliebiges NMR-Paper nur wegen des Tokens "NMR"
            # vor einer Lehrfolie mit chemischer Verschiebung und J-Kopplung
            # landet.
            if concept_required_total:
                required_coverage = concept_required_groups / concept_required_total
                if required_coverage == 0:
                    concept_penalty += 2.8
                elif required_coverage < 1.0:
                    concept_penalty += 0.9
                else:
                    concept_bonus += 1.4

            score += concept_bonus
            score -= concept_penalty

            # Exakte Phrase ist bei Fachbegriffen extrem aussagekräftig.
            phrase_hits = 0
            for phrase in phrases:
                phrase_norm = self._normalize_search_text(phrase)
                phrase_compact = self._compact_search_text(phrase)
                if phrase_norm in doc_norm or phrase_compact in doc_compact:
                    phrase_hits += 1

            if phrase_hits:
                score *= 1.0 + min(0.65, 0.18 * phrase_hits)

            # Die komplette Frage als zusammenhängender Text ist ein sehr
            # starker Hinweis, aber nur bei kurzen/inhaltlichen Fragen.
            if len(query_compact) >= 10 and query_compact in doc_compact:
                score += 3.0
                exact_matches += 1

            # Dateiname/Titel zählt ebenfalls, aber nur moderat.
            source = str(meta.get("source", ""))
            source_norm = self._normalize_search_text(source)
            source_terms = set(source_norm.split())
            source_hits = len(source_terms.intersection(terms))
            if source_hits:
                score += min(2.5, 0.8 * source_hits)
            score += self._source_quality_signal(source, question) * 2.0
            explicit_target_bonus = self._explicit_target_score(
                question,
                documents[idx],
                source,
            )
            score += explicit_target_bonus * 4.0

            if score <= 0:
                continue

            candidates.append({
                "idx": idx,
                "score": score,
                "coverage": coverage,
                "phrase_hits": phrase_hits,
                "exact_matches": exact_matches,
                "concept_bonus": concept_bonus,
                "concept_penalty": concept_penalty,
                "concept_anchor_count": concept_anchor_count,
                "concept_negative_count": concept_negative_count,
                "concept_required_groups": concept_required_groups,
                "concept_required_total": concept_required_total,
                "explicit_target_bonus": explicit_target_bonus,
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        candidates = candidates[:n_results]

        # Auf [0,1] normalisieren; Distanz ist nur noch eine technische
        # Rückgabe für die bestehende Result-Struktur.
        max_score = candidates[0]["score"] if candidates else 1.0
        for item in candidates:
            item["normalized_score"] = item["score"] / max(max_score, 1e-9)

        print(f"[RETRIEVAL] BM25-Kandidaten: {len(candidates)}")

        return {
            "documents": [[documents[item["idx"]] for item in candidates]],
            "metadatas": [[metadatas[item["idx"]] for item in candidates]],
            "distances": [[1.0 - item["normalized_score"] for item in candidates]],
            "_retrieval_scores": [[item for item in candidates]],
        }

    def _semantic_search(self, question, is_cache=None):
        embedding = self._embedding(question)
        return self.database.search(
            embedding,
            self.RETRIEVAL_SEMANTIC_CANDIDATES,
            is_cache=is_cache,
        )

    def _merge_retrieval_results(self, semantic, lexical, n_results):
        """RRF-Fusion auf Chunk-Ebene, danach Zusammenführung auf Seitenebene."""
        merged = {}

        def ingest(results, channel):
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            retrieval_scores = results.get("_retrieval_scores", [[]])[0]

            for rank, (doc, meta) in enumerate(zip(docs, metas), start=1):
                score_info = retrieval_scores[rank - 1] if rank - 1 < len(retrieval_scores) else {}
                meta = meta or {}
                key = (meta.get("source"), meta.get("page"), str(doc))
                item = merged.setdefault(key, {
                    "document": doc,
                    "metadata": meta,
                    "semantic_rank": None,
                    "lexical_rank": None,
                    "lexical_score": 0.0,
                    "concept_anchor_count": 0,
                    "concept_negative_count": 0,
                    "concept_bonus": 0.0,
                    "concept_penalty": 0.0,
                    "concept_required_groups": 0,
                    "concept_required_total": 0,
                    "explicit_target_bonus": 0.0,
                })
                if channel == "semantic" and item["semantic_rank"] is None:
                    item["semantic_rank"] = rank
                if channel == "lexical" and item["lexical_rank"] is None:
                    item["lexical_rank"] = rank
                    item["lexical_score"] = float(score_info.get("normalized_score", 0.0))
                    item["concept_anchor_count"] = int(score_info.get("concept_anchor_count", 0))
                    item["concept_negative_count"] = int(score_info.get("concept_negative_count", 0))
                    item["concept_bonus"] = float(score_info.get("concept_bonus", 0.0))
                    item["concept_penalty"] = float(score_info.get("concept_penalty", 0.0))
                    item["concept_required_groups"] = int(score_info.get("concept_required_groups", 0))
                    item["concept_required_total"] = int(score_info.get("concept_required_total", 0))
                    item["explicit_target_bonus"] = float(score_info.get("explicit_target_bonus", 0.0))

        ingest(semantic, "semantic")
        ingest(lexical, "lexical")

        ranked = []
        for item in merged.values():
            sem_rank = item["semantic_rank"]
            lex_rank = item["lexical_rank"]

            # Rang-Scores statt Chroma-Distanzen: Die beiden Retrieval-
            # Kanäle werden auf derselben, robusten Skala kombiniert.
            # Logarithmische Abwertung verhindert, dass Rang 20 fast so
            # stark zählt wie Rang 1.
            sem_score = (
                1.0 / __import__("math").log2(sem_rank + 1.0)
                if sem_rank is not None else 0.0
            )
            lex_score = (
                1.0 / __import__("math").log2(lex_rank + 1.0)
                if lex_rank is not None else 0.0
            )

            # Lexikalische Evidenz ist bei chemischen Fachbegriffen bewusst
            # etwas wichtiger als reine semantische Ähnlichkeit.
            fused = 0.55 * lex_score + 0.45 * sem_score

            # Fachkonzept-Evidenz aus BM25 wirkt auf den finalen Rang, ohne
            # die Rohscores der beiden Retrieval-Kanäle zu vermischen.
            if item.get("lexical_rank") is not None:
                fused += min(0.20, 0.06 * item.get("concept_anchor_count", 0))
                fused += min(0.12, 0.03 * item.get("concept_bonus", 0.0))
                fused -= min(0.24, 0.08 * item.get("concept_negative_count", 0))
                fused -= min(0.12, 0.04 * item.get("concept_penalty", 0.0))
                fused += min(0.10, item.get("explicit_target_bonus", 0.0) * 0.20)

            # Exakt in beiden Retrieval-Kanälen gefunden = besonders stabil.
            agreement = 1.0 if sem_rank is not None and lex_rank is not None else 0.0

            item["rrf_score"] = fused
            item["agreement"] = agreement
            ranked.append(item)

        ranked.sort(key=lambda x: x["rrf_score"], reverse=True)
        ranked = ranked[:self.RETRIEVAL_FUSED_CANDIDATES]

        return self._aggregate_pages(ranked, n_results, lexical_available=bool(lexical.get("documents", [[]])[0]))

    def _aggregate_pages(self, chunks, n_results, lexical_available=True):
        """Mehrere gute Chunks derselben PDF-Seite werden zu einer Seite fusioniert."""
        pages = {}

        for item in chunks:
            meta = item["metadata"]
            key = (meta.get("source"), meta.get("page"), bool(meta.get("is_cache", False)))
            page = pages.setdefault(key, {
                "metadata": meta,
                "documents": [],
                "rrf_sum": 0.0,
                "best_rrf": 0.0,
                "agreement": 0.0,
                "chunk_count": 0,
                "lexical_rank": None,
                "semantic_rank": None,
                "concept_anchor_count": 0,
                "concept_negative_count": 0,
                "concept_required_groups": 0,
                "concept_required_total": 0,
                "explicit_target_bonus": 0.0,
            })

            page["documents"].append(item["document"])
            page["rrf_sum"] += item["rrf_score"]
            page["concept_anchor_count"] = max(page["concept_anchor_count"], item.get("concept_anchor_count", 0))
            page["concept_negative_count"] = max(page["concept_negative_count"], item.get("concept_negative_count", 0))
            page["concept_required_groups"] = max(page["concept_required_groups"], item.get("concept_required_groups", 0))
            page["concept_required_total"] = max(page["concept_required_total"], item.get("concept_required_total", 0))
            page["explicit_target_bonus"] = max(page["explicit_target_bonus"], item.get("explicit_target_bonus", 0.0))
            page["best_rrf"] = max(page["best_rrf"], item["rrf_score"])
            page["agreement"] = max(page["agreement"], item["agreement"])
            page["chunk_count"] += 1
            if item["lexical_rank"] is not None:
                if page["lexical_rank"] is None:
                    page["lexical_rank"] = item["lexical_rank"]
                else:
                    page["lexical_rank"] = min(page["lexical_rank"], item["lexical_rank"])
            if item["semantic_rank"] is not None:
                if page["semantic_rank"] is None:
                    page["semantic_rank"] = item["semantic_rank"]
                else:
                    page["semantic_rank"] = min(page["semantic_rank"], item["semantic_rank"])

        ranked_pages = []

        for page in pages.values():
            # Der Fusionsscore liegt bereits sinnvoll auf ungefähr [0,1].
            # Kein künstliches Normieren auf den besten Treffer: Ein
            # schlechtes Gesamtergebnis soll auch schlecht bleiben.
            base = page["best_rrf"]

            # Zweiter guter Chunk derselben Seite ist ein echtes Signal,
            # aber mit abnehmendem Zusatznutzen.
            chunk_bonus = min(0.10, 0.05 * max(page["chunk_count"] - 1, 0))
            agreement_bonus = 0.06 * page["agreement"]
            concept_bonus = min(0.12, 0.06 * page["concept_anchor_count"])
            concept_penalty = min(0.20, 0.10 * page["concept_negative_count"])
            explicit_target_bonus = min(0.12, page.get("explicit_target_bonus", 0.0) * 0.40)

            required_penalty = 0.0
            if page["concept_required_total"]:
                coverage = page["concept_required_groups"] / page["concept_required_total"]
                if coverage == 0:
                    required_penalty = 0.22
                elif coverage < 1.0:
                    required_penalty = 0.07

            page_score = min(
                1.0,
                max(
                    0.0,
                    base + chunk_bonus + agreement_bonus + concept_bonus
                    + explicit_target_bonus
                    - concept_penalty - required_penalty
                )
            )

            page["final_score"] = page_score
            page["document"] = "\n\n".join(page["documents"])
            ranked_pages.append(page)

        ranked_pages.sort(key=lambda p: p["final_score"], reverse=True)

        # Relevanz-Gate: schwache semantische Zufallstreffer dürfen nicht
        # automatisch in Vision und Qwen gelangen. Für Seiten mit starker
        # lexikalischer Evidenz lassen wir den Score bewusst großzügiger.
        selected = []
        concept_profiles = self._query_concept_profile(getattr(self, "_last_retrieval_question", ""))
        for page in ranked_pages:
            # Harte Negativregel: Wenn ein Query einen eindeutigen Fachanker
            # hat und die Seite nur den falschen verwandten Begriff enthält,
            # soll sie nicht als Vision-/Qwen-Quelle weitergereicht werden.
            if concept_profiles:
                if page["concept_negative_count"] > 0 and page["concept_anchor_count"] == 0:
                    continue

            score = page["final_score"]
            has_strong_evidence = (
                page["agreement"] > 0
                or page["chunk_count"] >= 2
            )

            # Wenn die Frage lexikalisch Treffer erzeugt, sollen reine
            # Semantic-Fallbacks nicht massenhaft durchgelassen werden.
            # Das ist der wichtigste Schutz gegen "Elektrochemie bei
            # Citratzyklus"-Fehltreffer.
            has_lexical = page["lexical_rank"] is not None
            if lexical_available and not has_lexical:
                # Semantic-only Seiten sind bei einer Frage mit vorhandener
                # lexikalischer Evidenz zunächst nur Fallback. Wir lassen
                # sie nur zu, wenn die semantische Platzierung außergewöhnlich
                # gut ist und keine bessere Seite existiert.
                if page["semantic_rank"] is None or page["semantic_rank"] > 3:
                    continue
                if score < 0.55:
                    continue

            if lexical_available and not has_strong_evidence and score < self.MIN_STRONG_SCORE:
                continue

            if score >= self.MIN_FINAL_SCORE or (
                has_strong_evidence and score >= self.MIN_STRONG_SCORE * 0.72
            ):
                selected.append(page)

            if len(selected) >= self.RETRIEVAL_FINAL_CANDIDATES:
                break

        print(f"[RETRIEVAL] Relevante Seiten: {len(selected)}")
        for index, page in enumerate(selected[:8], start=1):
            meta = page["metadata"]
            print(
                f"    {index}. {meta.get('source')} | Seite {meta.get('page')} | "
                f"score={page['final_score']:.3f} | chunks={page['chunk_count']}"
            )

        return {
            "documents": [[p["document"] for p in selected[:n_results]]],
            "metadatas": [[p["metadata"] for p in selected[:n_results]]],
            "distances": [[1.0 - p["final_score"] for p in selected[:n_results]]],
        }

    def search_documents(self, question, is_cache=None):
        self._last_retrieval_question = question
        if self.active_source_keys is not None and not self.active_source_keys:
            print("[SCOPE] Keine Dokumente für diesen Chat freigegeben; Retrieval bleibt leer.")
            return self._empty_results()
        semantic = self._semantic_search(question, is_cache=is_cache)
        lexical = self._lexical_search(
            question,
            self.RETRIEVAL_LEXICAL_CANDIDATES,
            is_cache=is_cache,
        )

        results = self._merge_retrieval_results(
            semantic,
            lexical,
            self.RETRIEVAL_PAGE_CANDIDATES,
        )

        return results

    def quick_search_files(self, question, limit=30):
        """Schnelle lokale Volltext-/Dateisuche direkt in den PDFs des aktiven Projekts.

        Sie benötigt keinen Chroma-Treffer und funktioniert deshalb auch dann,
        wenn ein PDF noch nicht indexiert wurde oder dessen Index repariert werden muss.
        Es werden ausschließlich die Dokument- und Cache-Ordner dieser RAG-Instanz
        durchsucht; andere Projekte werden nicht berührt.
        """
        query = str(question or "").strip()
        if not query:
            return []

        normalized_query = self._normalize_search_text(query)
        compact_query = self._compact_search_text(query)
        terms = []
        for term in normalized_query.split():
            if len(term) >= 2 and term not in self.SEARCH_STOPWORDS and term not in terms:
                terms.append(term)
        if not terms:
            terms = [term for term in normalized_query.split() if term]
        if not terms and not compact_query:
            return []

        candidates = []
        for base_dir, is_cache in (
            (self.config.DOCUMENTS_DIR, False),
            (self.config.CACHE_DOCUMENTS_DIR, True),
        ):
            if not Path(base_dir).exists():
                continue
            try:
                pdf_paths = sorted(
                    (p for p in Path(base_dir).rglob("*")
                     if p.is_file() and p.suffix.casefold() == ".pdf"
                     and not p.name.startswith("._")),
                    key=lambda p: str(p).casefold(),
                )
            except OSError as error:
                print(f"[QUICKSEARCH] Ordner konnte nicht gelesen werden ({base_dir}): {error}")
                continue

            for pdf_path in pdf_paths:
                try:
                    stat = pdf_path.stat()
                    cache_key = str(pdf_path.resolve())
                    signature = (stat.st_mtime_ns, stat.st_size)
                    cached = self._quick_pdf_text_cache.get(cache_key)
                    if cached is None or cached[0] != signature:
                        with pymupdf.open(pdf_path) as pdf:
                            pages = [page.get_text("text") or "" for page in pdf]
                        self._quick_pdf_text_cache[cache_key] = (signature, pages)
                    else:
                        pages = cached[1]
                except Exception as error:
                    print(f"[QUICKSEARCH] PDF konnte nicht gelesen werden: {pdf_path} ({error})")
                    continue

                try:
                    relative = pdf_path.relative_to(base_dir).as_posix()
                except ValueError:
                    relative = pdf_path.name
                display_source = f"Cache/{relative}" if is_cache else relative
                normalized_name = self._normalize_search_text(pdf_path.name)
                compact_name = self._compact_search_text(pdf_path.stem)
                filename_hit = bool(
                    (normalized_query and normalized_query in normalized_name)
                    or (compact_query and len(compact_query) >= 3 and compact_query in compact_name)
                    or (terms and sum(term in normalized_name for term in terms) >= max(1, min(2, len(terms))))
                )

                file_results = []
                for page_no, page_text in enumerate(pages, start=1):
                    if not page_text.strip():
                        continue
                    normalized_text = self._normalize_search_text(page_text)
                    compact_text = self._compact_search_text(page_text)
                    phrase_hit = bool(normalized_query and normalized_query in normalized_text)
                    compact_hit = bool(compact_query and len(compact_query) >= 3 and compact_query in compact_text)
                    term_hits = [term for term in terms if term in normalized_text]
                    minimum_hits = max(1, min(2, len(terms)))
                    if not (phrase_hit or compact_hit or len(term_hits) >= minimum_hits):
                        continue

                    score = (5.0 if phrase_hit else 0.0) + (3.0 if compact_hit else 0.0)
                    score += len(term_hits) / max(1, len(terms)) * 2.0
                    if filename_hit:
                        score += 0.5
                    # Erzeuge einen Originaltext-Ausschnitt um das erste passende Wort.
                    lower_page = page_text.casefold()
                    positions = [lower_page.find(term.casefold()) for term in terms if lower_page.find(term.casefold()) >= 0]
                    start_pos = max(0, (min(positions) if positions else 0) - 150)
                    excerpt = " ".join(page_text[start_pos:start_pos + 430].split())
                    if start_pos > 0:
                        excerpt = "…" + excerpt
                    if start_pos + 430 < len(page_text):
                        excerpt += "…"
                    file_results.append({
                        "source": display_source,
                        "page": page_no,
                        "excerpt": excerpt,
                        "is_cache": is_cache,
                        "score": score,
                        "kind": "PDF-Volltext",
                    })

                if file_results:
                    candidates.extend(file_results[:3])
                elif filename_hit:
                    candidates.append({
                        "source": display_source,
                        "page": "Dateiname",
                        "excerpt": f"Der Dateiname passt zur Suchanfrage: {pdf_path.name}",
                        "is_cache": is_cache,
                        "score": 1.0,
                        "kind": "Dateiname",
                    })

        candidates.sort(key=lambda item: (-float(item["score"]), str(item["source"]).casefold(), str(item["page"])))
        return candidates[:max(1, int(limit))]

    # =========================================================
    # Cache-Fragen erkennen
    # =========================================================

    def _mentions_cache(
        self,
        question
    ):

        text = (
            question
            .lower()
            .strip()
        )

        keywords = [
            "cache",
            "zwischenspeicher",
            "temporär",
            "temporären",
            "temporäre",
        ]

        return any(
            keyword in text
            for keyword in keywords
        )

    def _is_summary_request(
        self,
        question
    ):

        text = (
            question
            .lower()
        )

        keywords = [
            "fasse",
            "zusammenfassung",
            "zusammenfassen",
            "überblick",
            "inhalt zusammen",
            "kurz zusammen",
            "summary",
            "summarize",
        ]

        return any(
            keyword in text
            for keyword in keywords
        )

    # =========================================================
    # Ergebnisse diversifizieren
    # =========================================================

    def _diversify_results(
        self,
        results
    ):

        docs = results.get(
            "documents",
            [[]]
        )[0]

        metas = results.get(
            "metadatas",
            [[]]
        )[0]

        distances = results.get(
            "distances",
            [[]]
        )[0]

        selected_docs = []

        selected_metas = []

        selected_distances = []

        seen_pages = set()

        for doc, meta, distance in zip(
            docs,
            metas,
            distances
        ):

            key = (
                meta.get(
                    "source"
                ),
                meta.get(
                    "page"
                )
            )

            if key in seen_pages:
                continue

            seen_pages.add(
                key
            )

            selected_docs.append(
                doc
            )

            selected_metas.append(
                meta
            )

            selected_distances.append(
                distance
            )

            if (
                len(
                    selected_docs
                )
                >=
                self.config
                    .MAX_RETRIEVED_PAGES
            ):
                break

        return {
            "documents": [
                selected_docs
            ],

            "metadatas": [
                selected_metas
            ],

            "distances": [
                selected_distances
            ],
        }

    # =========================================================
    # Kontext
    # =========================================================

    def _context(
        self,
        results
    ):

        docs = results.get(
            "documents",
            [[]]
        )[0]

        metas = results.get(
            "metadatas",
            [[]]
        )[0]

        pages = {}

        for doc, meta in zip(
            docs,
            metas
        ):

            key = (
                meta.get(
                    "source"
                ),
                meta.get(
                    "page"
                )
            )

            pages.setdefault(
                key,
                []
            ).append(
                doc
            )

        blocks = []

        total = 0

        for (
            source,
            page
        ), chunks in pages.items():

            text = (
                "\n\n".join(
                    chunks
                )
            )

            text = text[
                :self.config
                .MAX_SOURCE_CHARS
            ]

            cache_label = ""

            if source.startswith(
                "cache/"
            ):

                cache_label = (
                    " [CACHE / "
                    "TEMPORÄR]"
                )

            block = (
                "[QUELLE]\n"
                f"Datei: {source}"
                f"{cache_label}\n"
                f"Seite: {page}\n\n"
                f"{text}"
            )

            if (
                total + len(block)
                >
                self.config
                    .MAX_CONTEXT_CHARS
            ):

                break

            blocks.append(
                block
            )

            total += len(
                block
            )

        return "\n\n".join(
            blocks
        )

    # =========================================================
    # Originalbilder / visuelle Kandidaten
    # =========================================================

    def _image_item_from_meta(self, meta):
        source = meta.get("source")
        page = meta.get("page")
        image_rel = meta.get("image_path")

        if not source or page is None:
            return None

        if not image_rel:
            image_rel = str(
                self._image_path(source, int(page)).relative_to(
                    self.config.PROJECT_ROOT
                )
            )

        image_path = self.config.PROJECT_ROOT / image_rel
        if not image_path.exists():
            return None

        image_bytes = image_path.read_bytes()
        raw_hash = self._hash_bytes(image_bytes)
        vision_hash = self._hash_bytes(
            f"{raw_hash}|{self.config.VISION_PROMPT_VERSION}".encode("utf-8")
        )
        return {
            "source": source,
            "page": page,
            "image": image_bytes,
            "vision_hash": vision_hash,
            "visual_score": float(meta.get("visual_score", 0.0)),
            "is_cache": bool(meta.get("is_cache", False)),
        }

    def _source_images(
        self,
        results,
        extra_visual_candidates=None,
        question="",
    ):
        """Select original pages using relevance plus source suitability.

        For explanatory/learning questions, a cached image from a research paper
        must not outrank a closely matching teaching page solely because its
        old vision description happens to be semantically similar.
        """
        metas = results.get("metadatas", [[]])[0]
        result = []
        seen = set()
        candidates_by_key = {}

        for meta in metas:
            meta = dict(meta or {})
            meta["visual_score"] = max(
                float(meta.get("visual_score", 0.0)),
                float(meta.get("final_score", 0.0)),
            )
            key = (meta.get("source"), meta.get("page"))
            candidates_by_key[key] = meta

        for meta in extra_visual_candidates or []:
            meta = dict(meta or {})
            key = (meta.get("source"), meta.get("page"))
            previous = candidates_by_key.get(key)
            if previous:
                # Keep richer metadata (especially the persisted analysis),
                # but retain the strongest relevance score from either channel.
                merged = {**previous, **meta}
                merged["visual_score"] = max(
                    float(previous.get("visual_score", 0.0)),
                    float(meta.get("visual_score", 0.0)),
                    float(previous.get("final_score", 0.0)),
                )
                candidates_by_key[key] = merged
            else:
                candidates_by_key[key] = meta

        candidates = list(candidates_by_key.values())
        for meta in candidates:
            source_evidence = " ".join((
                str(meta.get("source", "")),
                str(meta.get("vision_analysis", "")),
                str(meta.get("_visual_text", "")),
            ))
            quality = self._source_quality_signal(source_evidence, question)
            # Quality adjustment only affects source selection; it does not
            # alter the scores stored in the database.
            meta["_image_selection_score"] = (
                float(meta.get("visual_score", 0.0)) + quality
            )

        candidates.sort(
            key=lambda m: float(m.get("_image_selection_score", 0.0)),
            reverse=True,
        )

        for meta in candidates:
            if not meta.get("has_visual") and not meta.get("image_path"):
                continue

            key = (meta.get("source"), meta.get("page"))
            if key in seen:
                continue
            if len(result) >= self.config.MAX_SOURCE_IMAGES:
                break

            item = self._image_item_from_meta(meta)
            if item is None:
                continue

            seen.add(key)
            result.append(item)

        return result

    def _visual_candidates_from_lexical(self, question, is_cache=None):
        """Sucht unter den bereits gefundenen Textkandidaten gezielt visuelle Seiten.

        Wichtig: Es wird keine Vision-Analyse ausgelöst. Wir nutzen nur die bereits
        vorhandenen Seitenbilder und Metadaten. Dadurch bleibt Vision lazy.
        """
        try:
            lexical = self._lexical_search(
                question,
                self.RETRIEVAL_LEXICAL_CANDIDATES,
                is_cache=is_cache,
            )
        except Exception:
            return []

        docs = lexical.get("documents", [[]])[0]
        metas = lexical.get("metadatas", [[]])[0]
        infos = lexical.get("_retrieval_scores", [[]])[0]

        candidates = []
        for idx, meta in enumerate(metas):
            if not meta:
                continue

            # Ältere Datenbanken haben has_visual/image_path nicht zwingend
            # als Metadatum. Wenn die gerenderte Seite bereits existiert,
            # ist sie trotzdem eine valide visuelle Kandidatin.
            has_visual = bool(meta.get("has_visual")) or bool(meta.get("image_path"))
            if not has_visual:
                try:
                    source = meta.get("source")
                    page = meta.get("page")
                    if source and page is not None:
                        has_visual = self._image_path(source, int(page)).exists()
                except Exception:
                    has_visual = False

            if not has_visual:
                continue

            info = infos[idx] if idx < len(infos) else {}
            score = float(info.get("normalized_score", 0.0))
            doc_text = docs[idx] if idx < len(docs) else ""

            # Backward compatibility: older Chroma entries may still mark
            # text-rich pages as visual. Do not send these pages to Vision
            # just because stale has_visual/image_path metadata exists.
            native_text = doc_text.split("[PDF-TEXT]", 1)[-1].strip() if "[PDF-TEXT]" in doc_text else doc_text.strip()
            if len(native_text) >= self.config.OCR_TEXT_THRESHOLD:
                continue
            score += 0.20 * self._vision_lexical_score(
                question, doc_text, str(meta.get("source", ""))
            )
            score = max(0.0, min(1.0, score))
            candidates.append({
                **meta,
                "visual_score": score,
                "_visual_text": doc_text,
            })

        candidates.sort(
            key=lambda m: float(m.get("visual_score", 0.0)),
            reverse=True,
        )

        return candidates[:self.config.VISUAL_SHORTLIST_CANDIDATES]

    # =========================================================
    # Einzelne Vision-Analyse
    # =========================================================

    def _analyze_single_image(
        self,
        question,
        item
    ):

        print(
            f"[VISION] Analysiere: "
            f"{item['source']} "
            f"Seite {item['page']}"
        )

        prompt = f"""
Du analysierst eine Originalseite aus
wissenschaftlichen Chemie-Unterlagen.

Benutzerfrage:

{question}

Dokument:

{item['source']}

Seite:

{item['page']}

Analysiere das Originalbild. Die Benutzerfrage dient nur dazu,
relevante Elemente zuerst zu beschreiben. Die gespeicherte Analyse
muss trotzdem eine möglichst allgemeine, wiederverwendbare
Beschreibung des tatsächlichen Seiteninhalts sein.

Benutzerfrage als Priorität:
{question}

Achte insbesondere auf:

- Handschrift
- chemische Strukturformeln
- Stereochemie
- Reaktionsmechanismen
- Reaktionsbedingungen
- Reaktionsgleichungen
- chemische Formeln
- Spektren
- Diagramme
- Tabellen
- griechische Buchstaben
- Symbole
- mathematische Formeln
- Seitenrandnotizen
- Markierungen
- Pfeile
- Beispiele

Bei chemischen Strukturen und Formeln ist äußerste
Genauigkeit wichtig.

Wenn etwas nicht eindeutig lesbar ist, kennzeichne
es ausdrücklich als unsicher.

Erfinde niemals fehlende Inhalte.

Erstelle eine strukturierte, quellengetreue Analyse, die dauerhaft als
visueller Index dieser Seite gespeichert werden kann und auch für spätere
andere Fragen sinnvoll bleibt.

Wichtig: Trenne sichtbare Tatsachen von Interpretationen. Wenn ein
Peakwert, eine Beschriftung, Struktur, Zahl oder handschriftliche Notiz
nicht sicher aus dem Bild abgelesen werden kann, schreibe nicht einfach
einen plausiblen Wert, sondern markiere ihn als unsicher.

Beginne mit:
1. SICHTBARE INHALTE – welche chemischen/visuellen Elemente
   tatsächlich auf der Seite vorhanden sind.
2. FRAGENRELEVANTE INHALTE – was davon für die aktuelle Frage
   relevant ist.
3. UNSICHERHEITEN – was nicht eindeutig lesbar ist.

Erfasse, soweit vorhanden, ausdrücklich:
- Spektren mit Achsen, Signalen, δ/ppm, J-Werten und Integralen
- Strukturformeln und Stereochemie
- Reaktionspfeile, Reagenzien und Bedingungen
- Tabellen und Diagramme
- handschriftliche Ergänzungen
- Formeln und Symbole

Diese Analyse wird später für andere Fragen wiederverwendet.
Erfinde nichts.

Maximal:
{self.config.VISION_DESCRIPTION_TARGET} Zeichen.
"""

        response = ollama.chat(
            model=
                self.config.VISION_MODEL,

            messages=[
                {
                    "role":
                        "system",

                    "content":
                        (
                            "Du bist ein äußerst "
                            "sorgfältiger "
                            "wissenschaftlicher "
                            "Bildanalysator für "
                            "Chemie-Unterlagen."
                        ),
                },

                {
                    "role":
                        "user",

                    "content":
                        prompt,

                    "images": [
                        item["image"]
                    ],
                },
            ],

            options={
                "num_ctx":
                    self.config.NUM_CTX,

                "num_predict":
                    self.config.NUM_PREDICT,

                "temperature": 0.15,
            },
        )

        return (
            response["message"]
            ["content"]
            .strip()
        )

    # =========================================================
    # Persistente Lazy Vision
    # =========================================================

    def _vision_for_images(
        self,
        question,
        images
    ):

        if not images:
            return ""

        if not self.config.VISION_ENABLED:
            return ""

        analyses = []

        for item in images:

            vision_hash = (
                item["vision_hash"]
            )

            cached = (
                self.database
                .get_vision_analysis(
                    item["source"],
                    item["page"],
                    vision_hash
                )
            )

            if cached:

                print(
                    f"[VISION] Cache-Hit: "
                    f"{item['source']} "
                    f"Seite {item['page']}"
                )

                analysis = cached

            else:

                analysis = (
                    self._analyze_single_image(
                        question,
                        item
                    )
                )

                vision_embedding = self._embedding(
                    f"{item['source']} Seite {item['page']}\\n{analysis}"
                )

                self.database.add_vision_analysis(
                    item["source"],
                    item["page"],
                    vision_hash,
                    analysis,
                    embedding=vision_embedding,
                    is_cache=bool(item.get("is_cache", False)),
                )

                print(
                    "[VISION] Analyse "
                    "dauerhaft gespeichert."
                )

            analyses.append(
                (
                    f"[BILDANALYSE]\n"
                    f"Datei: "
                    f"{item['source']}\n"
                    f"Seite: "
                    f"{item['page']}\n\n"
                    f"{analysis}"
                )
            )

        return "\n\n".join(
            analyses
        )

    # =========================================================
    # Vision-Retrieval
    # =========================================================

    def _vision_retrieval(self, question, is_cache=None):
        """Persistierte Bildanalysen: semantische + lexikalische Relevanz."""
        if not self.config.VISION_ENABLED:
            return []
        try:
            embedding = self._embedding(question)
            raw = self.database.search_vision(
                embedding,
                self.config.VISION_LEXICAL_CANDIDATES,
                is_cache=is_cache,
            )
        except Exception as error:
            print(f"[VISION] Retrieval übersprungen: {error}")
            return []

        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        result = []
        seen = set()
        profiles = self._query_concept_profile(question)

        for rank, (doc, meta, distance) in enumerate(
            zip(docs, metas, distances), start=1
        ):
            meta = dict(meta or {})
            source = meta.get("source")
            page = meta.get("page")
            if not source or page is None:
                continue
            key = (source, page)
            if key in seen:
                continue

            semantic_score = 1.0 / __import__("math").log2(rank + 1.0)
            if rank > self.config.VISION_RANK_TOP_K:
                semantic_score *= 0.72
            lexical_score = self._vision_lexical_score(question, str(doc or ""), source)
            combined = 0.48 * semantic_score + 0.52 * lexical_score
            # Educational explanations should favor teaching materials over
            # incidental examples in research-paper figures or supplements.
            combined += self._source_quality_signal(
                f"{source} {doc or ''}", question
            )

            doc_norm = self._normalize_search_text(str(doc or ""))
            doc_compact = self._compact_search_text(str(doc or ""))
            for profile in profiles:
                positive, negative, anchors, required = self._concept_evidence(
                    doc_norm, doc_compact, profile
                )
                if anchors:
                    combined += min(0.16, 0.08 * len(anchors))
                if required and profile.get("required_groups"):
                    combined += min(0.12, 0.06 * len(required))
                if negative and not anchors:
                    combined -= min(0.30, 0.15 * len(negative))

            combined = max(0.0, min(1.0, combined))
            if combined < self.config.VISION_MIN_SCORE:
                continue

            meta["has_visual"] = True
            meta["visual_score"] = combined
            meta["vision_retrieval"] = True
            meta["vision_distance"] = float(distance)
            meta["vision_analysis"] = doc
            meta["vision_semantic_score"] = semantic_score
            meta["vision_lexical_score"] = lexical_score
            result.append(meta)
            seen.add(key)

        result.sort(key=lambda m: float(m.get("visual_score", 0.0)), reverse=True)
        result = result[:self.config.VISION_RETRIEVAL_CANDIDATES]
        if result:
            print(f"[VISION] Cache-Kandidaten: {len(result)}")
            for i, meta in enumerate(result[:5], start=1):
                print(
                    f"    {i}. {meta.get('source')} | Seite {meta.get('page')} | "
                    f"visual={meta.get('visual_score', 0.0):.3f} | "
                    f"lex={meta.get('vision_lexical_score', 0.0):.3f}"
                )
        return result

    # =========================================================
    # Antwort
    # =========================================================

    @staticmethod
    def _result_rows(results):
        docs = results.get("documents", [[]])[0] if results else []
        metas = results.get("metadatas", [[]])[0] if results else []
        distances = results.get("distances", [[]])[0] if results else []
        rows = []
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            rows.append((str(doc or ""), dict(meta or {}), float(distances[i]) if i < len(distances) else 0.0))
        return rows

    @staticmethod
    def _pack_result_rows(rows):
        return {
            "documents": [[r[0] for r in rows]],
            "metadatas": [[r[1] for r in rows]],
            "distances": [[r[2] for r in rows]],
        }

    def _resolve_target_cache_source(self, question):
        """Resolve the requested cache document conservatively.

        Exact names and explicit identifiers (e.g. T4 + author) take priority.
        Generic aliases are accepted only when they identify exactly one likely
        protocol. Never silently choose T1 merely because it is a protocol.
        """
        try:
            rows = self._result_rows(self.cache.get_all_results())
        except Exception as error:
            print(f"[CACHE][ZIELDOKUMENT] Cache-Metadaten nicht lesbar: {error}")
            return None
        sources = sorted({str(meta.get("source", "")).strip() for _, meta, _ in rows if meta.get("source")})
        if not sources:
            return None

        q_norm = self._normalize_search_text(question)
        q_compact = self._compact_search_text(question)
        stems = {src: Path(src.replace("\\", "/")).stem for src in sources}

        # 1) Vollständiger Dateiname / eindeutiger Stem gewinnt.
        exact = []
        for src, stem in stems.items():
            stem_norm = self._normalize_search_text(stem)
            stem_compact = self._compact_search_text(stem)
            if stem_norm and stem_norm in q_norm:
                exact.append(src)
            elif len(stem_compact) >= 8 and stem_compact in q_compact:
                exact.append(src)
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            print(f"[CACHE][ZIELDOKUMENT] Mehrdeutiger Dateiname: {exact}")
            return None

        # 2) Explizite Praktikums-/Dokumentkennung (T1, T4, BA etc.)
        #    verhindert, dass Autorname allein ein anderes Praktikum auswählt.
        t_matches = re.findall(r"(?<![a-z0-9])t\s*([0-9]{1,2})(?![0-9])", q_norm)
        if t_matches:
            wanted_t = {f"t{n}" for n in t_matches}
            candidates = []
            for src, stem in stems.items():
                stem_tokens = set(self._tokenize_search_text(stem))
                if wanted_t.intersection(stem_tokens):
                    candidates.append(src)
            if len(candidates) == 1:
                return candidates[0]
            if len(candidates) > 1:
                # Mit zusätzlichem Autor-/Namenssignal weiter eingrenzen.
                name_tokens = [t for t in self._tokenize_search_text(question)
                               if len(t) >= 4 and t not in wanted_t]
                scored = []
                for src in candidates:
                    st = set(self._tokenize_search_text(stems[src]))
                    score = sum(1 for token in name_tokens if token in st)
                    scored.append((score, src))
                best = max((score for score, _ in scored), default=0)
                winners = [src for score, src in scored if score == best and score > 0]
                if len(winners) == 1:
                    return winners[0]
                print(f"[CACHE][ZIELDOKUMENT] Mehrere Dateien mit Kennung {sorted(wanted_t)}: {candidates}")
                return None
            print(f"[CACHE][ZIELDOKUMENT] Explizite Kennung {sorted(wanted_t)} passt zu keiner Cache-Datei.")
            return None

        # 3) Autorname/weitere unterscheidende Dateinamens-Tokens.
        q_tokens = set(self._tokenize_search_text(question))
        scored = []
        for src, stem in stems.items():
            stem_tokens = set(self._tokenize_search_text(stem))
            informative = {t for t in stem_tokens if len(t) >= 4 and t not in {"block", "cache", "labor", "protokoll"}}
            overlap = q_tokens.intersection(informative)
            if overlap:
                scored.append((len(overlap), src))
        if scored:
            best = max(score for score, _ in scored)
            winners = [src for score, src in scored if score == best]
            if len(winners) == 1 and best >= 2:
                return winners[0]
            if len(winners) > 1:
                print(f"[CACHE][ZIELDOKUMENT] Namensangaben mehrdeutig: {winners}")
                return None

        # 4) Generischer Typ-Alias nur bei genau einem plausiblen Kandidaten.
        q = q_norm
        aliases = []
        if any(term in q for term in ("bachelorarbeit", "bachelor thesis", "meine thesis")):
            aliases = [src for src, stem in stems.items()
                       if "bachelorarbeit" in self._normalize_search_text(stem)
                       or self._normalize_search_text(stem).startswith("ba ")]
        elif any(term in q for term in ("laborprotokoll", "versuchsprotokoll", "protokoll im cache", "mein protokoll")):
            aliases = [src for src, stem in stems.items()
                       if any(tag in self._normalize_search_text(stem)
                              for tag in ("protokoll", "t1 block", "t2 block", "t3 block", "t4 block", "t5 block"))]
        if len(aliases) == 1:
            print(f"[CACHE][ZIELDOKUMENT] Typ-Alias eindeutig aufgelöst: {aliases[0]}")
            return aliases[0]
        if len(aliases) > 1:
            print(f"[CACHE][ZIELDOKUMENT] Mehrere mögliche Dokumente für Anfrage: {aliases}")
        return None

    def _target_candidates(self):
        """Return distinct cached sources for a clarification question."""
        try:
            rows = self._result_rows(self.cache.get_all_results())
        except Exception:
            return []
        return sorted({str(meta.get("source", "")).strip() for _, meta, _ in rows if meta.get("source")})

    @staticmethod
    def _pretty_source(source):
        return Path(str(source).replace("\\", "/")).name

    def _cache_results_for_source(self, source):
        """Return all indexed chunks for one exact source path."""
        rows = self._result_rows(self.cache.get_all_results())
        selected = [row for row in rows if str(row[1].get("source", "")) == str(source)]
        selected.sort(key=lambda row: (int(row[1].get("page", 0) or 0), str(row[1].get("chunk_index", ""))))
        return self._pack_result_rows(selected)

    def _select_cache_results_for_question(self, question):
        results = self.cache.get_all_results()
        rows = self._result_rows(results)
        if not rows:
            return results
        sources = sorted({meta.get("source", "") for _, meta, _ in rows if meta.get("source")})
        q = self._normalize_search_text(question)
        # Wenn ein Dateiname im Prompt genannt wird, wähle bevorzugt genau
        # diese Cache-Datei. Bei generischen Cache-Fragen bleiben alle Dateien.
        matched = [src for src in sources if Path(str(src).replace("\\", "/")).stem.lower() in q]
        if not matched:
            matched = [src for src in sources if any(
                len(token) >= 6 and token in self._normalize_search_text(Path(str(src).replace("\\", "/")).stem)
                for token in self._tokenize_search_text(question)
            )]
        if matched:
            rows = [r for r in rows if r[1].get("source") in matched]
        rows.sort(key=lambda r: (str(r[1].get("source", "")).casefold(), int(r[1].get("page", 0) or 0)))
        return self._pack_result_rows(rows)

    def _cache_page_text(self, cache_results):
        grouped = {}
        for doc, meta, _ in self._result_rows(cache_results):
            key = (str(meta.get("source", "")), int(meta.get("page", 0) or 0))
            grouped.setdefault(key, []).append(doc)
        blocks = []
        for (source, page), chunks in sorted(grouped.items(), key=lambda x: (x[0][0].casefold(), x[0][1])):
            text = "\n\n".join(chunk for chunk in chunks if chunk.strip()).strip()
            blocks.append(f"[KLAUSURDOKUMENT: {source} | SEITE {page}]\n{text or '[Kein extrahierbarer Text auf dieser Seite]'}")
        return "\n\n".join(blocks)

    def _cache_visual_images(self, cache_results):
        images = []
        seen = set()
        for _, meta, _ in self._result_rows(cache_results):
            if not meta.get("has_visual") and not meta.get("image_path"):
                continue
            key = (meta.get("source"), meta.get("page"))
            if key in seen:
                continue
            seen.add(key)
            item = self._image_item_from_meta(meta)
            if item:
                item["is_cache"] = True
                images.append(item)
        return images

    def _chat_text(self, system_prompt, content, images=None, num_predict=None):
        """Call Ollama with diagnostics and never silently hide an empty response."""
        message = {"role": "user", "content": content}
        image_count = 0
        if images:
            message["images"] = [item["image"] for item in images if item.get("image")]
            image_count = len(message["images"])

        requested_tokens = int(num_predict or self.config.NUM_PREDICT)
        print(
            f"[OLLAMA] Anfrage gestartet | Modell={self.config.LLM_MODEL} "
            f"| Promptzeichen={len(system_prompt) + len(content)} "
            f"| Bilder={image_count} | num_ctx={self.config.NUM_CTX} "
            f"| num_predict={requested_tokens}",
            flush=True,
        )

        try:
            chat_kwargs = {
                "model": self.config.LLM_MODEL,
                "messages": [{"role": "system", "content": system_prompt}, message],
                "options": {
                    "num_ctx": self.config.NUM_CTX,
                    "num_predict": requested_tokens,
                    "temperature": 0.15,
                },
            }
            # Qwen3.5 can spend the entire generation budget on its separate
            # thinking channel. Disable thinking for this structured RAG task.
            # Older ollama-python versions may not support the `think` argument;
            # in that case retry once without it rather than failing immediately.
            try:
                response = ollama.chat(**chat_kwargs, think=False)
                print("[OLLAMA] Thinking-Modus deaktiviert (think=False).", flush=True)
            except TypeError as think_error:
                if "think" not in str(think_error).lower():
                    raise
                print(
                    "[OLLAMA][HINWEIS] Diese ollama-python-Version unterstützt "
                    "think=False nicht; Wiederholung ohne diesen Parameter.",
                    flush=True,
                )
                response = ollama.chat(**chat_kwargs)
        except Exception as error:
            print(f"[OLLAMA][FEHLER] {type(error).__name__}: {error}", flush=True)
            return (
                "FEHLER bei der Antwortgenerierung über Ollama "
                f"({type(error).__name__}: {error}). "
                "Die Anfrage wurde nicht fachlich ausgewertet."
            )

        # ollama.chat may return a Pydantic ChatResponse rather than a dict.
        # Support both response shapes without calling .keys() on model objects.
        def field(obj, name, default=None):
            if isinstance(obj, dict):
                return obj.get(name, default)
            return getattr(obj, name, default)

        response_message = field(response, "message", {}) or {}
        answer = str(field(response_message, "content", "") or "").strip()
        thinking = str(field(response_message, "thinking", "") or "").strip()
        done_reason = field(response, "done_reason", "nicht angegeben")
        self._last_chat_done_reason = str(done_reason)
        eval_count = field(response, "eval_count", "n/a")
        prompt_eval_count = field(response, "prompt_eval_count", "n/a")
        print(
            f"[OLLAMA] Antwort erhalten | Zeichen={len(answer)} "
            f"| Thinking-Zeichen={len(thinking)} | done_reason={done_reason} "
            f"| eval_count={eval_count} | prompt_eval_count={prompt_eval_count}",
            flush=True,
        )

        if done_reason == "length":
            print(
                "[OLLAMA][WARNUNG] Antwortgenerierung endete am Ausgabelimit.",
                flush=True,
            )

        if not answer:
            print(
                "[OLLAMA][LEERE ANTWORT] Kein sichtbarer Nachrichtentext. "
                f"Response-Typ={type(response).__name__}; "
                f"Message-Typ={type(response_message).__name__}; "
                f"Thinking-Zeichen={len(thinking)}",
                flush=True,
            )
            return (
                "FEHLER: Ollama hat keinen sichtbaren Antworttext zurückgegeben "
                f"(done_reason={done_reason}, thinking_chars={len(thinking)}). "
                "Es wurde keine fachliche Bewertung für diesen Aufruf erzeugt."
            )

        return answer

    def _trace_exam(self, event, **details):
        """Schreibt einen nachvollziehbaren Ablauf-Trace, keine internen Gedankengänge."""
        try:
            if not hasattr(self.config, "DATABASE_DIR"):
                return  # z. B. isolierter Unit-Test mit minimalem Config-Mock
            trace_dir = self.config.DATABASE_DIR / "diagnostics"
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = trace_dir / "exam_diagnostics.jsonl"
            record = {
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                "event": str(event),
                **details,
            }
            with trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as error:
            print(f"[DIAGNOSE] Trace konnte nicht gespeichert werden: {error}")

    def _cache_all_page_images(self, cache_results):
        """Rendert für Klausuren jede Seite bei Bedarf, auch reine Vektorzeichnungen.

        Die normale Indexierung spart Bilder textreicher Seiten. Für eine vollständige
        Klausurerkennung ist diese Heuristik zu schwach: Strukturformeln/Diagramme können
        aus Vektorpfaden bestehen und daher nicht als eingebettetes Bild erkannt werden.
        """
        pages = {}
        for _, meta, _ in self._result_rows(cache_results):
            source = str(meta.get("source", ""))
            page = meta.get("page")
            if source and isinstance(page, int) and page > 0:
                pages[(source.replace("\\", "/").casefold(), page)] = (source, page, meta)

        images = []
        for _, (source, page, meta) in sorted(pages.items(), key=lambda pair: (pair[0][0], pair[0][1])):
            image_path = self._image_path(source, page)
            try:
                if not image_path.exists():
                    pdf_path = self.config.PROJECT_ROOT.joinpath(*source.replace("\\", "/").split("/"))
                    if not pdf_path.exists():
                        self._trace_exam("image_render_missing_pdf", source=source, page=page, pdf_path=str(pdf_path))
                        continue
                    with pymupdf.open(pdf_path) as document:
                        if page > len(document):
                            continue
                        image_path.parent.mkdir(parents=True, exist_ok=True)
                        image_path.write_bytes(self._render_page(document[page - 1]))
                item = self._image_item_from_meta({
                    **meta,
                    "source": source,
                    "page": page,
                    "image_path": str(image_path.relative_to(self.config.PROJECT_ROOT)),
                    "is_cache": True,
                })
                if item:
                    item["is_cache"] = True
                    images.append(item)
            except Exception as error:
                print(f"[KLAUSUR] Originalseite {page} konnte nicht gerendert werden: {error}")
                self._trace_exam("image_render_error", source=source, page=page, error=repr(error))
        self._trace_exam("page_images_ready", page_count=len(pages), image_count=len(images), pages=[{"source": x[0], "page": x[1]} for x in pages.values()])
        return images

    @staticmethod
    def _parse_json_object(raw):
        """Liest das erste JSON-Objekt, ohne gierigen Regex über mehrere Objekte hinweg."""
        decoder = json.JSONDecoder()
        text = str(raw or "").strip()
        for start, char in enumerate(text):
            if char == "{":
                try:
                    value, _ = decoder.raw_decode(text[start:])
                    if isinstance(value, dict):
                        return value
                except json.JSONDecodeError:
                    continue
        raise ValueError("Keine vollständige JSON-Objektantwort gefunden")

    def _audit_exam_coverage(self, cache_text, tasks):
        """Prüft explizite Aufgaben-/Fragenummern im extrahierten PDF-Text gegen die Liste."""
        source_labels = set()
        for match in re.finditer(r"(?i)\b(?:aufgabe|frage|problem)\s*(\d{1,3})\s*([a-z])?", cache_text):
            source_labels.add(match.group(1))
        task_labels = set()
        for task in tasks:
            match = re.search(r"(?i)(?:aufgabe|frage|problem)?\s*(\d{1,3})", str(task.get("id", "")))
            if match:
                task_labels.add(match.group(1))
        missing = sorted(source_labels - task_labels, key=lambda value: int(value))
        report = {
            "explicit_source_question_numbers": sorted(source_labels, key=lambda value: int(value)),
            "recognized_question_numbers": sorted(task_labels, key=lambda value: int(value)),
            "missing_explicit_question_numbers": missing,
            "audit_available": bool(source_labels),
            "coverage_warning": bool(source_labels and missing),
        }
        self._trace_exam("coverage_audit", **report, task_count=len(tasks))
        if report["coverage_warning"]:
            print("[KLAUSUR] WARNUNG: Im PDF-Text erkennbare Aufgabennummern fehlen in der Aufgabenliste: " + ", ".join(missing))
        elif not report["audit_available"]:
            print("[KLAUSUR] Hinweis: Im PDF-Text wurden keine expliziten 'Aufgabe/Frage N'-Marker gefunden; automatische Vollständigkeitsprüfung eingeschränkt.")
        return report

    def diagnose_exam_extraction(self, question="Löse die Klausur im Cache vollständig"):
        """Diagnostiziert Cache-Seiten, Bilder und Aufgabenerkennung ohne die Aufgaben zu lösen."""
        cache_results = self._select_cache_results_for_question(question)
        rows = self._result_rows(cache_results)
        if not rows:
            print("[DIAGNOSE] Keine Cache-Seiten indexiert.")
            return {"tasks": [], "page_count": 0, "error": "no_cache_pages"}
        cache_text = self._cache_page_text(cache_results)
        images = self._cache_all_page_images(cache_results)
        self._trace_exam("diagnostic_start", question=question, cache_rows=len(rows), cache_text_chars=len(cache_text), image_count=len(images))
        print(f"[DIAGNOSE] Seiten/Chunks: {len(rows)} | gerenderte Originalseiten: {len(images)} | Textzeichen: {len(cache_text)}")
        visual = self._vision_for_images(
            "Inventarisiere diese Klausurseite vollständig für die Aufgabenextraktion. Lies sichtbare Aufgaben, Unteraufgaben, Formeln, Tabellen und Diagramme ab. Löse nichts; markiere unleserliche Stellen.",
            images,
        ) if images else ""
        tasks, raw = self._extract_exam_tasks(question, cache_text, visual, images)
        audit = self._audit_exam_coverage(cache_text, tasks)
        print(f"[DIAGNOSE] Erkannte Aufgaben-/Teilaufgaben: {len(tasks)}")
        for task in tasks:
            print(f"  - {task.get('id')} | Seite {task.get('page')} | {len(task.get('text', ''))} Zeichen")
        print(f"[DIAGNOSE] Trace: {self.config.DATABASE_DIR / 'diagnostics' / 'exam_diagnostics.jsonl'}")
        return {"tasks": tasks, "raw": raw, "page_count": len(rows), "image_count": len(images), "audit": audit}

    @staticmethod
    def _fallback_tasks_from_page_text(source, page, page_text):
        """Deterministischer Rückfall, wenn das Modell kein JSON liefert.

        Extrahiert nur explizit markierte Aufgaben/Teilaufgaben und erfindet
        keine Aufgaben aus bloßen Imperativwörtern. Das ist besonders wichtig,
        wenn die Modellantwort wegen zu vieler Tokens abgeschnitten wurde.
        """
        text = str(page_text or "").strip()
        if not text:
            return []
        pattern = re.compile(
            r"(?i)\b(?:aufgabe|frage|problem)\s*(\d{1,3})\s*([a-z])?\s*[:.)-]?"
        )
        matches = list(pattern.finditer(text))
        tasks = []
        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            block = re.sub(r"\s+", " ", text[start:end]).strip()
            # Seitenkopf/-fuß und bloße Referenzen wie "siehe Aufgabe 4" sind
            # keine belastbaren Aufgaben. Mindestinhalt nach dem Marker nötig.
            body = block[match.end() - start:].strip(" :.)-\t")
            if len(body) < 12:
                continue
            number = match.group(1)
            letter = (match.group(2) or "").lower()
            task_id = f"Aufgabe {number}{letter}"
            tasks.append({
                "id": task_id,
                "page": page if isinstance(page, int) else None,
                "text": block,
                "visual_required": bool(re.search(r"(?i)(abbildung|diagramm|spektrum|strukturformel|siehe bild|siehe grafik)", block)),
                "source": source,
            })
        return tasks

    def _has_indexed_cache_documents(self):
        """Prüft den Cache anhand der tatsächlichen indexierten Zeilen.

        Einige ältere Chroma-Indizes liefern bei `where={is_cache: True}`
        inkonsistente Ergebnisse. Daher wird nicht ausschließlich auf
        CacheManager.has_documents() vertraut.
        """
        try:
            all_results = self.cache.get_all_results()
            if self._result_rows(all_results):
                return True
        except Exception:
            pass
        try:
            if self.cache.has_documents():
                return True
        except Exception:
            pass
        # Letzter Fallback: Der Cache-Sync hat die PDF-Dateien gerade geprüft.
        # Alte Indizes können is_cache-Metadaten falsch typisieren, obwohl die
        # Cache-PDFs und ihre Chunks vorhanden sind.
        try:
            cache_dir = self.config.CACHE_DOCUMENTS_DIR
            if cache_dir.exists() and any(
                path.is_file() and path.suffix.lower() == ".pdf"
                for path in cache_dir.rglob("*")
            ):
                return True
        except Exception:
            pass
        return False

    def _extract_exam_tasks(self, question, cache_text, visual_analysis, cache_images):
        """Extrahiert Aufgaben seitenweise, damit lange Klausuren nicht abgeschnitten werden."""
        page_pattern = re.compile(
            r"\[KLAUSURDOKUMENT:\s*(.*?)\s*\|\s*SEITE\s+(\d+)\]\s*\n(.*?)"
            r"(?=\n\s*\n\[KLAUSURDOKUMENT:|\Z)",
            flags=re.S | re.I,
        )
        page_blocks = [
            {"source": source.strip(), "page": int(page), "text": text.strip()}
            for source, page, text in page_pattern.findall(cache_text)
        ]

        # Die visuellen Analysen werden ebenfalls nach Quelle und Seite zugeordnet.
        visual_by_page = {}
        if visual_analysis:
            visual_blocks = re.split(r"(?=\[BILDANALYSE\])", visual_analysis)
            for block in visual_blocks:
                source_match = re.search(r"(?m)^Datei:\s*(.+?)\s*$", block)
                page_match = re.search(r"(?m)^Seite:\s*(\d+)\s*$", block)
                if source_match and page_match:
                    key = (
                        source_match.group(1).strip().replace("\\", "/").casefold(),
                        int(page_match.group(1)),
                    )
                    visual_by_page[key] = block.strip()

        image_by_page = {}
        for item in cache_images or []:
            key = (str(item.get("source", "")).replace("\\", "/").casefold(), item.get("page"))
            image_by_page.setdefault(key, item)

        # Alte/unerwartete Textformate werden weiterhin unterstützt, aber klar als
        # ein Gesamtblock verarbeitet statt stillschweigend Inhalt zu verlieren.
        if not page_blocks:
            page_blocks = [{"source": "unbekannt", "page": None, "text": cache_text}]

        tasks = []
        raw_outputs = []
        seen_task_ids = {}
        for page_block in page_blocks:
            source = page_block["source"]
            page = page_block["page"]
            page_text = page_block["text"]
            source_key = source.replace("\\", "/").casefold()
            visual_text = visual_by_page.get((source_key, page), "")
            page_image = image_by_page.get((source_key, page))
            prompt = f"""
Lies NUR diese eine Seite einer Chemie-Klausur aus. Extrahiere alle auf dieser Seite
sichtbaren Aufgaben und Teilaufgaben; löse nichts. Ignoriere Deckblatt, organisatorische
Hinweise und Lösungsschlüssel als Aufgaben. Wenn eine Aufgabe auf dieser Seite fortgesetzt
wird, gib nur den auf dieser Seite erkennbaren Wortlaut aus und verwende dieselbe Aufgaben-ID.
Erfinde keinen fehlenden Text. Jede Teilaufgabe soll einen eigenen Eintrag erhalten.

Gib ausschließlich gültiges JSON zurück:
{{"tasks":[{{"id":"Aufgabe 1a","page":{page if page is not None else 'null'},"text":"Wortlaut dieser Aufgabe/Teilaufgabe","visual_required":false}}]}}
Wenn diese Seite keine Aufgaben enthält, gib {{"tasks":[]}} zurück.
Wenn etwas nur im Bild lesbar ist, übernimm es vorsichtig und markiere unleserliche Stellen.

Nutzerauftrag: {question}
Quelldatei: {source}
Seite: {page if page is not None else 'unbekannt'}

PDF-TEXT DIESER SEITE:
{page_text[:9000]}

VISUELLE ANALYSE DIESER SEITE:
{visual_text[:4500] if visual_text else 'Keine persistierte Seitenanalyse verfügbar.'}
"""
            raw = ""
            parse_error = None
            page_tasks = None
            for attempt in range(2):
                try:
                    attempt_prompt = prompt
                    if attempt:
                        attempt_prompt += "\n\nWIEDERHOLUNG: Deine vorige Ausgabe war kein gültiges JSON. Gib exakt ein JSON-Objekt mit dem Schlüssel tasks aus. Keine Markdown-Codezäune, keine Kommentare, keine Einleitung."
                    raw = self._chat_text(
                        "Du bist ein sorgfältiger Klausur-Parser. Antworte ausschließlich mit gültigem JSON. Keine Lösungen und keine erfundenen Aufgaben.",
                        attempt_prompt,
                        images=[page_image] if page_image else None,
                        num_predict=max(1600, min(2600, self.config.NUM_PREDICT)),
                    )
                    data = self._parse_json_object(raw)
                    candidate_tasks = data.get("tasks", [])
                    if not isinstance(candidate_tasks, list):
                        raise ValueError("JSON-Schlüssel 'tasks' ist keine Liste")
                    task_cues = re.search(
                        r"(?i)\b(aufgabe|frage|berechnen|bestimmen|erklären|erklaeren|nennen|beschreiben|begründen|begruenden|zeichnen|geben sie an|leiten sie|formulieren|reaktionsgleichung)\b",
                        page_text + "\n" + visual_text,
                    )
                    if not candidate_tasks and task_cues:
                        raise ValueError("leere Aufgabenliste trotz aufgabentypischer Marker auf der Seite")
                    page_tasks = candidate_tasks
                    parse_error = None
                    break
                except Exception as error:
                    parse_error = f"{type(error).__name__}: {error}"
            raw_outputs.append(f"[SEITE {page}]\n{raw}")
            if parse_error:
                fallback_tasks = self._fallback_tasks_from_page_text(source, page, page_text)
                if fallback_tasks:
                    page_tasks = fallback_tasks
                    print(
                        f"[KLAUSUR][FALLBACK] Seite {page}: "
                        f"{len(fallback_tasks)} explizit markierte Aufgaben aus PDF-Text übernommen "
                        f"(JSON-Extraktion fehlgeschlagen)."
                    )
                    self._trace_exam(
                        "page_extraction_fallback", source=source, page=page,
                        text_chars=len(page_text), has_image=bool(page_image),
                        error=parse_error, task_ids=[item["id"] for item in fallback_tasks],
                        raw_tail=raw[-1800:],
                    )
                    raw_outputs.append(f"[SEITE {page} – DETERMINISTISCHER FALLBACK]\n{raw}")
                else:
                    print(f"[KLAUSUR] JSON-Extraktion auf Seite {page} nach 2 Versuchen fehlgeschlagen: {parse_error}")
                    raw_outputs.append(f"[PARSERFEHLER Seite {page}: {parse_error}]")
                    self._trace_exam("page_extraction_failed", source=source, page=page, text_chars=len(page_text), has_image=bool(page_image), raw_tail=raw[-2500:], error=parse_error)
                    continue
            page_task_ids = [str(x.get("id", "")) for x in page_tasks if isinstance(x, dict)]
            print(f"[KLAUSUR][EXTRAKTION] Seite {page}: {len(page_tasks)} Aufgaben/Teilaufgaben" + (f" ({', '.join(page_task_ids)})" if page_task_ids else ""))
            self._trace_exam("page_extraction", source=source, page=page, text_chars=len(page_text), has_image=bool(page_image), extracted_task_count=len(page_tasks), task_ids=page_task_ids, raw_response_chars=len(raw), raw_response_tail=raw[-1800:])

            for task in page_tasks:
                if not isinstance(task, dict):
                    continue
                task_text = str(task.get("text", "")).strip()
                if not task_text:
                    continue
                task_id = str(task.get("id") or f"Aufgabe auf Seite {page}").strip()
                cleaned = {
                    "id": task_id,
                    # Die aktuelle Seitenquelle ist zuverlässiger als eine vom Modell
                    # halluzinierte oder falsch zugeordnete Seitenzahl.
                    "page": page if isinstance(page, int) else None,
                    "text": task_text,
                    "visual_required": bool(task.get("visual_required", False)),
                    "source": source,
                }
                identity = re.sub(r"\s+", " ", task_id.casefold()).strip()
                if identity in seen_task_ids:
                    previous = tasks[seen_task_ids[identity]]
                    previous_norm = self._normalize_search_text(previous["text"])
                    current_norm = self._normalize_search_text(task_text)
                    if current_norm and current_norm not in previous_norm:
                        previous["text"] += f"\n\n[Fortsetzung auf Seite {page}]\n{task_text}"
                        previous["visual_required"] = previous["visual_required"] or cleaned["visual_required"]
                    continue
                seen_task_ids[identity] = len(tasks)
                tasks.append(cleaned)

        return tasks, "\n\n".join(raw_outputs)

    def _answer_cached_exam(self, question):
        """Löst Cache-Klausuren Aufgabe für Aufgabe und prüft die Abdeckung."""
        cache_results = self._select_cache_results_for_question(question)
        cache_rows = self._result_rows(cache_results)
        if not cache_rows:
            return (
                "Im Cache sind keine indexierten PDF-Inhalte vorhanden. "
                "Lege die Klausur als PDF unter cache/ ab und versuche es erneut.",
                cache_results,
            )

        cache_text = self._cache_page_text(cache_results)
        # Bei einer explizit angeforderten Einzel-/Hauptaufgabe zuerst nur die
        # wahrscheinlichen Aufgabenseiten behalten. Damit werden nicht mehr
        # pauschal alle Klausurseiten visuell analysiert. Wenn keine sichere
        # Seitenerkennung gelingt, brechen wir gezielt ab statt die ganze Klausur
        # als Ersatz zu bearbeiten.
        normalized_request = self._normalize_search_text(question)
        early_target = re.search(r"\b(?:teil)?aufgabe\s*(\d+\s*[a-z]?)\b", normalized_request)
        early_ordinal = re.search(
            r"\b(erste[nrsm]?|zweite[nrsm]?|dritte[nrsm]?|vierte[nrsm]?|fuenfte[nrsm]?|funfte[nrsm]?|"
            r"sechste[nrsm]?|siebte[nrsm]?|achte[nrsm]?|neunte[nrsm]?|zehnte[nrsm]?)\s+(?:teil)?aufgaben?\b",
            normalized_request,
        )
        early_num = early_target.group(1).replace(" ", "").casefold() if early_target else None
        if not early_num and early_ordinal:
            early_words = {"erste":"1", "zweite":"2", "dritte":"3", "vierte":"4", "fuenfte":"5", "funfte":"5", "sechste":"6", "siebte":"7", "achte":"8", "neunte":"9", "zehnte":"10"}
            word = early_ordinal.group(1)
            early_num = next((v for k,v in early_words.items() if word.startswith(k)), None)
        explicit_target_request = bool(early_num or re.search(r"\b(nur|ausschließlich|einzelne?|bestimmte|konkrete)\b.{0,50}\b(aufgabe|teilaufgabe)\b", normalized_request))
        targeted_pages = set()
        targeted_page_pairs = set()
        if early_num:
            target_number = re.match(r"(\d+)", early_num).group(1)
            target_letter = re.search(r"\d+([a-z])$", early_num)
            page_pattern = re.compile(rf"(?i)(?:teil)?aufgabe\s*{re.escape(target_number)}[a-z]?|\b{re.escape(target_number)}[a-z]\b")
            page_pattern_blocks = re.compile(r"\[KLAUSURDOKUMENT:\s*(.*?)\s*\|\s*SEITE\s+(\d+)\]\s*\n(.*?)(?=\n\s*\n\[KLAUSURDOKUMENT:|\Z)", re.S|re.I)
            matched_blocks=[]
            for source_name, page_num, page_body in page_pattern_blocks.findall(cache_text):
                if page_pattern.search(page_body):
                    targeted_pages.add(int(page_num))
                    targeted_page_pairs.add((str(source_name).replace("\\", "/").casefold(), int(page_num)))
                    matched_blocks.append((source_name, page_num, page_body))
            if not targeted_pages:
                self._trace_exam("target_page_not_found", requested_task=early_num, pages_examined=len(page_pattern_blocks.findall(cache_text)))
                return (f"Ich konnte die Seiten für Aufgabe {early_num} im freigegebenen Klausurtext nicht sicher lokalisieren. "
                        "Ich analysiere deshalb nicht ersatzweise die gesamte Klausur. Prüfe die PDF-Auswahl/OCR oder nenne die Seitenzahl.", cache_results)
            cache_text = "\n\n".join(f"[KLAUSURDOKUMENT: {src} | SEITE {pg}]\n{body}" for src,pg,body in matched_blocks)
            cache_rows = [row for row in cache_rows if (str((row[1] or {}).get("source", "")).replace("\\", "/").casefold(), int((row[1] or {}).get("page", 0) or 0)) in targeted_page_pairs]
            # Filter the underlying Chroma result too, so rendering and citations
            # cannot accidentally reintroduce unrelated pages from the same exam.
            if isinstance(cache_results, dict):
                filtered = dict(cache_results)
                for field in ("documents", "metadatas", "distances", "ids"):
                    groups = cache_results.get(field)
                    if not groups or not isinstance(groups, list):
                        continue
                    filtered_groups=[]
                    for group in groups:
                        if field == "metadatas":
                            indexes=[i for i,meta in enumerate(group or []) if (str((meta or {}).get("source", "")).replace("\\", "/").casefold(), int((meta or {}).get("page", 0) or 0)) in targeted_page_pairs]
                        else:
                            metas=(cache_results.get("metadatas") or [[]])[0]
                            indexes=[i for i,meta in enumerate(metas or []) if (str((meta or {}).get("source", "")).replace("\\", "/").casefold(), int((meta or {}).get("page", 0) or 0)) in targeted_page_pairs]
                        filtered_groups.append([group[i] for i in indexes if i < len(group)])
                    filtered[field]=filtered_groups
                cache_results=filtered
            print(f"[KLAUSUR] Zielseitenfilter: Aufgabe {early_num} -> Seiten {sorted(targeted_pages)}; übrige Klausurseiten werden nicht weiter analysiert.", flush=True)
        elif explicit_target_request:
            self._trace_exam("target_task_unclear", question=question)
            return ("Du möchtest eine einzelne Aufgabe bearbeiten, aber ich konnte die Nummer nicht sicher erkennen. "
                    "Ich starte deshalb keine Vollklausuranalyse. Bitte nenne Aufgabe/Teilaufgabe oder Seitenzahl.", cache_results)

        # Nur die im Zielseitenfilter verbleibenden Seiten rendern. Bei einer
        # vollständigen Klausuranfrage bleibt der vollständige Ablauf erhalten.
        if early_num and targeted_pages:
            all_images = self._cache_all_page_images(cache_results)
            cache_images = [img for img in all_images if img.get("page") is not None and (str(img.get("source", "")).replace("\\", "/").casefold(), int(img.get("page"))) in targeted_page_pairs]
        else:
            cache_images = self._cache_all_page_images(cache_results)
        print(
            f"[KLAUSUR] Cache-Seiten/Chunks: {len(cache_rows)} | "
            f"gerenderte Originalseiten: {len(cache_images)} | Textzeichen: {len(cache_text)}"
        )
        self._trace_exam("exam_start", question=question, cache_rows=len(cache_rows), cache_text_chars=len(cache_text), image_count=len(cache_images), sources=sorted({str(meta.get('source')) for _, meta, _ in cache_rows}))

        visual_analysis = self._vision_for_images(
            "Erfasse die Seite vollständig zur Aufgabenerkennung. Beschreibe Aufgaben, "
            "Teilaufgaben, Tabellen, Formeln, Strukturen, Reaktionspfeile und unleserliche "
            "Stellen. Löse nichts und ergänze nichts, was nicht sichtbar ist.",
            cache_images,
        ) if cache_images else ""

        tasks, extraction_raw = self._extract_exam_tasks(
            question, cache_text, visual_analysis, cache_images
        )
        if not tasks:
            tasks = [{
                "id": "Gesamte Klausur (Aufgabenerkennung unsicher)",
                "page": None,
                "text": (
                    "Bearbeite alle Aufgaben, die im folgenden Klausurtext sicher erkennbar "
                    "sind. Nenne unleserliche oder nicht sicher trennbare Stellen ausdrücklich.\n\n"
                    + cache_text[:16000]
                ),
                "visual_required": bool(cache_images),
            }]
            print(
                "[KLAUSUR] WARNUNG: Aufgabenliste nicht sicher extrahiert; "
                "Gesamtdokument-Fallback aktiv. Vollständigkeit ist nicht garantiert."
            )

        extraction_uncertain = "[PARSERFEHLER" in extraction_raw
        coverage_audit = self._audit_exam_coverage(cache_text, tasks)
        if extraction_uncertain:
            print("[KLAUSUR] WARNUNG: Mindestens eine Seite konnte nicht zuverlässig als Aufgabenliste geparst werden.")
            print("[KLAUSUR] Erkannte Aufgabenliste kann unvollständig sein; Einzelaufgaben werden nur bei eindeutigem Treffer gelöst.")

        # Doppelte IDs verhindern, dass eine Aufgabe bei der Kontrolle überschrieben wird.
        seen_ids = set()
        for i, task in enumerate(tasks, start=1):
            task_id = str(task.get("id") or f"Aufgabe {i}").strip()
            if task_id in seen_ids:
                task_id = f"{task_id} (Eintrag {i})"
            seen_ids.add(task_id)
            task["id"] = task_id

        # Wenn der Nutzer eine konkrete Aufgabennummer nennt (z. B. "Aufgabe 1a"),
        # darf der Klausurmodus ausschließlich diesen Eintrag lösen. Die Extraktion
        # kann weiterhin alle Seiten prüfen, aber die teuren Retrieval-/LLM-Aufrufe
        # werden auf die gewünschte Teilaufgabe begrenzt.
        normalized_question = self._normalize_search_text(question)
        requested_match = re.search(
            r"\b(?:teilaufgabe|aufgabe)\s*(\d+\s*[a-z]?)\b",
            normalized_question,
        )
        # Auch ausgeschriebene Ordinalzahlen wie „zweite Aufgabe“ erkennen.
        # Ohne diese Zuordnung fiel die Anfrage in den Vollklausurmodus.
        ordinal_map = {
            "erste": "1", "ersten": "1", "erstes": "1",
            "zweite": "2", "zweiten": "2", "zweites": "2",
            "dritte": "3", "dritten": "3", "drittes": "3",
            "vierte": "4", "vierten": "4", "viertes": "4",
            "fuenfte": "5", "funfte": "5", "fünfte": "5", "fünften": "5",
            "sechste": "6", "sechsten": "6", "siebte": "7", "siebten": "7",
            "achte": "8", "achten": "8", "neunte": "9", "neunten": "9",
            "zehnte": "10", "zehnten": "10",
        }
        ordinal_match = re.search(
            r"\b(erste[nrsm]?|zweite[nrsm]?|dritte[nrsm]?|vierte[nrsm]?|"
            r"fuenfte[nrsm]?|funfte[nrsm]?|fünfte[nrsm]?|sechste[nrsm]?|"
            r"siebte[nrsm]?|achte[nrsm]?|neunte[nrsm]?|zehnte[nrsm]?)\s+"
            r"(?:teil)?aufgaben?\b",
            normalized_question,
        )
        if not requested_match and ordinal_match:
            ordinal_word = ordinal_match.group(1)
            requested_number = next(
                (number for word, number in ordinal_map.items() if ordinal_word.startswith(word)),
                None,
            )
            if requested_number:
                requested_match = re.match(r"(.*)", requested_number)
        # Detect requests that clearly ask for a single task even if the numbering
        # phrase was unusual. In that case never fall through to full-exam mode.
        targeted_intent = bool(re.search(
            r"\b(nur|ausschließlich|einzelne?|bestimmte|konkrete)\b.{0,55}\b(aufgabe|teilaufgabe|frage)\b|"
            r"\b(aufgabe|teilaufgabe)\b.{0,30}\b(löse|loese|lose|lösen|loesen|bearbeite|beantworte|rechne)\b",
            normalized_question,
        ))
        if requested_match:
            requested_id = re.sub(r"\s+", "", requested_match.group(1)).casefold()
            # Ein Ziel mit Buchstaben (z. B. 2b) ist eine konkrete Teilaufgabe.
            # Eine reine Nummer (z. B. "zweite Aufgabe" / "Aufgabe 2") bezeichnet
            # dagegen die ganze Hauptaufgabe und muss 2a, 2b, 2c usw. umfassen.
            explicit_subtask = bool(re.search(r"\d+[a-z]$", requested_id))
            matching_tasks = []
            for task in tasks:
                task_id_norm = re.sub(r"\s+", "", str(task.get("id", "")).casefold())
                task_match = re.search(r"(?:teil)?aufgabe(\d+)([a-z]?)", task_id_norm)
                if not task_match:
                    continue
                number, letter = task_match.group(1), task_match.group(2)
                requested_num_match = re.fullmatch(r"(\d+)([a-z]?)", requested_id)
                if not requested_num_match:
                    continue
                requested_num, requested_letter = requested_num_match.groups()
                if number != requested_num:
                    continue
                if explicit_subtask:
                    if letter == requested_letter:
                        matching_tasks.append(task)
                else:
                    matching_tasks.append(task)
            if not matching_tasks:
                msg = (
                    f"Ich habe Aufgabe {requested_id} in der erkannten Aufgabenliste nicht sicher gefunden. "
                    "Ich habe deshalb nicht ersatzweise die gesamte Klausur gelöst. "
                    "Prüfe die Aufgabenextraktion oder nenne die Seitenzahl."
                )
                print(f"[KLAUSUR] Einzelaufgabe {requested_id} nicht gefunden; keine Massenbearbeitung gestartet.")
                self._trace_exam("requested_task_not_found", requested_task=requested_id, available_task_ids=[str(t.get("id", "")) for t in tasks])
                return msg, cache_results
            if explicit_subtask:
                tasks = matching_tasks[:1]
                print(f"[KLAUSUR] Einzelteilaufgabenmodus: nur Aufgabe {requested_id} wird gelöst.")
            else:
                tasks = matching_tasks
                print(
                    f"[KLAUSUR] Hauptaufgabenmodus: Aufgabe {requested_id} umfasst "
                    f"{len(tasks)} erkannte Teilaufgabe(n): "
                    + ", ".join(str(t.get("id", "")) for t in tasks)
                )
        elif targeted_intent:
            msg = ("Du hast eine einzelne Aufgabe angefordert, aber ich konnte die gewünschte "
                   "Aufgabennummer nicht sicher erkennen. Ich löse deshalb nicht die gesamte Klausur. "
                   "Bitte nenne z. B. 'Aufgabe 2', 'Aufgabe 2b' oder die PDF-Seite.")
            print("[KLAUSUR] Zielaufgabe nicht sicher erkennbar; Vollklausurmodus blockiert.")
            self._trace_exam("targeted_task_not_parsed", question=question, available_task_ids=[str(t.get("id", "")) for t in tasks])
            return msg, cache_results
        else:
            print(f"[KLAUSUR] {len(tasks)} Aufgaben-/Teilaufgaben-Einträge erkannt; vollständiger Klausurmodus.")

        combined_rows = list(cache_rows)
        answer_blocks = []
        completion = []

        for index, task in enumerate(tasks, start=1):
            task_id = task["id"]
            task_query = task["text"]
            print(f"[KLAUSUR] Löse {task_id} ({index}/{len(tasks)}) …")

            try:
                retrieved = self.search_documents(task_query, is_cache=False)
                retrieved = self._diversify_results(retrieved)
                retrieved_rows = self._result_rows(retrieved)
                combined_rows.extend(retrieved_rows)
                permanent_context = self._context(retrieved).strip()
                self._trace_exam("task_retrieval", task_id=task_id, query_chars=len(task_query), result_count=len(retrieved_rows), sources=[{"source": m.get("source"), "page": m.get("page"), "is_cache": bool(m.get("is_cache", False))} for _, m, _ in retrieved_rows[:12]], context_chars=len(permanent_context))
                if not permanent_context:
                    permanent_context = (
                        "Keine ausreichend relevanten permanenten RAG-Fachquellen gefunden. "
                        "Falls allgemeines Chemiewissen verwendet wird, kennzeichne es ausdrücklich "
                        "als [ALLGEMEINES FACHWISSEN]. Erfinde keine Quellen."
                    )

                page = task.get("page")
                task_cache_images = [
                    item for item in cache_images
                    if page is not None and item.get("page") is not None
                    and int(item.get("page")) == int(page)
                ]
                # Wenn die Seite unbekannt ist, nur wenige Bilder beilegen, damit nicht
                # sämtliche Klausurseiten bei jeder Teilaufgabe erneut im Kontext landen.
                if not task_cache_images and page is None:
                    task_cache_images = cache_images[:2]

                rag_images = self._source_images(retrieved, question=task_query)
                task_images = []
                image_seen = set()
                for item in task_cache_images + rag_images[:2]:
                    image_key = (
                        str(item.get("source", "")).replace("\\", "/").casefold(),
                        item.get("page"),
                    )
                    if image_key not in image_seen and len(task_images) < 3:
                        task_images.append(item)
                        image_seen.add(image_key)

                page_note = f"Seite {page}" if page else "Seite nicht sicher erkannt"
                content = f"""
BEARBEITE NUR DIESE EINE KLAUSURAUFGABE
Aufgaben-ID: {task_id}
Dokumentseite: {page_note}

AUFGABENSTELLUNG AUS DEM CACHE (maßgeblich für Zahlen und Wortlaut):
{task_query}

PASSENDE FACHQUELLEN AUS DEM PERMANENTEN RAG:
{permanent_context[:5200]}

QUELLENREGELN
- Das Cache-Dokument liefert nur Aufgabenstellung, Werte und Abbildungen, nicht den Beleg für die fachliche Lösung.
- Bevorzuge passende permanente RAG-Fachquellen und nenne Dokument/Seite, wenn vorhanden.
- Wenn die permanenten Quellen nicht ausreichen, darfst du Fachwissen verwenden, musst es aber als [ALLGEMEINES FACHWISSEN] markieren.
- Erfinde keine Quellen, Messwerte, Strukturdetails oder Reaktionsbedingungen.
- Bei unleserlicher Abbildung oder Handschrift benenne genau die Unsicherheit.
- Bei Rechnungen: Formel, eingesetzte Werte, Einheiten und Ergebnis zeigen.
- Beantworte alle Teilfragen, die in dieser Aufgabenstellung enthalten sind.
- Gib ausschließlich die Lösung für {task_id} aus. Keine anderen Klausuraufgaben bearbeiten.
"""
                solution = self._chat_text(
                    self.config.SYSTEM_PROMPT,
                    content,
                    images=task_images,
                    num_predict=max(1800, min(2600, self.config.NUM_PREDICT)),
                ).strip()
                if not solution:
                    raise RuntimeError("Das Modell hat eine leere Antwort zurückgegeben.")
                # Separate verification pass: check derivations, chemistry, arithmetic,
                # units and whether the response answered only the requested task.
                # This is a visible quality-control pass, not a claim of infallibility.
                if getattr(self.config, "VERIFY_TASK_ANSWERS", True) and not solution.startswith("FEHLER:"):
                    verify_prompt = f"""Prüfe die folgende vorgeschlagene Lösung kritisch und unabhängig.
Prüfe: (1) fachliche Aussagen, (2) Rechenweg und Zahlen, (3) Einheiten und Vorzeichen,
(4) ob alle Teilfragen der angegebenen Aufgabe beantwortet wurden, (5) ob etwas erfunden
oder nicht durch die Aufgabe/Fachquellen gestützt wurde, (6) ob versehentlich andere
Aufgaben gelöst wurden. Nutze die Aufgabenstellung und Quellen unten. Wenn du einen Fehler
findest, korrigiere ihn. Wenn die Lösung korrekt ist, gib sie klar und ohne unnötige
Wiederholung aus. Markiere verbleibende Unsicherheiten. Erfinde keine Quellen.

AUFGABE {task_id}:
{task_query[:7000]}

FACHQUELLEN:
{permanent_context[:3500]}

ZU PRÜFENDE LÖSUNG:
{solution[:10000]}"""
                    reviewed = self._chat_text(
                        "Du bist ein unabhängiger fachlicher Prüfer für Chemie-Aufgaben. Suche aktiv nach Fehlern; bestätige nicht automatisch.",
                        verify_prompt,
                        images=task_images,
                        num_predict=int(getattr(self.config, "TASK_VERIFICATION_TOKENS", 1800)),
                    ).strip()
                    if reviewed and not reviewed.startswith("FEHLER:"):
                        solution = reviewed
                    self._trace_exam("task_verification", task_id=task_id, status="completed" if reviewed else "empty", reviewed_chars=len(reviewed or ""))
                self._trace_exam("task_answer", task_id=task_id, status="ok", answer_chars=len(solution), image_count=len(task_images))
                answer_blocks.append(f"## {task_id} ({page_note})\n\n{solution}")
                completion.append((task_id, True, "Antwort erzeugt"))
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                answer_blocks.append(
                    f"## {task_id} (NICHT VOLLSTÄNDIG BEARBEITET)\n\n"
                    f"Für diese Aufgabe konnte keine Antwort erzeugt werden. Technischer Fehler: {message}"
                )
                completion.append((task_id, False, message))
                self._trace_exam("task_answer", task_id=task_id, status="error", error=message)
                print(f"[KLAUSUR] FEHLER bei {task_id}: {message}")

        completed = sum(1 for _, ok, _ in completion if ok)
        checklist = [
            "## Vollständigkeitskontrolle",
            f"Bearbeitete Aufgaben in diesem Auftrag: {len(tasks)} | Antworten erzeugt: {completed}/{len(tasks)}",
            "",
        ]
        checklist.extend(
            f"- [{'x' if ok else ' '}] {task_id} — {status}"
            for task_id, ok, status in completion
        )
        if completed != len(tasks):
            checklist.extend([
                "",
                "**WARNUNG:** Mindestens eine Aufgabe wurde technisch nicht beantwortet. "
                "Die Klausurlösung ist daher nicht vollständig.",
            ])
        else:
            checklist.extend([
                "",
                "Hinweis: Diese Kontrolle bestätigt, dass für jeden erkannten Eintrag eine "
                "Antwort erzeugt wurde. Sie beweist nicht, dass die Aufgabenerkennung selbst "
                "alle Aufgaben des PDF erfasst hat.",
            ])
        if extraction_uncertain or coverage_audit.get("coverage_warning"):
            checklist.extend([
                "",
                "**WARNUNG ZUR AUFGABENERKENNUNG:** Die Vollständigkeit der Aufgabenliste ist nicht bestätigt.",
            ])
            if coverage_audit.get("missing_explicit_question_numbers"):
                checklist.append("Im PDF-Text gefundene, aber nicht sicher erkannte Aufgabennummern: " + ", ".join(coverage_audit["missing_explicit_question_numbers"]))
        self._trace_exam("exam_complete", recognized_tasks=len(tasks), completed=completed, failed=len(tasks)-completed, extraction_uncertain=extraction_uncertain, coverage_audit=coverage_audit)

        answer = "\n\n---\n\n".join(answer_blocks + ["\n".join(checklist)])
        self.conversation_history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer[:self.config.MAX_HISTORY_MESSAGE_CHARS]},
        ])
        self.conversation_history = self.conversation_history[-self.config.MAX_HISTORY_MESSAGES:]

        unique = []
        seen = set()
        for row in combined_rows:
            meta = row[1]
            source_norm = str(meta.get("source", "")).replace("\\", "/").casefold()
            key = (source_norm, meta.get("page"), bool(meta.get("is_cache", False)), row[0][:100])
            if key in seen:
                continue
            seen.add(key)
            unique.append(row)
        return answer, self._pack_result_rows(unique)

    @staticmethod
    def _build_review_document_context(page_blocks):
        """Build compact, deterministic cross-page context before page-level review."""
        if not page_blocks:
            return "Kein seitenübergreifender Kontext verfügbar."
        page_count = len(page_blocks)
        headings = []
        identifiers = []
        for page_no, text in page_blocks:
            for line in str(text).splitlines():
                line = re.sub(r"\s+", " ", line).strip()
                if not line:
                    continue
                # Preserve likely section headings, compound names and analytic labels.
                if (len(line) <= 150 and (line.isupper() or re.match(
                    r"^(?:\d+(?:\.\d+)*\s+)?(?:synthese|experiment|ergebnis|diskussion|analytik|nmr|ir|uv|cv|literatur|fazit|zusammenfassung)\b",
                    line, re.I))):
                    headings.append(f"S.{page_no}: {line}")
            for match in re.finditer(r"\b\d{1,3}\s*pg\s*\d{1,3}(?:\s*\d{1,3})?\b", str(text), re.I):
                label = re.sub(r"\s+", " ", match.group(0)).strip()
                item = f"S.{page_no}: {label}"
                if item not in identifiers:
                    identifiers.append(item)
        # Extract document metadata and likely sample-ID lines. Treat these as
        # hypotheses/context, never as proof unless the document explicitly defines them.
        all_text = "\\n".join(str(text) for _, text in page_blocks)
        author_matches = []
        for pattern in (
            r"(?im)^.{0,30}(?:autor|ersteller|verfasser|name)\\s*[:：]\\s*([^\\n]{2,80})",
            r"(?im)^.{0,30}(?:protokollant|bearbeiter)\\s*[:：]\\s*([^\\n]{2,80})",
        ):
            author_matches.extend(m.group(1).strip() for m in re.finditer(pattern, all_text))
        sample_like = []
        for page_no, text in page_blocks:
            for line in str(text).splitlines():
                line_clean = re.sub(r"\\s+", " ", line).strip()
                if re.search(r"(?i)\\b(?:analytik(?:kürzel|kuerzel)?|probe|proben[- ]?(?:nr|nummer|kennung)|sample|nmr|spektrum|spectrum|messung)\\b", line_clean):
                    if re.search(r"(?i)\\b\\d{1,3}\\s+[a-z]{1,4}(?:[_ -]?\\d{1,3})?\\b|\\b[a-z]{1,4}[_ -]?\\d{1,3}\\b", line_clean):
                        sample_like.append(f"S.{page_no}: {line_clean[:220]}")
        # First pages often define abbreviations; include them as explicit context.
        opening = "\\n\\n".join(
            f"[SEITE {n}]\\n{text[:2200]}" for n, text in page_blocks[:2]
        )
        return (
            f"Dokumentumfang: {page_count} indexierte Seiten.\\n"
            "WICHTIG: Dies ist ein Navigations-/Kontextanker, keine fachliche Bewertung.\\n"
            "Explizite Autor-/Erstellerangaben (falls erkannt):\\n"
            + ("; ".join(dict.fromkeys(author_matches[:10])) if author_matches else "Keine eindeutige Angabe automatisch erkannt.")
            + "\\n\\nAnfang des Dokuments (Definitionen, Probenlisten und Abkürzungen können hier erklärt sein):\\n"
            f"{opening}\\n\\n"
            "Erkannte Überschriften/Abschnittsmarker:\\n"
            + ("\\n".join(headings[:100]) or "Keine sicher erkannten Überschriften.")
            + "\\n\\nErkannte analytische Kurzlabels mit Fundseite:\\n"
            + ("\\n".join(identifiers[:80]) or "Keine passenden Labels erkannt.")
            + "\\n\\nMögliche Proben-/Analytikkennungen aus beschrifteten Zeilen (nur Kandidaten):\\n"
            + ("\\n".join(sample_like[:60]) or "Keine eindeutigen Kandidaten erkannt.")
            + "\\n\\nInterpretationsregel: Kürzel wie PG, Zahlen wie 41/42 und Suffixe wie 1/2 können interne Proben-, Produkt-, Fraktions- oder Spektrum-IDs sein. Erst Autorname, Dokumenttitel/Praktikumskennung, Zahl der Produkte, Anfangsdefinitionen und wiederholte Bildunterschriften gemeinsam prüfen. Nicht automatisch als Einheit oder Seitenzahl interpretieren."
        )

    @staticmethod
    def _deterministic_review_checks(page_blocks):
        """Flag arithmetic-looking values for review without claiming chemistry conclusions."""
        checks = []
        for page_no, text in page_blocks:
            for match in re.finditer(r"(?<!\d)(\d{2,3}(?:[,.]\d+)?)\s*%", str(text)):
                try:
                    value = float(match.group(1).replace(",", "."))
                except ValueError:
                    continue
                if value > 100:
                    excerpt_start = max(0, match.start() - 180)
                    excerpt_end = min(len(text), match.end() + 180)
                    excerpt = re.sub(r"\s+", " ", str(text)[excerpt_start:excerpt_end]).strip()
                    checks.append(
                        f"SEITE {page_no}: Zahlenwert {match.group(1)} % (>100 %). "
                        "Automatischer Plausibilitäts-Hinweis, KEIN Beweis für einen Fehler. "
                        f"Kontext: {excerpt}"
                    )
        if not checks:
            return "Keine automatisch erkannten Prozentwerte über 100 %. Das beweist nicht, dass alle Rechnungen korrekt sind."
        return "\n".join(checks[:30])

    def _evaluate_target_document(self, question, source, results):
        """Review every indexed page in bounded batches, then synthesize evidence.

        Unlike _context(), this path does not stop after MAX_CONTEXT_CHARS and
        accidentally leave later pages unreviewed.
        """
        rows = self._result_rows(results)
        pages = {}
        page_meta = {}
        for doc, meta, _distance in rows:
            page_no = int(meta.get("page", 0) or 0)
            if page_no <= 0:
                continue
            pages.setdefault(page_no, []).append(doc)
            page_meta.setdefault(page_no, meta)

        if not pages:
            return (
                "Die Bewertung wurde abgebrochen: Im Cache wurden keine "
                "auswertbaren Seiten-Chunks für das Zieldokument gefunden."
            )

        page_blocks = []
        for page_no in sorted(pages):
            chunks = pages[page_no]
            page_text = "\\n\\n".join(c.strip() for c in chunks if c and c.strip()).strip()
            page_blocks.append((page_no, page_text or "[Kein extrahierbarer Text auf dieser Seite]"))

        # Build cross-page context before reviewing individual pages. This prevents
        # abbreviations defined on page 1 from being misread as quantities on later pages.
        document_context = self._build_review_document_context(page_blocks)
        arithmetic_checks = self._deterministic_review_checks(page_blocks)
        self._trace_exam(
            "review_preflight",
            source=source,
            indexed_pages=len(page_blocks),
            automatic_over_100_percent_flags=arithmetic_checks.count("SEITE "),
        )

        # One page per call limits reasoning demand; every indexed page is still
        # processed, rather than silently truncating the document context.
        batch_size = 1
        batches = [page_blocks[i:i + batch_size] for i in range(0, len(page_blocks), batch_size)]
        all_images = self._source_images(results, extra_visual_candidates=[], question=question)
        findings = []

        print(
            f"[BEWERTUNG] {len(page_blocks)} indexierte Seiten in "
            f"{len(batches)} Prüfabschnitten; Bilder verfügbar: {len(all_images)}",
            flush=True,
        )

        for batch_no, batch in enumerate(batches, start=1):
            first_page, last_page = batch[0][0], batch[-1][0]
            text_block = "\\n\\n".join(
                f"[ZIELDOKUMENT: {source} | SEITE {page_no}]\\n{text}"
                for page_no, text in batch
            )
            batch_images = [
                item for item in all_images
                if first_page <= int(item.get("page", 0) or 0) <= last_page
            ][:2]
            image_labels = "\\n".join(
                f"[ORIGINALBILD] {item['source']} – Seite {item['page']}"
                for item in batch_images
            )
            content = f"""
Antworte direkt mit sichtbarem Text. Keine lange interne Analyse, keine ausführlichen
Gedankengänge. Nutze höchstens 5 kurze Stichpunkte; wenn keine Auffälligkeit erkennbar
ist, sage das ausdrücklich und nenne die Grenzen.

Du prüfst Abschnitt {batch_no}/{len(batches)} eines chemischen Laborprotokolls.
Nutzerauftrag: {question}
Zieldokument: {source}
Zu prüfende Seiten: {first_page} bis {last_page}

SEITENÜBERGREIFENDER KONTEXT (zur Auflösung von Abkürzungen/Labels; kein Beweis für einen Befund):
{document_context[:7000]}

SEITENTEXT:
{text_block}

ORIGINALBILDER IN DIESEM ABSCHNITT:
{image_labels if image_labels else "Keine Originalbilder für diesen Abschnitt übergeben."}

Erstelle eine evidenzbasierte Zwischenprüfung und antworte knapp. Unterscheide klar:
A. Direkt belegte Beobachtung (Seite + wörtlicher kurzer Beleg, soweit vorhanden)
B. Plausibler fachlicher Prüfverdacht (erkläre kurz, warum; keine vorschnelle Gewissheit)
C. Mit diesem Ausschnitt nicht verifizierbar
D. Fehlende oder unklare Angaben
Strenge Regeln zur fachlichen Zuverlässigkeit:
- Nenne einen Befund nur dann „belegt“, wenn der konkrete Widerspruch aus dem hier enthaltenen Text/Bild folgt. Sonst „Prüfverdacht“ oder „nicht verifizierbar“.
- Eine Ausbeute >100 % ist bei einem als rein angenommenen Produkt ungewöhnlich und prüfenswert; behaupte nicht, sie sei ausnahmslos physikalisch unmöglich. Prüfe erst, ob theoretische Ausbeute, limitierendes Edukt, Produktform, Solvat-/Salzanteil, Feuchte und Reinheit eindeutig angegeben sind.
- Interpretiere Kürzel und Zahlenfolgen zuerst als mögliche Proben-/Analytik-IDs, bevor du sie als physikalische Einheiten liest. In Laborprotokollen sind Labels wie „PG_42“, „42 PG 1“, „41 pg 2“ oder ähnliche Muster oft interne Kennungen für Person/Arbeitsgruppe, Probe, Produkt, Fraktion oder Spektrum. Das sind Beispiele, keine universelle Regel.
- Löse solche Labels anhand des gesamten Dokuments auf: Autor/Ersteller, Dokument-/Praktikumskennung (z. B. T4), Zahl der hergestellten Produkte, Probenliste, Definitionen am Anfang sowie Wiederholungen in NMR-/IR-/UV-Vis-/CV-Bildunterschriften. Wenn ein Autor „Paul Grote“ heißt und „PG“ in Analytiklabels wiederholt vorkommt, ist „PG“ als Initialen-/Probenkürzel eine plausible Hypothese; behaupte sie nur als gesichert, wenn das Dokument sie definiert oder das Muster klar bestätigt.
- Ein Dokument mit 25 Seiten kann legitime Verweise auf Proben/Spektren „41“ oder „42“ enthalten; die Zahl muss nicht die PDF-Seite bezeichnen. Prüfe, ob die Kennung über mehrere Messarten konsistent wiederkehrt und zu unterschiedlichen Produkten passt. Wenn das Label wahrscheinlich eine Probenkennung ist, darf es nicht als pg (Pikogramm) oder OCR-Fehler beanstandet werden.
- Leite aus „42 pg“ allein keinen sicheren Einheitenfehler ab. Prüfe zunächst Proben-ID, Initialen, Analytikkürzel, Bildunterschriften und den Originalseitenkontext; wenn die Bedeutung nicht auflösbar ist, markiere sie als unklar statt als Fehler.
- Erfinde keine chemische Begründung für einen Lösungsmittel-Fehler. Behaupte insbesondere nicht, dass gewöhnliche aromatische Pyridinring-Protonen in D2O allein deshalb austauschen oder dass D2O automatisch ein ungeeignetes Lösungsmittel sei. Beurteile Lösungsmittelwahl anhand der konkret genannten Verbindung, Löslichkeit, Messbedingungen und des tatsächlichen Spektrums; wenn diese Belege fehlen, kennzeichne nur eine offene Frage.
- Leite die erwartete Zahl von NMR-Signalen nicht nur aus einer vereinfachten Symmetrieannahme ab. Berücksichtige Lösungsmittel, Isomerie, Diastereotopie, Auflösung, Signalüberlagerung und die tatsächlich beschriebene Struktur; wenn diese Informationen fehlen, formuliere nur einen Prüfhinweis.
- Fehlendes 13C-NMR, fehlende Reinheitsdaten oder unvollständige Messbedingungen sind nicht automatisch Fehler, sofern Aufgabenstellung/Methodenstandard dies nicht verlangt.
- Eine unvollständige Referenz im extrahierten Text kann OCR-/Extraktionsursachen haben; erkläre eine Quelle nicht für falsch, ohne die Bibliografie zu sehen.
- Behaupte keine Übereinstimmung mit einer Referenzquelle, wenn sie in diesem Abschnitt nicht vorliegt.
- Das Fehlen von Text zu einer Abbildung beweist nicht, dass die Abbildung im PDF fehlt.
- Erfinde keine Zahlen, Messwerte, Zitate, Literaturangaben oder Korrekturwerte.
- Bewerte nur den hier enthaltenen Abschnitt. Keine Gesamtaussage zur Vollständigkeit des ganzen Protokolls.
Gib eine knappe, strukturierte Zwischenprüfung aus.
"""
            print(
                f"[BEWERTUNG] Prüfe Abschnitt {batch_no}/{len(batches)} "
                f"(Seiten {first_page}-{last_page})",
                flush=True,
            )
            result = self._chat_text(
                self.config.SYSTEM_PROMPT,
                content,
                images=batch_images,
                num_predict=1600,
            )
            # A response cut off by num_predict is not accepted as a complete review.
            # Retry once with a larger budget and a shorter instruction set.
            if getattr(self, "_last_chat_done_reason", None) == "length" and not result.startswith("FEHLER:"):
                print(
                    f"[BEWERTUNG][RETRY] Abschnitt {batch_no} endete am Tokenlimit; "
                    "wiederhole mit höherem Ausgabelimit.", flush=True
                )
                retry_content = (
                    content
                    + "\n\nWIEDERHOLUNG: Antworte kompakt und vollständig. Maximal 4 Befunde. "
                    "Jeder Befund braucht Seite, kurzen Beleg und Status. Keine Einleitung."
                )
                result = self._chat_text(
                    self.config.SYSTEM_PROMPT,
                    retry_content,
                    images=batch_images,
                    num_predict=2600,
                )
            if result.startswith("FEHLER:"):
                return (
                    f"Die Protokollprüfung wurde bei Seiten {first_page}–{last_page} "
                    "abgebrochen, weil Ollama keine verwertbare Zwischenprüfung "
                    f"geliefert hat. Technische Rückmeldung: {result}"
                )
            findings.append(
                f"## Zwischenprüfung: Seiten {first_page}–{last_page}\\n{result}"
            )

        synthesis_content = f"""
Erstelle die abschließende Bewertung des chemischen Laborprotokolls anhand der folgenden Zwischenprüfungen.
Nutzerauftrag: {question}
Zieldokument: {source}
Geprüfte Seiten: {len(page_blocks)} indexierte Seiten in {len(batches)} Abschnitten.

SEITENÜBERGREIFENDER KONTEXT:
{document_context[:7000]}

AUTOMATISCHE ZAHLEN-PLAUSIBILITÄTSHINWEISE (nur Prüfsignale; nicht ungeprüft als Fehler übernehmen):
{arithmetic_checks}

ZWISCHENPRÜFUNGEN:
{chr(10).join(findings)}

Vorgaben für die Endbewertung:
0. Nutzerauftrag richtig auslegen: Prüfe den tatsächlichen Auftrag und den ausdrücklich genannten Umfang. „fachliche Fehler“ bedeutet fachliche Plausibilität, Rechenwege, Einheiten, interne Widersprüche und Auswertung; es ist keine Aufforderung, fehlende Angaben automatisch als Fehler zu werten. Wenn eine notwendige Annahme nicht klar ist, stelle eine Rückfrage statt die Aufgabe umzudeuten.
1. Fasse nur konkrete Befunde zusammen, die in den Zwischenprüfungen mit Seitenbeleg gestützt sind. Die Zwischenprüfungen selbst können falsch liegen: prüfe chemische Schlussfolgerungen kritisch und übernimm sie nicht automatisch.
2. Gib zu jedem Befund die Seite(n), den konkreten Beleg und den Status an: „belegt“, „plausibler Prüfverdacht“ oder „mit den vorliegenden Daten nicht verifizierbar“.
3. Trenne Beobachtung, fachliche Interpretation und empfohlene Prüfung/Korrektur.
4. Verwende „kritischer Fehler“ oder „nicht akzeptabel“ nur bei einem eindeutig nachgewiesenen, für das Ergebnis wesentlichen Widerspruch. Sonst formuliere zurückhaltend.
5. Bei Rechnungen: rechne nur nach, wenn die benötigten Ausgangswerte ausdrücklich vorliegen; zeige Formel, Werte und Einheiten. Fehlen Werte, benenne diese.
6. Werte wie >100 % oder sehr kleine Probenmengen sind Warnsignale, aber allein kein Beweis für eine bestimmte Ursache. Behaupte bei „pg“ keinen gesicherten Einheitenfehler, solange die Originalabbildung/Methodik nicht kontrolliert wurde.
7. NMR-Signalzahl/-integration darf nicht allein aus einer vereinfachten Symmetrieannahme als Fehler erklärt werden. Fehlendes 13C-NMR, Reinheitsdaten oder einzelne Messbedingungen sind nicht automatisch Fehler, solange die Aufgabenstellung/der Standard dies nicht verlangt.
8. Behaupte keine Übereinstimmung mit externen Referenzspektren oder RAG-Dokumenten, sofern diese nicht tatsächlich eingesehen wurden.
9. Eine Abbildung, die im extrahierten Text nicht erscheint, ist nicht automatisch im PDF fehlend.
10. Nenne zum Schluss den Umfang und die Grenzen: Die indexierten Text-Chunks wurden seitenweise geprüft; nicht automatisch jede Grafik, jedes Spektrum oder jede Literaturquelle unabhängig verifiziert.
11. Keine pauschale Behauptung, das gesamte Dokument sei fehlerfrei oder „fachlich nicht akzeptabel“, wenn das aus den konkreten Belegen nicht folgt.
12. Wenn Zwischenprüfungen spekulative oder widersprüchliche Behauptungen enthalten, kennzeichne sie ausdrücklich als unbestätigt oder lasse sie weg.
13. Suche ausdrücklich nach Widersprüchen zwischen verschiedenen Dokumentteilen (z. B. eine Stelle nennt geringe/ungeprüfte Reinheit, eine andere behauptet hohe Reinheit). Nenne beide Seiten mit Fundseite und zitiere die jeweiligen Aussagen kurz.
14. Interpretiere Kürzel nicht isoliert: prüfe zuerst Dokumenttitel/Praktikumskennung, Autor-/Erstellername, Zahl der synthetisierten Produkte, Probenlisten, Tabellenlegenden und Bildunterschriften. Labels wie „41 pg 2“, „42 pg 1“ oder „PG_42“ können interne Proben-/Analytikkennungen sein. Bei einem Autor namens Paul Grote ist „PG“ als Initialen-Kürzel plausibel, aber ohne ausdrückliche Definition nur als begründete Hypothese zu nennen. Wiederholungen derselben Kennung bei NMR, IR, UV-Vis und CV sprechen eher für eine Proben-ID als für eine Masseinheit. Solche Labels dürfen nicht allein wegen „pg“ als Pikogramm-Einheit oder OCR-Fehler beanstandet werden.
15. Prüfe Lösungsmittelbehauptungen auf konkrete Belege. Behaupte nicht ohne Beleg, dass aromatische Pyridinring-Protonen in D2O austauschen oder dass D2O allein die NMR-Zuordnung verfälscht. Ungewöhnlich ist nicht gleich falsch; wenn keine konkreten Daten vorliegen, formuliere eine offene Frage statt einer Korrekturanweisung.
16. Automatische Zahlenhinweise sind keine Befunde. Verifiziere den Bezug der Zahl, die Rechenbasis und den Kontext vor einer Schlussfolgerung.
17. Prüfe die Originalseite/Bilddaten, wenn du aus Text-Extraktion/OCR einen Satzabbruch, eine fehlende Abbildung, ein ungewöhnliches Reagenz oder eine Beschriftung ableitest. Trenne nachweisbare PDF-Inhalte von OCR-/Extraktionsartefakten.
18. Behaupte keine „eindeutige Verunreinigung“ nur aus einem Signal im NMR. Prüfe erst, ob das Molekül Protonen enthält, welches Signal/Integral genau beschrieben wird, ob das Lösungsmittel/Referenz/Restlösemittel bekannt ist und ob das Signal tatsächlich zum Produkt gehören soll.
19. Symmetrie- und Signalzahlargumente müssen zur korrekten Struktur passen. Keine Aussage wie „alle aromatischen Protonen äquivalent“ ohne Positionen/Substitutionsmuster und erwartete Äquivalenzklassen zu prüfen.
20. Fehlendes Lösungsmittel oder eine nicht erklärte Verfahrensangabe ist zunächst eine Dokumentationslücke/Prüffrage, kein Beweis, dass die Reaktion nicht möglich war oder die Ausbeute ungültig ist.
21. Verwende externe chemische Allgemeinannahmen nicht als Ersatz für die konkrete Quelle. Wenn der Text die Aussage nicht trägt, markiere sie als Hypothese und sage, welche Information zur Prüfung fehlt.
"""
        print("[BEWERTUNG] Erstelle abschließende, evidenzbasierte Zusammenfassung.", flush=True)
        return self._chat_text(
            self.config.SYSTEM_PROMPT,
            synthesis_content,
            images=None,
            num_predict=max(6000, int(self.config.NUM_PREDICT)),
        )

    def answer_question(self, question):
        # Wenn die letzte Antwort eine Dokumentauswahl erbeten hat, darf eine
        # knappe Antwort wie „T4“ oder ein Dateiname als Auswahl gelten.
        pending = self._pending_target_clarification
        if pending:
            candidates = pending.get("candidates", [])
            qn = self._normalize_search_text(question)
            selected = []
            for src in candidates:
                stem = Path(str(src).replace("\\", "/")).stem
                stem_n = self._normalize_search_text(stem)
                if stem_n and (stem_n in qn or self._compact_search_text(stem) in self._compact_search_text(question)):
                    selected.append(src)
                else:
                    # Kurze Auswahl wie T4 muss eindeutig zum Dateinamen passen.
                    for token in re.findall(r"(?<![a-z0-9])t\s*([0-9]{1,2})(?![0-9])", qn):
                        if f"t{token}" in set(self._tokenize_search_text(stem)):
                            selected.append(src)
            selected = sorted(set(selected))
            if len(selected) == 1:
                original = pending.get("original_question", "Prüfe das Dokument")
                self._pending_target_clarification = None
                question = f"{original}\nAusgewähltes Zieldokument: {self._pretty_source(selected[0])}"
                print(f"[CACHE][ZIELDOKUMENT] Nutzer hat Ziel bestätigt: {selected[0]}")
            elif re.search(r"\b(abbrechen|egal|nein|stop)\b", qn):
                self._pending_target_clarification = None
                return "Okay, ich starte keine Dokumentprüfung. Nenne mir bei Bedarf den gewünschten Dateinamen.", self._empty_results()
            else:
                options = "\n".join(f"- {self._pretty_source(src)}" for src in candidates[:12])
                return ("Ich bin noch nicht sicher, welches Dokument gemeint ist. Bitte antworte mit dem Dateinamen "
                        "oder der Kennung (z. B. T4):\n" + options), self._empty_results()

        self.sync_cache()

        # Evaluation/review requests should be anchored to the requested cached
        # document before general RAG retrieval is allowed to introduce unrelated files.
        # _normalize_search_text decomposes umlauts ("prüfe" -> "prufe",
        # "überprüfe" -> "uberprufe"). Match the normalized forms here;
        # otherwise common German review requests silently fall through to
        # general retrieval and can use unrelated documents.
        normalized_question = self._normalize_search_text(question)
        # Ein fachliches "bewerte den Trend" ist keine Dokumentprüfung.
        # Erst ein expliziter Dokument-/Protokollbezug aktiviert den Auditpfad.
        review_verb = bool(re.search(
            r"\b(bewerte|bewerten|bewertung|beurteile|beurteilen|prufe|prufen|"
            r"uberprufe|uberprufen|analysiere|analysieren|kritisiere|kritisiere|"
            r"kontrolliere|kontrollieren|begutachte|begutachten|feedback|fehler|widerspruche)\b",
            normalized_question,
        ))
        document_scope = bool(re.search(
            r"\b(protokoll|laborbericht|bachelorarbeit|thesis|dokument|pdf|datei|"
            r"t\s*[0-9]{1,2}|klausur|arbeit im cache|cache-dokument)\b",
            normalized_question,
        ))
        # Explizit genannter Dateiname zählt ebenfalls als Dokumentbezug.
        target_source = self._resolve_target_cache_source(question) if (review_verb and document_scope) else None
        evaluation_request = bool(review_verb and document_scope)
        if evaluation_request and not target_source:
            candidates = self._target_candidates()
            # Nur bei tatsächlich mehreren Dateien eine Auswahl anbieten; bei
            # leerem Cache/keinem Treffer gezielt um den Dateinamen bitten.
            if len(candidates) > 1:
                self._pending_target_clarification = {
                    "original_question": question,
                    "candidates": candidates,
                }
                options = "\n".join(f"- {self._pretty_source(src)}" for src in candidates[:12])
                message = (
                    "Ich verstehe den Auftrag als fachliche Prüfung eines Laborprotokolls. "
                    "Im Cache liegen aber mehrere mögliche Zieldokumente; ich möchte nicht "
                    "auf Verdacht eines auswählen. Welches soll ich prüfen?\n" + options
                    + "\nBitte antworte mit dem Dateinamen oder der Kennung (z. B. T4)."
                )
            else:
                message = (
                    "Ich verstehe den Auftrag als Dokumentprüfung, kann aber kein passendes "
                    "Zieldokument eindeutig im Cache finden. Ich werde deshalb keine andere Datei "
                    "ersatzweise bewerten. Bitte nenne den genauen Dateinamen oder prüfe, ob die "
                    "PDF im Cache indexiert ist."
                )
            print("[CACHE][ZIELDOKUMENT] Keine eindeutige Quelle; Rückfrage statt stiller Auswahl.")
            return message, self._empty_results()
        if target_source:
            print(f"[CACHE][ZIELDOKUMENT] Primäre Bewertungsquelle: {target_source}")
            results = self._cache_results_for_source(target_source)
            answer = self._evaluate_target_document(question, target_source, results)
            self.conversation_history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ])
            self.conversation_history = self.conversation_history[-self.config.MAX_HISTORY_MESSAGES:]
            return answer, results

        summary_request = self._is_summary_request(question)
        # Resolve a specifically named cache document even for ordinary factual
        # questions (not only reviews/summaries). Otherwise questions such as
        # "What is the first task in PCI_...pdf?" fall through to global RAG.
        resolved_source = self._resolve_target_cache_source(question)
        q_compact = self._compact_search_text(question)
        resolved_stem = Path(str(resolved_source).replace("\\", "/")).stem if resolved_source else ""
        stem_compact = self._compact_search_text(resolved_stem)
        explicit_filename = bool(
            resolved_source and stem_compact and len(stem_compact) >= 8
            and stem_compact in q_compact
        )
        explicit_identifier = bool(
            resolved_source and re.search(r"(?<![a-z0-9])t\s*[0-9]{1,2}(?![0-9])", normalized_question)
            and re.search(r"(?<![a-z0-9])t\s*[0-9]{1,2}(?![0-9])", self._normalize_search_text(resolved_stem))
        )
        explicit_thesis = bool(
            resolved_source and re.search(r"\b(bachelorarbeit|bachelor thesis|meine thesis)\b", normalized_question)
        )
        direct_document_question = bool(
            resolved_source and not evaluation_request and not summary_request
            and (explicit_filename or explicit_identifier or explicit_thesis)
        )
        # A named document's factual questions use every indexed chunk of that
        # document, rather than unrelated top-k retrieval results.
        summary_source = resolved_source if summary_request else None
        cache_summary = (
            summary_request
            and self.cache.has_documents()
            and (self._mentions_cache(question) or summary_source is not None)
        )
        if direct_document_question:
            print(f"[CACHE][ZIELDOKUMENT] Direkte Inhaltsfrage; Quelle festgelegt: {resolved_source}")
            results = self._cache_results_for_source(resolved_source)
        elif cache_summary and summary_source:
            results = self._cache_results_for_source(summary_source)
        elif cache_summary:
            results = self.cache.get_summary_results()
            results = self._diversify_summary_results(results) if results else self._empty_results()
        else:
            # Für einen Klausur-/Aufgabenauftrag bleibt der Cache die Quelle der
            # Aufgabenstellungen; permanent indexierte Dokumente liefern getrennt
            # dazu das Fachwissen. Die alte Cache-only-Suche war der Kernfehler.
            exam_request = bool(re.search(
                r"\b(klausur|altklausur|musterklausur|prüfung|pruefung|aufgabenblatt|aufgabe|aufgaben|übungsblatt|uebungsblatt|lösen|löse|loesen|bearbeiten|beantworte)\b",
                self._normalize_search_text(question),
            ))
            cache_mentioned = self._mentions_cache(question)
            # Wenn im Chat explizit Klausur-/Cache-PDFs freigegeben sind,
            # gelten Aufgabenaufträge auch ohne das Wort "Cache" als darauf
            # bezogen. Die Entscheidung bleibt durch den Chat-Scope begrenzt.
            scoped_cache_selected = bool(
                self.active_source_keys is not None
                and any(str(key).startswith("cache::") for key in self.active_source_keys)
            )
            should_check_cache = cache_mentioned or scoped_cache_selected
            cache_available = self._has_indexed_cache_documents() if should_check_cache else False
            if cache_available and should_check_cache and exam_request:
                print("[KLAUSUR] Cache-Klausurmodus: seitenweise Extraktion und Einzelantworten aktiv.")
                self._trace_exam("exam_route_selected", question=question, cache_available=True)
                return self._answer_cached_exam(question)

            if cache_mentioned and cache_available:
                raw_results = self.search_documents(question, is_cache=True)
            else:
                raw_results = self.search_documents(question)
            results = self._diversify_results(raw_results)

        query_is_cache = bool(direct_document_question or cache_summary or self._mentions_cache(question))
        vision_candidates = self._vision_retrieval(
            question,
            is_cache=(True if query_is_cache else None),
        )
        visual_candidates = self._visual_candidates_from_lexical(
            question,
            is_cache=(True if query_is_cache else None),
        )
        if direct_document_question:
            # Keep image-side retrieval aligned with the same named source; a
            # cache-only filter alone still allows pages from other cached PDFs.
            def _same_target(item):
                item_source = str(item.get("source", "")).replace("\\\\", "/").casefold()
                target = str(resolved_source).replace("\\\\", "/").casefold()
                return item_source == target
            raw_vision = list(vision_candidates)
            raw_visual = list(visual_candidates)
            raw_vision_count = len(raw_vision)
            raw_visual_count = len(raw_visual)
            rejected_sources = sorted({
                str(item.get("source", ""))
                for item in (raw_vision + raw_visual)
                if not _same_target(item)
            })
            vision_candidates = [item for item in raw_vision if _same_target(item)]
            visual_candidates = [item for item in raw_visual if _same_target(item)]
            print(
                f"[VISION][ZIELDOKUMENT] Filter nach Quelle: {resolved_source} | "
                f"Vision {raw_vision_count}->{len(vision_candidates)} | "
                f"Lexikalisch {raw_visual_count}->{len(visual_candidates)}"
            )
            if rejected_sources:
                print("[VISION][ZIELDOKUMENT] Ausgeschlossen: " + " ; ".join(rejected_sources[:8]))
        images = self._source_images(
            results,
            extra_visual_candidates=(vision_candidates + visual_candidates),
            question=question,
        )
        if direct_document_question:
            # Final safety check: no image from a different source may reach
            # the vision analyzer or the multimodal answer model.
            target = str(resolved_source).replace("\\\\", "/").casefold()
            before_image_filter = len(images)
            images = [
                item for item in images
                if str(item.get("source", "")).replace("\\\\", "/").casefold() == target
            ]
            print(
                f"[VISION][ZIELDOKUMENT] Originalbilder freigegeben: "
                f"{before_image_filter}->{len(images)}"
            )
            for image_no, item in enumerate(images, start=1):
                print(
                    f"    [VISION][FREIGEGEBEN {image_no}] "
                    f"{item.get('source')} | Seite {item.get('page')}"
                )
        context = self._context(results)
        visual_analysis = self._vision_for_images(question, images)
        labels = "\n".join(
            f"[ORIGINALBILD {i}] {item['source']} – Seite {item['page']}"
            for i, item in enumerate(images, start=1)
        )
        summary_instruction = """
ZUSAMMENFASSUNGS-MODUS: Erstelle eine inhaltlich substanzielle, gut strukturierte
Zusammenfassung statt einer Antwort von nur wenigen Sätzen. Nutze je nach Umfang
ca. 6–10 aussagekräftige Abschnitte oder Stichpunktgruppen: (1) Thema und Lernziele,
(2) zentrale Begriffe/Definitionen, (3) Modelle und Zusammenhänge, (4) wichtige Regeln,
Formeln oder Diagramme samt Interpretation, (5) Beispiele/Anwendungen, (6) typische
Fehler oder Verwechslungen, (7) prüfungsrelevante Kernaussagen. Verwende nur Abschnitte,
die durch den Quelleninhalt gestützt sind. Wenn das Dokument länger ist oder Themen
fehlen, benenne die Grenzen. Führe keine nicht gefundenen Kapitel oder Seiten als Quelle an.
Zielumfang bei einer normalen Vorlesung: ungefähr 700–1200 Wörter, bei knapper Quelle
entsprechend kürzer. Belege wichtige Aussagen mit Seitenzahlen, sofern verfügbar.
""" if cache_summary else ""
        # Der persistierte Gesprächsverlauf wird gezielt als Kontext für den
        # aktuellen Chat übergeben. Er ist nicht global und wird beim Wechsel
        # des Chats von der GUI aus dem jeweiligen SQLite-Verlauf geladen.
        recent_history = self.conversation_history[-self.config.MAX_HISTORY_MESSAGES:]
        history_lines = []
        for item in recent_history:
            role = str(item.get("role", "")).strip().upper()
            text = str(item.get("content", "")).strip()
            if role in {"USER", "ASSISTANT"} and text:
                text = text[:self.config.MAX_HISTORY_MESSAGE_CHARS]
                history_lines.append(f"{role}: {text}")
        history_context = "\n\n".join(history_lines) or "Kein früherer Nachrichtenverlauf vorhanden."
        memory_context = self.active_memory_summary or "Kein zusätzliches Chat-Memory gespeichert."

        content = f"""
Beantworte die folgende Frage auf Deutsch:

{question}

CHAT-MEMORY (gespeicherte Zusammenfassung nur dieses Chats):
{memory_context}

RELEVANTER GESPRÄCHSVERLAUF DIESES CHATS:
{history_context}

{summary_instruction}

RELEVANTES QUELLENWISSEN:
{context}

PERSISTIERTE BILDANALYSEN:
{visual_analysis if visual_analysis else 'Keine relevante Bildanalyse verfügbar.'}

ORIGINALBILDER:
{labels if labels else 'Keine visuellen Originalseiten vorhanden.'}

QUELLENTRENNUNG:
- Bei gewöhnlichen Fachfragen verwende passende permanente RAG-Quellen.
- Wenn ausdrücklich nach einem Cache-Dokument gefragt wird, ist dieses die Quelle für den Dokumentinhalt, nicht automatisch der Beleg für Fachwissen.
- Eine Bildanalyse ist nur ein Such- und Lesehinweis. Originalbilder und extrahierter PDF-Text haben Vorrang.
- Kennzeichne dokumentgestützte Aussagen als [QUELLENWISSEN] und ergänzendes Fachwissen als [ALLGEMEINES FACHWISSEN].
- Erfinde keine Seiten, Quellen, Zahlen, Strukturdetails oder Messwerte.
- Wenn mehrere Teilfragen gestellt werden, beantworte jede einzeln und prüfe am Ende, ob alle beantwortet sind.
"""
        answer = self._chat_text(self.config.SYSTEM_PROMPT, content, images=images)
        self.conversation_history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])
        self.conversation_history = self.conversation_history[-self.config.MAX_HISTORY_MESSAGES:]
        return answer, results

    # =========================================================
    # Zusammenfassungs-Ergebnisse
    # =========================================================

    def _diversify_summary_results(
        self,
        results
    ):

        docs = results.get(
            "documents",
            [[]]
        )[0]

        metas = results.get(
            "metadatas",
            [[]]
        )[0]

        selected_docs = []

        selected_metas = []

        selected_distances = []

        seen_pages = set()

        for doc, meta in zip(
            docs,
            metas
        ):

            key = (
                meta.get(
                    "source"
                ),
                meta.get(
                    "page"
                )
            )

            if key in seen_pages:
                continue

            seen_pages.add(
                key
            )

            selected_docs.append(
                doc
            )

            selected_metas.append(
                meta
            )

            selected_distances.append(
                0.0
            )

        return {
            "documents": [
                selected_docs
            ],

            "metadatas": [
                selected_metas
            ],

            "distances": [
                selected_distances
            ],
        }

    # =========================================================
    # Chat
    # =========================================================

    def clear_history(
        self
    ):

        self.conversation_history = []

        print(
            "[CHAT] Gesprächshistorie "
            "gelöscht."
        )

    # =========================================================
    # Quellen
    # =========================================================

    def get_sources(
        self,
        results
    ):

        metas = results.get(
            "metadatas",
            [[]]
        )[0]

        sources = []

        seen = set()

        for meta in metas:

            key = (
                meta.get(
                    "source"
                ),
                meta.get(
                    "page"
                ),
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            sources.append(
                key
            )

        return sources