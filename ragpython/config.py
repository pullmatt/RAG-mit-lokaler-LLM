from pathlib import Path
from types import SimpleNamespace


class Config:
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()

    DOCUMENTS_DIR = PROJECT_ROOT / "documents"
    DATABASE_DIR = PROJECT_ROOT / "database"
    CHROMA_DIR = DATABASE_DIR / "chroma"
    PAGE_IMAGES_DIR = DATABASE_DIR / "page_images"
    CACHE_DOCUMENTS_DIR = PROJECT_ROOT / "cache"

    COLLECTION_NAME = "chemie_qwen3_embedding_8b"
    VISION_COLLECTION_NAME = "chemie_vision_qwen3_embedding_8b"

    EMBEDDING_MODEL = "qwen3-embedding:8b"
    LLM_MODEL = "gemma4:12b"
    VISION_MODEL = "gemma4:12b"

    # Qwen 3.5 9B unterstützt laut Ollama ein deutlich größeres Kontextfenster.
    # 32K ist für den lokalen Betrieb ein guter Kompromiss aus Kontext und RAM/VRAM.
    NUM_CTX = 32768
    NUM_PREDICT = 3072
    VERIFY_TASK_ANSWERS = True
    TASK_VERIFICATION_TOKENS = 1800

    RETRIEVAL_CANDIDATES = 20
    MAX_RETRIEVED_PAGES = 8
    MAX_SOURCE_IMAGES = 4
    MAX_CONTEXT_CHARS = 18000
    MAX_SOURCE_CHARS = 4500

    MAX_HISTORY_MESSAGES = 6
    MAX_HISTORY_MESSAGE_CHARS = 3000

    CHUNK_SIZE = 3000
    CHUNK_OVERLAP = 500
    EMBEDDING_BATCH_SIZE = 64

    OCR_ENABLED = True
    OCR_LANGUAGE = "deu+eng"
    OCR_DPI = 180
    OCR_TEXT_THRESHOLD = 300

    VISION_ENABLED = True
    VISION_DPI = 150
    VISION_DESCRIPTION_TARGET = 2200

    # Multimodales Retrieval
    VISION_RETRIEVAL_CANDIDATES = 8
    VISUAL_SHORTLIST_CANDIDATES = 8
    VISION_PAGE_BONUS = 0.18

    # Phase A.2 – multimodale Relevanz
    VISION_LEXICAL_CANDIDATES = 24
    VISION_MIN_SCORE = 0.30
    VISION_RANK_TOP_K = 6
    VISION_PROMPT_VERSION = "vision-v3-page-content-inventory"

    INDEX_VERSION = "chemie-rag-v7-qwen3-embedding-8b"

    SYSTEM_PROMPT = """
Du bist ein wissenschaftlicher Lernassistent für Chemie.

Deine Aufgabe besteht aus zwei getrennten Wissensquellen:

[QUELLENWISSEN]
Informationen aus den bereitgestellten Dokumenten,
persistierten Bildanalysen und Originalbildern des Benutzers.

[ALLGEMEINES FACHWISSEN]
Dein allgemeines chemisches Fachwissen.

Regeln:

1. Verwende die bereitgestellten Quellen bevorzugt, wenn sie
   für die Frage relevant sind.

2. Du darfst allgemeines chemisches Fachwissen verwenden, wenn
   die Quellen einen Begriff nur erwähnen, die Frage nicht
   vollständig beantworten oder keine passende Erklärung enthalten.

3. Trenne Quellenwissen und allgemeines Fachwissen klar.

4. Behaupte niemals, dass eine Information aus einer Quelle stammt,
   wenn sie nur aus deinem allgemeinen Fachwissen kommt.

5. Originalbilder haben Vorrang vor automatisch erzeugten Bildanalysen.

6. Das gilt insbesondere für Handschrift, chemische Strukturformeln,
   Stereochemie, Reaktionsmechanismen, Reaktionsbedingungen, Spektren,
   Diagramme, Tabellen, griechische Buchstaben und mathematische Formeln.

7. Wenn eine Handschrift oder Struktur nicht eindeutig lesbar ist,
   sage das ausdrücklich. Erfinde keine fehlenden Informationen.

8. Chemische Abkürzungen dürfen nicht aufgrund ähnlicher Schreibweise
   umgedeutet werden.

9. Korrigiere falsche Prämissen des Benutzers ausdrücklich.

10. Wenn die Frage allgemeines chemisches Wissen betrifft und die
    Quellen dazu nichts Relevantes enthalten, beantworte sie trotzdem
    mit deinem Fachwissen und kennzeichne diesen Teil als
    [ALLGEMEINES FACHWISSEN].

11. Quellen, Seitenzahlen oder chemische Strukturen dürfen niemals
    erfunden werden.

12. Eine persistierte Bildanalyse ist ein Hinweis auf den tatsächlichen
    Seiteninhalt, ersetzt aber bei Strukturformeln, Spektren, Handschrift
    und Mechanismen niemals das Originalbild.

13. Bei Klausuren und Aufgaben im temporären Cache gilt eine strikte Trennung:
    Das Cache-Dokument liefert die Aufgabenstellung, Abbildungen und gegebenen
    Werte. Die fachliche Lösung muss zusätzlich über passende permanente RAG-
    Quellen gesucht werden. Die Cache-Aufgabenstellung darf nicht als Ersatz
    für Fachquellen behandelt werden. Bearbeite alle erkannten Teilaufgaben.
"""


def config_for_project(project_id: str):
    """Erstellt eine vollständig getrennte lokale RAG-Konfiguration pro Projekt.

    Bestehende globale Ordner bleiben als Legacy-Quelle erhalten, werden von
    Projekt-RAG-Instanzen aber nicht indexiert. Jede Projektinstanz erhält
    eigene Dokument-/Cache-Ordner, Hashdateien, Seitenbilder und Chroma-Dateien.
    """
    root = Config.PROJECT_ROOT
    safe_id = "".join(ch for ch in str(project_id) if ch.isalnum())[:32]
    if not safe_id:
        raise ValueError("Ungültige Projekt-ID")

    project_data_dir = root / "data" / "projects" / safe_id
    project_database_dir = root / "database" / "project_indexes" / safe_id
    values = {
        key: getattr(Config, key)
        for key in dir(Config)
        if not key.startswith("_") and not callable(getattr(Config, key))
    }
    values.update({
        "PROJECT_ID": safe_id,
        "PROJECT_DATA_DIR": project_data_dir,
        "PROJECT_ROOT": root,
        "DOCUMENTS_DIR": project_data_dir / "documents",
        "CACHE_DOCUMENTS_DIR": project_data_dir / "cache",
        "DATABASE_DIR": project_database_dir,
        "CHROMA_DIR": project_database_dir / "chroma",
        "PAGE_IMAGES_DIR": project_database_dir / "page_images",
        "COLLECTION_NAME": f"chemie_text_{safe_id[:32]}",
        "VISION_COLLECTION_NAME": f"chemie_vision_{safe_id[:27]}",
    })
    return SimpleNamespace(**values)
