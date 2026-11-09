import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rag import ChemistryRAG


def main():
    rag = ChemistryRAG()
    print()
    print("=" * 80)
    print("                         CHEMIE-RAG v6")
    print("=" * 80)
    print()
    print("Dokumente:", rag.config.DOCUMENTS_DIR)
    print("Datenbank:", rag.config.DATABASE_DIR)
    print("Einträge:", rag.database.count())
    print()
    print("Indexierung wird geprüft...")
    rag.index_documents()
    print()
    print("Bereit. Befehle: exit, clear, reindex, diagnose_exam")
    print()
    while True:
        try:
            question = input("Du: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nChat beendet.")
            break
        if not question:
            continue
        command = question.lower()
        if command in ("exit", "quit"):
            print("\nChat beendet.")
            break
        if command == "clear":
            rag.clear_history()
            continue
        if command == "reindex":
            print("WARNUNG: reindex löscht den aktuellen Projektindex und baut ihn neu auf.")
            confirm = input("Zum Fortfahren REBUILD eingeben: ").strip()
            if confirm == "REBUILD":
                rag.index_documents(force_rebuild=True)
                rag.sync_cache()
            else:
                print("Abgebrochen; vorhandener Index bleibt erhalten.")
            continue
        if command == "diagnose_exam":
            rag.sync_cache()
            rag.diagnose_exam_extraction("Löse die Klausur im Cache vollständig")
            continue
        try:
            print("\nSuche relevante Quellen...")
            answer, results = rag.answer_question(question)
            print("\nQwen:\n")
            print(answer)
            print("\nQuellen:")
            for source, page in rag.get_sources(results):
                print(f"- {source}, Seite {page}")
            print("\n" + "-" * 80 + "\n")
        except Exception as error:
            print("\nFEHLER:")
            print(error)
            print()


if __name__ == "__main__":
    main()
