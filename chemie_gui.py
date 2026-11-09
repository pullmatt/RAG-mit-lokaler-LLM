import sys
import threading
import re
import io
import json
import base64
import shutil
import filecmp
import gc
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog

# Projektpfad
PROJECT_ROOT = Path(__file__).parent.resolve()
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from rag import ChemistryRAG
from config import config_for_project
from workspace_manager import WorkspaceStore, discover_pdf_library


class ChemistryGUI:
    """
    Einfache Desktop-Oberfläche für die Chemie-RAG.

    Die RAG bleibt vollständig im Hintergrund.
    Die GUI zeigt:
      - Frage und Antwort
      - Quellen
      - Originalseiten als echte PDF-Seitenbilder
      - Ladezustand während Suche/Vision/Qwen
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Chemie-RAG")
        self.root.geometry("1450x900")
        self.root.minsize(1050, 700)

        self.rag = None
        self.rag_by_project = {}
        self.loading_project_ids = set()
        self.busy = False
        self.image_refs = []
        self.math_image_refs = []
        self.workspace = WorkspaceStore(PROJECT_ROOT / "database" / "workspace.sqlite3")
        self.active_project_id, self.active_chat_id = self.workspace.ensure_default_workspace()
        self.active_source_keys = self.workspace.get_project_sources(self.active_project_id)
        self.project_rows = []
        self.chat_rows = []
        self.pending_selected_sources = set()
        self._migrate_legacy_documents_once(self.active_project_id)
        self.workspace_header_var = tk.StringVar(value="Lokale Dokumente · Projekte und persistente Chats")

        self._configure_style()
        self._build_ui()

        self._set_status("Starte projektbezogene Chemie-RAG ...")
        self.loading_project_ids.add(self.active_project_id)
        self.send_button.configure(state="disabled")
        threading.Thread(
            target=self._initialize_rag,
            args=(self.active_project_id,),
            daemon=True
        ).start()

    # =========================================================
    # Oberfläche
    # =========================================================

    def _configure_style(self):
        style = ttk.Style()

        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "Title.TLabel",
            font=("Segoe UI", 20, "bold"),
        )

        style.configure(
            "Subtitle.TLabel",
            font=("Segoe UI", 10),
        )

        style.configure(
            "Status.TLabel",
            font=("Segoe UI", 9),
        )

        style.configure(
            "Send.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=(16, 8),
        )

    def _build_ui(self):
        # -----------------------------------------------------
        # Kopf
        # -----------------------------------------------------

        header = ttk.Frame(self.root, padding=(18, 14, 18, 8))
        header.pack(fill="x")

        ttk.Label(
            header,
            text="Chemie-RAG",
            style="Title.TLabel",
        ).pack(side="left")

        ttk.Label(
            header,
            textvariable=self.workspace_header_var,
            style="Subtitle.TLabel",
        ).pack(side="left", padx=(18, 0))

        self.status_var = tk.StringVar(
            value="Wird gestartet ..."
        )

        ttk.Label(
            header,
            textvariable=self.status_var,
            style="Status.TLabel",
        ).pack(side="right")

        # -----------------------------------------------------
        # Hauptbereich
        # -----------------------------------------------------

        main = ttk.PanedWindow(
            self.root,
            orient="horizontal"
        )
        main.pack(
            fill="both",
            expand=True,
            padx=12,
            pady=(0, 8)
        )

        # Linke Seite: Projekte und Chats
        workspace_frame = ttk.Frame(main, padding=(6, 8))
        main.add(workspace_frame, weight=1)
        self._build_workspace_sidebar(workspace_frame)

        # Mitte: aktiver Chat
        chat_frame = ttk.Frame(main, padding=8)
        main.add(chat_frame, weight=3)

        # Rechte Seite: Originalbilder
        image_frame = ttk.Frame(main, padding=8)
        main.add(image_frame, weight=2)

        # -----------------------------------------------------
        # Chatbereich
        # -----------------------------------------------------

        ttk.Label(
            chat_frame,
            text="Antwort  ·  Formeln werden, sofern möglich, gesetzt dargestellt",
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w", pady=(0, 6))

        chat_container = ttk.Frame(chat_frame)
        chat_container.pack(
            fill="both",
            expand=True
        )

        self.answer_text = tk.Text(
            chat_container,
            wrap="word",
            font=("Segoe UI", 11),
            padx=16,
            pady=14,
            relief="flat",
            bg="#ffffff",
            fg="#202124",
            insertbackground="#202124",
        )

        answer_scroll = ttk.Scrollbar(
            chat_container,
            orient="vertical",
            command=self.answer_text.yview
        )

        self.answer_text.configure(
            yscrollcommand=answer_scroll.set
        )

        self.answer_text.pack(
            side="left",
            fill="both",
            expand=True
        )

        answer_scroll.pack(
            side="right",
            fill="y"
        )

        # Textformatierung
        self.answer_text.tag_configure(
            "question",
            font=("Segoe UI", 11, "bold"),
            foreground="#444444",
            spacing3=10,
        )

        self.answer_text.tag_configure(
            "heading",
            font=("Segoe UI", 14, "bold"),
            spacing1=12,
            spacing3=6,
        )

        self.answer_text.tag_configure(
            "bold",
            font=("Segoe UI", 11, "bold"),
        )

        self.answer_text.tag_configure(
            "code",
            font=("Consolas", 10),
            background="#f0f2f4",
            foreground="#30343b",
        )

        self.answer_text.tag_configure(
            "source",
            font=("Segoe UI", 9),
            foreground="#555555",
        )

        self.answer_text.tag_configure(
            "normal",
            font=("Segoe UI", 11),
            spacing3=5,
        )

        # -----------------------------------------------------
        # Quellen
        # -----------------------------------------------------

        ttk.Label(
            chat_frame,
            text="Verwendete Quellen",
            font=("Segoe UI", 11, "bold"),
        ).pack(
            anchor="w",
            pady=(12, 5)
        )

        sources_container = ttk.Frame(chat_frame)
        sources_container.pack(
            fill="x"
        )

        self.sources_text = tk.Text(
            sources_container,
            height=7,
            wrap="word",
            font=("Segoe UI", 9),
            padx=10,
            pady=8,
            relief="flat",
            bg="#f5f5f5",
            fg="#444444",
        )

        sources_scroll = ttk.Scrollbar(
            sources_container,
            orient="vertical",
            command=self.sources_text.yview
        )

        self.sources_text.configure(
            yscrollcommand=sources_scroll.set
        )

        self.sources_text.pack(
            side="left",
            fill="both",
            expand=True
        )

        sources_scroll.pack(
            side="right",
            fill="y"
        )

        self.sources_text.configure(
            state="disabled"
        )

        # -----------------------------------------------------
        # Eingabe
        # -----------------------------------------------------

        input_frame = ttk.Frame(
            chat_frame,
            padding=(0, 12, 0, 0)
        )
        input_frame.pack(fill="x")

        self.question_entry = tk.Text(
            input_frame,
            height=4,
            wrap="word",
            font=("Segoe UI", 11),
            padx=12,
            pady=10,
            relief="solid",
            borderwidth=1,
        )

        self.question_entry.pack(
            side="left",
            fill="both",
            expand=True
        )

        self.question_entry.bind(
            "<Control-Return>",
            self._on_send
        )

        self.send_button = ttk.Button(
            input_frame,
            text="Fragen",
            style="Send.TButton",
            command=self._on_send,
        )

        self.send_button.pack(
            side="right",
            padx=(10, 0),
            anchor="s"
        )

        # -----------------------------------------------------
        # Rechte Seite: Bilder
        # -----------------------------------------------------

        ttk.Label(
            image_frame,
            text="Originalseiten",
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w")

        ttk.Label(
            image_frame,
            text=(
                "Hier werden die tatsächlich aus dem PDF "
                "gerenderten Seiten angezeigt."
            ),
            font=("Segoe UI", 9),
        ).pack(
            anchor="w",
            pady=(2, 8)
        )

        image_container = ttk.Frame(image_frame)
        image_container.pack(
            fill="both",
            expand=True
        )

        self.image_canvas = tk.Canvas(
            image_container,
            bg="#eeeeee",
            highlightthickness=0,
        )

        image_scroll = ttk.Scrollbar(
            image_container,
            orient="vertical",
            command=self.image_canvas.yview
        )

        self.image_canvas.configure(
            yscrollcommand=image_scroll.set
        )

        self.image_canvas.pack(
            side="left",
            fill="both",
            expand=True
        )

        image_scroll.pack(
            side="right",
            fill="y"
        )

        self.image_inner = ttk.Frame(
            self.image_canvas
        )

        self.image_window = (
            self.image_canvas.create_window(
                (0, 0),
                window=self.image_inner,
                anchor="nw"
            )
        )

        self.image_inner.bind(
            "<Configure>",
            self._update_image_scrollregion
        )

        self.image_canvas.bind(
            "<Configure>",
            self._resize_image_inner
        )

        self._show_image_placeholder()

        # -----------------------------------------------------
        # Tastenkürzel
        # -----------------------------------------------------

        self.root.bind(
            "<Control-l>",
            lambda event: self._clear_question()
        )

        self.root.bind(
            "<Escape>",
            lambda event: self.root.destroy()
        )

    # =========================================================
    # Projekte und Chats
    # =========================================================

    def _build_workspace_sidebar(self, parent):
        ttk.Label(parent, text="Projekte", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 5))
        project_box = ttk.Frame(parent)
        project_box.pack(fill="both", expand=True)
        self.project_listbox = tk.Listbox(project_box, height=8, exportselection=False, activestyle="dotbox")
        project_scroll = ttk.Scrollbar(project_box, orient="vertical", command=self.project_listbox.yview)
        self.project_listbox.configure(yscrollcommand=project_scroll.set)
        self.project_listbox.pack(side="left", fill="both", expand=True)
        project_scroll.pack(side="right", fill="y")
        self.project_listbox.bind("<<ListboxSelect>>", self._on_project_selected)

        project_actions = ttk.Frame(parent)
        project_actions.pack(fill="x", pady=(5, 12))
        ttk.Button(project_actions, text="+ Projekt", command=self._create_project).pack(side="left", fill="x", expand=True)
        ttk.Button(project_actions, text="Umbenennen", command=self._rename_project).pack(side="left", fill="x", expand=True, padx=(4, 0))
        ttk.Button(parent, text="Projekt löschen", command=self._delete_project).pack(fill="x", pady=(0, 12))

        ttk.Separator(parent).pack(fill="x", pady=(0, 10))
        ttk.Label(parent, text="Chats im Projekt", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 5))
        chat_box = ttk.Frame(parent)
        chat_box.pack(fill="both", expand=True)
        self.chat_listbox = tk.Listbox(chat_box, height=12, exportselection=False, activestyle="dotbox")
        chat_scroll = ttk.Scrollbar(chat_box, orient="vertical", command=self.chat_listbox.yview)
        self.chat_listbox.configure(yscrollcommand=chat_scroll.set)
        self.chat_listbox.pack(side="left", fill="both", expand=True)
        chat_scroll.pack(side="right", fill="y")
        self.chat_listbox.bind("<<ListboxSelect>>", self._on_chat_selected)

        ttk.Button(parent, text="+ Neuer Chat", command=self._create_chat).pack(fill="x", pady=(6, 4))
        ttk.Button(parent, text="Chat umbenennen", command=self._rename_chat).pack(fill="x", pady=2)
        ttk.Button(parent, text="Chat-Memory …", command=self._edit_memory).pack(fill="x", pady=2)
        ttk.Button(parent, text="Schnellsuche  Ctrl+K", command=self._quick_search).pack(fill="x", pady=2)
        self.root.bind_all("<Control-k>", lambda event: (self._quick_search(), "break"))
        ttk.Button(parent, text="Chat löschen", command=self._delete_chat).pack(fill="x", pady=2)
        ttk.Button(parent, text="Dateien verwalten …", command=self._manage_sources).pack(fill="x", pady=(10, 2))

    def _refresh_projects(self, select_id=None):
        self.project_rows = self.workspace.list_projects()
        self.project_listbox.delete(0, "end")
        selected_index = None
        for index, project in enumerate(self.project_rows):
            self.project_listbox.insert("end", f"{project['name']}  ·  {project['chat_count']} Chats")
            if project["id"] == (select_id or self.active_project_id):
                selected_index = index
        if selected_index is not None:
            self.project_listbox.selection_clear(0, "end")
            self.project_listbox.selection_set(selected_index)
            self.project_listbox.activate(selected_index)
            self.project_listbox.see(selected_index)

    def _refresh_chats(self, select_id=None):
        self.chat_rows = self.workspace.list_chats(self.active_project_id)
        self.chat_listbox.delete(0, "end")
        selected_index = None
        for index, chat in enumerate(self.chat_rows):
            title = chat["title"] or "Neuer Chat"
            count = int(chat.get("message_count", 0))
            self.chat_listbox.insert("end", f"{title}  ·  {count}")
            if chat["id"] == (select_id or self.active_chat_id):
                selected_index = index
        if selected_index is not None:
            self.chat_listbox.selection_clear(0, "end")
            self.chat_listbox.selection_set(selected_index)
            self.chat_listbox.activate(selected_index)
            self.chat_listbox.see(selected_index)

    def _select_current_project_row(self):
        selected = self.project_listbox.curselection()
        if not selected or selected[0] >= len(self.project_rows):
            return None
        return self.project_rows[selected[0]]

    def _select_current_chat_row(self):
        selected = self.chat_listbox.curselection()
        if not selected or selected[0] >= len(self.chat_rows):
            return None
        return self.chat_rows[selected[0]]

    def _on_project_selected(self, event=None):
        if self.busy:
            self._refresh_projects(self.active_project_id)
            return
        row = self._select_current_project_row()
        if not row or row["id"] == self.active_project_id:
            return
        self.active_project_id = row["id"]
        chats = self.workspace.list_chats(self.active_project_id)
        if not chats:
            self.active_chat_id = self.workspace.create_chat(self.active_project_id, "Neuer Chat")
        else:
            self.active_chat_id = chats[0]["id"]
        self.rag = self.rag_by_project.get(self.active_project_id)
        self.active_source_keys = self.workspace.get_project_sources(self.active_project_id)
        self._refresh_projects(self.active_project_id)
        self._refresh_chats(self.active_chat_id)
        self._activate_chat(self.active_chat_id)
        self._ensure_project_rag(self.active_project_id)

    def _on_chat_selected(self, event=None):
        if self.busy:
            self._refresh_chats(self.active_chat_id)
            return
        row = self._select_current_chat_row()
        if row and row["id"] != self.active_chat_id:
            self._activate_chat(row["id"])

    def _activate_chat(self, chat_id):
        self.active_chat_id = chat_id
        self.active_source_keys = self.workspace.get_project_sources(self.active_project_id)
        messages = self.workspace.get_messages(chat_id)
        self.answer_text.configure(state="normal")
        self.answer_text.delete("1.0", "end")
        latest_sources = []
        history = []
        for message in messages:
            role, content = message["role"], message["content"]
            if role == "user":
                self._append_question(content)
                history.append({"role": "user", "content": content})
            elif role == "assistant":
                self._append_answer(content)
                latest_sources = message.get("sources", []) or []
                history.append({"role": "assistant", "content": content})
        if not messages:
            self.answer_text.insert("end", "Neuer Chat\n\nStelle eine Frage. Der Verlauf wird lokal gespeichert und beim nächsten Start wiederhergestellt.\n", "normal")
        self._show_sources([(str(item[0]), item[1]) for item in latest_sources if isinstance(item, (list, tuple)) and len(item) >= 2])
        self._show_image_placeholder()
        if self.rag is not None:
            self.rag.conversation_history = history[-self.rag.config.MAX_HISTORY_MESSAGES:]
            self.rag.set_document_scope(self.active_source_keys, self.workspace.get_memory_summary(chat_id))
        project_name = next((p["name"] for p in self.workspace.list_projects() if p["id"] == self.active_project_id), "Projekt")
        chat_name = next((c["title"] for c in self.workspace.list_chats(self.active_project_id) if c["id"] == chat_id), "Chat")
        self.workspace_header_var.set(f"{project_name}  /  {chat_name}  ·  {len(self.active_source_keys)} Projektdateien aktiv")
        self._refresh_chats(chat_id)
        self.answer_text.see("end")
        self.question_entry.focus_set()

    def _create_project(self):
        if self.busy:
            return
        name = simpledialog.askstring("Neues Projekt", "Wie soll das Projekt heißen?", parent=self.root)
        if not name or not name.strip():
            return
        try:
            project_id = self.workspace.create_project(name)
            chat_id = self.workspace.create_chat(project_id, "Neuer Chat")
            self.active_project_id, self.active_chat_id = project_id, chat_id
            self.rag = self.rag_by_project.get(project_id)
            self.active_source_keys = self.workspace.get_project_sources(project_id)
            self._refresh_projects(project_id)
            self._refresh_chats(chat_id)
            self._activate_chat(chat_id)
            self._ensure_project_rag(project_id)
        except Exception as error:
            messagebox.showerror("Projekt konnte nicht angelegt werden", str(error), parent=self.root)

    def _rename_project(self):
        project = next((p for p in self.workspace.list_projects() if p["id"] == self.active_project_id), None)
        if not project or self.busy:
            return
        name = simpledialog.askstring("Projekt umbenennen", "Neuer Projektname:", initialvalue=project["name"], parent=self.root)
        if not name or not name.strip():
            return
        try:
            self.workspace.rename_project(self.active_project_id, name)
            self._refresh_projects(self.active_project_id)
            self._activate_chat(self.active_chat_id)
        except Exception as error:
            messagebox.showerror("Projekt konnte nicht umbenannt werden", str(error), parent=self.root)

    def _delete_project(self):
        if self.busy:
            return
        if self.active_project_id in self.loading_project_ids:
            messagebox.showinfo("Projekt wird geladen", "Bitte warte, bis der Projektindex geladen wurde, bevor du das Projekt löschst.", parent=self.root)
            return
        project = next((p for p in self.workspace.list_projects() if p["id"] == self.active_project_id), None)
        if not project:
            return
        cfg = config_for_project(self.active_project_id)
        prompt = (
            f"Projekt '{project['name']}' samt Chats, lokalem Cache, PDF-Dateien und eigenem Suchindex löschen?\n\n"
            "Diese Dateien werden dauerhaft aus dem Projektordner entfernt."
        )
        if not messagebox.askyesno("Projekt vollständig löschen", prompt, parent=self.root):
            return
        deleted_project_id = self.active_project_id
        try:
            self.workspace.delete_project(deleted_project_id)
            old_rag = self.rag_by_project.pop(deleted_project_id, None)
            self.rag = None
            self.loading_project_ids.discard(deleted_project_id)
            del old_rag
            gc.collect()
            shutil.rmtree(cfg.PROJECT_DATA_DIR, ignore_errors=True)
            shutil.rmtree(cfg.DATABASE_DIR, ignore_errors=True)
            projects = self.workspace.list_projects()
            self.active_project_id = projects[0]["id"]
            chats = self.workspace.list_chats(self.active_project_id)
            self.active_chat_id = chats[0]["id"] if chats else self.workspace.create_chat(self.active_project_id, "Neuer Chat")
            self.rag = self.rag_by_project.get(self.active_project_id)
            self.active_source_keys = self.workspace.get_project_sources(self.active_project_id)
            self._refresh_projects(self.active_project_id)
            self._refresh_chats(self.active_chat_id)
            self._activate_chat(self.active_chat_id)
            self._ensure_project_rag(self.active_project_id)
        except Exception as error:
            messagebox.showerror("Projekt konnte nicht gelöscht werden", str(error), parent=self.root)

    def _create_chat(self):
        if self.busy:
            return
        title = simpledialog.askstring("Neuer Chat", "Chatname (optional):", initialvalue="Neuer Chat", parent=self.root)
        if title is None:
            return
        try:
            self.active_chat_id = self.workspace.create_chat(self.active_project_id, title or "Neuer Chat")
            self._refresh_projects(self.active_project_id)
            self._refresh_chats(self.active_chat_id)
            self._activate_chat(self.active_chat_id)
        except Exception as error:
            messagebox.showerror("Chat konnte nicht erstellt werden", str(error), parent=self.root)

    def _rename_chat(self):
        if self.busy:
            return
        chat = next((c for c in self.workspace.list_chats(self.active_project_id) if c["id"] == self.active_chat_id), None)
        if not chat:
            return
        title = simpledialog.askstring("Chat umbenennen", "Neuer Chatname:", initialvalue=chat["title"], parent=self.root)
        if not title or not title.strip():
            return
        try:
            self.workspace.rename_chat(self.active_chat_id, title)
            self._refresh_chats(self.active_chat_id)
            self._activate_chat(self.active_chat_id)
        except Exception as error:
            messagebox.showerror("Chat konnte nicht umbenannt werden", str(error), parent=self.root)

    def _quick_search(self):
        """Lokale Google-artige Suche: PDFs auf Platte, Index-Treffer und frühere Chats."""
        win = tk.Toplevel(self.root)
        win.title("Schnellsuche")
        win.geometry("900x650")
        win.transient(self.root)
        top = ttk.Frame(win, padding=12)
        top.pack(fill="x")
        ttk.Label(top, text="Schnellsuche", font=("Segoe UI", 15, "bold")).pack(anchor="w")
        ttk.Label(
            top,
            text="Durchsucht PDF-Inhalte auch dann, wenn sie noch nicht indexiert sind. Dokumente bleiben auf das aktuelle Projekt begrenzt.",
            wraplength=850,
        ).pack(anchor="w", pady=(3, 0))

        query_var = tk.StringVar()
        row = ttk.Frame(top)
        row.pack(fill="x", pady=(8, 0))
        entry = ttk.Entry(row, textvariable=query_var, font=("Segoe UI", 11))
        entry.pack(side="left", fill="x", expand=True)
        mode_var = tk.StringVar(value="Beides")
        ttk.Combobox(
            row, textvariable=mode_var,
            values=("Beides", "Dokumente", "Chats"),
            state="readonly", width=12,
        ).pack(side="left", padx=8)

        result_box = tk.Text(win, wrap="word", font=("Segoe UI", 10), padx=12, pady=10)
        scroll = ttk.Scrollbar(win, orient="vertical", command=result_box.yview)
        result_box.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        result_box.pack(side="left", fill="both", expand=True, padx=(12, 0), pady=(0, 12))
        result_box.tag_configure("heading", font=("Segoe UI", 11, "bold"), spacing1=8, spacing3=3)
        result_box.tag_configure("meta", foreground="#666666")
        result_box.tag_configure("error", foreground="#a32121")

        search_button = ttk.Button(row, text="Suchen")
        search_button.pack(side="left")

        def run_search(event=None):
            query = query_var.get().strip()
            if not query:
                return "break"
            mode = mode_var.get()
            project_id = self.active_project_id
            result_box.configure(state="normal")
            result_box.delete("1.0", "end")
            result_box.insert("end", "Suche läuft …\n", "meta")
            search_button.configure(state="disabled")

            def worker():
                document_hits = []
                indexed_hits = []
                chat_hits = []
                errors = []
                document_search_paths = []

                if mode in ("Beides", "Dokumente"):
                    rag = self.rag_by_project.get(project_id)
                    search_config = rag.config if rag is not None else config_for_project(project_id)
                    document_search_paths = [str(search_config.DOCUMENTS_DIR), str(search_config.CACHE_DOCUMENTS_DIR)]
                    # Die Direktsuche benötigt weder Ollama noch einen geladenen Chroma-Index.
                    try:
                        if rag is not None:
                            document_hits = rag.quick_search_files(query, limit=30)
                        else:
                            lightweight_searcher = object.__new__(ChemistryRAG)
                            lightweight_searcher.config = config_for_project(project_id)
                            lightweight_searcher._quick_pdf_text_cache = {}
                            document_hits = lightweight_searcher.quick_search_files(query, limit=30)
                    except Exception as error:
                        errors.append(f"Lokale PDF-Suche fehlgeschlagen: {error}")

                    if rag is not None:
                        # Ergänzend semantische Treffer aus dem persistierten Projektindex.
                        try:
                            results = rag.search_documents(query)
                            docs = results.get("documents", [[]])[0] if isinstance(results, dict) else []
                            metas = results.get("metadatas", [[]])[0] if isinstance(results, dict) else []
                            seen = set()
                            for doc, meta in zip(docs, metas):
                                meta = meta or {}
                                source = str(meta.get("source", "Unbekannte Datei"))
                                page = meta.get("page", "?")
                                ident = (source.casefold(), str(page))
                                if ident in seen:
                                    continue
                                seen.add(ident)
                                indexed_hits.append({
                                    "source": source,
                                    "page": page,
                                    "excerpt": " ".join(str(doc).strip().split())[:700],
                                    "is_cache": bool(meta.get("is_cache", False)),
                                })
                                if len(indexed_hits) >= 20:
                                    break
                        except Exception as error:
                            # Index-/Ollama-Fehler dürfen die direkte PDF-Suche nicht zunichtemachen.
                            errors.append(f"Semantische Suche derzeit nicht verfügbar: {error}")
                    else:
                        errors.append("Der Projektindex wird noch geladen; die direkte PDF-Suche wurde trotzdem ausgeführt.")

                if mode in ("Beides", "Chats"):
                    try:
                        chat_hits = self.workspace.search_messages(query, limit=40)
                    except Exception as error:
                        errors.append(f"Chatsuche fehlgeschlagen: {error}")

                def render_results():
                    if not win.winfo_exists():
                        return
                    result_box.configure(state="normal")
                    result_box.delete("1.0", "end")
                    if mode in ("Beides", "Dokumente"):
                        result_box.insert("end", "PDF-Dateien im aktuellen Projekt (Direktsuche)\n", "heading")
                        if document_hits:
                            for hit in document_hits:
                                label = "Klausur/Cache" if hit.get("is_cache") else "Fachwissen"
                                result_box.insert("end", f"{hit['source']} · Seite {hit['page']} · {label}\n", "meta")
                                result_box.insert("end", hit.get("excerpt", "") + "\n\n")
                        else:
                            result_box.insert("end", "Keine passenden Textstellen in den lokalen PDFs gefunden.\n")
                            result_box.insert("end", "Durchsuchte Ordner:\n", "meta")
                            for folder_path in document_search_paths:
                                result_box.insert("end", f"  {folder_path}\n", "meta")

                        result_box.insert("end", "\nSemantische Treffer aus dem Projektindex\n", "heading")
                        if indexed_hits:
                            for hit in indexed_hits:
                                label = "Klausur/Cache" if hit.get("is_cache") else "Fachwissen"
                                result_box.insert("end", f"{hit['source']} · Seite {hit['page']} · {label}\n", "meta")
                                result_box.insert("end", hit.get("excerpt", "") + "\n\n")
                        else:
                            result_box.insert("end", "Keine semantischen Treffer im Index. Die Direktsuche kann trotzdem Ergebnisse liefern.\n")

                    if mode in ("Beides", "Chats"):
                        result_box.insert("end", "\nFrühere Chats (alle Projekte)\n", "heading")
                        if not chat_hits:
                            result_box.insert("end", "Keine passenden Nachrichten gefunden.\n")
                        for hit in chat_hits:
                            result_box.insert(
                                "end",
                                f"{hit['project_name']} / {hit['chat_title']} · {hit['role']} · {hit['created_at']}\n",
                                "meta",
                            )
                            result_box.insert("end", hit.get("excerpt", hit.get("content", "")) + "\n\n")

                    if errors:
                        result_box.insert("end", "\nHinweise\n", "heading")
                        for error in errors:
                            result_box.insert("end", f"• {error}\n", "error")
                    result_box.configure(state="disabled")
                    search_button.configure(state="normal")

                self.root.after(0, render_results)

            threading.Thread(target=worker, daemon=True).start()
            return "break"

        search_button.configure(command=run_search)
        entry.bind("<Return>", run_search)
        entry.focus_set()

    def _edit_memory(self):
        if self.busy:
            messagebox.showinfo("Chat beschäftigt", "Bitte warte, bis die Antwort fertig ist.", parent=self.root)
            return
        win = tk.Toplevel(self.root)
        win.title("Chat-Memory bearbeiten")
        win.geometry("650x430")
        win.transient(self.root)
        win.grab_set()
        ttk.Label(win, text="Dauerhaftes Memory dieses Chats", font=("Segoe UI", 14, "bold")).pack(anchor="w", padx=14, pady=(14, 4))
        ttk.Label(win, text="Diese Notizen werden bei neuen Fragen dieses Chats zusätzlich zum jüngeren Gesprächsverlauf als Kontext übergeben. Das Memory wird nicht mit anderen Chats geteilt.", wraplength=610).pack(anchor="w", padx=14, pady=(0, 8))
        editor = tk.Text(win, wrap="word", font=("Segoe UI", 10), padx=10, pady=8, height=14)
        editor.pack(fill="both", expand=True, padx=14, pady=(0, 10))
        editor.insert("1.0", self.workspace.get_memory_summary(self.active_chat_id))
        actions = ttk.Frame(win, padding=(14, 0, 14, 14))
        actions.pack(fill="x")
        def save_memory():
            summary = editor.get("1.0", "end").strip()
            self.workspace.set_memory_summary(self.active_chat_id, summary)
            if self.rag is not None:
                self.rag.set_document_scope(self.workspace.get_project_sources(self.active_project_id), summary)
            self._set_status("Chat-Memory gespeichert")
            win.destroy()
        ttk.Button(actions, text="Abbrechen", command=win.destroy).pack(side="right")
        ttk.Button(actions, text="Memory speichern", command=save_memory).pack(side="right", padx=(0, 8))

    def _delete_chat(self):
        if self.busy:
            return
        chat = next((c for c in self.workspace.list_chats(self.active_project_id) if c["id"] == self.active_chat_id), None)
        if not chat or not messagebox.askyesno("Chat löschen", f"Chat '{chat['title']}' und seinen Verlauf löschen?", parent=self.root):
            return
        try:
            self.workspace.delete_chat(self.active_chat_id)
            chats = self.workspace.list_chats(self.active_project_id)
            self.active_chat_id = chats[0]["id"] if chats else self.workspace.create_chat(self.active_project_id, "Neuer Chat")
            self._refresh_chats(self.active_chat_id)
            self._activate_chat(self.active_chat_id)
        except Exception as error:
            messagebox.showerror("Chat konnte nicht gelöscht werden", str(error), parent=self.root)

    def _manage_sources(self):
        """Einzelne, projektweite Dateiauswahl mit Ordnerstruktur und Projektindex."""
        if self.busy:
            messagebox.showinfo("Chat beschäftigt", "Bitte warte, bis die Antwort fertig ist.", parent=self.root)
            return
        if self.rag is None:
            messagebox.showinfo("Noch nicht bereit", "Der Projektindex wird noch geladen.", parent=self.root)
            return

        project_id = self.active_project_id
        rag = self.rag
        project_name = next((p["name"] for p in self.workspace.list_projects() if p["id"] == project_id), "Projekt")
        win = tk.Toplevel(self.root)
        win.title(f"Dateien verwalten – {project_name}")
        win.geometry("980x720")
        win.minsize(760, 520)
        win.transient(self.root)
        win.grab_set()

        ttk.Label(win, text=f"Projektdateien · {project_name}", font=("Segoe UI", 15, "bold")).pack(anchor="w", padx=16, pady=(14, 2))
        ttk.Label(
            win,
            text="Eine Auswahl pro Projekt: Alle Chats dieses Projekts verwenden die markierten PDFs. Fachwissen und Klausur-Cache liegen in getrennten Ordnern und in einem eigenen Chroma-Index dieses Projekts.",
            wraplength=920,
        ).pack(anchor="w", padx=16, pady=(0, 10))

        tree_frame = ttk.Frame(win, padding=(12, 0))
        tree_frame.pack(fill="both", expand=True)
        tree = ttk.Treeview(tree_frame, columns=("pages", "kind"), show="tree headings", selectmode="browse")
        tree.heading("#0", text="Ordner / PDF – anklicken zum An- oder Abwählen")
        tree.heading("pages", text="Seiten")
        tree.heading("kind", text="Typ")
        tree.column("#0", width=690, minwidth=360, stretch=True)
        tree.column("pages", width=70, anchor="e", stretch=False)
        tree.column("kind", width=115, anchor="center", stretch=False)
        scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        selected_keys = set(self.workspace.get_project_sources(project_id))
        node_keys = {}
        node_base_labels = {}
        source_by_key = {}
        saved_open_nodes = None
        try:
            saved_open_json = self.workspace.project_setting(project_id, "source_tree_open_nodes", "")
            if saved_open_json:
                parsed_open = json.loads(saved_open_json)
                if isinstance(parsed_open, list):
                    saved_open_nodes = {str(iid) for iid in parsed_open}
        except Exception:
            saved_open_nodes = None

        def _display_path(item):
            raw = str(item["source"]).replace("\\", "/").strip("/")
            if item["is_cache"]:
                try:
                    prefix = rag.config.CACHE_DOCUMENTS_DIR.relative_to(rag.config.PROJECT_ROOT).as_posix().strip("/")
                    if raw.startswith(prefix + "/"):
                        raw = raw[len(prefix) + 1:]
                except Exception:
                    pass
            return raw or Path(raw).name or "Unbenannt.pdf"

        def _symbol(keys):
            if not keys or not any(key in selected_keys for key in keys):
                return "☐"
            if all(key in selected_keys for key in keys):
                return "☑"
            return "◩"

        def refresh_tree():
            # Beim Neuaufbau (z. B. nach einem Import) offene Ordner und
            # Scrollposition erhalten. Das An-/Abwählen selbst baut den Baum
            # nicht mehr neu auf, damit er weder zuklappt noch nach oben springt.
            previous_open = saved_open_nodes if not tree.get_children() else None
            previous_top = 0.0
            if tree.get_children():
                previous_open = set()
                def collect_open(parent=""):
                    for child in tree.get_children(parent):
                        try:
                            if tree.item(child, "open"):
                                previous_open.add(child)
                        except tk.TclError:
                            pass
                        collect_open(child)
                collect_open()
                try:
                    previous_top = float(tree.yview()[0])
                except (tk.TclError, TypeError, ValueError):
                    previous_top = 0.0

            tree.delete(*tree.get_children())
            node_keys.clear()
            node_base_labels.clear()
            source_by_key.clear()
            # Union aus indexierten Quellen UND PDFs auf der Platte. Bisher wurden
            # nur Chroma-Treffer gezeigt; frisch abgelegte/noch nicht indexierte PDFs
            # waren dadurch unsichtbar und konnten nicht ausgewählt werden.
            indexed_sources = rag.get_indexed_sources()
            merged = {item["key"]: dict(item) for item in indexed_sources}
            for is_cache, base_dir in ((False, rag.config.DOCUMENTS_DIR), (True, rag.config.CACHE_DOCUMENTS_DIR)):
                if not base_dir.exists():
                    continue
                for pdf in base_dir.rglob("*"):
                    if not pdf.is_file() or pdf.suffix.casefold() != ".pdf" or pdf.name.startswith("._"):
                        continue
                    try:
                        rel_source = pdf.relative_to(rag.config.PROJECT_ROOT).as_posix() if is_cache else pdf.relative_to(base_dir).as_posix()
                    except ValueError:
                        rel_source = pdf.relative_to(base_dir).as_posix()
                    key = rag.database.source_key(rel_source, is_cache)
                    merged.setdefault(key, {"key": key, "source": rel_source, "is_cache": is_cache,
                                            "page_count": 0, "chunks": 0, "not_indexed": True})
            all_sources = list(merged.values())
            entries = []
            category_keys = {"Fachwissen": set(), "Klausur-Cache": set()}
            folder_keys = {}
            for item in all_sources:
                key = item["key"]
                source_by_key[key] = item
                category = "Klausur-Cache" if item["is_cache"] else "Fachwissen"
                rel = _display_path(item)
                parts = [part for part in rel.split("/") if part and part != "."]
                if not parts:
                    parts = [str(item["source"])]
                entries.append((category, parts, item))
                category_keys[category].add(key)
                for depth in range(1, len(parts)):
                    folder = "/".join(parts[:depth])
                    folder_keys.setdefault((category, folder), set()).add(key)

            roots = {}
            for category in ("Fachwissen", "Klausur-Cache"):
                root_iid = f"root::{category}"
                roots[category] = root_iid
                root_count = len(category_keys[category])
                base_label = f"{category} ({root_count} PDF)"
                open_state = True if previous_open is None else root_iid in previous_open
                tree.insert("", "end", iid=root_iid, text=f"{_symbol(category_keys[category])}  {base_label}", values=("", "Projektordner"), open=open_state)
                node_keys[root_iid] = category_keys[category]
                node_base_labels[root_iid] = base_label

            folder_iids = {}
            for category, folder in sorted(folder_keys, key=lambda row: (row[0], row[1].count("/"), row[1].casefold())):
                parts = folder.split("/")
                parent_folder = "/".join(parts[:-1])
                parent_iid = folder_iids.get((category, parent_folder), roots[category])
                iid = f"folder::{category}::{folder}"
                keys = folder_keys[(category, folder)]
                base_label = parts[-1]
                open_state = True if previous_open is None else iid in previous_open
                tree.insert(parent_iid, "end", iid=iid, text=f"{_symbol(keys)}  {base_label}", values=("", "Ordner"), open=open_state)
                folder_iids[(category, folder)] = iid
                node_keys[iid] = keys
                node_base_labels[iid] = base_label

            for category, parts, item in sorted(entries, key=lambda row: (row[0], "/".join(row[1]).casefold())):
                key = item["key"]
                folder = "/".join(parts[:-1])
                parent_iid = folder_iids.get((category, folder), roots[category])
                iid = f"file::{key}"
                base_label = parts[-1]
                tree.insert(
                    parent_iid, "end", iid=iid,
                    text=f"{_symbol({key})}  {base_label}",
                    values=(item.get("page_count", 0), ("Noch nicht indexiert" if item.get("not_indexed") else ("Klausur/Cache" if item["is_cache"] else "Fachwissen"))),
                    open=False,
                )
                node_keys[iid] = {key}
                node_base_labels[iid] = base_label

            if not all_sources:
                empty_id = "empty::project"
                tree.insert("", "end", iid=empty_id, text="Noch keine PDFs im Projekt. Importiere eine Datei oder einen Ordner.", values=("", ""))
            tree.update_idletasks()
            if previous_open is not None:
                def restore_scroll():
                    try:
                        tree.yview_moveto(max(0.0, min(1.0, previous_top)))
                    except tk.TclError:
                        pass
                win.after_idle(restore_scroll)

        def update_tree_symbols():
            """Aktualisiert nur die Checkbox-Symbole; keine Treeview-Neuerstellung."""
            for iid, keys in list(node_keys.items()):
                if not tree.exists(iid):
                    continue
                base_label = node_base_labels.get(iid, tree.item(iid, "text"))
                tree.item(iid, text=f"{_symbol(keys)}  {base_label}")

        def toggle_node(event):
            region = tree.identify("region", event.x, event.y)
            if region != "tree":
                return
            element = tree.identify("element", event.x, event.y)
            if "indicator" in str(element).lower():
                return
            iid = tree.identify_row(event.y)
            keys = node_keys.get(iid)
            if not iid or not keys:
                return "break"
            should_select = not all(key in selected_keys for key in keys)
            if should_select:
                selected_keys.update(keys)
            else:
                selected_keys.difference_update(keys)
            update_tree_symbols()
            return "break"

        tree.bind("<Button-1>", toggle_node)

        def source_key_for_path(path, kind):
            if kind == "cache":
                source = path.relative_to(rag.config.PROJECT_ROOT).as_posix()
            else:
                source = path.relative_to(rag.config.DOCUMENTS_DIR).as_posix()
            return rag.database.source_key(source, kind == "cache")

        def run_indexing(target_paths, kind, display_name):
            if not target_paths:
                messagebox.showinfo("Keine PDFs gefunden", "Der ausgewählte Ordner enthält keine PDF-Dateien.", parent=win)
                return
            self._set_status(f"Indexiere {len(target_paths)} PDF(s) im Projekt …")

            def worker():
                try:
                    if kind == "cache":
                        rag.sync_cache()
                    else:
                        rag.index_documents()
                    requested_keys = {source_key_for_path(path, kind) for path in target_paths}
                    indexed_keys = {item["key"] for item in rag.get_indexed_sources()}
                    keys = requested_keys.intersection(indexed_keys)
                    if not keys:
                        raise RuntimeError("Die kopierten PDFs haben keine indexierbaren Text-/Cache-Einträge erzeugt.")
                    self.root.after(0, lambda: import_done(keys, None, display_name))
                except Exception as error:
                    self.root.after(0, lambda err=error: import_done(set(), err, display_name))

            def import_done(keys, error, name):
                if error:
                    messagebox.showerror("Indexierung fehlgeschlagen", f"{name} wurde in den Projektordner kopiert, konnte aber nicht vollständig indexiert werden.\n\n{error}", parent=self.root)
                    self._set_status("Indexierung fehlgeschlagen")
                    refresh_tree()
                    return
                selected_keys.update(keys)
                self.workspace.set_project_sources(project_id, selected_keys)
                self.workspace.mark_project_sources_initialized(project_id)
                if self.active_project_id == project_id:
                    self.active_source_keys = set(selected_keys)
                    rag.set_document_scope(self.active_source_keys, self.workspace.get_memory_summary(self.active_chat_id))
                    self._refresh_projects(project_id)
                    self._activate_chat(self.active_chat_id)
                    self._set_status(f"Importiert und im Projekt indexiert: {name}")
                refresh_tree()

            threading.Thread(target=worker, daemon=True).start()

        def import_pdf(kind):
            selected = filedialog.askopenfilename(parent=win, title="PDF zum Projekt hinzufügen", filetypes=[("PDF-Dateien", "*.pdf")])
            if not selected:
                return
            source_path = Path(selected)
            base_dir = rag.config.CACHE_DOCUMENTS_DIR if kind == "cache" else rag.config.DOCUMENTS_DIR
            base_dir.mkdir(parents=True, exist_ok=True)
            target = base_dir / source_path.name
            counter = 2
            while target.exists():
                target = base_dir / f"{source_path.stem} ({counter}){source_path.suffix}"
                counter += 1
            try:
                shutil.copy2(source_path, target)
            except Exception as error:
                messagebox.showerror("Import fehlgeschlagen", str(error), parent=win)
                return
            run_indexing([target], kind, source_path.name)

        def import_folder(kind):
            selected = filedialog.askdirectory(parent=win, title="Ordner mit PDFs auswählen")
            if not selected:
                return
            source_dir = Path(selected)
            pdfs = sorted(p for p in source_dir.rglob("*") if p.is_file() and p.suffix.casefold() == ".pdf" and not p.name.startswith("._"))
            if not pdfs:
                messagebox.showinfo("Keine PDFs gefunden", "Im ausgewählten Ordner wurden keine PDFs gefunden.", parent=win)
                return
            base_dir = rag.config.CACHE_DOCUMENTS_DIR if kind == "cache" else rag.config.DOCUMENTS_DIR
            base_dir.mkdir(parents=True, exist_ok=True)
            target_root = base_dir / source_dir.name
            counter = 2
            original_target_root = target_root
            while target_root.exists():
                target_root = base_dir / f"{original_target_root.name} ({counter})"
                counter += 1
            copied = []
            try:
                for pdf in pdfs:
                    relative = pdf.relative_to(source_dir)
                    target = target_root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(pdf, target)
                    copied.append(target)
            except Exception as error:
                messagebox.showerror("Ordnerimport fehlgeschlagen", f"Ein Teil des Ordners konnte nicht kopiert werden.\n\n{error}", parent=win)
                return
            run_indexing(copied, kind, f"{source_dir.name} ({len(copied)} PDFs)")

        def select_from_library():
            """Importiert vorhandene PDFs aus alten/globalen Ordnern und anderen Projekten."""
            try:
                library_entries = discover_pdf_library(
                    PROJECT_ROOT, self.workspace.list_projects(), project_id
                )
            except Exception as error:
                messagebox.showerror("Bibliothek konnte nicht gelesen werden", str(error), parent=win)
                return
            if not library_entries:
                messagebox.showinfo(
                    "Keine weiteren PDFs gefunden",
                    "In alten globalen Ordnern und anderen Projekten wurden keine weiteren PDF-Dateien gefunden.",
                    parent=win,
                )
                return

            picker = tk.Toplevel(win)
            picker.title("Vorhandene PDFs in dieses Projekt übernehmen")
            picker.geometry("900x640")
            picker.minsize(680, 460)
            picker.transient(win)
            picker.grab_set()
            ttk.Label(
                picker,
                text="PDF-Bibliothek",
                font=("Segoe UI", 14, "bold"),
            ).pack(anchor="w", padx=14, pady=(12, 2))
            ttk.Label(
                picker,
                text="Wähle Dateien oder ganze Ordner aus alten globalen Ordnern und anderen Projekten. Ausgewählte PDFs werden in dieses Projekt kopiert und hier separat indexiert.",
                wraplength=860,
            ).pack(anchor="w", padx=14, pady=(0, 8))
            lib_frame = ttk.Frame(picker, padding=(12, 0))
            lib_frame.pack(fill="both", expand=True)
            lib_tree = ttk.Treeview(lib_frame, columns=("kind", "path"), show="tree headings", selectmode="browse")
            lib_tree.heading("#0", text="Quelle – anklicken zum An-/Abwählen")
            lib_tree.heading("kind", text="Typ")
            lib_tree.heading("path", text="Relativer Pfad")
            lib_tree.column("#0", width=390, minwidth=250, stretch=True)
            lib_tree.column("kind", width=120, stretch=False, anchor="center")
            lib_tree.column("path", width=300, stretch=True)
            lib_scroll = ttk.Scrollbar(lib_frame, orient="vertical", command=lib_tree.yview)
            lib_tree.configure(yscrollcommand=lib_scroll.set)
            lib_tree.pack(side="left", fill="both", expand=True)
            lib_scroll.pack(side="right", fill="y")

            selected_entries = set()
            lib_node_entries = {}
            lib_node_labels = {}
            library_by_index = {i: item for i, item in enumerate(library_entries)}
            origin_iids = {}
            kind_iids = {}
            folder_iids = {}

            def entry_symbol(indices):
                if not indices or not any(idx in selected_entries for idx in indices):
                    return "☐"
                if all(idx in selected_entries for idx in indices):
                    return "☑"
                return "◩"

            # Gruppen: Quelle/Projekt -> Fachwissen oder Klausur-Cache -> Ordner -> PDF.
            grouped = {}
            for idx, item in library_by_index.items():
                grouped.setdefault((item["origin_key"], item["origin_label"]), []).append((idx, item))
            group_keys = {}
            for origin_tuple, entries_for_origin in grouped.items():
                origin_key, origin_label = origin_tuple
                root_iid = f"origin::{origin_key}"
                origin_indices = {idx for idx, _ in entries_for_origin}
                lib_tree.insert("", "end", iid=root_iid, text=f"☐  {origin_label} ({len(origin_indices)} PDFs)", values=("Quelle", ""), open=False)
                origin_iids[origin_key] = root_iid
                group_keys[root_iid] = origin_indices
                lib_node_entries[root_iid] = origin_indices
                lib_node_labels[root_iid] = f"{origin_label} ({len(origin_indices)} PDFs)"

                for kind in ("document", "cache"):
                    kind_items = [(idx, item) for idx, item in entries_for_origin if item["kind"] == kind]
                    if not kind_items:
                        continue
                    category_label = "Fachwissen" if kind == "document" else "Klausur-Cache"
                    kind_iid = f"kind::{origin_key}::{kind}"
                    indices = {idx for idx, _ in kind_items}
                    lib_tree.insert(root_iid, "end", iid=kind_iid, text=f"☐  {category_label} ({len(indices)} PDFs)", values=(category_label, ""), open=False)
                    kind_iids[(origin_key, kind)] = kind_iid
                    lib_node_entries[kind_iid] = indices
                    lib_node_labels[kind_iid] = f"{category_label} ({len(indices)} PDFs)"

                    local_folders = {}
                    for idx, item in kind_items:
                        parts = [part for part in item["relative_path"].split("/") if part]
                        for depth in range(1, len(parts)):
                            folder = "/".join(parts[:depth])
                            local_folders.setdefault(folder, set()).add(idx)
                    for folder in sorted(local_folders, key=lambda value: (value.count("/"), value.casefold())):
                        parent_path = "/".join(folder.split("/")[:-1])
                        parent_iid = folder_iids.get((origin_key, kind, parent_path), kind_iid)
                        iid = f"libfolder::{origin_key}::{kind}::{folder}"
                        base = folder.split("/")[-1]
                        indices = local_folders[folder]
                        lib_tree.insert(parent_iid, "end", iid=iid, text=f"☐  {base}", values=("Ordner", folder), open=False)
                        folder_iids[(origin_key, kind, folder)] = iid
                        lib_node_entries[iid] = indices
                        lib_node_labels[iid] = base
                    for idx, item in sorted(kind_items, key=lambda pair: pair[1]["relative_path"].casefold()):
                        parts = [part for part in item["relative_path"].split("/") if part]
                        parent_folder = "/".join(parts[:-1])
                        parent_iid = folder_iids.get((origin_key, kind, parent_folder), kind_iid)
                        iid = f"libfile::{idx}"
                        lib_tree.insert(
                            parent_iid, "end", iid=iid,
                            text=f"☐  {parts[-1]}",
                            values=(category_label, item["relative_path"]),
                            open=False,
                        )
                        lib_node_entries[iid] = {idx}
                        lib_node_labels[iid] = parts[-1]

            def update_library_symbols():
                for iid, indices in list(lib_node_entries.items()):
                    if lib_tree.exists(iid):
                        lib_tree.item(iid, text=f"{entry_symbol(indices)}  {lib_node_labels.get(iid, '')}")

            def toggle_library_item(event):
                if lib_tree.identify("region", event.x, event.y) != "tree":
                    return
                element = lib_tree.identify("element", event.x, event.y)
                if "indicator" in str(element).lower():
                    return
                iid = lib_tree.identify_row(event.y)
                indices = lib_node_entries.get(iid)
                if not indices:
                    return "break"
                if all(idx in selected_entries for idx in indices):
                    selected_entries.difference_update(indices)
                else:
                    selected_entries.update(indices)
                update_library_symbols()
                return "break"

            lib_tree.bind("<Button-1>", toggle_library_item)
            update_library_symbols()

            def import_selected_library_entries():
                if not selected_entries:
                    messagebox.showinfo("Keine Dateien ausgewählt", "Wähle mindestens eine PDF aus.", parent=picker)
                    return
                import_entries = [library_by_index[idx] for idx in sorted(selected_entries)]
                prepared = []
                try:
                    for item in import_entries:
                        source_path = Path(item["path"])
                        kind = item["kind"]
                        base_dir = rag.config.CACHE_DOCUMENTS_DIR if kind == "cache" else rag.config.DOCUMENTS_DIR
                        # Jeder Import bekommt einen eigenen Herkunftsordner. Die Originale
                        # bleiben unangetastet und der Zielindex bleibt projektlokal.
                        readable_origin = re.sub(r"[^A-Za-z0-9_-]+", "_", item["origin_label"]).strip("_") or "Bibliothek"
                        unique_suffix = re.sub(r"[^A-Za-z0-9_-]+", "_", item["origin_key"]).strip("_")[-12:]
                        safe_origin = f"{readable_origin[:48]}_{unique_suffix}"
                        relative = Path(item["relative_path"])
                        target = base_dir / "Bibliothek" / safe_origin / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if target.exists():
                            try:
                                same_file = filecmp.cmp(source_path, target, shallow=False)
                            except OSError:
                                same_file = False
                            if not same_file:
                                counter = 2
                                original = target
                                while target.exists():
                                    target = original.with_name(f"{original.stem} ({counter}){original.suffix}")
                                    counter += 1
                            else:
                                prepared.append((target, kind))
                                continue
                        shutil.copy2(source_path, target)
                        prepared.append((target, kind))
                except Exception as error:
                    messagebox.showerror("Import fehlgeschlagen", f"Mindestens eine PDF konnte nicht kopiert werden. Die bereits kopierten Dateien bleiben erhalten.\n\n{error}", parent=picker)
                    refresh_tree()
                    return

                picker.destroy()
                win.title(f"Dateien verwalten – {project_name} (Indexierung …)")
                self._set_status(f"Übernehme {len(prepared)} Bibliotheks-PDF(s) ins Projekt …")

                def worker():
                    try:
                        # Beide Suchbereiche inkrementell synchronisieren. Es gibt
                        # keinen globalen Index und keinen Zugriff auf andere Projekte.
                        rag.index_documents()
                        rag.sync_cache()
                        indexed_rows = rag.get_indexed_sources()
                        indexed_keys = {item["key"] for item in indexed_rows}
                        requested = set()
                        for path, kind in prepared:
                            try:
                                if kind == "cache":
                                    source = path.relative_to(rag.config.PROJECT_ROOT).as_posix()
                                else:
                                    source = path.relative_to(rag.config.DOCUMENTS_DIR).as_posix()
                                requested.add(rag.database.source_key(source, kind == "cache"))
                            except ValueError:
                                continue
                        keys = requested.intersection(indexed_keys)
                        if not keys:
                            raise RuntimeError("Die ausgewählten PDFs wurden kopiert, aber es entstanden keine passenden Indexeinträge.")
                        self.root.after(0, lambda: library_import_done(keys, None, len(prepared)))
                    except Exception as error:
                        self.root.after(0, lambda err=error: library_import_done(set(), err, len(prepared)))

                def library_import_done(keys, error, count):
                    if error:
                        messagebox.showerror(
                            "Bibliotheksimport fehlgeschlagen",
                            f"Die PDFs wurden soweit möglich in dieses Projekt kopiert, konnten aber nicht vollständig indexiert werden.\n\n{error}",
                            parent=self.root,
                        )
                        self._set_status("Bibliotheksimport: Indexierung fehlgeschlagen")
                        try:
                            if win.winfo_exists():
                                refresh_tree()
                        except tk.TclError:
                            pass
                        return
                    selected_keys.update(keys)
                    self.workspace.set_project_sources(project_id, selected_keys)
                    self.workspace.mark_project_sources_initialized(project_id)
                    if self.active_project_id == project_id:
                        self.active_source_keys = set(selected_keys)
                        rag.set_document_scope(self.active_source_keys, self.workspace.get_memory_summary(self.active_chat_id))
                        self._refresh_projects(project_id)
                        self._activate_chat(self.active_chat_id)
                        self._set_status(f"{len(keys)} Bibliotheksdatei(en) übernommen und indexiert")
                    try:
                        if win.winfo_exists():
                            refresh_tree()
                    except tk.TclError:
                        pass

                threading.Thread(target=worker, daemon=True).start()

            lib_footer = ttk.Frame(picker, padding=12)
            lib_footer.pack(fill="x")
            ttk.Label(lib_footer, text="Ordner-Klick wählt alle enthaltenen PDFs aus oder ab.", foreground="#555555").pack(side="left")
            ttk.Button(lib_footer, text="Abbrechen", command=picker.destroy).pack(side="right")
            ttk.Button(lib_footer, text="Auswahl ins Projekt übernehmen", command=import_selected_library_entries).pack(side="right", padx=(0, 8))

        def save_selection():
            save_tree_state()
            self.workspace.set_project_sources(project_id, selected_keys)
            self.workspace.mark_project_sources_initialized(project_id)
            self.active_source_keys = set(selected_keys)
            rag.set_document_scope(self.active_source_keys, self.workspace.get_memory_summary(self.active_chat_id))
            self._refresh_projects(project_id)
            self._activate_chat(self.active_chat_id)
            self._set_status(f"{len(selected_keys)} Projektdateien aktiviert")
            win.destroy()

        def save_tree_state():
            open_nodes = []
            def collect_open(parent=""):
                for child in tree.get_children(parent):
                    try:
                        if tree.item(child, "open"):
                            open_nodes.append(child)
                    except tk.TclError:
                        pass
                    collect_open(child)
            collect_open()
            try:
                self.workspace.set_project_setting(
                    project_id, "source_tree_open_nodes", json.dumps(open_nodes, ensure_ascii=False)
                )
            except Exception as error:
                print(f"[GUI] Ordnerzustand konnte nicht gespeichert werden: {error}")

        def close_manager():
            save_tree_state()
            try:
                win.grab_release()
            except tk.TclError:
                pass
            win.destroy()

        refresh_tree()
        win.protocol("WM_DELETE_WINDOW", close_manager)
        file_actions = ttk.Frame(win, padding=(12, 10, 12, 4))
        file_actions.pack(fill="x")
        ttk.Button(file_actions, text="Fach-PDF hinzufügen …", command=lambda: import_pdf("document")).pack(side="left")
        ttk.Button(file_actions, text="Klausur-PDF hinzufügen …", command=lambda: import_pdf("cache")).pack(side="left", padx=(6, 0))
        ttk.Button(file_actions, text="Fachwissen-Ordner importieren …", command=lambda: import_folder("document")).pack(side="left", padx=(6, 0))
        ttk.Button(file_actions, text="Klausurordner importieren …", command=lambda: import_folder("cache")).pack(side="left", padx=(6, 0))
        ttk.Button(file_actions, text="Aus Datenbank/Bibliothek wählen …", command=select_from_library).pack(side="left", padx=(6, 0))

        footer = ttk.Frame(win, padding=12)
        footer.pack(fill="x")
        ttk.Label(footer, text="Ordner-Klick = alle enthaltenen PDFs wählen oder abwählen", foreground="#555555").pack(side="left")
        ttk.Button(footer, text="Abbrechen", command=close_manager).pack(side="right")
        ttk.Button(footer, text="Auswahl speichern", command=save_selection).pack(side="right", padx=(0, 8))

    # =========================================================
    # Initialisierung
    # =========================================================

    def _migrate_legacy_documents_once(self, active_project_id):
        """Übernimmt vorhandene PDFs einmalig in Projektordner, ohne global zu indexieren.

        Vorherige Projektfreigaben werden berücksichtigt. Dokumente, die keinem
        alten Projekt zugeordnet waren, landen im aktiven Projekt als Ausgangsbasis.
        Die alten Originaldateien bleiben als Sicherheitskopie unberührt.
        """
        marker = Path(__file__).parent / "database" / "project_storage_migration_v2.json"
        legacy_documents = Path(__file__).parent / "documents"
        legacy_cache = Path(__file__).parent / "cache"
        def pdfs_in(folder):
            if not folder.exists():
                return []
            return sorted(
                (path for path in folder.rglob("*")
                 if path.is_file() and path.suffix.casefold() == ".pdf"
                 and not path.name.startswith("._")),
                key=lambda path: str(path).casefold(),
            )

        document_files = pdfs_in(legacy_documents)
        cache_files = pdfs_in(legacy_cache)
        if marker.exists():
            active_cfg = config_for_project(active_project_id)
            processed_sources = set()
            try:
                marker_data = json.loads(marker.read_text(encoding="utf-8"))
                if isinstance(marker_data, dict):
                    processed_sources = set(str(x) for x in marker_data.get("processed_sources", []))
            except Exception:
                # Alte V2-Marker waren Klartext. Nutze die bisher gespeicherten
                # Freigaben, damit ein PDF, das einem anderen Projekt gehört,
                # nicht bei der Reparatur in das aktive Projekt kopiert wird.
                active_old_keys = {
                    str(key).replace("\\", "/")
                    for key in self.workspace.get_project_sources(active_project_id)
                }
                for chat in self.workspace.list_chats(active_project_id):
                    active_old_keys.update(
                        str(key).replace("\\", "/")
                        for key in self.workspace.get_chat_sources(chat["id"])
                    )
                other_old_keys = set()
                for project in self.workspace.list_projects():
                    if project["id"] == active_project_id:
                        continue
                    other_old_keys.update(
                        str(key).replace("\\", "/")
                        for key in self.workspace.get_project_sources(project["id"])
                    )
                    for chat in self.workspace.list_chats(project["id"]):
                        other_old_keys.update(
                            str(key).replace("\\", "/")
                            for key in self.workspace.get_chat_sources(chat["id"])
                        )
                for pdf in document_files:
                    relative = pdf.relative_to(legacy_documents)
                    source_key = f"document::{relative.as_posix()}"
                    if (active_cfg.DOCUMENTS_DIR / relative).exists() or (
                        source_key in other_old_keys and source_key not in active_old_keys
                    ):
                        processed_sources.add(f"documents:{relative.as_posix()}")
                for pdf in cache_files:
                    relative = pdf.relative_to(legacy_cache)
                    old_source_key = f"cache::{pdf.relative_to(Path(__file__).parent).as_posix()}"
                    if (active_cfg.CACHE_DOCUMENTS_DIR / relative).exists() or (
                        old_source_key in other_old_keys and old_source_key not in active_old_keys
                    ):
                        processed_sources.add(f"cache:{relative.as_posix()}")

            active_cfg.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
            active_cfg.CACHE_DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
            copied = 0
            for pdf in document_files:
                relative = pdf.relative_to(legacy_documents)
                source_key = f"documents:{relative.as_posix()}"
                if source_key in processed_sources:
                    continue
                target = active_cfg.DOCUMENTS_DIR / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copy2(pdf, target)
                    copied += 1
                processed_sources.add(source_key)
            for pdf in cache_files:
                relative = pdf.relative_to(legacy_cache)
                source_key = f"cache:{relative.as_posix()}"
                if source_key in processed_sources:
                    continue
                target = active_cfg.CACHE_DOCUMENTS_DIR / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copy2(pdf, target)
                    copied += 1
                processed_sources.add(source_key)

            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({
                "version": 3,
                "active_project": active_project_id,
                "processed_sources": sorted(processed_sources),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            if copied:
                print(f"[WORKSPACE] Migration repariert: {copied} neue PDFs in das aktive Projekt kopiert.")
            return

        document_files = pdfs_in(legacy_documents)
        cache_files = pdfs_in(legacy_cache)
        assigned_docs = {}
        assigned_cache = {}
        projects = self.workspace.list_projects()

        for project in projects:
            project_id = project["id"]
            allowed = {str(key).replace("\\", "/") for key in self.workspace.get_project_sources(project_id)}
            for chat in self.workspace.list_chats(project_id):
                allowed.update(str(key).replace("\\", "/") for key in self.workspace.get_chat_sources(chat["id"]))
            for pdf in document_files:
                relative = pdf.relative_to(legacy_documents).as_posix()
                key = f"document::{relative}"
                if key in allowed:
                    assigned_docs.setdefault(project_id, set()).add(pdf)
            for pdf in cache_files:
                relative_root = pdf.relative_to(Path(__file__).parent).as_posix()
                key = f"cache::{relative_root}"
                if key in allowed:
                    assigned_cache.setdefault(project_id, set()).add(pdf)

        assigned_any = set()
        copied_count = 0
        for project in projects:
            project_id = project["id"]
            config = config_for_project(project_id)
            config.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
            config.CACHE_DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

            for pdf in assigned_docs.get(project_id, set()):
                relative = pdf.relative_to(legacy_documents)
                target = config.DOCUMENTS_DIR / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copy2(pdf, target)
                    copied_count += 1
                assigned_any.add(pdf)

            for pdf in assigned_cache.get(project_id, set()):
                relative = pdf.relative_to(legacy_cache)
                target = config.CACHE_DOCUMENTS_DIR / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copy2(pdf, target)
                    copied_count += 1
                assigned_any.add(pdf)

        # Alte globale Dateien, für die es keine gespeicherte Projektzuordnung
        # gibt, werden nur in das aktive Projekt übernommen.
        active_config = config_for_project(active_project_id)
        active_config.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        active_config.CACHE_DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        for pdf in document_files:
            if pdf in assigned_any:
                continue
            target = active_config.DOCUMENTS_DIR / pdf.relative_to(legacy_documents)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(pdf, target)
                copied_count += 1
        for pdf in cache_files:
            if pdf in assigned_any:
                continue
            target = active_config.CACHE_DOCUMENTS_DIR / pdf.relative_to(legacy_cache)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(pdf, target)
                copied_count += 1

        marker.parent.mkdir(parents=True, exist_ok=True)
        processed_sources = {
            *(f"documents:{pdf.relative_to(legacy_documents).as_posix()}" for pdf in document_files),
            *(f"cache:{pdf.relative_to(legacy_cache).as_posix()}" for pdf in cache_files),
        }
        marker.write_text(json.dumps({
            "version": 3,
            "active_project": active_project_id,
            "copied_pdfs": copied_count,
            "processed_sources": sorted(processed_sources),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        if copied_count:
            print(f"[WORKSPACE] {copied_count} bestehende PDFs wurden projektbezogen übernommen.")

    def _initialize_rag(self, project_id):
        try:
            config = config_for_project(project_id)
            rag = ChemistryRAG(config=config)
            rag.index_documents()
            rag.sync_cache()

            indexed_items = rag.get_indexed_sources()
            indexed_keys = {item["key"] for item in indexed_items}
            existing_keys = self.workspace.get_project_sources(project_id)
            legacy_selected_keys = {str(key).replace("\\", "/") for key in existing_keys}
            for old_chat in self.workspace.list_chats(project_id):
                legacy_selected_keys.update(str(key).replace("\\", "/") for key in self.workspace.get_chat_sources(old_chat["id"]))
            if not self.workspace.project_sources_initialized(project_id):
                valid_existing = legacy_selected_keys.intersection(indexed_keys)
                # Vorherige Cache-Keys begannen mit cache/<relativer Pfad>.
                # Übersetze sie auf den neuen projektlokalen Cache-Pfad.
                config = rag.config
                for item in indexed_items:
                    if not item.get("is_cache"):
                        continue
                    try:
                        rel = Path(item["source"]).relative_to(config.CACHE_DOCUMENTS_DIR.relative_to(config.PROJECT_ROOT)).as_posix()
                    except Exception:
                        prefix = config.CACHE_DOCUMENTS_DIR.relative_to(config.PROJECT_ROOT).as_posix().rstrip("/") + "/"
                        source = str(item["source"]).replace("\\", "/")
                        rel = source[len(prefix):] if source.startswith(prefix) else source
                    old_key = rag.database.source_key(f"cache/{rel}", True)
                    if old_key in legacy_selected_keys:
                        valid_existing.add(item["key"])
                self.workspace.set_project_sources(project_id, valid_existing if valid_existing else indexed_keys)
                self.workspace.mark_project_sources_initialized(project_id)
            else:
                self.workspace.set_project_sources(project_id, existing_keys.intersection(indexed_keys))

            self.root.after(0, lambda: self._project_rag_ready(project_id, rag))
        except Exception as error:
            self.root.after(0, lambda err=error: self._project_rag_failed(project_id, err))

    def _ensure_project_rag(self, project_id):
        if project_id in self.rag_by_project:
            self.rag = self.rag_by_project[project_id]
            self.active_source_keys = self.workspace.get_project_sources(project_id)
            self.rag.set_document_scope(self.active_source_keys, self.workspace.get_memory_summary(self.active_chat_id))
            self.send_button.configure(state="normal")
            self._set_status(f"Projekt bereit · {self.rag.database.count()} Chunks · {len(self.active_source_keys)} Dateien aktiv")
            self._activate_chat(self.active_chat_id)
            return
        if project_id in self.loading_project_ids:
            self.send_button.configure(state="disabled")
            self._set_status("Projektindex wird geladen …")
            return
        self.loading_project_ids.add(project_id)
        self.rag = None
        self.send_button.configure(state="disabled")
        self._set_status("Lade eigenen Cache und Index des Projekts …")
        threading.Thread(target=self._initialize_rag, args=(project_id,), daemon=True).start()

    def _project_rag_ready(self, project_id, rag):
        self.loading_project_ids.discard(project_id)
        self.rag_by_project[project_id] = rag
        if project_id != self.active_project_id:
            return
        self.rag = rag
        self.active_source_keys = self.workspace.get_project_sources(project_id)
        self.rag.set_document_scope(self.active_source_keys, self.workspace.get_memory_summary(self.active_chat_id))
        self._refresh_projects(project_id)
        self._refresh_chats(self.active_chat_id)
        self._activate_chat(self.active_chat_id)
        self.send_button.configure(state="normal")
        self._set_status(f"Projekt bereit · {rag.database.count()} Chunks · {len(self.active_source_keys)} Dateien aktiv")
        self.question_entry.focus_set()

    def _project_rag_failed(self, project_id, error):
        self.loading_project_ids.discard(project_id)
        if project_id != self.active_project_id:
            print(f"[WORKSPACE] Projekt {project_id} konnte nicht geladen werden: {error}")
            return
        self.rag = None
        self.send_button.configure(state="disabled")
        self._set_status("Fehler beim Laden des Projektindex")
        messagebox.showerror("Chemie-RAG", f"Der Index dieses Projekts konnte nicht gestartet werden.\n\n{error}", parent=self.root)

    # =========================================================
    # Fragen
    # =========================================================

    def _on_send(self, event=None):
        if self.busy:
            return "break"

        if self.rag is None:
            messagebox.showinfo(
                "Noch nicht bereit",
                "Die Chemie-RAG wird noch gestartet."
            )
            return "break"

        question = (
            self.question_entry
            .get("1.0", "end")
            .strip()
        )

        if not question:
            return "break"

        self._clear_question()

        if not self.active_chat_id:
            messagebox.showerror("Kein Chat ausgewählt", "Bitte erstelle oder wähle einen Chat aus.")
            return "break"
        self.workspace.add_message(self.active_chat_id, "user", question)
        # Der erste Turn benennt einen unbenannten Chat automatisch nach dem Thema.
        current_chat = next((c for c in self.workspace.list_chats(self.active_project_id) if c["id"] == self.active_chat_id), None)
        if current_chat and current_chat["title"].strip().lower() == "neuer chat":
            title = re.sub(r"\s+", " ", question).strip()[:42]
            self.workspace.rename_chat(self.active_chat_id, title or "Chat")
            self._refresh_chats(self.active_chat_id)
        self._append_question(question)

        self.busy = True
        self.send_button.configure(
            state="disabled"
        )

        self._set_status(
            "Suche relevante Quellen ..."
        )

        threading.Thread(
            target=self._answer_worker,
            args=(question, self.active_chat_id, set(self.active_source_keys)),
            daemon=True
        ).start()

        return "break"

    def _answer_worker(self, question, chat_id, source_keys):
        try:
            # Das Suchlimit wird pro Turn gesetzt und bleibt auch für Cache- und
            # Vision-Retrieval aktiv. Kein stiller Rückfall auf fremde Quellen.
            self.rag.set_document_scope(source_keys, self.workspace.get_memory_summary(chat_id))
            answer, results = self.rag.answer_question(question)

            images = (
                self.rag._source_images(
                    results
                )
            )

            sources = (
                self.rag.get_sources(
                    results
                )
            )

            self.root.after(
                0,
                lambda: self._show_result(
                    answer, sources, images, chat_id, question
                )
            )

        except Exception as error:
            self.root.after(
                0,
                lambda err=error: self._show_error(err)
            )

    def _show_result(
        self,
        answer,
        sources,
        images,
        chat_id=None,
        question=None,
    ):
        self._append_answer(answer)
        source_payload = [[source, page] for source, page in (sources or [])]
        if chat_id:
            self.workspace.add_message(chat_id, "assistant", answer, source_payload)
            # Automatische Verdichtung längerer Chats. Läuft nach der Antwort
            # im Hintergrund; vorhandenes Memory wird aktualisiert statt ersetzt.
            try:
                message_count = self.workspace.message_count(chat_id)
                if message_count >= 12 and message_count % 8 == 4:
                    threading.Thread(target=self._auto_summarize_chat, args=(chat_id,), daemon=True).start()
            except Exception as summary_trigger_error:
                print(f"[MEMORY] Automatische Verdichtung nicht gestartet: {summary_trigger_error}")
        # Spezialrouten (z. B. Klausur-Einzelaufgaben) aktualisieren die interne
        # RAG-Historie nicht immer. Hier wird sichergestellt, dass der Austausch
        # spätestens nach der Antwort im aktiven Chat-Kontext vorhanden ist.
        if self.rag is not None and question:
            history = self.rag.conversation_history
            already_added = len(history) >= 2 and history[-2].get("role") == "user" and history[-2].get("content") == question and history[-1].get("role") == "assistant" and history[-1].get("content") == answer
            if not already_added:
                history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
                self.rag.conversation_history = history[-self.rag.config.MAX_HISTORY_MESSAGES:]
        self._show_sources(sources)
        self._refresh_chats(chat_id or self.active_chat_id)
        self._show_images(images)

        self.busy = False
        self.send_button.configure(
            state="normal"
        )

        self._set_status(
            f"Fertig · {len(sources)} Quellen"
        )

        self.question_entry.focus_set()

    def _auto_summarize_chat(self, chat_id):
        """Verdichtet einen langen Verlauf in ein überprüfbares Chat-Memory."""
        try:
            messages = self.workspace.get_messages(chat_id)
            if len(messages) < 12:
                return
            old_memory = self.workspace.get_memory_summary(chat_id)
            # Keep the newest exchanges verbatim in the normal conversation context;
            # the summary retains stable preferences, decisions and unresolved items.
            transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages[-24:])
            rag = self.rag_by_project.get(self.workspace.project_for_chat(chat_id)) if hasattr(self.workspace, "project_for_chat") else self.rag
            if rag is None:
                return
            prompt = f"""Aktualisiere ein kompaktes, dauerhaftes MEMORY für genau diesen Chat.
Nutze nur Informationen aus dem vorhandenen Memory und dem Verlauf. Erfinde nichts.
Behalte stabile Nutzerpräferenzen, bestätigte Entscheidungen, Projekt-/Aufgabenkontext,
offene Fragen und relevante bereits geklärte Fakten. Entferne Wiederholungen und
kurzlebige Details. Trenne gesicherte Fakten von Vermutungen. Maximal 900 Wörter.
Gib nur das aktualisierte Memory aus.

BISHERIGES MEMORY:
{old_memory or '(leer)'}

LETZTE NACHRICHTEN:
{transcript[:18000]}"""
            summary = rag._chat_text("Du bist ein sorgfältiger Zusammenfassungsassistent. Bewahre Fakten und Unsicherheiten korrekt.", prompt, num_predict=1200).strip()
            if summary and not summary.startswith("FEHLER:"):
                self.workspace.set_memory_summary(chat_id, summary)
                if chat_id == self.active_chat_id and self.rag is rag:
                    self.rag.set_document_scope(self.workspace.get_project_sources(self.active_project_id), summary)
                print(f"[MEMORY] Chat {chat_id}: Memory automatisch aktualisiert ({len(summary)} Zeichen).")
        except Exception as e:
            print(f"[MEMORY] Automatische Verdichtung fehlgeschlagen: {e}")

    def _show_error(self, error):
        error_text = "FEHLER:\n\n" + str(error)
        self._append_answer(error_text)
        if self.active_chat_id:
            self.workspace.add_message(self.active_chat_id, "assistant", error_text, [])
        self._refresh_chats(self.active_chat_id)

        self.busy = False
        self.send_button.configure(
            state="normal"
        )

        self._set_status("Fehler")
        messagebox.showerror(
            "Chemie-RAG",
            str(error)
        )

    # =========================================================
    # Antwortdarstellung
    # =========================================================

    def _append_question(self, question):
        self.answer_text.insert(
            "end",
            "\nFrage\n",
            "question"
        )

        self.answer_text.insert(
            "end",
            question + "\n\n",
            "normal"
        )

        self.answer_text.see("end")

    def _append_answer(self, answer):
        self._insert_markdown(answer)
        self.answer_text.insert(
            "end",
            "\n" + ("─" * 80) + "\n",
            "source"
        )
        self.answer_text.see("end")

    def _insert_markdown(self, text):
        """Rendert Markdown und setzt LaTeX-/Chemieformeln als Satzbilder."""
        lines = str(text or "").splitlines()
        index = 0
        in_code = False
        code_buffer = []
        code_language = ""

        while index < len(lines):
            line = lines[index]
            stripped = line.strip()

            # Markdown-Codeblöcke: LaTeX-/Math-Blöcke werden gesetzt, gewöhnliche
            # Codeblöcke behalten einen lesbaren monospace Stil.
            if stripped.startswith("```"):
                language = stripped[3:].strip().lower()
                if not in_code:
                    in_code = True
                    code_buffer = []
                    code_language = language
                else:
                    self._finish_code_block(code_buffer, code_language)
                    in_code = False
                    code_buffer = []
                    code_language = ""
                index += 1
                continue
            if in_code:
                code_buffer.append(line)
                index += 1
                continue

            # Mehrzeilige Display-Mathematik mit \[ ... \], $$ ... $$ sowie
            # equation/align-Umgebungen zusammenfassen, bevor Zeilen getrennt
            # verarbeitet werden.
            opener = None
            closer = None
            if stripped in (r"\[", "$$"):
                opener, closer = stripped, (r"\]" if stripped == r"\[" else "$$")
            elif re.match(r"\\begin\{(?:equation\*?|align\*?|gather\*?|displaymath)\}", stripped):
                env_match = re.search(r"\\begin\{([^}]+)\}", stripped)
                env = env_match.group(1) if env_match else "equation"
                opener, closer = rf"\begin{{{env}}}", rf"\end{{{env}}}"
            if opener:
                formula_lines = []
                # Inhalt hinter dem öffnenden Marker in derselben Zeile erhalten.
                tail = stripped[len(opener):].strip()
                if tail:
                    formula_lines.append(tail)
                index += 1
                while index < len(lines):
                    current = lines[index].strip()
                    if closer in current:
                        before = current.split(closer, 1)[0].strip()
                        if before:
                            formula_lines.append(before)
                        index += 1
                        break
                    formula_lines.append(lines[index])
                    index += 1
                formula = "\n".join(formula_lines).strip()
                self.answer_text.insert("end", "\n")
                if not self._insert_math_image(formula, display=True):
                    self.answer_text.insert("end", self._latex_readable(formula), "normal")
                self.answer_text.insert("end", "\n\n")
                continue

            if not stripped:
                self.answer_text.insert("end", "\n")
                index += 1
                continue
            if stripped.startswith("### "):
                self.answer_text.insert("end", stripped[4:] + "\n", "heading")
                index += 1
                continue
            if stripped.startswith("## "):
                self.answer_text.insert("end", stripped[3:] + "\n", "heading")
                index += 1
                continue
            if stripped.startswith("# "):
                self.answer_text.insert("end", stripped[2:] + "\n", "heading")
                index += 1
                continue
            if stripped.startswith("- ") or stripped.startswith("* "):
                self.answer_text.insert("end", "• ", "normal")
                self._insert_bold_text(stripped[2:])
                self.answer_text.insert("end", "\n")
                index += 1
                continue
            if re_numbered(stripped):
                prefix, body = re_numbered(stripped)
                self.answer_text.insert("end", prefix, "normal")
                self._insert_bold_text(body)
                self.answer_text.insert("end", "\n")
                index += 1
                continue

            # Manche Modellantworten schreiben eine Formel ohne $-Begrenzer als
            # separate Zeile oder geben sie als `...` aus. Erkenne diese Fälle.
            has_inline_markup = bool(re.search(r"(?:\\\(|\\\[|\$|`)", stripped))
            if not has_inline_markup and self._looks_like_formula(stripped) and len(stripped) <= 240:
                formula_line = stripped.strip("`").strip()
                if self._insert_math_image(formula_line, display=True):
                    self.answer_text.insert("end", "\n")
                    index += 1
                    continue
            self._insert_bold_text(stripped)
            self.answer_text.insert("end", "\n")
            index += 1

        # Falls die Antwort mit einem nicht geschlossenen Codeblock endet.
        if in_code and code_buffer:
            self._finish_code_block(code_buffer, code_language)

    def _finish_code_block(self, lines, language):
        content = "\n".join(lines).strip()
        if not content:
            self.answer_text.insert("end", "\n")
            return
        math_languages = {"latex", "tex", "math", "maths", "formula", "equation", "katex"}
        if language in math_languages or (not language and self._looks_like_formula(content)):
            if not self._insert_math_image(content, display=True):
                self.answer_text.insert("end", self._latex_readable(content), "normal")
            self.answer_text.insert("end", "\n\n")
            return
        for code_line in lines:
            self.answer_text.insert("end", code_line + "\n", "code")
        self.answer_text.insert("end", "\n")

    @staticmethod
    def _looks_like_formula(text):
        value = str(text or "").strip().strip("`")
        if not value or len(value) > 260:
            return False
        if re.search(r"\\(?:frac|sqrt|sum|prod|int|Delta|delta|alpha|beta|gamma|theta|mathrm|text|ce|pu|cdot|times|rightarrow|leftarrow|leftrightarrow|approx|leq|geq|neq|mu|sigma|pi|log|ln|exp|pm|partial|nabla)\b", value):
            return True
        # Unmarkierte Gleichungen / Reaktionsgleichungen, aber keine normalen
        # Prosa-Sätze in eine Formel umwandeln.
        has_operator = bool(re.search(r"(?:=|\^|_[{A-Za-z0-9]|->|<=>|→|⇌|±|≈|≤|≥)", value))
        has_wordy_prose = bool(re.search(r"\b(?:because|therefore|which|where|wobei|damit|deshalb|hingegen|außerdem|allerdings|ist|sind|war|wird|werden|eine|einer|eines|ein|der|die|das|den|dem|von|mit|für|aus|durch|kann|können|soll|sollte|formel|freie|enthalpie|energie|reaction|formula)\b", value, re.I))
        # Ein LaTeX-Befehl in einem ganzen deutschen Satz macht nicht den ganzen
        # Satz zur Formel. Unmarkierte Formeln sind meist kurz und operatorreich.
        if has_wordy_prose:
            return False
        return has_operator and len(value.split()) <= 20

    @staticmethod
    def _normalize_math_formula(formula):
        value = str(formula or "").strip()
        value = value.replace(r"\displaystyle", "")
        value = re.sub(r"\\begin\{(?:aligned|align\*?|equation\*?|gather\*?)\}", "", value)
        value = re.sub(r"\\end\{(?:aligned|align\*?|equation\*?|gather\*?)\}", "", value)
        value = value.replace("&", " ").replace(r"\\", " ")
        value = re.sub(r"\\text\{([^{}]*)\}", r"\\mathrm{\1}", value)

        def chemistry_to_math(match):
            chemistry = match.group(1).strip()
            chemistry = chemistry.replace("<=>", r"\rightleftharpoons").replace("⇌", r"\rightleftharpoons")
            chemistry = chemistry.replace("->", r"\rightarrow").replace("→", r"\rightarrow")
            pieces = re.split(r"(\s+|\+|\\rightarrow|\\rightleftharpoons|\\leftarrow|=)", chemistry)
            output = []
            for piece in pieces:
                if not piece or piece.isspace() or piece in {"+", "=", r"\rightarrow", r"\rightleftharpoons", r"\leftarrow"}:
                    output.append(piece)
                else:
                    token = re.sub(r"([A-Za-z\)])(\d+)", r"\1_{\2}", piece)
                    output.append(r"\mathrm{" + token + "}")
            return "".join(output)

        value = re.sub(r"\\ce\{([^{}]*)\}", chemistry_to_math, value)
        value = re.sub(r"\\pu\{([^{}]*)\}", r"\\mathrm{\1}", value)
        # Entferne einzelne Markdown-Codebackticks, die Modelle gelegentlich
        # versehentlich um Formeln setzen.
        value = value.strip("`").strip()
        return value

    @staticmethod
    def _latex_readable(formula):
        """Lesbarer Formelfallback mit Unicode-Hoch-/Tiefstellung für einfache Chemieformeln."""
        value = str(formula or "").strip().strip("$").replace(r"\[", "").replace(r"\]", "")
        value = re.sub(r"\\(?:mathrm|text|operatorname)\{([^{}]*)\}", r"\1", value)
        def chemistry_to_unicode(match):
            formula = match.group(1)
            digit_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
            formula = re.sub(
                r"([A-Za-z\)])(\d+)",
                lambda part: part.group(1) + part.group(2).translate(digit_map),
                formula,
            )
            if formula.endswith("+"):
                formula = formula[:-1] + "⁺"
            elif formula.endswith("-"):
                formula = formula[:-1] + "⁻"
            return formula
        value = re.sub(r"\\ce\{([^{}]*)\}", chemistry_to_unicode, value)
        value = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", value)
        value = re.sub(r"\\sqrt\{([^{}]*)\}", r"√(\1)", value)
        replacements = {
            r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
            r"\Delta": "Δ", r"\theta": "θ", r"\mu": "μ", r"\sigma": "σ",
            r"\pi": "π", r"\lambda": "λ", r"\rightarrow": "→", r"\leftarrow": "←",
            r"\leftrightarrow": "↔", r"\rightleftharpoons": "⇌", r"\cdot": "·",
            r"\times": "×", r"\pm": "±", r"\leq": "≤", r"\geq": "≥",
            r"\neq": "≠", r"\approx": "≈", r"\infty": "∞", r"\degree": "°",
            r"\log": "log", r"\ln": "ln", r"\exp": "exp", r"\mathrm": "",
        }
        for old, replacement in replacements.items():
            value = value.replace(old, replacement)

        subscripts = str.maketrans({
            "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
            "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
            "+": "₊", "-": "₋", "=": "₌", "(": "₍", ")": "₎",
            "a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ",
            "k": "ₖ", "l": "ₗ", "m": "ₘ", "n": "ₙ", "o": "ₒ",
            "p": "ₚ", "r": "ᵣ", "s": "ₛ", "t": "ₜ", "u": "ᵤ",
            "v": "ᵥ", "x": "ₓ",
        })
        superscripts = str.maketrans({
            "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
            "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
            "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾",
            "a": "ᵃ", "b": "ᵇ", "c": "ᶜ", "d": "ᵈ", "e": "ᵉ",
            "f": "ᶠ", "g": "ᵍ", "h": "ʰ", "i": "ⁱ", "j": "ʲ",
            "k": "ᵏ", "l": "ˡ", "m": "ᵐ", "n": "ⁿ", "o": "ᵒ",
            "p": "ᵖ", "r": "ʳ", "s": "ˢ", "t": "ᵗ", "u": "ᵘ",
            "v": "ᵛ", "w": "ʷ", "x": "ˣ", "y": "ʸ", "z": "ᶻ",
        })

        def format_script(match):
            marker = match.group(1)
            content = match.group(2) if match.group(2) is not None else match.group(3)
            table = subscripts if marker == "_" else superscripts
            translated = content.translate(table)
            # Bei mehrstelligen Inhalten ohne vollständige Unicode-Entsprechung
            # bleiben Klammern sichtbar, statt ein kaputtes Codefragment auszugeben.
            simple = all(char.translate(table) != char for char in content) or all(char.isdigit() or char in "+-=()" for char in content)
            if len(content) == 1 and simple:
                return translated
            if simple:
                return translated
            bracket_open, bracket_close = ("₍", "₎") if marker == "_" else ("⁽", "⁾")
            return bracket_open + translated + bracket_close

        value = re.sub(r"([_^])(?:\{([^{}]+)\}|([0-9+\-=()]+|[A-Za-z]))", format_script, value)
        value = value.replace("{", "").replace("}", "")
        value = re.sub(r"\\([A-Za-z]+)", r"\1", value)
        return value

    def _insert_bold_text(self, text):
        """Unterstützt Markdown-Fettdruck, Inline-LaTeX und code-umschlossene Formeln."""
        bold_parts = text.split("**")
        bare_math_token = r'(?<![\w])[A-Za-zΑ-Ωα-ω][A-Za-z0-9Α-Ωα-ω]*(?:[_^](?:\{[^{}\n]+\}|[0-9+\-=()]+)[A-Za-z0-9Α-Ωα-ω]*)+'
        bare_chemistry_token = r'(?<![\w])(?:[A-Z][a-z]?\d*){2,}(?:[+-])?(?![\w])'
        bare_hybrid_token = r'(?<![\w])(?:d\d+sp\d+|dsp\d+|sp\d+d\d*|sp\d+)(?![\w])'
        pattern = re.compile(
            r'(\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|\$\$[^$]+\$\$|\$[^$\n]+\$|`[^`\n]+`|'
            + bare_math_token + r'|' + bare_chemistry_token + r'|' + bare_hybrid_token + r')'
        )
        for bold_index, bold_part in enumerate(bold_parts):
            if not bold_part:
                continue
            tag = "bold" if bold_index % 2 == 1 else "normal"
            cursor = 0
            for match in pattern.finditer(bold_part):
                if match.start() > cursor:
                    self.answer_text.insert("end", bold_part[cursor:match.start()], tag)
                token = match.group(0)
                is_backtick = token.startswith("`") and token.endswith("`")
                formula = token
                if is_backtick:
                    formula = token[1:-1].strip()
                elif formula.startswith("$$") and formula.endswith("$$"):
                    formula = formula[2:-2]
                elif formula.startswith("$") and formula.endswith("$"):
                    formula = formula[1:-1]
                elif formula.startswith(r"\(") and formula.endswith(r"\)"):
                    formula = formula[2:-2]
                elif formula.startswith(r"\[") and formula.endswith(r"\]"):
                    formula = formula[2:-2]
                explicit_math = (not is_backtick) and (
                    token.startswith(("$", r"\(", r"\["))
                )
                bare_sub_or_sup = bool(re.search(r"(?:_|\^)", formula)) and bool(
                    re.fullmatch(r"[A-Za-zΑ-Ωα-ω][A-Za-z0-9Α-Ωα-ω]*(?:[_^](?:\{[^{}]+\}|[0-9+\-=()]+)[A-Za-z0-9Α-Ωα-ω]*)+", formula)
                )
                bare_chemistry = bool(re.fullmatch(r"(?:[A-Z][a-z]?\d*){2,}(?:[+-])?", formula))
                bare_hybrid = bool(re.fullmatch(r"(?:d\d+sp\d+|dsp\d+|sp\d+d\d*|sp\d+)", formula))
                render_formula = formula
                if bare_chemistry:
                    render_formula = r"\ce{" + formula + "}"
                elif bare_hybrid:
                    render_formula = re.sub(r"([spdf])(\d+)", r"\1^{\2}", formula)
                looks_math = explicit_math or self._looks_like_formula(formula) or bare_sub_or_sup or bare_chemistry or bare_hybrid
                if looks_math and self._insert_math_image(render_formula.strip()):
                    pass
                elif is_backtick and (self._looks_like_formula(formula) or bare_sub_or_sup or bare_chemistry or bare_hybrid):
                    # Matplotlib kann nicht jedes Chemie-Kürzel setzen; der Fallback
                    # wandelt Zahlen/Indizes in echte Unicode-Tief-/Hochstellungen um.
                    self.answer_text.insert("end", self._latex_readable(render_formula), tag)
                elif is_backtick:
                    self.answer_text.insert("end", formula, "code")
                else:
                    self.answer_text.insert("end", self._latex_readable(render_formula) if ("\\" in formula or bare_sub_or_sup or bare_chemistry or bare_hybrid) else token, tag)
                cursor = match.end()
            if cursor < len(bold_part):
                tail = bold_part[cursor:]
                if self._looks_like_formula(tail) and len(tail) <= 200 and self._insert_math_image(tail.strip()):
                    continue
                self.answer_text.insert("end", tail, tag)

    def _insert_math_image(self, formula, display=False):
        """Setzt eine LaTeX-Formel als transparentes Bild im Antwort-Textfeld."""
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            normalized = self._normalize_math_formula(formula)
            if not normalized:
                return False
            figure_width = max(1.1, min(12.0, len(normalized) * (0.115 if display else 0.105)))
            figure_height = 0.62 if display or any(token in normalized for token in (r"\frac", r"\sum", r"\int", r"\sqrt")) else 0.40
            fig = Figure(figsize=(figure_width, figure_height), dpi=160)
            fig.patch.set_alpha(0.0)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_axis_off()
            ax.text(
                0.5 if display else 0.01,
                0.5,
                f"${normalized}$",
                fontsize=14 if display else 12.5,
                color="#202124",
                va="center",
                ha="center" if display else "left",
            )
            canvas = FigureCanvasAgg(fig)
            buffer = io.BytesIO()
            canvas.print_png(buffer, transparent=True, bbox_inches="tight", pad_inches=0.06)
            buffer.seek(0)
            encoded = base64.b64encode(buffer.read()).decode("ascii")
            image = tk.PhotoImage(data=encoded)
            self.math_image_refs.append(image)
            self.answer_text.image_create("end", image=image, align="baseline")
            return True
        except Exception:
            return False

    # =========================================================
    # Quellen
    # =========================================================

    def _show_sources(self, sources):
        self.sources_text.configure(
            state="normal"
        )
        self.sources_text.delete(
            "1.0",
            "end"
        )

        if not sources:
            self.sources_text.insert(
                "end",
                "Keine Quellen gefunden.",
                "source"
            )
        else:
            for source, page in sources:
                self.sources_text.insert(
                    "end",
                    f"• {source} — Seite {page}\n",
                    "source"
                )

        self.sources_text.configure(
            state="disabled"
        )

    # =========================================================
    # Originalbilder
    # =========================================================

    def _clear_images(self):
        for widget in self.image_inner.winfo_children():
            widget.destroy()

        self.image_refs = []

    def _show_image_placeholder(self):
        self._clear_images()

        label = ttk.Label(
            self.image_inner,
            text=(
                "Noch keine Originalseiten.\n\n"
                "Wenn eine relevante PDF-Seite\n"
                "gefunden wird, erscheint sie hier."
            ),
            justify="center",
            padding=30,
        )

        label.pack(
            fill="both",
            expand=True
        )

    def _show_images(self, images):
        self._clear_images()

        if not images:
            self._show_image_placeholder()
            return

        for number, item in enumerate(
            images,
            start=1
        ):
            card = ttk.Frame(
                self.image_inner,
                padding=(8, 8, 8, 16)
            )
            card.pack(
                fill="x"
            )

            title = ttk.Label(
                card,
                text=(
                    f"Originalbild {number} · "
                    f"Seite {item['page']}"
                ),
                font=("Segoe UI", 10, "bold"),
                wraplength=450,
            )
            title.pack(
                anchor="w",
                pady=(0, 4)
            )

            source = ttk.Label(
                card,
                text=item["source"],
                font=("Segoe UI", 8),
                wraplength=450,
            )
            source.pack(
                anchor="w",
                pady=(0, 6)
            )

            try:
                image = tk.PhotoImage(
                    data=item["image"]
                )

                # Bei sehr großen Seiten verkleinern.
                width = image.width()
                height = image.height()

                max_width = 520
                max_height = 720

                factor = max(
                    width / max_width,
                    height / max_height,
                    1
                )

                if factor > 1:
                    reduction = int(factor) + 1
                    image = image.subsample(
                        reduction,
                        reduction
                    )

                self.image_refs.append(image)

                image_label = tk.Label(
                    card,
                    image=image,
                    bg="white",
                    bd=1,
                    relief="solid",
                )

                image_label.pack(
                    anchor="center"
                )

            except Exception as error:
                error_label = ttk.Label(
                    card,
                    text=(
                        "Originalbild konnte nicht "
                        f"angezeigt werden:\n{error}"
                    ),
                    foreground="red",
                    wraplength=450,
                )
                error_label.pack(
                    anchor="w"
                )

        self.image_inner.update_idletasks()
        self.image_canvas.configure(
            scrollregion=self.image_canvas.bbox("all")
        )

        self.image_canvas.yview_moveto(0)

    def _update_image_scrollregion(self, event=None):
        self.image_canvas.configure(
            scrollregion=self.image_canvas.bbox("all")
        )

    def _resize_image_inner(self, event):
        self.image_canvas.itemconfigure(
            self.image_window,
            width=event.width
        )

    # =========================================================
    # Hilfsfunktionen
    # =========================================================

    def _clear_question(self):
        self.question_entry.delete(
            "1.0",
            "end"
        )

    def _set_status(self, text):
        self.status_var.set(text)


def re_numbered(text):
    """
    Erkennt z.B.:
        1. Schritt ...
        2. Schritt ...
    """
    match = __import__("re").match(
        r"^(\d+\.\s+)(.*)$",
        text
    )

    if not match:
        return None

    return match.group(1), match.group(2)


def main():
    root = tk.Tk()

    app = ChemistryGUI(root)

    root.mainloop()


if __name__ == "__main__":
    main()
