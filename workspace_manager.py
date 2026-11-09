"""Persistente Verwaltung von Projekten, Chats, Dokumentfreigaben und Nachrichten.

Die Datenbank enthält nur Metadaten und Gesprächsinhalte. PDF-Dateien und
Embeddings bleiben in den bestehenden Projektordnern bzw. in Chroma.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _id() -> str:
    return uuid.uuid4().hex


def discover_pdf_library(
    project_root: str | Path,
    projects: list[dict[str, Any]],
    current_project_id: str,
) -> list[dict[str, Any]]:
    """Find existing PDFs that can be imported into another project.

    The function only discovers files; it never indexes or changes them. Selected
    files are copied into the destination project's private document/cache tree
    by the GUI before indexing, which preserves strict per-project retrieval.
    """
    root = Path(project_root).resolve()
    result: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_root(base: Path, kind: str, origin_key: str, origin_label: str, project_id: str = ""):
        if not base.is_dir():
            return
        try:
            paths = base.rglob("*")
            for pdf in paths:
                try:
                    if not pdf.is_file() or pdf.suffix.casefold() != ".pdf" or pdf.name.startswith("._"):
                        continue
                    resolved = str(pdf.resolve())
                    if resolved in seen:
                        continue
                    relative = pdf.relative_to(base).as_posix()
                    seen.add(resolved)
                    result.append({
                        "path": resolved,
                        "kind": kind,
                        "origin_key": origin_key,
                        "origin_label": origin_label,
                        "relative_path": relative,
                        "project_id": project_id,
                    })
                except (OSError, ValueError):
                    continue
        except OSError:
            return

    # Legacy/global sources are included because older installations stored PDFs
    # directly in these folders before per-project isolation was introduced.
    add_root(root / "documents", "document", "legacy-documents", "Alte globale Dokumente")
    add_root(root / "cache", "cache", "legacy-cache", "Alter globaler Klausur-Cache")

    names_by_safe_id: dict[str, str] = {}
    for project in projects:
        safe_id = "".join(ch for ch in str(project.get("id", "")) if ch.isalnum())[:32]
        if safe_id:
            names_by_safe_id[safe_id] = str(project.get("name") or safe_id)

    projects_root = root / "data" / "projects"
    if projects_root.is_dir():
        try:
            project_dirs = sorted((p for p in projects_root.iterdir() if p.is_dir()), key=lambda p: p.name.casefold())
        except OSError:
            project_dirs = []
        for project_dir in project_dirs:
            if project_dir.name == str(current_project_id):
                continue
            # config_for_project removes non-alphanumeric characters from IDs.
            clean_current = "".join(ch for ch in str(current_project_id) if ch.isalnum())[:32]
            if project_dir.name == clean_current:
                continue
            project_label = names_by_safe_id.get(project_dir.name, f"Archiviertes Projekt ({project_dir.name[:8]})")
            origin_key = f"project-{project_dir.name}"
            origin_label = f"Projekt: {project_label}"
            add_root(project_dir / "documents", "document", origin_key, origin_label, project_dir.name)
            add_root(project_dir / "cache", "cache", origin_key, origin_label, project_dir.name)

    result.sort(key=lambda item: (item["origin_label"].casefold(), item["kind"], item["relative_path"].casefold()))
    return result


class WorkspaceStore:
    def __init__(self, database_path: str | Path):
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self):
        """Yield a connection and ALWAYS close it (SQLite's `with` alone does not)."""
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS project_documents (
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    source_key TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY(project_id, source_key)
                );
                CREATE TABLE IF NOT EXISTS project_settings (
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    PRIMARY KEY(project_id, key)
                );
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    memory_summary TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chats_project ON chats(project_id, updated_at DESC);
                CREATE TABLE IF NOT EXISTS chat_documents (
                    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                    source_key TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY(chat_id, source_key)
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
                    content TEXT NOT NULL,
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id);
            """)

    def ensure_default_workspace(self) -> tuple[str, str]:
        projects = self.list_projects()
        if not projects:
            project_id = self.create_project("Allgemein")
        else:
            project_id = projects[0]["id"]
        chats = self.list_chats(project_id)
        chat_id = chats[0]["id"] if chats else self.create_chat(project_id, "Neuer Chat")
        return project_id, chat_id

    def create_project(self, name: str) -> str:
        name = str(name).strip()
        if not name:
            raise ValueError("Der Projektname darf nicht leer sein.")
        project_id, stamp = _id(), _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO projects(id,name,created_at,updated_at) VALUES(?,?,?,?)",
                (project_id, name, stamp, stamp),
            )
        return project_id

    def list_projects(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT p.id, p.name, p.created_at, p.updated_at,
                       (SELECT COUNT(*) FROM project_documents d WHERE d.project_id=p.id AND d.enabled=1) AS document_count,
                       (SELECT COUNT(*) FROM chats c WHERE c.project_id=p.id) AS chat_count
                FROM projects p ORDER BY p.updated_at DESC, p.name COLLATE NOCASE
            """).fetchall()
        return [dict(row) for row in rows]

    def rename_project(self, project_id: str, name: str):
        name = str(name).strip()
        if not name:
            raise ValueError("Der Projektname darf nicht leer sein.")
        with self._connect() as conn:
            conn.execute("UPDATE projects SET name=?,updated_at=? WHERE id=?", (name, _now(), project_id))

    def delete_project(self, project_id: str):
        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
            if count <= 1:
                raise ValueError("Das letzte Projekt kann nicht gelöscht werden.")
            conn.execute("DELETE FROM projects WHERE id=?", (project_id,))

    def create_chat(self, project_id: str, title: str = "Neuer Chat") -> str:
        chat_id, stamp = _id(), _now()
        with self._connect() as conn:
            exists = conn.execute("SELECT id FROM projects WHERE id=?", (project_id,)).fetchone()
            if not exists:
                raise ValueError("Das Projekt existiert nicht mehr.")
            conn.execute(
                "INSERT INTO chats(id,project_id,title,created_at,updated_at) VALUES(?,?,?,?,?)",
                (chat_id, project_id, str(title).strip() or "Neuer Chat", stamp, stamp),
            )
            conn.execute("""
                INSERT INTO chat_documents(chat_id,source_key,enabled)
                SELECT ?,source_key,enabled FROM project_documents
                WHERE project_id=? AND enabled=1
            """, (chat_id, project_id))
            conn.execute("UPDATE projects SET updated_at=? WHERE id=?", (stamp, project_id))
        return chat_id

    def list_chats(self, project_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT c.id,c.project_id,c.title,c.memory_summary,c.created_at,c.updated_at,
                       (SELECT COUNT(*) FROM messages m WHERE m.chat_id=c.id) AS message_count
                FROM chats c WHERE c.project_id=?
                ORDER BY c.updated_at DESC, c.created_at DESC
            """, (project_id,)).fetchall()
        return [dict(row) for row in rows]

    def rename_chat(self, chat_id: str, title: str):
        title = str(title).strip()
        if not title:
            raise ValueError("Der Chatname darf nicht leer sein.")
        with self._connect() as conn:
            conn.execute("UPDATE chats SET title=?,updated_at=? WHERE id=?", (title, _now(), chat_id))

    def delete_chat(self, chat_id: str):
        with self._connect() as conn:
            row = conn.execute("SELECT project_id FROM chats WHERE id=?", (chat_id,)).fetchone()
            if not row:
                return
            count = conn.execute("SELECT COUNT(*) FROM chats WHERE project_id=?", (row["project_id"],)).fetchone()[0]
            if count <= 1:
                raise ValueError("Das letzte Chatfenster eines Projekts kann nicht gelöscht werden.")
            conn.execute("DELETE FROM chats WHERE id=?", (chat_id,))
            conn.execute("UPDATE projects SET updated_at=? WHERE id=?", (_now(), row["project_id"]))

    def get_project_sources(self, project_id: str) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT source_key FROM project_documents WHERE project_id=? AND enabled=1", (project_id,)).fetchall()
        return {row[0] for row in rows}

    def get_chat_sources(self, chat_id: str) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT source_key FROM chat_documents WHERE chat_id=? AND enabled=1", (chat_id,)).fetchall()
        return {row[0] for row in rows}

    def set_project_sources(self, project_id: str, source_keys: set[str] | list[str]):
        source_keys = set(source_keys)
        with self._connect() as conn:
            conn.execute("DELETE FROM project_documents WHERE project_id=?", (project_id,))
            conn.executemany(
                "INSERT INTO project_documents(project_id,source_key,enabled) VALUES(?,?,1)",
                [(project_id, key) for key in sorted(source_keys)],
            )
            conn.execute("UPDATE projects SET updated_at=? WHERE id=?", (_now(), project_id))

    def set_chat_sources(self, chat_id: str, source_keys: set[str] | list[str]):
        source_keys = set(source_keys)
        with self._connect() as conn:
            conn.execute("DELETE FROM chat_documents WHERE chat_id=?", (chat_id,))
            conn.executemany(
                "INSERT INTO chat_documents(chat_id,source_key,enabled) VALUES(?,?,1)",
                [(chat_id, key) for key in sorted(source_keys)],
            )
            conn.execute("UPDATE chats SET updated_at=? WHERE id=?", (_now(), chat_id))

    def project_setting(self, project_id: str, key: str, default: str = "") -> str:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM project_settings WHERE project_id=? AND key=?",
                (project_id, key),
            ).fetchone()
        return str(row[0]) if row else default

    def set_project_setting(self, project_id: str, key: str, value: str):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO project_settings(project_id,key,value) VALUES(?,?,?) "
                "ON CONFLICT(project_id,key) DO UPDATE SET value=excluded.value",
                (project_id, key, str(value)),
            )

    def project_sources_initialized(self, project_id: str) -> bool:
        return self.project_setting(project_id, "sources_initialized", "0") == "1"

    def mark_project_sources_initialized(self, project_id: str):
        self.set_project_setting(project_id, "sources_initialized", "1")

    def add_message(self, chat_id: str, role: str, content: str, sources=None) -> int:
        if role not in ("user", "assistant", "system"):
            raise ValueError(f"Ungültige Nachrichtenrolle: {role}")
        stamp = _now()
        sources_json = json.dumps(sources or [], ensure_ascii=False)
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO messages(chat_id,role,content,sources_json,created_at) VALUES(?,?,?,?,?)",
                (chat_id, role, str(content), sources_json, stamp),
            )
            conn.execute("UPDATE chats SET updated_at=? WHERE id=?", (stamp, chat_id))
            return int(cur.lastrowid)

    def get_messages(self, chat_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id,chat_id,role,content,sources_json,created_at FROM messages WHERE chat_id=? ORDER BY id",
                (chat_id,),
            ).fetchall()
        output = []
        for row in rows:
            item = dict(row)
            try:
                item["sources"] = json.loads(item.pop("sources_json") or "[]")
            except json.JSONDecodeError:
                item["sources"] = []
                item.pop("sources_json", None)
            output.append(item)
        return output

    def get_memory_summary(self, chat_id: str) -> str:
        with self._connect() as conn:
            row = conn.execute("SELECT memory_summary FROM chats WHERE id=?", (chat_id,)).fetchone()
        return str(row[0] or "") if row else ""

    def set_memory_summary(self, chat_id: str, summary: str):
        with self._connect() as conn:
            conn.execute("UPDATE chats SET memory_summary=?,updated_at=? WHERE id=?", (str(summary or ""), _now(), chat_id))

    def search_messages(self, query: str, limit: int = 100) -> list[dict[str, Any]]:
        """Case-insensitive literal search over persisted chats and messages."""
        query = str(query or "").strip()
        if not query:
            return []
        needle = query.casefold()
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT m.id,m.chat_id,m.role,m.content,m.created_at,c.title AS chat_title,
                       p.id AS project_id,p.name AS project_name
                FROM messages m JOIN chats c ON c.id=m.chat_id
                JOIN projects p ON p.id=c.project_id
                ORDER BY m.id DESC
            """).fetchall()
        out=[]
        for row in rows:
            item=dict(row)
            if needle in str(item.get("content", "")).casefold() or needle in str(item.get("chat_title", "")).casefold():
                item["excerpt"] = str(item.get("content", "")).replace("\n", " ")[:420]
                out.append(item)
                if len(out) >= max(1, int(limit)):
                    break
        return out

    def message_count(self, chat_id: str) -> int:
        with self._connect() as conn:
            row=conn.execute("SELECT COUNT(*) FROM messages WHERE chat_id=?", (chat_id,)).fetchone()
        return int(row[0]) if row else 0

    def project_for_chat(self, chat_id: str) -> str | None:
        with self._connect() as conn:
            row=conn.execute("SELECT project_id FROM chats WHERE id=?", (chat_id,)).fetchone()
        return str(row[0]) if row else None
