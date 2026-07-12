"""
SQLite-backed chat sessions for the demo.

The browser tab can be closed/refreshed at any time — conversation history
lives in a SQLite file on the server, keyed by a session id (a short opaque
token the browser keeps in localStorage). This is separate from the GPU KV
cache: sessions are pure text history; "reset" drops the live KV cache +
working history on the GPU, but the saved session rows stay so you can re-open
the chat later.

Schema:
  sessions(id TEXT PRIMARY KEY, title TEXT, model_id TEXT, created_at REAL, updated_at REAL)
  messages(id INTEGER PRIMARY KEY AUTOINCREMENT,
           session_id TEXT, role TEXT, content TEXT, ts REAL,
           FOREIGN KEY(session_id) REFERENCES sessions(id))
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path


class SessionStore:
    """Thread-safe wrapper over a SQLite file (one writer lock)."""

    def __init__(self, db_path: str | os.PathLike):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _conn(self):
        # check_same_thread=False + the lock make it safe across request threads.
        c = sqlite3.connect(self.db_path, check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    def _init_schema(self):
        with self._lock:
            c = self._conn()
            c.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    model_id TEXT,
                    created_at REAL,
                    updated_at REAL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    ts REAL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(session_id, id);
            """)
            c.commit()
            c.close()

    # ------------------------------------------------------------- sessions

    def create_session(self, model_id: str | None = None, title: str | None = None) -> dict:
        sid = uuid.uuid4().hex[:16]
        now = time.time()
        with self._lock:
            c = self._conn()
            c.execute(
                "INSERT INTO sessions(id, title, model_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (sid, title or "New chat", model_id, now, now),
            )
            c.commit()
            c.close()
        return {"id": sid, "title": title or "New chat", "model_id": model_id,
                "created_at": now, "messages": []}

    def list_sessions(self) -> list[dict]:
        with self._lock:
            c = self._conn()
            rows = c.execute(
                "SELECT id, title, model_id, created_at, updated_at "
                "FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
            c.close()
        return [dict(r) for r in rows]

    def get_session(self, session_id: str) -> dict | None:
        with self._lock:
            c = self._conn()
            s = c.execute(
                "SELECT id, title, model_id, created_at, updated_at "
                "FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
            if s is None:
                c.close()
                return None
            msgs = c.execute(
                "SELECT role, content FROM messages WHERE session_id=? ORDER BY id",
                (session_id,)
            ).fetchall()
            c.close()
        d = dict(s)
        d["messages"] = [dict(m) for m in msgs]
        return d

    def rename_session(self, session_id: str, title: str) -> bool:
        with self._lock:
            c = self._conn()
            cur = c.execute(
                "UPDATE sessions SET title=?, updated_at=? WHERE id=?",
                (title, time.time(), session_id),
            )
            c.commit()
            c.close()
            return cur.rowcount > 0

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            c = self._conn()
            c.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
            cur = c.execute("DELETE FROM sessions WHERE id=?", (session_id,))
            c.commit()
            c.close()
            return cur.rowcount > 0

    # ------------------------------------------------------------ messages

    def append_message(self, session_id: str, role: str, content: str) -> dict:
        now = time.time()
        with self._lock:
            c = self._conn()
            cur = c.execute(
                "INSERT INTO messages(session_id, role, content, ts) VALUES (?, ?, ?, ?)",
                (session_id, role, content, now),
            )
            # Auto-title the session from the first user message.
            if role == "user":
                row = c.execute(
                    "SELECT COUNT(*) AS n FROM messages WHERE session_id=? AND role='user'",
                    (session_id,)
                ).fetchone()
                if row["n"] == 1:
                    c.execute(
                        "UPDATE sessions SET title=?, updated_at=? WHERE id=?",
                        (content[:60], now, session_id),
                    )
            c.execute(
                "UPDATE sessions SET updated_at=? WHERE id=?", (now, session_id)
            )
            c.commit()
            msg_id = cur.lastrowid
            c.close()
        return {"id": msg_id, "session_id": session_id, "role": role,
                "content": content, "ts": now}
