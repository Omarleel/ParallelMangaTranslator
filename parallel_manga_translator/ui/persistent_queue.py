from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import List, Optional


class PersistentJobQueue:
    """Cola FIFO duradera respaldada por SQLite.

    Cada operación abre una conexión corta, por lo que la cola tolera reinicios y no
    comparte conexiones SQLite entre hilos.
    """

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.database_path), timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _initialize(self) -> None:
        with self._init_lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS job_queue (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 100,
                    enqueued_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_job_queue_next "
                "ON job_queue(status, priority, enqueued_at)"
            )

    def enqueue(self, job_id: str, *, priority: int = 100, preserve_time: bool = True) -> None:
        now = time.time()
        with self._connect() as conn:
            existing = conn.execute("SELECT enqueued_at FROM job_queue WHERE job_id = ?", (job_id,)).fetchone()
            enqueued_at = float(existing["enqueued_at"]) if existing and preserve_time else now
            conn.execute(
                """
                INSERT INTO job_queue(job_id, status, priority, enqueued_at, updated_at)
                VALUES (?, 'queued', ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status='queued', priority=excluded.priority,
                    enqueued_at=excluded.enqueued_at, updated_at=excluded.updated_at
                """,
                (job_id, int(priority), enqueued_at, now),
            )

    def remove(self, job_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM job_queue WHERE job_id = ?", (job_id,))

    def pop_next(self) -> Optional[str]:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT job_id FROM job_queue WHERE status='queued' "
                "ORDER BY priority ASC, enqueued_at ASC LIMIT 1"
            ).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None
            job_id = str(row["job_id"])
            conn.execute(
                "UPDATE job_queue SET status='processing', updated_at=? WHERE job_id=?",
                (time.time(), job_id),
            )
            conn.execute("COMMIT")
            return job_id


    def complete(self, job_id: str) -> None:
        """Elimina solo la reserva activa; conserva una reencolación concurrente."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM job_queue WHERE job_id = ? AND status = 'processing'",
                (job_id,),
            )

    def recover_processing(self) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE job_queue SET status='queued', updated_at=? WHERE status='processing'",
                (time.time(),),
            )
            return int(cursor.rowcount or 0)

    def queued_ids(self) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT job_id FROM job_queue WHERE status='queued' "
                "ORDER BY priority ASC, enqueued_at ASC"
            ).fetchall()
        return [str(row["job_id"]) for row in rows]

    def all_ids(self) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT job_id FROM job_queue").fetchall()
        return [str(row["job_id"]) for row in rows]

    def position(self, job_id: str) -> int | None:
        ids = self.queued_ids()
        try:
            return ids.index(job_id) + 1
        except ValueError:
            return None
