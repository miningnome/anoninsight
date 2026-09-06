"""SQLite access and schema migrations.

Uses the Python standard library `sqlite3` plus numbered plain-SQL
migrations - the same pattern the sibling product in this repo uses
(server/migrations/). All SQL lives behind Repository, so moving to
PostgreSQL later (see ARCHITECTURE.md section 2.3) is a change confined to
this package.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._connection: sqlite3.Connection | None = None

    def connect(self) -> None:
        if self._connection is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path), check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=15000")
        self._connection = connection
        self._apply_migrations()

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("database is not connected")
        return self._connection

    def read(self, sql: str, parameters: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self.connection.execute(sql, parameters).fetchall()

    def write(self, sql: str, parameters: tuple = ()) -> None:
        with self._lock:
            with self.connection:
                self.connection.execute(sql, parameters)

    def write_many(self, statements: list[tuple[str, tuple]]) -> None:
        """Applies several statements in one transaction, so a partially
        written person/embedding set can never be committed."""
        with self._lock:
            with self.connection:
                for sql, parameters in statements:
                    self.connection.execute(sql, parameters)

    def _apply_migrations(self) -> None:
        connection = self.connection
        with self._lock:
            # executescript() commits any pending transaction, so migrations
            # are committed one by one instead of inside a `with connection`
            # block that would fight it.
            connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                " name TEXT PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            connection.commit()
            applied = {
                row["name"] for row in connection.execute("SELECT name FROM schema_migrations")
            }
            for migration in sorted(MIGRATIONS_DIR.glob("*.sql")):
                if migration.name in applied:
                    continue
                connection.executescript(migration.read_text(encoding="utf-8"))
                connection.execute(
                    "INSERT INTO schema_migrations (name) VALUES (?)", (migration.name,)
                )
                connection.commit()
                logger.info("applied migration %s", migration.name)
