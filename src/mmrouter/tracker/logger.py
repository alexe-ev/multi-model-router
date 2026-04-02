"""SQLite tracker: log every routed request, query stats."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from mmrouter.models import RequestLog

_DEFAULT_DB = "mmrouter.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    complexity TEXT NOT NULL,
    category TEXT NOT NULL,
    confidence REAL NOT NULL,
    model TEXT NOT NULL,
    tokens_in INTEGER NOT NULL,
    tokens_out INTEGER NOT NULL,
    cost REAL NOT NULL,
    latency_ms REAL NOT NULL,
    fallback_used INTEGER NOT NULL DEFAULT 0,
    cascade_used INTEGER NOT NULL DEFAULT 0,
    cascade_attempts INTEGER NOT NULL DEFAULT 1
)
"""

_MIGRATIONS = [
    "ALTER TABLE requests ADD COLUMN cascade_used INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE requests ADD COLUMN cascade_attempts INTEGER NOT NULL DEFAULT 1",
]

_INSERT = """
INSERT INTO requests (
    timestamp, prompt_hash, complexity, category, confidence,
    model, tokens_in, tokens_out, cost, latency_ms, fallback_used,
    cascade_used, cascade_attempts
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class Tracker:
    """SQLite-based request logger and stats provider."""

    def __init__(self, db_path: str | Path = _DEFAULT_DB):
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_TABLE)
        self._run_migrations()
        self._conn.commit()

    def _run_migrations(self) -> None:
        """Add columns that may be missing from older databases."""
        cur = self._conn.execute("PRAGMA table_info(requests)")
        existing_columns = {row[1] for row in cur.fetchall()}
        for stmt in _MIGRATIONS:
            # Extract column name from ALTER TABLE ... ADD COLUMN <name> ...
            col_name = stmt.split("ADD COLUMN")[1].strip().split()[0]
            if col_name not in existing_columns:
                self._conn.execute(stmt)

    def log(self, entry: RequestLog) -> None:
        self._conn.execute(_INSERT, (
            entry.timestamp.isoformat(),
            entry.prompt_hash,
            entry.classification.complexity.value,
            entry.classification.category.value,
            entry.classification.confidence,
            entry.model_used,
            entry.completion.tokens_in,
            entry.completion.tokens_out,
            entry.completion.cost,
            entry.completion.latency_ms,
            int(entry.fallback_used),
            int(entry.cascade_used),
            entry.cascade_attempts,
        ))
        self._conn.commit()

    def get_stats(self) -> dict:
        cur = self._conn.execute("""
            SELECT
                COUNT(*) as total_requests,
                COALESCE(SUM(cost), 0) as total_cost,
                COALESCE(AVG(latency_ms), 0) as avg_latency_ms,
                COALESCE(SUM(tokens_in), 0) as total_tokens_in,
                COALESCE(SUM(tokens_out), 0) as total_tokens_out,
                COALESCE(SUM(fallback_used), 0) as fallback_count
            FROM requests
        """)
        row = cur.fetchone()

        model_cur = self._conn.execute("""
            SELECT model, COUNT(*) as count, SUM(cost) as cost
            FROM requests
            GROUP BY model
            ORDER BY count DESC
        """)
        model_distribution = {
            r["model"]: {"count": r["count"], "cost": r["cost"]}
            for r in model_cur.fetchall()
        }

        return {
            "total_requests": row["total_requests"],
            "total_cost": round(row["total_cost"], 6),
            "avg_latency_ms": round(row["avg_latency_ms"], 1),
            "total_tokens_in": row["total_tokens_in"],
            "total_tokens_out": row["total_tokens_out"],
            "fallback_count": row["fallback_count"],
            "model_distribution": model_distribution,
        }

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    def close(self) -> None:
        self._conn.close()
