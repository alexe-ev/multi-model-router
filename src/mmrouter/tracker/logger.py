"""SQLite tracker: log every routed request, query stats."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
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
    cascade_attempts INTEGER NOT NULL DEFAULT 1,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    cache_creation_tokens INTEGER NOT NULL DEFAULT 0
)
"""

_CREATE_FEEDBACK_TABLE = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id INTEGER NOT NULL REFERENCES requests(id),
    rating INTEGER NOT NULL CHECK (rating IN (-1, 1)),
    timestamp TEXT NOT NULL,
    UNIQUE(request_id)
)
"""

_MIGRATIONS = [
    "ALTER TABLE requests ADD COLUMN cascade_used INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE requests ADD COLUMN cascade_attempts INTEGER NOT NULL DEFAULT 1",
    "ALTER TABLE requests ADD COLUMN cache_read_tokens INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE requests ADD COLUMN cache_creation_tokens INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE requests ADD COLUMN experiment_id INTEGER",
    "ALTER TABLE requests ADD COLUMN variant TEXT",
]

_INSERT = """
INSERT INTO requests (
    timestamp, prompt_hash, complexity, category, confidence,
    model, tokens_in, tokens_out, cost, latency_ms, fallback_used,
    cascade_used, cascade_attempts, cache_read_tokens, cache_creation_tokens,
    experiment_id, variant
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class Tracker:
    """SQLite-based request logger and stats provider."""

    def __init__(self, db_path: str | Path = _DEFAULT_DB):
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_TABLE)
        self._conn.execute(_CREATE_FEEDBACK_TABLE)
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

    def log(self, entry: RequestLog) -> int:
        """Log a request and return the inserted request_id."""
        cur = self._conn.execute(_INSERT, (
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
            entry.completion.cache_read_tokens,
            entry.completion.cache_creation_tokens,
            entry.experiment_id,
            entry.variant,
        ))
        self._conn.commit()
        return cur.lastrowid

    def submit_feedback(self, request_id: int, rating: int) -> None:
        """Submit feedback for a request. Overwrites if already exists.

        Args:
            request_id: ID from the requests table.
            rating: 1 (thumbs up) or -1 (thumbs down).

        Raises:
            ValueError: If rating is not 1 or -1, or request_id doesn't exist.
        """
        if rating not in (1, -1):
            raise ValueError(f"Rating must be 1 or -1, got {rating}")

        # Verify request exists
        cur = self._conn.execute(
            "SELECT id FROM requests WHERE id = ?", (request_id,)
        )
        if not cur.fetchone():
            raise ValueError(f"Request {request_id} not found")

        ts = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT INTO feedback (request_id, rating, timestamp)
               VALUES (?, ?, ?)
               ON CONFLICT(request_id) DO UPDATE SET rating = excluded.rating, timestamp = excluded.timestamp""",
            (request_id, rating, ts),
        )
        self._conn.commit()

    def get_feedback_stats(self) -> dict:
        """Aggregated feedback stats: per (model, complexity, category) bucket."""
        cur = self._conn.execute("""
            SELECT
                r.model,
                r.complexity,
                r.category,
                COUNT(*) as total,
                SUM(CASE WHEN f.rating = 1 THEN 1 ELSE 0 END) as positive,
                SUM(CASE WHEN f.rating = -1 THEN 1 ELSE 0 END) as negative
            FROM feedback f
            JOIN requests r ON f.request_id = r.id
            GROUP BY r.model, r.complexity, r.category
        """)
        rows = cur.fetchall()

        total_feedback = self._conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        total_requests = self._conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]

        buckets = []
        for row in rows:
            total = row["total"]
            positive = row["positive"]
            buckets.append({
                "model": row["model"],
                "complexity": row["complexity"],
                "category": row["category"],
                "total": total,
                "positive": positive,
                "negative": row["negative"],
                "success_rate": round(positive / total, 4) if total > 0 else 0.0,
            })

        return {
            "total_feedback": total_feedback,
            "total_requests": total_requests,
            "feedback_rate": round(total_feedback / total_requests, 4) if total_requests > 0 else 0.0,
            "buckets": buckets,
        }

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
