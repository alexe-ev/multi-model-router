from __future__ import annotations
import sqlite3
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from mmrouter.tracker.logger import Tracker, _CREATE_TABLE, _CREATE_FEEDBACK_TABLE
from mmrouter.tracker.analytics import CostAnalytics


def _open_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_CREATE_TABLE)
    conn.execute(_CREATE_FEEDBACK_TABLE)
    conn.commit()
    return conn


def create_app(db_path: str = "mmrouter.db") -> FastAPI:
    conn = _open_conn(db_path)
    # Build Tracker around our thread-safe connection
    tracker = Tracker.__new__(Tracker)
    tracker._db_path = str(db_path)
    tracker._conn = conn
    analytics = CostAnalytics(conn)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        try:
            conn.close()
        except Exception:
            pass

    app = FastAPI(title="mmrouter dashboard", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/stats")
    def get_stats():
        stats = tracker.get_stats()
        savings = analytics.savings_vs_baseline()
        return {**stats, "savings": savings}

    @app.get("/api/stats/daily")
    def get_daily():
        return analytics.daily_costs()

    @app.get("/api/stats/distribution")
    def get_distribution():
        return analytics.distribution()

    @app.get("/api/requests")
    def get_requests(
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
        model: str | None = None,
        complexity: str | None = None,
        category: str | None = None,
    ):
        query = "SELECT id, timestamp, prompt_hash, complexity, category, confidence, model, tokens_in, tokens_out, cost, latency_ms, fallback_used FROM requests"
        conditions = []
        params = []

        if model:
            conditions.append("model = ?")
            params.append(model)
        if complexity:
            conditions.append("complexity = ?")
            params.append(complexity)
        if category:
            conditions.append("category = ?")
            params.append(category)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        # Get total count
        count_query = "SELECT COUNT(*) FROM requests"
        if conditions:
            count_query += " WHERE " + " AND ".join(conditions)
        total = conn.execute(count_query, params).fetchone()[0]

        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = conn.execute(query, params).fetchall()
        items = [dict(row) for row in rows]

        return {"total": total, "items": items, "limit": limit, "offset": offset}

    @app.get("/api/stats/feedback")
    def get_feedback_stats():
        return tracker.get_feedback_stats()

    @app.get("/api/models")
    def get_models():
        rows = conn.execute("""
            SELECT model, COUNT(*) as count, SUM(cost) as total_cost,
                   AVG(latency_ms) as avg_latency_ms,
                   SUM(tokens_in) as total_tokens_in,
                   SUM(tokens_out) as total_tokens_out
            FROM requests GROUP BY model ORDER BY count DESC
        """).fetchall()
        return [dict(row) for row in rows]

    return app
