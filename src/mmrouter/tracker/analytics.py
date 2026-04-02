"""Cost analytics: daily costs, savings estimation, distribution breakdowns."""

from __future__ import annotations

import sqlite3

MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-6": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
    "gpt-4o": {"input": 2.50, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.0},
}


class CostAnalytics:
    """Analyzes routing costs from the tracker database."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def daily_costs(self) -> list[dict]:
        cur = self._conn.execute("""
            SELECT
                date(timestamp) as day,
                model,
                COUNT(*) as request_count,
                SUM(cost) as total_cost
            FROM requests
            GROUP BY date(timestamp), model
            ORDER BY date(timestamp) DESC, SUM(cost) DESC
        """)
        return [
            {
                "date": row[0],
                "model": row[1],
                "request_count": row[2],
                "total_cost": row[3],
            }
            for row in cur.fetchall()
        ]

    def savings_vs_baseline(self, baseline_model: str = "claude-sonnet-4-6") -> dict:
        cur = self._conn.execute(
            "SELECT tokens_in, tokens_out, cost, model FROM requests"
        )
        rows = cur.fetchall()

        if not rows:
            return {
                "actual_cost": 0.0,
                "baseline_cost": 0.0,
                "savings": 0.0,
                "savings_pct": 0.0,
            }

        pricing = MODEL_PRICING.get(baseline_model)
        if not pricing:
            return {
                "actual_cost": 0.0,
                "baseline_cost": 0.0,
                "savings": 0.0,
                "savings_pct": 0.0,
            }

        actual_cost = 0.0
        baseline_cost = 0.0

        for row in rows:
            tokens_in, tokens_out, cost, _model = row
            actual_cost += cost
            baseline_cost += (
                tokens_in * pricing["input"] / 1_000_000
                + tokens_out * pricing["output"] / 1_000_000
            )

        savings = baseline_cost - actual_cost
        savings_pct = (savings / baseline_cost * 100) if baseline_cost > 0 else 0.0

        return {
            "actual_cost": round(actual_cost, 6),
            "baseline_cost": round(baseline_cost, 6),
            "savings": round(savings, 6),
            "savings_pct": round(savings_pct, 2),
        }

    def distribution(self) -> dict:
        complexity_cur = self._conn.execute("""
            SELECT complexity, COUNT(*) as count, SUM(cost) as cost
            FROM requests
            GROUP BY complexity
            ORDER BY count DESC
        """)
        by_complexity = {
            row[0]: {"count": row[1], "cost": row[2]}
            for row in complexity_cur.fetchall()
        }

        category_cur = self._conn.execute("""
            SELECT category, COUNT(*) as count, SUM(cost) as cost
            FROM requests
            GROUP BY category
            ORDER BY count DESC
        """)
        by_category = {
            row[0]: {"count": row[1], "cost": row[2]}
            for row in category_cur.fetchall()
        }

        return {
            "by_complexity": by_complexity,
            "by_category": by_category,
        }

    def cache_stats(self) -> dict:
        """Cache hit rate: cache_read_tokens / total input tokens over time."""
        cur = self._conn.execute(
            "SELECT name FROM pragma_table_info('requests') WHERE name = 'cache_read_tokens'"
        )
        if not cur.fetchone():
            return {
                "total_cache_read_tokens": 0,
                "total_cache_creation_tokens": 0,
                "total_input_tokens": 0,
                "cache_hit_rate": 0.0,
            }

        cur = self._conn.execute("""
            SELECT
                COALESCE(SUM(cache_read_tokens), 0) as total_cache_read,
                COALESCE(SUM(cache_creation_tokens), 0) as total_cache_creation,
                COALESCE(SUM(tokens_in), 0) as total_input
            FROM requests
        """)
        row = cur.fetchone()
        total_read = row[0]
        total_creation = row[1]
        total_input = row[2]

        hit_rate = (total_read / total_input * 100) if total_input > 0 else 0.0

        return {
            "total_cache_read_tokens": total_read,
            "total_cache_creation_tokens": total_creation,
            "total_input_tokens": total_input,
            "cache_hit_rate": round(hit_rate, 2),
        }

    def feedback_stats(self) -> dict:
        """Feedback analytics: success rates per model, per bucket."""
        # Check if feedback table exists
        cur = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'"
        )
        if not cur.fetchone():
            return {"total_feedback": 0, "by_model": {}, "by_bucket": {}}

        total = self._conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]

        model_cur = self._conn.execute("""
            SELECT r.model,
                   COUNT(*) as total,
                   SUM(CASE WHEN f.rating = 1 THEN 1 ELSE 0 END) as positive
            FROM feedback f
            JOIN requests r ON f.request_id = r.id
            GROUP BY r.model
        """)
        by_model = {}
        for row in model_cur.fetchall():
            model, cnt, pos = row[0], row[1], row[2]
            by_model[model] = {
                "total": cnt,
                "positive": pos,
                "negative": cnt - pos,
                "success_rate": round(pos / cnt, 4) if cnt > 0 else 0.0,
            }

        bucket_cur = self._conn.execute("""
            SELECT r.complexity, r.category,
                   COUNT(*) as total,
                   SUM(CASE WHEN f.rating = 1 THEN 1 ELSE 0 END) as positive
            FROM feedback f
            JOIN requests r ON f.request_id = r.id
            GROUP BY r.complexity, r.category
        """)
        by_bucket = {}
        for row in bucket_cur.fetchall():
            key = f"{row[0]}/{row[1]}"
            cnt, pos = row[2], row[3]
            by_bucket[key] = {
                "total": cnt,
                "positive": pos,
                "negative": cnt - pos,
                "success_rate": round(pos / cnt, 4) if cnt > 0 else 0.0,
            }

        return {
            "total_feedback": total,
            "by_model": by_model,
            "by_bucket": by_bucket,
        }

    def cascade_savings(self) -> dict:
        """Compare actual cost of cascade requests vs what the originally-classified model would have cost."""
        cur = self._conn.execute(
            "SELECT cascade_used, cascade_attempts, cost, tokens_in, tokens_out, model "
            "FROM requests WHERE cascade_used = 1"
        )
        rows = cur.fetchall()

        if not rows:
            return {
                "cascade_requests": 0,
                "cascade_actual_cost": 0.0,
                "cascade_attempts_total": 0,
                "avg_attempts": 0.0,
            }

        total_cost = 0.0
        total_attempts = 0
        count = len(rows)

        for row in rows:
            total_cost += row[2]
            total_attempts += row[1]

        return {
            "cascade_requests": count,
            "cascade_actual_cost": round(total_cost, 6),
            "cascade_attempts_total": total_attempts,
            "avg_attempts": round(total_attempts / count, 2) if count > 0 else 0.0,
        }
