"""Tests for cost analytics."""

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from mmrouter.tracker.analytics import MODEL_PRICING, CostAnalytics
from mmrouter.tracker.logger import Tracker

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
    fallback_used INTEGER NOT NULL DEFAULT 0
)
"""

_INSERT = """
INSERT INTO requests (
    timestamp, prompt_hash, complexity, category, confidence,
    model, tokens_in, tokens_out, cost, latency_ms, fallback_used
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(_CREATE_TABLE)
    conn.commit()
    return conn


def _insert_requests(conn: sqlite3.Connection, requests: list[dict]) -> None:
    for r in requests:
        conn.execute(_INSERT, (
            r.get("timestamp", datetime.now(timezone.utc).isoformat()),
            r.get("prompt_hash", "abc123"),
            r.get("complexity", "simple"),
            r.get("category", "factual"),
            r.get("confidence", 0.9),
            r["model"],
            r["tokens_in"],
            r["tokens_out"],
            r["cost"],
            r.get("latency_ms", 100.0),
            r.get("fallback_used", 0),
        ))
    conn.commit()


class TestDailyCosts:
    def test_daily_costs_groups_by_date_and_model(self):
        conn = _make_conn()
        today = datetime.now(timezone.utc).date().isoformat()
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()

        _insert_requests(conn, [
            {"model": "claude-haiku-4-5-20251001", "tokens_in": 100, "tokens_out": 50, "cost": 0.001, "timestamp": f"{today}T10:00:00"},
            {"model": "claude-haiku-4-5-20251001", "tokens_in": 100, "tokens_out": 50, "cost": 0.002, "timestamp": f"{today}T11:00:00"},
            {"model": "claude-sonnet-4-6", "tokens_in": 200, "tokens_out": 100, "cost": 0.01, "timestamp": f"{today}T12:00:00"},
            {"model": "claude-haiku-4-5-20251001", "tokens_in": 100, "tokens_out": 50, "cost": 0.001, "timestamp": f"{yesterday}T10:00:00"},
        ])

        analytics = CostAnalytics(conn)
        result = analytics.daily_costs()

        assert len(result) == 3

        # Today first (DESC), then ordered by cost DESC within day
        assert result[0]["date"] == today
        assert result[0]["model"] == "claude-sonnet-4-6"
        assert result[0]["total_cost"] == 0.01

        assert result[1]["date"] == today
        assert result[1]["model"] == "claude-haiku-4-5-20251001"
        assert result[1]["request_count"] == 2
        assert result[1]["total_cost"] == pytest.approx(0.003)

        assert result[2]["date"] == yesterday
        assert result[2]["model"] == "claude-haiku-4-5-20251001"

    def test_daily_costs_empty_db(self):
        conn = _make_conn()
        analytics = CostAnalytics(conn)
        assert analytics.daily_costs() == []


class TestSavingsVsBaseline:
    def test_savings_vs_baseline(self):
        conn = _make_conn()
        # Haiku request: cheap model, baseline would cost more
        # tokens_in=1000, tokens_out=500
        # Haiku actual cost: 0.001
        # Sonnet baseline: 1000 * 3.0/1M + 500 * 15.0/1M = 0.003 + 0.0075 = 0.0105
        _insert_requests(conn, [
            {"model": "claude-haiku-4-5-20251001", "tokens_in": 1000, "tokens_out": 500, "cost": 0.001},
            {"model": "claude-opus-4-6", "tokens_in": 2000, "tokens_out": 1000, "cost": 0.1},
        ])

        analytics = CostAnalytics(conn)
        result = analytics.savings_vs_baseline()

        # Actual: 0.001 + 0.1 = 0.101
        assert result["actual_cost"] == pytest.approx(0.101, abs=1e-6)

        # Baseline (sonnet): (1000*3/1M + 500*15/1M) + (2000*3/1M + 1000*15/1M)
        # = 0.0105 + 0.021 = 0.0315
        assert result["baseline_cost"] == pytest.approx(0.0315, abs=1e-6)

        # Savings = 0.0315 - 0.101 = -0.0695 (negative: routing to opus cost more)
        assert result["savings"] == pytest.approx(-0.0695, abs=1e-4)
        assert result["savings_pct"] < 0

    def test_savings_empty_db(self):
        conn = _make_conn()
        analytics = CostAnalytics(conn)
        result = analytics.savings_vs_baseline()

        assert result["actual_cost"] == 0.0
        assert result["baseline_cost"] == 0.0
        assert result["savings"] == 0.0
        assert result["savings_pct"] == 0.0

    def test_savings_unknown_model_uses_actual_cost(self):
        """If baseline model is not in pricing table, returns zeros."""
        conn = _make_conn()
        _insert_requests(conn, [
            {"model": "claude-haiku-4-5-20251001", "tokens_in": 1000, "tokens_out": 500, "cost": 0.001},
        ])

        analytics = CostAnalytics(conn)
        result = analytics.savings_vs_baseline(baseline_model="nonexistent-model-xyz")

        assert result["actual_cost"] == 0.0
        assert result["baseline_cost"] == 0.0
        assert result["savings"] == 0.0
        assert result["savings_pct"] == 0.0


class TestDistribution:
    def test_distribution_by_complexity_and_category(self):
        conn = _make_conn()
        _insert_requests(conn, [
            {"model": "m1", "tokens_in": 100, "tokens_out": 50, "cost": 0.001, "complexity": "simple", "category": "factual"},
            {"model": "m2", "tokens_in": 200, "tokens_out": 100, "cost": 0.01, "complexity": "simple", "category": "code"},
            {"model": "m3", "tokens_in": 300, "tokens_out": 150, "cost": 0.05, "complexity": "complex", "category": "reasoning"},
        ])

        analytics = CostAnalytics(conn)
        result = analytics.distribution()

        by_c = result["by_complexity"]
        assert by_c["simple"]["count"] == 2
        assert by_c["simple"]["cost"] == pytest.approx(0.011)
        assert by_c["complex"]["count"] == 1
        assert by_c["complex"]["cost"] == pytest.approx(0.05)

        by_cat = result["by_category"]
        assert by_cat["factual"]["count"] == 1
        assert by_cat["code"]["count"] == 1
        assert by_cat["reasoning"]["count"] == 1

    def test_distribution_empty_db(self):
        conn = _make_conn()
        analytics = CostAnalytics(conn)
        result = analytics.distribution()

        assert result["by_complexity"] == {}
        assert result["by_category"] == {}


class TestModelPricing:
    def test_model_pricing_has_expected_models(self):
        expected = [
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
            "gpt-4o",
            "gpt-4o-mini",
            "gemini-2.0-flash",
            "gemini-2.5-pro",
        ]
        for model in expected:
            assert model in MODEL_PRICING
            assert "input" in MODEL_PRICING[model]
            assert "output" in MODEL_PRICING[model]


class TestCascadeSavings:
    def test_cascade_savings_with_cascade_requests(self, tmp_path):
        tracker = Tracker(tmp_path / "test.db")
        conn = tracker.connection
        from mmrouter.tracker.logger import _INSERT as tracker_insert
        conn.execute(tracker_insert, (
            "2026-04-01T10:00:00", "abc123", "simple", "factual", 0.9,
            "claude-haiku-4-5-20251001", 100, 50, 0.001, 100.0, 0, 1, 2, 0, 0, None, None,
        ))
        conn.execute(tracker_insert, (
            "2026-04-01T11:00:00", "def456", "medium", "reasoning", 0.8,
            "claude-sonnet-4-6", 200, 100, 0.01, 200.0, 0, 1, 3, 0, 0, None, None,
        ))
        conn.execute(tracker_insert, (
            "2026-04-01T12:00:00", "ghi789", "simple", "factual", 0.9,
            "claude-haiku-4-5-20251001", 50, 25, 0.0005, 50.0, 0, 0, 1, 0, 0, None, None,
        ))
        conn.commit()

        analytics = CostAnalytics(conn)
        result = analytics.cascade_savings()

        assert result["cascade_requests"] == 2
        assert result["cascade_actual_cost"] == pytest.approx(0.011, abs=1e-6)
        assert result["cascade_attempts_total"] == 5
        assert result["avg_attempts"] == 2.5
        tracker.close()

    def test_cascade_savings_no_cascade_requests(self, tmp_path):
        tracker = Tracker(tmp_path / "test.db")
        analytics = CostAnalytics(tracker.connection)
        result = analytics.cascade_savings()

        assert result["cascade_requests"] == 0
        assert result["cascade_actual_cost"] == 0.0
        assert result["cascade_attempts_total"] == 0
        assert result["avg_attempts"] == 0.0
        tracker.close()


class TestTrackerConnectionProperty:
    def test_tracker_exposes_connection(self, tmp_path):
        tracker = Tracker(tmp_path / "test.db")
        assert tracker.connection is not None
        assert isinstance(tracker.connection, sqlite3.Connection)
        tracker.close()
