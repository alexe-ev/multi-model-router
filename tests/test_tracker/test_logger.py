"""Tests for SQLite tracker."""

import pytest

from mmrouter.models import (
    ClassificationResult,
    CompletionResult,
    RequestLog,
)
from mmrouter.tracker.logger import Tracker


def _make_log(model="claude-haiku", cost=0.001, fallback=False):
    return RequestLog(
        prompt_hash=RequestLog.hash_prompt("test"),
        classification=ClassificationResult(
            complexity="simple", category="factual", confidence=0.9
        ),
        model_used=model,
        completion=CompletionResult(
            content="ok",
            model=model,
            tokens_in=10,
            tokens_out=5,
            cost=cost,
            latency_ms=150.0,
        ),
        fallback_used=fallback,
    )


class TestTracker:
    def test_log_and_stats(self, tmp_path):
        tracker = Tracker(tmp_path / "test.db")
        tracker.log(_make_log())
        stats = tracker.get_stats()

        assert stats["total_requests"] == 1
        assert stats["total_cost"] == 0.001
        assert stats["avg_latency_ms"] == 150.0
        assert stats["model_distribution"]["claude-haiku"]["count"] == 1
        tracker.close()

    def test_multiple_models(self, tmp_path):
        tracker = Tracker(tmp_path / "test.db")
        tracker.log(_make_log(model="claude-haiku", cost=0.001))
        tracker.log(_make_log(model="claude-haiku", cost=0.001))
        tracker.log(_make_log(model="claude-sonnet", cost=0.01))

        stats = tracker.get_stats()
        assert stats["total_requests"] == 3
        assert stats["model_distribution"]["claude-haiku"]["count"] == 2
        assert stats["model_distribution"]["claude-sonnet"]["count"] == 1
        tracker.close()

    def test_fallback_count(self, tmp_path):
        tracker = Tracker(tmp_path / "test.db")
        tracker.log(_make_log(fallback=False))
        tracker.log(_make_log(fallback=True))
        tracker.log(_make_log(fallback=True))

        stats = tracker.get_stats()
        assert stats["fallback_count"] == 2
        tracker.close()

    def test_empty_stats(self, tmp_path):
        tracker = Tracker(tmp_path / "test.db")
        stats = tracker.get_stats()

        assert stats["total_requests"] == 0
        assert stats["total_cost"] == 0
        assert stats["model_distribution"] == {}
        tracker.close()

    def test_wal_mode(self, tmp_path):
        tracker = Tracker(tmp_path / "test.db")
        cur = tracker._conn.execute("PRAGMA journal_mode")
        mode = cur.fetchone()[0]
        assert mode == "wal"
        tracker.close()

    def test_db_created_automatically(self, tmp_path):
        db_path = tmp_path / "new.db"
        assert not db_path.exists()
        tracker = Tracker(db_path)
        assert db_path.exists()
        tracker.close()
