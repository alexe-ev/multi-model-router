"""Tests for feedback table CRUD and stats in tracker."""

import pytest

from mmrouter.models import (
    ClassificationResult,
    CompletionResult,
    RequestLog,
)
from mmrouter.tracker.logger import Tracker


def _make_log(model="claude-haiku", complexity="simple", category="factual", cost=0.001):
    return RequestLog(
        prompt_hash=RequestLog.hash_prompt("test"),
        classification=ClassificationResult(
            complexity=complexity, category=category, confidence=0.9
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
    )


class TestFeedbackTable:
    def test_feedback_table_created(self, tmp_path):
        tracker = Tracker(tmp_path / "test.db")
        cur = tracker.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'"
        )
        assert cur.fetchone() is not None
        tracker.close()

    def test_submit_feedback_positive(self, tmp_path):
        tracker = Tracker(tmp_path / "test.db")
        request_id = tracker.log(_make_log())
        tracker.submit_feedback(request_id, 1)

        cur = tracker.connection.execute(
            "SELECT rating FROM feedback WHERE request_id = ?", (request_id,)
        )
        assert cur.fetchone()["rating"] == 1
        tracker.close()

    def test_submit_feedback_negative(self, tmp_path):
        tracker = Tracker(tmp_path / "test.db")
        request_id = tracker.log(_make_log())
        tracker.submit_feedback(request_id, -1)

        cur = tracker.connection.execute(
            "SELECT rating FROM feedback WHERE request_id = ?", (request_id,)
        )
        assert cur.fetchone()["rating"] == -1
        tracker.close()

    def test_submit_feedback_overwrites(self, tmp_path):
        """Second feedback for same request overwrites, not duplicates."""
        tracker = Tracker(tmp_path / "test.db")
        request_id = tracker.log(_make_log())
        tracker.submit_feedback(request_id, 1)
        tracker.submit_feedback(request_id, -1)

        cur = tracker.connection.execute(
            "SELECT COUNT(*) as cnt FROM feedback WHERE request_id = ?",
            (request_id,),
        )
        assert cur.fetchone()["cnt"] == 1

        cur = tracker.connection.execute(
            "SELECT rating FROM feedback WHERE request_id = ?", (request_id,)
        )
        assert cur.fetchone()["rating"] == -1
        tracker.close()

    def test_submit_feedback_invalid_rating(self, tmp_path):
        tracker = Tracker(tmp_path / "test.db")
        request_id = tracker.log(_make_log())
        with pytest.raises(ValueError, match="Rating must be 1 or -1"):
            tracker.submit_feedback(request_id, 0)
        tracker.close()

    def test_submit_feedback_nonexistent_request(self, tmp_path):
        tracker = Tracker(tmp_path / "test.db")
        with pytest.raises(ValueError, match="not found"):
            tracker.submit_feedback(9999, 1)
        tracker.close()

    def test_log_returns_request_id(self, tmp_path):
        tracker = Tracker(tmp_path / "test.db")
        rid1 = tracker.log(_make_log())
        rid2 = tracker.log(_make_log())
        assert isinstance(rid1, int)
        assert rid2 == rid1 + 1
        tracker.close()


class TestFeedbackStats:
    def test_empty_feedback_stats(self, tmp_path):
        tracker = Tracker(tmp_path / "test.db")
        stats = tracker.get_feedback_stats()
        assert stats["total_feedback"] == 0
        assert stats["feedback_rate"] == 0.0
        assert stats["buckets"] == []
        tracker.close()

    def test_feedback_stats_aggregation(self, tmp_path):
        tracker = Tracker(tmp_path / "test.db")

        # Log requests for two models
        for _ in range(3):
            rid = tracker.log(_make_log(model="model-a"))
            tracker.submit_feedback(rid, 1)

        rid = tracker.log(_make_log(model="model-a"))
        tracker.submit_feedback(rid, -1)

        for _ in range(2):
            rid = tracker.log(_make_log(model="model-b"))
            tracker.submit_feedback(rid, -1)

        stats = tracker.get_feedback_stats()
        assert stats["total_feedback"] == 6
        assert stats["total_requests"] == 6
        assert stats["feedback_rate"] == 1.0

        # Find model-a bucket
        model_a = [b for b in stats["buckets"] if b["model"] == "model-a"][0]
        assert model_a["total"] == 4
        assert model_a["positive"] == 3
        assert model_a["negative"] == 1
        assert model_a["success_rate"] == 0.75

        model_b = [b for b in stats["buckets"] if b["model"] == "model-b"][0]
        assert model_b["total"] == 2
        assert model_b["positive"] == 0
        assert model_b["success_rate"] == 0.0

        tracker.close()

    def test_feedback_rate_partial(self, tmp_path):
        """Feedback rate when only some requests have feedback."""
        tracker = Tracker(tmp_path / "test.db")
        rid1 = tracker.log(_make_log())
        tracker.log(_make_log())  # no feedback for this one
        tracker.submit_feedback(rid1, 1)

        stats = tracker.get_feedback_stats()
        assert stats["total_feedback"] == 1
        assert stats["total_requests"] == 2
        assert stats["feedback_rate"] == 0.5
        tracker.close()
