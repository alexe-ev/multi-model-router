"""Tests for FeedbackScorer: scoring, reranking, thresholds, decay."""

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from mmrouter.models import AdaptiveConfig
from mmrouter.router.adaptive import FeedbackScorer
from mmrouter.tracker.logger import Tracker, _CREATE_TABLE, _CREATE_FEEDBACK_TABLE, _INSERT


def _setup_db(tmp_path):
    """Create a tracker DB and return (tracker, conn)."""
    tracker = Tracker(tmp_path / "test.db")
    return tracker, tracker.connection


def _insert_request(conn, model="model-a", complexity="simple", category="factual"):
    """Insert a request row and return its id."""
    cur = conn.execute(
        _INSERT,
        (
            datetime.now(timezone.utc).isoformat(),
            "abc123",
            complexity,
            category,
            0.9,
            model,
            10,
            20,
            0.001,
            100.0,
            0,
            0,
            1,
            0,
            0,
        ),
    )
    conn.commit()
    return cur.lastrowid


def _insert_feedback(conn, request_id, rating, ts=None):
    """Insert a feedback row."""
    if ts is None:
        ts = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO feedback (request_id, rating, timestamp) VALUES (?, ?, ?)",
        (request_id, rating, ts),
    )
    conn.commit()


class TestGetModelScores:
    def test_no_feedback_returns_empty(self, tmp_path):
        tracker, conn = _setup_db(tmp_path)
        scorer = FeedbackScorer(conn, AdaptiveConfig(min_feedback_count=1))
        scores = scorer.get_model_scores("simple", "factual")
        assert scores == {}
        tracker.close()

    def test_below_min_feedback_excluded(self, tmp_path):
        tracker, conn = _setup_db(tmp_path)
        # Insert 3 feedbacks but require 5
        for _ in range(3):
            rid = _insert_request(conn, "model-a")
            _insert_feedback(conn, rid, 1)

        scorer = FeedbackScorer(conn, AdaptiveConfig(min_feedback_count=5))
        scores = scorer.get_model_scores("simple", "factual")
        assert "model-a" not in scores
        tracker.close()

    def test_at_min_feedback_included(self, tmp_path):
        tracker, conn = _setup_db(tmp_path)
        for _ in range(5):
            rid = _insert_request(conn, "model-a")
            _insert_feedback(conn, rid, 1)

        scorer = FeedbackScorer(conn, AdaptiveConfig(min_feedback_count=5))
        scores = scorer.get_model_scores("simple", "factual")
        assert scores["model-a"] == 1.0
        tracker.close()

    def test_mixed_feedback_score(self, tmp_path):
        tracker, conn = _setup_db(tmp_path)
        # 3 positive, 2 negative = 0.6 success rate
        for _ in range(3):
            rid = _insert_request(conn, "model-a")
            _insert_feedback(conn, rid, 1)
        for _ in range(2):
            rid = _insert_request(conn, "model-a")
            _insert_feedback(conn, rid, -1)

        scorer = FeedbackScorer(conn, AdaptiveConfig(min_feedback_count=5))
        scores = scorer.get_model_scores("simple", "factual")
        assert scores["model-a"] == pytest.approx(0.6)
        tracker.close()

    def test_scores_scoped_to_bucket(self, tmp_path):
        tracker, conn = _setup_db(tmp_path)
        # model-a in simple/factual: all positive
        for _ in range(5):
            rid = _insert_request(conn, "model-a", "simple", "factual")
            _insert_feedback(conn, rid, 1)
        # model-a in medium/reasoning: all negative
        for _ in range(5):
            rid = _insert_request(conn, "model-a", "medium", "reasoning")
            _insert_feedback(conn, rid, -1)

        scorer = FeedbackScorer(conn, AdaptiveConfig(min_feedback_count=5))

        simple_scores = scorer.get_model_scores("simple", "factual")
        assert simple_scores["model-a"] == 1.0

        medium_scores = scorer.get_model_scores("medium", "reasoning")
        assert medium_scores["model-a"] == 0.0

        tracker.close()

    def test_decay_excludes_old_feedback(self, tmp_path):
        tracker, conn = _setup_db(tmp_path)
        old_ts = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        recent_ts = datetime.now(timezone.utc).isoformat()

        # 5 old positive feedbacks (outside 30-day window)
        for _ in range(5):
            rid = _insert_request(conn, "model-a")
            _insert_feedback(conn, rid, 1, ts=old_ts)

        # 5 recent negative feedbacks
        for _ in range(5):
            rid = _insert_request(conn, "model-a")
            _insert_feedback(conn, rid, -1, ts=recent_ts)

        scorer = FeedbackScorer(conn, AdaptiveConfig(min_feedback_count=5, decay_days=30))
        scores = scorer.get_model_scores("simple", "factual")
        # Only recent (negative) count
        assert scores["model-a"] == 0.0
        tracker.close()

    def test_multiple_models_scored(self, tmp_path):
        tracker, conn = _setup_db(tmp_path)
        for _ in range(5):
            rid = _insert_request(conn, "model-a")
            _insert_feedback(conn, rid, 1)
        for _ in range(5):
            rid = _insert_request(conn, "model-b")
            _insert_feedback(conn, rid, -1)

        scorer = FeedbackScorer(conn, AdaptiveConfig(min_feedback_count=5))
        scores = scorer.get_model_scores("simple", "factual")
        assert scores["model-a"] == 1.0
        assert scores["model-b"] == 0.0
        tracker.close()


class TestRerankModels:
    def test_no_feedback_no_change(self, tmp_path):
        tracker, conn = _setup_db(tmp_path)
        scorer = FeedbackScorer(conn, AdaptiveConfig(min_feedback_count=5))
        models = ["model-a", "model-b", "model-c"]
        result, reranked = scorer.rerank_models(models, "simple", "factual")
        assert result == models
        assert not reranked
        tracker.close()

    def test_penalized_model_moves_to_end(self, tmp_path):
        tracker, conn = _setup_db(tmp_path)
        # model-a: 20% success rate (below penalty_threshold 0.4)
        for _ in range(4):
            rid = _insert_request(conn, "model-a")
            _insert_feedback(conn, rid, -1)
        rid = _insert_request(conn, "model-a")
        _insert_feedback(conn, rid, 1)

        scorer = FeedbackScorer(
            conn,
            AdaptiveConfig(min_feedback_count=5, penalty_threshold=0.4, boost_threshold=0.8),
        )
        result, reranked = scorer.rerank_models(
            ["model-a", "model-b"], "simple", "factual"
        )
        assert result == ["model-b", "model-a"]
        assert reranked
        tracker.close()

    def test_boosted_model_moves_to_front(self, tmp_path):
        tracker, conn = _setup_db(tmp_path)
        # model-b: 100% success rate (above boost_threshold 0.8)
        for _ in range(5):
            rid = _insert_request(conn, "model-b")
            _insert_feedback(conn, rid, 1)

        scorer = FeedbackScorer(
            conn,
            AdaptiveConfig(min_feedback_count=5, boost_threshold=0.8),
        )
        result, reranked = scorer.rerank_models(
            ["model-a", "model-b"], "simple", "factual"
        )
        assert result == ["model-b", "model-a"]
        assert reranked
        tracker.close()

    def test_never_adds_or_removes_models(self, tmp_path):
        tracker, conn = _setup_db(tmp_path)
        for _ in range(5):
            rid = _insert_request(conn, "model-a")
            _insert_feedback(conn, rid, 1)

        scorer = FeedbackScorer(conn, AdaptiveConfig(min_feedback_count=5))
        original = ["model-a", "model-b", "model-c"]
        result, _ = scorer.rerank_models(original, "simple", "factual")
        assert set(result) == set(original)
        assert len(result) == len(original)
        tracker.close()

    def test_insufficient_feedback_keeps_order(self, tmp_path):
        tracker, conn = _setup_db(tmp_path)
        # Only 3 feedbacks, min is 5
        for _ in range(3):
            rid = _insert_request(conn, "model-a")
            _insert_feedback(conn, rid, -1)

        scorer = FeedbackScorer(conn, AdaptiveConfig(min_feedback_count=5))
        models = ["model-a", "model-b"]
        result, reranked = scorer.rerank_models(models, "simple", "factual")
        assert result == models
        assert not reranked
        tracker.close()

    def test_boost_and_penalty_together(self, tmp_path):
        tracker, conn = _setup_db(tmp_path)
        # model-a: 0% (penalized)
        for _ in range(5):
            rid = _insert_request(conn, "model-a")
            _insert_feedback(conn, rid, -1)
        # model-c: 100% (boosted)
        for _ in range(5):
            rid = _insert_request(conn, "model-c")
            _insert_feedback(conn, rid, 1)

        scorer = FeedbackScorer(
            conn,
            AdaptiveConfig(min_feedback_count=5, penalty_threshold=0.4, boost_threshold=0.8),
        )
        result, reranked = scorer.rerank_models(
            ["model-a", "model-b", "model-c"], "simple", "factual"
        )
        # boosted first, neutral middle, penalized last
        assert result == ["model-c", "model-b", "model-a"]
        assert reranked
        tracker.close()

    def test_all_negative_feedback(self, tmp_path):
        tracker, conn = _setup_db(tmp_path)
        for model in ["model-a", "model-b"]:
            for _ in range(5):
                rid = _insert_request(conn, model)
                _insert_feedback(conn, rid, -1)

        scorer = FeedbackScorer(
            conn,
            AdaptiveConfig(min_feedback_count=5, penalty_threshold=0.4),
        )
        result, reranked = scorer.rerank_models(
            ["model-a", "model-b"], "simple", "factual"
        )
        # Both penalized, but order within penalized group preserved
        assert set(result) == {"model-a", "model-b"}
        assert not reranked  # both in same bucket, order unchanged
        tracker.close()

    def test_single_model_no_change(self, tmp_path):
        tracker, conn = _setup_db(tmp_path)
        for _ in range(5):
            rid = _insert_request(conn, "model-a")
            _insert_feedback(conn, rid, -1)

        scorer = FeedbackScorer(
            conn,
            AdaptiveConfig(min_feedback_count=5, penalty_threshold=0.4),
        )
        result, reranked = scorer.rerank_models(
            ["model-a"], "simple", "factual"
        )
        assert result == ["model-a"]
        assert not reranked  # single model, can't reorder
        tracker.close()
