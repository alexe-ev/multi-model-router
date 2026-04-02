"""Tests for dashboard feedback stats endpoint."""

import pytest
from fastapi.testclient import TestClient

from mmrouter.dashboard.app import create_app
from mmrouter.tracker.logger import Tracker, _INSERT


def _seed_db(db_path):
    """Insert test request data and some feedback."""
    tracker = Tracker(db_path)
    conn = tracker.connection
    test_data = [
        ("2026-04-01T10:00:00", "abc123", "simple", "factual", 0.9, "claude-haiku-4-5-20251001", 10, 20, 0.0001, 50.0, 0, 0, 1, 0, 0),
        ("2026-04-01T11:00:00", "def456", "simple", "factual", 0.8, "claude-haiku-4-5-20251001", 50, 100, 0.001, 200.0, 0, 0, 1, 0, 0),
        ("2026-04-01T12:00:00", "ghi789", "medium", "code", 0.7, "claude-sonnet-4-6", 100, 200, 0.01, 500.0, 1, 0, 1, 0, 0),
    ]
    for row in test_data:
        conn.execute(_INSERT, row)
    conn.commit()

    # Add feedback: request 1 positive, request 2 negative
    tracker.submit_feedback(1, 1)
    tracker.submit_feedback(2, -1)

    return tracker


@pytest.fixture
def client(tmp_path):
    db_path = str(tmp_path / "test.db")
    tracker = _seed_db(db_path)
    app = create_app(db_path)
    with TestClient(app) as c:
        yield c
    tracker.close()


@pytest.fixture
def empty_client(tmp_path):
    db_path = str(tmp_path / "empty.db")
    app = create_app(db_path)
    with TestClient(app) as c:
        yield c


class TestFeedbackStatsEndpoint:
    def test_returns_feedback_stats(self, client):
        r = client.get("/api/stats/feedback")
        assert r.status_code == 200
        data = r.json()
        assert data["total_feedback"] == 2
        assert data["total_requests"] == 3
        assert len(data["buckets"]) > 0

    def test_feedback_rate(self, client):
        r = client.get("/api/stats/feedback")
        data = r.json()
        # 2 feedbacks / 3 requests
        assert data["feedback_rate"] == pytest.approx(2 / 3, abs=0.01)

    def test_bucket_details(self, client):
        r = client.get("/api/stats/feedback")
        data = r.json()
        # Both feedbacks are for simple/factual/claude-haiku
        haiku_bucket = [
            b for b in data["buckets"]
            if b["model"] == "claude-haiku-4-5-20251001"
        ]
        assert len(haiku_bucket) == 1
        assert haiku_bucket[0]["total"] == 2
        assert haiku_bucket[0]["positive"] == 1
        assert haiku_bucket[0]["negative"] == 1
        assert haiku_bucket[0]["success_rate"] == 0.5

    def test_empty_feedback(self, empty_client):
        r = empty_client.get("/api/stats/feedback")
        assert r.status_code == 200
        data = r.json()
        assert data["total_feedback"] == 0
        assert data["buckets"] == []
