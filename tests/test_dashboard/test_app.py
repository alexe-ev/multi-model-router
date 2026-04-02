"""Tests for dashboard FastAPI endpoints."""
import sqlite3
import pytest
from fastapi.testclient import TestClient
from mmrouter.dashboard.app import create_app
from mmrouter.tracker.logger import Tracker, _CREATE_TABLE, _INSERT


def _seed_db(db_path):
    """Insert test request data."""
    tracker = Tracker(db_path)
    conn = tracker.connection
    test_data = [
        ("2026-04-01T10:00:00", "abc123", "simple", "factual", 0.9, "claude-haiku-4-5-20251001", 10, 20, 0.0001, 50.0, 0, 0, 1, 0, 0, None, None),
        ("2026-04-01T11:00:00", "def456", "medium", "reasoning", 0.8, "claude-sonnet-4-6", 50, 100, 0.001, 200.0, 0, 0, 1, 0, 0, None, None),
        ("2026-04-01T12:00:00", "ghi789", "complex", "code", 0.7, "claude-opus-4-6", 100, 200, 0.01, 500.0, 1, 0, 1, 0, 0, None, None),
        ("2026-04-02T10:00:00", "jkl012", "simple", "factual", 0.95, "claude-haiku-4-5-20251001", 15, 25, 0.00015, 45.0, 0, 0, 1, 0, 0, None, None),
        ("2026-04-02T11:00:00", "mno345", "medium", "creative", 0.75, "claude-sonnet-4-6", 60, 120, 0.0012, 180.0, 0, 0, 1, 0, 0, None, None),
    ]
    for row in test_data:
        conn.execute(_INSERT, row)
    conn.commit()
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


class TestStatsEndpoint:
    def test_returns_stats(self, client):
        r = client.get("/api/stats")
        assert r.status_code == 200
        data = r.json()
        assert data["total_requests"] == 5
        assert data["total_cost"] > 0
        assert "savings" in data
        assert "model_distribution" in data

    def test_empty_db(self, empty_client):
        r = empty_client.get("/api/stats")
        assert r.status_code == 200
        data = r.json()
        assert data["total_requests"] == 0


class TestDailyEndpoint:
    def test_returns_daily(self, client):
        r = client.get("/api/stats/daily")
        assert r.status_code == 200
        data = r.json()
        assert len(data) > 0
        assert "date" in data[0]
        assert "model" in data[0]


class TestDistributionEndpoint:
    def test_returns_distribution(self, client):
        r = client.get("/api/stats/distribution")
        assert r.status_code == 200
        data = r.json()
        assert "by_complexity" in data
        assert "by_category" in data


class TestRequestsEndpoint:
    def test_returns_paginated(self, client):
        r = client.get("/api/requests?limit=2&offset=0")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 5
        assert len(data["items"]) == 2
        assert data["limit"] == 2
        assert data["offset"] == 0

    def test_offset(self, client):
        r = client.get("/api/requests?limit=10&offset=3")
        data = r.json()
        assert len(data["items"]) == 2  # 5 total - 3 offset = 2

    def test_filter_by_model(self, client):
        r = client.get("/api/requests?model=claude-haiku-4-5-20251001")
        data = r.json()
        assert data["total"] == 2
        assert all(item["model"] == "claude-haiku-4-5-20251001" for item in data["items"])

    def test_filter_by_complexity(self, client):
        r = client.get("/api/requests?complexity=simple")
        data = r.json()
        assert data["total"] == 2

    def test_filter_by_category(self, client):
        r = client.get("/api/requests?category=factual")
        data = r.json()
        assert data["total"] == 2

    def test_no_prompt_in_response(self, client):
        r = client.get("/api/requests")
        data = r.json()
        for item in data["items"]:
            assert "prompt" not in item
            assert "prompt_hash" in item


class TestModelsEndpoint:
    def test_returns_models(self, client):
        r = client.get("/api/models")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 3  # haiku, sonnet, opus
        assert "model" in data[0]
        assert "count" in data[0]
        assert "total_cost" in data[0]
        assert "avg_latency_ms" in data[0]
