"""Tests for the dashboard's alert-incident endpoints."""

import sqlite3

import pytest
from fastapi.testclient import TestClient

from mmrouter.alerts.channels import Alert
from mmrouter.alerts.store import AlertIncidentStore
from mmrouter.dashboard.app import create_app
from mmrouter.tracker.logger import Tracker, _INSERT


def _seed_db(db_path):
    """A database as it looks before this feature: requests only, no incidents."""
    tracker = Tracker(db_path)
    conn = tracker.connection
    conn.execute(
        _INSERT,
        ("2026-04-01T10:00:00", "abc123", "simple", "factual", 0.9,
         "claude-haiku-4-5-20251001", 10, 20, 0.0001, 50.0, 0, 0, 1, 0, 0, None, None),
    )
    conn.commit()
    tracker.close()


def _seed_incidents(db_path):
    """Two incidents: error_rate fired twice, cost_spike once."""
    conn = sqlite3.connect(db_path)
    store = AlertIncidentStore(conn)
    store.record_fire(Alert(
        rule_name="cost_spike", message="cost is up", severity="warning",
        details={"multiplier": 5.0},
    ))
    store.record_fire(Alert(
        rule_name="error_rate", message="fallback rate 20%", severity="critical",
        details={"rate": 0.2},
    ))
    store.record_fire(Alert(
        rule_name="error_rate", message="fallback rate 30%", severity="critical",
        details={"rate": 0.3},
    ))
    conn.close()


@pytest.fixture
def client(tmp_path):
    db_path = str(tmp_path / "test.db")
    _seed_db(db_path)
    _seed_incidents(db_path)
    app = create_app(db_path)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def upgraded_client(tmp_path):
    """A pre-feature database: nothing ever fired, no alert_incidents table."""
    db_path = str(tmp_path / "old.db")
    _seed_db(db_path)
    app = create_app(db_path)
    with TestClient(app) as c:
        yield c


class TestGetAlerts:
    def test_shape_and_counts(self, client):
        r = client.get("/api/alerts")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 2
        assert data["open_count"] == 2
        assert data["handled_count"] == 0
        assert data["limit"] == 200
        assert len(data["items"]) == 2

        by_rule = {i["rule_name"]: i for i in data["items"]}
        assert by_rule["error_rate"]["fire_count"] == 2
        assert by_rule["error_rate"]["severity"] == "critical"
        assert by_rule["error_rate"]["message"] == "fallback rate 30%"
        assert by_rule["error_rate"]["details"] == {"rate": 0.3}
        assert by_rule["cost_spike"]["severity"] == "warning"
        assert by_rule["cost_spike"]["cleared_at"] is None
        assert by_rule["cost_spike"]["handled_at"] is None

    def test_ordering_is_most_recent_activity_first(self, client):
        items = client.get("/api/alerts").json()["items"]
        assert [i["rule_name"] for i in items] == ["error_rate", "cost_spike"]

    def test_limit_applies(self, client):
        data = client.get("/api/alerts?limit=1").json()
        assert len(data["items"]) == 1
        assert data["limit"] == 1
        # Counts come from their own query, so the limit never distorts them.
        assert data["total"] == 2
        assert data["open_count"] == 2

    def test_limit_zero_rejected(self, client):
        assert client.get("/api/alerts?limit=0").status_code == 422

    def test_limit_above_ceiling_rejected(self, client):
        assert client.get("/api/alerts?limit=501").status_code == 422

    def test_limit_ceiling_accepted(self, client):
        assert client.get("/api/alerts?limit=500").status_code == 200

    def test_pre_feature_db_reads_empty(self, upgraded_client):
        r = upgraded_client.get("/api/alerts")
        assert r.status_code == 200
        data = r.json()
        assert data["items"] == []
        assert data["total"] == 0
        assert data["open_count"] == 0
        assert data["handled_count"] == 0

    def test_pre_feature_db_still_serves_the_read_endpoints(self, upgraded_client):
        assert upgraded_client.get("/api/stats").status_code == 200
        assert upgraded_client.get("/api/models").status_code == 200


class TestSetHandled:
    def _first_id(self, client):
        return client.get("/api/alerts").json()["items"][0]["id"]

    def test_mark_returns_the_updated_row(self, client):
        incident_id = self._first_id(client)
        r = client.post(f"/api/alerts/{incident_id}/handled", json={"handled": True})
        assert r.status_code == 200
        item = r.json()
        assert item["id"] == incident_id
        assert item["handled_at"] is not None
        assert item["fire_count"] == 2

    def test_mark_moves_the_counts(self, client):
        incident_id = self._first_id(client)
        client.post(f"/api/alerts/{incident_id}/handled", json={"handled": True})
        data = client.get("/api/alerts").json()
        assert data["total"] == 2
        assert data["open_count"] == 1
        assert data["handled_count"] == 1

    def test_unmark(self, client):
        incident_id = self._first_id(client)
        client.post(f"/api/alerts/{incident_id}/handled", json={"handled": True})
        r = client.post(f"/api/alerts/{incident_id}/handled", json={"handled": False})
        assert r.status_code == 200
        assert r.json()["handled_at"] is None
        assert client.get("/api/alerts").json()["open_count"] == 2

    def test_unmark_something_never_handled_is_a_no_op(self, client):
        incident_id = self._first_id(client)
        r = client.post(f"/api/alerts/{incident_id}/handled", json={"handled": False})
        assert r.status_code == 200
        assert r.json()["handled_at"] is None
        assert client.get("/api/alerts").json()["total"] == 2

    def test_unknown_id_is_404(self, client):
        r = client.post("/api/alerts/999999/handled", json={"handled": True})
        assert r.status_code == 404

    def test_id_past_sqlites_integer_range_is_404(self, client):
        """Python ints are unbounded; SQLite's are not. An id it cannot hold
        still names no incident, so it gets the same answer as any unknown id."""
        r = client.post("/api/alerts/99999999999999999999/handled", json={"handled": True})
        assert r.status_code == 404

    def test_negative_and_zero_ids_are_404(self, client):
        assert client.post("/api/alerts/-1/handled", json={"handled": True}).status_code == 404
        assert client.post("/api/alerts/0/handled", json={"handled": True}).status_code == 404

    def test_store_failure_is_503_and_leaves_the_row_alone(self, client, monkeypatch):
        incident_id = self._first_id(client)

        def boom(self, incident_id, handled):
            raise sqlite3.OperationalError("attempt to write a readonly database")

        monkeypatch.setattr(AlertIncidentStore, "set_handled", boom)

        r = client.post(f"/api/alerts/{incident_id}/handled", json={"handled": True})
        assert r.status_code == 503
        detail = r.json()["detail"]
        assert detail == "Could not write to alert history on this database"
        # Nothing internal in the message: no path, no SQL, no driver text.
        assert "/" not in detail
        assert "UPDATE" not in detail.upper()
        assert "readonly" not in detail

        data = client.get("/api/alerts").json()
        assert data["open_count"] == 2
        assert data["handled_count"] == 0
        assert all(i["handled_at"] is None for i in data["items"])

    def test_body_without_handled_is_422(self, client):
        incident_id = self._first_id(client)
        assert client.post(f"/api/alerts/{incident_id}/handled", json={}).status_code == 422


class TestStoreUnavailable:
    """The DDL failing at construction must cost the alerts endpoints only."""

    def test_alerts_endpoints_are_503_while_the_read_endpoints_serve(
        self, tmp_path, monkeypatch
    ):
        db_path = str(tmp_path / "unavailable.db")
        _seed_db(db_path)

        def boom(self, conn):
            raise sqlite3.OperationalError("attempt to write a readonly database")

        monkeypatch.setattr(AlertIncidentStore, "__init__", boom)

        app = create_app(db_path)
        with TestClient(app) as c:
            assert c.get("/api/alerts").status_code == 503
            assert c.post("/api/alerts/1/handled", json={"handled": True}).status_code == 503
            for path in (
                "/api/stats",
                "/api/stats/daily",
                "/api/stats/distribution",
                "/api/requests",
                "/api/stats/feedback",
                "/api/models",
            ):
                assert c.get(path).status_code == 200, path
