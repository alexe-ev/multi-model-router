"""Tests for AlertIncidentStore against a real temp-file DB.

A file, not ":memory:", so the two-connection cases are real: one live incident
per rule has to hold across processes, which is what the partial unique index
buys and what an in-memory database cannot exercise.
"""

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from mmrouter.alerts.channels import Alert
from mmrouter.alerts.store import AlertIncidentStore


def _alert(rule_name="error_rate", message="fired", severity="critical", details=None):
    return Alert(
        rule_name=rule_name,
        message=message,
        severity=severity,
        details=details if details is not None else {"rate": 0.2},
    )


def _age(conn, rule_name, seconds):
    """Push the live incident's last_seen back, instead of sleeping."""
    old = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()
    conn.execute(
        "UPDATE alert_incidents SET last_seen = ? "
        "WHERE rule_name = ? AND cleared_at IS NULL",
        (old, rule_name),
    )
    conn.commit()


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "alerts.db")


@pytest.fixture
def conn(db_path):
    c = sqlite3.connect(db_path)
    yield c
    c.close()


@pytest.fixture
def store(conn):
    return AlertIncidentStore(conn)


class TestRecordFire:
    def test_first_fire_opens_incident(self, store):
        store.record_fire(_alert())
        items = store.list_incidents()
        assert len(items) == 1
        row = items[0]
        assert row["rule_name"] == "error_rate"
        assert row["severity"] == "critical"
        assert row["message"] == "fired"
        assert row["fire_count"] == 1
        assert row["first_seen"] == row["last_seen"]
        assert row["cleared_at"] is None
        assert row["handled_at"] is None

    def test_repeat_fire_extends_the_same_row(self, store):
        store.record_fire(_alert(message="first", severity="warning", details={"n": 1}))
        store.record_fire(_alert(message="second", severity="critical", details={"n": 2}))

        items = store.list_incidents()
        assert len(items) == 1
        row = items[0]
        assert row["fire_count"] == 2
        assert row["message"] == "second"
        assert row["severity"] == "critical"
        assert row["details"] == {"n": 2}
        assert row["last_seen"] > row["first_seen"]

    def test_extends_even_while_handled(self, store):
        store.record_fire(_alert())
        incident_id = store.list_incidents()[0]["id"]
        store.set_handled(incident_id, True)

        store.record_fire(_alert(message="again"))

        items = store.list_incidents()
        assert len(items) == 1
        assert items[0]["fire_count"] == 2
        assert items[0]["message"] == "again"
        assert items[0]["handled_at"] is not None

    def test_two_rules_are_two_incidents(self, store):
        store.record_fire(_alert(rule_name="error_rate", severity="critical"))
        store.record_fire(_alert(rule_name="cost_spike", severity="warning"))

        by_rule = {i["rule_name"]: i for i in store.list_incidents()}
        assert set(by_rule) == {"error_rate", "cost_spike"}
        assert by_rule["error_rate"]["severity"] == "critical"
        assert by_rule["cost_spike"]["severity"] == "warning"

    def test_details_round_trip_as_dict(self, store):
        store.record_fire(_alert(details={"spent_today": 1.5, "nested": {"a": [1, 2]}}))
        assert store.list_incidents()[0]["details"] == {
            "spent_today": 1.5,
            "nested": {"a": [1, 2]},
        }

    def test_corrupt_details_degrade_to_empty_dict(self, store, conn):
        store.record_fire(_alert())
        conn.execute("UPDATE alert_incidents SET details = 'not json'")
        conn.commit()
        assert store.list_incidents()[0]["details"] == {}

    def test_ordering_is_most_recent_activity_first(self, store, conn):
        store.record_fire(_alert(rule_name="cost_spike", severity="warning"))
        store.record_fire(_alert(rule_name="error_rate"))
        # The cost_spike incident is older overall but just fired again.
        _age(conn, "error_rate", 60)

        names = [i["rule_name"] for i in store.list_incidents()]
        assert names == ["cost_spike", "error_rate"]

    def test_limit_caps_the_list(self, store, conn):
        store.record_fire(_alert(rule_name="error_rate"))
        store.record_fire(_alert(rule_name="cost_spike"))
        assert len(store.list_incidents(limit=1)) == 1


class TestClear:
    def test_clears_an_aged_incident(self, store, conn):
        store.record_fire(_alert())
        _age(conn, "error_rate", 600)

        assert store.clear("error_rate") is True
        assert store.list_incidents()[0]["cleared_at"] is not None

    def test_refuses_an_incident_younger_than_the_grace(self, store, conn):
        store.record_fire(_alert())

        assert store.clear("error_rate", min_age_seconds=300.0) is False
        assert store.list_incidents()[0]["cleared_at"] is None

        _age(conn, "error_rate", 301)
        assert store.clear("error_rate", min_age_seconds=300.0) is True
        assert store.list_incidents()[0]["cleared_at"] is not None

    def test_no_live_incident_returns_false_and_writes_nothing(self, store, conn, db_path):
        """The steady state: every routed request evaluates every rule silently."""
        observer = sqlite3.connect(db_path)
        try:
            def data_version():
                return observer.execute("PRAGMA data_version").fetchone()[0]

            before = data_version()
            statements = []
            conn.set_trace_callback(statements.append)

            assert store.clear("never_fired") is False

            conn.set_trace_callback(None)
            assert data_version() == before
            assert not any("UPDATE" in s.upper() for s in statements)

            # Positive control: a real write does move the observer's version.
            store.record_fire(_alert(rule_name="never_fired"))
            assert data_version() != before
        finally:
            observer.close()

    def test_re_fire_after_clear_opens_a_second_row(self, store, conn):
        store.record_fire(_alert())
        _age(conn, "error_rate", 600)
        store.clear("error_rate")

        store.record_fire(_alert(message="back again"))

        items = store.list_incidents()
        assert len(items) == 2
        live = [i for i in items if i["cleared_at"] is None]
        assert len(live) == 1
        assert live[0]["fire_count"] == 1
        assert live[0]["message"] == "back again"

    def test_new_row_is_open_even_when_the_old_one_was_handled(self, store, conn):
        store.record_fire(_alert())
        first_id = store.list_incidents()[0]["id"]
        store.set_handled(first_id, True)
        _age(conn, "error_rate", 600)
        store.clear("error_rate")

        store.record_fire(_alert())

        items = {i["id"]: i for i in store.list_incidents()}
        assert len(items) == 2
        assert items[first_id]["handled_at"] is not None
        new_id = next(i for i in items if i != first_id)
        assert items[new_id]["handled_at"] is None
        assert items[new_id]["cleared_at"] is None


class TestTwoConnections:
    def test_one_live_incident_per_rule_across_connections(self, db_path):
        conn_a = sqlite3.connect(db_path)
        conn_b = sqlite3.connect(db_path)
        try:
            store_a = AlertIncidentStore(conn_a)
            store_b = AlertIncidentStore(conn_b)

            store_a.record_fire(_alert())
            store_b.record_fire(_alert(message="from the other process"))

            items = store_a.list_incidents()
            assert len(items) == 1
            assert items[0]["fire_count"] == 2
            assert items[0]["message"] == "from the other process"
        finally:
            conn_a.close()
            conn_b.close()

    def test_flapping_across_connections_leaves_one_row(self, db_path):
        """A second process's silent evaluation must not close a fresh incident.

        Without the age guard in clear() this sequence produces 21 rows.
        """
        conn_a = sqlite3.connect(db_path)
        conn_b = sqlite3.connect(db_path)
        try:
            store_a = AlertIncidentStore(conn_a)
            store_b = AlertIncidentStore(conn_b)

            store_a.record_fire(_alert())
            for _ in range(20):
                store_b.clear("error_rate")
                store_a.record_fire(_alert())

            items = store_a.list_incidents()
            assert len(items) == 1
            assert items[0]["fire_count"] == 21
            assert items[0]["cleared_at"] is None
        finally:
            conn_a.close()
            conn_b.close()

    def test_two_concurrent_clears_close_the_incident_once(self, db_path):
        conn_a = sqlite3.connect(db_path)
        conn_b = sqlite3.connect(db_path)
        try:
            store_a = AlertIncidentStore(conn_a)
            store_b = AlertIncidentStore(conn_b)
            store_a.record_fire(_alert())
            _age(conn_a, "error_rate", 600)

            assert store_a.clear("error_rate") is True
            assert store_b.clear("error_rate") is False

            rows = conn_a.execute(
                "SELECT cleared_at FROM alert_incidents"
            ).fetchall()
            assert len(rows) == 1
            assert rows[0][0] is not None
        finally:
            conn_a.close()
            conn_b.close()


class TestHandledAndCounts:
    def test_counts_on_an_empty_table(self, store):
        assert store.counts() == {"total": 0, "open_count": 0, "handled_count": 0}

    def test_counts_split_open_and_handled(self, store):
        store.record_fire(_alert(rule_name="error_rate"))
        store.record_fire(_alert(rule_name="cost_spike"))
        handled_id = store.list_incidents()[0]["id"]
        store.set_handled(handled_id, True)

        assert store.counts() == {"total": 2, "open_count": 1, "handled_count": 1}

    def test_mark_and_unmark(self, store):
        store.record_fire(_alert())
        incident_id = store.list_incidents()[0]["id"]

        marked = store.set_handled(incident_id, True)
        assert marked["handled_at"] is not None
        assert marked["id"] == incident_id

        unmarked = store.set_handled(incident_id, False)
        assert unmarked["handled_at"] is None

    def test_mark_touches_nothing_else(self, store):
        store.record_fire(_alert())
        before = store.list_incidents()[0]

        after = store.set_handled(before["id"], True)

        for key in ("fire_count", "first_seen", "last_seen", "message", "severity", "cleared_at"):
            assert after[key] == before[key]
        assert after["details"] == before["details"]

    def test_unknown_id_returns_none(self, store):
        assert store.set_handled(999999, True) is None

    def test_details_survive_a_mark(self, store, conn):
        store.record_fire(_alert(details={"rate": 0.42}))
        incident_id = store.list_incidents()[0]["id"]
        store.set_handled(incident_id, True)
        raw = conn.execute("SELECT details FROM alert_incidents").fetchone()[0]
        assert json.loads(raw) == {"rate": 0.42}
