"""Tests for alert rules engine."""

import sqlite3
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

import pytest

from mmrouter.alerts.channels import Alert
from mmrouter.alerts.rules import (
    BUILTIN_RULES,
    AlertManager,
    AlertRule,
    create_budget_warning_rule,
    _check_cost_spike,
    _check_error_rate,
)


@pytest.fixture
def db():
    """In-memory SQLite with requests table."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE requests (
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
            fallback_used INTEGER NOT NULL DEFAULT 0,
            cascade_used INTEGER NOT NULL DEFAULT 0,
            cascade_attempts INTEGER NOT NULL DEFAULT 1,
            cache_read_tokens INTEGER NOT NULL DEFAULT 0,
            cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
            experiment_id INTEGER,
            variant TEXT
        )
    """)
    conn.commit()
    return conn


def _insert_request(conn, *, cost=0.001, fallback=0, hours_ago=0):
    """Insert a dummy request at a given time offset."""
    ts = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    # Use consistent format that sorts correctly with SQLite comparisons
    ts_str = ts.strftime("%Y-%m-%dT%H:%M:%S")
    conn.execute(
        """INSERT INTO requests (timestamp, prompt_hash, complexity, category,
           confidence, model, tokens_in, tokens_out, cost, latency_ms, fallback_used)
           VALUES (?, 'abc', 'simple', 'factual', 0.9, 'test-model', 100, 50, ?, 200.0, ?)""",
        (ts_str, cost, fallback),
    )
    conn.commit()


class TestCostSpikeRule:
    def test_no_requests_returns_none(self, db):
        assert _check_cost_spike(db) is None

    def test_no_history_returns_none(self, db):
        """Current hour has cost but no prior history to compare."""
        _insert_request(db, cost=1.0, hours_ago=0)
        assert _check_cost_spike(db) is None

    def test_normal_cost_returns_none(self, db):
        """Cost is within normal range."""
        # History: 10 distinct hours, $0.01 each (well outside current hour)
        for h in range(2, 12):
            _insert_request(db, cost=0.01, hours_ago=h)
        # Current hour: $0.01 (1x average, below 2x)
        _insert_request(db, cost=0.01, hours_ago=0)
        result = _check_cost_spike(db)
        assert result is None

    def test_spike_fires(self, db):
        """Cost is >2x average, should fire."""
        # History: 10 distinct hours at $0.10 each, avg = $0.10/hour
        for h in range(2, 12):
            _insert_request(db, cost=0.10, hours_ago=h)
        # Current hour: $0.50 (5x average)
        _insert_request(db, cost=0.50, hours_ago=0)
        alert = _check_cost_spike(db)
        assert alert is not None
        assert alert.rule_name == "cost_spike"
        assert alert.details["multiplier"] == 5.0


class TestErrorRateRule:
    def test_too_few_requests(self, db):
        for _ in range(5):
            _insert_request(db, fallback=1)
        assert _check_error_rate(db) is None

    def test_low_error_rate(self, db):
        for _ in range(95):
            _insert_request(db, fallback=0)
        for _ in range(5):
            _insert_request(db, fallback=1)
        assert _check_error_rate(db) is None

    def test_high_error_rate_fires(self, db):
        for _ in range(80):
            _insert_request(db, fallback=0)
        for _ in range(20):
            _insert_request(db, fallback=1)
        alert = _check_error_rate(db)
        assert alert is not None
        assert alert.rule_name == "error_rate"
        assert alert.severity == "critical"
        assert alert.details["rate"] == 0.2


class TestBudgetWarningRule:
    def test_no_limit_returns_none(self, db):
        rule = create_budget_warning_rule(daily_limit=0.0)
        assert rule.check(db) is None

    def test_under_threshold_returns_none(self, db):
        _insert_request(db, cost=0.05)
        rule = create_budget_warning_rule(daily_limit=1.0, warn_pct=0.90)
        assert rule.check(db) is None

    def test_over_threshold_fires(self, db):
        _insert_request(db, cost=0.95)
        rule = create_budget_warning_rule(daily_limit=1.0, warn_pct=0.90)
        alert = rule.check(db)
        assert alert is not None
        assert alert.rule_name == "budget_warning"
        assert alert.details["usage_pct"] == 95.0

    def test_over_100_pct_is_critical(self, db):
        _insert_request(db, cost=1.5)
        rule = create_budget_warning_rule(daily_limit=1.0)
        alert = rule.check(db)
        assert alert is not None
        assert alert.severity == "critical"


class TestAlertManager:
    def test_no_rules_returns_empty(self, db):
        mgr = AlertManager(db, rules=[])
        assert mgr.check_all() == []

    def test_fires_when_condition_met(self, db):
        def always_fire(conn):
            return Alert(
                rule_name="always",
                message="always fires",
                severity="warning",
                details={},
            )

        rule = AlertRule(name="always", check=always_fire, cooldown_seconds=0)
        mgr = AlertManager(db, rules=[rule])
        fired = mgr.check_all()
        assert len(fired) == 1
        assert fired[0].rule_name == "always"

    def test_cooldown_prevents_repeat(self, db):
        call_count = 0

        def counting_check(conn):
            nonlocal call_count
            call_count += 1
            return Alert(
                rule_name="counted",
                message="fired",
                severity="warning",
                details={},
            )

        rule = AlertRule(name="counted", check=counting_check, cooldown_seconds=9999)
        mgr = AlertManager(db, rules=[rule])

        fired1 = mgr.check_all()
        assert len(fired1) == 1

        fired2 = mgr.check_all()
        assert len(fired2) == 0  # Cooldown blocks it

    def test_cooldown_expires(self, db):
        def always_fire(conn):
            return Alert(
                rule_name="expiring",
                message="fired",
                severity="warning",
                details={},
            )

        rule = AlertRule(name="expiring", check=always_fire, cooldown_seconds=0)
        mgr = AlertManager(db, rules=[rule])

        fired1 = mgr.check_all()
        assert len(fired1) == 1
        # cooldown_seconds=0 means it can fire again immediately
        fired2 = mgr.check_all()
        assert len(fired2) == 1

    def test_webhook_called_on_fire(self, db):
        def always_fire(conn):
            return Alert(
                rule_name="webhook_test",
                message="test",
                severity="warning",
                details={},
            )

        rule = AlertRule(name="webhook_test", check=always_fire, cooldown_seconds=0)
        mgr = AlertManager(db, rules=[rule], webhook_url="http://example.com/hook")

        # Mock the webhook send
        mgr._webhook.send = MagicMock(return_value=True)
        mgr.check_all()
        mgr._webhook.send.assert_called_once()

    def test_condition_returns_none_skips(self, db):
        rule = AlertRule(name="silent", check=lambda conn: None, cooldown_seconds=0)
        mgr = AlertManager(db, rules=[rule])
        assert mgr.check_all() == []

    def test_is_in_cooldown(self, db):
        def always_fire(conn):
            return Alert(
                rule_name="cd_test",
                message="fired",
                severity="warning",
                details={},
            )

        rule = AlertRule(name="cd_test", check=always_fire, cooldown_seconds=9999)
        mgr = AlertManager(db, rules=[rule])
        assert not mgr.is_in_cooldown("cd_test")
        mgr.check_all()
        assert mgr.is_in_cooldown("cd_test")

    def test_get_status(self, db):
        rule = AlertRule(name="status_rule", check=lambda c: None, cooldown_seconds=300, severity="warning")
        mgr = AlertManager(db, rules=[rule], webhook_url="http://example.com")
        status = mgr.get_status()
        assert status["webhook_configured"] is True
        assert len(status["rules"]) == 1
        assert status["rules"][0]["name"] == "status_rule"
        assert status["rules"][0]["in_cooldown"] is False

    def test_get_status_no_webhook(self, db):
        mgr = AlertManager(db, rules=[])
        status = mgr.get_status()
        assert status["webhook_configured"] is False
        assert status["webhook_url"] is None

    def test_builtin_rules_exist(self):
        assert "cost_spike" in BUILTIN_RULES
        assert "error_rate" in BUILTIN_RULES

    def test_cooldown_override(self, db):
        rule = AlertRule(name="override", check=lambda c: None, cooldown_seconds=300)
        mgr = AlertManager(db, rules=[rule], cooldown_seconds=600)
        assert mgr.rules[0].cooldown_seconds == 600

    def test_last_fired(self, db):
        def always_fire(conn):
            return Alert(rule_name="lf_test", message="fired", severity="warning", details={})

        rule = AlertRule(name="lf_test", check=always_fire, cooldown_seconds=9999)
        mgr = AlertManager(db, rules=[rule])
        assert mgr.last_fired("lf_test") is None
        mgr.check_all()
        assert mgr.last_fired("lf_test") is not None
