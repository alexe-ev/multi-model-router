"""Alert rules engine: evaluate conditions, manage cooldowns, fire alerts."""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable

from mmrouter.alerts.channels import Alert, LogChannel, WebhookChannel
from mmrouter.alerts.store import AlertIncidentStore

logger = logging.getLogger("mmrouter.alerts")


def _utc_iso(dt: datetime) -> str:
    """Format a UTC datetime as ISO string matching SQLite's stored format."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class AlertRule:
    """A single alert rule with condition and cooldown."""

    name: str
    check: Callable[[sqlite3.Connection], Alert | None]
    cooldown_seconds: int = 300
    severity: str = "warning"


def _check_cost_spike(conn: sqlite3.Connection) -> Alert | None:
    """Fire if current hour's cost > 2x the average hourly cost."""
    now = datetime.now(timezone.utc)
    one_hour_ago = _utc_iso(now - timedelta(hours=1))
    twenty_five_hours_ago = _utc_iso(now - timedelta(hours=25))

    cur = conn.execute(
        "SELECT COALESCE(SUM(cost), 0) FROM requests WHERE timestamp >= ?",
        (one_hour_ago,),
    )
    current_hour_cost = cur.fetchone()[0]

    if current_hour_cost == 0:
        return None

    # Average hourly cost over last 24h (excluding current hour)
    cur = conn.execute(
        """SELECT COALESCE(SUM(cost), 0),
                  COUNT(DISTINCT strftime('%Y-%m-%d %H', timestamp))
           FROM requests
           WHERE timestamp >= ? AND timestamp < ?""",
        (twenty_five_hours_ago, one_hour_ago),
    )
    row = cur.fetchone()
    total_cost, hour_count = row[0], row[1]

    if hour_count == 0:
        return None

    avg_hourly = total_cost / hour_count

    if avg_hourly > 0 and current_hour_cost > 2 * avg_hourly:
        return Alert(
            rule_name="cost_spike",
            message=f"Hourly cost ${current_hour_cost:.4f} exceeds 2x average ${avg_hourly:.4f}",
            severity="warning",
            details={
                "current_hour_cost": round(current_hour_cost, 6),
                "avg_hourly_cost": round(avg_hourly, 6),
                "multiplier": round(current_hour_cost / avg_hourly, 2),
            },
        )
    return None


def _check_error_rate(conn: sqlite3.Connection) -> Alert | None:
    """Fire if >10% of last 100 requests used fallback (indicates primary failures)."""
    cur = conn.execute("""
        SELECT COUNT(*) as total,
               COALESCE(SUM(fallback_used), 0) as failures
        FROM (
            SELECT fallback_used FROM requests ORDER BY id DESC LIMIT 100
        )
    """)
    row = cur.fetchone()
    total, failures = row[0], row[1]

    if total < 10:
        return None

    rate = failures / total
    if rate > 0.10:
        return Alert(
            rule_name="error_rate",
            message=f"Fallback rate {rate:.0%} in last {total} requests (threshold: 10%)",
            severity="critical",
            details={
                "total_checked": total,
                "failures": failures,
                "rate": round(rate, 4),
            },
        )
    return None


def _check_budget_warning(conn: sqlite3.Connection) -> Alert | None:
    """Placeholder: actual impl uses factory below."""
    return None


def create_budget_warning_rule(
    daily_limit: float, warn_pct: float = 0.90, cooldown: int = 300
) -> AlertRule:
    """Create a budget warning rule with the actual daily limit."""

    def check(conn: sqlite3.Connection) -> Alert | None:
        if daily_limit <= 0:
            return None
        cur = conn.execute(
            "SELECT COALESCE(SUM(cost), 0) FROM requests WHERE date(timestamp) = date('now')"
        )
        spent = cur.fetchone()[0]
        ratio = spent / daily_limit

        if ratio >= warn_pct:
            return Alert(
                rule_name="budget_warning",
                message=f"Budget usage {ratio:.0%}: ${spent:.4f} of ${daily_limit:.2f} daily limit",
                severity="critical" if ratio >= 1.0 else "warning",
                details={
                    "spent_today": round(spent, 6),
                    "daily_limit": daily_limit,
                    "usage_pct": round(ratio * 100, 1),
                },
            )
        return None

    return AlertRule(
        name="budget_warning",
        check=check,
        cooldown_seconds=cooldown,
        severity="warning",
    )


# Built-in rules (cost_spike and error_rate don't need external config)
BUILTIN_RULES: dict[str, AlertRule] = {
    "cost_spike": AlertRule(
        name="cost_spike",
        check=_check_cost_spike,
        cooldown_seconds=300,
        severity="warning",
    ),
    "error_rate": AlertRule(
        name="error_rate",
        check=_check_error_rate,
        cooldown_seconds=300,
        severity="critical",
    ),
}


class AlertManager:
    """Evaluates alert rules, manages cooldowns, dispatches to channels."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        rules: list[AlertRule] | None = None,
        webhook_url: str | None = None,
        cooldown_seconds: int = 300,
        write_lock=None,
    ):
        self._conn = conn
        self._rules = rules or []
        self._cooldown_override = cooldown_seconds
        self._last_fired: dict[str, float] = {}
        self._write_lock = write_lock
        # AlertManager was single-threaded before this ticket; the streaming path
        # now runs check_all from a different thread than non-streamed requests
        # do, so its own in-process state needs its own guard. Never the write
        # lock, which must not be held across a webhook send.
        self._state_lock = threading.Lock()
        # History is best-effort from the first statement on. Router.__init__ builds
        # this manager OUTSIDE any try (engine.py:93-108), so a raise here would take
        # down routing itself — the one thing alerting is required never to do.
        self._store: AlertIncidentStore | None = None
        try:
            self._store = AlertIncidentStore(conn, lock=self._write_lock)
        except Exception:
            logger.warning(
                "Alert history unavailable; alerts will fire but will not be recorded",
                exc_info=True,
            )

        # Channels
        self._log_channel = LogChannel()
        self._webhook: WebhookChannel | None = None
        if webhook_url:
            self._webhook = WebhookChannel(webhook_url)

        # Apply cooldown override to rules that use default
        for rule in self._rules:
            if rule.cooldown_seconds == 300 and cooldown_seconds != 300:
                rule.cooldown_seconds = cooldown_seconds

    @property
    def rules(self) -> list[AlertRule]:
        return list(self._rules)

    @property
    def webhook(self) -> WebhookChannel | None:
        return self._webhook

    def last_fired(self, rule_name: str) -> float | None:
        """Return monotonic timestamp of last fire, or None."""
        return self._last_fired.get(rule_name)

    def is_in_cooldown(self, rule_name: str) -> bool:
        """Check if a rule is currently in cooldown."""
        last = self._last_fired.get(rule_name)
        if last is None:
            return False
        rule = next((r for r in self._rules if r.name == rule_name), None)
        if rule is None:
            return False
        return (time.monotonic() - last) < rule.cooldown_seconds

    def check_all(self) -> list[Alert]:
        """Evaluate all rules, fire alerts for those that trigger. Returns fired alerts."""
        fired: list[Alert] = []
        now = time.monotonic()

        for rule in self._rules:
            # Cooldown check (unlocked fast path -- avoids the rule's SQL for a
            # rule that is clearly still in cooldown).
            last = self._last_fired.get(rule.name)
            if last is not None and (now - last) < rule.cooldown_seconds:
                continue

            alert = rule.check(self._conn)
            if alert is None:
                # Evaluated and silent: this closes the live incident, if any.
                # The rule's own cooldown is the grace window — see store.clear.
                self._record_clear(rule.name, rule.cooldown_seconds)
                continue

            # Fire -- the authoritative check-and-set. Two threads racing the
            # unlocked check above can both reach here; only the one that wins
            # the lock may set last_fired and proceed, or both would fire,
            # sending the webhook twice and double-incrementing fire_count.
            with self._state_lock:
                last = self._last_fired.get(rule.name)
                if last is not None and (now - last) < rule.cooldown_seconds:
                    continue
                self._last_fired[rule.name] = now

            self._log_channel.send(alert)
            if self._webhook:
                self._webhook.send(alert)
            fired.append(alert)
            # After the channels: history must never cost a delivery.
            self._record_fire(alert)

        return fired

    def _ensure_store(self) -> AlertIncidentStore | None:
        """Re-arm history if the DDL failed at construction. Called only from the
        fire path: a transient lock at startup must not disable history for the
        whole life of a long-running server, but retrying on every silent
        evaluation would re-run DDL on the steady path."""
        with self._state_lock:
            if self._store is None:
                try:
                    self._store = AlertIncidentStore(self._conn, lock=self._write_lock)
                except Exception:
                    return None
            return self._store

    def _record_fire(self, alert: Alert) -> None:
        store = self._ensure_store()
        if store is None:
            return
        try:
            store.record_fire(alert)
        except Exception:
            logger.warning(
                "Alert history write failed for rule %s", alert.rule_name, exc_info=True
            )

    def _record_clear(self, rule_name: str, min_age_seconds: float) -> None:
        if self._store is None:
            return
        try:
            self._store.clear(rule_name, min_age_seconds)
        except Exception:
            logger.warning(
                "Alert history clear failed for rule %s", rule_name, exc_info=True
            )

    def get_status(self) -> dict:
        """Status for CLI display."""
        now = time.monotonic()
        rules_status = []
        for rule in self._rules:
            last = self._last_fired.get(rule.name)
            in_cooldown = False
            cooldown_remaining = 0
            if last is not None:
                elapsed = now - last
                if elapsed < rule.cooldown_seconds:
                    in_cooldown = True
                    cooldown_remaining = int(rule.cooldown_seconds - elapsed)

            rules_status.append({
                "name": rule.name,
                "severity": rule.severity,
                "cooldown_seconds": rule.cooldown_seconds,
                "in_cooldown": in_cooldown,
                "cooldown_remaining": cooldown_remaining,
            })

        return {
            "rules": rules_status,
            "webhook_configured": self._webhook is not None,
            "webhook_url": self._webhook.url if self._webhook else None,
        }
