"""SQLite-backed alert incident history: one row per rule per condition."""

from __future__ import annotations

import contextlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone

from mmrouter.alerts.channels import Alert

_CREATE_INCIDENTS_TABLE = """
CREATE TABLE IF NOT EXISTS alert_incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_name TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    details TEXT NOT NULL DEFAULT '{}',
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    fire_count INTEGER NOT NULL DEFAULT 1,
    cleared_at TEXT,
    handled_at TEXT
)
"""

# At most one live (not cleared) incident per rule, enforced by the database so
# that a `mmrouter serve` process and a `mmrouter route` one-shot cannot both
# open one. Also the conflict target of the upsert below.
_CREATE_LIVE_INDEX = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_alert_incidents_live
ON alert_incidents (rule_name) WHERE cleared_at IS NULL
"""

_UPSERT_FIRE = """
INSERT INTO alert_incidents (
    rule_name, severity, message, details, first_seen, last_seen, fire_count
) VALUES (?, ?, ?, ?, ?, ?, 1)
ON CONFLICT (rule_name) WHERE cleared_at IS NULL DO UPDATE SET
    severity = excluded.severity,
    message = excluded.message,
    details = excluded.details,
    last_seen = excluded.last_seen,
    fire_count = fire_count + 1
"""

_COLUMNS = (
    "id", "rule_name", "severity", "message", "details",
    "first_seen", "last_seen", "fire_count", "cleared_at", "handled_at",
)
_SELECT = f"SELECT {', '.join(_COLUMNS)} FROM alert_incidents"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row) -> dict:
    item = dict(zip(_COLUMNS, row))
    try:
        item["details"] = json.loads(item["details"])
    except (TypeError, ValueError):
        item["details"] = {}
    return item


class AlertIncidentStore:
    """Durable alert history. Owns the alert_incidents table."""

    def __init__(self, conn: sqlite3.Connection, *, lock=None):
        self._conn = conn
        self._lock = lock or contextlib.nullcontext()
        with self._lock:
            self._conn.execute(_CREATE_INCIDENTS_TABLE)
            self._conn.execute(_CREATE_LIVE_INDEX)
            self._conn.commit()

    def record_fire(self, alert: Alert) -> None:
        """Open a live incident for this rule, or extend the existing one."""
        now = _now_iso()
        with self._lock:
            self._conn.execute(
                _UPSERT_FIRE,
                (
                    alert.rule_name,
                    alert.severity,
                    alert.message,
                    json.dumps(alert.details),
                    now,
                    now,
                ),
            )
            self._conn.commit()

    def clear(self, rule_name: str, min_age_seconds: float = 300.0) -> bool:
        """Close the live incident for this rule. Returns True if one was closed.

        Two guards, both load-bearing:

        - It SELECTs before it UPDATEs. A silent evaluation happens on every
          routed request for every configured rule, and an unconditional UPDATE
          takes a write transaction whether or not it matches anything. That
          would put N write locks per request on the steady, nothing-is-wrong
          path and is where `database is locked` would surface first.
        - It refuses to close an incident younger than `min_age_seconds`.
          Cooldown is per process (`rules.py:231-234`), so a process that just
          fired stops evaluating that rule for its whole cooldown. A silent
          evaluation from a SECOND process inside that window is not evidence
          the condition ended -- the authoritative observer is muted. Without
          this guard a flapping rule plus mixed serve/CLI traffic closes and
          reopens an incident on every oscillation, producing exactly the wall
          of rows this ticket exists to remove.
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(seconds=min_age_seconds)
        ).isoformat()
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM alert_incidents "
                "WHERE rule_name = ? AND cleared_at IS NULL AND last_seen <= ? LIMIT 1",
                (rule_name, cutoff),
            ).fetchone()
            if row is None:
                return False
            self._conn.execute(
                "UPDATE alert_incidents SET cleared_at = ? WHERE id = ? AND cleared_at IS NULL",
                (_now_iso(), row[0]),
            )
            self._conn.commit()
            return True

    def list_incidents(self, limit: int = 200) -> list[dict]:
        """Incidents by most recent activity first."""
        rows = self._conn.execute(
            _SELECT + " ORDER BY last_seen DESC, id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def counts(self) -> dict:
        """Totals, unaffected by the list limit. Open means unhandled."""
        row = self._conn.execute(
            """SELECT COUNT(*),
                      COALESCE(SUM(handled_at IS NULL), 0),
                      COALESCE(SUM(handled_at IS NOT NULL), 0)
               FROM alert_incidents"""
        ).fetchone()
        return {"total": row[0], "open_count": row[1], "handled_count": row[2]}

    def set_handled(self, incident_id: int, handled: bool) -> dict | None:
        """Mark or unmark. Returns the updated row, or None for an unknown id."""
        with self._lock:
            self._conn.execute(
                "UPDATE alert_incidents SET handled_at = ? WHERE id = ?",
                (_now_iso() if handled else None, incident_id),
            )
            self._conn.commit()
        row = self._conn.execute(_SELECT + " WHERE id = ?", (incident_id,)).fetchone()
        return _row_to_dict(row) if row is not None else None
