"""SQLite-backed experiment storage. Max 1 active experiment at a time."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from mmrouter.models import Experiment, ExperimentStatus

_CREATE_EXPERIMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    control_config TEXT NOT NULL,
    treatment_config TEXT NOT NULL,
    traffic_split REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL,
    stopped_at TEXT
)
"""


class ExperimentStore:
    """CRUD for experiments, backed by SQLite."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._conn.execute(_CREATE_EXPERIMENTS_TABLE)
        self._conn.commit()

    def create(self, experiment: Experiment) -> Experiment:
        """Create an experiment. Raises ValueError if one is already active."""
        active = self.get_active()
        if active is not None:
            raise ValueError(
                f"Experiment '{active.name}' (id={active.id}) is already active. "
                f"Stop it before creating a new one."
            )

        cur = self._conn.execute(
            """INSERT INTO experiments (name, status, control_config, treatment_config,
               traffic_split, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                experiment.name,
                experiment.status.value,
                experiment.control_config,
                experiment.treatment_config,
                experiment.traffic_split,
                experiment.created_at.isoformat(),
            ),
        )
        self._conn.commit()
        experiment.id = cur.lastrowid
        return experiment

    def get_active(self) -> Experiment | None:
        """Return the active experiment, or None."""
        cur = self._conn.execute(
            "SELECT * FROM experiments WHERE status = ? LIMIT 1",
            (ExperimentStatus.ACTIVE.value,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return self._row_to_experiment(row)

    def get(self, experiment_id: int) -> Experiment | None:
        """Get experiment by ID."""
        cur = self._conn.execute(
            "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
        )
        row = cur.fetchone()
        if not row:
            return None
        return self._row_to_experiment(row)

    def list_all(self) -> list[Experiment]:
        """List all experiments, newest first."""
        cur = self._conn.execute(
            "SELECT * FROM experiments ORDER BY created_at DESC"
        )
        return [self._row_to_experiment(row) for row in cur.fetchall()]

    def stop(self, experiment_id: int) -> Experiment:
        """Stop an active experiment."""
        exp = self.get(experiment_id)
        if exp is None:
            raise ValueError(f"Experiment {experiment_id} not found")
        if exp.status != ExperimentStatus.ACTIVE:
            raise ValueError(
                f"Experiment {experiment_id} is not active (status={exp.status})"
            )
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE experiments SET status = ?, stopped_at = ? WHERE id = ?",
            (ExperimentStatus.STOPPED.value, now, experiment_id),
        )
        self._conn.commit()
        exp.status = ExperimentStatus.STOPPED
        exp.stopped_at = datetime.now(timezone.utc)
        return exp

    def stop_active(self) -> Experiment | None:
        """Stop whatever experiment is currently active. Returns it, or None."""
        active = self.get_active()
        if active is None:
            return None
        return self.stop(active.id)

    def _row_to_experiment(self, row) -> Experiment:
        stopped_at = None
        if row["stopped_at"]:
            stopped_at = datetime.fromisoformat(row["stopped_at"])
        return Experiment(
            id=row["id"],
            name=row["name"],
            status=ExperimentStatus(row["status"]),
            control_config=row["control_config"],
            treatment_config=row["treatment_config"],
            traffic_split=row["traffic_split"],
            created_at=datetime.fromisoformat(row["created_at"]),
            stopped_at=stopped_at,
        )
