"""Tests for ExperimentStore (SQLite CRUD)."""

import pytest

from mmrouter.experiments.store import ExperimentStore
from mmrouter.models import Experiment, ExperimentStatus
from mmrouter.tracker.logger import Tracker


@pytest.fixture
def store(tmp_path):
    tracker = Tracker(tmp_path / "test.db")
    s = ExperimentStore(tracker.connection)
    yield s
    tracker.close()


@pytest.fixture
def tracker_and_store(tmp_path):
    tracker = Tracker(tmp_path / "test.db")
    s = ExperimentStore(tracker.connection)
    yield tracker, s
    tracker.close()


class TestExperimentStore:
    def test_create_experiment(self, store):
        exp = store.create(Experiment(
            name="test_exp",
            control_config="/path/to/control.yaml",
            treatment_config="/path/to/treatment.yaml",
            traffic_split=0.5,
        ))
        assert exp.id is not None
        assert exp.id > 0
        assert exp.name == "test_exp"
        assert exp.status == ExperimentStatus.ACTIVE

    def test_get_active(self, store):
        store.create(Experiment(
            name="active_exp",
            control_config="/a.yaml",
            treatment_config="/b.yaml",
        ))
        active = store.get_active()
        assert active is not None
        assert active.name == "active_exp"

    def test_no_active(self, store):
        assert store.get_active() is None

    def test_only_one_active(self, store):
        store.create(Experiment(
            name="first",
            control_config="/a.yaml",
            treatment_config="/b.yaml",
        ))
        with pytest.raises(ValueError, match="already active"):
            store.create(Experiment(
                name="second",
                control_config="/c.yaml",
                treatment_config="/d.yaml",
            ))

    def test_get_by_id(self, store):
        exp = store.create(Experiment(
            name="findme",
            control_config="/a.yaml",
            treatment_config="/b.yaml",
        ))
        found = store.get(exp.id)
        assert found is not None
        assert found.name == "findme"

    def test_get_nonexistent(self, store):
        assert store.get(999) is None

    def test_list_all(self, store):
        exp1 = store.create(Experiment(
            name="first",
            control_config="/a.yaml",
            treatment_config="/b.yaml",
        ))
        store.stop(exp1.id)

        store.create(Experiment(
            name="second",
            control_config="/c.yaml",
            treatment_config="/d.yaml",
        ))

        all_exps = store.list_all()
        assert len(all_exps) == 2

    def test_stop_experiment(self, store):
        exp = store.create(Experiment(
            name="to_stop",
            control_config="/a.yaml",
            treatment_config="/b.yaml",
        ))
        stopped = store.stop(exp.id)
        assert stopped.status == ExperimentStatus.STOPPED
        assert stopped.stopped_at is not None

        # Verify persisted
        fetched = store.get(exp.id)
        assert fetched.status == ExperimentStatus.STOPPED

    def test_stop_nonexistent(self, store):
        with pytest.raises(ValueError, match="not found"):
            store.stop(999)

    def test_stop_already_stopped(self, store):
        exp = store.create(Experiment(
            name="already",
            control_config="/a.yaml",
            treatment_config="/b.yaml",
        ))
        store.stop(exp.id)
        with pytest.raises(ValueError, match="not active"):
            store.stop(exp.id)

    def test_stop_active_convenience(self, store):
        store.create(Experiment(
            name="active",
            control_config="/a.yaml",
            treatment_config="/b.yaml",
        ))
        stopped = store.stop_active()
        assert stopped is not None
        assert stopped.status == ExperimentStatus.STOPPED

    def test_stop_active_when_none(self, store):
        assert store.stop_active() is None

    def test_create_after_stop(self, store):
        exp1 = store.create(Experiment(
            name="first",
            control_config="/a.yaml",
            treatment_config="/b.yaml",
        ))
        store.stop(exp1.id)

        exp2 = store.create(Experiment(
            name="second",
            control_config="/c.yaml",
            treatment_config="/d.yaml",
        ))
        assert exp2.id is not None
        assert exp2.name == "second"
        assert exp2.status == ExperimentStatus.ACTIVE

    def test_traffic_split_stored(self, store):
        exp = store.create(Experiment(
            name="split_test",
            control_config="/a.yaml",
            treatment_config="/b.yaml",
            traffic_split=0.3,
        ))
        fetched = store.get(exp.id)
        assert fetched.traffic_split == 0.3
