"""Tests for experiment-aware routing integration."""

import pytest

from mmrouter.classifier import ClassifierBase
from mmrouter.models import (
    ClassificationResult,
    CompletionResult,
    Complexity,
    Category,
    Experiment,
    ExperimentStatus,
    RequestLog,
)
from mmrouter.providers.base import ProviderBase
from mmrouter.providers.litellm_provider import ProviderError
from mmrouter.router.engine import Router
from mmrouter.tracker.logger import Tracker


class MockProvider(ProviderBase):
    """Provider that returns canned responses including model name."""

    def __init__(self):
        self.calls = []

    def complete(self, prompt, model, **kwargs):
        self.calls.append((prompt, model))
        return CompletionResult(
            content=f"Response from {model}",
            model=model,
            tokens_in=10,
            tokens_out=20,
            cost=0.001,
            latency_ms=100.0,
        )


class TestExperimentRouting:
    def test_route_without_experiment(self, tmp_path):
        """Normal routing when no experiment is active."""
        provider = MockProvider()
        tracker = Tracker(tmp_path / "test.db")
        router = Router("configs/default.yaml", provider=provider, tracker=tracker)

        result = router.route("What is 2+2?")
        assert result.experiment_id is None
        assert result.variant is None
        router.close()

    def test_route_with_experiment_control(self, tmp_path):
        """Route through control config when assigned to control."""
        provider = MockProvider()
        tracker = Tracker(tmp_path / "test.db")
        router = Router("configs/default.yaml", provider=provider, tracker=tracker)

        # Create experiment: control = default, treatment = cascade
        store = router.experiment_store
        exp = store.create(Experiment(
            name="test",
            control_config="configs/default.yaml",
            treatment_config="configs/cascade.yaml",
            traffic_split=0.0,  # All traffic to control
        ))

        result = router.route("What is the capital of France?")
        assert result.experiment_id == exp.id
        assert result.variant == "control"
        router.close()

    def test_route_with_experiment_treatment(self, tmp_path):
        """Route through treatment config when assigned to treatment."""
        provider = MockProvider()
        tracker = Tracker(tmp_path / "test.db")
        router = Router("configs/default.yaml", provider=provider, tracker=tracker)

        store = router.experiment_store
        exp = store.create(Experiment(
            name="test",
            control_config="configs/default.yaml",
            treatment_config="configs/cascade.yaml",
            traffic_split=1.0,  # All traffic to treatment
        ))

        result = router.route("What is the capital of France?")
        assert result.experiment_id == exp.id
        assert result.variant == "treatment"
        router.close()

    def test_experiment_variant_logged_to_db(self, tmp_path):
        """Experiment ID and variant are stored in the requests table."""
        provider = MockProvider()
        tracker = Tracker(tmp_path / "test.db")
        router = Router("configs/default.yaml", provider=provider, tracker=tracker)

        store = router.experiment_store
        exp = store.create(Experiment(
            name="db_test",
            control_config="configs/default.yaml",
            treatment_config="configs/default.yaml",
            traffic_split=0.0,
        ))

        result = router.route("Hello")

        # Check DB directly
        cur = tracker.connection.execute(
            "SELECT experiment_id, variant FROM requests WHERE id = ?",
            (result.request_id,),
        )
        row = cur.fetchone()
        assert row["experiment_id"] == exp.id
        assert row["variant"] == "control"
        router.close()

    def test_deterministic_variant_assignment(self, tmp_path):
        """Same prompt always gets same variant."""
        provider = MockProvider()
        tracker = Tracker(tmp_path / "test.db")
        router = Router("configs/default.yaml", provider=provider, tracker=tracker)

        store = router.experiment_store
        store.create(Experiment(
            name="deterministic",
            control_config="configs/default.yaml",
            treatment_config="configs/default.yaml",
            traffic_split=0.5,
        ))

        prompt = "What is the meaning of life?"
        r1 = router.route(prompt)
        r2 = router.route(prompt)
        assert r1.variant == r2.variant
        router.close()

    def test_no_experiment_after_stop(self, tmp_path):
        """After stopping, routing returns to normal (no experiment)."""
        provider = MockProvider()
        tracker = Tracker(tmp_path / "test.db")
        router = Router("configs/default.yaml", provider=provider, tracker=tracker)

        store = router.experiment_store
        exp = store.create(Experiment(
            name="to_stop",
            control_config="configs/default.yaml",
            treatment_config="configs/default.yaml",
            traffic_split=0.5,
        ))
        store.stop(exp.id)

        result = router.route("Hello")
        assert result.experiment_id is None
        assert result.variant is None
        router.close()

    def test_experiment_uses_treatment_config_routes(self, tmp_path):
        """Treatment config with cascade enabled actually uses cascade routing."""
        provider = MockProvider()
        tracker = Tracker(tmp_path / "test.db")
        router = Router("configs/default.yaml", provider=provider, tracker=tracker)

        store = router.experiment_store
        exp = store.create(Experiment(
            name="cascade_test",
            control_config="configs/default.yaml",
            treatment_config="configs/cascade.yaml",
            traffic_split=1.0,
        ))

        # cascade.yaml has cascade enabled, so this should use cascade routing
        result = router.route("What is 2+2?")
        assert result.experiment_id == exp.id
        assert result.variant == "treatment"
        # The cascade config should trigger cascade routing
        assert result.cascade_used is True
        router.close()

    def test_null_experiment_fields_without_experiment(self, tmp_path):
        """When no experiment, DB fields are NULL."""
        provider = MockProvider()
        tracker = Tracker(tmp_path / "test.db")
        router = Router("configs/default.yaml", provider=provider, tracker=tracker)

        result = router.route("Hello")

        cur = tracker.connection.execute(
            "SELECT experiment_id, variant FROM requests WHERE id = ?",
            (result.request_id,),
        )
        row = cur.fetchone()
        assert row["experiment_id"] is None
        assert row["variant"] is None
        router.close()
