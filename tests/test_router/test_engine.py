"""Tests for router engine (mocked provider)."""

import pytest

from mmrouter.classifier import ClassifierBase
from mmrouter.classifier.rules import RuleClassifier
from mmrouter.models import (
    ClassificationResult,
    CompletionResult,
    Complexity,
    Category,
    ProviderConfig,
)
from mmrouter.providers.base import ProviderBase
from mmrouter.providers.litellm_provider import ProviderError
from mmrouter.router.engine import Router
from mmrouter.tracker.logger import Tracker


class MockProvider(ProviderBase):
    """Provider that returns canned responses."""

    def __init__(self, fail_models=None):
        self._fail_models = fail_models or set()
        self.calls = []

    def complete(self, prompt, model, **kwargs):
        self.calls.append((prompt, model))
        if model in self._fail_models:
            raise ProviderError(f"{model} is down", retryable=True)
        return CompletionResult(
            content=f"Response from {model}",
            model=model,
            tokens_in=10,
            tokens_out=20,
            cost=0.001,
            latency_ms=100.0,
        )


class TestRouter:
    def test_route_simple_factual(self, tmp_path):
        provider = MockProvider()
        tracker = Tracker(tmp_path / "test.db")
        router = Router(
            "configs/default.yaml",
            provider=provider,
            tracker=tracker,
        )

        result = router.route("What is the capital of France?")

        assert result.classification.complexity == Complexity.SIMPLE
        assert result.classification.category == Category.FACTUAL
        assert "haiku" in result.model_used.lower()
        assert not result.fallback_used
        assert result.completion.content.startswith("Response from")
        router.close()

    def test_fallback_on_primary_failure(self, tmp_path):
        provider = MockProvider(fail_models={"claude-haiku-4-5-20251001"})
        tracker = Tracker(tmp_path / "test.db")
        router = Router(
            "configs/default.yaml",
            provider=provider,
            tracker=tracker,
        )

        result = router.route("What is 2+2?")

        assert result.fallback_used
        assert "sonnet" in result.model_used.lower()
        assert len(provider.calls) == 2  # tried haiku, then sonnet
        router.close()

    def test_all_models_fail(self, tmp_path):
        provider = MockProvider(
            fail_models={"claude-haiku-4-5-20251001", "claude-sonnet-4-6"}
        )
        tracker = Tracker(tmp_path / "test.db")
        router = Router(
            "configs/default.yaml",
            provider=provider,
            tracker=tracker,
        )

        with pytest.raises(RuntimeError, match="All models failed"):
            router.route("What is 2+2?")
        router.close()

    def test_request_logged(self, tmp_path):
        provider = MockProvider()
        tracker = Tracker(tmp_path / "test.db")
        router = Router(
            "configs/default.yaml",
            provider=provider,
            tracker=tracker,
        )

        router.route("What is the capital of France?")
        stats = router.get_stats()

        assert stats["total_requests"] == 1
        assert stats["total_cost"] > 0
        router.close()

    def test_classify_only(self, tmp_path):
        provider = MockProvider()
        tracker = Tracker(tmp_path / "test.db")
        router = Router(
            "configs/default.yaml",
            provider=provider,
            tracker=tracker,
        )

        result = router.classify("Write a poem about rain")

        assert result.category == Category.CREATIVE
        assert len(provider.calls) == 0  # no provider call
        router.close()

    def test_complex_code_routes_to_strong_model(self, tmp_path):
        provider = MockProvider()
        tracker = Tracker(tmp_path / "test.db")
        router = Router(
            "configs/default.yaml",
            provider=provider,
            tracker=tracker,
        )

        result = router.route(
            "Design a distributed microservice architecture for a real-time "
            "event processing system with exactly-once delivery guarantees"
        )

        assert result.classification.complexity == Complexity.COMPLEX
        assert "opus" in result.model_used.lower() or "sonnet" in result.model_used.lower()
        router.close()


class MockClassifier(ClassifierBase):
    """Classifier that returns a fixed result."""

    def __init__(self, complexity: Complexity, category: Category, confidence: float = 0.9):
        self._result = ClassificationResult(
            complexity=complexity, category=category, confidence=confidence
        )

    def classify(self, prompt: str) -> ClassificationResult:
        return self._result


class FailingMockProvider(ProviderBase):
    """Provider with configurable retryable flag per failing model."""

    def __init__(self, fail_models: dict[str, bool] | None = None):
        self._fail_models = fail_models or {}
        self.calls: list[tuple[str, str]] = []

    def complete(self, prompt, model, **kwargs):
        self.calls.append((prompt, model))
        if model in self._fail_models:
            retryable = self._fail_models[model]
            raise ProviderError(f"{model} is down", retryable=retryable)
        return CompletionResult(
            content=f"Response from {model}",
            model=model,
            tokens_in=10,
            tokens_out=20,
            cost=0.001,
            latency_ms=100.0,
        )


class TestCircuitBreakerIntegration:
    def test_circuit_opens_after_repeated_failures(self, tmp_path):
        """After 5 transient failures, haiku circuit opens and is skipped."""
        provider = FailingMockProvider(
            fail_models={"claude-haiku-4-5-20251001": True}
        )
        tracker = Tracker(tmp_path / "test.db")
        classifier = MockClassifier(Complexity.SIMPLE, Category.FACTUAL)
        router = Router(
            "configs/default.yaml",
            classifier=classifier,
            provider=provider,
            tracker=tracker,
        )

        # Route 5 times: each tries haiku (fails) then falls back to sonnet
        for _ in range(5):
            result = router.route("test")
            assert "sonnet" in result.model_used.lower()

        # After 5 transient failures, haiku circuit should be open
        # 6th call should skip haiku entirely, only call sonnet
        provider.calls.clear()
        result = router.route("test")
        assert "sonnet" in result.model_used.lower()
        assert len(provider.calls) == 1  # only sonnet, haiku skipped
        router.close()

    def test_permanent_error_doesnt_trip_circuit(self, tmp_path):
        """Permanent errors (retryable=False) don't trip the circuit breaker."""
        provider = FailingMockProvider(
            fail_models={"claude-haiku-4-5-20251001": False}
        )
        tracker = Tracker(tmp_path / "test.db")
        classifier = MockClassifier(Complexity.SIMPLE, Category.FACTUAL)
        router = Router(
            "configs/default.yaml",
            classifier=classifier,
            provider=provider,
            tracker=tracker,
        )

        # Route 10 times with permanent haiku failure
        for _ in range(10):
            result = router.route("test")
            assert "sonnet" in result.model_used.lower()

        # 11th call should still try haiku first (circuit never opened)
        provider.calls.clear()
        result = router.route("test")
        assert "sonnet" in result.model_used.lower()
        assert len(provider.calls) == 2  # tried haiku, then sonnet
        router.close()


class TestConfidenceRouting:
    def _make_router(self, tmp_path, complexity, category, confidence):
        classifier = MockClassifier(complexity, category, confidence)
        provider = MockProvider()
        tracker = Tracker(tmp_path / "test.db")
        router = Router(
            "configs/default.yaml",
            classifier=classifier,
            provider=provider,
            tracker=tracker,
        )
        return router

    def test_low_confidence_simple_escalates_to_medium(self, tmp_path):
        router = self._make_router(tmp_path, Complexity.SIMPLE, Category.FACTUAL, 0.5)
        result = router.route("test")
        assert result.escalated is True
        assert "sonnet" in result.model_used.lower()
        router.close()

    def test_low_confidence_medium_escalates_to_complex(self, tmp_path):
        router = self._make_router(tmp_path, Complexity.MEDIUM, Category.REASONING, 0.4)
        result = router.route("test")
        assert result.escalated is True
        assert "opus" in result.model_used.lower()
        router.close()

    def test_low_confidence_complex_stays_complex(self, tmp_path):
        router = self._make_router(tmp_path, Complexity.COMPLEX, Category.CODE, 0.3)
        result = router.route("test")
        assert result.escalated is False
        assert "opus" in result.model_used.lower()
        router.close()

    def test_high_confidence_no_escalation(self, tmp_path):
        router = self._make_router(tmp_path, Complexity.SIMPLE, Category.FACTUAL, 0.9)
        result = router.route("test")
        assert result.escalated is False
        assert "haiku" in result.model_used.lower()
        router.close()

    def test_confidence_at_threshold_no_escalation(self, tmp_path):
        router = self._make_router(tmp_path, Complexity.SIMPLE, Category.FACTUAL, 0.7)
        result = router.route("test")
        assert result.escalated is False
        assert "haiku" in result.model_used.lower()
        router.close()

    def test_confidence_just_below_threshold_escalates(self, tmp_path):
        router = self._make_router(tmp_path, Complexity.SIMPLE, Category.FACTUAL, 0.69)
        result = router.route("test")
        assert result.escalated is True
        assert "sonnet" in result.model_used.lower()
        router.close()
