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


class CascadeMockProvider(ProviderBase):
    """Provider with per-model response content for cascade testing."""

    def __init__(self, responses: dict[str, str] | None = None, fail_models: set | None = None):
        self._responses = responses or {}
        self._fail_models = fail_models or set()
        self.calls: list[tuple[str, str]] = []

    def complete(self, prompt, model, **kwargs):
        self.calls.append((prompt, model))
        if model in self._fail_models:
            raise ProviderError(f"{model} is down", retryable=True)
        content = self._responses.get(model, f"Response from {model}")
        return CompletionResult(
            content=content,
            model=model,
            tokens_in=10,
            tokens_out=len(content),
            cost=0.001,
            latency_ms=100.0,
        )


def _write_cascade_config(tmp_path, *, enabled=True, strategy="heuristic",
                          min_response_length=50, per_route_cascade=None):
    """Write a cascade-enabled YAML config to tmp_path and return the path."""
    cascade_section = ""
    if per_route_cascade:
        cascade_lines = "\n".join(f"        - {m}" for m in per_route_cascade)
        cascade_section = f"\n      cascade:\n{cascade_lines}"

    cfg = tmp_path / "cascade_test.yaml"
    cfg.write_text(f"""
version: "1"

routes:
  simple:
    factual:
      model: claude-haiku-4-5-20251001
      fallbacks:
        - claude-sonnet-4-6{cascade_section}
    reasoning:
      model: claude-haiku-4-5-20251001
      fallbacks:
        - claude-sonnet-4-6
    creative:
      model: claude-sonnet-4-6
      fallbacks:
        - claude-haiku-4-5-20251001
    code:
      model: claude-sonnet-4-6
      fallbacks:
        - claude-haiku-4-5-20251001

  medium:
    factual:
      model: claude-sonnet-4-6
      fallbacks:
        - claude-haiku-4-5-20251001
    reasoning:
      model: claude-sonnet-4-6
      fallbacks:
        - claude-opus-4-6
    creative:
      model: claude-sonnet-4-6
      fallbacks:
        - claude-opus-4-6
    code:
      model: claude-sonnet-4-6
      fallbacks:
        - claude-opus-4-6

  complex:
    factual:
      model: claude-sonnet-4-6
      fallbacks:
        - claude-opus-4-6
    reasoning:
      model: claude-opus-4-6
      fallbacks:
        - claude-sonnet-4-6
    creative:
      model: claude-opus-4-6
      fallbacks:
        - claude-sonnet-4-6
    code:
      model: claude-opus-4-6
      fallbacks:
        - claude-sonnet-4-6

classifier:
  strategy: rules
  threshold: "0.7"

provider:
  timeout_ms: 30000
  max_retries: 2
  circuit_breaker_threshold: 5
  circuit_breaker_reset_ms: 60000

cascade:
  enabled: {str(enabled).lower()}
  strategy: {strategy}
  min_response_length: {min_response_length}
""")
    return cfg


class TestCascadeRouting:
    def test_cascade_disabled_zero_behavior_change(self, tmp_path):
        """With cascade disabled, routing is identical to standard behavior."""
        cfg = _write_cascade_config(tmp_path, enabled=False)
        provider = MockProvider()
        tracker = Tracker(tmp_path / "test.db")
        classifier = MockClassifier(Complexity.SIMPLE, Category.FACTUAL)
        router = Router(str(cfg), classifier=classifier, provider=provider, tracker=tracker)

        result = router.route("test")

        assert not result.cascade_used
        assert result.cascade_attempts == 1
        assert "haiku" in result.model_used.lower()
        router.close()

    def test_cascade_short_response_escalates(self, tmp_path):
        """Short response from cheap model triggers escalation."""
        cfg = _write_cascade_config(tmp_path, min_response_length=50)
        provider = CascadeMockProvider(responses={
            "claude-haiku-4-5-20251001": "Short.",
            "claude-sonnet-4-6": "A" * 100,
        })
        tracker = Tracker(tmp_path / "test.db")
        classifier = MockClassifier(Complexity.SIMPLE, Category.FACTUAL)
        router = Router(str(cfg), classifier=classifier, provider=provider, tracker=tracker)

        result = router.route("test")

        assert result.cascade_used
        assert result.cascade_attempts == 2
        assert "sonnet" in result.model_used.lower()
        assert len(provider.calls) == 2
        router.close()

    def test_cascade_good_cheap_response_stops_early(self, tmp_path):
        """Good response from cheap model stops cascade after first model."""
        cfg = _write_cascade_config(tmp_path, min_response_length=50)
        provider = CascadeMockProvider(responses={
            "claude-haiku-4-5-20251001": "A" * 200,
            "claude-sonnet-4-6": "B" * 200,
        })
        tracker = Tracker(tmp_path / "test.db")
        classifier = MockClassifier(Complexity.SIMPLE, Category.FACTUAL)
        router = Router(str(cfg), classifier=classifier, provider=provider, tracker=tracker)

        result = router.route("test")

        assert result.cascade_used
        assert result.cascade_attempts == 1
        assert "haiku" in result.model_used.lower()
        assert len(provider.calls) == 1
        router.close()

    def test_cascade_hedging_detection(self, tmp_path):
        """Response containing hedging phrase triggers escalation."""
        cfg = _write_cascade_config(tmp_path, min_response_length=10)
        provider = CascadeMockProvider(responses={
            "claude-haiku-4-5-20251001": "I'm not sure about this, but I think the answer might be 42.",
            "claude-sonnet-4-6": "The answer is 42, this is well established.",
        })
        tracker = Tracker(tmp_path / "test.db")
        classifier = MockClassifier(Complexity.SIMPLE, Category.FACTUAL)
        router = Router(str(cfg), classifier=classifier, provider=provider, tracker=tracker)

        result = router.route("test")

        assert result.cascade_used
        assert result.cascade_attempts == 2
        assert "sonnet" in result.model_used.lower()
        router.close()

    def test_cascade_per_route_chain(self, tmp_path):
        """Per-route cascade chain overrides default ordering."""
        cfg = _write_cascade_config(
            tmp_path,
            min_response_length=50,
            per_route_cascade=[
                "claude-sonnet-4-6",
                "claude-haiku-4-5-20251001",
            ],
        )
        provider = CascadeMockProvider(responses={
            "claude-sonnet-4-6": "Short.",
            "claude-haiku-4-5-20251001": "A" * 200,
        })
        tracker = Tracker(tmp_path / "test.db")
        classifier = MockClassifier(Complexity.SIMPLE, Category.FACTUAL)
        router = Router(str(cfg), classifier=classifier, provider=provider, tracker=tracker)

        result = router.route("test")

        # Should try sonnet first (per cascade chain), then haiku
        assert result.cascade_used
        assert result.cascade_attempts == 2
        assert "haiku" in result.model_used.lower()
        assert provider.calls[0][1] == "claude-sonnet-4-6"
        assert provider.calls[1][1] == "claude-haiku-4-5-20251001"
        router.close()

    def test_cascade_default_chain_from_route(self, tmp_path):
        """Without per-route cascade, chain is derived from primary + fallbacks."""
        cfg = _write_cascade_config(tmp_path, min_response_length=50)
        provider = CascadeMockProvider(responses={
            "claude-haiku-4-5-20251001": "Short.",
            "claude-sonnet-4-6": "A" * 200,
        })
        tracker = Tracker(tmp_path / "test.db")
        classifier = MockClassifier(Complexity.SIMPLE, Category.FACTUAL)
        router = Router(str(cfg), classifier=classifier, provider=provider, tracker=tracker)

        result = router.route("test")

        # Default chain: haiku (primary), then sonnet (fallback)
        assert provider.calls[0][1] == "claude-haiku-4-5-20251001"
        assert provider.calls[1][1] == "claude-sonnet-4-6"
        router.close()

    def test_cascade_circuit_breaker_skips_open(self, tmp_path):
        """Cascade skips models with open circuit breakers."""
        cfg = _write_cascade_config(tmp_path, min_response_length=50)
        fail_provider = CascadeMockProvider(
            responses={"claude-sonnet-4-6": "A" * 200},
            fail_models={"claude-haiku-4-5-20251001"},
        )
        tracker = Tracker(tmp_path / "test.db")
        classifier = MockClassifier(Complexity.SIMPLE, Category.FACTUAL)
        router = Router(str(cfg), classifier=classifier, provider=fail_provider, tracker=tracker)

        # Trip the circuit breaker by failing 5 times
        for _ in range(5):
            result = router.route("test")
            assert "sonnet" in result.model_used.lower()

        # Now haiku circuit is open. Next cascade should skip it.
        fail_provider.calls.clear()
        result = router.route("test")

        assert result.cascade_used
        assert "sonnet" in result.model_used.lower()
        assert len(fail_provider.calls) == 1  # haiku skipped
        router.close()

    def test_cascade_provider_error_tries_next(self, tmp_path):
        """ProviderError in cascade skips to next model (not fatal)."""
        cfg = _write_cascade_config(tmp_path, min_response_length=10)
        provider = CascadeMockProvider(
            responses={"claude-sonnet-4-6": "A" * 200},
            fail_models={"claude-haiku-4-5-20251001"},
        )
        tracker = Tracker(tmp_path / "test.db")
        classifier = MockClassifier(Complexity.SIMPLE, Category.FACTUAL)
        router = Router(str(cfg), classifier=classifier, provider=provider, tracker=tracker)

        result = router.route("test")

        assert result.cascade_used
        assert "sonnet" in result.model_used.lower()
        router.close()

    def test_cascade_logging(self, tmp_path):
        """Tracker logs cascade_used and cascade_attempts correctly."""
        cfg = _write_cascade_config(tmp_path, min_response_length=50)
        provider = CascadeMockProvider(responses={
            "claude-haiku-4-5-20251001": "Short.",
            "claude-sonnet-4-6": "A" * 100,
        })
        tracker = Tracker(tmp_path / "test.db")
        classifier = MockClassifier(Complexity.SIMPLE, Category.FACTUAL)
        router = Router(str(cfg), classifier=classifier, provider=provider, tracker=tracker)

        router.route("test")

        # Check the logged row
        cur = tracker.connection.execute(
            "SELECT cascade_used, cascade_attempts FROM requests ORDER BY id DESC LIMIT 1"
        )
        row = cur.fetchone()
        assert row["cascade_used"] == 1
        assert row["cascade_attempts"] == 2
        router.close()

    def test_cascade_all_fail_quality_returns_last(self, tmp_path):
        """When all models fail quality gate, return last response (best effort)."""
        cfg = _write_cascade_config(tmp_path, min_response_length=500)
        provider = CascadeMockProvider(responses={
            "claude-haiku-4-5-20251001": "Short response from haiku.",
            "claude-sonnet-4-6": "Short response from sonnet.",
        })
        tracker = Tracker(tmp_path / "test.db")
        classifier = MockClassifier(Complexity.SIMPLE, Category.FACTUAL)
        router = Router(str(cfg), classifier=classifier, provider=provider, tracker=tracker)

        result = router.route("test")

        # Should return last model's response as best effort
        assert result.cascade_used
        assert result.cascade_attempts == 2
        assert "sonnet" in result.model_used.lower()
        router.close()

    def test_cascade_all_providers_unavailable(self, tmp_path):
        """When all models in cascade are unavailable, raise RuntimeError."""
        cfg = _write_cascade_config(tmp_path, min_response_length=10)
        provider = CascadeMockProvider(
            fail_models={"claude-haiku-4-5-20251001", "claude-sonnet-4-6"},
        )
        tracker = Tracker(tmp_path / "test.db")
        classifier = MockClassifier(Complexity.SIMPLE, Category.FACTUAL)
        router = Router(str(cfg), classifier=classifier, provider=provider, tracker=tracker)

        with pytest.raises(RuntimeError, match="Cascade failed"):
            router.route("test")
        router.close()

    def test_cascade_result_fields(self, tmp_path):
        """Verify RoutingResult has correct cascade metadata."""
        cfg = _write_cascade_config(tmp_path, min_response_length=50)
        provider = CascadeMockProvider(responses={
            "claude-haiku-4-5-20251001": "A" * 100,
        })
        tracker = Tracker(tmp_path / "test.db")
        classifier = MockClassifier(Complexity.SIMPLE, Category.FACTUAL)
        router = Router(str(cfg), classifier=classifier, provider=provider, tracker=tracker)

        result = router.route("test")

        assert result.cascade_used is True
        assert result.cascade_attempts == 1
        assert result.fallback_used is False
        router.close()

    def test_non_cascade_result_has_default_cascade_fields(self, tmp_path):
        """Standard (non-cascade) routing result has cascade_used=False."""
        cfg = _write_cascade_config(tmp_path, enabled=False)
        provider = MockProvider()
        tracker = Tracker(tmp_path / "test.db")
        classifier = MockClassifier(Complexity.SIMPLE, Category.FACTUAL)
        router = Router(str(cfg), classifier=classifier, provider=provider, tracker=tracker)

        result = router.route("test")

        assert result.cascade_used is False
        assert result.cascade_attempts == 1
        router.close()
