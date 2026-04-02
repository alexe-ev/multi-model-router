"""Tests for router engine (mocked provider)."""

import pytest

from mmrouter.classifier.rules import RuleClassifier
from mmrouter.models import CompletionResult, Complexity, Category, ProviderConfig
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
