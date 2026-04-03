"""Tests for Router.route_messages and route_messages_stream."""

from typing import Iterator

import pytest

from mmrouter.classifier import ClassifierBase
from mmrouter.models import (
    ClassificationResult,
    CompletionResult,
    Complexity,
    Category,
    StreamChunk,
)
from mmrouter.providers.base import ProviderBase
from mmrouter.providers.litellm_provider import ProviderError
from mmrouter.router.engine import Router
from mmrouter.tracker.logger import Tracker


class MockProvider(ProviderBase):
    def __init__(self, fail_models=None):
        self._fail_models = fail_models or set()
        self.calls: list[tuple] = []

    def complete(self, prompt, model, **kwargs):
        self.calls.append(("complete", prompt, model))
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

    def complete_messages(self, messages, model, **kwargs):
        self.calls.append(("complete_messages", messages, model))
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

    def stream_messages(self, messages, model, **kwargs) -> Iterator[StreamChunk]:
        self.calls.append(("stream_messages", messages, model))
        if model in self._fail_models:
            raise ProviderError(f"{model} is down", retryable=True)
        yield StreamChunk(content="Hello ", model=model)
        yield StreamChunk(content="world!", model=model, finish_reason="stop")


class MockClassifier(ClassifierBase):
    def __init__(self, complexity: Complexity, category: Category, confidence: float = 0.9):
        self._result = ClassificationResult(
            complexity=complexity, category=category, confidence=confidence
        )

    def classify(self, prompt: str) -> ClassificationResult:
        return self._result


class TestRouteMessages:
    def test_routes_based_on_last_user_message(self, tmp_path):
        provider = MockProvider()
        tracker = Tracker(tmp_path / "test.db")
        router = Router(
            "configs/default.yaml",
            provider=provider,
            tracker=tracker,
        )

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is the capital of France?"},
        ]
        result = router.route_messages(messages)

        assert result.classification.complexity == Complexity.SIMPLE
        assert result.classification.category == Category.FACTUAL
        assert "haiku" in result.model_used.lower()
        router.close()

    def test_system_messages_passed_to_provider(self, tmp_path):
        provider = MockProvider()
        tracker = Tracker(tmp_path / "test.db")
        router = Router(
            "configs/default.yaml",
            provider=provider,
            tracker=tracker,
        )

        messages = [
            {"role": "system", "content": "You are a poet."},
            {"role": "user", "content": "What is the capital of France?"},
        ]
        router.route_messages(messages)

        # Provider should receive the full messages array
        last_call = provider.calls[-1]
        assert last_call[0] == "complete_messages"
        assert len(last_call[1]) == 2
        assert last_call[1][0]["role"] == "system"
        router.close()

    def test_fallback_on_primary_failure(self, tmp_path):
        provider = MockProvider(fail_models={"claude-haiku-4-5-20251001"})
        tracker = Tracker(tmp_path / "test.db")
        router = Router(
            "configs/default.yaml",
            provider=provider,
            tracker=tracker,
        )

        messages = [{"role": "user", "content": "What is 2+2?"}]
        result = router.route_messages(messages)

        assert result.fallback_used
        assert "sonnet" in result.model_used.lower()
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

        messages = [{"role": "user", "content": "What is 2+2?"}]
        with pytest.raises(RuntimeError, match="All models failed"):
            router.route_messages(messages)
        router.close()

    def test_low_confidence_escalation(self, tmp_path):
        classifier = MockClassifier(Complexity.SIMPLE, Category.FACTUAL, 0.5)
        provider = MockProvider()
        tracker = Tracker(tmp_path / "test.db")
        router = Router(
            "configs/default.yaml",
            classifier=classifier,
            provider=provider,
            tracker=tracker,
        )

        messages = [{"role": "user", "content": "test"}]
        result = router.route_messages(messages)

        assert result.escalated is True
        assert "sonnet" in result.model_used.lower()
        router.close()

    def test_request_logged(self, tmp_path):
        provider = MockProvider()
        tracker = Tracker(tmp_path / "test.db")
        router = Router(
            "configs/default.yaml",
            provider=provider,
            tracker=tracker,
        )

        messages = [{"role": "user", "content": "What is 2+2?"}]
        router.route_messages(messages)
        stats = router.get_stats()

        assert stats["total_requests"] == 1
        router.close()

    def test_kwargs_passed_through(self, tmp_path):
        provider = MockProvider()
        tracker = Tracker(tmp_path / "test.db")
        router = Router(
            "configs/default.yaml",
            provider=provider,
            tracker=tracker,
        )

        messages = [{"role": "user", "content": "Hi"}]
        router.route_messages(messages, temperature=0.5, max_tokens=100)

        # MockProvider doesn't capture kwargs in our simplified version,
        # but this test ensures the method signature accepts **kwargs
        router.close()


class TestRouteMessagesStream:
    def test_returns_classification_and_chunks(self, tmp_path):
        provider = MockProvider()
        tracker = Tracker(tmp_path / "test.db")
        router = Router(
            "configs/default.yaml",
            provider=provider,
            tracker=tracker,
        )

        messages = [{"role": "user", "content": "What is 2+2?"}]
        result = router.route_messages_stream(messages)

        assert result.classification.complexity == Complexity.SIMPLE
        assert "haiku" in result.model.lower()
        assert not result.fallback_used
        assert not result.escalated

        chunk_list = list(result.chunks)
        assert len(chunk_list) == 2
        assert chunk_list[0].content == "Hello "
        assert chunk_list[1].finish_reason == "stop"
        router.close()

    def test_stream_with_escalation(self, tmp_path):
        classifier = MockClassifier(Complexity.SIMPLE, Category.FACTUAL, 0.5)
        provider = MockProvider()
        tracker = Tracker(tmp_path / "test.db")
        router = Router(
            "configs/default.yaml",
            classifier=classifier,
            provider=provider,
            tracker=tracker,
        )

        messages = [{"role": "user", "content": "test"}]
        result = router.route_messages_stream(messages)

        assert result.escalated is True
        assert "sonnet" in result.model.lower()
        list(result.chunks)  # consume
        router.close()

    def test_stream_error_raised_during_iteration(self, tmp_path):
        """When the provider fails during streaming, the error surfaces at iteration time."""
        provider = MockProvider(
            fail_models={"claude-haiku-4-5-20251001"}
        )
        tracker = Tracker(tmp_path / "test.db")
        router = Router(
            "configs/default.yaml",
            provider=provider,
            tracker=tracker,
        )

        messages = [{"role": "user", "content": "What is 2+2?"}]
        # route_messages_stream returns immediately (generator is lazy).
        # The model selected is haiku (first in list), but error surfaces during iteration.
        result = router.route_messages_stream(messages)
        assert "haiku" in result.model.lower()
        with pytest.raises(ProviderError):
            list(result.chunks)
        router.close()
