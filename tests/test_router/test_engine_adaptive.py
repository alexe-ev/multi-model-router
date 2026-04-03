"""Tests for Router integration with adaptive reranking."""

from datetime import datetime, timezone

import pytest

from mmrouter.models import (
    ClassificationResult,
    CompletionResult,
    Complexity,
    Category,
)
from mmrouter.providers.base import ProviderBase
from mmrouter.providers.litellm_provider import ProviderError
from mmrouter.router.engine import Router
from mmrouter.tracker.logger import Tracker


class MockProvider(ProviderBase):
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


def _write_config(tmp_path, adaptive_enabled=False, min_feedback=5):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"""
version: "1"
routes:
  simple:
    factual:
      model: model-a
      fallbacks:
        - model-b
    reasoning:
      model: model-a
      fallbacks:
        - model-b
    creative:
      model: model-a
      fallbacks:
        - model-b
    code:
      model: model-a
      fallbacks:
        - model-b
  medium:
    factual:
      model: model-a
      fallbacks:
        - model-b
    reasoning:
      model: model-a
      fallbacks:
        - model-b
    creative:
      model: model-a
      fallbacks:
        - model-b
    code:
      model: model-a
      fallbacks:
        - model-b
  complex:
    factual:
      model: model-a
      fallbacks:
        - model-b
    reasoning:
      model: model-a
      fallbacks:
        - model-b
    creative:
      model: model-a
      fallbacks:
        - model-b
    code:
      model: model-a
      fallbacks:
        - model-b
classifier:
  strategy: rules
  threshold: "0.7"
provider:
  timeout_ms: 30000
  max_retries: 0
adaptive:
  enabled: {"true" if adaptive_enabled else "false"}
  min_feedback_count: {min_feedback}
  decay_days: 30
  penalty_threshold: 0.4
  boost_threshold: 0.8
  cache_ttl: 0
""")
    return cfg


class TestRouterAdaptiveDisabled:
    def test_no_reranking_when_disabled(self, tmp_path):
        cfg = _write_config(tmp_path, adaptive_enabled=False)
        provider = MockProvider()
        tracker = Tracker(tmp_path / "test.db")
        router = Router(str(cfg), provider=provider, tracker=tracker)

        result = router.route("What is 2+2?")
        assert not result.adaptive_reranked
        # model-a should be tried first (config order)
        assert result.model_used == "model-a"
        router.close()


class TestRouterAdaptiveEnabled:
    def test_reranking_with_enough_feedback(self, tmp_path):
        cfg = _write_config(tmp_path, adaptive_enabled=True, min_feedback=5)
        provider = MockProvider()
        tracker = Tracker(tmp_path / "test.db")
        router = Router(str(cfg), provider=provider, tracker=tracker)

        # Build up negative feedback for model-a in simple/factual
        for _ in range(5):
            result = router.route("What is the capital of France?")
            router.submit_feedback(result.request_id, -1)

        # Build up positive feedback for model-b in simple/factual
        # We need to get model-b used. Log requests manually.
        from mmrouter.models import RequestLog, ClassificationResult, CompletionResult
        for _ in range(5):
            rid = tracker.log(RequestLog(
                prompt_hash="test",
                classification=ClassificationResult(
                    complexity="simple", category="factual", confidence=0.9
                ),
                model_used="model-b",
                completion=CompletionResult(
                    content="ok", model="model-b",
                    tokens_in=10, tokens_out=5, cost=0.001, latency_ms=100.0,
                ),
            ))
            tracker.submit_feedback(rid, 1)

        # Now route again: model-b should be boosted, model-a penalized
        result = router.route("What is the capital of Germany?")
        assert result.adaptive_reranked
        # model-b should be tried first now
        assert result.model_used == "model-b"
        router.close()

    def test_no_reranking_with_insufficient_feedback(self, tmp_path):
        cfg = _write_config(tmp_path, adaptive_enabled=True, min_feedback=20)
        provider = MockProvider()
        tracker = Tracker(tmp_path / "test.db")
        router = Router(str(cfg), provider=provider, tracker=tracker)

        # Only 3 feedbacks, min is 20
        for _ in range(3):
            result = router.route("What is 2+2?")
            router.submit_feedback(result.request_id, -1)

        result = router.route("What is 3+3?")
        assert not result.adaptive_reranked
        assert result.model_used == "model-a"  # original order
        router.close()

    def test_request_id_returned(self, tmp_path):
        cfg = _write_config(tmp_path, adaptive_enabled=True)
        provider = MockProvider()
        tracker = Tracker(tmp_path / "test.db")
        router = Router(str(cfg), provider=provider, tracker=tracker)

        result = router.route("What is 2+2?")
        assert result.request_id is not None
        assert isinstance(result.request_id, int)
        router.close()

    def test_submit_feedback_via_router(self, tmp_path):
        cfg = _write_config(tmp_path, adaptive_enabled=True)
        provider = MockProvider()
        tracker = Tracker(tmp_path / "test.db")
        router = Router(str(cfg), provider=provider, tracker=tracker)

        result = router.route("What is 2+2?")
        router.submit_feedback(result.request_id, 1)

        stats = router.get_feedback_stats()
        assert stats["total_feedback"] == 1
        router.close()

    def test_request_id_in_route_messages(self, tmp_path):
        """route_messages also returns request_id."""
        cfg = _write_config(tmp_path, adaptive_enabled=True)
        provider = MockProvider()
        # Need complete_messages on mock
        provider.complete_messages = lambda messages, model, **kw: CompletionResult(
            content=f"Response from {model}",
            model=model,
            tokens_in=10,
            tokens_out=20,
            cost=0.001,
            latency_ms=100.0,
        )
        tracker = Tracker(tmp_path / "test.db")
        router = Router(str(cfg), provider=provider, tracker=tracker)

        result = router.route_messages(
            [{"role": "user", "content": "What is 2+2?"}]
        )
        assert result.request_id is not None
        router.close()
