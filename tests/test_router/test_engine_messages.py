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
    StreamUsage,
)
from mmrouter.providers.base import ProviderBase
from mmrouter.providers.litellm_provider import ProviderError
from mmrouter.router.budget import BudgetExceededError
from mmrouter.router.engine import Router
from mmrouter.tracker.logger import Tracker


class MockProvider(ProviderBase):
    def __init__(self, fail_models=None, fail_models_open=None, report_usage=True):
        self._fail_models = fail_models or set()
        # Models that fail before the stream ever opened -- nothing was billed.
        # Separate from fail_models (which raises mid_stream=True) so existing
        # tests relying on that behavior are untouched.
        self._fail_models_open = fail_models_open or set()
        # Switch to suppress usage reporting, for tests exercising the
        # "provider sent nothing" (no_usage) path.
        self._report_usage = report_usage
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
        if model in self._fail_models_open:
            # The call never opened: nothing was billed.
            raise ProviderError(f"{model} is down", retryable=True, mid_stream=False)
        if model in self._fail_models:
            # Raised from inside the generator body, i.e. after the stream has
            # already "opened" in the sense this mock has one -- mid_stream=True.
            raise ProviderError(f"{model} is down", retryable=True, mid_stream=True)
        yield StreamChunk(content="Hello ", model=model)
        yield StreamChunk(content="world!", model=model, finish_reason="stop")
        if self._report_usage:
            return StreamUsage(tokens_in=10, tokens_out=20, cost=0.001)
        return None


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


class TestRouteMessagesAlerts:
    """Alerts fire on both the stream and non-stream messages paths (MMR-3)."""

    @pytest.fixture(autouse=True)
    def restore_builtin_cooldown(self):
        # AlertManager rewrites the cooldown on the shared BUILTIN_RULES instance.
        from mmrouter.alerts.rules import BUILTIN_RULES

        original = BUILTIN_RULES["error_rate"].cooldown_seconds
        yield
        BUILTIN_RULES["error_rate"].cooldown_seconds = original

    def _config(self, tmp_path):
        cfg = tmp_path / "alerting.yaml"
        cfg.write_text("""
version: "1"
routes:
  simple:
    factual:
      model: claude-haiku-4-5-20251001
alerts:
  enabled: true
  cooldown_seconds: 0
  rules:
    - error_rate
""")
        return str(cfg)

    def _router(self, tmp_path, provider=None):
        """A Router with alerting on, over a tracker already in a firing state."""
        from mmrouter.tracker.logger import _INSERT

        tracker = Tracker(tmp_path / "alerts.db")
        conn = tracker.connection
        for i in range(10):
            conn.execute(_INSERT, (
                "2026-04-01T10:00:00", "abc", "simple", "factual", 0.9,
                "claude-haiku-4-5-20251001", 10, 20, 0.0001, 50.0,
                1 if i < 2 else 0, 0, 1, 0, 0, None, None,
            ))
        conn.commit()

        return Router(
            self._config(tmp_path),
            classifier=MockClassifier(Complexity.SIMPLE, Category.FACTUAL),
            provider=provider or MockProvider(),
            tracker=tracker,
        )

    def _incidents(self, tmp_path):
        import sqlite3

        from mmrouter.alerts.store import AlertIncidentStore

        conn = sqlite3.connect(str(tmp_path / "alerts.db"))
        try:
            return AlertIncidentStore(conn).list_incidents()
        finally:
            conn.close()

    def test_route_messages_records_an_incident(self, tmp_path):
        router = self._router(tmp_path)
        router.route_messages([{"role": "user", "content": "What is 2+2?"}])

        items = self._incidents(tmp_path)
        assert len(items) == 1
        assert items[0]["rule_name"] == "error_rate"
        assert items[0]["fire_count"] == 1
        router.close()

    def test_route_messages_extends_the_same_incident(self, tmp_path):
        router = self._router(tmp_path)
        messages = [{"role": "user", "content": "What is 2+2?"}]
        router.route_messages(messages)
        router.route_messages(messages)

        items = self._incidents(tmp_path)
        assert len(items) == 1
        assert items[0]["fire_count"] == 2
        router.close()

    def test_stream_path_records_an_incident(self, tmp_path):
        """Self-check 5: replaces test_stream_path_records_nothing. The stream now
        opens the incident that the non-stream sibling opens (same fixture)."""
        router = self._router(tmp_path)
        messages = [{"role": "user", "content": "What is 2+2?"}]

        result = router.route_messages_stream(messages)
        list(result.chunks)  # consume

        items = self._incidents(tmp_path)
        assert len(items) == 1
        assert items[0]["rule_name"] == "error_rate"
        assert items[0]["fire_count"] == 1
        router.close()

    def test_stream_no_usage_does_not_evaluate_alerts(self, tmp_path):
        """Self-check 2 (alerts half): no usage -> no row, and alert evaluation
        never runs, even though the seeded fixture is primed to fire."""
        router = self._router(tmp_path, provider=MockProvider(report_usage=False))
        messages = [{"role": "user", "content": "What is 2+2?"}]

        result = router.route_messages_stream(messages)
        list(result.chunks)  # consume

        assert self._incidents(tmp_path) == []
        # Fixture seeds 10 rows; the count must stay there -- nothing new logged.
        assert router.get_stats()["total_requests"] == 10
        rows = router._tracker.connection.execute(
            "SELECT model, reason FROM unrecorded_streams"
        ).fetchall()
        assert [(r[0], r[1]) for r in rows] == [(result.model, "no_usage")]
        router.close()


class TestRecordedStream:
    """Streamed requests are logged, so alerts, budget and stats can see them (MMR-3)."""

    def _router(self, tmp_path, provider=None):
        tracker = Tracker(tmp_path / "test.db")
        router = Router(
            "configs/default.yaml",
            provider=provider or MockProvider(),
            tracker=tracker,
        )
        return router, tracker

    def _unrecorded(self, tracker):
        rows = tracker.connection.execute(
            "SELECT model, reason FROM unrecorded_streams"
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def test_usage_reported_writes_one_row(self, tmp_path):
        """Self-check 1: drained stream with usage -> one requests row with
        those tokens and cost, latency_ms > 0, unrecorded_streams empty."""
        router, tracker = self._router(tmp_path)
        messages = [{"role": "user", "content": "What is 2+2?"}]

        result = router.route_messages_stream(messages)
        list(result.chunks)  # consume

        stats = router.get_stats()
        assert stats["total_requests"] == 1
        assert stats["total_tokens_in"] == 10
        assert stats["total_tokens_out"] == 20
        assert stats["total_cost"] == pytest.approx(0.001)
        row = tracker.connection.execute("SELECT latency_ms FROM requests").fetchone()
        assert row[0] > 0
        assert self._unrecorded(tracker) == []
        router.close()

    def test_provider_error_mid_stream_counts_provider_error(self, tmp_path):
        """Self-check 3: ProviderError raised from inside the generator ->
        propagates, no row, reason provider_error."""
        provider = MockProvider(fail_models={"claude-haiku-4-5-20251001"})
        router, tracker = self._router(tmp_path, provider)
        messages = [{"role": "user", "content": "What is 2+2?"}]

        result = router.route_messages_stream(messages)
        with pytest.raises(ProviderError):
            list(result.chunks)

        assert router.get_stats()["total_requests"] == 0
        assert self._unrecorded(tracker) == [(result.model, "provider_error")]
        router.close()

    def test_provider_error_open_time_counts_provider_unavailable(self, tmp_path):
        """Self-check 15 (open-time half): ProviderError with mid_stream=False
        -- the call never opened, nothing was spent -- propagates, no row,
        reason provider_unavailable. Distinct from provider_error (mid-flight,
        may have been billed): that is the whole point of the reason column."""
        provider = MockProvider(fail_models_open={"claude-haiku-4-5-20251001"})
        router, tracker = self._router(tmp_path, provider)
        messages = [{"role": "user", "content": "What is 2+2?"}]

        result = router.route_messages_stream(messages)
        with pytest.raises(ProviderError):
            list(result.chunks)

        assert router.get_stats()["total_requests"] == 0
        assert self._unrecorded(tracker) == [(result.model, "provider_unavailable")]
        router.close()

    def test_abandoned_iterator_counts_aborted(self, tmp_path):
        """Self-check 4: closing the generator after one chunk -> reason
        aborted, no row."""
        router, tracker = self._router(tmp_path)
        messages = [{"role": "user", "content": "What is 2+2?"}]

        result = router.route_messages_stream(messages)
        next(result.chunks)
        result.chunks.close()

        assert router.get_stats()["total_requests"] == 0
        assert self._unrecorded(tracker) == [(result.model, "aborted")]
        router.close()

    def test_budget_rejected_counts_budget_rejected(self, tmp_path):
        """Self-check 6: budget in reject mode over the limit ->
        route_messages_stream raises BudgetExceededError, one unrecorded_streams
        row, reason budget_rejected, model NULL."""
        from datetime import datetime, timezone

        cfg = tmp_path / "budget.yaml"
        cfg.write_text("""
version: "1"
routes:
  simple:
    factual:
      model: claude-haiku-4-5-20251001
budget:
  enabled: true
  daily_limit: 1.0
  hard_limit_action: reject
""")
        tracker = Tracker(tmp_path / "test.db")
        tracker.connection.execute(
            """INSERT INTO requests
               (timestamp, prompt_hash, complexity, category, confidence,
                model, tokens_in, tokens_out, cost, latency_ms, fallback_used,
                cascade_used, cascade_attempts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now(timezone.utc).isoformat(), "seed", "simple", "factual", 0.9,
             "claude-haiku-4-5-20251001", 10, 20, 5.0, 100.0, 0, 0, 1),
        )
        tracker.connection.commit()
        classifier = MockClassifier(Complexity.SIMPLE, Category.FACTUAL)
        router = Router(str(cfg), classifier=classifier, provider=MockProvider(), tracker=tracker)
        messages = [{"role": "user", "content": "test"}]

        with pytest.raises(BudgetExceededError):
            router.route_messages_stream(messages)

        assert self._unrecorded(tracker) == [(None, "budget_rejected")]
        router.close()

    def test_stream_spend_counts_toward_daily_budget(self, tmp_path):
        """Self-check 11: BudgetManager.get_daily_spend() includes a row
        written by the stream recorder."""
        from mmrouter.models import BudgetConfig
        from mmrouter.router.budget import BudgetManager

        router, tracker = self._router(tmp_path)
        messages = [{"role": "user", "content": "What is 2+2?"}]

        result = router.route_messages_stream(messages)
        list(result.chunks)  # consume

        budget = BudgetManager(BudgetConfig(enabled=True, daily_limit=10.0), tracker.connection)
        assert budget.get_daily_spend() == pytest.approx(0.001)
        router.close()
