"""Tests for budget manager and budget-aware routing."""

import sqlite3
from datetime import datetime, timezone

import pytest

from mmrouter.classifier import ClassifierBase
from mmrouter.models import (
    BudgetConfig,
    ClassificationResult,
    CompletionResult,
    Complexity,
    Category,
)
from mmrouter.providers.base import ProviderBase
from mmrouter.router.budget import BudgetExceededError, BudgetManager, BudgetTier
from mmrouter.router.engine import Router
from mmrouter.tracker.logger import Tracker


# -- Helpers --


def _make_tracker_with_spend(tmp_path, spend: float) -> Tracker:
    """Create a tracker and insert a fake request with the given cost for today."""
    tracker = Tracker(tmp_path / "test.db")
    if spend > 0:
        now = datetime.now(timezone.utc).isoformat()
        tracker.connection.execute(
            """INSERT INTO requests
               (timestamp, prompt_hash, complexity, category, confidence,
                model, tokens_in, tokens_out, cost, latency_ms, fallback_used,
                cascade_used, cascade_attempts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (now, "test", "simple", "factual", 0.9,
             "claude-haiku-4-5-20251001", 10, 20, spend, 100.0, 0, 0, 1),
        )
        tracker.connection.commit()
    return tracker


class MockClassifier(ClassifierBase):
    def __init__(self, complexity: Complexity, category: Category, confidence: float = 0.9):
        self._result = ClassificationResult(
            complexity=complexity, category=category, confidence=confidence
        )

    def classify(self, prompt: str) -> ClassificationResult:
        return self._result


class MockProvider(ProviderBase):
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

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


# -- BudgetManager unit tests --


class TestBudgetManager:
    def test_disabled_returns_normal(self, tmp_path):
        tracker = _make_tracker_with_spend(tmp_path, 0)
        config = BudgetConfig(enabled=False, daily_limit=1.0)
        mgr = BudgetManager(config, tracker.connection)
        assert mgr.get_budget_tier() == BudgetTier.NORMAL
        assert not mgr.enabled
        tracker.close()

    def test_disabled_when_zero_limit(self, tmp_path):
        tracker = _make_tracker_with_spend(tmp_path, 0)
        config = BudgetConfig(enabled=True, daily_limit=0.0)
        mgr = BudgetManager(config, tracker.connection)
        assert not mgr.enabled
        tracker.close()

    def test_normal_tier(self, tmp_path):
        tracker = _make_tracker_with_spend(tmp_path, 0.50)
        config = BudgetConfig(enabled=True, daily_limit=10.0)
        mgr = BudgetManager(config, tracker.connection)
        assert mgr.get_budget_tier() == BudgetTier.NORMAL
        tracker.close()

    def test_warn_tier(self, tmp_path):
        tracker = _make_tracker_with_spend(tmp_path, 7.50)
        config = BudgetConfig(enabled=True, daily_limit=10.0, warn_threshold=0.75)
        mgr = BudgetManager(config, tracker.connection)
        assert mgr.get_budget_tier() == BudgetTier.WARN
        tracker.close()

    def test_downgrade_tier(self, tmp_path):
        tracker = _make_tracker_with_spend(tmp_path, 9.50)
        config = BudgetConfig(enabled=True, daily_limit=10.0, downgrade_threshold=0.90)
        mgr = BudgetManager(config, tracker.connection)
        assert mgr.get_budget_tier() == BudgetTier.DOWNGRADE
        tracker.close()

    def test_hard_limit_tier(self, tmp_path):
        tracker = _make_tracker_with_spend(tmp_path, 10.0)
        config = BudgetConfig(enabled=True, daily_limit=10.0)
        mgr = BudgetManager(config, tracker.connection)
        assert mgr.get_budget_tier() == BudgetTier.HARD_LIMIT
        tracker.close()

    def test_hard_limit_over_budget(self, tmp_path):
        tracker = _make_tracker_with_spend(tmp_path, 15.0)
        config = BudgetConfig(enabled=True, daily_limit=10.0)
        mgr = BudgetManager(config, tracker.connection)
        assert mgr.get_budget_tier() == BudgetTier.HARD_LIMIT
        tracker.close()

    def test_get_remaining(self, tmp_path):
        tracker = _make_tracker_with_spend(tmp_path, 3.0)
        config = BudgetConfig(enabled=True, daily_limit=10.0)
        mgr = BudgetManager(config, tracker.connection)
        assert mgr.get_remaining() == pytest.approx(7.0, abs=0.01)
        tracker.close()

    def test_get_remaining_disabled(self, tmp_path):
        tracker = _make_tracker_with_spend(tmp_path, 3.0)
        config = BudgetConfig(enabled=False)
        mgr = BudgetManager(config, tracker.connection)
        assert mgr.get_remaining() == float("inf")
        tracker.close()

    def test_get_remaining_over_budget(self, tmp_path):
        tracker = _make_tracker_with_spend(tmp_path, 15.0)
        config = BudgetConfig(enabled=True, daily_limit=10.0)
        mgr = BudgetManager(config, tracker.connection)
        assert mgr.get_remaining() == 0.0
        tracker.close()

    def test_apply_budget_normal_no_change(self, tmp_path):
        tracker = _make_tracker_with_spend(tmp_path, 1.0)
        config = BudgetConfig(enabled=True, daily_limit=10.0)
        mgr = BudgetManager(config, tracker.connection)
        assert mgr.apply_budget(Complexity.COMPLEX) == Complexity.COMPLEX
        tracker.close()

    def test_apply_budget_warn_no_change(self, tmp_path):
        tracker = _make_tracker_with_spend(tmp_path, 8.0)
        config = BudgetConfig(enabled=True, daily_limit=10.0, warn_threshold=0.75)
        mgr = BudgetManager(config, tracker.connection)
        # Warn tier doesn't downgrade
        assert mgr.apply_budget(Complexity.COMPLEX) == Complexity.COMPLEX
        tracker.close()

    def test_apply_budget_downgrade_complex_to_medium(self, tmp_path):
        tracker = _make_tracker_with_spend(tmp_path, 9.5)
        config = BudgetConfig(enabled=True, daily_limit=10.0, downgrade_threshold=0.90)
        mgr = BudgetManager(config, tracker.connection)
        assert mgr.apply_budget(Complexity.COMPLEX) == Complexity.MEDIUM
        tracker.close()

    def test_apply_budget_downgrade_medium_to_simple(self, tmp_path):
        tracker = _make_tracker_with_spend(tmp_path, 9.5)
        config = BudgetConfig(enabled=True, daily_limit=10.0, downgrade_threshold=0.90)
        mgr = BudgetManager(config, tracker.connection)
        assert mgr.apply_budget(Complexity.MEDIUM) == Complexity.SIMPLE
        tracker.close()

    def test_apply_budget_downgrade_simple_stays_simple(self, tmp_path):
        tracker = _make_tracker_with_spend(tmp_path, 9.5)
        config = BudgetConfig(enabled=True, daily_limit=10.0, downgrade_threshold=0.90)
        mgr = BudgetManager(config, tracker.connection)
        assert mgr.apply_budget(Complexity.SIMPLE) == Complexity.SIMPLE
        tracker.close()

    def test_apply_budget_hard_limit_cheapest(self, tmp_path):
        tracker = _make_tracker_with_spend(tmp_path, 10.0)
        config = BudgetConfig(enabled=True, daily_limit=10.0, hard_limit_action="cheapest")
        mgr = BudgetManager(config, tracker.connection)
        assert mgr.apply_budget(Complexity.COMPLEX) == Complexity.SIMPLE
        tracker.close()

    def test_apply_budget_hard_limit_reject(self, tmp_path):
        tracker = _make_tracker_with_spend(tmp_path, 10.0)
        config = BudgetConfig(enabled=True, daily_limit=10.0, hard_limit_action="reject")
        mgr = BudgetManager(config, tracker.connection)
        with pytest.raises(BudgetExceededError, match="exhausted"):
            mgr.apply_budget(Complexity.COMPLEX)
        tracker.close()

    def test_get_status_enabled(self, tmp_path):
        tracker = _make_tracker_with_spend(tmp_path, 5.0)
        config = BudgetConfig(enabled=True, daily_limit=10.0)
        mgr = BudgetManager(config, tracker.connection)
        status = mgr.get_status()
        assert status["enabled"] is True
        assert status["daily_limit"] == 10.0
        assert status["spent_today"] == pytest.approx(5.0, abs=0.01)
        assert status["remaining"] == pytest.approx(5.0, abs=0.01)
        assert status["usage_pct"] == pytest.approx(50.0, abs=0.1)
        assert status["tier"] == "normal"
        tracker.close()

    def test_get_status_disabled(self, tmp_path):
        tracker = _make_tracker_with_spend(tmp_path, 0)
        config = BudgetConfig(enabled=False)
        mgr = BudgetManager(config, tracker.connection)
        status = mgr.get_status()
        assert status["enabled"] is False
        tracker.close()

    def test_cache_invalidation(self, tmp_path):
        tracker = _make_tracker_with_spend(tmp_path, 5.0)
        config = BudgetConfig(enabled=True, daily_limit=10.0)
        mgr = BudgetManager(config, tracker.connection)

        spend1 = mgr.get_daily_spend()
        assert spend1 == pytest.approx(5.0, abs=0.01)

        # Insert more spend
        now = datetime.now(timezone.utc).isoformat()
        tracker.connection.execute(
            """INSERT INTO requests
               (timestamp, prompt_hash, complexity, category, confidence,
                model, tokens_in, tokens_out, cost, latency_ms, fallback_used,
                cascade_used, cascade_attempts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (now, "test2", "simple", "factual", 0.9,
             "claude-haiku-4-5-20251001", 10, 20, 3.0, 100.0, 0, 0, 1),
        )
        tracker.connection.commit()

        # Still cached
        spend2 = mgr.get_daily_spend()
        assert spend2 == pytest.approx(5.0, abs=0.01)

        # After invalidation
        mgr.invalidate_cache()
        spend3 = mgr.get_daily_spend()
        assert spend3 == pytest.approx(8.0, abs=0.01)
        tracker.close()


# -- Config parsing tests --


class TestBudgetConfig:
    def test_budget_config_from_yaml(self, tmp_path):
        from mmrouter.router.config import load_config

        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("""
version: "1"

routes:
  simple:
    factual:
      model: claude-haiku-4-5-20251001

budget:
  enabled: true
  daily_limit: 5.0
  warn_threshold: 0.70
  downgrade_threshold: 0.85
  hard_limit_action: reject
""")
        config = load_config(str(cfg_file))
        assert config.budget.enabled is True
        assert config.budget.daily_limit == 5.0
        assert config.budget.warn_threshold == 0.70
        assert config.budget.downgrade_threshold == 0.85
        assert config.budget.hard_limit_action == "reject"

    def test_budget_config_defaults(self, tmp_path):
        from mmrouter.router.config import load_config

        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("""
version: "1"

routes:
  simple:
    factual:
      model: claude-haiku-4-5-20251001
""")
        config = load_config(str(cfg_file))
        assert config.budget.enabled is False
        assert config.budget.daily_limit == 0.0
        assert config.budget.hard_limit_action == "cheapest"

    def test_budget_config_invalid_action(self, tmp_path):
        from mmrouter.router.config import ConfigError, load_config

        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("""
version: "1"

routes:
  simple:
    factual:
      model: claude-haiku-4-5-20251001

budget:
  enabled: true
  daily_limit: 5.0
  hard_limit_action: explode
""")
        with pytest.raises(ConfigError, match="Invalid hard_limit_action"):
            load_config(str(cfg_file))


# -- Router integration tests --


def _write_budget_config(tmp_path, *, daily_limit=10.0, spend=0.0,
                         hard_limit_action="cheapest",
                         warn_threshold=0.75, downgrade_threshold=0.90):
    """Write a budget-enabled config and return (config_path, tracker)."""
    cfg = tmp_path / "budget_test.yaml"
    cfg.write_text(f"""
version: "1"

routes:
  simple:
    factual:
      model: claude-haiku-4-5-20251001
      fallbacks:
        - claude-sonnet-4-6
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

budget:
  enabled: true
  daily_limit: {daily_limit}
  warn_threshold: {warn_threshold}
  downgrade_threshold: {downgrade_threshold}
  hard_limit_action: {hard_limit_action}
""")
    tracker = _make_tracker_with_spend(tmp_path, spend)
    return str(cfg), tracker


class TestBudgetRouterIntegration:
    def test_budget_disabled_no_downgrade(self, tmp_path):
        """With budget disabled, complex routes stay complex."""
        provider = MockProvider()
        tracker = Tracker(tmp_path / "test.db")
        classifier = MockClassifier(Complexity.COMPLEX, Category.REASONING)
        router = Router(
            "configs/default.yaml",
            classifier=classifier,
            provider=provider,
            tracker=tracker,
        )
        result = router.route("test")
        assert not result.budget_downgraded
        assert "opus" in result.model_used.lower()
        router.close()

    def test_budget_normal_no_downgrade(self, tmp_path):
        """Under budget: complex routes stay complex."""
        cfg, tracker = _write_budget_config(tmp_path, daily_limit=10.0, spend=1.0)
        provider = MockProvider()
        classifier = MockClassifier(Complexity.COMPLEX, Category.REASONING)
        router = Router(cfg, classifier=classifier, provider=provider, tracker=tracker)
        result = router.route("test")
        assert not result.budget_downgraded
        assert "opus" in result.model_used.lower()
        router.close()

    def test_budget_downgrade_complex_to_medium(self, tmp_path):
        """At downgrade threshold: complex -> medium."""
        cfg, tracker = _write_budget_config(tmp_path, daily_limit=10.0, spend=9.5)
        provider = MockProvider()
        classifier = MockClassifier(Complexity.COMPLEX, Category.REASONING)
        router = Router(cfg, classifier=classifier, provider=provider, tracker=tracker)
        result = router.route("test")
        assert result.budget_downgraded
        assert "sonnet" in result.model_used.lower()
        router.close()

    def test_budget_hard_limit_cheapest(self, tmp_path):
        """At hard limit with 'cheapest' action: forces simple tier."""
        cfg, tracker = _write_budget_config(
            tmp_path, daily_limit=10.0, spend=10.0, hard_limit_action="cheapest"
        )
        provider = MockProvider()
        classifier = MockClassifier(Complexity.COMPLEX, Category.FACTUAL)
        router = Router(cfg, classifier=classifier, provider=provider, tracker=tracker)
        result = router.route("test")
        assert result.budget_downgraded
        assert "haiku" in result.model_used.lower()
        router.close()

    def test_budget_hard_limit_reject(self, tmp_path):
        """At hard limit with 'reject' action: raises BudgetExceededError."""
        cfg, tracker = _write_budget_config(
            tmp_path, daily_limit=10.0, spend=10.0, hard_limit_action="reject"
        )
        provider = MockProvider()
        classifier = MockClassifier(Complexity.COMPLEX, Category.FACTUAL)
        router = Router(cfg, classifier=classifier, provider=provider, tracker=tracker)
        with pytest.raises(BudgetExceededError):
            router.route("test")
        router.close()

    def test_budget_warn_no_downgrade(self, tmp_path):
        """Warn tier does not downgrade, just warns."""
        cfg, tracker = _write_budget_config(tmp_path, daily_limit=10.0, spend=8.0)
        provider = MockProvider()
        classifier = MockClassifier(Complexity.COMPLEX, Category.REASONING)
        router = Router(cfg, classifier=classifier, provider=provider, tracker=tracker)
        result = router.route("test")
        assert not result.budget_downgraded
        assert "opus" in result.model_used.lower()
        router.close()

    def test_budget_applies_to_route_messages(self, tmp_path):
        """Budget downgrade also works for route_messages."""
        cfg, tracker = _write_budget_config(tmp_path, daily_limit=10.0, spend=9.5)
        provider = MockProvider()
        provider.complete_messages = lambda msgs, model, **kw: CompletionResult(
            content=f"Response from {model}",
            model=model, tokens_in=10, tokens_out=20, cost=0.001, latency_ms=100.0,
        )
        classifier = MockClassifier(Complexity.COMPLEX, Category.REASONING)
        router = Router(cfg, classifier=classifier, provider=provider, tracker=tracker)
        messages = [{"role": "user", "content": "test"}]
        result = router.route_messages(messages)
        assert result.budget_downgraded
        assert "sonnet" in result.model_used.lower()
        router.close()

    def test_budget_applies_to_stream(self, tmp_path):
        """Budget downgrade also works for route_messages_stream."""
        from mmrouter.models import StreamChunk

        cfg, tracker = _write_budget_config(tmp_path, daily_limit=10.0, spend=9.5)
        provider = MockProvider()

        def mock_stream(msgs, model, **kw):
            yield StreamChunk(content="hi", model=model, finish_reason="stop")

        provider.stream_messages = mock_stream
        classifier = MockClassifier(Complexity.COMPLEX, Category.REASONING)
        router = Router(cfg, classifier=classifier, provider=provider, tracker=tracker)
        messages = [{"role": "user", "content": "test"}]
        classification, model, fallback_used, escalated, budget_downgraded, chunks = (
            router.route_messages_stream(messages)
        )
        assert budget_downgraded
        assert "sonnet" in model.lower()
        list(chunks)
        router.close()

    def test_budget_get_status(self, tmp_path):
        """Router.get_budget_status() returns correct data."""
        cfg, tracker = _write_budget_config(tmp_path, daily_limit=10.0, spend=5.0)
        provider = MockProvider()
        classifier = MockClassifier(Complexity.SIMPLE, Category.FACTUAL)
        router = Router(cfg, classifier=classifier, provider=provider, tracker=tracker)
        status = router.get_budget_status()
        assert status["enabled"] is True
        assert status["daily_limit"] == 10.0
        assert status["spent_today"] == pytest.approx(5.0, abs=0.01)
        assert status["tier"] == "normal"
        router.close()

    def test_budget_interacts_with_confidence_escalation(self, tmp_path):
        """Budget downgrade can override confidence escalation."""
        # Low confidence would escalate simple -> medium,
        # but budget downgrade at 95% forces it back to simple
        cfg, tracker = _write_budget_config(tmp_path, daily_limit=10.0, spend=9.5)
        provider = MockProvider()
        classifier = MockClassifier(Complexity.SIMPLE, Category.FACTUAL, confidence=0.5)
        router = Router(cfg, classifier=classifier, provider=provider, tracker=tracker)
        result = router.route("test")
        # Confidence escalation: simple -> medium
        # Budget downgrade: medium -> simple
        assert result.escalated is True
        assert result.budget_downgraded is True
        assert "haiku" in result.model_used.lower()
        router.close()
