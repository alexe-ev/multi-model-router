"""Tests for circuit breaker, provider circuit breaker, and registry."""

import time

import pytest

from mmrouter.models import ProviderConfig
from mmrouter.router.fallback import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
    ProviderCircuitBreaker,
    extract_provider,
)


def _make_breaker(threshold: int = 5, reset_ms: int = 60000) -> CircuitBreaker:
    config = ProviderConfig(
        circuit_breaker_threshold=threshold,
        circuit_breaker_reset_ms=reset_ms,
    )
    return CircuitBreaker("test-model", config)


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = _make_breaker()
        assert cb.state == CircuitState.CLOSED

    def test_stays_closed_below_threshold(self):
        cb = _make_breaker(threshold=5)
        for _ in range(4):
            cb.record_failure(retryable=True)
        assert cb.state == CircuitState.CLOSED

    def test_opens_at_threshold(self):
        cb = _make_breaker(threshold=5)
        for _ in range(5):
            cb.record_failure(retryable=True)
        assert cb.state == CircuitState.OPEN

    def test_permanent_errors_dont_trip(self):
        cb = _make_breaker(threshold=5)
        for _ in range(10):
            cb.record_failure(retryable=False)
        assert cb.state == CircuitState.CLOSED

    def test_mixed_errors_only_transient_count(self):
        cb = _make_breaker(threshold=5)
        for _ in range(4):
            cb.record_failure(retryable=True)
        for _ in range(10):
            cb.record_failure(retryable=False)
        assert cb.state == CircuitState.CLOSED
        cb.record_failure(retryable=True)
        assert cb.state == CircuitState.OPEN

    def test_check_raises_when_open(self):
        cb = _make_breaker(threshold=1)
        cb.record_failure(retryable=True)
        with pytest.raises(CircuitOpenError):
            cb.check()

    def test_check_ok_when_closed(self):
        cb = _make_breaker()
        cb.check()  # should not raise

    def test_success_resets_to_closed(self):
        cb = _make_breaker(threshold=5)
        for _ in range(4):
            cb.record_failure(retryable=True)
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        # After reset, need full threshold again to open
        for _ in range(4):
            cb.record_failure(retryable=True)
        assert cb.state == CircuitState.CLOSED

    def test_transitions_to_half_open_after_reset(self):
        cb = _make_breaker(threshold=1, reset_ms=50)
        cb.record_failure(retryable=True)
        assert cb.state == CircuitState.OPEN
        time.sleep(0.06)
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_success_closes(self):
        cb = _make_breaker(threshold=1, reset_ms=50)
        cb.record_failure(retryable=True)
        time.sleep(0.06)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        cb = _make_breaker(threshold=1, reset_ms=50)
        cb.record_failure(retryable=True)
        time.sleep(0.06)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_failure(retryable=True)
        assert cb.state == CircuitState.OPEN

    def test_circuit_open_error_has_retry_after(self):
        cb = _make_breaker(threshold=1, reset_ms=60000)
        cb.record_failure(retryable=True)
        with pytest.raises(CircuitOpenError) as exc_info:
            cb.check()
        assert exc_info.value.retry_after_ms > 0
        assert exc_info.value.model == "test-model"


class TestCircuitBreakerRegistry:
    def test_creates_breakers_on_demand(self):
        registry = CircuitBreakerRegistry(ProviderConfig())
        breaker = registry.get("model-a")
        assert isinstance(breaker, CircuitBreaker)

    def test_returns_same_breaker_for_same_model(self):
        registry = CircuitBreakerRegistry(ProviderConfig())
        b1 = registry.get("model-a")
        b2 = registry.get("model-a")
        assert b1 is b2

    def test_different_models_get_different_breakers(self):
        registry = CircuitBreakerRegistry(ProviderConfig())
        b1 = registry.get("model-a")
        b2 = registry.get("model-b")
        assert b1 is not b2


class TestExtractProvider:
    def test_claude_models(self):
        assert extract_provider("claude-haiku-4-5-20251001") == "anthropic"
        assert extract_provider("claude-sonnet-4-6") == "anthropic"
        assert extract_provider("claude-opus-4-6") == "anthropic"

    def test_openai_gpt_models(self):
        assert extract_provider("gpt-4o") == "openai"
        assert extract_provider("gpt-4o-mini") == "openai"
        assert extract_provider("gpt-3.5-turbo") == "openai"

    def test_openai_o_models(self):
        assert extract_provider("o1-preview") == "openai"
        assert extract_provider("o1-mini") == "openai"
        assert extract_provider("o3-mini") == "openai"

    def test_openai_chatgpt_models(self):
        assert extract_provider("chatgpt-4o-latest") == "openai"

    def test_google_models(self):
        assert extract_provider("gemini-2.5-pro") == "google"
        assert extract_provider("gemini-2.0-flash") == "google"

    def test_unknown_model(self):
        assert extract_provider("some-custom-model") == "unknown"
        assert extract_provider("llama-3-70b") == "unknown"

    def test_provider_map_override(self):
        provider_map = {"my-custom-model": "openai", "llama-3-70b": "meta"}
        assert extract_provider("my-custom-model", provider_map) == "openai"
        assert extract_provider("llama-3-70b", provider_map) == "meta"

    def test_provider_map_takes_precedence(self):
        provider_map = {"claude-sonnet-4-6": "custom-provider"}
        assert extract_provider("claude-sonnet-4-6", provider_map) == "custom-provider"

    def test_provider_map_none(self):
        assert extract_provider("gpt-4o", None) == "openai"

    def test_empty_provider_map(self):
        assert extract_provider("gpt-4o", {}) == "openai"


class TestProviderCircuitBreaker:
    def _make_provider_breaker(self, reset_ms: int = 60000) -> ProviderCircuitBreaker:
        config = ProviderConfig(provider_circuit_breaker_reset_ms=reset_ms)
        return ProviderCircuitBreaker("anthropic", config)

    def test_starts_closed(self):
        pb = self._make_provider_breaker()
        assert pb.state == CircuitState.CLOSED

    def test_trip_opens(self):
        pb = self._make_provider_breaker()
        pb.trip()
        assert pb.state == CircuitState.OPEN

    def test_check_raises_when_open(self):
        pb = self._make_provider_breaker()
        pb.trip()
        with pytest.raises(CircuitOpenError):
            pb.check()

    def test_check_ok_when_closed(self):
        pb = self._make_provider_breaker()
        pb.check()  # should not raise

    def test_transitions_to_half_open_after_reset(self):
        pb = self._make_provider_breaker(reset_ms=50)
        pb.trip()
        assert pb.state == CircuitState.OPEN
        time.sleep(0.06)
        assert pb.state == CircuitState.HALF_OPEN

    def test_half_open_success_closes(self):
        pb = self._make_provider_breaker(reset_ms=50)
        pb.trip()
        time.sleep(0.06)
        assert pb.state == CircuitState.HALF_OPEN
        pb.record_success()
        assert pb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        pb = self._make_provider_breaker(reset_ms=50)
        pb.trip()
        time.sleep(0.06)
        assert pb.state == CircuitState.HALF_OPEN
        pb.record_failure()
        assert pb.state == CircuitState.OPEN

    def test_circuit_open_error_has_provider_name(self):
        pb = self._make_provider_breaker()
        pb.trip()
        with pytest.raises(CircuitOpenError) as exc_info:
            pb.check()
        assert exc_info.value.model == "anthropic"
        assert exc_info.value.retry_after_ms > 0


class TestProviderBreakerRegistry:
    def _make_registry(self, threshold: int = 2, model_threshold: int = 1) -> CircuitBreakerRegistry:
        config = ProviderConfig(
            circuit_breaker_threshold=model_threshold,
            circuit_breaker_reset_ms=60000,
            provider_circuit_breaker_threshold=threshold,
            provider_circuit_breaker_reset_ms=60000,
        )
        return CircuitBreakerRegistry(config)

    def test_check_provider_ok_when_no_failures(self):
        registry = self._make_registry()
        registry.check_provider("claude-sonnet-4-6")  # should not raise

    def test_check_provider_ok_for_unknown(self):
        registry = self._make_registry()
        registry.check_provider("some-unknown-model")  # should not raise

    def test_provider_trips_when_enough_models_open(self):
        registry = self._make_registry(threshold=2, model_threshold=1)

        # Open breaker for model 1
        b1 = registry.get("claude-haiku-4-5-20251001")
        b1.record_failure(retryable=True)
        assert b1.state == CircuitState.OPEN
        registry.record_model_open("claude-haiku-4-5-20251001")

        # Provider not tripped yet (only 1 model open, threshold=2)
        registry.check_provider("claude-sonnet-4-6")  # should not raise

        # Open breaker for model 2
        b2 = registry.get("claude-sonnet-4-6")
        b2.record_failure(retryable=True)
        assert b2.state == CircuitState.OPEN
        registry.record_model_open("claude-sonnet-4-6")

        # Provider should now be tripped
        with pytest.raises(CircuitOpenError):
            registry.check_provider("claude-opus-4-6")

    def test_provider_not_tripped_below_threshold(self):
        registry = self._make_registry(threshold=3, model_threshold=1)

        # Open 2 models but threshold is 3
        b1 = registry.get("claude-haiku-4-5-20251001")
        b1.record_failure(retryable=True)
        registry.record_model_open("claude-haiku-4-5-20251001")

        b2 = registry.get("claude-sonnet-4-6")
        b2.record_failure(retryable=True)
        registry.record_model_open("claude-sonnet-4-6")

        registry.check_provider("claude-opus-4-6")  # should not raise

    def test_different_providers_independent(self):
        registry = self._make_registry(threshold=1, model_threshold=1)

        # Trip anthropic
        b1 = registry.get("claude-haiku-4-5-20251001")
        b1.record_failure(retryable=True)
        registry.record_model_open("claude-haiku-4-5-20251001")

        # Anthropic tripped, OpenAI still fine
        with pytest.raises(CircuitOpenError):
            registry.check_provider("claude-sonnet-4-6")

        registry.check_provider("gpt-4o")  # should not raise

    def test_get_provider_state_closed_by_default(self):
        registry = self._make_registry()
        assert registry.get_provider_state("anthropic") == CircuitState.CLOSED

    def test_get_provider_state_open_after_trip(self):
        registry = self._make_registry(threshold=1, model_threshold=1)

        b1 = registry.get("claude-haiku-4-5-20251001")
        b1.record_failure(retryable=True)
        registry.record_model_open("claude-haiku-4-5-20251001")

        assert registry.get_provider_state("anthropic") == CircuitState.OPEN

    def test_provider_success_closes_half_open(self):
        config = ProviderConfig(
            circuit_breaker_threshold=1,
            circuit_breaker_reset_ms=60000,
            provider_circuit_breaker_threshold=1,
            provider_circuit_breaker_reset_ms=50,
        )
        registry = CircuitBreakerRegistry(config)

        b1 = registry.get("claude-haiku-4-5-20251001")
        b1.record_failure(retryable=True)
        registry.record_model_open("claude-haiku-4-5-20251001")

        assert registry.get_provider_state("anthropic") == CircuitState.OPEN
        time.sleep(0.06)
        assert registry.get_provider_state("anthropic") == CircuitState.HALF_OPEN

        registry.record_provider_success("claude-sonnet-4-6")
        assert registry.get_provider_state("anthropic") == CircuitState.CLOSED

    def test_provider_failure_reopens_half_open(self):
        config = ProviderConfig(
            circuit_breaker_threshold=1,
            circuit_breaker_reset_ms=60000,
            provider_circuit_breaker_threshold=1,
            provider_circuit_breaker_reset_ms=50,
        )
        registry = CircuitBreakerRegistry(config)

        b1 = registry.get("claude-haiku-4-5-20251001")
        b1.record_failure(retryable=True)
        registry.record_model_open("claude-haiku-4-5-20251001")

        time.sleep(0.06)
        assert registry.get_provider_state("anthropic") == CircuitState.HALF_OPEN

        registry.record_provider_failure("claude-sonnet-4-6")
        assert registry.get_provider_state("anthropic") == CircuitState.OPEN

    def test_record_model_open_for_unknown_provider_noop(self):
        registry = self._make_registry(threshold=1, model_threshold=1)

        b = registry.get("some-custom-model")
        b.record_failure(retryable=True)
        registry.record_model_open("some-custom-model")  # should not crash

        registry.check_provider("some-custom-model")  # should not raise

    def test_provider_map_in_registry(self):
        config = ProviderConfig(
            circuit_breaker_threshold=1,
            circuit_breaker_reset_ms=60000,
            provider_circuit_breaker_threshold=1,
            provider_circuit_breaker_reset_ms=60000,
            provider_map={"my-custom-llm": "anthropic"},
        )
        registry = CircuitBreakerRegistry(config)

        b = registry.get("my-custom-llm")
        b.record_failure(retryable=True)
        registry.record_model_open("my-custom-llm")

        # Provider breaker should trip for anthropic
        with pytest.raises(CircuitOpenError):
            registry.check_provider("claude-sonnet-4-6")
