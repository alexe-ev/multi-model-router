"""Tests for circuit breaker and registry."""

import time

import pytest

from mmrouter.models import ProviderConfig
from mmrouter.router.fallback import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
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
