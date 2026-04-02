"""Circuit breaker: per-model failure tracking and automatic skip."""

from __future__ import annotations

import time
from enum import StrEnum

from mmrouter.models import ProviderConfig


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a circuit breaker is open for a model."""

    def __init__(self, model: str, retry_after_ms: float):
        super().__init__(f"Circuit open for {model}, retry after {retry_after_ms:.0f}ms")
        self.model = model
        self.retry_after_ms = retry_after_ms


class CircuitBreaker:
    """Per-model circuit breaker. Tracks transient failures, skips known-bad models."""

    def __init__(self, model: str, config: ProviderConfig):
        self._model = model
        self._threshold = config.circuit_breaker_threshold
        self._reset_ms = config.circuit_breaker_reset_ms
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at: float = 0.0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            elapsed_ms = (time.monotonic() - self._opened_at) * 1000
            if elapsed_ms >= self._reset_ms:
                return CircuitState.HALF_OPEN
        return self._state

    def check(self) -> None:
        current = self.state
        if current == CircuitState.OPEN:
            elapsed_ms = (time.monotonic() - self._opened_at) * 1000
            retry_after_ms = self._reset_ms - elapsed_ms
            raise CircuitOpenError(self._model, retry_after_ms)

    def record_success(self) -> None:
        self._failure_count = 0
        self._state = CircuitState.CLOSED

    def record_failure(self, retryable: bool) -> None:
        if not retryable:
            return
        self._failure_count += 1
        if self.state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            return
        if self._failure_count >= self._threshold:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()


class CircuitBreakerRegistry:
    """Lazy registry of per-model circuit breakers."""

    def __init__(self, config: ProviderConfig):
        self._config = config
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(self, model: str) -> CircuitBreaker:
        if model not in self._breakers:
            self._breakers[model] = CircuitBreaker(model, self._config)
        return self._breakers[model]
