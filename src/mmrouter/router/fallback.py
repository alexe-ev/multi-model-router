"""Circuit breaker: per-model and per-provider failure tracking and automatic skip."""

from __future__ import annotations

import time
from enum import StrEnum

from mmrouter.models import ProviderConfig


_PREFIX_TO_PROVIDER: list[tuple[str, str]] = [
    ("claude-", "anthropic"),
    ("gpt-", "openai"),
    ("o1-", "openai"),
    ("o3-", "openai"),
    ("chatgpt-", "openai"),
    ("gemini-", "google"),
]


def extract_provider(model: str, provider_map: dict[str, str] | None = None) -> str:
    """Map a model name to its provider string.

    Checks explicit provider_map first, then falls back to prefix heuristics.
    """
    if provider_map and model in provider_map:
        return provider_map[model]
    for prefix, provider in _PREFIX_TO_PROVIDER:
        if model.startswith(prefix):
            return provider
    return "unknown"


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


class ProviderCircuitBreaker:
    """Provider-level circuit breaker. Trips when multiple models from the same provider are OPEN."""

    def __init__(self, provider: str, config: ProviderConfig):
        self._provider = provider
        self._reset_ms = config.provider_circuit_breaker_reset_ms
        self._state = CircuitState.CLOSED
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
            raise CircuitOpenError(self._provider, retry_after_ms)

    def trip(self) -> None:
        """Force the provider breaker to OPEN state."""
        self._state = CircuitState.OPEN
        self._opened_at = time.monotonic()

    def record_success(self) -> None:
        self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Re-open after a half-open probe fails."""
        if self.state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()


class CircuitBreakerRegistry:
    """Lazy registry of per-model and per-provider circuit breakers."""

    def __init__(self, config: ProviderConfig):
        self._config = config
        self._breakers: dict[str, CircuitBreaker] = {}
        self._provider_breakers: dict[str, ProviderCircuitBreaker] = {}
        self._provider_threshold = config.provider_circuit_breaker_threshold

    def get(self, model: str) -> CircuitBreaker:
        if model not in self._breakers:
            self._breakers[model] = CircuitBreaker(model, self._config)
        return self._breakers[model]

    def _get_provider_breaker(self, provider: str) -> ProviderCircuitBreaker:
        if provider not in self._provider_breakers:
            self._provider_breakers[provider] = ProviderCircuitBreaker(provider, self._config)
        return self._provider_breakers[provider]

    def check_provider(self, model: str) -> None:
        """Raise CircuitOpenError if the provider for this model is tripped."""
        provider = extract_provider(model, self._config.provider_map)
        if provider == "unknown":
            return
        breaker = self._get_provider_breaker(provider)
        breaker.check()

    def record_model_open(self, model: str) -> None:
        """Called when a per-model breaker opens. Checks if provider threshold is met."""
        provider = extract_provider(model, self._config.provider_map)
        if provider == "unknown":
            return

        open_count = sum(
            1 for m, b in self._breakers.items()
            if extract_provider(m, self._config.provider_map) == provider
            and b.state == CircuitState.OPEN
        )

        if open_count >= self._provider_threshold:
            self._get_provider_breaker(provider).trip()

    def record_provider_success(self, model: str) -> None:
        """Called on a successful call to potentially close the provider breaker."""
        provider = extract_provider(model, self._config.provider_map)
        if provider == "unknown":
            return
        if provider in self._provider_breakers:
            pb = self._provider_breakers[provider]
            if pb.state == CircuitState.HALF_OPEN:
                pb.record_success()

    def record_provider_failure(self, model: str) -> None:
        """Called on failure during half-open probe to re-open provider breaker."""
        provider = extract_provider(model, self._config.provider_map)
        if provider == "unknown":
            return
        if provider in self._provider_breakers:
            self._provider_breakers[provider].record_failure()

    def get_provider_state(self, provider: str) -> CircuitState:
        """Get the current state of a provider-level breaker."""
        if provider not in self._provider_breakers:
            return CircuitState.CLOSED
        return self._provider_breakers[provider].state
