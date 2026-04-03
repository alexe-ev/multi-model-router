"""LiteLLM provider: wraps litellm behind ProviderBase interface."""

from __future__ import annotations

import time
from typing import Iterator

import litellm

from mmrouter.models import CompletionResult, ProviderConfig, StreamChunk
from mmrouter.providers.base import ProviderBase
from mmrouter.providers.cache import annotate_cache_control

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True

# Transient error types that should be retried
_TRANSIENT_ERRORS = (
    litellm.RateLimitError,
    litellm.Timeout,
    litellm.ServiceUnavailableError,
    litellm.APIConnectionError,
)

# Permanent errors that should not be retried
_PERMANENT_ERRORS = (
    litellm.AuthenticationError,
    litellm.NotFoundError,
    litellm.BadRequestError,
)


class ProviderError(Exception):
    """Raised when provider call fails after all retries."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class LiteLLMProvider(ProviderBase):
    """LLM provider using litellm for multi-provider access."""

    def __init__(self, config: ProviderConfig | None = None):
        self._config = config or ProviderConfig()

    def complete(self, prompt: str, model: str, **kwargs) -> CompletionResult:
        last_error = None

        for attempt in range(self._config.max_retries + 1):
            try:
                return self._call(prompt, model, **kwargs)
            except _PERMANENT_ERRORS as e:
                raise ProviderError(
                    f"Permanent error from {model}: {e}", retryable=False
                ) from e
            except _TRANSIENT_ERRORS as e:
                last_error = e
                if attempt < self._config.max_retries:
                    delay = 2**attempt * 0.5  # 0.5s, 1s, 2s...
                    time.sleep(delay)
            except Exception as e:
                raise ProviderError(
                    f"Unexpected error from {model}: {e}", retryable=False
                ) from e

        raise ProviderError(
            f"Failed after {self._config.max_retries + 1} attempts: {last_error}",
            retryable=True,
        ) from last_error

    def complete_messages(self, messages: list[dict], model: str, **kwargs) -> CompletionResult:
        last_error = None

        for attempt in range(self._config.max_retries + 1):
            try:
                return self._call_messages(messages, model, **kwargs)
            except _PERMANENT_ERRORS as e:
                raise ProviderError(
                    f"Permanent error from {model}: {e}", retryable=False
                ) from e
            except _TRANSIENT_ERRORS as e:
                last_error = e
                if attempt < self._config.max_retries:
                    delay = 2**attempt * 0.5
                    time.sleep(delay)
            except Exception as e:
                raise ProviderError(
                    f"Unexpected error from {model}: {e}", retryable=False
                ) from e

        raise ProviderError(
            f"Failed after {self._config.max_retries + 1} attempts: {last_error}",
            retryable=True,
        ) from last_error

    def stream_messages(self, messages: list[dict], model: str, **kwargs) -> Iterator[StreamChunk]:
        """Stream response chunks for a messages array."""
        # Apply prompt caching annotation if enabled
        if self._config.prompt_caching:
            messages = annotate_cache_control(
                messages, model, self._config.provider_map or None
            )

        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                stream=True,
                timeout=self._config.timeout_ms / 1000,
                **kwargs,
            )
        except _PERMANENT_ERRORS as e:
            raise ProviderError(
                f"Permanent error from {model}: {e}", retryable=False
            ) from e
        except _TRANSIENT_ERRORS as e:
            raise ProviderError(
                f"Transient error from {model}: {e}", retryable=True
            ) from e
        except Exception as e:
            raise ProviderError(
                f"Unexpected error from {model}: {e}", retryable=False
            ) from e

        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = delta.content if delta and delta.content else ""
            finish_reason = chunk.choices[0].finish_reason
            yield StreamChunk(
                content=content,
                model=chunk.model or model,
                finish_reason=finish_reason,
            )

    def _call(self, prompt: str, model: str, **kwargs) -> CompletionResult:
        start = time.perf_counter()

        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            timeout=self._config.timeout_ms / 1000,
            **kwargs,
        )

        latency_ms = (time.perf_counter() - start) * 1000

        usage = response.usage
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0

        # Extract cache metrics from usage
        cache_read_tokens = 0
        cache_creation_tokens = 0
        if usage:
            cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
            cache_creation_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0

        # Try litellm's cost calculation, fall back to 0
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            cost = 0.0

        content = response.choices[0].message.content or ""

        return CompletionResult(
            content=content,
            model=response.model or model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=cost,
            latency_ms=round(latency_ms, 1),
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
        )

    def _call_messages(self, messages: list[dict], model: str, **kwargs) -> CompletionResult:
        start = time.perf_counter()

        # Apply prompt caching annotation if enabled
        if self._config.prompt_caching:
            messages = annotate_cache_control(
                messages, model, self._config.provider_map or None
            )

        response = litellm.completion(
            model=model,
            messages=messages,
            timeout=self._config.timeout_ms / 1000,
            **kwargs,
        )

        latency_ms = (time.perf_counter() - start) * 1000

        usage = response.usage
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0

        # Extract cache metrics from usage (Anthropic returns these via LiteLLM)
        cache_read_tokens = 0
        cache_creation_tokens = 0
        if usage:
            cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
            cache_creation_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0

        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            cost = 0.0

        content = response.choices[0].message.content or ""

        return CompletionResult(
            content=content,
            model=response.model or model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=cost,
            latency_ms=round(latency_ms, 1),
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
        )
