"""LiteLLM provider: wraps litellm behind ProviderBase interface."""

from __future__ import annotations

import time
from typing import Generator, Iterator

import litellm

from mmrouter.models import CompletionResult, ProviderConfig, StreamChunk, StreamUsage
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

    def __init__(self, message: str, *, retryable: bool = False, mid_stream: bool = False):
        super().__init__(message)
        self.retryable = retryable
        # True only for a stream that had already opened. The caller uses this to
        # decide whether an unrecorded stream may still have cost money.
        self.mid_stream = mid_stream


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

    def stream_messages(
        self, messages: list[dict], model: str, **kwargs
    ) -> Generator[StreamChunk, None, StreamUsage | None]:
        """Stream response chunks for a messages array."""
        # Apply prompt caching annotation if enabled
        if self._config.prompt_caching:
            messages = annotate_cache_control(
                messages, model, self._config.provider_map or None
            )

        # OpenAI only reports usage on a stream when asked; Anthropic and Gemini
        # always do. litellm drops the option for providers that do not take it
        # (utils.py, the `k == "stream_options"` exemption in _check_valid_arg),
        # verified 2026-08-28 on 1.82.6: anthropic and gemini come back without it.
        kwargs.setdefault("stream_options", {"include_usage": True})

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

        yield from self._iter_stream(response, model)
        return self._stream_usage(response, model)

    def _iter_stream(self, response, model: str) -> Iterator[StreamChunk]:
        """Yield content chunks, translating a failure raised while iterating.

        This closes the open half of KN-151 (`docs/integrations.md`): the
        try/except above wraps only the call that OPENS the stream, so a
        provider dying mid-stream reached the server as a raw litellm
        exception. `generate()` in `server/app.py` catches only
        `ProviderError`, `RuntimeError` and `BudgetExceededError`, so a raw
        litellm exception became a truncated SSE stream with no error frame
        and no `[DONE]` -- which is what AC9 promises does not happen.
        """
        try:
            for chunk in response:
                # litellm strips usage from every ordinary chunk and returns it on
                # one terminal chunk of its own, whose `choices` is non-empty with
                # a None content -- so the `not chunk.choices` filter below does
                # NOT catch it, and yielding it would add an empty frame to the
                # SSE stream, on the passthrough path too.
                #
                # `break`, not `continue`: pulling again re-enters litellm's
                # `except StopIteration` handler, which re-runs its whole
                # finalisation -- a second stream_chunk_builder pass, a second
                # cache write and a second success callback per stream.
                if getattr(chunk, "usage", None) is not None:
                    break
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                content = delta.content if delta and delta.content else ""
                yield StreamChunk(
                    content=content,
                    model=chunk.model or model,
                    finish_reason=chunk.choices[0].finish_reason,
                )
        except ProviderError:
            raise
        except _TRANSIENT_ERRORS as e:
            raise ProviderError(
                f"Transient error from {model} mid-stream: {e}",
                retryable=True,
                mid_stream=True,
            ) from e
        except Exception as e:
            raise ProviderError(
                f"Unexpected error from {model} mid-stream: {e}",
                retryable=False,
                mid_stream=True,
            ) from e

    _USAGE_FIELDS = (
        "prompt_tokens",
        "completion_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    )

    def _stream_usage(self, response, model: str) -> StreamUsage | None:
        """Usage the provider actually sent, or None if it sent none.

        Accumulated PER FIELD, last non-zero wins. That is the rule litellm's
        own accumulators use (`streaming_chunk_builder_utils.py`, the
        `usage_chunk_dict[...] > 0` guards), and it is not optional: Anthropic
        reports usage across TWO events. `message_start` carries
        `input_tokens`; `message_delta` carries only `output_tokens` and
        converts to a Usage with `prompt_tokens = 0`
        (`llms/anthropic/chat/transformation.py`, `calculate_usage`). Keeping
        the last whole object would log `tokens_in = 0` for every Anthropic
        stream -- the repo's default provider family.

        Read from the raw chunks litellm accumulated, never from
        `stream_chunk_builder`: it computes `prompt_tokens or token_counter(...)`
        and the same for completion tokens, substituting a local tokenizer
        estimate indistinguishable from a billed count.
        """
        totals = dict.fromkeys(self._USAGE_FIELDS, 0)
        reported = False
        for raw in getattr(response, "chunks", None) or ():
            usage = getattr(raw, "usage", None)
            if usage is None:
                continue
            reported = True
            for name in self._USAGE_FIELDS:
                value = getattr(usage, name, 0) or 0
                if value > 0:
                    totals[name] = value
        if not reported:
            return None

        usage = litellm.Usage(
            prompt_tokens=totals["prompt_tokens"],
            completion_tokens=totals["completion_tokens"],
            total_tokens=totals["prompt_tokens"] + totals["completion_tokens"],
        )
        for name in ("cache_read_input_tokens", "cache_creation_input_tokens"):
            setattr(usage, name, totals[name])
        try:
            cost = litellm.completion_cost(
                completion_response=litellm.ModelResponse(model=model, usage=usage)
            )
        except Exception:
            # Same rule `_call` and `_call_messages` already follow: a model
            # litellm cannot price is logged at $0. See
            # decisions/2026-08-28-streamed-request-accounting.md.
            cost = 0.0
        return StreamUsage(
            tokens_in=totals["prompt_tokens"],
            tokens_out=totals["completion_tokens"],
            cost=cost,
            cache_read_tokens=totals["cache_read_input_tokens"],
            cache_creation_tokens=totals["cache_creation_input_tokens"],
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
