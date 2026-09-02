"""Tests for LiteLLM provider error handling."""

from unittest.mock import MagicMock, patch

import litellm
import pytest

from mmrouter.models import ProviderConfig, StreamChunk
from mmrouter.providers.litellm_provider import LiteLLMProvider, ProviderError


@pytest.fixture
def provider():
    return LiteLLMProvider(ProviderConfig(max_retries=0))


class TestStreamMessagesErrorWrapping:
    """stream_messages wraps litellm exceptions into ProviderError."""

    def test_permanent_error_not_retryable(self, provider):
        with patch("litellm.completion", side_effect=litellm.AuthenticationError(
            message="Invalid key", model="test", llm_provider="openai"
        )):
            with pytest.raises(ProviderError, match="Permanent error") as exc_info:
                list(provider.stream_messages(
                    [{"role": "user", "content": "hi"}], "gpt-4o"
                ))
            assert exc_info.value.retryable is False

    def test_not_found_error_not_retryable(self, provider):
        with patch("litellm.completion", side_effect=litellm.NotFoundError(
            message="Model not found", model="test", llm_provider="openai"
        )):
            with pytest.raises(ProviderError, match="Permanent error") as exc_info:
                list(provider.stream_messages(
                    [{"role": "user", "content": "hi"}], "gpt-4o"
                ))
            assert exc_info.value.retryable is False

    def test_transient_error_retryable(self, provider):
        with patch("litellm.completion", side_effect=litellm.RateLimitError(
            message="Rate limited", model="test", llm_provider="openai"
        )):
            with pytest.raises(ProviderError, match="Transient error") as exc_info:
                list(provider.stream_messages(
                    [{"role": "user", "content": "hi"}], "gpt-4o"
                ))
            assert exc_info.value.retryable is True

    def test_timeout_error_retryable(self, provider):
        with patch("litellm.completion", side_effect=litellm.Timeout(
            message="Timeout", model="test", llm_provider="openai"
        )):
            with pytest.raises(ProviderError, match="Transient error") as exc_info:
                list(provider.stream_messages(
                    [{"role": "user", "content": "hi"}], "gpt-4o"
                ))
            assert exc_info.value.retryable is True

    def test_unexpected_error_not_retryable(self, provider):
        with patch("litellm.completion", side_effect=RuntimeError("Something broke")):
            with pytest.raises(ProviderError, match="Unexpected error") as exc_info:
                list(provider.stream_messages(
                    [{"role": "user", "content": "hi"}], "gpt-4o"
                ))
            assert exc_info.value.retryable is False

    def test_successful_stream(self, provider):
        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = "hello"
        mock_chunk.choices[0].finish_reason = None
        mock_chunk.model = "gpt-4o"
        mock_chunk.usage = None

        with patch("litellm.completion", return_value=[mock_chunk]):
            chunks = list(provider.stream_messages(
                [{"role": "user", "content": "hi"}], "gpt-4o"
            ))
            assert len(chunks) == 1
            assert chunks[0].content == "hello"


class TestStreamMessagesMidStreamPhaseMarker:
    """MMR-3 self-check 15 (provider half): ProviderError.mid_stream tells apart
    a stream that never opened from one that died mid-flight."""

    def test_open_time_failure_is_not_mid_stream(self, provider):
        """A failure raised by the call that OPENS the stream (never iterated)."""
        with patch("litellm.completion", side_effect=litellm.RateLimitError(
            message="Rate limited", model="test", llm_provider="openai"
        )):
            with pytest.raises(ProviderError) as exc_info:
                list(provider.stream_messages(
                    [{"role": "user", "content": "hi"}], "gpt-4o"
                ))
            assert exc_info.value.mid_stream is False
            assert exc_info.value.retryable is True

    def test_transient_failure_while_iterating_is_mid_stream(self, provider):
        """Closes KN-151: a transient litellm error raised while ITERATING
        (after at least one chunk) is translated with mid_stream=True."""
        def _dying_response():
            chunk = MagicMock()
            chunk.usage = None
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = "partial"
            chunk.choices[0].finish_reason = None
            chunk.model = "gpt-4o"
            yield chunk
            raise litellm.RateLimitError(
                message="dropped mid-stream", model="gpt-4o", llm_provider="openai"
            )

        with patch("litellm.completion", return_value=_dying_response()):
            with pytest.raises(ProviderError) as exc_info:
                list(provider.stream_messages(
                    [{"role": "user", "content": "hi"}], "gpt-4o"
                ))
            assert exc_info.value.mid_stream is True
            assert exc_info.value.retryable is True

    def test_unexpected_failure_while_iterating_is_mid_stream(self, provider):
        """A raw, untranslated exception (not one of litellm's own error types)
        raised while iterating is still caught and marked mid_stream=True --
        this is what closes AC9 for a real provider dying mid-stream."""
        def _dying_response():
            chunk = MagicMock()
            chunk.usage = None
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = "partial"
            chunk.choices[0].finish_reason = None
            chunk.model = "gpt-4o"
            yield chunk
            raise RuntimeError("connection dropped mid-stream")

        with patch("litellm.completion", return_value=_dying_response()):
            with pytest.raises(ProviderError) as exc_info:
                list(provider.stream_messages(
                    [{"role": "user", "content": "hi"}], "gpt-4o"
                ))
            assert exc_info.value.mid_stream is True
            assert exc_info.value.retryable is False


class TestStreamUsageCostParity:
    """MMR-3 self-check 12: the stream path and the non-stream path price the
    same tokens through the same litellm entry point (completion_cost) and
    must not drift apart."""

    def test_stream_and_non_stream_cost_match(self, provider):
        model = "claude-haiku-4-5-20251001"
        tokens_in, tokens_out = 270, 250

        usage = litellm.Usage(
            prompt_tokens=tokens_in, completion_tokens=tokens_out,
            total_tokens=tokens_in + tokens_out,
        )
        usage.cache_read_input_tokens = 0
        usage.cache_creation_input_tokens = 0

        # Stream path: usage read from raw chunks, priced via _stream_usage.
        raw_chunk = MagicMock()
        raw_chunk.usage = usage
        stream_response = MagicMock()
        stream_response.chunks = [raw_chunk]
        stream_usage = provider._stream_usage(stream_response, model)
        assert stream_usage is not None

        # Non-stream path: the same tokens, priced via the same
        # completion_cost call _call_messages uses (litellm_provider.py:210).
        # A real litellm.ModelResponse, not a MagicMock: completion_cost needs
        # a genuine `.model` field to look up pricing.
        non_stream_response = litellm.ModelResponse(model=model, usage=usage)
        with patch("litellm.completion", return_value=non_stream_response):
            non_stream_result = provider.complete_messages(
                [{"role": "user", "content": "hi"}], model
            )

        assert stream_usage.cost == non_stream_result.cost
        assert stream_usage.cost > 0


class TestStreamUsageAccumulation:
    """MMR-3 self-check 13: usage is accumulated PER FIELD, last non-zero wins."""

    def test_anthropics_split_usage_across_two_chunks(self, provider):
        """Anthropic reports usage across two events: message_start carries
        input_tokens, message_delta carries only output_tokens with
        prompt_tokens=0. Keeping the last whole object would log tokens_in=0."""
        model = "claude-haiku-4-5-20251001"
        usage1 = litellm.Usage(prompt_tokens=270, completion_tokens=1, total_tokens=271)
        usage2 = litellm.Usage(prompt_tokens=0, completion_tokens=250, total_tokens=250)

        chunk1 = MagicMock()
        chunk1.usage = usage1
        chunk2 = MagicMock()
        chunk2.usage = usage2

        response = MagicMock()
        response.chunks = [chunk1, chunk2]

        usage = provider._stream_usage(response, model)

        assert usage is not None
        assert usage.tokens_in == 270
        assert usage.tokens_out == 250


class _CountingStreamResponse:
    """Fake litellm streaming response: an iterator that pops from `_queue`
    and appends every raw item pulled into `.chunks` -- mirroring how
    litellm's real CustomStreamWrapper accumulates chunks as a side effect of
    iteration, growing rather than draining."""

    def __init__(self, items):
        self._queue = list(items)
        self.chunks: list = []
        self.pull_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.pull_count += 1
        if not self._queue:
            raise StopIteration
        item = self._queue.pop(0)
        self.chunks.append(item)
        return item


def _content_chunk(text, finish_reason=None, model="gpt-4o"):
    chunk = MagicMock()
    chunk.usage = None
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = text
    chunk.choices[0].finish_reason = finish_reason
    chunk.model = model
    return chunk


class TestStreamMessagesUsageChunkGuard:
    """MMR-3 self-check 8 (AC11 guard) and 14: the terminal usage-only chunk
    litellm returns produces no extra visible frame and is pulled exactly
    once -- via `break`, not `continue`."""

    def test_usage_chunk_yields_no_extra_frame(self, provider):
        usage_chunk = MagicMock()
        usage_chunk.usage = litellm.Usage(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )
        content_chunks = [_content_chunk("hello "), _content_chunk("world!", "stop")]
        response = _CountingStreamResponse(content_chunks + [usage_chunk])

        with patch("litellm.completion", return_value=response):
            chunks = list(provider.stream_messages(
                [{"role": "user", "content": "hi"}], "gpt-4o"
            ))

        # Only the two content chunks are visible -- the usage chunk added no frame.
        assert len(chunks) == 2
        assert [c.content for c in chunks] == ["hello ", "world!"]

    def test_break_pulls_the_iterator_exactly_once_for_the_usage_chunk(self, provider):
        usage_chunk = MagicMock()
        usage_chunk.usage = litellm.Usage(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )
        content_chunks = [_content_chunk("hello "), _content_chunk("world!", "stop")]
        response = _CountingStreamResponse(content_chunks + [usage_chunk])

        with patch("litellm.completion", return_value=response):
            list(provider.stream_messages([{"role": "user", "content": "hi"}], "gpt-4o"))

        # 2 content chunks + 1 pull that finds the usage chunk and breaks = 3.
        # `continue` instead of `break` would pull a 4th time (StopIteration)
        # before returning, re-running litellm's whole finalisation.
        assert response.pull_count == 3
