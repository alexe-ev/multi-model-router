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

        with patch("litellm.completion", return_value=[mock_chunk]):
            chunks = list(provider.stream_messages(
                [{"role": "user", "content": "hi"}], "gpt-4o"
            ))
            assert len(chunks) == 1
            assert chunks[0].content == "hello"
