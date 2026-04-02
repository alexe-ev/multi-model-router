"""Tests for provider messages and streaming interface."""

from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest

from mmrouter.models import CompletionResult, ProviderConfig, StreamChunk
from mmrouter.providers.base import ProviderBase, _extract_last_user_message
from mmrouter.providers.litellm_provider import LiteLLMProvider, ProviderError


class TestExtractLastUserMessage:
    def test_single_user_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        assert _extract_last_user_message(messages) == "Hello"

    def test_multi_turn(self):
        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Second"},
        ]
        assert _extract_last_user_message(messages) == "Second"

    def test_system_and_user(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        assert _extract_last_user_message(messages) == "Hi"

    def test_no_user_message(self):
        messages = [{"role": "system", "content": "Hello"}]
        assert _extract_last_user_message(messages) == ""

    def test_empty_messages(self):
        assert _extract_last_user_message([]) == ""

    def test_list_content_format(self):
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "Look at this"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
            ]}
        ]
        assert _extract_last_user_message(messages) == "Look at this"


class ConcreteProvider(ProviderBase):
    """Minimal concrete provider for testing base class defaults."""

    def complete(self, prompt, model, **kwargs):
        return CompletionResult(
            content=f"Echo: {prompt}",
            model=model,
            tokens_in=5,
            tokens_out=10,
            cost=0.0,
            latency_ms=1.0,
        )


class TestProviderBaseDefaults:
    def test_complete_messages_default_extracts_user_message(self):
        provider = ConcreteProvider()
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hello world"},
        ]
        result = provider.complete_messages(messages, "test-model")
        assert result.content == "Echo: Hello world"

    def test_stream_messages_default_yields_single_chunk(self):
        provider = ConcreteProvider()
        messages = [{"role": "user", "content": "Hi"}]
        chunks = list(provider.stream_messages(messages, "test-model"))
        assert len(chunks) == 1
        assert chunks[0].content == "Echo: Hi"
        assert chunks[0].finish_reason == "stop"


def _mock_response(content="Hello", model="claude-haiku", prompt_tokens=10, completion_tokens=5):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.model = model
    response.usage = MagicMock()
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    response.usage.cache_read_input_tokens = 0
    response.usage.cache_creation_input_tokens = 0
    return response


class TestLiteLLMProviderMessages:
    @patch("mmrouter.providers.litellm_provider.litellm")
    def test_complete_messages_sends_full_array(self, mock_litellm):
        mock_litellm.completion.return_value = _mock_response()
        mock_litellm.completion_cost.return_value = 0.001

        provider = LiteLLMProvider()
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hi"},
        ]
        result = provider.complete_messages(messages, "claude-haiku")

        # Verify litellm.completion was called with the full messages array
        # System message gets cache_control annotation for Anthropic models
        call_kwargs = mock_litellm.completion.call_args
        sent_messages = call_kwargs.kwargs["messages"]
        assert len(sent_messages) == 2
        assert sent_messages[0]["role"] == "system"
        assert sent_messages[0]["content"] == "Be helpful"
        assert sent_messages[0]["cache_control"] == {"type": "ephemeral"}
        assert sent_messages[1] == {"role": "user", "content": "Hi"}
        assert result.content == "Hello"

    @patch("mmrouter.providers.litellm_provider.litellm")
    def test_complete_messages_no_annotation_when_disabled(self, mock_litellm):
        mock_litellm.completion.return_value = _mock_response()
        mock_litellm.completion_cost.return_value = 0.001

        provider = LiteLLMProvider(ProviderConfig(prompt_caching=False))
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hi"},
        ]
        result = provider.complete_messages(messages, "claude-haiku")

        call_kwargs = mock_litellm.completion.call_args
        assert call_kwargs.kwargs["messages"] == messages
        assert result.content == "Hello"

    @patch("mmrouter.providers.litellm_provider.litellm")
    def test_complete_messages_retries_transient_errors(self, mock_litellm):
        mock_litellm.RateLimitError = type("RateLimitError", (Exception,), {})
        mock_litellm.Timeout = type("Timeout", (Exception,), {})
        mock_litellm.ServiceUnavailableError = type("ServiceUnavailableError", (Exception,), {})
        mock_litellm.APIConnectionError = type("APIConnectionError", (Exception,), {})
        mock_litellm.AuthenticationError = type("AuthenticationError", (Exception,), {})
        mock_litellm.NotFoundError = type("NotFoundError", (Exception,), {})
        mock_litellm.BadRequestError = type("BadRequestError", (Exception,), {})

        import mmrouter.providers.litellm_provider as mod
        orig = mod._TRANSIENT_ERRORS
        mod._TRANSIENT_ERRORS = (mock_litellm.RateLimitError,)

        try:
            mock_litellm.completion.side_effect = [
                mock_litellm.RateLimitError("rate limited"),
                _mock_response(),
            ]
            mock_litellm.completion_cost.return_value = 0.001

            provider = LiteLLMProvider(ProviderConfig(max_retries=1))
            result = provider.complete_messages(
                [{"role": "user", "content": "Hi"}], "claude-haiku"
            )
            assert result.content == "Hello"
            assert mock_litellm.completion.call_count == 2
        finally:
            mod._TRANSIENT_ERRORS = orig

    @patch("mmrouter.providers.litellm_provider.litellm")
    def test_stream_messages_yields_chunks(self, mock_litellm):
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello "
        chunk1.choices[0].finish_reason = None
        chunk1.model = "claude-haiku"

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = "world!"
        chunk2.choices[0].finish_reason = "stop"
        chunk2.model = "claude-haiku"

        mock_litellm.completion.return_value = [chunk1, chunk2]

        provider = LiteLLMProvider()
        chunks = list(provider.stream_messages(
            [{"role": "user", "content": "Hi"}], "claude-haiku"
        ))

        assert len(chunks) == 2
        assert chunks[0].content == "Hello "
        assert chunks[0].finish_reason is None
        assert chunks[1].content == "world!"
        assert chunks[1].finish_reason == "stop"

        # Verify stream=True was passed
        call_kwargs = mock_litellm.completion.call_args
        assert call_kwargs.kwargs["stream"] is True
