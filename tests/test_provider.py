"""Tests for LiteLLM provider (mocked, no real API calls)."""

from unittest.mock import MagicMock, patch

import pytest

from mmrouter.models import ProviderConfig
from mmrouter.providers.litellm_provider import LiteLLMProvider, ProviderError


def _mock_response(content="Hello", model="claude-haiku", prompt_tokens=10, completion_tokens=5):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.model = model
    response.usage = MagicMock()
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    return response


class TestLiteLLMProvider:
    @patch("mmrouter.providers.litellm_provider.litellm")
    def test_successful_completion(self, mock_litellm):
        mock_litellm.completion.return_value = _mock_response()
        mock_litellm.completion_cost.return_value = 0.001

        provider = LiteLLMProvider()
        result = provider.complete("test prompt", "claude-haiku")

        assert result.content == "Hello"
        assert result.model == "claude-haiku"
        assert result.tokens_in == 10
        assert result.tokens_out == 5
        assert result.cost == 0.001
        assert result.latency_ms >= 0

    @patch("mmrouter.providers.litellm_provider.litellm")
    def test_cost_calculation_fallback(self, mock_litellm):
        mock_litellm.completion.return_value = _mock_response()
        mock_litellm.completion_cost.side_effect = Exception("cost calc failed")

        provider = LiteLLMProvider()
        result = provider.complete("test", "claude-haiku")

        assert result.cost == 0.0

    @patch("mmrouter.providers.litellm_provider.litellm")
    def test_permanent_error_no_retry(self, mock_litellm):
        mock_litellm.AuthenticationError = type("AuthenticationError", (Exception,), {})
        mock_litellm.completion.side_effect = mock_litellm.AuthenticationError("bad key")
        # Re-patch the module-level tuple
        import mmrouter.providers.litellm_provider as mod
        original = mod._PERMANENT_ERRORS
        mod._PERMANENT_ERRORS = (mock_litellm.AuthenticationError,)

        try:
            provider = LiteLLMProvider()
            with pytest.raises(ProviderError, match="Permanent error") as exc_info:
                provider.complete("test", "claude-haiku")
            assert not exc_info.value.retryable
            assert mock_litellm.completion.call_count == 1
        finally:
            mod._PERMANENT_ERRORS = original

    @patch("mmrouter.providers.litellm_provider.litellm")
    @patch("mmrouter.providers.litellm_provider.time")
    def test_transient_error_retries(self, mock_time, mock_litellm):
        mock_time.perf_counter.return_value = 0
        mock_litellm.RateLimitError = type("RateLimitError", (Exception,), {})
        mock_litellm.Timeout = type("Timeout", (Exception,), {})
        mock_litellm.ServiceUnavailableError = type("ServiceUnavailableError", (Exception,), {})
        mock_litellm.APIConnectionError = type("APIConnectionError", (Exception,), {})
        mock_litellm.AuthenticationError = type("AuthenticationError", (Exception,), {})
        mock_litellm.NotFoundError = type("NotFoundError", (Exception,), {})
        mock_litellm.BadRequestError = type("BadRequestError", (Exception,), {})

        import mmrouter.providers.litellm_provider as mod
        orig_transient = mod._TRANSIENT_ERRORS
        orig_permanent = mod._PERMANENT_ERRORS
        mod._TRANSIENT_ERRORS = (mock_litellm.RateLimitError,)
        mod._PERMANENT_ERRORS = (mock_litellm.AuthenticationError,)

        try:
            # Fail twice, succeed on third
            mock_litellm.completion.side_effect = [
                mock_litellm.RateLimitError("rate limited"),
                mock_litellm.RateLimitError("rate limited"),
                _mock_response(),
            ]
            mock_litellm.completion_cost.return_value = 0.001

            provider = LiteLLMProvider(ProviderConfig(max_retries=2))
            result = provider.complete("test", "claude-haiku")

            assert result.content == "Hello"
            assert mock_litellm.completion.call_count == 3
        finally:
            mod._TRANSIENT_ERRORS = orig_transient
            mod._PERMANENT_ERRORS = orig_permanent

    @patch("mmrouter.providers.litellm_provider.litellm")
    @patch("mmrouter.providers.litellm_provider.time")
    def test_transient_error_exhausted(self, mock_time, mock_litellm):
        mock_time.perf_counter.return_value = 0
        mock_litellm.RateLimitError = type("RateLimitError", (Exception,), {})
        mock_litellm.Timeout = type("Timeout", (Exception,), {})
        mock_litellm.ServiceUnavailableError = type("ServiceUnavailableError", (Exception,), {})
        mock_litellm.APIConnectionError = type("APIConnectionError", (Exception,), {})
        mock_litellm.AuthenticationError = type("AuthenticationError", (Exception,), {})
        mock_litellm.NotFoundError = type("NotFoundError", (Exception,), {})
        mock_litellm.BadRequestError = type("BadRequestError", (Exception,), {})

        import mmrouter.providers.litellm_provider as mod
        orig_transient = mod._TRANSIENT_ERRORS
        orig_permanent = mod._PERMANENT_ERRORS
        mod._TRANSIENT_ERRORS = (mock_litellm.RateLimitError,)
        mod._PERMANENT_ERRORS = (mock_litellm.AuthenticationError,)

        try:
            mock_litellm.completion.side_effect = mock_litellm.RateLimitError("rate limited")

            provider = LiteLLMProvider(ProviderConfig(max_retries=1))
            with pytest.raises(ProviderError, match="Failed after 2 attempts") as exc_info:
                provider.complete("test", "claude-haiku")
            assert exc_info.value.retryable
        finally:
            mod._TRANSIENT_ERRORS = orig_transient
            mod._PERMANENT_ERRORS = orig_permanent
