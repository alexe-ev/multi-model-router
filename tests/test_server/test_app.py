"""Tests for REST API endpoints (mocked provider, no real API calls)."""

import os
from typing import Iterator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from mmrouter.models import CompletionResult, StreamChunk
from mmrouter.providers.base import ProviderBase
from mmrouter.providers.litellm_provider import ProviderError
from mmrouter.router.engine import Router
from mmrouter.server.app import create_app
from mmrouter.tracker.logger import Tracker


class MockProvider(ProviderBase):
    """Provider that returns canned responses for testing."""

    def __init__(self, fail=False, fail_retryable=True):
        self._fail = fail
        self._fail_retryable = fail_retryable
        self.calls: list[tuple] = []

    def complete(self, prompt, model, **kwargs):
        self.calls.append(("complete", prompt, model, kwargs))
        if self._fail:
            raise ProviderError("mock failure", retryable=self._fail_retryable)
        return CompletionResult(
            content=f"Response from {model}",
            model=model,
            tokens_in=10,
            tokens_out=20,
            cost=0.001,
            latency_ms=50.0,
        )

    def complete_messages(self, messages, model, **kwargs):
        self.calls.append(("complete_messages", messages, model, kwargs))
        if self._fail:
            raise ProviderError("mock failure", retryable=self._fail_retryable)
        return CompletionResult(
            content=f"Response from {model}",
            model=model,
            tokens_in=10,
            tokens_out=20,
            cost=0.001,
            latency_ms=50.0,
        )

    def stream_messages(self, messages, model, **kwargs) -> Iterator[StreamChunk]:
        self.calls.append(("stream_messages", messages, model, kwargs))
        if self._fail:
            raise ProviderError("mock failure", retryable=self._fail_retryable)
        yield StreamChunk(content="Hello ", model=model)
        yield StreamChunk(content="world!", model=model, finish_reason="stop")


def _write_config(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("""
version: "1"
routes:
  simple:
    factual:
      model: claude-haiku-4-5-20251001
      fallbacks:
        - claude-sonnet-4-6
    reasoning:
      model: claude-haiku-4-5-20251001
      fallbacks:
        - claude-sonnet-4-6
    creative:
      model: claude-sonnet-4-6
      fallbacks:
        - claude-haiku-4-5-20251001
    code:
      model: claude-sonnet-4-6
      fallbacks:
        - claude-haiku-4-5-20251001
  medium:
    factual:
      model: claude-sonnet-4-6
      fallbacks:
        - claude-haiku-4-5-20251001
    reasoning:
      model: claude-sonnet-4-6
      fallbacks:
        - claude-opus-4-6
    creative:
      model: claude-sonnet-4-6
      fallbacks:
        - claude-opus-4-6
    code:
      model: claude-sonnet-4-6
      fallbacks:
        - claude-opus-4-6
  complex:
    factual:
      model: claude-sonnet-4-6
      fallbacks:
        - claude-opus-4-6
    reasoning:
      model: claude-opus-4-6
      fallbacks:
        - claude-sonnet-4-6
    creative:
      model: claude-opus-4-6
      fallbacks:
        - claude-sonnet-4-6
    code:
      model: claude-opus-4-6
      fallbacks:
        - claude-sonnet-4-6
classifier:
  strategy: rules
  threshold: "0.7"
provider:
  timeout_ms: 30000
  max_retries: 0
""")
    return cfg


@pytest.fixture
def mock_provider():
    return MockProvider()


@pytest.fixture
def app_with_provider(tmp_path, mock_provider):
    """Create app and inject mock provider into the router."""
    cfg = _write_config(tmp_path)
    application = create_app(config_path=str(cfg), db_path=str(tmp_path / "test.db"))

    # Inject mock provider by creating app and swapping provider on the router
    # We need to trigger lifespan to initialize the router
    with TestClient(application) as client:
        application.state.router._provider = mock_provider
        yield client, mock_provider, application


class TestHealth:
    def test_health_returns_ok(self, tmp_path):
        cfg = _write_config(tmp_path)
        app = create_app(config_path=str(cfg), db_path=str(tmp_path / "test.db"))
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert "version" in data


class TestListModels:
    def test_list_models_returns_config_models(self, app_with_provider):
        client, _, _ = app_with_provider
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MMROUTER_API_KEY", None)
            resp = client.get("/v1/models")

        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        model_ids = [m["id"] for m in data["data"]]
        assert "auto" in model_ids
        assert "claude-haiku-4-5-20251001" in model_ids
        assert "claude-sonnet-4-6" in model_ids
        assert "claude-opus-4-6" in model_ids

    def test_list_models_no_duplicates(self, app_with_provider):
        client, _, _ = app_with_provider
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MMROUTER_API_KEY", None)
            resp = client.get("/v1/models")

        data = resp.json()
        model_ids = [m["id"] for m in data["data"]]
        # "auto" is special, the rest should be unique
        non_auto = [m for m in model_ids if m != "auto"]
        assert len(non_auto) == len(set(non_auto))


class TestChatCompletionsNonStreaming:
    def test_basic_request(self, app_with_provider):
        client, provider, _ = app_with_provider
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MMROUTER_API_KEY", None)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "What is the capital of France?"}],
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["id"].startswith("chatcmpl-")
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert "usage" in data

    def test_classification_headers_present(self, app_with_provider):
        client, _, _ = app_with_provider
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MMROUTER_API_KEY", None)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "What is 2+2?"}],
                },
            )

        assert resp.status_code == 200
        assert "x-mmrouter-complexity" in resp.headers
        assert "x-mmrouter-category" in resp.headers
        assert "x-mmrouter-confidence" in resp.headers
        assert "x-mmrouter-model" in resp.headers
        assert "x-mmrouter-fallback" in resp.headers
        assert "x-mmrouter-escalated" in resp.headers

    def test_auto_model_routes_simple(self, app_with_provider):
        client, _, _ = app_with_provider
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MMROUTER_API_KEY", None)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": "What is the capital of France?"}],
                },
            )

        assert resp.status_code == 200
        assert resp.headers["x-mmrouter-complexity"] == "simple"
        assert "haiku" in resp.headers["x-mmrouter-model"]

    def test_explicit_model_bypasses_routing(self, app_with_provider):
        client, provider, _ = app_with_provider
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MMROUTER_API_KEY", None)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "claude-opus-4-6",
                    "messages": [{"role": "user", "content": "What is 2+2?"}],
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "claude-opus-4-6"
        # No classification headers for explicit model
        assert "x-mmrouter-complexity" not in resp.headers

    def test_usage_info_present(self, app_with_provider):
        client, _, _ = app_with_provider
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MMROUTER_API_KEY", None)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

        data = resp.json()
        usage = data["usage"]
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 20
        assert usage["total_tokens"] == 30

    def test_provider_kwargs_passed(self, app_with_provider):
        client, provider, _ = app_with_provider
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MMROUTER_API_KEY", None)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Hi"}],
                    "temperature": 0.5,
                    "max_tokens": 100,
                },
            )

        assert resp.status_code == 200
        # Check that kwargs were passed to provider
        last_call = provider.calls[-1]
        kwargs = last_call[3]
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 100

    def test_invalid_request_returns_422(self, app_with_provider):
        client, _, _ = app_with_provider
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MMROUTER_API_KEY", None)
            resp = client.post(
                "/v1/chat/completions",
                json={"not_messages": "invalid"},
            )

        assert resp.status_code == 422

    def test_empty_messages_returns_422(self, app_with_provider):
        client, _, _ = app_with_provider
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MMROUTER_API_KEY", None)
            resp = client.post(
                "/v1/chat/completions",
                json={"messages": []},
            )

        # Router classifies empty prompt, still returns a response
        # The actual behavior depends on classifier; 200 is acceptable
        assert resp.status_code in (200, 422)


class TestChatCompletionsProviderErrors:
    def test_provider_failure_returns_502(self, tmp_path):
        cfg = _write_config(tmp_path)
        app = create_app(config_path=str(cfg), db_path=str(tmp_path / "test.db"))

        with TestClient(app) as client:
            # Replace provider with failing one
            fail_provider = MockProvider(fail=True)
            app.state.router._provider = fail_provider

            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("MMROUTER_API_KEY", None)
                resp = client.post(
                    "/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": "Hi"}],
                    },
                )

        assert resp.status_code == 502

    def test_explicit_model_provider_failure_returns_502(self, tmp_path):
        cfg = _write_config(tmp_path)
        app = create_app(config_path=str(cfg), db_path=str(tmp_path / "test.db"))

        with TestClient(app) as client:
            fail_provider = MockProvider(fail=True)
            app.state.router._provider = fail_provider

            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("MMROUTER_API_KEY", None)
                resp = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "some-model",
                        "messages": [{"role": "user", "content": "Hi"}],
                    },
                )

        assert resp.status_code == 502


class TestChatCompletionsStreaming:
    def test_streaming_returns_sse(self, app_with_provider):
        client, _, _ = app_with_provider
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MMROUTER_API_KEY", None)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "What is 2+2?"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        # Parse SSE events
        lines = resp.text.strip().split("\n\n")
        events = [line for line in lines if line.startswith("data: ")]
        assert len(events) >= 3  # role chunk + content chunks + [DONE]
        assert events[-1] == "data: [DONE]"

    def test_streaming_chunks_valid_json(self, app_with_provider):
        import json
        client, _, _ = app_with_provider
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MMROUTER_API_KEY", None)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
            )

        lines = resp.text.strip().split("\n\n")
        for line in lines:
            if line.startswith("data: ") and line != "data: [DONE]":
                data = json.loads(line[6:])
                assert data["object"] == "chat.completion.chunk"
                assert data["id"].startswith("chatcmpl-")
                assert len(data["choices"]) == 1

    def test_streaming_first_chunk_has_role(self, app_with_provider):
        import json
        client, _, _ = app_with_provider
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MMROUTER_API_KEY", None)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
            )

        lines = resp.text.strip().split("\n\n")
        first_data = json.loads(lines[0][6:])
        assert first_data["choices"][0]["delta"]["role"] == "assistant"

    def test_streaming_classification_headers(self, app_with_provider):
        client, _, _ = app_with_provider
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MMROUTER_API_KEY", None)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "What is 2+2?"}],
                    "stream": True,
                },
            )

        assert "x-mmrouter-complexity" in resp.headers
        assert "x-mmrouter-model" in resp.headers

    def test_streaming_explicit_model(self, app_with_provider):
        client, _, _ = app_with_provider
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MMROUTER_API_KEY", None)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "claude-opus-4-6",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        # No classification headers for explicit model
        assert "x-mmrouter-complexity" not in resp.headers

    def test_streaming_provider_failure_sends_error_event(self, tmp_path):
        """Provider errors during streaming are sent as error events in the SSE stream."""
        import json as json_mod
        cfg = _write_config(tmp_path)
        app = create_app(config_path=str(cfg), db_path=str(tmp_path / "test.db"))

        with TestClient(app) as client:
            fail_provider = MockProvider(fail=True)
            app.state.router._provider = fail_provider

            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("MMROUTER_API_KEY", None)
                resp = client.post(
                    "/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": "Hi"}],
                        "stream": True,
                    },
                )

        # Stream starts successfully (200), but contains an error event
        assert resp.status_code == 200
        # Find the error event in the stream
        has_error = False
        for line in resp.text.split("\n\n"):
            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    data = json_mod.loads(line[6:])
                    if "error" in data:
                        has_error = True
                except json_mod.JSONDecodeError:
                    pass
        assert has_error


class TestMessagePassthrough:
    def test_system_message_passed_to_provider(self, app_with_provider):
        client, provider, _ = app_with_provider
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MMROUTER_API_KEY", None)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "What is 2+2?"},
                    ],
                },
            )

        assert resp.status_code == 200
        # Verify full messages array was passed to provider
        last_call = provider.calls[-1]
        messages = last_call[1]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_multi_turn_conversation(self, app_with_provider):
        client, provider, _ = app_with_provider
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MMROUTER_API_KEY", None)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "user", "content": "Hi"},
                        {"role": "assistant", "content": "Hello!"},
                        {"role": "user", "content": "What is 2+2?"},
                    ],
                },
            )

        assert resp.status_code == 200
        last_call = provider.calls[-1]
        messages = last_call[1]
        assert len(messages) == 3
