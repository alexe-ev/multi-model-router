"""Tests for POST /v1/feedback endpoint and X-MMRouter-Request-Id header."""

import os
from typing import Iterator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from mmrouter.models import CompletionResult, StreamChunk
from mmrouter.providers.base import ProviderBase
from mmrouter.server.app import create_app


class MockProvider(ProviderBase):
    def complete(self, prompt, model, **kwargs):
        return CompletionResult(
            content=f"Response from {model}",
            model=model,
            tokens_in=10,
            tokens_out=20,
            cost=0.001,
            latency_ms=50.0,
        )

    def complete_messages(self, messages, model, **kwargs):
        return CompletionResult(
            content=f"Response from {model}",
            model=model,
            tokens_in=10,
            tokens_out=20,
            cost=0.001,
            latency_ms=50.0,
        )

    def stream_messages(self, messages, model, **kwargs) -> Iterator[StreamChunk]:
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
def client_with_provider(tmp_path):
    cfg = _write_config(tmp_path)
    app = create_app(config_path=str(cfg), db_path=str(tmp_path / "test.db"))
    provider = MockProvider()
    with TestClient(app) as client:
        app.state.router._provider = provider
        yield client, app


class TestRequestIdHeader:
    def test_non_stream_has_request_id(self, client_with_provider):
        client, _ = client_with_provider
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MMROUTER_API_KEY", None)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "What is 2+2?"}],
                },
            )

        assert resp.status_code == 200
        assert "x-mmrouter-request-id" in resp.headers
        request_id = resp.headers["x-mmrouter-request-id"]
        assert request_id.isdigit()

    def test_adaptive_reranked_header_present(self, client_with_provider):
        client, _ = client_with_provider
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MMROUTER_API_KEY", None)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "What is 2+2?"}],
                },
            )

        assert resp.status_code == 200
        assert "x-mmrouter-adaptive-reranked" in resp.headers
        assert resp.headers["x-mmrouter-adaptive-reranked"] == "false"


class TestFeedbackEndpoint:
    def test_submit_feedback_success(self, client_with_provider):
        client, _ = client_with_provider
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MMROUTER_API_KEY", None)

            # First create a request
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "What is 2+2?"}],
                },
            )
            request_id = int(resp.headers["x-mmrouter-request-id"])

            # Submit feedback
            resp = client.post(
                "/v1/feedback",
                json={"request_id": request_id, "rating": 1},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["request_id"] == request_id

    def test_submit_negative_feedback(self, client_with_provider):
        client, _ = client_with_provider
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MMROUTER_API_KEY", None)

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "What is 2+2?"}],
                },
            )
            request_id = int(resp.headers["x-mmrouter-request-id"])

            resp = client.post(
                "/v1/feedback",
                json={"request_id": request_id, "rating": -1},
            )

        assert resp.status_code == 200

    def test_feedback_nonexistent_request_returns_404(self, client_with_provider):
        client, _ = client_with_provider
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MMROUTER_API_KEY", None)
            resp = client.post(
                "/v1/feedback",
                json={"request_id": 99999, "rating": 1},
            )

        assert resp.status_code == 404

    def test_feedback_invalid_rating_returns_404(self, client_with_provider):
        client, _ = client_with_provider
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MMROUTER_API_KEY", None)

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "What is 2+2?"}],
                },
            )
            request_id = int(resp.headers["x-mmrouter-request-id"])

            resp = client.post(
                "/v1/feedback",
                json={"request_id": request_id, "rating": 0},
            )

        assert resp.status_code == 404

    def test_feedback_overwrites(self, client_with_provider):
        client, _ = client_with_provider
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MMROUTER_API_KEY", None)

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "What is 2+2?"}],
                },
            )
            request_id = int(resp.headers["x-mmrouter-request-id"])

            # Submit positive then negative
            client.post("/v1/feedback", json={"request_id": request_id, "rating": 1})
            resp = client.post("/v1/feedback", json={"request_id": request_id, "rating": -1})

        assert resp.status_code == 200
