"""Tests for REST API auth middleware."""

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from mmrouter.server.app import create_app


@pytest.fixture
def app(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("""
version: "1"
routes:
  simple:
    factual:
      model: test-model
      fallbacks:
        - test-fallback
    reasoning:
      model: test-model
    creative:
      model: test-model
    code:
      model: test-model
  medium:
    factual:
      model: test-model
    reasoning:
      model: test-model
    creative:
      model: test-model
    code:
      model: test-model
  complex:
    factual:
      model: test-model
    reasoning:
      model: test-model
    creative:
      model: test-model
    code:
      model: test-model
classifier:
  strategy: rules
  threshold: "0.7"
provider:
  timeout_ms: 30000
  max_retries: 0
""")
    return create_app(config_path=str(config_path), db_path=str(tmp_path / "test.db"))


class TestAuthDisabled:
    """When MMROUTER_API_KEY is not set, auth is disabled."""

    def test_health_no_auth(self, app):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MMROUTER_API_KEY", None)
            with TestClient(app) as client:
                resp = client.get("/health")
                assert resp.status_code == 200

    def test_models_no_auth(self, app):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MMROUTER_API_KEY", None)
            with TestClient(app) as client:
                resp = client.get("/v1/models")
                assert resp.status_code == 200


class TestAuthEnabled:
    """When MMROUTER_API_KEY is set, auth is enforced on /v1/ routes."""

    def test_missing_key_returns_401(self, app):
        with patch.dict(os.environ, {"MMROUTER_API_KEY": "secret-key-123"}):
            with TestClient(app) as client:
                resp = client.get("/v1/models")
                assert resp.status_code == 401

    def test_wrong_key_returns_401(self, app):
        with patch.dict(os.environ, {"MMROUTER_API_KEY": "secret-key-123"}):
            with TestClient(app) as client:
                resp = client.get(
                    "/v1/models",
                    headers={"Authorization": "Bearer wrong-key"},
                )
                assert resp.status_code == 401

    def test_correct_key_passes(self, app):
        with patch.dict(os.environ, {"MMROUTER_API_KEY": "secret-key-123"}):
            with TestClient(app) as client:
                resp = client.get(
                    "/v1/models",
                    headers={"Authorization": "Bearer secret-key-123"},
                )
                assert resp.status_code == 200

    def test_health_no_auth_required(self, app):
        """Health endpoint never requires auth."""
        with patch.dict(os.environ, {"MMROUTER_API_KEY": "secret-key-123"}):
            with TestClient(app) as client:
                resp = client.get("/health")
                assert resp.status_code == 200

    def test_missing_bearer_prefix_returns_401(self, app):
        with patch.dict(os.environ, {"MMROUTER_API_KEY": "secret-key-123"}):
            with TestClient(app) as client:
                resp = client.get(
                    "/v1/models",
                    headers={"Authorization": "secret-key-123"},
                )
                assert resp.status_code == 401

    def test_chat_completions_requires_auth(self, app):
        with patch.dict(os.environ, {"MMROUTER_API_KEY": "secret-key-123"}):
            with TestClient(app) as client:
                resp = client.post(
                    "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hi"}]},
                )
                assert resp.status_code == 401
