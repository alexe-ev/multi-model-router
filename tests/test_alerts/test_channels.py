"""Tests for alert delivery channels."""

import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

import pytest

from mmrouter.alerts.channels import Alert, LogChannel, WebhookChannel


@pytest.fixture
def sample_alert():
    return Alert(
        rule_name="test_rule",
        message="Something happened",
        severity="warning",
        details={"value": 42},
    )


class TestLogChannel:
    def test_send_warning(self, sample_alert, caplog):
        channel = LogChannel()
        with caplog.at_level(logging.WARNING, logger="mmrouter.alerts"):
            channel.send(sample_alert)
        assert "test_rule" in caplog.text
        assert "Something happened" in caplog.text

    def test_send_critical(self, caplog):
        alert = Alert(
            rule_name="critical_rule",
            message="Critical issue",
            severity="critical",
            details={},
        )
        channel = LogChannel()
        with caplog.at_level(logging.ERROR, logger="mmrouter.alerts"):
            channel.send(alert)
        assert "critical_rule" in caplog.text


class _WebhookHandler(BaseHTTPRequestHandler):
    """Simple handler that records POST payloads."""

    received: list = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        _WebhookHandler.received.append(json.loads(body))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format, *args):
        pass  # Suppress server logs in tests


class TestWebhookChannel:
    def test_send_to_server(self, sample_alert):
        _WebhookHandler.received = []
        server = HTTPServer(("127.0.0.1", 0), _WebhookHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()

        channel = WebhookChannel(f"http://127.0.0.1:{port}/webhook")
        result = channel.send(sample_alert)
        thread.join(timeout=5)
        server.server_close()

        assert result is True
        assert len(_WebhookHandler.received) == 1
        payload = _WebhookHandler.received[0]
        assert "test_rule" in payload["text"]
        assert payload["alert"]["rule"] == "test_rule"
        assert payload["alert"]["severity"] == "warning"
        assert payload["alert"]["details"]["value"] == 42

    def test_send_to_invalid_url(self, sample_alert):
        channel = WebhookChannel("http://127.0.0.1:1/nonexistent", timeout=0.5)
        result = channel.send(sample_alert)
        assert result is False

    def test_url_property(self):
        channel = WebhookChannel("http://example.com/hook")
        assert channel.url == "http://example.com/hook"
