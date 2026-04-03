"""Alert delivery channels: webhook and log."""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from dataclasses import dataclass

logger = logging.getLogger("mmrouter.alerts")


@dataclass
class Alert:
    """A fired alert payload."""

    rule_name: str
    message: str
    severity: str  # "warning" or "critical"
    details: dict


class LogChannel:
    """Logs alerts via Python logger. Always active."""

    def send(self, alert: Alert) -> None:
        log_fn = logger.warning if alert.severity == "warning" else logger.error
        log_fn(
            "ALERT [%s] %s | %s",
            alert.rule_name,
            alert.message,
            json.dumps(alert.details),
        )


class WebhookChannel:
    """POST JSON to a webhook URL. Compatible with Slack incoming webhooks."""

    def __init__(self, url: str, timeout: float = 5.0):
        self._url = url
        self._timeout = timeout

    @property
    def url(self) -> str:
        return self._url

    def send(self, alert: Alert) -> bool:
        """Send alert. Returns True on success, False on failure. Never raises."""
        payload = {
            "text": f"[{alert.severity.upper()}] {alert.rule_name}: {alert.message}",
            "alert": {
                "rule": alert.rule_name,
                "severity": alert.severity,
                "message": alert.message,
                "details": alert.details,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout):
                pass
            return True
        except (urllib.error.URLError, OSError, ValueError) as e:
            logger.warning("Webhook delivery failed to %s: %s", self._url, e)
            return False
