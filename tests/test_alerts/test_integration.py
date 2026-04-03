"""Integration tests: config parsing, router integration."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from mmrouter.models import AlertsConfig, RoutingConfig
from mmrouter.router.config import ConfigError, load_config


@pytest.fixture
def base_yaml():
    """Minimal valid YAML config content."""
    return """
version: "1"
routes:
  simple:
    factual:
      model: test-model
"""


class TestAlertsConfigParsing:
    def test_default_alerts_disabled(self, base_yaml, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(base_yaml)
        config = load_config(cfg_file)
        assert config.alerts.enabled is False
        assert config.alerts.webhook_url is None

    def test_alerts_enabled(self, base_yaml, tmp_path):
        yaml_content = base_yaml + """
alerts:
  enabled: true
  webhook_url: "http://example.com/hook"
  cooldown_seconds: 120
  rules:
    - cost_spike
    - error_rate
"""
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml_content)
        config = load_config(cfg_file)
        assert config.alerts.enabled is True
        assert config.alerts.webhook_url == "http://example.com/hook"
        assert config.alerts.cooldown_seconds == 120
        assert config.alerts.rules == ["cost_spike", "error_rate"]

    def test_alerts_partial_config(self, base_yaml, tmp_path):
        yaml_content = base_yaml + """
alerts:
  enabled: true
"""
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml_content)
        config = load_config(cfg_file)
        assert config.alerts.enabled is True
        # Defaults
        assert config.alerts.cooldown_seconds == 300
        assert "cost_spike" in config.alerts.rules

    def test_invalid_rule_name_rejected(self, base_yaml, tmp_path):
        yaml_content = base_yaml + """
alerts:
  enabled: true
  rules:
    - nonexistent_rule
"""
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml_content)
        with pytest.raises(ConfigError, match="Unknown alert rule"):
            load_config(cfg_file)

    def test_alerts_with_budget_warning_only(self, base_yaml, tmp_path):
        yaml_content = base_yaml + """
alerts:
  enabled: true
  rules:
    - budget_warning
"""
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml_content)
        config = load_config(cfg_file)
        assert config.alerts.rules == ["budget_warning"]


class TestAlertsConfigModel:
    def test_defaults(self):
        cfg = AlertsConfig()
        assert cfg.enabled is False
        assert cfg.webhook_url is None
        assert cfg.cooldown_seconds == 300
        assert set(cfg.rules) == {"cost_spike", "error_rate", "budget_warning"}

    def test_custom(self):
        cfg = AlertsConfig(
            enabled=True,
            webhook_url="http://slack.example.com/hook",
            cooldown_seconds=60,
            rules=["cost_spike"],
        )
        assert cfg.enabled is True
        assert cfg.webhook_url == "http://slack.example.com/hook"
        assert cfg.cooldown_seconds == 60
        assert cfg.rules == ["cost_spike"]


class TestRoutingConfigIncludesAlerts:
    def test_routing_config_has_alerts(self):
        config = RoutingConfig()
        assert hasattr(config, "alerts")
        assert isinstance(config.alerts, AlertsConfig)
        assert config.alerts.enabled is False
