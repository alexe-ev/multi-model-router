"""Tests for YAML config loader."""

import pytest

from mmrouter.models import Complexity, Category
from mmrouter.router.config import ConfigError, load_config


class TestLoadDefaultConfig:
    def test_loads_successfully(self):
        config = load_config("configs/default.yaml")
        assert config.version == "1"
        assert config.classifier.strategy == "rules"

    def test_all_complexity_category_pairs_present(self):
        config = load_config("configs/default.yaml")
        for complexity in Complexity:
            for category in Category:
                route = config.get_route(complexity, category)
                assert route is not None, f"Missing route: {complexity}/{category}"
                assert route.model, f"Empty model: {complexity}/{category}"

    def test_fallbacks_present(self):
        config = load_config("configs/default.yaml")
        route = config.get_route(Complexity.SIMPLE, Category.FACTUAL)
        assert len(route.fallbacks) > 0

    def test_provider_config(self):
        config = load_config("configs/default.yaml")
        assert config.provider.timeout_ms == 30000
        assert config.provider.max_retries == 2


class TestConfigValidation:
    def test_file_not_found(self):
        with pytest.raises(ConfigError, match="not found"):
            load_config("nonexistent.yaml")

    def test_invalid_complexity(self, tmp_path):
        cfg = tmp_path / "bad.yaml"
        cfg.write_text("""
routes:
  easy:
    factual:
      model: claude-haiku
""")
        with pytest.raises(ConfigError, match="Unknown complexity 'easy'"):
            load_config(cfg)

    def test_invalid_category(self, tmp_path):
        cfg = tmp_path / "bad.yaml"
        cfg.write_text("""
routes:
  simple:
    math:
      model: claude-haiku
""")
        with pytest.raises(ConfigError, match="Unknown category 'math'"):
            load_config(cfg)

    def test_missing_routes(self, tmp_path):
        cfg = tmp_path / "bad.yaml"
        cfg.write_text("""
classifier:
  strategy: rules
""")
        with pytest.raises(ConfigError, match="Invalid config"):
            load_config(cfg)

    def test_missing_model_in_route(self, tmp_path):
        cfg = tmp_path / "bad.yaml"
        cfg.write_text("""
routes:
  simple:
    factual:
      fallbacks:
        - claude-sonnet
""")
        with pytest.raises(ConfigError, match="Invalid config"):
            load_config(cfg)

    def test_minimal_valid_config(self, tmp_path):
        cfg = tmp_path / "minimal.yaml"
        cfg.write_text("""
routes:
  simple:
    factual:
      model: claude-haiku
""")
        config = load_config(cfg)
        route = config.get_route(Complexity.SIMPLE, Category.FACTUAL)
        assert route.model == "claude-haiku"
        assert route.fallbacks == []
