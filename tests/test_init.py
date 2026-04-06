"""Tests for mmrouter.init module and related CLI features."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from mmrouter.cli import _format_error, cli
from mmrouter.init import PROVIDER_PRESETS, check_api_key, generate_config
from mmrouter.router.config import load_config


# ---------------------------------------------------------------------------
# generate_config
# ---------------------------------------------------------------------------


def test_generate_config_anthropic():
    yaml_str = generate_config("anthropic")
    assert "claude-haiku" in yaml_str
    assert "claude-sonnet" in yaml_str
    assert "claude-opus" in yaml_str
    assert 'version: "1"' in yaml_str


def test_generate_config_openai():
    yaml_str = generate_config("openai")
    assert "gpt-4o-mini" in yaml_str
    assert "gpt-4o" in yaml_str
    assert 'version: "1"' in yaml_str


def test_generate_config_google():
    yaml_str = generate_config("google")
    assert "gemini" in yaml_str
    assert 'version: "1"' in yaml_str


def test_generate_config_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        generate_config("unknown")


# ---------------------------------------------------------------------------
# load_config validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", ["anthropic", "openai", "google"])
def test_generated_config_parses(provider, tmp_path):
    yaml_str = generate_config(provider)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_str)
    cfg = load_config(config_file)
    assert cfg.version == "1"
    # All 12 complexity/category combinations must be present
    for complexity in ("simple", "medium", "complex"):
        assert complexity in cfg.routes
        for category in ("factual", "reasoning", "creative", "code"):
            assert category in cfg.routes[complexity]
            route = cfg.routes[complexity][category]
            assert route.model


# ---------------------------------------------------------------------------
# check_api_key
# ---------------------------------------------------------------------------


def test_check_api_key_present(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-value")
    env_var, is_set = check_api_key("anthropic")
    assert env_var == "ANTHROPIC_API_KEY"
    assert is_set is True


def test_check_api_key_missing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env_var, is_set = check_api_key("anthropic")
    assert env_var == "ANTHROPIC_API_KEY"
    assert is_set is False


def test_check_api_key_openai_present(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    env_var, is_set = check_api_key("openai")
    assert env_var == "OPENAI_API_KEY"
    assert is_set is True


def test_check_api_key_google_missing(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    env_var, is_set = check_api_key("google")
    assert env_var == "GOOGLE_API_KEY"
    assert is_set is False


def test_check_api_key_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        check_api_key("unknown")


# ---------------------------------------------------------------------------
# _format_error
# ---------------------------------------------------------------------------


def test_format_error_auth_anthropic():
    err = Exception("litellm.AuthenticationError: ANTHROPIC API key not set")
    result = _format_error(err)
    assert "ANTHROPIC_API_KEY" in result
    assert "console.anthropic.com" in result


def test_format_error_auth_openai():
    err = Exception("litellm.AuthenticationError: openai API key invalid")
    result = _format_error(err)
    assert "OPENAI_API_KEY" in result
    assert "platform.openai.com" in result


def test_format_error_auth_google():
    err = Exception("litellm.AuthenticationError: google gemini key not found")
    result = _format_error(err)
    assert "GOOGLE_API_KEY" in result
    assert "aistudio.google.com" in result


def test_format_error_auth_gemini_upper():
    err = Exception("litellm.AuthenticationError: GEMINI key error")
    result = _format_error(err)
    assert "GOOGLE_API_KEY" in result


def test_format_error_auth_generic():
    err = Exception("litellm.AuthenticationError: unknown provider")
    result = _format_error(err)
    assert "mmrouter init" in result


def test_format_error_config_not_found():
    err = Exception("Config file not found: configs/missing.yaml")
    result = _format_error(err)
    assert "mmrouter init" in result


def test_format_error_generic():
    original = "some random error with no special keywords"
    err = Exception(original)
    result = _format_error(err)
    assert result == original


# ---------------------------------------------------------------------------
# init CLI command
# ---------------------------------------------------------------------------


def test_init_command_creates_config(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    output_file = tmp_path / "config.yaml"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["init", "--provider", "anthropic", "--output", str(output_file)],
    )
    assert result.exit_code == 0, result.output
    assert output_file.exists()
    content = output_file.read_text()
    assert "claude-haiku" in content
    assert "claude-sonnet" in content
    assert 'version: "1"' in content


def test_init_command_creates_config_openai(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    output_file = tmp_path / "config.yaml"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["init", "--provider", "openai", "--output", str(output_file)],
    )
    assert result.exit_code == 0, result.output
    assert output_file.exists()
    content = output_file.read_text()
    assert "gpt-4o" in content


def test_init_command_creates_config_google(tmp_path, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    output_file = tmp_path / "config.yaml"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["init", "--provider", "google", "--output", str(output_file)],
    )
    assert result.exit_code == 0, result.output
    assert output_file.exists()
    content = output_file.read_text()
    assert "gemini" in content


def test_init_command_no_overwrite(tmp_path):
    output_file = tmp_path / "config.yaml"
    original_content = "original content"
    output_file.write_text(original_content)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["init", "--provider", "anthropic", "--output", str(output_file)],
        input="n\n",
    )
    assert result.exit_code == 0, result.output
    # File must remain unchanged
    assert output_file.read_text() == original_content


def test_init_command_overwrite_yes(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    output_file = tmp_path / "config.yaml"
    output_file.write_text("old content")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["init", "--provider", "anthropic", "--output", str(output_file)],
        input="y\n",
    )
    assert result.exit_code == 0, result.output
    content = output_file.read_text()
    assert "claude-haiku" in content


def test_init_command_shows_key_set(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    output_file = tmp_path / "config.yaml"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["init", "--provider", "anthropic", "--output", str(output_file)],
    )
    assert result.exit_code == 0, result.output
    assert "is set" in result.output
    assert "mmrouter route" in result.output


def test_init_command_shows_key_missing_hint(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    output_file = tmp_path / "config.yaml"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["init", "--provider", "anthropic", "--output", str(output_file)],
    )
    assert result.exit_code == 0, result.output
    assert "is not set" in result.output
    assert "console.anthropic.com" in result.output


def test_init_command_creates_parent_dirs(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    output_file = tmp_path / "nested" / "deep" / "config.yaml"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["init", "--provider", "anthropic", "--output", str(output_file)],
    )
    assert result.exit_code == 0, result.output
    assert output_file.exists()


def test_init_command_interactive_prompt(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    output_file = tmp_path / "config.yaml"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["init", "--output", str(output_file)],
        input="openai\n",
    )
    assert result.exit_code == 0, result.output
    assert output_file.exists()
    content = output_file.read_text()
    assert "gpt-4o" in content


# ---------------------------------------------------------------------------
# PROVIDER_PRESETS sanity checks
# ---------------------------------------------------------------------------


def test_provider_presets_keys():
    assert set(PROVIDER_PRESETS.keys()) == {"anthropic", "openai", "google"}


def test_provider_presets_required_fields():
    for name, preset in PROVIDER_PRESETS.items():
        for field in ("simple", "medium", "complex", "env_var", "key_url"):
            assert field in preset, f"Provider '{name}' missing field '{field}'"
