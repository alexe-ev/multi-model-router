"""Init logic for mmrouter: provider presets, config generation, API key check."""

from __future__ import annotations

import os

PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "anthropic": {
        "simple": "claude-haiku-4-5-20251001",
        "medium": "claude-sonnet-4-6",
        "complex": "claude-opus-4-6",
        "env_var": "ANTHROPIC_API_KEY",
        "key_url": "https://console.anthropic.com/",
    },
    "openai": {
        "simple": "gpt-4o-mini",
        "medium": "gpt-4o",
        "complex": "gpt-4o",
        "env_var": "OPENAI_API_KEY",
        "key_url": "https://platform.openai.com/api-keys",
    },
    "google": {
        "simple": "gemini/gemini-2.0-flash",
        "medium": "gemini/gemini-2.5-pro",
        "complex": "gemini/gemini-2.5-pro",
        "env_var": "GOOGLE_API_KEY",
        "key_url": "https://aistudio.google.com/apikey",
    },
}


def generate_config(provider: str) -> str:
    """Return a YAML config string for the given provider.

    The generated YAML passes load_config() validation (strictyaml).
    """
    if provider not in PROVIDER_PRESETS:
        raise ValueError(f"Unknown provider '{provider}'. Valid: {', '.join(PROVIDER_PRESETS)}")

    p = PROVIDER_PRESETS[provider]
    simple = p["simple"]
    medium = p["medium"]
    complex_ = p["complex"]

    return f"""version: "1"

routes:
  simple:
    factual:
      model: {simple}
      fallbacks:
        - {medium}
    reasoning:
      model: {simple}
      fallbacks:
        - {medium}
    creative:
      model: {simple}
      fallbacks:
        - {medium}
    code:
      model: {simple}
      fallbacks:
        - {medium}

  medium:
    factual:
      model: {medium}
      fallbacks:
        - {simple}
    reasoning:
      model: {medium}
      fallbacks:
        - {simple}
    creative:
      model: {medium}
      fallbacks:
        - {simple}
    code:
      model: {medium}
      fallbacks:
        - {simple}

  complex:
    factual:
      model: {medium}
      fallbacks:
        - {complex_}
    reasoning:
      model: {complex_}
      fallbacks:
        - {medium}
    creative:
      model: {complex_}
      fallbacks:
        - {medium}
    code:
      model: {complex_}
      fallbacks:
        - {medium}

classifier:
  strategy: rules
  threshold: "0.7"

provider:
  timeout_ms: 30000
  max_retries: 2
  circuit_breaker_threshold: 5
  circuit_breaker_reset_ms: 60000
  prompt_caching: true

cascade:
  enabled: false

budget:
  enabled: false

adaptive:
  enabled: false

alerts:
  enabled: false
"""


def check_api_key(provider: str) -> tuple[str, bool]:
    """Return (env_var_name, is_set) for the given provider."""
    if provider not in PROVIDER_PRESETS:
        raise ValueError(f"Unknown provider '{provider}'. Valid: {', '.join(PROVIDER_PRESETS)}")
    env_var = PROVIDER_PRESETS[provider]["env_var"]
    is_set = bool(os.environ.get(env_var, "").strip())
    return env_var, is_set
