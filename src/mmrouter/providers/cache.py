"""Prompt caching: provider-specific cache_control annotation for messages."""

from __future__ import annotations

import copy

from mmrouter.router.fallback import extract_provider


def _is_anthropic_model(model: str, provider_map: dict[str, str] | None = None) -> bool:
    """Check if a model is an Anthropic model."""
    return extract_provider(model, provider_map) == "anthropic"


def annotate_cache_control(
    messages: list[dict],
    model: str,
    provider_map: dict[str, str] | None = None,
) -> list[dict]:
    """Add cache_control annotations to messages for provider-native caching.

    For Anthropic models: adds cache_control: {"type": "ephemeral"} to the last
    system message (if present and not already annotated). Also annotates the last
    message before the final user turn in long conversations.

    For other models: returns messages unchanged (OpenAI caching is automatic,
    Google requires separate API).

    Never mutates the original list. Returns a shallow copy with modified dicts.
    """
    if not _is_anthropic_model(model, provider_map):
        return messages

    if not messages:
        return messages

    # Shallow copy the list; we'll replace only dicts we modify
    result = list(messages)

    # Find the last system message index
    last_system_idx: int | None = None
    for i in range(len(result) - 1, -1, -1):
        if result[i].get("role") == "system":
            last_system_idx = i
            break

    # Annotate last system message
    if last_system_idx is not None:
        msg = result[last_system_idx]
        if "cache_control" not in msg:
            result[last_system_idx] = {**msg, "cache_control": {"type": "ephemeral"}}

    # For long conversations (4+ messages), also annotate the last message
    # before the final user turn to cache the conversation prefix
    if len(result) >= 4:
        last_user_idx: int | None = None
        for i in range(len(result) - 1, -1, -1):
            if result[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx is not None and last_user_idx > 0:
            prefix_idx = last_user_idx - 1
            # Don't double-annotate if it's the same as the system message we already handled
            if prefix_idx != last_system_idx:
                msg = result[prefix_idx]
                if "cache_control" not in msg:
                    result[prefix_idx] = {**msg, "cache_control": {"type": "ephemeral"}}

    return result
