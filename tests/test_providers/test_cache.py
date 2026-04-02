"""Tests for prompt caching annotation logic."""

import copy

import pytest

from mmrouter.providers.cache import annotate_cache_control


class TestAnnotateCacheControlAnthropic:
    """Annotation for Anthropic (claude-*) models."""

    def test_annotates_system_message(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        result = annotate_cache_control(messages, "claude-sonnet-4-6")
        assert result[0]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in result[1]

    def test_annotates_last_system_message(self):
        messages = [
            {"role": "system", "content": "First system."},
            {"role": "system", "content": "Second system."},
            {"role": "user", "content": "Hi"},
        ]
        result = annotate_cache_control(messages, "claude-haiku-4-5-20251001")
        # Only the last system message gets annotated
        assert "cache_control" not in result[0]
        assert result[1]["cache_control"] == {"type": "ephemeral"}

    def test_no_double_annotate(self):
        messages = [
            {"role": "system", "content": "You are helpful.", "cache_control": {"type": "ephemeral"}},
            {"role": "user", "content": "Hi"},
        ]
        result = annotate_cache_control(messages, "claude-sonnet-4-6")
        # Should not modify existing cache_control
        assert result[0]["cache_control"] == {"type": "ephemeral"}
        assert result[0] is messages[0]  # Same object, not replaced

    def test_no_system_message(self):
        messages = [
            {"role": "user", "content": "Hi"},
        ]
        result = annotate_cache_control(messages, "claude-sonnet-4-6")
        assert "cache_control" not in result[0]

    def test_empty_messages(self):
        result = annotate_cache_control([], "claude-sonnet-4-6")
        assert result == []

    def test_long_conversation_annotates_prefix(self):
        """For 4+ messages, annotate last message before final user turn."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "What is 2+2?"},
        ]
        result = annotate_cache_control(messages, "claude-sonnet-4-6")
        # System message annotated
        assert result[0]["cache_control"] == {"type": "ephemeral"}
        # Assistant message (before last user turn) also annotated
        assert result[2]["cache_control"] == {"type": "ephemeral"}
        # User messages not annotated
        assert "cache_control" not in result[1]
        assert "cache_control" not in result[3]

    def test_short_conversation_no_prefix_annotation(self):
        """For < 4 messages, only system message is annotated."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = annotate_cache_control(messages, "claude-sonnet-4-6")
        assert result[0]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in result[1]
        assert "cache_control" not in result[2]

    def test_long_conversation_system_is_prefix(self):
        """If system message is right before the last user turn, don't double-annotate."""
        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "Answer"},
            {"role": "system", "content": "Updated system."},
            {"role": "user", "content": "Second question"},
        ]
        result = annotate_cache_control(messages, "claude-sonnet-4-6")
        # Last system (index 3) is annotated
        assert result[3]["cache_control"] == {"type": "ephemeral"}
        # Index 3 is also the prefix message (last before final user turn)
        # so no separate annotation needed, no double-annotate


class TestAnnotateCacheControlNonAnthropic:
    """No-op for non-Anthropic models."""

    def test_openai_noop(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        result = annotate_cache_control(messages, "gpt-4o")
        assert "cache_control" not in result[0]
        assert "cache_control" not in result[1]

    def test_gemini_noop(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        result = annotate_cache_control(messages, "gemini-2.0-flash")
        assert "cache_control" not in result[0]

    def test_unknown_model_noop(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        result = annotate_cache_control(messages, "some-custom-model")
        assert "cache_control" not in result[0]


class TestAnnotateImmutability:
    """Original messages must not be mutated."""

    def test_original_not_mutated(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        original = copy.deepcopy(messages)
        annotate_cache_control(messages, "claude-sonnet-4-6")
        assert messages == original

    def test_returns_new_list(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        result = annotate_cache_control(messages, "claude-sonnet-4-6")
        assert result is not messages

    def test_noop_returns_same_list_for_non_anthropic(self):
        """For non-Anthropic models, the original list reference is returned."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        result = annotate_cache_control(messages, "gpt-4o")
        assert result is messages


class TestAnnotateWithProviderMap:
    """Provider map overrides for model -> provider mapping."""

    def test_custom_model_mapped_to_anthropic(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        provider_map = {"my-custom-model": "anthropic"}
        result = annotate_cache_control(messages, "my-custom-model", provider_map)
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_claude_model_mapped_to_other(self):
        """If provider_map overrides claude- to a non-anthropic provider, no annotation."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        # extract_provider checks provider_map first, then prefix
        # But provider_map only matches exact model names, and prefix still matches claude-
        # So this test verifies the provider_map lookup (exact match takes priority)
        provider_map = {"claude-custom": "openai"}
        result = annotate_cache_control(messages, "claude-custom", provider_map)
        assert "cache_control" not in result[0]
