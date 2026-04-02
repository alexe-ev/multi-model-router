"""Provider base class: abstract interface for LLM API calls."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from mmrouter.models import CompletionResult, StreamChunk


class ProviderBase(ABC):
    """Abstract base for all LLM providers."""

    @abstractmethod
    def complete(self, prompt: str, model: str, **kwargs) -> CompletionResult:
        """Send a prompt to the specified model and return the result."""

    def complete_messages(self, messages: list[dict], model: str, **kwargs) -> CompletionResult:
        """Send a messages array to the model. Default: extract last user message."""
        prompt = _extract_last_user_message(messages)
        return self.complete(prompt, model, **kwargs)

    def stream_messages(self, messages: list[dict], model: str, **kwargs) -> Iterator[StreamChunk]:
        """Stream response chunks for a messages array. Default: yield single chunk from complete."""
        result = self.complete_messages(messages, model, **kwargs)
        yield StreamChunk(content=result.content, model=result.model, finish_reason="stop")


def _extract_last_user_message(messages: list[dict]) -> str:
    """Extract the content of the last user message from a messages array."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            # Handle list-format content (vision messages etc.)
            if isinstance(content, list):
                parts = [p.get("text", "") for p in content if p.get("type") == "text"]
                return " ".join(parts)
    return ""
