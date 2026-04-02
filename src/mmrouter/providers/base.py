"""Provider base class: abstract interface for LLM API calls."""

from __future__ import annotations

from abc import ABC, abstractmethod

from mmrouter.models import CompletionResult


class ProviderBase(ABC):
    """Abstract base for all LLM providers."""

    @abstractmethod
    def complete(self, prompt: str, model: str, **kwargs) -> CompletionResult:
        """Send a prompt to the specified model and return the result."""
