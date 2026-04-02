"""Classifier module: classify prompts by complexity and category."""

from __future__ import annotations

from abc import ABC, abstractmethod

from mmrouter.models import ClassificationResult


class ClassifierBase(ABC):
    """Abstract base for all classifiers."""

    @abstractmethod
    def classify(self, prompt: str) -> ClassificationResult:
        """Classify a prompt into complexity + category with confidence score."""
