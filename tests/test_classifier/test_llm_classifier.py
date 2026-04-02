"""Tests for LLMClassifier."""

from __future__ import annotations

import inspect

import pytest

from mmrouter.classifier import ClassifierBase
from mmrouter.classifier.llm_classifier import LLMClassifier
from mmrouter.models import Category, ClassificationResult, CompletionResult, Complexity
from mmrouter.providers.base import ProviderBase


class MockProvider(ProviderBase):
    def __init__(self, response_content: str):
        self._content = response_content

    def complete(self, prompt: str, model: str, **kwargs) -> CompletionResult:
        return CompletionResult(
            content=self._content,
            model=model,
            tokens_in=50,
            tokens_out=30,
            cost=0.0001,
            latency_ms=100,
        )


class CapturingProvider(ProviderBase):
    """Provider that records the prompt it receives."""

    def __init__(self, response_content: str):
        self._content = response_content
        self.last_prompt: str = ""

    def complete(self, prompt: str, model: str, **kwargs) -> CompletionResult:
        self.last_prompt = prompt
        return CompletionResult(
            content=self._content,
            model=model,
            tokens_in=50,
            tokens_out=30,
            cost=0.0001,
            latency_ms=100,
        )


def _valid_json(complexity: str, category: str, confidence: float = 0.9) -> str:
    return f'{{"complexity": "{complexity}", "category": "{category}", "confidence": {confidence}}}'


def test_valid_classification():
    provider = MockProvider(_valid_json("medium", "reasoning"))
    classifier = LLMClassifier(provider, model="test-model", few_shot=False)
    result = classifier.classify("Why do companies go bankrupt?")
    assert isinstance(result, ClassificationResult)
    assert result.complexity == Complexity.MEDIUM
    assert result.category == Category.REASONING
    assert result.confidence == pytest.approx(0.9)


@pytest.mark.parametrize("complexity", ["simple", "medium", "complex"])
def test_all_complexity_values(complexity):
    provider = MockProvider(_valid_json(complexity, "factual"))
    classifier = LLMClassifier(provider, model="test-model", few_shot=False)
    result = classifier.classify("test prompt")
    assert result.complexity == Complexity(complexity)


@pytest.mark.parametrize("category", ["factual", "reasoning", "creative", "code"])
def test_all_category_values(category):
    provider = MockProvider(_valid_json("medium", category))
    classifier = LLMClassifier(provider, model="test-model", few_shot=False)
    result = classifier.classify("test prompt")
    assert result.category == Category(category)


def test_malformed_response_fallback():
    provider = MockProvider("this is not json at all !!!")
    classifier = LLMClassifier(provider, model="test-model", few_shot=False)
    result = classifier.classify("some prompt")
    assert result.complexity == Complexity.MEDIUM
    assert result.category == Category.FACTUAL
    assert result.confidence == pytest.approx(0.3)


def test_missing_field_fallback():
    provider = MockProvider('{"complexity": "simple", "confidence": 0.8}')
    classifier = LLMClassifier(provider, model="test-model", few_shot=False)
    result = classifier.classify("some prompt")
    assert result.complexity == Complexity.MEDIUM
    assert result.category == Category.FACTUAL
    assert result.confidence == pytest.approx(0.3)


def test_few_shot_in_prompt():
    from mmrouter.classifier.few_shot_examples import FEW_SHOT_EXAMPLES

    provider = CapturingProvider(_valid_json("simple", "factual"))
    classifier = LLMClassifier(provider, model="test-model", few_shot=True)
    classifier.classify("What color is the sky?")

    # The prompt sent to the provider must contain at least one example prompt
    example_prompts = [ex["prompt"] for ex in FEW_SHOT_EXAMPLES]
    assert any(ep in provider.last_prompt for ep in example_prompts), (
        "Expected few-shot example prompts to appear in the classification prompt"
    )


def test_no_few_shot_in_prompt():
    from mmrouter.classifier.few_shot_examples import FEW_SHOT_EXAMPLES

    provider = CapturingProvider(_valid_json("simple", "factual"))
    classifier = LLMClassifier(provider, model="test-model", few_shot=False)
    classifier.classify("What color is the sky?")

    example_prompts = [ex["prompt"] for ex in FEW_SHOT_EXAMPLES]
    assert not any(ep in provider.last_prompt for ep in example_prompts)


def test_no_litellm_import():
    import mmrouter.classifier.llm_classifier as mod

    source = inspect.getsource(mod)
    assert "import litellm" not in source, (
        "llm_classifier.py must not import litellm directly"
    )


def test_is_classifier_base_subclass():
    provider = MockProvider(_valid_json("simple", "code"))
    classifier = LLMClassifier(provider, model="test-model")
    assert isinstance(classifier, ClassifierBase)


def test_markdown_fenced_json_response():
    """LLM sometimes wraps JSON in code fences; classifier should handle it."""
    fenced = '```json\n{"complexity": "complex", "category": "code", "confidence": 0.95}\n```'
    provider = MockProvider(fenced)
    classifier = LLMClassifier(provider, model="test-model", few_shot=False)
    result = classifier.classify("Implement a B-tree in Rust.")
    assert result.complexity == Complexity.COMPLEX
    assert result.category == Category.CODE


def test_empty_prompt():
    """Empty string returns simple/factual/0.5 without calling the provider."""
    provider = CapturingProvider(_valid_json("complex", "code"))
    classifier = LLMClassifier(provider, model="test-model", few_shot=False)
    result = classifier.classify("")
    assert result.complexity == Complexity.SIMPLE
    assert result.category == Category.FACTUAL
    assert result.confidence == pytest.approx(0.5)
    assert provider.last_prompt == "", "Provider must not be called for empty prompt"


def test_confidence_out_of_range():
    """confidence=5.0 from LLM is clamped to 1.0."""
    provider = MockProvider(_valid_json("simple", "factual", confidence=5.0))
    classifier = LLMClassifier(provider, model="test-model", few_shot=False)
    result = classifier.classify("What is water?")
    assert result.confidence == pytest.approx(1.0)


def test_invalid_enum_fallback():
    """Unknown complexity value triggers fallback."""
    provider = MockProvider('{"complexity": "extreme", "category": "factual", "confidence": 0.8}')
    classifier = LLMClassifier(provider, model="test-model", few_shot=False)
    result = classifier.classify("test prompt")
    assert result.complexity == Complexity.MEDIUM
    assert result.category == Category.FACTUAL
    assert result.confidence == pytest.approx(0.3)


class RaisingProvider(ProviderBase):
    """Provider that always raises an exception."""

    def complete(self, prompt: str, model: str, **kwargs) -> CompletionResult:
        raise RuntimeError("network error")


def test_provider_exception_fallback():
    """Exception from provider returns fallback result."""
    classifier = LLMClassifier(RaisingProvider(), model="test-model", few_shot=False)
    result = classifier.classify("Does this work?")
    assert result.complexity == Complexity.MEDIUM
    assert result.category == Category.FACTUAL
    assert result.confidence == pytest.approx(0.3)
