"""Tests for core models and ABCs."""

import pytest
from pydantic import ValidationError

from mmrouter.classifier import ClassifierBase
from mmrouter.models import (
    Category,
    ClassificationResult,
    Complexity,
    CompletionResult,
    ModelRoute,
    RequestLog,
    RoutingConfig,
)
from mmrouter.providers.base import ProviderBase


class TestEnums:
    def test_complexity_values(self):
        assert list(Complexity) == [Complexity.SIMPLE, Complexity.MEDIUM, Complexity.COMPLEX]

    def test_category_values(self):
        assert list(Category) == [
            Category.FACTUAL,
            Category.REASONING,
            Category.CREATIVE,
            Category.CODE,
        ]

    def test_strenum_string_comparison(self):
        assert Complexity.SIMPLE == "simple"
        assert Category.CODE == "code"


class TestClassificationResult:
    def test_valid(self):
        r = ClassificationResult(complexity="simple", category="factual", confidence=0.95)
        assert r.complexity == Complexity.SIMPLE
        assert r.category == Category.FACTUAL
        assert r.confidence == 0.95

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            ClassificationResult(complexity="simple", category="factual", confidence=1.5)
        with pytest.raises(ValidationError):
            ClassificationResult(complexity="simple", category="factual", confidence=-0.1)

    def test_invalid_complexity(self):
        with pytest.raises(ValidationError):
            ClassificationResult(complexity="unknown", category="factual", confidence=0.5)


class TestCompletionResult:
    def test_valid(self):
        r = CompletionResult(
            content="Hello",
            model="claude-haiku",
            tokens_in=10,
            tokens_out=5,
            cost=0.001,
            latency_ms=150.0,
        )
        assert r.content == "Hello"
        assert r.model == "claude-haiku"


class TestRequestLog:
    def test_prompt_hash(self):
        h = RequestLog.hash_prompt("test prompt")
        assert len(h) == 16
        assert h == RequestLog.hash_prompt("test prompt")
        assert h != RequestLog.hash_prompt("different prompt")

    def test_serialization_roundtrip(self):
        log = RequestLog(
            prompt_hash=RequestLog.hash_prompt("test"),
            classification=ClassificationResult(
                complexity="simple", category="factual", confidence=0.9
            ),
            model_used="claude-haiku",
            completion=CompletionResult(
                content="ok", model="claude-haiku", tokens_in=5, tokens_out=2, cost=0.0001, latency_ms=100.0
            ),
        )
        data = log.model_dump()
        restored = RequestLog(**data)
        assert restored.prompt_hash == log.prompt_hash
        assert restored.classification.complexity == Complexity.SIMPLE


class TestRoutingConfig:
    def test_get_route(self):
        config = RoutingConfig(
            routes={
                "simple": {
                    "factual": ModelRoute(model="claude-haiku", fallbacks=["claude-sonnet"]),
                },
            }
        )
        route = config.get_route(Complexity.SIMPLE, Category.FACTUAL)
        assert route is not None
        assert route.model == "claude-haiku"
        assert route.fallbacks == ["claude-sonnet"]

    def test_get_route_missing(self):
        config = RoutingConfig()
        assert config.get_route(Complexity.COMPLEX, Category.CODE) is None


class TestABCs:
    def test_classifier_base_not_instantiable(self):
        with pytest.raises(TypeError):
            ClassifierBase()

    def test_provider_base_not_instantiable(self):
        with pytest.raises(TypeError):
            ProviderBase()

    def test_classifier_subclass_works(self):
        class DummyClassifier(ClassifierBase):
            def classify(self, prompt):
                return ClassificationResult(complexity="simple", category="factual", confidence=1.0)

        c = DummyClassifier()
        result = c.classify("test")
        assert result.complexity == Complexity.SIMPLE

    def test_provider_subclass_works(self):
        class DummyProvider(ProviderBase):
            def complete(self, prompt, model, **kwargs):
                return CompletionResult(
                    content="ok", model=model, tokens_in=1, tokens_out=1, cost=0.0, latency_ms=0.0
                )

        p = DummyProvider()
        result = p.complete("test", "claude-haiku")
        assert result.model == "claude-haiku"
