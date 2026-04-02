"""Tests for embedding-based classifier."""

from pathlib import Path

import pytest

sentence_transformers = pytest.importorskip("sentence_transformers")

from mmrouter.classifier import ClassifierBase
from mmrouter.classifier.embeddings import EmbeddingClassifier
from mmrouter.models import Category, ClassificationResult, Complexity


EXAMPLES_PATH = Path(__file__).resolve().parent.parent.parent / "eval_data" / "embedding_examples.yaml"


@pytest.fixture(scope="module")
def classifier():
    return EmbeddingClassifier(examples_path=EXAMPLES_PATH, k=5)


def test_is_subclass_of_classifier_base():
    assert issubclass(EmbeddingClassifier, ClassifierBase)


def test_classify_returns_valid_result(classifier):
    result = classifier.classify("What is the capital of Germany?")
    assert isinstance(result, ClassificationResult)
    assert result.complexity in list(Complexity)
    assert result.category in list(Category)
    assert 0.0 <= result.confidence <= 1.0


def test_exact_training_example_complexity(classifier):
    # This prompt is in embedding_examples.yaml as simple/factual
    result = classifier.classify("What is the capital of Germany?")
    assert result.complexity == Complexity.SIMPLE


def test_exact_training_example_category(classifier):
    # This prompt is in embedding_examples.yaml as simple/factual
    result = classifier.classify("What is the capital of Germany?")
    assert result.category == Category.FACTUAL


def test_k1_returns_valid_result():
    clf = EmbeddingClassifier(examples_path=EXAMPLES_PATH, k=1)
    result = clf.classify("Write a haiku about mountains.")
    assert isinstance(result, ClassificationResult)
    assert 0.0 <= result.confidence <= 1.0


def test_k1_vs_k5_same_interface():
    clf1 = EmbeddingClassifier(examples_path=EXAMPLES_PATH, k=1)
    clf5 = EmbeddingClassifier(examples_path=EXAMPLES_PATH, k=5)
    r1 = clf1.classify("Implement a binary search function in Python.")
    r5 = clf5.classify("Implement a binary search function in Python.")
    # Both must return valid results regardless of k
    assert isinstance(r1, ClassificationResult)
    assert isinstance(r5, ClassificationResult)


def test_empty_prompt_returns_default(classifier):
    result = classifier.classify("")
    assert result.complexity == Complexity.SIMPLE
    assert result.category == Category.FACTUAL
    assert result.confidence == 0.5


def test_complex_code_prompt(classifier):
    result = classifier.classify(
        "Design and implement a connection pool for PostgreSQL with thread-safe checkout and release."
    )
    assert result.category == Category.CODE
    assert result.complexity == Complexity.COMPLEX
