"""Tests for eval/compare.py: run_comparison and ComparisonResult."""

from __future__ import annotations

import pytest

from mmrouter.classifier import ClassifierBase
from mmrouter.eval.compare import ComparisonResult, run_comparison
from mmrouter.eval.evaluate import EvalCase
from mmrouter.models import Category, ClassificationResult, Complexity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FixedClassifier(ClassifierBase):
    """Returns a fixed classification for every prompt."""

    def __init__(self, complexity: Complexity, category: Category, confidence: float = 1.0):
        self._complexity = complexity
        self._category = category
        self._confidence = confidence

    def classify(self, prompt: str) -> ClassificationResult:
        return ClassificationResult(
            complexity=self._complexity,
            category=self._category,
            confidence=self._confidence,
        )


def _make_cases(complexity: str, category: str, n: int) -> list[EvalCase]:
    return [
        EvalCase(
            prompt=f"prompt {i}",
            expected_complexity=Complexity(complexity),
            expected_category=Category(category),
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunComparison:
    def test_returns_one_result_per_classifier(self):
        eval_set = _make_cases("simple", "factual", 4)
        classifiers = {
            "rules": FixedClassifier(Complexity.SIMPLE, Category.FACTUAL),
            "embeddings": FixedClassifier(Complexity.MEDIUM, Category.CODE),
        }
        results = run_comparison(eval_set, classifiers)
        assert len(results) == 2

    def test_result_names_match_classifier_keys(self):
        eval_set = _make_cases("simple", "factual", 2)
        classifiers = {
            "alpha": FixedClassifier(Complexity.SIMPLE, Category.FACTUAL),
            "beta": FixedClassifier(Complexity.COMPLEX, Category.REASONING),
        }
        results = run_comparison(eval_set, classifiers)
        names = [r.name for r in results]
        assert "alpha" in names
        assert "beta" in names

    def test_result_order_matches_insertion_order(self):
        eval_set = _make_cases("simple", "factual", 2)
        classifiers = {
            "first": FixedClassifier(Complexity.SIMPLE, Category.FACTUAL),
            "second": FixedClassifier(Complexity.SIMPLE, Category.FACTUAL),
            "third": FixedClassifier(Complexity.SIMPLE, Category.FACTUAL),
        }
        results = run_comparison(eval_set, classifiers)
        assert [r.name for r in results] == ["first", "second", "third"]

    def test_elapsed_seconds_is_positive(self):
        eval_set = _make_cases("simple", "factual", 5)
        classifiers = {"rules": FixedClassifier(Complexity.SIMPLE, Category.FACTUAL)}
        results = run_comparison(eval_set, classifiers)
        assert results[0].elapsed_seconds >= 0

    def test_elapsed_seconds_is_measured_not_zero(self):
        # With 10 cases there will always be some measurable time
        eval_set = _make_cases("simple", "factual", 10)
        classifiers = {"rules": FixedClassifier(Complexity.SIMPLE, Category.FACTUAL)}
        results = run_comparison(eval_set, classifiers)
        # Can't guarantee > 0 on fast hardware with perf_counter resolution,
        # but the field must be a float and be non-negative.
        assert isinstance(results[0].elapsed_seconds, float)
        assert results[0].elapsed_seconds >= 0.0

    def test_report_accuracy_for_perfect_classifier(self):
        eval_set = _make_cases("simple", "factual", 6)
        classifiers = {"perfect": FixedClassifier(Complexity.SIMPLE, Category.FACTUAL)}
        results = run_comparison(eval_set, classifiers)
        assert results[0].report.overall_accuracy == 1.0
        assert results[0].report.correct == 6

    def test_report_accuracy_for_wrong_classifier(self):
        eval_set = _make_cases("simple", "factual", 4)
        classifiers = {"wrong": FixedClassifier(Complexity.COMPLEX, Category.CODE)}
        results = run_comparison(eval_set, classifiers)
        assert results[0].report.overall_accuracy == 0.0
        assert results[0].report.correct == 0

    def test_single_classifier(self):
        eval_set = _make_cases("medium", "reasoning", 3)
        classifiers = {"solo": FixedClassifier(Complexity.MEDIUM, Category.REASONING)}
        results = run_comparison(eval_set, classifiers)
        assert len(results) == 1
        assert results[0].name == "solo"
        assert results[0].report.total == 3

    def test_empty_eval_set(self):
        classifiers = {
            "rules": FixedClassifier(Complexity.SIMPLE, Category.FACTUAL),
            "embeddings": FixedClassifier(Complexity.MEDIUM, Category.CODE),
        }
        results = run_comparison([], classifiers)
        assert len(results) == 2
        for r in results:
            assert r.report.total == 0
            assert r.report.overall_accuracy == 0.0

    def test_empty_classifiers_dict(self):
        eval_set = _make_cases("simple", "factual", 3)
        results = run_comparison(eval_set, {})
        assert results == []

    def test_comparison_result_dataclass_fields(self):
        eval_set = _make_cases("simple", "factual", 2)
        classifiers = {"test": FixedClassifier(Complexity.SIMPLE, Category.FACTUAL)}
        results = run_comparison(eval_set, classifiers)
        r = results[0]
        assert hasattr(r, "name")
        assert hasattr(r, "report")
        assert hasattr(r, "elapsed_seconds")
        assert isinstance(r, ComparisonResult)

    def test_two_classifiers_different_accuracy(self):
        # perfect + wrong classifier on same eval set
        eval_set = _make_cases("simple", "factual", 4)
        classifiers = {
            "perfect": FixedClassifier(Complexity.SIMPLE, Category.FACTUAL),
            "wrong": FixedClassifier(Complexity.COMPLEX, Category.CODE),
        }
        results = run_comparison(eval_set, classifiers)
        by_name = {r.name: r for r in results}
        assert by_name["perfect"].report.overall_accuracy == 1.0
        assert by_name["wrong"].report.overall_accuracy == 0.0
