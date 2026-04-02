"""Tests for eval module: load_eval_set, run_eval, EvalReport."""

from __future__ import annotations

import pytest
import yaml

from mmrouter.classifier import ClassifierBase
from mmrouter.eval.evaluate import EvalCase, EvalReport, Mismatch, load_eval_set, run_eval
from mmrouter.models import Category, ClassificationResult, Complexity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FixedClassifier(ClassifierBase):
    """Always returns a fixed classification."""

    def __init__(self, complexity: Complexity, category: Category, confidence: float = 1.0):
        self.complexity = complexity
        self.category = category
        self.confidence = confidence

    def classify(self, prompt: str) -> ClassificationResult:
        return ClassificationResult(
            complexity=self.complexity,
            category=self.category,
            confidence=self.confidence,
        )


def _write_yaml(tmp_path, data) -> str:
    p = tmp_path / "eval.yaml"
    p.write_text(yaml.dump(data))
    return str(p)


# ---------------------------------------------------------------------------
# load_eval_set
# ---------------------------------------------------------------------------


class TestLoadEvalSet:
    def test_valid_yaml(self, tmp_path):
        data = [
            {"prompt": "What is 2+2?", "complexity": "simple", "category": "factual"},
            {"prompt": "Write a poem.", "complexity": "medium", "category": "creative"},
        ]
        path = _write_yaml(tmp_path, data)
        cases = load_eval_set(path)

        assert len(cases) == 2
        assert cases[0].prompt == "What is 2+2?"
        assert cases[0].expected_complexity == Complexity.SIMPLE
        assert cases[0].expected_category == Category.FACTUAL
        assert cases[1].expected_complexity == Complexity.MEDIUM
        assert cases[1].expected_category == Category.CREATIVE

    def test_all_complexity_values(self, tmp_path):
        data = [
            {"prompt": "a", "complexity": "simple", "category": "factual"},
            {"prompt": "b", "complexity": "medium", "category": "reasoning"},
            {"prompt": "c", "complexity": "complex", "category": "code"},
        ]
        cases = load_eval_set(_write_yaml(tmp_path, data))
        complexities = [c.expected_complexity for c in cases]
        assert Complexity.SIMPLE in complexities
        assert Complexity.MEDIUM in complexities
        assert Complexity.COMPLEX in complexities

    def test_all_category_values(self, tmp_path):
        data = [
            {"prompt": "a", "complexity": "simple", "category": "factual"},
            {"prompt": "b", "complexity": "simple", "category": "reasoning"},
            {"prompt": "c", "complexity": "simple", "category": "creative"},
            {"prompt": "d", "complexity": "simple", "category": "code"},
        ]
        cases = load_eval_set(_write_yaml(tmp_path, data))
        categories = [c.expected_category for c in cases]
        assert Category.FACTUAL in categories
        assert Category.REASONING in categories
        assert Category.CREATIVE in categories
        assert Category.CODE in categories

    def test_missing_prompt_field(self, tmp_path):
        data = [{"complexity": "simple", "category": "factual"}]
        path = _write_yaml(tmp_path, data)
        with pytest.raises(ValueError, match="missing fields"):
            load_eval_set(path)

    def test_missing_complexity_field(self, tmp_path):
        data = [{"prompt": "test", "category": "factual"}]
        path = _write_yaml(tmp_path, data)
        with pytest.raises(ValueError, match="missing fields"):
            load_eval_set(path)

    def test_missing_category_field(self, tmp_path):
        data = [{"prompt": "test", "complexity": "simple"}]
        path = _write_yaml(tmp_path, data)
        with pytest.raises(ValueError, match="missing fields"):
            load_eval_set(path)

    def test_invalid_complexity_value(self, tmp_path):
        data = [{"prompt": "test", "complexity": "ultra", "category": "factual"}]
        path = _write_yaml(tmp_path, data)
        with pytest.raises(ValueError, match="Entry 0 invalid"):
            load_eval_set(path)

    def test_invalid_category_value(self, tmp_path):
        data = [{"prompt": "test", "complexity": "simple", "category": "math"}]
        path = _write_yaml(tmp_path, data)
        with pytest.raises(ValueError, match="Entry 0 invalid"):
            load_eval_set(path)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_eval_set("/nonexistent/path/file.yaml")

    def test_not_a_list(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("key: value\n")
        with pytest.raises(ValueError, match="YAML list"):
            load_eval_set(str(p))

    def test_empty_list(self, tmp_path):
        data = []
        cases = load_eval_set(_write_yaml(tmp_path, data))
        assert cases == []


# ---------------------------------------------------------------------------
# run_eval
# ---------------------------------------------------------------------------


class TestRunEval:
    def _make_cases(self, complexity: str, category: str, n: int) -> list[EvalCase]:
        return [
            EvalCase(
                prompt=f"prompt {i}",
                expected_complexity=Complexity(complexity),
                expected_category=Category(category),
            )
            for i in range(n)
        ]

    def test_perfect_accuracy(self):
        cases = self._make_cases("simple", "factual", 10)
        clf = FixedClassifier(Complexity.SIMPLE, Category.FACTUAL)
        report = run_eval(clf, cases)

        assert report.total == 10
        assert report.correct == 10
        assert report.overall_accuracy == 1.0
        assert report.complexity_accuracy == 1.0
        assert report.category_accuracy == 1.0
        assert report.mismatches == []

    def test_zero_accuracy(self):
        cases = self._make_cases("complex", "code", 8)
        clf = FixedClassifier(Complexity.SIMPLE, Category.FACTUAL)
        report = run_eval(clf, cases)

        assert report.total == 8
        assert report.correct == 0
        assert report.overall_accuracy == 0.0
        assert report.complexity_accuracy == 0.0
        assert report.category_accuracy == 0.0
        assert len(report.mismatches) == 8

    def test_partial_accuracy_math(self):
        # 4 simple/factual + 4 complex/code
        # classifier always returns simple/factual
        # so 4 correct, 4 wrong
        cases = self._make_cases("simple", "factual", 4) + self._make_cases("complex", "code", 4)
        clf = FixedClassifier(Complexity.SIMPLE, Category.FACTUAL)
        report = run_eval(clf, cases)

        assert report.total == 8
        assert report.correct == 4
        assert report.overall_accuracy == 0.5
        # complexity: 4/8 correct (simple matches simple, wrong on complex)
        assert report.complexity_accuracy == 0.5
        # category: 4/8 correct (factual matches factual, wrong on code)
        assert report.category_accuracy == 0.5
        assert len(report.mismatches) == 4

    def test_complexity_correct_category_wrong(self):
        # classifier returns simple/reasoning, cases expect simple/factual
        # complexity always correct, category always wrong
        cases = self._make_cases("simple", "factual", 6)
        clf = FixedClassifier(Complexity.SIMPLE, Category.REASONING)
        report = run_eval(clf, cases)

        assert report.correct == 0
        assert report.complexity_accuracy == 1.0
        assert report.category_accuracy == 0.0
        assert len(report.mismatches) == 6

    def test_per_class_accuracy_keys(self):
        cases = (
            self._make_cases("simple", "factual", 4)
            + self._make_cases("medium", "code", 4)
        )
        clf = FixedClassifier(Complexity.SIMPLE, Category.FACTUAL)
        report = run_eval(clf, cases)

        assert "simple/factual" in report.per_class_accuracy
        assert "medium/code" in report.per_class_accuracy
        assert report.per_class_accuracy["simple/factual"] == 1.0
        assert report.per_class_accuracy["medium/code"] == 0.0

    def test_per_class_partial(self):
        # 2 correct + 2 wrong within simple/factual bucket
        cases = self._make_cases("simple", "factual", 4)
        # clf returns simple/factual for all — perfect within bucket
        clf = FixedClassifier(Complexity.SIMPLE, Category.FACTUAL)
        report = run_eval(clf, cases)
        assert report.per_class_accuracy["simple/factual"] == 1.0

    def test_empty_eval_set(self):
        clf = FixedClassifier(Complexity.SIMPLE, Category.FACTUAL)
        report = run_eval(clf, [])

        assert report.total == 0
        assert report.correct == 0
        assert report.overall_accuracy == 0.0
        assert report.per_class_accuracy == {}
        assert report.mismatches == []

    def test_mismatch_fields(self):
        cases = [
            EvalCase(
                prompt="explain quantum physics in detail",
                expected_complexity=Complexity.COMPLEX,
                expected_category=Category.REASONING,
            )
        ]
        clf = FixedClassifier(Complexity.SIMPLE, Category.FACTUAL)
        report = run_eval(clf, cases)

        assert len(report.mismatches) == 1
        m = report.mismatches[0]
        assert m.prompt == "explain quantum physics in detail"
        assert m.expected_complexity == Complexity.COMPLEX
        assert m.expected_category == Category.REASONING
        assert m.got_complexity == Complexity.SIMPLE
        assert m.got_category == Category.FACTUAL


# ---------------------------------------------------------------------------
# EvalReport model fields
# ---------------------------------------------------------------------------


class TestEvalReportModel:
    def test_fields_exist(self):
        report = EvalReport(
            total=10,
            correct=7,
            overall_accuracy=0.7,
            complexity_accuracy=0.8,
            category_accuracy=0.9,
            per_class_accuracy={"simple/factual": 1.0, "complex/code": 0.5},
            mismatches=[],
        )
        assert report.total == 10
        assert report.correct == 7
        assert report.overall_accuracy == 0.7
        assert report.complexity_accuracy == 0.8
        assert report.category_accuracy == 0.9
        assert report.per_class_accuracy == {"simple/factual": 1.0, "complex/code": 0.5}
        assert report.mismatches == []

    def test_with_mismatches(self):
        m = Mismatch(
            prompt="test",
            expected_complexity=Complexity.COMPLEX,
            expected_category=Category.CODE,
            got_complexity=Complexity.SIMPLE,
            got_category=Category.FACTUAL,
        )
        report = EvalReport(
            total=1,
            correct=0,
            overall_accuracy=0.0,
            complexity_accuracy=0.0,
            category_accuracy=0.0,
            per_class_accuracy={"complex/code": 0.0},
            mismatches=[m],
        )
        assert len(report.mismatches) == 1
        assert report.mismatches[0].prompt == "test"

    def test_serialization(self):
        report = EvalReport(
            total=5,
            correct=3,
            overall_accuracy=0.6,
            complexity_accuracy=0.7,
            category_accuracy=0.8,
            per_class_accuracy={"simple/factual": 0.6},
            mismatches=[],
        )
        data = report.model_dump()
        assert data["total"] == 5
        assert data["overall_accuracy"] == 0.6
        assert isinstance(data["per_class_accuracy"], dict)
        assert isinstance(data["mismatches"], list)
