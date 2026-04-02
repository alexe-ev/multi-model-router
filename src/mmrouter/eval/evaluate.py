"""Eval runner: load labeled dataset, run classifier, produce accuracy report."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from mmrouter.classifier import ClassifierBase
from mmrouter.models import Category, Complexity


class EvalCase(BaseModel):
    prompt: str
    expected_complexity: Complexity
    expected_category: Category


class Mismatch(BaseModel):
    prompt: str
    expected_complexity: Complexity
    expected_category: Category
    got_complexity: Complexity
    got_category: Category


class EvalReport(BaseModel):
    total: int
    correct: int
    overall_accuracy: float
    complexity_accuracy: float
    category_accuracy: float
    per_class_accuracy: dict[str, float]
    mismatches: list[Mismatch]


def load_eval_set(path: str | Path) -> list[EvalCase]:
    """Parse YAML eval dataset, validate each entry, raise ValueError on invalid."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, list):
        raise ValueError(f"Eval dataset must be a YAML list, got {type(raw)}")

    cases: list[EvalCase] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"Entry {i} is not a dict: {entry!r}")
        missing = {"prompt", "complexity", "category"} - entry.keys()
        if missing:
            raise ValueError(f"Entry {i} missing fields: {missing}")
        try:
            cases.append(
                EvalCase(
                    prompt=entry["prompt"],
                    expected_complexity=Complexity(entry["complexity"]),
                    expected_category=Category(entry["category"]),
                )
            )
        except (ValueError, ValidationError) as e:
            raise ValueError(f"Entry {i} invalid: {e}") from e

    return cases


def run_eval(classifier: ClassifierBase, eval_set: list[EvalCase]) -> EvalReport:
    """Run classifier against eval set, compute accuracy metrics."""
    if not eval_set:
        return EvalReport(
            total=0,
            correct=0,
            overall_accuracy=0.0,
            complexity_accuracy=0.0,
            category_accuracy=0.0,
            per_class_accuracy={},
            mismatches=[],
        )

    # Per-bucket counters: key = "complexity/category"
    bucket_total: dict[str, int] = {}
    bucket_correct: dict[str, int] = {}

    correct_both = 0
    correct_complexity = 0
    correct_category = 0
    mismatches: list[Mismatch] = []

    for case in eval_set:
        bucket_key = f"{case.expected_complexity}/{case.expected_category}"
        bucket_total[bucket_key] = bucket_total.get(bucket_key, 0) + 1

        result = classifier.classify(case.prompt)

        complexity_ok = result.complexity == case.expected_complexity
        category_ok = result.category == case.expected_category

        if complexity_ok:
            correct_complexity += 1
        if category_ok:
            correct_category += 1
        if complexity_ok and category_ok:
            correct_both += 1
            bucket_correct[bucket_key] = bucket_correct.get(bucket_key, 0) + 1
        else:
            mismatches.append(
                Mismatch(
                    prompt=case.prompt,
                    expected_complexity=case.expected_complexity,
                    expected_category=case.expected_category,
                    got_complexity=result.complexity,
                    got_category=result.category,
                )
            )

    total = len(eval_set)
    per_class: dict[str, float] = {}
    for key, t in bucket_total.items():
        c = bucket_correct.get(key, 0)
        per_class[key] = round(c / t, 4) if t > 0 else 0.0

    return EvalReport(
        total=total,
        correct=correct_both,
        overall_accuracy=round(correct_both / total, 4),
        complexity_accuracy=round(correct_complexity / total, 4),
        category_accuracy=round(correct_category / total, 4),
        per_class_accuracy=per_class,
        mismatches=mismatches,
    )
