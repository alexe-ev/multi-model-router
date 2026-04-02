"""Comparison runner: run multiple classifiers on the same eval set and collect results."""

from __future__ import annotations

import time
from dataclasses import dataclass

from mmrouter.classifier import ClassifierBase
from mmrouter.eval.evaluate import EvalCase, EvalReport, run_eval


@dataclass
class ComparisonResult:
    """Result for a single classifier in a comparison run."""

    name: str
    report: EvalReport
    elapsed_seconds: float


def run_comparison(
    eval_set: list[EvalCase],
    classifiers: dict[str, ClassifierBase],
) -> list[ComparisonResult]:
    """Run each classifier on the eval set, measure wall time, return results in insertion order."""
    results: list[ComparisonResult] = []
    for name, classifier in classifiers.items():
        start = time.perf_counter()
        report = run_eval(classifier, eval_set)
        elapsed = time.perf_counter() - start
        results.append(ComparisonResult(name=name, report=report, elapsed_seconds=elapsed))
    return results
