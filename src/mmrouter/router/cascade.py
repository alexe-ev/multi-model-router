"""Quality gates for cascade routing: check if a response is good enough or needs escalation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from mmrouter.models import CascadeConfig, CompletionResult


@dataclass
class QualityGateResult:
    passed: bool
    reason: str
    score: float = 1.0


class QualityGate(ABC):
    """Abstract quality gate: decides if a response is good enough."""

    @abstractmethod
    def check(self, prompt: str, response: CompletionResult) -> QualityGateResult:
        """Return whether the response passes the quality gate."""


class HeuristicGate(QualityGate):
    """Checks response length and hedging phrases. Free, no extra API calls."""

    def __init__(self, config: CascadeConfig):
        self._min_length = config.min_response_length
        self._hedging_phrases = [p.lower() for p in config.hedging_phrases]

    def check(self, prompt: str, response: CompletionResult) -> QualityGateResult:
        content = response.content.strip()

        if len(content) < self._min_length:
            return QualityGateResult(
                passed=False,
                reason=f"Response too short ({len(content)} < {self._min_length} chars)",
                score=len(content) / self._min_length if self._min_length > 0 else 0.0,
            )

        content_lower = content.lower()
        for phrase in self._hedging_phrases:
            if phrase in content_lower:
                return QualityGateResult(
                    passed=False,
                    reason=f"Response contains hedging phrase: '{phrase}'",
                    score=0.5,
                )

        return QualityGateResult(passed=True, reason="Passed heuristic checks", score=1.0)


class LLMJudgeGate(QualityGate):
    """Uses LLM-as-judge to evaluate response quality. Requires a provider."""

    def __init__(self, config: CascadeConfig, provider):
        from mmrouter.providers.base import ProviderBase

        if not isinstance(provider, ProviderBase):
            raise TypeError("LLMJudgeGate requires a ProviderBase instance")
        self._provider = provider
        self._judge_model = config.judge_model
        self._threshold = config.judge_threshold

    def check(self, prompt: str, response: CompletionResult) -> QualityGateResult:
        if not self._judge_model:
            return QualityGateResult(passed=True, reason="No judge model configured", score=1.0)

        from mmrouter.eval.quality import judge_response

        score = judge_response(self._provider, self._judge_model, prompt, response.content)

        if score.score >= self._threshold:
            return QualityGateResult(
                passed=True,
                reason=f"Judge score {score.score} >= threshold {self._threshold}",
                score=float(score.score) / 5.0,
            )
        return QualityGateResult(
            passed=False,
            reason=f"Judge score {score.score} < threshold {self._threshold}",
            score=float(score.score) / 5.0,
        )


def create_quality_gate(config: CascadeConfig, provider=None) -> QualityGate:
    """Factory: create the appropriate quality gate from config."""
    if config.strategy == "llm_judge":
        if provider is None:
            raise ValueError("LLM judge strategy requires a provider")
        return LLMJudgeGate(config, provider)
    return HeuristicGate(config)
