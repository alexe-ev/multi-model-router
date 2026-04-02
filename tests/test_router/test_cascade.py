"""Tests for cascade routing quality gates."""

import pytest

from mmrouter.models import CascadeConfig, CompletionResult
from mmrouter.router.cascade import (
    HeuristicGate,
    LLMJudgeGate,
    QualityGateResult,
    create_quality_gate,
)


def _make_completion(content: str, model: str = "test-model") -> CompletionResult:
    return CompletionResult(
        content=content,
        model=model,
        tokens_in=10,
        tokens_out=len(content),
        cost=0.001,
        latency_ms=100.0,
    )


class TestHeuristicGate:
    def test_short_response_fails(self):
        config = CascadeConfig(min_response_length=50)
        gate = HeuristicGate(config)

        result = gate.check("test", _make_completion("Short."))

        assert not result.passed
        assert "too short" in result.reason
        assert result.score < 1.0

    def test_long_response_passes(self):
        config = CascadeConfig(min_response_length=50)
        gate = HeuristicGate(config)

        result = gate.check("test", _make_completion("A" * 100))

        assert result.passed
        assert result.score == 1.0

    def test_exact_min_length_passes(self):
        config = CascadeConfig(min_response_length=50)
        gate = HeuristicGate(config)

        result = gate.check("test", _make_completion("A" * 50))

        assert result.passed

    def test_hedging_phrase_fails(self):
        config = CascadeConfig(min_response_length=10, hedging_phrases=["I'm not sure"])
        gate = HeuristicGate(config)

        result = gate.check("test", _make_completion("I'm not sure about that, but here's my answer."))

        assert not result.passed
        assert "hedging" in result.reason

    def test_hedging_case_insensitive(self):
        config = CascadeConfig(min_response_length=10, hedging_phrases=["i cannot"])
        gate = HeuristicGate(config)

        result = gate.check("test", _make_completion("I CANNOT provide that information at this time."))

        assert not result.passed

    def test_no_hedging_passes(self):
        config = CascadeConfig(min_response_length=10, hedging_phrases=["I'm not sure"])
        gate = HeuristicGate(config)

        result = gate.check("test", _make_completion("The capital of France is Paris."))

        assert result.passed

    def test_empty_response_fails(self):
        config = CascadeConfig(min_response_length=50)
        gate = HeuristicGate(config)

        result = gate.check("test", _make_completion(""))

        assert not result.passed
        assert result.score == 0.0

    def test_whitespace_only_response_fails(self):
        config = CascadeConfig(min_response_length=50)
        gate = HeuristicGate(config)

        result = gate.check("test", _make_completion("   \n\t  "))

        assert not result.passed

    def test_zero_min_length_always_passes_length(self):
        config = CascadeConfig(min_response_length=0, hedging_phrases=[])
        gate = HeuristicGate(config)

        result = gate.check("test", _make_completion(""))

        assert result.passed

    def test_multiple_hedging_phrases(self):
        config = CascadeConfig(
            min_response_length=10,
            hedging_phrases=["I'm not sure", "I don't know", "I cannot"],
        )
        gate = HeuristicGate(config)

        # First phrase
        r1 = gate.check("test", _make_completion("I'm not sure what you mean by that question."))
        assert not r1.passed

        # Second phrase
        r2 = gate.check("test", _make_completion("I don't know the answer to this question."))
        assert not r2.passed

        # Third phrase
        r3 = gate.check("test", _make_completion("I cannot help with that particular request."))
        assert not r3.passed

    def test_length_checked_before_hedging(self):
        """Short response fails on length even if it also contains hedging."""
        config = CascadeConfig(min_response_length=100, hedging_phrases=["I cannot"])
        gate = HeuristicGate(config)

        result = gate.check("test", _make_completion("I cannot"))

        assert not result.passed
        assert "too short" in result.reason


class TestQualityGateFactory:
    def test_heuristic_gate_created_by_default(self):
        config = CascadeConfig(strategy="heuristic")
        gate = create_quality_gate(config)

        assert isinstance(gate, HeuristicGate)

    def test_unknown_strategy_falls_back_to_heuristic(self):
        config = CascadeConfig(strategy="unknown")
        gate = create_quality_gate(config)

        assert isinstance(gate, HeuristicGate)

    def test_llm_judge_requires_provider(self):
        config = CascadeConfig(strategy="llm_judge")

        with pytest.raises(ValueError, match="requires a provider"):
            create_quality_gate(config, provider=None)

    def test_llm_judge_created_with_provider(self):
        from mmrouter.providers.base import ProviderBase

        class FakeProvider(ProviderBase):
            def complete(self, prompt, model, **kwargs):
                pass

        config = CascadeConfig(strategy="llm_judge", judge_model="test-model")
        gate = create_quality_gate(config, provider=FakeProvider())

        assert isinstance(gate, LLMJudgeGate)


class TestQualityGateResult:
    def test_passed_result(self):
        r = QualityGateResult(passed=True, reason="ok", score=1.0)
        assert r.passed
        assert r.score == 1.0

    def test_failed_result(self):
        r = QualityGateResult(passed=False, reason="too short", score=0.3)
        assert not r.passed
        assert r.score == 0.3
