"""Tests for LLM-as-judge quality evaluation."""
import json
import pytest
from mmrouter.eval.quality import (
    QualityScore, QualityReport, JUDGE_PROMPT,
    judge_response, run_quality_eval, compare_quality,
    _strip_code_fences,
)
from mmrouter.models import CompletionResult
from mmrouter.providers.base import ProviderBase


class MockJudgeProvider(ProviderBase):
    def __init__(self, response_content: str):
        self._content = response_content
        self.last_prompt = None

    def complete(self, prompt, model, **kwargs):
        self.last_prompt = prompt
        return CompletionResult(
            content=self._content, model=model,
            tokens_in=100, tokens_out=50, cost=0.001, latency_ms=200,
        )


def _make_judge_json(score=4, relevance=4, accuracy=4, completeness=4, reasoning="Good"):
    return json.dumps({
        "score": score, "relevance": relevance,
        "accuracy": accuracy, "completeness": completeness,
        "reasoning": reasoning,
    })


class TestStripCodeFences:
    def test_no_fences(self):
        assert _strip_code_fences('{"score": 4}') == '{"score": 4}'

    def test_json_fences(self):
        assert _strip_code_fences('```json\n{"score": 4}\n```') == '{"score": 4}'

    def test_plain_fences(self):
        assert _strip_code_fences('```\n{"score": 4}\n```') == '{"score": 4}'


class TestJudgeResponse:
    def test_parses_valid_json(self):
        provider = MockJudgeProvider(_make_judge_json(5, 5, 4, 4, "Excellent"))
        result = judge_response(provider, "test-model", "What is 2+2?", "4")
        assert result.score == 5
        assert result.relevance == 5
        assert result.accuracy == 4
        assert result.completeness == 4
        assert result.reasoning == "Excellent"

    def test_handles_markdown_fences(self):
        content = f"```json\n{_make_judge_json(4, 4, 3, 3)}\n```"
        provider = MockJudgeProvider(content)
        result = judge_response(provider, "test-model", "prompt", "response")
        assert result.score == 4

    def test_handles_parse_error(self):
        provider = MockJudgeProvider("This is not JSON at all")
        result = judge_response(provider, "test-model", "prompt", "response")
        assert result.score == 3
        assert result.reasoning == "Judge parse error"

    def test_prompt_has_xml_tags(self):
        provider = MockJudgeProvider(_make_judge_json())
        judge_response(provider, "test-model", "my prompt", "my response")
        assert "<prompt_to_judge>" in provider.last_prompt
        assert "my prompt" in provider.last_prompt
        assert "<response_to_judge>" in provider.last_prompt
        assert "my response" in provider.last_prompt

    def test_handles_invalid_scores(self):
        bad_json = json.dumps({"score": 10, "relevance": 0, "accuracy": 4, "completeness": 4, "reasoning": "bad"})
        provider = MockJudgeProvider(bad_json)
        result = judge_response(provider, "test-model", "p", "r")
        assert result.score == 3  # fallback

    def test_handles_braces_in_content(self):
        provider = MockJudgeProvider(_make_judge_json(4, 4, 4, 4))
        result = judge_response(
            provider, "test-model",
            'Write {"key": "value"} in Python',
            'Use json: {"key": "value"}',
        )
        assert result.score == 4
        assert '{"key": "value"}' in provider.last_prompt


class TestRunQualityEval:
    def test_aggregates_scores(self):
        provider = MockJudgeProvider(_make_judge_json(4, 5, 3, 4))
        pairs = [("p1", "r1"), ("p2", "r2"), ("p3", "r3")]
        report = run_quality_eval(provider, "model", pairs)
        assert report.total == 3
        assert report.avg_score == 4.0
        assert report.avg_relevance == 5.0
        assert report.avg_accuracy == 3.0
        assert report.avg_completeness == 4.0
        assert len(report.scores) == 3

    def test_empty_list(self):
        provider = MockJudgeProvider(_make_judge_json())
        report = run_quality_eval(provider, "model", [])
        assert report.total == 0
        assert report.avg_score == 0


class TestCompareQuality:
    def test_positive_delta(self):
        # Router gets 5s, baseline gets 3s
        call_count = 0
        class AlternatingProvider(ProviderBase):
            def complete(self, prompt, model, **kwargs):
                nonlocal call_count
                call_count += 1
                # First N calls are for router eval, next N for baseline
                if call_count <= 2:
                    content = _make_judge_json(5, 5, 5, 5, "Great")
                else:
                    content = _make_judge_json(3, 3, 3, 3, "OK")
                return CompletionResult(
                    content=content, model=model,
                    tokens_in=100, tokens_out=50, cost=0.001, latency_ms=200,
                )

        result = compare_quality(
            AlternatingProvider(), "judge",
            [("p1", "r1"), ("p2", "r2")],
            [("p1", "r1"), ("p2", "r2")],
        )
        assert result["router"].avg_score == 5.0
        assert result["baseline"].avg_score == 3.0
        assert result["delta"] == 2.0
        assert result["delta_pct"] > 0

    def test_equal_scores(self):
        provider = MockJudgeProvider(_make_judge_json(4, 4, 4, 4))
        result = compare_quality(
            provider, "judge",
            [("p1", "r1")], [("p1", "r1")],
        )
        assert result["delta"] == 0.0
        assert result["delta_pct"] == 0.0
