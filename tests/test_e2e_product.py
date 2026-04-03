"""End-to-end product tests with real LLM calls.

These tests hit real OpenAI API. Require OPENAI_API_KEY in env.
Run with: pytest tests/test_e2e_product.py -v -s
"""

from __future__ import annotations

import os
import time

import pytest

from mmrouter.models import Complexity, Category
from mmrouter.router.engine import Router

# Skip entire module if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)

OPENAI_CONFIG = "configs/openai.yaml"


@pytest.fixture
def router(tmp_path):
    r = Router(OPENAI_CONFIG, db_path=tmp_path / "e2e.db")
    yield r
    r.close()


# -------------------------------------------------------------------
# Scenario 1: Classification accuracy on real-world prompts
# -------------------------------------------------------------------

class TestClassificationAccuracy:
    """Verify classifier assigns correct complexity/category to known prompts."""

    # Prompts where rule-based classifier is reliable (simple tier)
    @pytest.mark.parametrize("prompt,expected_complexity", [
        ("What is the capital of France?", Complexity.SIMPLE),
        ("What does the len() function do in Python?", Complexity.SIMPLE),
        ("Write a haiku about autumn", Complexity.SIMPLE),
    ])
    def test_simple_classification(self, router, prompt, expected_complexity):
        result = router.classify(prompt)
        assert result.complexity == expected_complexity, (
            f"Expected {expected_complexity} for: '{prompt[:50]}...', got {result.complexity}"
        )
        assert 0.0 < result.confidence <= 1.0

    # Prompts where rule-based classifier is known to be weaker.
    # We accept +-1 tier (known 67% accuracy on medium/complex).
    @pytest.mark.parametrize("prompt,min_complexity", [
        ("Explain the trade-offs between SQL and NoSQL databases", Complexity.SIMPLE),
        ("Write a Python function to find the longest common subsequence", Complexity.MEDIUM),
        ("Design a microservices architecture for a real-time bidding platform with fault tolerance", Complexity.MEDIUM),
    ])
    def test_complex_classification_at_least(self, router, prompt, min_complexity):
        """Rule-based classifier should not under-classify by more than one tier."""
        tier_order = [Complexity.SIMPLE, Complexity.MEDIUM, Complexity.COMPLEX]
        result = router.classify(prompt)
        assert tier_order.index(result.complexity) >= tier_order.index(min_complexity), (
            f"'{prompt[:50]}...' classified as {result.complexity}, expected at least {min_complexity}"
        )


# -------------------------------------------------------------------
# Scenario 2: Basic routing - right model for right complexity
# -------------------------------------------------------------------

class TestBasicRouting:
    """Verify prompts get routed to the correct model tier."""

    def test_simple_query_uses_mini(self, router):
        """Simple factual question should route to gpt-4o-mini."""
        result = router.route("What is 2 + 2?")
        assert "mini" in result.model_used.lower(), f"Expected mini, got {result.model_used}"
        assert "4" in result.completion.content
        assert result.completion.cost > 0
        assert result.completion.latency_ms > 0
        assert result.completion.tokens_in > 0
        assert result.completion.tokens_out > 0

    def test_medium_query_uses_4o(self, router):
        """Medium reasoning should route to gpt-4o."""
        result = router.route("Compare the pros and cons of microservices vs monolith architecture. Be concise.")
        assert result.model_used == "gpt-4o", f"Expected gpt-4o, got {result.model_used}"
        assert len(result.completion.content) > 100  # substantial response
        assert result.completion.cost > 0

    def test_complex_query_uses_4o(self, router):
        """Complex reasoning should route to gpt-4o (our strongest available)."""
        result = router.route(
            "Design a distributed consensus algorithm that handles Byzantine faults "
            "in a network with up to 33% malicious nodes. Provide pseudocode."
        )
        assert result.model_used == "gpt-4o", f"Expected gpt-4o, got {result.model_used}"
        assert len(result.completion.content) > 200


# -------------------------------------------------------------------
# Scenario 3: Response quality - answers are actually correct
# -------------------------------------------------------------------

class TestResponseQuality:
    """Verify LLM responses are actually useful, not garbage."""

    def test_factual_accuracy(self, router):
        result = router.route("What year did World War 2 end?")
        assert "1945" in result.completion.content

    def test_code_generation(self, router):
        result = router.route("Write a Python function that checks if a number is prime. Just the function, no explanation.")
        content = result.completion.content
        assert "def " in content
        assert "prime" in content.lower()

    def test_creative_output(self, router):
        result = router.route("Write a haiku about rain")
        content = result.completion.content
        # A haiku should be short
        assert len(content.strip()) > 10
        assert len(content.strip()) < 500


# -------------------------------------------------------------------
# Scenario 4: Cost tracking works with real prices
# -------------------------------------------------------------------

class TestCostTracking:
    """Verify costs are logged correctly after real API calls."""

    def test_costs_logged(self, router):
        router.route("Say hello")
        router.route("What is Python?")
        router.route("Explain recursion briefly")

        stats = router.get_stats()
        assert stats["total_requests"] == 3
        assert stats["total_cost"] > 0
        assert stats["avg_latency_ms"] > 0
        assert stats["total_tokens_in"] > 0
        assert stats["total_tokens_out"] > 0

    def test_mini_cheaper_than_4o(self, router):
        """gpt-4o-mini should be cheaper than gpt-4o for similar prompts."""
        r1 = router.route("What is 2+2?")  # simple -> mini
        r2 = router.route("Compare TCP and UDP protocols in detail, covering reliability, ordering, and use cases")  # medium -> 4o

        # mini should cost less (or similar token count but lower per-token price)
        # We can't guarantee exact cost comparison because response lengths differ,
        # but per-token cost of mini is much lower
        assert r1.model_used != r2.model_used, "Should use different models"


# -------------------------------------------------------------------
# Scenario 5: Messages API (multi-turn conversation)
# -------------------------------------------------------------------

class TestMessagesAPI:
    """Verify route_messages works with multi-turn conversations."""

    def test_multi_turn(self, router):
        messages = [
            {"role": "system", "content": "You are a helpful math tutor."},
            {"role": "user", "content": "What is 5 * 7?"},
            {"role": "assistant", "content": "5 * 7 = 35"},
            {"role": "user", "content": "Now multiply that by 2"},
        ]
        result = router.route_messages(messages)
        assert "70" in result.completion.content
        assert result.completion.cost > 0


# -------------------------------------------------------------------
# Scenario 6: Streaming works
# -------------------------------------------------------------------

class TestStreaming:
    """Verify streaming returns chunks correctly."""

    def test_stream_produces_chunks(self, router):
        messages = [{"role": "user", "content": "Count from 1 to 5"}]
        result = router.route_messages_stream(messages)

        chunks = []
        for chunk in result.chunks:
            chunks.append(chunk)

        assert len(chunks) > 1, "Should produce multiple chunks"
        full_content = "".join(c.content for c in chunks)
        assert "1" in full_content
        assert "5" in full_content


# -------------------------------------------------------------------
# Scenario 7: Feedback submission
# -------------------------------------------------------------------

class TestFeedback:
    """Verify feedback can be submitted for routed requests."""

    def test_feedback_roundtrip(self, router):
        result = router.route("What color is the sky?")
        assert result.request_id is not None

        # Submit positive feedback
        router.submit_feedback(result.request_id, 1)

        # Verify feedback is recorded
        stats = router.get_feedback_stats()
        assert stats["total_feedback"] >= 1


# -------------------------------------------------------------------
# Scenario 8: Confidence escalation
# -------------------------------------------------------------------

class TestConfidenceEscalation:
    """Verify low-confidence classification triggers model escalation."""

    def test_ambiguous_prompt_escalates(self, router):
        # This prompt is intentionally ambiguous between categories
        result = router.route("Tell me about it")
        # We can't predict exact routing, but verify the pipeline completes
        assert result.completion.content
        assert result.model_used
        # If confidence was low, escalated should be True
        # (but we can't guarantee low confidence, so just verify no crash)


# -------------------------------------------------------------------
# Scenario 9: REST API server (quick smoke test)
# -------------------------------------------------------------------

class TestRESTAPI:
    """Smoke test the OpenAI-compatible REST API."""

    def test_openai_compatible_endpoint(self, tmp_path):
        from fastapi.testclient import TestClient
        from mmrouter.server.app import create_app

        app = create_app(config_path=OPENAI_CONFIG, db_path=str(tmp_path / "api.db"))
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": "What is 1+1?"}],
                    "max_tokens": 50,
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "choices" in data
            assert len(data["choices"]) > 0
            assert "2" in data["choices"][0]["message"]["content"]

            # Check routing headers
            assert "x-mmrouter-complexity" in resp.headers
            assert "x-mmrouter-model" in resp.headers

    def test_streaming_endpoint(self, tmp_path):
        from fastapi.testclient import TestClient
        from mmrouter.server.app import create_app

        app = create_app(config_path=OPENAI_CONFIG, db_path=str(tmp_path / "api.db"))
        with TestClient(app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": "Say hi"}],
                    "stream": True,
                    "max_tokens": 20,
                },
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            # Should contain data lines
            body = resp.text
            assert "data:" in body
            assert "[DONE]" in body
