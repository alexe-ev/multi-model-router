"""Tests for rule-based classifier."""

from mmrouter.classifier.rules import RuleClassifier
from mmrouter.models import Category, Complexity


def make_classifier():
    return RuleClassifier()


class TestSimpleFactual:
    def test_what_is(self):
        r = make_classifier().classify("What is the capital of France?")
        assert r.complexity == Complexity.SIMPLE
        assert r.category == Category.FACTUAL

    def test_short_question(self):
        r = make_classifier().classify("Who is Einstein?")
        assert r.complexity == Complexity.SIMPLE

    def test_define(self):
        r = make_classifier().classify("Define photosynthesis")
        assert r.complexity == Complexity.SIMPLE

    def test_how_many(self):
        r = make_classifier().classify("How many planets are there?")
        assert r.complexity == Complexity.SIMPLE


class TestComplexCode:
    def test_implement_complex(self):
        r = make_classifier().classify(
            "Implement a binary search tree with red-black balancing"
        )
        assert r.category == Category.CODE
        assert r.complexity in (Complexity.MEDIUM, Complexity.COMPLEX)

    def test_architecture_design(self):
        r = make_classifier().classify(
            "Design a distributed microservice architecture for a real-time "
            "event processing system with exactly-once delivery guarantees"
        )
        assert r.complexity == Complexity.COMPLEX

    def test_debug_code(self):
        r = make_classifier().classify("Debug this Python function that throws a runtime error")
        assert r.category == Category.CODE


class TestReasoning:
    def test_explain_why(self):
        r = make_classifier().classify("Why is the sky blue? Explain the physics.")
        assert r.category == Category.REASONING

    def test_compare(self):
        r = make_classifier().classify(
            "Compare the pros and cons of microservices versus monolithic architecture"
        )
        assert r.category == Category.REASONING

    def test_analyze_tradeoffs(self):
        r = make_classifier().classify(
            "Analyze the trade-off between consistency and availability in distributed systems"
        )
        assert r.category == Category.REASONING


class TestCreative:
    def test_write_story(self):
        r = make_classifier().classify("Write a short story about a robot learning to paint")
        assert r.category == Category.CREATIVE

    def test_compose_poem(self):
        r = make_classifier().classify("Compose a poem about autumn")
        assert r.category == Category.CREATIVE

    def test_brainstorm(self):
        r = make_classifier().classify("Brainstorm creative marketing ideas for a coffee shop")
        assert r.category == Category.CREATIVE


class TestEdgeCases:
    def test_empty_string(self):
        r = make_classifier().classify("")
        assert r.complexity == Complexity.SIMPLE
        assert r.confidence == 0.5

    def test_whitespace_only(self):
        r = make_classifier().classify("   ")
        assert r.complexity == Complexity.SIMPLE
        assert r.confidence == 0.5

    def test_very_short(self):
        r = make_classifier().classify("Hi")
        assert r.complexity == Complexity.SIMPLE

    def test_confidence_range(self):
        r = make_classifier().classify("Tell me about quantum computing")
        assert 0.0 <= r.confidence <= 1.0


class TestConfidence:
    def test_clear_simple_high_confidence(self):
        r = make_classifier().classify("What is 2+2?")
        assert r.confidence >= 0.7

    def test_clear_category_high_confidence(self):
        r = make_classifier().classify(
            "Write a creative story with dialogue and metaphor"
        )
        assert r.confidence >= 0.8
        assert r.category == Category.CREATIVE
