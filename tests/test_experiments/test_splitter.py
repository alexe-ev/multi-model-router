"""Tests for deterministic hash-based traffic splitter."""

import pytest

from mmrouter.experiments.splitter import assign_variant
from mmrouter.models import RequestLog


class TestAssignVariant:
    def test_deterministic(self):
        """Same hash always gets same variant."""
        h = "abcdef1234567890"
        result1 = assign_variant(h, 0.5)
        result2 = assign_variant(h, 0.5)
        assert result1 == result2

    def test_returns_control_or_treatment(self):
        h = "abcdef1234567890"
        result = assign_variant(h, 0.5)
        assert result in ("control", "treatment")

    def test_all_treatment_at_split_1(self):
        """With split=1.0, everything goes to treatment."""
        for i in range(100):
            h = RequestLog.hash_prompt(f"prompt_{i}")
            assert assign_variant(h, 1.0) == "treatment"

    def test_all_control_at_split_0(self):
        """With split=0.0, everything goes to control."""
        for i in range(100):
            h = RequestLog.hash_prompt(f"prompt_{i}")
            assert assign_variant(h, 0.0) == "control"

    def test_approximate_split(self):
        """With split=0.5, roughly half should be treatment."""
        treatment_count = 0
        n = 1000
        for i in range(n):
            h = RequestLog.hash_prompt(f"prompt_{i}")
            if assign_variant(h, 0.5) == "treatment":
                treatment_count += 1
        # Should be roughly 50% +/- 10%
        ratio = treatment_count / n
        assert 0.35 < ratio < 0.65, f"Expected ~50% treatment, got {ratio:.1%}"

    def test_different_splits_change_assignment(self):
        """Higher split means more treatment assignments."""
        h = RequestLog.hash_prompt("test_prompt")
        # At some split threshold, assignment flips
        results = [assign_variant(h, s / 10.0) for s in range(11)]
        # Should have at least one control and one treatment across the range
        assert "control" in results
        assert "treatment" in results

    def test_uses_real_prompt_hash(self):
        """Works with actual prompt hashes from RequestLog."""
        h = RequestLog.hash_prompt("What is the capital of France?")
        result = assign_variant(h, 0.5)
        assert result in ("control", "treatment")
