"""Deterministic hash-based traffic splitting for A/B experiments."""

from __future__ import annotations


def assign_variant(prompt_hash: str, traffic_split: float) -> str:
    """Assign a variant based on prompt hash and traffic split.

    Uses the prompt_hash (already a hex digest) to deterministically assign
    "control" or "treatment". Same prompt always gets same variant.

    Args:
        prompt_hash: hex string from RequestLog.hash_prompt()
        traffic_split: fraction of traffic to send to treatment (0.0-1.0)

    Returns:
        "control" or "treatment"
    """
    # Use last 8 hex chars of prompt_hash as a stable numeric value
    hash_int = int(prompt_hash[-8:], 16)
    # Normalize to [0, 1)
    normalized = (hash_int % 10000) / 10000.0
    if normalized < traffic_split:
        return "treatment"
    return "control"
