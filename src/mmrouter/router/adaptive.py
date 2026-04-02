"""Feedback-driven adaptive model reranking."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

from mmrouter.models import AdaptiveConfig


class FeedbackScorer:
    """Computes per-model success rates per (complexity, category) bucket
    and reranks the fallback chain accordingly."""

    def __init__(self, conn: sqlite3.Connection, config: AdaptiveConfig):
        self._conn = conn
        self._config = config

    def get_model_scores(self, complexity: str, category: str) -> dict[str, float]:
        """Return {model_name: success_rate} for the given bucket.

        Only includes models with >= min_feedback_count ratings.
        Only considers feedback from last decay_days.
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=self._config.decay_days)
        ).isoformat()

        cur = self._conn.execute(
            """
            SELECT
                r.model,
                COUNT(*) as total,
                SUM(CASE WHEN f.rating = 1 THEN 1 ELSE 0 END) as positive
            FROM feedback f
            JOIN requests r ON f.request_id = r.id
            WHERE r.complexity = ?
              AND r.category = ?
              AND f.timestamp >= ?
            GROUP BY r.model
            """,
            (complexity, category, cutoff),
        )

        scores: dict[str, float] = {}
        for row in cur.fetchall():
            model, total, positive = row[0], row[1], row[2]
            if total >= self._config.min_feedback_count:
                scores[model] = positive / total if total > 0 else 0.0

        return scores

    def rerank_models(
        self, models: list[str], complexity: str, category: str
    ) -> tuple[list[str], bool]:
        """Re-rank model list based on feedback scores.

        Models with score > boost_threshold move to front.
        Models with score < penalty_threshold move to end.
        Models with insufficient feedback keep original position.

        Returns (reranked_list, was_reranked).
        Never adds or removes models from the list.
        """
        scores = self.get_model_scores(complexity, category)
        if not scores:
            return models, False

        boosted = []
        neutral = []
        penalized = []

        for model in models:
            if model not in scores:
                neutral.append(model)
            elif scores[model] >= self._config.boost_threshold:
                boosted.append(model)
            elif scores[model] < self._config.penalty_threshold:
                penalized.append(model)
            else:
                neutral.append(model)

        reranked = boosted + neutral + penalized
        was_reranked = reranked != models
        return reranked, was_reranked
