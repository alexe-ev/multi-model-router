"""Budget manager: track daily spend, enforce limits, downgrade models."""

from __future__ import annotations

import sqlite3
import time
from enum import StrEnum

from mmrouter.models import BudgetConfig, Complexity
from mmrouter.tracker.analytics import MODEL_PRICING


class BudgetTier(StrEnum):
    NORMAL = "normal"
    WARN = "warn"
    DOWNGRADE = "downgrade"
    HARD_LIMIT = "hard_limit"


class BudgetExceededError(Exception):
    """Raised when budget is exhausted and hard_limit_action is 'reject'."""


class BudgetManager:
    """Tracks daily spend and determines allowed model tier."""

    def __init__(self, config: BudgetConfig, conn: sqlite3.Connection):
        self._config = config
        self._conn = conn
        self._cache_value: float | None = None
        self._cache_time: float = 0.0
        self._cache_ttl = 1.0  # seconds

    @property
    def enabled(self) -> bool:
        return self._config.enabled and self._config.daily_limit > 0

    def get_daily_spend(self) -> float:
        """Query today's total spend from SQLite, with 1s TTL cache."""
        now = time.monotonic()
        if self._cache_value is not None and (now - self._cache_time) < self._cache_ttl:
            return self._cache_value

        cur = self._conn.execute(
            "SELECT COALESCE(SUM(cost), 0) FROM requests WHERE date(timestamp) = date('now')"
        )
        spend = cur.fetchone()[0]
        self._cache_value = float(spend)
        self._cache_time = now
        return self._cache_value

    def get_budget_tier(self) -> BudgetTier:
        """Determine which budget tier we're in based on daily spend."""
        if not self.enabled:
            return BudgetTier.NORMAL

        spend = self.get_daily_spend()
        ratio = spend / self._config.daily_limit

        if ratio >= 1.0:
            return BudgetTier.HARD_LIMIT
        if ratio >= self._config.downgrade_threshold:
            return BudgetTier.DOWNGRADE
        if ratio >= self._config.warn_threshold:
            return BudgetTier.WARN
        return BudgetTier.NORMAL

    def get_remaining(self) -> float:
        """Return remaining budget for today."""
        if not self.enabled:
            return float("inf")
        return max(0.0, self._config.daily_limit - self.get_daily_spend())

    def apply_budget(self, complexity: Complexity) -> Complexity:
        """Downgrade complexity based on budget tier. Raises BudgetExceededError if reject mode."""
        tier = self.get_budget_tier()

        if tier == BudgetTier.NORMAL or tier == BudgetTier.WARN:
            return complexity

        if tier == BudgetTier.HARD_LIMIT:
            if self._config.hard_limit_action == "reject":
                raise BudgetExceededError(
                    f"Daily budget ${self._config.daily_limit:.2f} exhausted. "
                    f"Spent: ${self.get_daily_spend():.6f}"
                )
            # hard_limit_action == "cheapest": force to simplest
            return Complexity.SIMPLE

        # DOWNGRADE tier: step down one level
        if complexity == Complexity.COMPLEX:
            return Complexity.MEDIUM
        if complexity == Complexity.MEDIUM:
            return Complexity.SIMPLE
        return Complexity.SIMPLE

    def get_status(self) -> dict:
        """Return budget status for CLI/API display."""
        if not self.enabled:
            return {"enabled": False}

        spend = self.get_daily_spend()
        tier = self.get_budget_tier()
        return {
            "enabled": True,
            "daily_limit": self._config.daily_limit,
            "spent_today": round(spend, 6),
            "remaining": round(self.get_remaining(), 6),
            "usage_pct": round(spend / self._config.daily_limit * 100, 1) if self._config.daily_limit > 0 else 0.0,
            "tier": tier.value,
            "hard_limit_action": self._config.hard_limit_action,
        }

    def invalidate_cache(self) -> None:
        """Force next get_daily_spend() to query the database."""
        self._cache_value = None
