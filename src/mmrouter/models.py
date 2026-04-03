"""Core data models and enums for mmrouter."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import StrEnum
from typing import Iterator

from pydantic import BaseModel, Field


class Complexity(StrEnum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class Category(StrEnum):
    FACTUAL = "factual"
    REASONING = "reasoning"
    CREATIVE = "creative"
    CODE = "code"


class ClassificationResult(BaseModel):
    complexity: Complexity
    category: Category
    confidence: float = Field(ge=0.0, le=1.0)


class CompletionResult(BaseModel):
    content: str
    model: str
    tokens_in: int
    tokens_out: int
    cost: float
    latency_ms: float
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


class StreamChunk(BaseModel):
    content: str
    model: str
    finish_reason: str | None = None


class StreamRouteResult(BaseModel):
    """Result from route_messages_stream: classification + model selection + chunk iterator."""

    classification: ClassificationResult
    model: str
    fallback_used: bool = False
    escalated: bool = False
    budget_downgraded: bool = False
    chunks: Iterator[StreamChunk] = None  # type: ignore[assignment]

    model_config = {"arbitrary_types_allowed": True}


class RequestLog(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    prompt_hash: str = ""
    classification: ClassificationResult
    model_used: str
    completion: CompletionResult
    fallback_used: bool = False
    cascade_used: bool = False
    cascade_attempts: int = 1
    experiment_id: int | None = None
    variant: str | None = None

    @staticmethod
    def hash_prompt(prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]


class AlertsConfig(BaseModel):
    """Configuration for alerting system."""

    enabled: bool = False
    webhook_url: str | None = None
    cooldown_seconds: int = 300
    rules: list[str] = Field(default_factory=lambda: ["cost_spike", "error_rate", "budget_warning"])


class AdaptiveConfig(BaseModel):
    """Configuration for feedback-driven adaptive routing."""

    enabled: bool = False
    min_feedback_count: int = 20
    decay_days: int = 30
    penalty_threshold: float = 0.4
    boost_threshold: float = 0.8


class BudgetConfig(BaseModel):
    """Configuration for daily budget limits with automatic model downgrade."""

    enabled: bool = False
    daily_limit: float = 0.0
    warn_threshold: float = 0.75
    downgrade_threshold: float = 0.90
    hard_limit_action: str = "cheapest"  # "cheapest" or "reject"


class CascadeConfig(BaseModel):
    """Configuration for cascade routing (try cheap models first, escalate on low quality)."""

    enabled: bool = False
    strategy: str = "heuristic"
    min_response_length: int = 50
    hedging_phrases: list[str] = Field(default_factory=lambda: [
        "I'm not sure",
        "I don't know",
        "I cannot",
        "I'm unable",
        "I don't have enough information",
    ])
    judge_model: str | None = None
    judge_threshold: int = 3


class ModelRoute(BaseModel):
    """Single model route: primary model + optional fallbacks."""

    model: str
    fallbacks: list[str] = Field(default_factory=list)
    cascade: list[str] = Field(default_factory=list)


class ClassifierConfig(BaseModel):
    strategy: str = "rules"
    model: str | None = None
    threshold: float = 0.7
    trained_model: str | None = None


class ProviderConfig(BaseModel):
    timeout_ms: int = 30000
    max_retries: int = 2
    circuit_breaker_threshold: int = 5
    circuit_breaker_reset_ms: int = 60000
    provider_map: dict[str, str] = Field(default_factory=dict)
    provider_circuit_breaker_threshold: int = 2
    provider_circuit_breaker_reset_ms: int = 120000
    prompt_caching: bool = True


class ExperimentStatus(StrEnum):
    ACTIVE = "active"
    STOPPED = "stopped"
    COMPLETED = "completed"


class Experiment(BaseModel):
    """A/B testing experiment: routes traffic between two configs."""

    id: int | None = None
    name: str
    status: ExperimentStatus = ExperimentStatus.ACTIVE
    control_config: str  # path to control YAML
    treatment_config: str  # path to treatment YAML
    traffic_split: float = Field(ge=0.0, le=1.0, default=0.5)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    stopped_at: datetime | None = None


class RoutingConfig(BaseModel):
    """Parsed YAML routing config."""

    version: str = "1"
    routes: dict[str, dict[str, ModelRoute]] = Field(default_factory=dict)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    cascade: CascadeConfig = Field(default_factory=CascadeConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    adaptive: AdaptiveConfig = Field(default_factory=AdaptiveConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)

    def get_route(self, complexity: Complexity, category: Category) -> ModelRoute | None:
        complexity_routes = self.routes.get(complexity.value)
        if not complexity_routes:
            return None
        return complexity_routes.get(category.value)
