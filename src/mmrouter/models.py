"""Core data models and enums for mmrouter."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import StrEnum

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


class RequestLog(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    prompt_hash: str = ""
    classification: ClassificationResult
    model_used: str
    completion: CompletionResult
    fallback_used: bool = False

    @staticmethod
    def hash_prompt(prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]


class ModelRoute(BaseModel):
    """Single model route: primary model + optional fallbacks."""

    model: str
    fallbacks: list[str] = Field(default_factory=list)


class ClassifierConfig(BaseModel):
    strategy: str = "rules"
    model: str | None = None
    threshold: float = 0.7


class ProviderConfig(BaseModel):
    timeout_ms: int = 30000
    max_retries: int = 2
    circuit_breaker_threshold: int = 5
    circuit_breaker_reset_ms: int = 60000


class RoutingConfig(BaseModel):
    """Parsed YAML routing config."""

    version: str = "1"
    routes: dict[str, dict[str, ModelRoute]] = Field(default_factory=dict)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)

    def get_route(self, complexity: Complexity, category: Category) -> ModelRoute | None:
        complexity_routes = self.routes.get(complexity.value)
        if not complexity_routes:
            return None
        return complexity_routes.get(category.value)
