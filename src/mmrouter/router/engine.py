"""Router engine: classify -> select model -> call provider -> track."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from mmrouter.classifier import ClassifierBase
from mmrouter.classifier.rules import RuleClassifier
from mmrouter.models import (
    ClassificationResult,
    CompletionResult,
    RequestLog,
    RoutingConfig,
)
from mmrouter.providers.base import ProviderBase
from mmrouter.providers.litellm_provider import LiteLLMProvider, ProviderError
from mmrouter.router.config import load_config
from mmrouter.tracker.logger import Tracker


class RoutingResult(BaseModel):
    classification: ClassificationResult
    completion: CompletionResult
    model_used: str
    fallback_used: bool = False


class Router:
    """Core router: classify prompt, select model, call provider, log result."""

    def __init__(
        self,
        config_path: str | Path = "configs/default.yaml",
        *,
        classifier: ClassifierBase | None = None,
        provider: ProviderBase | None = None,
        tracker: Tracker | None = None,
        db_path: str | Path = "mmrouter.db",
    ):
        self._config: RoutingConfig = load_config(config_path)
        self._classifier = classifier or RuleClassifier()
        self._provider = provider or LiteLLMProvider(self._config.provider)
        self._tracker = tracker or Tracker(db_path)

    def route(self, prompt: str) -> RoutingResult:
        classification = self._classifier.classify(prompt)

        route = self._config.get_route(classification.complexity, classification.category)
        if not route:
            raise ValueError(
                f"No route for {classification.complexity}/{classification.category}"
            )

        models_to_try = [route.model] + route.fallbacks
        fallback_used = False
        last_error = None

        for i, model in enumerate(models_to_try):
            try:
                completion = self._provider.complete(prompt, model)
                if i > 0:
                    fallback_used = True

                result = RoutingResult(
                    classification=classification,
                    completion=completion,
                    model_used=model,
                    fallback_used=fallback_used,
                )

                self._tracker.log(RequestLog(
                    prompt_hash=RequestLog.hash_prompt(prompt),
                    classification=classification,
                    model_used=model,
                    completion=completion,
                    fallback_used=fallback_used,
                ))

                return result
            except ProviderError as e:
                last_error = e
                if not e.retryable:
                    continue
                continue

        raise RuntimeError(
            f"All models failed for {classification.complexity}/{classification.category}. "
            f"Tried: {models_to_try}. Last error: {last_error}"
        )

    def classify(self, prompt: str) -> ClassificationResult:
        return self._classifier.classify(prompt)

    def get_stats(self) -> dict:
        return self._tracker.get_stats()

    def close(self) -> None:
        self._tracker.close()
