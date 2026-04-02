"""Router engine: classify -> select model -> call provider -> track."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from pydantic import BaseModel

from mmrouter.classifier import ClassifierBase
from mmrouter.classifier.rules import RuleClassifier
from mmrouter.models import (
    CascadeConfig,
    ClassificationResult,
    CompletionResult,
    Complexity,
    ModelRoute,
    RequestLog,
    RoutingConfig,
    StreamChunk,
)
from mmrouter.providers.base import ProviderBase, _extract_last_user_message
from mmrouter.providers.litellm_provider import LiteLLMProvider, ProviderError
from mmrouter.router.cascade import QualityGateResult, create_quality_gate
from mmrouter.router.config import load_config
from mmrouter.router.fallback import CircuitBreakerRegistry, CircuitOpenError
from mmrouter.tracker.logger import Tracker

_ESCALATION_MAP = {
    Complexity.SIMPLE: Complexity.MEDIUM,
    Complexity.MEDIUM: Complexity.COMPLEX,
    Complexity.COMPLEX: Complexity.COMPLEX,
}


class RoutingResult(BaseModel):
    classification: ClassificationResult
    completion: CompletionResult
    model_used: str
    fallback_used: bool = False
    escalated: bool = False
    cascade_used: bool = False
    cascade_attempts: int = 1


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
        self._breakers = CircuitBreakerRegistry(self._config.provider)

    def _get_cascade_chain(self, route: ModelRoute) -> list[str]:
        """Get cascade chain for a route: per-route if defined, else all unique models cheap-first."""
        if route.cascade:
            return list(route.cascade)
        # Deduplicate: route.model + fallbacks, preserving order
        seen = set()
        chain = []
        for m in [route.model] + route.fallbacks:
            if m not in seen:
                seen.add(m)
                chain.append(m)
        return chain

    def _route_cascade(
        self,
        prompt: str,
        classification: ClassificationResult,
        route: ModelRoute,
        escalated: bool,
    ) -> RoutingResult:
        """Try models cheapest-first, escalate if quality gate fails."""
        cascade_chain = self._get_cascade_chain(route)
        gate = create_quality_gate(self._config.cascade, self._provider)

        last_completion: CompletionResult | None = None
        last_model: str | None = None
        attempts = 0

        for model in cascade_chain:
            breaker = self._breakers.get(model)
            try:
                breaker.check()
            except CircuitOpenError:
                continue

            try:
                completion = self._provider.complete(prompt, model)
                breaker.record_success()
                attempts += 1
                last_completion = completion
                last_model = model

                gate_result = gate.check(prompt, completion)
                if gate_result.passed:
                    result = RoutingResult(
                        classification=classification,
                        completion=completion,
                        model_used=model,
                        escalated=escalated,
                        cascade_used=True,
                        cascade_attempts=attempts,
                    )
                    self._tracker.log(RequestLog(
                        prompt_hash=RequestLog.hash_prompt(prompt),
                        classification=classification,
                        model_used=model,
                        completion=completion,
                        cascade_used=True,
                        cascade_attempts=attempts,
                    ))
                    return result
                # Quality gate failed, try next model
            except ProviderError as e:
                breaker.record_failure(e.retryable)
                continue

        # All models tried, none passed quality gate. Return last response (best effort).
        if last_completion and last_model:
            result = RoutingResult(
                classification=classification,
                completion=last_completion,
                model_used=last_model,
                escalated=escalated,
                cascade_used=True,
                cascade_attempts=attempts,
            )
            self._tracker.log(RequestLog(
                prompt_hash=RequestLog.hash_prompt(prompt),
                classification=classification,
                model_used=last_model,
                completion=last_completion,
                cascade_used=True,
                cascade_attempts=attempts,
            ))
            return result

        raise RuntimeError(
            f"Cascade failed: all models unavailable for "
            f"{classification.complexity}/{classification.category}. "
            f"Chain: {cascade_chain}"
        )

    def route(self, prompt: str) -> RoutingResult:
        classification = self._classifier.classify(prompt)

        complexity = classification.complexity
        escalated = False
        if classification.confidence < self._config.classifier.threshold:
            new_complexity = _ESCALATION_MAP[complexity]
            if new_complexity != complexity:
                escalated = True
                complexity = new_complexity

        route = self._config.get_route(complexity, classification.category)
        if not route:
            raise ValueError(
                f"No route for {complexity}/{classification.category}"
            )

        # Cascade routing path
        if self._config.cascade.enabled:
            return self._route_cascade(prompt, classification, route, escalated)

        # Standard routing path (unchanged)
        models_to_try = [route.model] + route.fallbacks
        fallback_used = False
        last_error = None

        for i, model in enumerate(models_to_try):
            breaker = self._breakers.get(model)
            try:
                breaker.check()
            except CircuitOpenError as e:
                last_error = e
                continue

            try:
                completion = self._provider.complete(prompt, model)
                breaker.record_success()
                if i > 0:
                    fallback_used = True

                result = RoutingResult(
                    classification=classification,
                    completion=completion,
                    model_used=model,
                    fallback_used=fallback_used,
                    escalated=escalated,
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
                breaker.record_failure(e.retryable)
                last_error = e
                continue

        raise RuntimeError(
            f"All models failed for {classification.complexity}/{classification.category}. "
            f"Tried: {models_to_try}. Last error: {last_error}"
        )

    def route_messages(self, messages: list[dict], **kwargs) -> RoutingResult:
        """Route a messages array. Classify based on last user message."""
        prompt = _extract_last_user_message(messages)
        classification = self._classifier.classify(prompt)

        complexity = classification.complexity
        escalated = False
        if classification.confidence < self._config.classifier.threshold:
            new_complexity = _ESCALATION_MAP[complexity]
            if new_complexity != complexity:
                escalated = True
                complexity = new_complexity

        route = self._config.get_route(complexity, classification.category)
        if not route:
            raise ValueError(
                f"No route for {complexity}/{classification.category}"
            )

        # Standard routing path for messages (no cascade for messages yet)
        models_to_try = [route.model] + route.fallbacks
        fallback_used = False
        last_error = None

        for i, model in enumerate(models_to_try):
            breaker = self._breakers.get(model)
            try:
                breaker.check()
            except CircuitOpenError as e:
                last_error = e
                continue

            try:
                completion = self._provider.complete_messages(messages, model, **kwargs)
                breaker.record_success()
                if i > 0:
                    fallback_used = True

                result = RoutingResult(
                    classification=classification,
                    completion=completion,
                    model_used=model,
                    fallback_used=fallback_used,
                    escalated=escalated,
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
                breaker.record_failure(e.retryable)
                last_error = e
                continue

        raise RuntimeError(
            f"All models failed for {classification.complexity}/{classification.category}. "
            f"Tried: {models_to_try}. Last error: {last_error}"
        )

    def route_messages_stream(
        self, messages: list[dict], **kwargs
    ) -> tuple[ClassificationResult, str, bool, bool, Iterator[StreamChunk]]:
        """Stream version: returns (classification, model_name, fallback_used, escalated, chunk_iterator).

        Classification and model selection happen immediately.
        The actual LLM call is lazy via the iterator.
        """
        prompt = _extract_last_user_message(messages)
        classification = self._classifier.classify(prompt)

        complexity = classification.complexity
        escalated = False
        if classification.confidence < self._config.classifier.threshold:
            new_complexity = _ESCALATION_MAP[complexity]
            if new_complexity != complexity:
                escalated = True
                complexity = new_complexity

        route = self._config.get_route(complexity, classification.category)
        if not route:
            raise ValueError(
                f"No route for {complexity}/{classification.category}"
            )

        models_to_try = [route.model] + route.fallbacks
        last_error = None

        for i, model in enumerate(models_to_try):
            breaker = self._breakers.get(model)
            try:
                breaker.check()
            except CircuitOpenError as e:
                last_error = e
                continue

            fallback_used = i > 0
            chunks = self._provider.stream_messages(messages, model, **kwargs)
            return classification, model, fallback_used, escalated, chunks

        raise RuntimeError(
            f"All models failed for {classification.complexity}/{classification.category}. "
            f"Tried: {models_to_try}. Last error: {last_error}"
        )

    def classify(self, prompt: str) -> ClassificationResult:
        return self._classifier.classify(prompt)

    def get_config(self) -> RoutingConfig:
        """Return the current routing config."""
        return self._config

    def get_stats(self) -> dict:
        return self._tracker.get_stats()

    def close(self) -> None:
        self._tracker.close()
