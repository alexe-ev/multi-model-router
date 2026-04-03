"""Router engine: classify -> select model -> call provider -> track."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from pydantic import BaseModel

from mmrouter.classifier import ClassifierBase
from mmrouter.classifier.rules import RuleClassifier
from mmrouter.experiments.splitter import assign_variant
from mmrouter.experiments.store import ExperimentStore
from mmrouter.alerts.rules import (
    BUILTIN_RULES,
    AlertManager,
    create_budget_warning_rule,
)
from mmrouter.models import (
    CascadeConfig,
    ClassificationResult,
    CompletionResult,
    Complexity,
    Experiment,
    ModelRoute,
    RequestLog,
    RoutingConfig,
    StreamChunk,
)
from mmrouter.providers.base import ProviderBase, _extract_last_user_message
from mmrouter.providers.litellm_provider import LiteLLMProvider, ProviderError
from mmrouter.router.adaptive import FeedbackScorer
from mmrouter.router.budget import BudgetExceededError, BudgetManager
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
    request_id: int | None = None
    fallback_used: bool = False
    escalated: bool = False
    cascade_used: bool = False
    cascade_attempts: int = 1
    budget_downgraded: bool = False
    adaptive_reranked: bool = False
    experiment_id: int | None = None
    variant: str | None = None


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
        self._config_path = str(config_path)
        self._config: RoutingConfig = load_config(config_path)
        self._classifier = classifier or RuleClassifier()
        self._provider = provider or LiteLLMProvider(self._config.provider)
        self._tracker = tracker or Tracker(db_path)
        self._breakers = CircuitBreakerRegistry(self._config.provider)
        self._budget = BudgetManager(self._config.budget, self._tracker.connection)
        self._scorer: FeedbackScorer | None = None
        if self._config.adaptive.enabled:
            self._scorer = FeedbackScorer(
                self._tracker.connection, self._config.adaptive
            )
        self._experiment_store = ExperimentStore(self._tracker.connection)
        # Cache for loaded experiment configs (config_path -> RoutingConfig)
        self._config_cache: dict[str, RoutingConfig] = {
            self._config_path: self._config,
        }
        # Alerting
        self._alert_manager: AlertManager | None = None
        if self._config.alerts.enabled:
            alert_rules = []
            for rule_name in self._config.alerts.rules:
                if rule_name == "budget_warning":
                    alert_rules.append(create_budget_warning_rule(
                        daily_limit=self._config.budget.daily_limit,
                        cooldown=self._config.alerts.cooldown_seconds,
                    ))
                elif rule_name in BUILTIN_RULES:
                    alert_rules.append(BUILTIN_RULES[rule_name])
            self._alert_manager = AlertManager(
                self._tracker.connection,
                rules=alert_rules,
                webhook_url=self._config.alerts.webhook_url,
                cooldown_seconds=self._config.alerts.cooldown_seconds,
            )

    def _load_config(self, path: str) -> RoutingConfig:
        """Load and cache a routing config."""
        if path not in self._config_cache:
            self._config_cache[path] = load_config(path)
        return self._config_cache[path]

    def _resolve_experiment(self, prompt_hash: str) -> tuple[RoutingConfig, int | None, str | None]:
        """Check active experiment, assign variant, return (config, experiment_id, variant).

        If no active experiment, returns (self._config, None, None).
        """
        experiment = self._experiment_store.get_active()
        if experiment is None:
            return self._config, None, None

        variant = assign_variant(prompt_hash, experiment.traffic_split)
        config_path = (
            experiment.treatment_config if variant == "treatment"
            else experiment.control_config
        )
        config = self._load_config(config_path)
        return config, experiment.id, variant

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

    def _record_failure_and_propagate(self, model: str, retryable: bool) -> None:
        """Record failure on model breaker and propagate to provider level if it opens."""
        breaker = self._breakers.get(model)
        was_open = breaker.state.value == "open"
        breaker.record_failure(retryable)
        is_open_now = breaker.state.value == "open"
        if not was_open and is_open_now:
            self._breakers.record_model_open(model)
        if retryable:
            self._breakers.record_provider_failure(model)

    def _record_success(self, model: str) -> None:
        """Record success on model breaker and propagate to provider level."""
        breaker = self._breakers.get(model)
        breaker.record_success()
        self._breakers.record_provider_success(model)

    def _route_cascade(
        self,
        prompt: str,
        classification: ClassificationResult,
        route: ModelRoute,
        escalated: bool,
        *,
        config: RoutingConfig | None = None,
        experiment_id: int | None = None,
        variant: str | None = None,
    ) -> RoutingResult:
        """Try models cheapest-first, escalate if quality gate fails."""
        effective_config = config or self._config
        cascade_chain = self._get_cascade_chain(route)

        # Adaptive reranking for cascade chain
        adaptive_reranked = False
        if self._scorer:
            cascade_chain, adaptive_reranked = self._scorer.rerank_models(
                cascade_chain,
                classification.complexity.value,
                classification.category.value,
            )

        gate = create_quality_gate(effective_config.cascade, self._provider)
        prompt_hash = RequestLog.hash_prompt(prompt)

        last_completion: CompletionResult | None = None
        last_model: str | None = None
        attempts = 0

        for model in cascade_chain:
            # Provider-level check first (fast skip entire provider)
            try:
                self._breakers.check_provider(model)
            except CircuitOpenError:
                continue

            breaker = self._breakers.get(model)
            try:
                breaker.check()
            except CircuitOpenError:
                continue

            try:
                completion = self._provider.complete(prompt, model)
                self._record_success(model)
                attempts += 1
                last_completion = completion
                last_model = model

                gate_result = gate.check(prompt, completion)
                if gate_result.passed:
                    request_id = self._tracker.log(RequestLog(
                        prompt_hash=prompt_hash,
                        classification=classification,
                        model_used=model,
                        completion=completion,
                        cascade_used=True,
                        cascade_attempts=attempts,
                        experiment_id=experiment_id,
                        variant=variant,
                    ))
                    result = RoutingResult(
                        classification=classification,
                        completion=completion,
                        model_used=model,
                        request_id=request_id,
                        escalated=escalated,
                        cascade_used=True,
                        cascade_attempts=attempts,
                        adaptive_reranked=adaptive_reranked,
                        experiment_id=experiment_id,
                        variant=variant,
                    )
                    self._check_alerts()
                    return result
                # Quality gate failed, try next model
            except ProviderError as e:
                self._record_failure_and_propagate(model, e.retryable)
                continue

        # All models tried, none passed quality gate. Return last response (best effort).
        if last_completion and last_model:
            request_id = self._tracker.log(RequestLog(
                prompt_hash=prompt_hash,
                classification=classification,
                model_used=last_model,
                completion=last_completion,
                cascade_used=True,
                cascade_attempts=attempts,
                experiment_id=experiment_id,
                variant=variant,
            ))
            result = RoutingResult(
                classification=classification,
                completion=last_completion,
                model_used=last_model,
                request_id=request_id,
                escalated=escalated,
                cascade_used=True,
                cascade_attempts=attempts,
                adaptive_reranked=adaptive_reranked,
                experiment_id=experiment_id,
                variant=variant,
            )
            return result

        raise RuntimeError(
            f"Cascade failed: all models unavailable for "
            f"{classification.complexity}/{classification.category}. "
            f"Chain: {cascade_chain}"
        )

    def route(self, prompt: str) -> RoutingResult:
        classification = self._classifier.classify(prompt)
        prompt_hash = RequestLog.hash_prompt(prompt)

        # Experiment: resolve which config to use
        config, experiment_id, variant = self._resolve_experiment(prompt_hash)

        complexity = classification.complexity
        escalated = False
        if classification.confidence < config.classifier.threshold:
            new_complexity = _ESCALATION_MAP[complexity]
            if new_complexity != complexity:
                escalated = True
                complexity = new_complexity

        # Budget enforcement: may downgrade complexity or reject
        budget_downgraded = False
        if self._budget.enabled:
            new_complexity = self._budget.apply_budget(complexity)
            if new_complexity != complexity:
                budget_downgraded = True
                complexity = new_complexity

        route = config.get_route(complexity, classification.category)
        if not route:
            raise ValueError(
                f"No route for {complexity}/{classification.category}"
            )

        # Cascade routing path
        if config.cascade.enabled:
            result = self._route_cascade(
                prompt, classification, route, escalated,
                config=config, experiment_id=experiment_id, variant=variant,
            )
            result.budget_downgraded = budget_downgraded
            return result

        # Standard routing path
        models_to_try = [route.model] + route.fallbacks

        # Adaptive reranking
        adaptive_reranked = False
        if self._scorer:
            models_to_try, adaptive_reranked = self._scorer.rerank_models(
                models_to_try, complexity.value, classification.category.value
            )

        fallback_used = False
        last_error = None

        for i, model in enumerate(models_to_try):
            # Provider-level check first (fast skip entire provider)
            try:
                self._breakers.check_provider(model)
            except CircuitOpenError as e:
                last_error = e
                continue

            breaker = self._breakers.get(model)
            try:
                breaker.check()
            except CircuitOpenError as e:
                last_error = e
                continue

            try:
                completion = self._provider.complete(prompt, model)
                self._record_success(model)
                if i > 0:
                    fallback_used = True

                request_id = self._tracker.log(RequestLog(
                    prompt_hash=prompt_hash,
                    classification=classification,
                    model_used=model,
                    completion=completion,
                    fallback_used=fallback_used,
                    experiment_id=experiment_id,
                    variant=variant,
                ))

                result = RoutingResult(
                    classification=classification,
                    completion=completion,
                    model_used=model,
                    request_id=request_id,
                    fallback_used=fallback_used,
                    escalated=escalated,
                    budget_downgraded=budget_downgraded,
                    adaptive_reranked=adaptive_reranked,
                    experiment_id=experiment_id,
                    variant=variant,
                )

                self._check_alerts()
                return result
            except ProviderError as e:
                self._record_failure_and_propagate(model, e.retryable)
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

        # Budget enforcement
        budget_downgraded = False
        if self._budget.enabled:
            new_complexity = self._budget.apply_budget(complexity)
            if new_complexity != complexity:
                budget_downgraded = True
                complexity = new_complexity

        route = self._config.get_route(complexity, classification.category)
        if not route:
            raise ValueError(
                f"No route for {complexity}/{classification.category}"
            )

        # Standard routing path for messages (no cascade for messages yet)
        models_to_try = [route.model] + route.fallbacks

        # Adaptive reranking
        adaptive_reranked = False
        if self._scorer:
            models_to_try, adaptive_reranked = self._scorer.rerank_models(
                models_to_try, complexity.value, classification.category.value
            )

        fallback_used = False
        last_error = None

        for i, model in enumerate(models_to_try):
            # Provider-level check first (fast skip entire provider)
            try:
                self._breakers.check_provider(model)
            except CircuitOpenError as e:
                last_error = e
                continue

            breaker = self._breakers.get(model)
            try:
                breaker.check()
            except CircuitOpenError as e:
                last_error = e
                continue

            try:
                completion = self._provider.complete_messages(messages, model, **kwargs)
                self._record_success(model)
                if i > 0:
                    fallback_used = True

                request_id = self._tracker.log(RequestLog(
                    prompt_hash=RequestLog.hash_prompt(prompt),
                    classification=classification,
                    model_used=model,
                    completion=completion,
                    fallback_used=fallback_used,
                ))

                result = RoutingResult(
                    classification=classification,
                    completion=completion,
                    model_used=model,
                    request_id=request_id,
                    fallback_used=fallback_used,
                    escalated=escalated,
                    budget_downgraded=budget_downgraded,
                    adaptive_reranked=adaptive_reranked,
                )

                return result
            except ProviderError as e:
                self._record_failure_and_propagate(model, e.retryable)
                last_error = e
                continue

        raise RuntimeError(
            f"All models failed for {classification.complexity}/{classification.category}. "
            f"Tried: {models_to_try}. Last error: {last_error}"
        )

    def route_messages_stream(
        self, messages: list[dict], **kwargs
    ) -> tuple[ClassificationResult, str, bool, bool, bool, Iterator[StreamChunk]]:
        """Stream version: returns (classification, model_name, fallback_used, escalated, budget_downgraded, chunk_iterator).

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

        # Budget enforcement
        budget_downgraded = False
        if self._budget.enabled:
            new_complexity = self._budget.apply_budget(complexity)
            if new_complexity != complexity:
                budget_downgraded = True
                complexity = new_complexity

        route = self._config.get_route(complexity, classification.category)
        if not route:
            raise ValueError(
                f"No route for {complexity}/{classification.category}"
            )

        models_to_try = [route.model] + route.fallbacks
        last_error = None

        for i, model in enumerate(models_to_try):
            # Provider-level check first (fast skip entire provider)
            try:
                self._breakers.check_provider(model)
            except CircuitOpenError as e:
                last_error = e
                continue

            breaker = self._breakers.get(model)
            try:
                breaker.check()
            except CircuitOpenError as e:
                last_error = e
                continue

            fallback_used = i > 0
            chunks = self._provider.stream_messages(messages, model, **kwargs)
            return classification, model, fallback_used, escalated, budget_downgraded, chunks

        raise RuntimeError(
            f"All models failed for {classification.complexity}/{classification.category}. "
            f"Tried: {models_to_try}. Last error: {last_error}"
        )

    def _check_alerts(self) -> None:
        """Evaluate alert rules after a request. Non-blocking: failures are logged, not raised."""
        if self._alert_manager is None:
            return
        try:
            self._alert_manager.check_all()
        except Exception:
            pass  # Alert failures must never break routing

    def classify(self, prompt: str) -> ClassificationResult:
        return self._classifier.classify(prompt)

    def submit_feedback(self, request_id: int, rating: int) -> None:
        """Submit feedback for a routed request."""
        self._tracker.submit_feedback(request_id, rating)

    def get_config(self) -> RoutingConfig:
        """Return the current routing config."""
        return self._config

    def get_stats(self) -> dict:
        return self._tracker.get_stats()

    def get_feedback_stats(self) -> dict:
        """Return feedback analytics."""
        return self._tracker.get_feedback_stats()

    def get_budget_status(self) -> dict:
        """Return current budget status."""
        return self._budget.get_status()

    def get_alerts_status(self) -> dict:
        """Return alerting system status."""
        if self._alert_manager is None:
            return {"enabled": False}
        status = self._alert_manager.get_status()
        status["enabled"] = True
        return status

    @property
    def experiment_store(self) -> ExperimentStore:
        return self._experiment_store

    def close(self) -> None:
        self._tracker.close()
