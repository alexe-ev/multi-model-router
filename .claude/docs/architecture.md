# Architecture

## Data flow

```
User Request (CLI / Python API / REST API)
  -> Experiment check (A/B testing: assign variant, load variant config)
  -> Classifier (rules | embeddings | llm) -> {complexity, category, confidence}
  -> Confidence escalation (low confidence -> bump complexity tier)
  -> Budget check (downgrade tier if approaching daily limit)
  -> Adaptive reranking (feedback-driven model reorder)
  -> Route lookup (YAML config) -> model chain
  -> Cascade routing (optional: try cheapest first, quality gate)
  -> Provider Layer (LiteLLM wrapper, prompt caching annotation)
       -> Circuit breaker check (per-model + per-provider)
       -> LLM API call (Anthropic / OpenAI / Google)
  -> Tracker (SQLite) -> log request + check alerts
  -> Response + metadata to user
```

## Components

### Core routing
- `mmrouter.router.engine` - orchestrates the full pipeline: classify -> route -> call -> track
- `mmrouter.router.config` - YAML config loader + validator (strictyaml)
- `mmrouter.router.fallback` - per-model + per-provider circuit breakers, CircuitBreakerRegistry
- `mmrouter.router.cascade` - quality gates (heuristic + LLM judge), cascade routing logic
- `mmrouter.router.budget` - BudgetManager: daily spend tracking, tier-based model downgrade
- `mmrouter.router.adaptive` - FeedbackScorer: model reranking based on user feedback

### Classification
- `mmrouter.classifier.*` - ClassifierBase ABC, three implementations (rules, embeddings, llm)
- Embedding classifier supports save/load for custom training

### Providers
- `mmrouter.providers.base` - ProviderBase ABC (complete, complete_messages, stream_messages)
- `mmrouter.providers.litellm_provider` - LiteLLM wrapper with retry logic
- `mmrouter.providers.cache` - Anthropic prompt caching auto-annotation

### Tracking & analytics
- `mmrouter.tracker.logger` - SQLite writer (WAL mode), feedback table, experiment columns
- `mmrouter.tracker.analytics` - cost aggregations, savings, cache stats, feedback stats

### Server & dashboard
- `mmrouter.server.*` - OpenAI-compatible REST API (FastAPI), Bearer auth, SSE streaming
- `mmrouter.dashboard.*` - Analytics dashboard backend (FastAPI) for React SPA

### Experiments & alerts
- `mmrouter.experiments.*` - A/B testing: ExperimentStore (SQLite), deterministic traffic splitter
- `mmrouter.alerts.*` - Rule-based alerting with webhook/log channels, cooldown

### Entry points
- `mmrouter.cli` - Click CLI (15+ commands)
- `mmrouter.api` - Programmatic Python API

## Key abstractions

- `ClassifierBase` - all classifiers implement `classify(prompt) -> ClassificationResult`
- `ProviderBase` - provider layer isolated behind interface (complete, complete_messages, stream_messages)
- `RoutingConfig` - parsed YAML, immutable after load. Includes routes, cascade, budget, adaptive, alerts configs.
- `RoutingResult` - classification + completion + metadata (model, fallback, cascade, budget, adaptive flags)
- `StreamRouteResult` - streaming variant: classification + model + chunk iterator (replaces tuple)
- `CircuitBreakerRegistry` - two-layer breakers (per-model + per-provider)
- `BudgetManager` - daily spend tracking with 1s TTL cache
- `FeedbackScorer` - per-(model, complexity, category) success rate from user feedback, 1s per-bucket TTL cache
- `ExperimentStore` - SQLite-backed experiment lifecycle (max 1 active)
- `AlertManager` - rule evaluation + cooldown + channel dispatch

## Dashboard

- Backend: FastAPI serving JSON from SQLite (6 endpoints)
- Frontend: React SPA (Vite + Recharts) with KPI cards, charts, request log
- Separate process from the REST API server

## REST API server

- OpenAI-compatible POST /v1/chat/completions
- model: "auto" triggers routing, explicit model names use Router.passthrough_messages()
- Streaming via SSE, classification metadata in X-MMRouter-* headers
- Bearer token auth (MMROUTER_API_KEY env var, constant-time comparison via hmac.compare_digest)
- POST /v1/feedback for adaptive routing feedback
