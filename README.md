# mmrouter

Intelligent LLM request routing. A classifier analyzes each prompt and routes it to the right model: simple queries go to cheap/fast models (Haiku), complex reasoning goes to powerful ones (Opus), everything else to the balanced middle (Sonnet).

This is product-driven routing, not a "user picks a model" proxy. The system decides which model fits the task.

## Why this exists

Most LLM applications send every request to the same model. That's wasteful. "What's the capital of France?" doesn't need Opus. A multi-step reasoning problem shouldn't go to Haiku.

The cost difference is real:

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|----------------------|
| Claude Haiku | $0.80 | $4.00 |
| Claude Sonnet | $3.00 | $15.00 |
| Claude Opus | $15.00 | $75.00 |

Haiku is 3.8x cheaper than Sonnet per request. Sonnet is 5x cheaper than Opus. If your traffic is mostly simple queries, routing saves real money.

### Cost math

Assume 1,000 requests/day, 500 input + 1,000 output tokens each.

| Scenario | Monthly cost | vs All-Sonnet | vs All-Opus |
|----------|-------------|---------------|-------------|
| All-Opus | $2,475 | +400% | baseline |
| All-Sonnet | $495 | baseline | -80% |
| Routed (60/30/10) | $475 | -4% | -81% |
| Routed (70/25/5) | $340 | -31% | -86% |

Where "60/30/10" means 60% simple (Haiku), 30% medium (Sonnet), 10% complex (Opus).

The savings depend entirely on your traffic mix. If most of your requests are simple lookups and straightforward tasks, routing pays off significantly. If your workload is mostly complex reasoning, you're sending most requests to Opus anyway and routing won't save much. The router also adds quality value: complex prompts get routed to more capable models instead of being handled by a cheaper one that might produce worse results.

## Architecture

```
Prompt -> Classifier -> Router Engine -> Provider (LiteLLM) -> Response
                             |
                    +--------+--------+
                    |        |        |
              Circuit    Cascade   Budget
              Breaker    Routing   Manager
                    |        |        |
                    +--------+--------+
                             |
                   Tracker (SQLite) -> Alerts
                             |
                 +-----------+-----------+
                 |                       |
          Dashboard               REST API
       (FastAPI+React)       (OpenAI-compatible)
```

**Classifier** analyzes the prompt along two dimensions:
- **Complexity**: simple, medium, complex
- **Category**: factual, reasoning, creative, code

**Router Engine** maps (complexity, category) to a model using YAML config. Supports confidence-based escalation, cascade routing, budget constraints, adaptive reranking, and A/B testing.

**Circuit Breaker** tracks failures per model and per provider. After consecutive failures, routes to fallback models automatically.

**Cascade Routing** tries the cheapest model first. If the response fails a quality gate (too short, hedging phrases), escalates to a stronger model.

**Budget Manager** enforces daily spending limits. Dynamically downgrades model selection as spend approaches the limit.

**Tracker** logs every request to SQLite (WAL mode): model used, tokens, cost, latency, classification result.

## Three classifier strategies

| Strategy | Accuracy | Cost | Speed | Notes |
|----------|----------|------|-------|-------|
| Rules | 67% overall (78% complexity, 83% category) | Free | <1ms | Pattern matching heuristics. No dependencies. |
| Embeddings | 78% overall (84% complexity, 90% category) | Free | ~50ms | kNN on sentence-transformers (MiniLM-L6-v2). Needs `pip install mmrouter[embeddings]`. |
| LLM | Not yet benchmarked | API cost per request | ~1s | Uses a cheap model (Haiku) to classify before routing. Requires API key. |

The embedding classifier is the best default for production. Rules work fine for development and testing. Custom training on your own data is supported via `mmrouter train`.

## Install

```bash
# Basic (rules classifier only)
pip install -e .

# With embedding classifier
pip install -e ".[embeddings]"

# With dashboard
pip install -e ".[dashboard]"

# With REST API server
pip install -e ".[server]"

# Everything (dev)
pip install -e ".[dev,embeddings,dashboard,server]"
```

Requires Python 3.11+.

## Configuration

Set your API key:

```bash
export ANTHROPIC_API_KEY=sk-...
```

Routing is configured in `configs/default.yaml`. The config maps (complexity, category) pairs to models with fallback chains:

```yaml
routes:
  simple:
    factual:
      model: claude-haiku-4-5-20251001
      fallbacks:
        - claude-sonnet-4-6
  medium:
    reasoning:
      model: claude-sonnet-4-6
      fallbacks:
        - claude-opus-4-6
  complex:
    reasoning:
      model: claude-opus-4-6
      fallbacks:
        - claude-sonnet-4-6

classifier:
  strategy: rules        # rules | embeddings | llm
  threshold: "0.7"       # confidence below this triggers escalation

provider:
  timeout_ms: 30000
  max_retries: 2
  circuit_breaker_threshold: 5
  circuit_breaker_reset_ms: 60000
  prompt_caching: true    # auto-annotate for Anthropic caching

cascade:
  enabled: false          # try cheapest model first, escalate on low quality
  strategy: heuristic     # heuristic | llm_judge
  min_response_length: 50

budget:
  enabled: false
  daily_limit: 0.0        # dollars, 0 = unlimited
  warn_threshold: 0.75
  downgrade_threshold: 0.90
  hard_limit_action: cheapest  # cheapest | reject

adaptive:
  enabled: false          # feedback-driven model reranking
  min_feedback_count: 20
  decay_days: 30

alerts:
  enabled: false
  cooldown_seconds: 300
  rules:
    - cost_spike
    - error_rate
    - budget_warning
```

The config is designed to be edited by PMs, not just engineers. Change which model handles what by editing the YAML.

Example configs:
- `configs/default.yaml` - single-provider Anthropic setup
- `configs/cascade.yaml` - cascade routing with quality gates enabled
- `configs/budget.yaml` - budget-constrained routing ($5/day limit)
- `configs/multi-provider.yaml` - cross-provider fallbacks (Anthropic + OpenAI + Google)

## CLI usage

### Route a prompt

```bash
mmrouter route "What is the capital of France?"
# Paris.
# [simple/factual] model=claude-haiku-4-5-20251001 cost=$0.000320 latency=245ms tokens=12+8

mmrouter route "Analyze the trade-offs between microservices and monoliths" -v
# (detailed response)
# [complex/reasoning] model=claude-opus-4-6 cost=$0.003200 latency=2100ms tokens=45+320
```

### Classify only (no API call)

```bash
mmrouter classify "Explain quantum entanglement"
# {
#   "complexity": "medium",
#   "category": "reasoning",
#   "confidence": 0.85
# }

mmrouter classify "Write a haiku about rain" --classifier embeddings
```

### Run accuracy eval

```bash
mmrouter eval --classifier rules
# Overall:    67.0%  (80/120)
# Complexity: 78.0%
# Category:   83.0%

mmrouter eval --classifier embeddings
# Overall:    78.0%  (94/120)
# Complexity: 84.0%
# Category:   90.0%
```

### Compare all classifiers

```bash
mmrouter compare
# Classifier   Overall  Complexity  Category   Time
# ----------   -------  ----------  --------   -----
# rules          67.0%       78.0%     83.0%  0.01s
# embeddings     78.0%       84.0%     90.0%  1.23s
```

### Train custom embedding classifier

```bash
mmrouter train --data my_data.yaml --output models/custom --eval-split 0.2
# Loaded 500 examples
# Eval split: 400 train, 100 eval
# Training... done
# Eval accuracy: 82.0%
# Saved to models/custom/

# Use trained model:
mmrouter classify "prompt" --classifier embeddings --trained-model models/custom
```

### Cost analytics

```bash
mmrouter stats
# Requests:      142
# Total cost:    $0.234500
# Avg latency:   380ms
# Tokens in/out: 15200/28400
# Fallbacks:     3

mmrouter stats --detailed
# (adds daily cost breakdown, savings vs all-Sonnet baseline,
#  distribution by complexity/category, cascade stats, budget status)
```

### LLM-as-judge quality eval

```bash
mmrouter quality --sample 20
# Evaluates whether routed responses match baseline (all-Sonnet) quality.
# Reports score, relevance, accuracy, completeness deltas.
```

### Feedback for adaptive routing

```bash
mmrouter feedback <request_id> up    # thumbs up
mmrouter feedback <request_id> down  # thumbs down
```

### A/B testing

```bash
mmrouter experiment create --name "cascade-test" \
  --control configs/default.yaml \
  --treatment configs/cascade.yaml \
  --split 0.5

mmrouter experiment status
mmrouter experiment stop
```

### Alerting

```bash
mmrouter alerts status
mmrouter alerts test --webhook-url https://hooks.slack.com/services/...
```

### REST API server (OpenAI-compatible)

```bash
mmrouter serve --port 8080
# Starts OpenAI-compatible API at http://localhost:8080

# Use with any OpenAI SDK client:
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "Hello"}]}'

# Or with the OpenAI Python SDK:
# client = OpenAI(base_url="http://localhost:8080/v1", api_key="any")
# response = client.chat.completions.create(model="auto", messages=[...])
```

`model: "auto"` triggers intelligent routing. Explicit model names bypass classification and go directly to the provider.

Classification metadata is returned in `X-MMRouter-*` response headers.

Auth: set `MMROUTER_API_KEY` env var to require Bearer token auth. Unset = no auth (local dev).

### Dashboard

```bash
mmrouter dashboard --port 8000
# Starts FastAPI backend + serves React SPA at http://localhost:8000
```

## Key features

### Multi-provider failover
Cross-provider fallback chains with provider-level circuit breaker. If Anthropic is down, routes to OpenAI/Google automatically. See `configs/multi-provider.yaml`.

### Cascade routing
Try the cheapest model first. If the response fails a quality gate (too short, contains hedging phrases), automatically escalate to a stronger model. Saves cost when cheap models can handle the task. See `configs/cascade.yaml`.

### Budget mode
Set a daily spending limit. The router dynamically downgrades model selection as spend approaches the limit:
- <75%: normal routing
- 75-90%: warn (log only)
- 90-100%: downgrade (complex->medium, medium->simple)
- 100%+: force cheapest model or reject requests

### Adaptive routing
Track user feedback (thumbs up/down) via API. The router learns which models perform best for each query type and reranks the fallback chain accordingly.

### A/B testing
Run two routing configs simultaneously. Traffic is split deterministically (same prompt always goes to same variant). Compare cost, latency, and error rates between strategies.

### Prompt caching
Automatic `cache_control` annotation for Anthropic models. System prompts get cached server-side, reducing cost by up to 90% on cached input tokens. OpenAI caching is automatic and requires no annotation.

### Alerting
Webhook notifications (works with Slack incoming webhooks) for cost spikes, high error rates, and budget warnings. Cooldown prevents alert spam.

## Project structure

```
src/mmrouter/
  classifier/          # RuleClassifier, EmbeddingClassifier, LLMClassifier
  router/              # Engine, Config, Cascade, Budget, Fallback, Adaptive
  providers/           # LiteLLM wrapper (ProviderBase ABC), cache annotation
  tracker/             # SQLite logger, cost analytics
  eval/                # Accuracy eval, classifier comparison, LLM-as-judge quality
  server/              # OpenAI-compatible REST API (FastAPI)
  dashboard/           # Dashboard backend (FastAPI)
  experiments/         # A/B testing engine (store, traffic splitter)
  alerts/              # Alert rules, webhook/log channels
  cli.py               # Click entry point
  api.py               # Programmatic API
  models.py            # Shared data models (Pydantic)
configs/               # YAML routing configs (default, cascade, budget, multi-provider)
eval_data/             # Labeled test queries for eval
dashboard/             # React + Vite + Recharts SPA
tests/                 # 481 tests (pytest)
```

## Stack

- Python 3.11+, Click (CLI), FastAPI (server + dashboard)
- LiteLLM for multi-provider model access (pinned version, isolated behind ProviderBase)
- sentence-transformers for embedding classifier (MiniLM-L6-v2, runs locally)
- SQLite with WAL mode for request/cost/feedback/experiment tracking
- React + Vite + Recharts for the dashboard
- strictyaml for config parsing
- pytest (481 tests)

## Running tests

```bash
pytest                              # all tests
pytest tests/test_classifier/       # classifier tests only
pytest tests/test_router/           # router tests only
pytest tests/test_server/           # REST API tests only
```

## Hard rules

- API keys only via environment variables. Never in code, config, or logs.
- LiteLLM is isolated behind `ProviderBase`. Never imported outside `providers/`.
- Model names live in YAML config. Never hardcoded in routing logic.
- All LLM calls go through the Router. No direct provider calls from outside `router/`.

## License

MIT
