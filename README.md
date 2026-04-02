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
                       Circuit Breaker
                             |
                        Tracker (SQLite)
                             |
                    Dashboard (FastAPI + React)
```

**Classifier** analyzes the prompt along two dimensions:
- **Complexity**: simple, medium, complex
- **Category**: factual, reasoning, creative, code

**Router Engine** maps (complexity, category) to a model using YAML config. If the primary model fails, it tries fallbacks.

**Circuit Breaker** tracks failures per provider. After 5 consecutive failures, the circuit opens and routes to fallback models for 60 seconds before retrying.

**Tracker** logs every request to SQLite (WAL mode): model used, tokens, cost, latency, classification result.

## Three classifier strategies

| Strategy | Accuracy | Cost | Speed | Notes |
|----------|----------|------|-------|-------|
| Rules | 67% overall (78% complexity, 83% category) | Free | <1ms | Pattern matching heuristics. No dependencies. |
| Embeddings | 78% overall (84% complexity, 90% category) | Free | ~50ms | kNN on sentence-transformers (MiniLM-L6-v2). Needs `pip install mmrouter[embeddings]`. |
| LLM | Not yet benchmarked | API cost per request | ~1s | Uses a cheap model (Haiku) to classify before routing. Requires API key. |

The embedding classifier is the best default for production. Rules work fine for development and testing.

## Install

```bash
# Basic (rules classifier only)
pip install -e .

# With embedding classifier
pip install -e ".[embeddings]"

# With dashboard
pip install -e ".[dashboard]"

# Everything (dev)
pip install -e ".[dev,embeddings,dashboard]"
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
    reasoning:
      model: claude-haiku-4-5-20251001
      fallbacks:
        - claude-sonnet-4-6
    creative:
      model: claude-sonnet-4-6
      fallbacks:
        - claude-haiku-4-5-20251001
    # ... code category omitted
  medium:
    reasoning:
      model: claude-sonnet-4-6
      fallbacks:
        - claude-opus-4-6
    # ... other categories follow same pattern
  complex:
    reasoning:
      model: claude-opus-4-6
      fallbacks:
        - claude-sonnet-4-6
    # ... see configs/default.yaml for full config

classifier:
  strategy: rules        # rules | embeddings | llm
  threshold: "0.7"       # confidence below this triggers escalation

provider:
  timeout_ms: 30000
  max_retries: 2
  circuit_breaker_threshold: 5
  circuit_breaker_reset_ms: 60000
```

The config is designed to be edited by PMs, not just engineers. Change which model handles what by editing the YAML. No code changes needed.

## CLI usage

### Route a prompt (classify + call model + log)

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

### Cost analytics

```bash
mmrouter stats
# Requests:      142
# Total cost:    $0.234500
# Avg latency:   380ms
# Tokens in/out: 15200/28400
# Fallbacks:     3

mmrouter stats --detailed
# (adds daily cost breakdown, savings vs all-Sonnet baseline, distribution by complexity/category)
```

### LLM-as-judge quality eval

```bash
mmrouter quality --sample 20
# Evaluates whether routed responses match baseline (all-Sonnet) quality.
# Reports score, relevance, accuracy, completeness deltas.
```

### Dashboard

```bash
mmrouter dashboard --port 8000
# Starts FastAPI backend + serves React SPA at http://localhost:8000
```

## Project structure

```
src/mmrouter/
  classifier/          # RuleClassifier, EmbeddingClassifier, LLMClassifier
  router/              # Engine (routing logic), Config (YAML loading), Fallback
  providers/           # LiteLLM wrapper (ProviderBase ABC)
  tracker/             # SQLite logger, cost analytics
  eval/                # Accuracy eval, classifier comparison, LLM-as-judge quality
  dashboard/           # FastAPI backend for React SPA
  cli.py               # Click entry point
  api.py               # Programmatic API
  models.py            # Shared data models (Pydantic)
configs/               # YAML routing configs
eval_data/             # Labeled test queries for eval
dashboard/             # React + Vite + Recharts SPA
tests/                 # pytest
```

## Stack

- Python 3.11+, Click (CLI), FastAPI (dashboard API)
- LiteLLM for multi-provider model access (pinned version, isolated behind ProviderBase)
- sentence-transformers for embedding classifier (MiniLM-L6-v2, runs locally)
- SQLite with WAL mode for request/cost tracking
- React + Vite + Recharts for the dashboard
- strictyaml for config parsing
- pytest for tests and eval

## Running tests

```bash
pytest                              # all tests
pytest tests/test_classifier/       # classifier tests only
```

## Hard rules

- API keys only via environment variables. Never in code, config, or logs.
- LiteLLM is isolated behind `ProviderBase`. Never imported outside `providers/`.
- Model names live in YAML config. Never hardcoded in routing logic.
- All LLM calls go through the Router. No direct provider calls from outside `router/`.

## License

MIT
