# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Multi-Model Router: intelligent LLM request routing system. Classifier analyzes incoming prompts and routes to the optimal model: simple queries to cheap/fast (Haiku), complex reasoning to powerful (Opus), standard tasks to balanced (Sonnet). Product-driven routing, not a user-picks-model proxy.

## Stack

- Python 3.11+ / Click (CLI) / FastAPI (dashboard backend)
- LiteLLM (multi-provider API, pinned version, wrapped in ProviderBase)
- sentence-transformers (embedding classification, local)
- SQLite + WAL mode (request/cost logging)
- React + Vite + Recharts (dashboard SPA)
- strictyaml (config parsing)
- pytest (tests and eval)

## Structure

```
src/mmrouter/          # Python package
  classifier/          # rules.py, embeddings.py, llm_classifier.py
  router/              # engine.py, config.py, fallback.py
  providers/           # litellm_provider.py (behind ProviderBase ABC)
  tracker/             # logger.py (SQLite), analytics.py
  eval/                # evaluate.py, compare.py, quality.py
  cli.py               # Click entry point
  api.py               # Programmatic API
dashboard/             # React SPA (Vite)
configs/               # YAML routing configs
eval_data/             # Labeled test queries
```

## Commands

```bash
pip install -e .                    # Install for development
pytest                              # Run all tests
pytest tests/test_classifier/       # Run classifier tests only
mmrouter route "prompt"             # Classify + route + respond
mmrouter classify "prompt"          # Classify only (debug)
mmrouter eval                       # Run eval set
mmrouter stats                      # Show cost/routing summary
```

## Conventions

- All classifiers implement `ClassifierBase.classify(prompt) -> ClassificationResult`
- All providers implement `ProviderBase.complete(prompt, model) -> CompletionResult`
- Routing config is YAML, PM-editable. Never hardcode model names in Python.
- LiteLLM is isolated behind `ProviderBase`. Never import litellm outside `providers/`.
- API keys only via environment variables. Never in code, config, or logs.
- Prompts logged to SQLite but never to stdout/stderr in production.
- Circuit breaker is per-provider, not global.

## Docs

- `.claude/docs/architecture.md` - component diagram, data flow, key abstractions

## Workflow

Tasks tracked in Linear (initiative: Multi-Model Router, projects P0-P7).

**Commands are in `.claude/commands/`:**
- `/plan <args>` : technical analysis, task document, save to Linear
- `/ship <args>` : branch, executor agent, QA/review/security agents, commit, PR

**Flow:**
1. Plan chunk scope (which tasks, which Linear issues)
2. `/ship KN-XXX` for each task (reads from Linear, full agent pipeline)
3. After chunk: update Linear + memory + PLAN.md + findings

If `/plan` and `/ship` can't be invoked as skills, read `.claude/commands/plan.md` and `.claude/commands/ship.md` and follow the steps manually.

**When to skip /ship:** quick simple fixes (typos, one-liner patches, config tweaks). Everything else goes through /ship.

## Hard rules

- Never commit API keys, tokens, or secrets
- Never import litellm outside of `providers/` directory
- Never hardcode model names in routing logic (use YAML config)
- Never log API keys or full prompts to stdout
- All LLM calls go through the Router, never direct provider calls from outside `router/`
- Never write implementation code outside of /ship workflow (except quick simple fixes)
