# Findings log

Non-blocking findings from automated review agents.
Review periodically - if something repeats across PRs, create a Linear task.

---

## 2026-04-02 | KN-129 | qa, code-reviewer

- Warning: `evaluate.py` uses `yaml.safe_load` while rest of project uses `strictyaml`. Intentional for eval data (not config), but inconsistency worth noting.
- Warning: magic number `20` for max mismatches shown in CLI eval command. Consider named constant.
- Warning: `test_per_class_partial` test name is misleading (tests 100% accuracy, not partial).
- Warning: bucket comments in test_set.yaml can become stale when additional entries are added.
- Suggestion: some simple/code entries in test_set.yaml are arguably factual ("What does len() do?", "What is an API?"). Audit ground truth labels.

## 2026-04-02 | KN-133 | qa, code-reviewer

- Warning: `CircuitBreaker.state` (computed property) vs `_state` (stored) distinction is subtle. Consider inline comment on line where `record_failure` reads `self.state`.
- Warning: integration tests depend on model names from default.yaml (consistent with existing tests but fragile).
- Warning: `time.sleep(0.06)` in circuit breaker tests could be flaky under CI load. Consider injectable clock if it becomes a problem.

## 2026-04-02 | KN-135 | qa, code-reviewer

- Warning: `Tracker.connection` exposes conn with `Row` row_factory, but analytics uses integer index access. Both work with sqlite3.Row, but mixed patterns.
- Warning: `distribution()` query lacks COALESCE on SUM(cost), unlike `get_stats()` in logger.py. Not a bug (column is NOT NULL) but inconsistent.
- Note: unknown baseline model in savings returns zero actual_cost too. Acceptable since CLI hardcodes valid baseline.

## 2026-04-02 | KN-136 | qa, code-reviewer, security-auditor

- **CRITICAL (fixed):** `.format()` in judge prompt allowed format string injection. Braces in user content could crash eval or bypass XML isolation. Fixed: switched to `.replace()`.
- Warning: `compare_quality` returns plain dict, not Pydantic model. Fine for CLI consumption.
- Note: `_FALLBACK_SCORE` uses all 3s (midpoint). Silent fallbacks could skew averages if judge failures are frequent.

## 2026-04-02 | KN-137 | qa, code-reviewer, security-auditor

- Warning (medium): `Tracker.__new__` bypass in dashboard app.py. Fragile if Tracker.__init__ changes.
- Warning (medium): shared SQLite connection across FastAPI thread pool. Low risk for read-only dashboard.
- Note: CORS wildcard (*) fine for dev, needs lockdown for production.
- Note: test gaps: no tests for invalid query params, combined filters, or daily/distribution with empty DB.

## 2026-04-02 | KN-140 | qa, code-reviewer, security-auditor

- Suggestion: `_get_cascade_chain()` docstring says "cheap-first" but without per-route cascade list, order comes from route.model + fallbacks. For complex routes this means expensive model first. Use per-route cascade chains (as shown in cascade.yaml) for proper cheapest-first ordering.
- Note: DB migration for cascade_used/cascade_attempts columns is safe (checks existing columns before ALTER TABLE).

## 2026-04-02 | KN-141 | qa, code-reviewer, security-auditor

- Warning: `router._provider` accessed directly in server/app.py (lines 152, 202) to bypass routing for explicit model names. Breaks encapsulation. Better: add `Router.passthrough_messages(messages, model)` public method.
- Warning: `stream_messages` in litellm_provider.py has no retry/error wrapping. Raw litellm exceptions can leak as 500 instead of proper 502. Non-critical but inconsistent with `complete_messages`.
- Suggestion: Use `hmac.compare_digest()` in auth.py for constant-time API key comparison (prevents timing attacks).
- Suggestion: `route_messages_stream` returns a 5-tuple. A dataclass/NamedTuple would be more maintainable. (Now 6-tuple after KN-142 added budget_downgraded.)

## 2026-04-02 | KN-142 | qa, code-reviewer, security-auditor

- Note: SQL uses `date('now')` for daily spend query. No user input in SQL. Safe.
- Note: 1s TTL cache on daily spend prevents SQLite hammering under burst traffic.
- Suggestion: `route_messages_stream` tuple keeps growing (now 6 elements). Should be refactored to a dataclass before adding more fields.

## 2026-04-02 | KN-143 | qa, code-reviewer, security-auditor

- Note: Provider detection uses prefix heuristics (_PREFIX_TO_PROVIDER) + configurable provider_map override. Extensible.
- Note: Provider breaker threshold defaults to 2 (trips when 2+ models from same provider are OPEN). Configurable.
- Suggestion: `extract_provider` iterates a list for every model check. Could use a dict for O(1) if the list grows, but with 6 entries it's fine.

## 2026-04-02 | KN-144 | self-review

- Note: EmbeddingClassifier.load() uses cls.__new__() to bypass __init__. Same pattern as dashboard's Tracker bypass. Fragile if __init__ changes.

## 2026-04-02 | KN-145 | self-review

- Note: prompt_caching defaults to true in ProviderConfig. Means existing configs without the field get auto-annotation enabled.
- Note: cache.py annotation only targets last system message. Multi-system-message scenarios may miss earlier ones.

## 2026-04-02 | KN-147 | self-review

- Note: FeedbackScorer queries SQLite on every route when adaptive enabled. Same 1s TTL cache pattern as BudgetManager would help.
- Suggestion: Feedback endpoint returns 404 for unknown request_id. Should document that request_id comes from X-MMRouter-Request-Id header.

## 2026-04-02 | KN-146 | self-review

- Note: Experiment configs are loaded per-request via _resolve_experiment(). Should cache parsed configs.
- Note: ExperimentStore enforces max 1 active experiment at app level, not DB constraint. Race condition possible under concurrent writers.

## 2026-04-03 | KN-150 | self-review

- Note: WebhookChannel has 5s timeout and never blocks routing on failure. Good pattern.
- Note: Alert rules query SQLite (last 100 requests, hourly costs). Under high traffic this could add latency. Cooldown mitigates but doesn't eliminate.

## 2026-04-03 | KN-155 | qa, code-reviewer

- Warning (fixed): `walk.* through` regex was overbroad, matched unrelated text. Fixed to `walk\s+(?:me\s+|us\s+)?through`.
- Warning (fixed): `trade.?off` pattern didn't match plural "trade-offs". Fixed to `trade.?offs?`.
- Warning (fixed): REASONING boost +2 couldn't override code_score=2 ties. Increased to +3.
- Suggestion: No unit tests for new patterns (_COMPLEX_VERB_PATTERN, REASONING boost, medium_signal_count paths). Covered by eval but not by targeted tests.
- Note: complex/factual (33%) and medium/factual (36%) remain weak. Category misclassification is the primary remaining issue (explain/describe overlap with reasoning/creative). Hard to fix with pure rules.
- Note: "design" now in _COMPLEX_VERB_PATTERN matches broadly. "Design a marketplace notification system" -> complex/creative instead of complex/code because "design" is in _CREATIVE_KEYWORDS.
