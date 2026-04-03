# Multi-Model Router: Implementation Plan

## 1. Positioning (revised 2026-04-03)

### Что это
LLM cost optimization through intelligent routing. Open-source Python-пакет, который анализирует каждый запрос и направляет к оптимальной модели. Экономия 40-85% на LLM-расходах без деградации качества. Zero code changes: замена одной строки base_url.

Details: `decisions/2026-04-03-positioning.md`

### Для кого
- **Self-hosted:** developers, $0. "Cut LLM costs 40-85%. Open-source."
- **Managed proxy:** маленькие команды (<$500/мес LLM). "One API key, we route to the cheapest model that works."
- **Cloud BYOK:** средние/крупные ($5K+/мес). "Your keys, your data, our routing intelligence."

Threshold: $500-1000/мес LLM spend. Ниже - pair of if-statements достаточно.

### Чем отличается от существующих решений

Конкурентный ландшафт (начало 2026):

| Решение | Тип | Проблема |
|---------|-----|----------|
| **RouteLLM** (LMSYS) | Open-source | Только binary routing (strong/weak). Нет multi-tier. Нет training scripts. Нет cost tracking. Нет дашборда. Плохой DX. |
| **LiteLLM** | Open-source | Proxy/load balancer, не intelligent router. Supply chain incident (март 2026). Тяжелый dependency tree. |
| **Martian** | Commercial | Opaque routing logic. $0.004/req. Нет self-hosted. |
| **Not Diamond** | Commercial | Routing отдельно от invocation. $0.001/recommendation + ты отдельно платишь провайдеру. |
| **OpenRouter** | Marketplace | Auto Router powered by Not Diamond. Нет custom rules. 5% markup. |
| **Portkey** | Commercial | Gateway, не intelligent router. Rule-based fallbacks, не quality-based routing. |

**Ни одно open-source решение не объединяет:** multi-tier classification + routing + cost tracking + dashboard в одном пакете.

### Наши дифференциаторы

1. **Multi-tier, не binary.** Три уровня сложности (simple/medium/complex) x четыре категории (factual/reasoning/creative/code). RouteLLM умеет только strong vs weak.

2. **PM-editable YAML config.** Routing rules меняются без кода и деплоя. Ни один конкурент этого не предлагает.

3. **Три стратегии классификации для сравнения.** Rule-based, embedding-based, LLM-as-classifier. Можно измерить accuracy каждой, выбрать лучшую для конкретного домена.

4. **Integrated observability.** Classification + routing + cost logging + quality tracking + dashboard в одном инструменте. Не нужно склеивать 3 сервиса.

5. **Open-source core.** Полный функционал бесплатно при self-hosting. Низкий порог входа.

---

## 2. Monetization (revised 2026-04-03)

### Модель: Hybrid (markup for proxy + % of savings for BYOK)

Пересмотрено после QA-сессии. Flat subscription не работает: маленьким невыгодно, большим мало. Details: `decisions/2026-04-03-monetization-v2.md`.

### Три сегмента

| Сегмент | Продукт | Как платит |
|---------|---------|------------|
| **Developers, хоббисты** | Self-hosted (free forever) | $0. Distribution engine. |
| **Маленькие команды** (<$500/мес LLM) | Managed proxy (наш API ключ) | 5% markup на pass-through. $100 spend = $5 нам. |
| **Средние/крупные** (>$500/мес LLM) | Cloud BYOK (свои ключи) | 15-20% от измеренной экономии. |

### Revenue math

Managed proxy (5% markup):
- 500 users x $200/мес avg spend x 5% = $5,000/мес

Cloud BYOK (% of savings):
- 50 users x $5K/мес spend x 35% savings x 17.5% fee = $15,300/мес
- 10 users x $20K/мес spend x 40% savings x 17.5% fee = $14,000/мес

**Total at scale: ~$34K/мес = ~$410K ARR**

### Что НЕ монетизируем
- Core routing engine (MIT, бесплатно навсегда)
- Все классификаторы, dashboard, CLI
- Self-hosted без ограничений

### Cloud-only features (conversion drivers)
1. Cross-client benchmarks (network effect, невозможно self-hosted)
2. Auto-tuning (continuous routing optimization)
3. Pre-trained industry classifiers (data moat)
4. Compliance package (SOC2, audit logs, data residency)
5. Collaborative config management (PR-like workflow с impact simulation)

Details: `decisions/2026-04-03-cloud-features.md`

---

## 3. Architecture

### Высокоуровневая схема

```
User Request
    |
    v
[Classifier] -- rules.py / embeddings.py / llm_classifier.py
    |
    | {complexity, category, confidence}
    v
[Router Engine] -- config.yaml --> model selection
    |                                    |
    | confidence < threshold?            |
    | --> escalate to stronger model     |
    v                                    v
[Provider Layer] -- LiteLLM --> Anthropic / OpenAI / Google
    |
    | on error: retry -> fallback -> circuit breaker
    v
[Tracker] -- SQLite --> log request
    |
    v
[Dashboard API] -- FastAPI --> React frontend
```

### Ключевые компоненты

```
multi-model-router/
  src/
    mmrouter/                  # Python package name
      __init__.py
      cli.py                   # Click CLI: mmrouter route "prompt"
      api.py                   # Programmatic API: router.route(prompt)
      
      classifier/
        __init__.py            # ClassifierBase ABC
        rules.py               # Rule-based: keywords + length
        embeddings.py          # Embedding-based: sentence-transformers + kNN
        llm_classifier.py      # LLM-as-classifier: zero/few-shot via cheap model
      
      router/
        __init__.py
        engine.py              # Core: classify -> select model -> call -> track
        config.py              # YAML config loader + validator
        fallback.py            # Fallback chain + circuit breaker
      
      providers/
        __init__.py
        base.py                # Provider ABC
        litellm_provider.py    # LiteLLM wrapper (primary)
        direct.py              # Direct API calls (fallback if LiteLLM issues)
      
      tracker/
        __init__.py
        logger.py              # SQLite writer: log every request
        analytics.py           # Aggregations: cost savings, quality, distribution
        models.py              # Pydantic models for request/response log
      
      eval/
        __init__.py
        evaluate.py            # Run eval set, measure classifier accuracy
        compare.py             # Compare routing vs single-model quality + cost
        quality.py             # LLM-as-judge quality scoring
  
  dashboard/
    src/                       # React + Recharts
    
  configs/
    default.yaml               # Default routing config
    budget.yaml                # Budget-constrained config
    
  eval_data/
    test_set.yaml              # 100+ labeled test queries
    
  tests/
    test_classifier/
    test_router/
    test_tracker/
    
  pyproject.toml
  README.md
```

### Tech stack (финальный)

| Компонент | Выбор | Почему, а не альтернатива |
|-----------|-------|---------------------------|
| Language | Python 3.11+ | Экосистема AI/ML. Click для CLI. |
| LLM access | LiteLLM | 100+ providers через единый API. Несмотря на проблемы, альтернатив по coverage нет. Обернем в свой провайдер для изоляции. |
| Embeddings | sentence-transformers (local) | Бесплатно, без API-зависимости. `all-MiniLM-L6-v2` для старта. |
| Storage | SQLite + WAL mode | Zero-config, single file, достаточно до 10M+ строк. |
| Config | YAML (strictyaml) | PM-editable. strictyaml для валидации без yaml.unsafe_load. |
| CLI | Click | Стандарт для Python CLI. |
| Dashboard backend | FastAPI | Async, автодокументация, Pydantic-native. |
| Dashboard frontend | React + Recharts | Спека, достаточно для MVP. Vite для сборки. |
| Tests | pytest | Стандарт. pytest-asyncio для async тестов. |
| Package | pyproject.toml + hatch | Modern Python packaging. |

---

## 4. Phased Delivery

### Phase 1: MVP (5-7 дней)

**Цель:** `mmrouter route "prompt"` работает end-to-end. Классифицирует, роутит, логирует.

**Scope:**

- [x] **Project scaffolding.** pyproject.toml, структура пакета, pytest setup, .gitignore.
- [x] **Rule-based classifier.** Keywords + length heuristics. Возвращает `{complexity, category, confidence}`.
- [x] **YAML config loader.** Парсит routing config, валидирует, возвращает model name по (complexity, category).
- [x] **Router engine.** `router.route(prompt)` : classify -> select model -> call LiteLLM -> return response + metadata.
- [x] **LiteLLM provider wrapper.** Изолирует LiteLLM за интерфейсом. Retry + basic error handling.
- [x] **Fallback chain.** При ошибке primary: пробуем fallback models из конфига.
- [x] **SQLite tracker.** Логирует каждый запрос: model, tokens, cost, latency, classification result.
- [x] **CLI.** `mmrouter route "What is the capital of France?"` -- выводит ответ + routing metadata.
- [x] **Programmatic API.** `from mmrouter import Router; r = Router(); result = r.route("...")`.
- [x] **Default config.** `configs/default.yaml` с маппингом на Claude models.
- [x] **Eval dataset.** 120 queries с ground truth labels в `eval_data/test_set.yaml`.
- [x] **Basic eval script.** Измеряет accuracy rule-based classifier на eval set.
- [x] **Tests.** 122 unit-теста для всех компонентов.

**Deliverable:** Работающий CLI и Python API. `pip install -e .` и поехали.

**Что показать:** "Вот 100 запросов. Rule-based classifier имеет accuracy X%. Routing экономит Y% vs all-Sonnet."

---

### Phase 2: V1 (7-10 дней после MVP)

**Цель:** Три стратегии классификации, dashboard, quality eval. Проект, который можно показать.

**Scope:**

- [x] **Embedding-based classifier.** sentence-transformers kNN, 78% accuracy (146 training examples).
- [x] **LLM-as-classifier.** Zero/few-shot через Haiku via ProviderBase. XML-tag prompt isolation.
- [x] **Classifier comparison.** `mmrouter compare` запускает все стратегии, выводит таблицу.
- [x] **Circuit breaker.** Per-model CLOSED/OPEN/HALF_OPEN. Transient errors trip, permanent don't.
- [x] **Confidence-based routing.** Low confidence -> escalate complexity one tier.
- [x] **Cost tracker analytics.** Daily costs, savings vs Sonnet baseline, model pricing table, CLI --detailed.
- [x] **Quality eval.** LLM-as-judge scoring (1-5 rubric), compare routed vs baseline, CLI `mmrouter quality`.
- [x] **Dashboard backend (FastAPI).** Endpoints: `/api/stats`, `/api/stats/daily`, `/api/stats/distribution`, `/api/requests`, `/api/models`.
- [x] **Dashboard frontend (React).** KPI cards, cost breakdown chart, model distribution pie, request log table with pagination + filters.
- [x] **README с business case.** Cost math table, real pricing, traffic mix scenarios.
- [x] **Сравнение стратегий.** Three classifiers compared: accuracy, cost, speed, use case.

**Deliverable:** Полноценный open-source пакет с dashboard. Можно написать пост и положить на GitHub.

**Что показать:** Dashboard с реальными данными. "Classifier accuracy 87%. Cost savings 52%. Quality delta 2.3%."

---

### Phase 3: V2 (опционально, 10-14 дней)

**Цель:** Features для монетизации. Cascade routing, adaptive learning, multi-provider, budget mode.

**Scope:**

- [x] **Cascade routing.** Cheap model first, escalate if low confidence. Measure cost savings vs quality trade-off.
- [x] **Adaptive routing.** Система учится на user feedback (thumbs up/down), корректирует routing weights.
- [x] **Multi-provider support.** Routing между Anthropic, OpenAI, Google с automatic failover при outages.
- [x] **Budget mode.** "Max quality within $X/day." Динамическое переключение моделей при приближении к бюджету.
- [x] **A/B testing layer.** Сравнение двух routing strategies на live трафике. Statistical significance.
- [x] **Alerting.** Slack/webhook уведомления при cost spike, quality drop, high error rate.
- [x] **REST API endpoint.** Deploy как microservice. OpenAI-compatible API (drop-in replacement).
- [x] **Custom classifier training.** CLI для обучения embedding-classifier на своих данных.
- [x] **Prompt caching integration.** Автоматический prompt caching для повторяющихся system prompts.

**Deliverable:** Production-ready сервис. Основа для cloud-hosted version.

---

## 5. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| LiteLLM dependency (supply chain, stability) | High | Обернуть в свой provider interface. Pin version. Lock file. При необходимости заменить на direct API calls. |
| Classifier accuracy <85% на реальных данных | Medium | Три стратегии для сравнения. Гибрид rules + embeddings. Confidence-based escalation как safety net. |
| Dashboard scope creep | Medium | MVP dashboard: 5 charts + request log. Не больше. Расширять в V2. |
| Pricing моделей изменится | Low | Pricing в отдельной таблице `model_pricing`, обновляется независимо от кода. |
| Конкуренты скопируют дифференциаторы | Medium | Ship fast. Community adoption = moat. YAML config + integrated dashboard сложно скопировать в binary-only RouteLLM. |

---

## 6. Success Criteria

### MVP
- [x] `mmrouter route "prompt"` работает end-to-end
- [x] Rule-based classifier accuracy >= 65% на eval set (78%, improved from 67%)
- [ ] Cost savings >= 30% vs single-model baseline (requires live traffic measurement)
- [x] Fallback работает при недоступности primary model

### V1
- [x] Три стратегии классификации реализованы и сравнены
- [ ] Best classifier accuracy >= 85% (embedding 78%, need LLM eval with API key)
- [ ] Cost savings >= 40% (requires live traffic measurement)
- [ ] Quality delta < 5% vs single-model (requires live traffic measurement)
- [x] Dashboard показывает: distribution, costs, savings, quality, errors
- [x] README с business case и реальными цифрами

### V2
- [x] Cascade routing дает дополнительную экономию vs базовый routing
- [x] Budget mode удерживает расходы в пределах заданного лимита
- [x] REST API совместим с OpenAI client (drop-in)
- [ ] 10+ GitHub stars (первый сигнал adoption)

Note: unchecked items require live traffic with API keys. Cannot be measured in dev environment.

---

## 7. Resolved Questions

1. **Package name.** Рабочее имя: `mmrouter`. Переименуем при публикации, когда подберем имя + домен.
2. **LiteLLM.** Pin safe version + обертка в свой provider interface. Не vendoring, но изоляция через абстракцию. При необходимости заменяем на direct API calls.
3. **Dashboard.** React SPA (Vite + Recharts). Отдельный build step, но production-grade.
4. **Eval dataset.** General для начала. Domain-specific добавим по мере необходимости.
5. **Launch strategy.** Закрытая бета сначала. Публичный launch позже.

---

## 8. Post-Phase 3: Code Quality (completed 2026-04-03)

Security and quality improvements from code review findings:

- [x] **KN-151** Security: constant-time API key comparison (hmac.compare_digest) + stream error wrapping (PR #21)
- [x] **KN-152** Refactor: StreamRouteResult dataclass, Router.passthrough_messages() (PR #22)
- [x] **KN-153** Perf: FeedbackScorer TTL cache for adaptive routing (PR #23)
- [x] **KN-154** Docs: update architecture, README, PLAN.md (PR #24)

---

## 9. Product Quality: Classifier + E2E Testing (completed 2026-04-03)

- [x] **KN-155** Classifier accuracy: 67% -> 78% overall, 78% -> 90% complexity (PR #25)
- [x] E2E product tests: 20 tests with real OpenAI API (routing, streaming, REST API, feedback, cost tracking)
- [x] configs/openai.yaml for OpenAI-based testing

### Stats
- 501 tests passing (481 unit + 20 e2e)
- 26 PRs merged
- All 3 phases + code quality + e2e complete

---

## 10. Strategic Decisions (2026-04-03)

QA-сессия выявила 5 стратегических вопросов. Исследование проведено, решения зафиксированы.

| Вопрос | Решение | Decision file |
|--------|---------|---------------|
| Monetization | Hybrid: 5% markup (proxy) + 15-20% of savings (BYOK) | `decisions/2026-04-03-monetization-v2.md` |
| Licensing | MIT now, add proprietary /ee later | `decisions/2026-04-03-opensource-strategy.md` |
| Privacy | Client-side classifier default, opt-in transient | `decisions/2026-04-03-privacy-architecture.md` |
| Positioning | "LLM cost optimization through intelligent routing" | `decisions/2026-04-03-positioning.md` |
| Cloud features | Network-effect first: benchmarks, auto-tuning | `decisions/2026-04-03-cloud-features.md` |

---

## 11. Cloud Platform Roadmap

### Phase 4: Open-Source Launch (P8) — prerequisite
- MIT LICENSE, PyPI publish, GitHub release v1.0
- Landing page, demo, positioning
- Backlog: KN-148/149 (A/B testing), KN-156 (LLM benchmark), KN-157 (e2e cascade/budget)

### Phase 5: Cloud MVP (P9 + P10, parallel)
- **P9: Auth & Multi-Tenancy**: registration, OAuth, workspace model, API keys, SQLite -> PostgreSQL
- **P10: Cloud Infrastructure**: Docker, CI/CD, hosting (Railway/Fly.io/Render), Sentry, managed PG

### Phase 6: Monetization (P11 + P12, parallel)
- **P11: Managed Proxy**: our provider keys, multi-tenant endpoint, 5% markup, Stripe, free tier
- **P12: Web UI**: personal cabinet, onboarding, analytics dashboard, landing page

### Phase 7: Advanced (P13)
- **P13: BYOK & Savings Billing**: encrypted key storage, savings measurement, % of savings billing, privacy mode

### Dependency chain
P8 -> P9 + P10 (parallel) -> P11 + P12 (parallel) -> P13
