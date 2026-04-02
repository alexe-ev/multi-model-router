# Multi-Model Router: Implementation Plan

## 1. Positioning

### Что это
Open-source Python-пакет для интеллектуальной маршрутизации LLM-запросов. Classifier анализирует входящий запрос, определяет сложность и тип, направляет к оптимальной модели. На выходе: экономия 40-70% на LLM-расходах без деградации качества.

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

## 2. Monetization

### Модель: Open-core + usage-based cloud

Паттерн, который работает в AI-тулинге 2025-2026: Langfuse, Portkey, PostHog, LiteLLM. Open-source для adoption, cloud для revenue.

### Tier structure

| Tier | Что входит | Цена |
|------|-----------|------|
| **Self-hosted (free forever)** | Полный функционал: все классификаторы, routing engine, cost tracker, dashboard. Без ограничений по запросам. | $0 |
| **Cloud Free** | Hosted version. До 25K routed requests/мес. 7-дневное хранение логов. 1 routing config. | $0 |
| **Cloud Pro** | До 500K req/мес. 90-дневные логи. Unlimited configs. Email alerts. API access к analytics. | $39/мес + $8 per 100K overage |
| **Cloud Team** | 2M req/мес. 1-year retention. Team access (5 seats). Slack alerts. Priority support. Custom evaluation rubrics. | $149/мес + $6 per 100K overage |
| **Enterprise** | Unlimited. VPC/on-prem deployment. Custom classifier training. SLA. SSO/SAML. Dedicated support. | Custom ($2K+/мес) |

### Почему эта модель

- **Не per-request pricing** (как Martian $0.004/req). Создает friction и воспринимается как "скрытый налог". Usage-based с generous free tier конвертирует лучше.
- **Не markup на модели** (как OpenRouter 5%). Мы не прокси, мы routing intelligence. Модели оплачиваются напрямую провайдерам.
- **Free tier достаточно для прототипа и small-scale.** Переход на paid происходит естественно при росте трафика.

### Revenue math (conservative)

При 200 платящих клиентах (реалистично через 12-18 мес после launch):
- 120 на Pro ($39) = $4,680/мес
- 60 на Team ($149) = $8,940/мес  
- 15 на Enterprise ($3K avg) = $45,000/мес
- Overage: ~$5,000/мес
- **Total: ~$63K/мес = ~$760K ARR**

Для сравнения: Requesty достиг $1.5M ARR с 25K developers. OpenRouter: $5M ARR.

### Что НЕ монетизируем

- Core routing engine (всегда open-source)
- Число моделей/провайдеров
- Число стратегий классификации
- Dashboard (basic version)

### Что монетизируем (cloud-only features)

- Hosted infrastructure (не нужно ставить и поддерживать самому)
- Длинное хранение логов (90 дней+ vs 7)
- Team collaboration (shared configs, role-based access)
- Advanced alerting (Slack, PagerDuty, webhooks)
- Custom classifier training на своих данных
- A/B testing layer для routing strategies
- Enterprise compliance (SOC2, audit logs, VPC)

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
- [ ] **README с business case.** "При N запросов/день routing экономит $X/мес" с реальными ценами.
- [ ] **Сравнение стратегий.** Таблица: три classifier'а по accuracy, latency, cost. Какой лучше для какого домена.

**Deliverable:** Полноценный open-source пакет с dashboard. Можно написать пост и положить на GitHub.

**Что показать:** Dashboard с реальными данными. "Classifier accuracy 87%. Cost savings 52%. Quality delta 2.3%."

---

### Phase 3: V2 (опционально, 10-14 дней)

**Цель:** Features для монетизации. Cascade routing, adaptive learning, multi-provider, budget mode.

**Scope:**

- [ ] **Cascade routing.** Cheap model first, escalate if low confidence. Measure cost savings vs quality trade-off.
- [ ] **Adaptive routing.** Система учится на user feedback (thumbs up/down), корректирует routing weights.
- [ ] **Multi-provider support.** Routing между Anthropic, OpenAI, Google с automatic failover при outages.
- [ ] **Budget mode.** "Max quality within $X/day." Динамическое переключение моделей при приближении к бюджету.
- [ ] **A/B testing layer.** Сравнение двух routing strategies на live трафике. Statistical significance.
- [ ] **Alerting.** Slack/webhook уведомления при cost spike, quality drop, high error rate.
- [ ] **REST API endpoint.** Deploy как microservice. OpenAI-compatible API (drop-in replacement).
- [ ] **Custom classifier training.** CLI для обучения embedding-classifier на своих данных.
- [ ] **Prompt caching integration.** Автоматический prompt caching для повторяющихся system prompts.

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
- [x] Rule-based classifier accuracy >= 65% на eval set (67%)
- [ ] Cost savings >= 30% vs single-model baseline (measured)
- [x] Fallback работает при недоступности primary model

### V1
- [x] Три стратегии классификации реализованы и сравнены
- [ ] Best classifier accuracy >= 85% (embedding 78%, need LLM eval with API key)
- [ ] Cost savings >= 40%
- [ ] Quality delta < 5% vs single-model
- [ ] Dashboard показывает: distribution, costs, savings, quality, errors
- [ ] README с business case и реальными цифрами

### V2
- [ ] Cascade routing дает дополнительную экономию vs базовый routing
- [ ] Budget mode удерживает расходы в пределах заданного лимита
- [ ] REST API совместим с OpenAI client (drop-in)
- [ ] 10+ GitHub stars (первый сигнал adoption)

---

## 7. Resolved Questions

1. **Package name.** Рабочее имя: `mmrouter`. Переименуем при публикации, когда подберем имя + домен.
2. **LiteLLM.** Pin safe version + обертка в свой provider interface. Не vendoring, но изоляция через абстракцию. При необходимости заменяем на direct API calls.
3. **Dashboard.** React SPA (Vite + Recharts). Отдельный build step, но production-grade.
4. **Eval dataset.** General для начала. Domain-specific добавим по мере необходимости.
5. **Launch strategy.** Закрытая бета сначала. Публичный launch позже.
