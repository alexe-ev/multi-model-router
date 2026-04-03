## Decision: MIT license now, add proprietary /ee directory later

## Context
Нужно определить лицензионную стратегию. Код построен за 2 дня, не является competitive moat. Trust критичен для инфраструктуры которая видит весь LLM-трафик.

## Alternatives considered

1. **MIT** (как LiteLLM, Langfuse, Supabase). Максимальный adoption, нулевая protection.
2. **AGPLv3** (как Grafana). OSI-approved, prevents cloud strip-mining. $270M ARR у Grafana.
3. **BSL/SSPL** (как HashiCorp, Elastic). Protection от конкурентов, но backlash и forks (OpenTofu, OpenSearch).
4. **MIT + /ee** (как PostHog). MIT core, proprietary enterprise features. $1.4B valuation.

## Reasoning
MIT сейчас потому что:
- Код тривиален (2 дня), protection через лицензию это театр
- Trust > protection для infra которая видит промпты
- Нулевой friction для adoption
- PostHog, Langfuse, Supabase доказали что MIT + managed service работает

Transition к MIT + /ee когда будут users и понятны enterprise needs (SSO, audit, compliance).

## Trade-offs accepted
- Нулевая protection от копирования (принимаем)
- Revenue позже чем с closed SaaS
- Нужно будет найти features достаточно сильные для /ee конверсии
