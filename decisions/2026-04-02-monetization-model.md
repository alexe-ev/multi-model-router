## Decision: Open-core + usage-based cloud pricing

## Context
Нужна модель монетизации для multi-model router. Исследованы все конкуренты: Martian (per-request), OpenRouter (markup), Not Diamond (per-recommendation), Portkey/Langfuse (usage-based cloud).

## Alternatives considered
1. **Per-request pricing** (как Martian, $0.004/req). Создает friction, воспринимается как скрытый налог на каждый API-вызов.
2. **Markup на модели** (как OpenRouter, 5%). Мы не прокси, нет pass-through spend для markup. Неприменимо.
3. **Pure SaaS subscription** (фикс/мес). Не масштабируется с usage. Невыгодно для крупных клиентов.
4. **Open-core + usage-based** (как Langfuse, Portkey, PostHog). Самый распространенный паттерн в AI-тулинге 2025-2026.

## Reasoning
- Open-source core дает adoption без friction. Это моат: community + integrations сложно догнать.
- Usage-based pricing растет с клиентом. Free tier (25K req/мес) достаточен для прототипа и small-scale.
- Монетизируем инфраструктуру и удобство (hosted, retention, alerts, team access), а не core functionality.
- Паттерн проверен: Langfuse ($29-$2499/мес), Portkey ($49+), PostHog (usage-based).

## Trade-offs accepted
- Self-hosted конкурирует с cloud: некоторые крупные клиенты поставят сами и не заплатят. Accepted: enterprise features (SSO, audit, VPC deployment) мотивируют paid tier.
- Revenue начнется не сразу: нужно сначала набрать community. Accepted: проект и так позиционируется как portfolio + learning.
