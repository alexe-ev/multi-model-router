## Decision: Hybrid privacy (client-side classifier default, opt-in transient processing)

## Context
mmrouter видит полный текст промптов для классификации. В cloud-версии это blocker для клиентов с sensitive data (PII, internal docs, CRM data).

## Alternatives considered

1. **"Trust us"** (как Portkey). Видим всё, SOC2 certified. Простейшая архитектура.
2. **Never store** (как AWS Bedrock). Transient processing, ничего не хранится. ZDR contracts.
3. **Client-side classification** (как RouteLLM self-hosted). Классификатор на стороне клиента, cloud видит только метаданные.
4. **Hybrid**: client-side по дефолту, opt-in transient для лучшей accuracy.

## Reasoning
Hybrid потому что:
- Privacy by default (client-side) это strongest trust signal
- Classifier уже существует (rules + embeddings), нужно только упаковать
- Opt-in transient позволяет enterprise клиентам получить лучшую accuracy если готовы
- OpenRouter и Azure оба предлагают default-private с opt-in, это proven pattern

Cloud получает: complexity, category, confidence, token count, latency. Этого достаточно для benchmarks, billing, analytics.

## Trade-offs accepted
- Два code path (client-side + transient)
- Client-side accuracy ниже (~78% rules vs потенциально 90%+ с полным текстом)
- Нужен механизм обновления client-side classifier
