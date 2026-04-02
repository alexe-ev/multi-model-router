## Decision: Positioning as integrated open-source routing toolkit (not proxy/gateway)

## Context
Рынок LLM routing в начале 2026: RouteLLM (binary, research-grade), LiteLLM (proxy), Martian/Not Diamond (commercial, opaque), Portkey (gateway). Нужно определить, чем наш продукт отличается.

## Alternatives considered
1. **Еще один LLM proxy/gateway** (как LiteLLM, Portkey). Crowded market, LiteLLM уже доминирует.
2. **Коммерческий routing service** (как Martian). Нужен ML-research, funding, infrastructure. Не наш масштаб.
3. **Integrated routing toolkit** (classification + routing + tracking + dashboard). Ни одного open-source решения в этой нише.

## Reasoning
Исследование показало пять незакрытых gaps:
1. Ни одно OSS решение не дает multi-tier routing (RouteLLM = binary only).
2. PM-editable YAML config не существует ни у кого.
3. Три сравниваемых стратегии классификации: уникально.
4. Integrated observability (classifier + router + tracker + dashboard) в одном пакете: не существует в OSS.
5. Timing: LiteLLM supply chain incident (март 2026) заставляет команды искать альтернативы.

## Trade-offs accepted
- Scope шире, чем "just a router": classifier + tracker + dashboard. Больше работы на MVP. Accepted: именно bundling создает дифференциацию.
- Зависимость от LiteLLM для provider layer. Accepted: обернем в interface, при необходимости заменим.
