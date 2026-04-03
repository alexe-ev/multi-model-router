## Decision: Position as "LLM cost optimization through intelligent routing"

## Context
QA показала что "routing" описывает mechanism, не outcome. Buyers ищут "cost optimization", не "routing". TAM: $15B LLM API spend к 2026, 50-90% addressable.

## Alternatives considered

1. **"LLM routing"** (как RouteLLM, Martian). Technical category label. Appeals to infra engineers.
2. **"LLM cost optimization"** Lead с outcome. Шире чем routing (caching, compression тоже), но сильнее как value prop.
3. **"Intelligent AI gateway"** (как Portkey). Requires enterprise features мы пока не имеем.
4. **"LLM cost optimization through intelligent routing"** Outcome + mechanism.

## Reasoning
Lead с outcome, qualify с mechanism:
- Buyer's pain = cost, не "я хочу router"
- "Routing" отличает от generic cost optimization advice
- "Zero code changes" это killer feature (замена одной строки base_url)
- Отличает от Portkey (governance), OpenRouter (marketplace)

Три сегмента, три pitch:
- Self-hosted: "Cut LLM costs 40-85%. Open-source."
- Managed proxy: "One API key, we route to the cheapest model that works."
- Cloud BYOK: "Your keys, your data, our routing intelligence."

Threshold: $500-1000/мес LLM spend. Ниже - экономия не оправдывает effort.

## Trade-offs accepted
- "Cost optimization" шире чем наш scope, нужно чётко ограничивать
- 80% команд хватит пары if-ов, наш рынок - 20% с высоким трафиком
