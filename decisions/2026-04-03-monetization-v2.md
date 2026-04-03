## Decision: Hybrid monetization (markup for proxy + % of savings for BYOK)

## Context
QA-сессия показала что flat subscription не работает: маленькие переплачивают относительно экономии, большие недоплачивают. Предыдущее решение (2026-04-02-monetization-model.md) с тиерами $39/$149/custom пересмотрено.

## Alternatives considered

1. **Flat subscription** ($39-149/mo). Простой биллинг, но misaligned с value: экономия $15/мес при подписке $39 не имеет смысла.
2. **% markup на pass-through** (5% как OpenRouter/Requesty). Работает для proxy, но не захватывает value от routing intelligence.
3. **% от экономии** (15-20% как Spot.io). Aligned с value, но counterfactual проблема и не работает для маленьких.
4. **Hybrid: markup для proxy + % от экономии для BYOK**. Каждый сегмент платит по модели которая имеет смысл для него.

## Reasoning
Hybrid закрывает оба конца рынка:
- Маленькие ($100-500/мес LLM): managed proxy, 5% markup = $5-25/мес. Модель доказана Requesty (15K+ devs, $3M seed).
- Средние/крупные ($5K+/мес): BYOK cloud, 15-20% от экономии. $5K spend, 35% savings = $1750 saved, нам $260-350. Revenue растёт с качеством классификатора.
- Self-hosted: бесплатно навсегда. Distribution engine.

Counterfactual решаем сравнением с baseline "все запросы на самую дорогую модель в конфиге". Прозрачно, логируется в dashboard.

## Trade-offs accepted
- Сложнее биллинг (два типа pricing)
- Counterfactual measurement может вызывать споры
- Нужна инфраструктура для proxy mode (API key management, billing)
