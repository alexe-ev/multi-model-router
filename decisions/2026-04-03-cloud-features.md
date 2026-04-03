## Decision: Cloud-only features prioritize network-effect over operational pain

## Context
Self-hosted mmrouter прост (SQLite + один сервис). Конверсия через "operational pain" не сработает (в отличие от PostHog/Sentry с ClickHouse/Kafka). Нужны features невозможные в self-hosted.

## Alternatives considered

1. **Operational convenience** (hosting, logs, backups). Weak: наш stack слишком простой.
2. **Scale-gated features** (SSO, RBAC, compliance). Works но скучно, generic.
3. **Network-effect features** (benchmarks, cross-client classifiers). Impossible in self-hosted, creates moat.

## Reasoning
Приоритет network-effect features потому что:
- Создают flywheel: больше users -> больше data -> лучше features -> больше users
- Невозможно replicate в self-hosted (нет агрегированных данных)
- Grafana доказала это с Adaptive Metrics

Top 5 cloud-only features (в порядке приоритета):
1. Cross-client benchmarks ("вы экономите 42%, медиана 37%")
2. Auto-tuning (continuous routing config optimization)
3. Pre-trained industry classifiers (training data = moat)
4. Compliance package (SOC2, audit logs, data residency)
5. Collaborative config management (PR-like workflow с impact simulation)

Первые два (benchmarks + auto-tuning) создают flywheel и строятся первыми.

## Trade-offs accepted
- Нужна critical mass users для benchmarks (chicken-and-egg)
- Auto-tuning требует ML infra
- Industry classifiers нужны domain-specific данные которых пока нет
