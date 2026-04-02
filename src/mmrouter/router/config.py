"""YAML config loader and validator for routing configs."""

from __future__ import annotations

from pathlib import Path

import strictyaml as sy

from mmrouter.models import (
    CascadeConfig,
    Category,
    ClassifierConfig,
    Complexity,
    ModelRoute,
    ProviderConfig,
    RoutingConfig,
)

_VALID_COMPLEXITIES = {c.value for c in Complexity}
_VALID_CATEGORIES = {c.value for c in Category}

_route_schema = sy.Map({
    "model": sy.Str(),
    sy.Optional("fallbacks"): sy.Seq(sy.Str()),
    sy.Optional("cascade"): sy.Seq(sy.Str()),
})

_routes_schema = sy.MapPattern(
    sy.Str(),  # complexity
    sy.MapPattern(sy.Str(), _route_schema),  # category -> route
)

_classifier_schema = sy.Map({
    sy.Optional("strategy"): sy.Str(),
    sy.Optional("model"): sy.Str(),
    sy.Optional("threshold"): sy.Float(),
})

_provider_schema = sy.Map({
    sy.Optional("timeout_ms"): sy.Int(),
    sy.Optional("max_retries"): sy.Int(),
    sy.Optional("circuit_breaker_threshold"): sy.Int(),
    sy.Optional("circuit_breaker_reset_ms"): sy.Int(),
})

_cascade_schema = sy.Map({
    sy.Optional("enabled"): sy.Bool(),
    sy.Optional("strategy"): sy.Str(),
    sy.Optional("min_response_length"): sy.Int(),
    sy.Optional("hedging_phrases"): sy.Seq(sy.Str()),
    sy.Optional("judge_model"): sy.Str(),
    sy.Optional("judge_threshold"): sy.Int(),
})

_config_schema = sy.Map({
    sy.Optional("version"): sy.Str(),
    "routes": _routes_schema,
    sy.Optional("classifier"): _classifier_schema,
    sy.Optional("provider"): _provider_schema,
    sy.Optional("cascade"): _cascade_schema,
})


class ConfigError(Exception):
    """Raised when a routing config is invalid."""


def load_config(path: str | Path) -> RoutingConfig:
    """Load and validate a YAML routing config file."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    raw = path.read_text()
    try:
        parsed = sy.load(raw, _config_schema)
    except sy.YAMLValidationError as e:
        raise ConfigError(f"Invalid config: {e}") from e

    data = parsed.data

    # Validate complexity/category values
    routes: dict[str, dict[str, ModelRoute]] = {}
    for complexity_key, category_map in data["routes"].items():
        if complexity_key not in _VALID_COMPLEXITIES:
            raise ConfigError(
                f"Unknown complexity '{complexity_key}'. "
                f"Valid: {', '.join(sorted(_VALID_COMPLEXITIES))}"
            )
        routes[complexity_key] = {}
        for category_key, route_data in category_map.items():
            if category_key not in _VALID_CATEGORIES:
                raise ConfigError(
                    f"Unknown category '{category_key}' under '{complexity_key}'. "
                    f"Valid: {', '.join(sorted(_VALID_CATEGORIES))}"
                )
            routes[complexity_key][category_key] = ModelRoute(
                model=route_data["model"],
                fallbacks=route_data.get("fallbacks", []),
                cascade=route_data.get("cascade", []),
            )

    classifier = ClassifierConfig()
    if "classifier" in data and data["classifier"]:
        cls_data = data["classifier"]
        classifier = ClassifierConfig(
            strategy=cls_data.get("strategy", "rules"),
            model=cls_data.get("model"),
            threshold=float(cls_data["threshold"]) if "threshold" in cls_data else 0.7,
        )

    provider = ProviderConfig()
    if "provider" in data and data["provider"]:
        prov_data = data["provider"]
        provider = ProviderConfig(**{k: v for k, v in prov_data.items() if v is not None})

    cascade = CascadeConfig()
    if "cascade" in data and data["cascade"]:
        cas_data = data["cascade"]
        cascade_kwargs = {}
        if "enabled" in cas_data:
            cascade_kwargs["enabled"] = cas_data["enabled"]
        if "strategy" in cas_data:
            cascade_kwargs["strategy"] = cas_data["strategy"]
        if "min_response_length" in cas_data:
            cascade_kwargs["min_response_length"] = cas_data["min_response_length"]
        if "hedging_phrases" in cas_data:
            cascade_kwargs["hedging_phrases"] = cas_data["hedging_phrases"]
        if "judge_model" in cas_data:
            cascade_kwargs["judge_model"] = cas_data["judge_model"]
        if "judge_threshold" in cas_data:
            cascade_kwargs["judge_threshold"] = cas_data["judge_threshold"]
        cascade = CascadeConfig(**cascade_kwargs)

    return RoutingConfig(
        version=data.get("version", "1"),
        routes=routes,
        classifier=classifier,
        provider=provider,
        cascade=cascade,
    )
