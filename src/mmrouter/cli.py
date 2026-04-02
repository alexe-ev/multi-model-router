"""CLI entry point for mmrouter."""

import json
import sys

import click

from mmrouter import __version__


@click.group()
@click.version_option(__version__, prog_name="mmrouter")
@click.option("--config", "-c", default="configs/default.yaml", help="Path to routing config YAML.")
@click.pass_context
def cli(ctx, config):
    """mmrouter: intelligent LLM request routing."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = config


@cli.command()
@click.argument("prompt")
@click.option("--verbose", "-v", is_flag=True, help="Show full routing metadata.")
@click.option("--db", default="mmrouter.db", help="Path to tracker database.")
@click.pass_context
def route(ctx, prompt, verbose, db):
    """Classify a prompt and route to the optimal model."""
    from mmrouter.router.engine import Router

    try:
        router = Router(ctx.obj["config"], db_path=db)
        result = router.route(prompt)
    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)

    click.echo(result.completion.content)
    click.echo()

    cl = result.classification
    click.secho(
        f"[{cl.complexity}/{cl.category}] "
        f"model={result.model_used} "
        f"cost=${result.completion.cost:.6f} "
        f"latency={result.completion.latency_ms:.0f}ms "
        f"tokens={result.completion.tokens_in}+{result.completion.tokens_out}",
        fg="bright_black",
    )

    if result.fallback_used:
        click.secho("  (fallback model used)", fg="yellow")

    if verbose:
        click.echo()
        click.secho("Classification:", fg="bright_black")
        click.secho(f"  complexity: {cl.complexity}", fg="bright_black")
        click.secho(f"  category:   {cl.category}", fg="bright_black")
        click.secho(f"  confidence: {cl.confidence}", fg="bright_black")

    router.close()


def _make_classifier(name: str, config_path: str = "configs/default.yaml"):
    """Factory: return the classifier instance for the given name."""
    if name == "rules":
        from mmrouter.classifier.rules import RuleClassifier
        return RuleClassifier()
    if name == "embeddings":
        try:
            from mmrouter.classifier.embeddings import EmbeddingClassifier
        except ImportError as e:
            click.secho(f"Error: {e}", fg="red", err=True)
            sys.exit(1)
        return EmbeddingClassifier()
    if name == "llm":
        try:
            from mmrouter.classifier.llm_classifier import LLMClassifier
            from mmrouter.providers.litellm_provider import LiteLLMProvider
            from mmrouter.router.config import load_config
        except ImportError as e:
            click.secho(f"Error: {e}", fg="red", err=True)
            sys.exit(1)
        try:
            cfg = load_config(config_path)
            model = cfg.classifier.model or "claude-haiku-4-5-20251001"
        except Exception:
            model = "claude-haiku-4-5-20251001"
        provider = LiteLLMProvider()
        return LLMClassifier(provider, model=model)
    click.secho(f"Error: unknown classifier '{name}'", fg="red", err=True)
    sys.exit(1)


@cli.command()
@click.argument("prompt")
@click.option(
    "--classifier",
    "classifier_name",
    type=click.Choice(["rules", "embeddings", "llm"]),
    default="rules",
    show_default=True,
    help="Classifier to use.",
)
@click.pass_context
def classify(ctx, prompt, classifier_name):
    """Classify a prompt without routing (debug)."""
    classifier = _make_classifier(classifier_name, ctx.obj["config"])
    result = classifier.classify(prompt)

    click.echo(json.dumps({
        "complexity": result.complexity.value,
        "category": result.category.value,
        "confidence": result.confidence,
    }, indent=2))


@cli.command()
@click.option("--db", default="mmrouter.db", help="Path to tracker database.")
@click.option("--json-output", "as_json", is_flag=True, help="Output as JSON.")
@click.option("--detailed", is_flag=True, help="Show detailed analytics.")
def stats(db, as_json, detailed):
    """Show routing stats from tracker database."""
    from mmrouter.tracker.logger import Tracker

    try:
        tracker = Tracker(db)
        data = tracker.get_stats()
    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)

    if detailed:
        from mmrouter.tracker.analytics import CostAnalytics

        analytics = CostAnalytics(tracker.connection)
        data["daily_costs"] = analytics.daily_costs()
        data["savings"] = analytics.savings_vs_baseline()
        data["distribution"] = analytics.distribution()

    tracker.close()

    if as_json:
        click.echo(json.dumps(data, indent=2))
        return

    click.secho("mmrouter stats", bold=True)
    click.echo(f"  Requests:      {data['total_requests']}")
    click.echo(f"  Total cost:    ${data['total_cost']:.6f}")
    click.echo(f"  Avg latency:   {data['avg_latency_ms']:.0f}ms")
    click.echo(f"  Tokens in/out: {data['total_tokens_in']}/{data['total_tokens_out']}")
    click.echo(f"  Fallbacks:     {data['fallback_count']}")

    if data["model_distribution"]:
        click.echo()
        click.secho("Model distribution:", bold=True)
        for model, info in data["model_distribution"].items():
            click.echo(f"  {model}: {info['count']} requests (${info['cost']:.6f})")

    if detailed:
        daily = data["daily_costs"]
        if daily:
            click.echo()
            click.secho("Daily costs:", bold=True)
            for entry in daily:
                click.echo(
                    f"  {entry['date']}  {entry['model']}: "
                    f"{entry['request_count']} requests, ${entry['total_cost']:.6f}"
                )

        savings = data["savings"]
        click.echo()
        click.secho("Savings vs baseline (claude-sonnet-4-6):", bold=True)
        click.echo(f"  Actual cost:   ${savings['actual_cost']:.6f}")
        click.echo(f"  Baseline cost: ${savings['baseline_cost']:.6f}")
        click.echo(f"  Savings:       ${savings['savings']:.6f} ({savings['savings_pct']:.1f}%)")

        dist = data["distribution"]
        if dist["by_complexity"]:
            click.echo()
            click.secho("By complexity:", bold=True)
            for key, info in dist["by_complexity"].items():
                click.echo(f"  {key}: {info['count']} requests, ${info['cost']:.6f}")
        if dist["by_category"]:
            click.echo()
            click.secho("By category:", bold=True)
            for key, info in dist["by_category"].items():
                click.echo(f"  {key}: {info['count']} requests, ${info['cost']:.6f}")


@cli.command(name="eval")
@click.option(
    "--dataset",
    default="eval_data/test_set.yaml",
    show_default=True,
    help="Path to labeled eval dataset YAML.",
)
@click.option(
    "--classifier",
    "classifier_name",
    type=click.Choice(["rules", "embeddings", "llm"]),
    default="rules",
    show_default=True,
    help="Classifier to use.",
)
def eval_cmd(dataset, classifier_name):
    """Run classifier accuracy evaluation against labeled dataset."""
    from mmrouter.eval.evaluate import load_eval_set, run_eval

    try:
        eval_set = load_eval_set(dataset)
    except (FileNotFoundError, ValueError) as e:
        click.secho(f"Error loading dataset: {e}", fg="red", err=True)
        sys.exit(1)

    click.echo(f"Loaded {len(eval_set)} cases from {dataset}")

    classifier = _make_classifier(classifier_name, "configs/default.yaml")
    report = run_eval(classifier, eval_set)

    click.echo()
    click.secho("Accuracy summary", bold=True)
    click.echo(f"  Overall:    {report.overall_accuracy:.1%}  ({report.correct}/{report.total})")
    click.echo(f"  Complexity: {report.complexity_accuracy:.1%}")
    click.echo(f"  Category:   {report.category_accuracy:.1%}")

    click.echo()
    click.secho("Per-class accuracy", bold=True)
    # Print as aligned table
    headers = ("bucket", "accuracy")
    col_w = max(len(k) for k in report.per_class_accuracy) if report.per_class_accuracy else 20
    col_w = max(col_w, len(headers[0]))
    click.echo(f"  {'bucket':<{col_w}}  accuracy")
    click.echo(f"  {'-' * col_w}  --------")
    for key in sorted(report.per_class_accuracy):
        acc = report.per_class_accuracy[key]
        color = "green" if acc >= 0.8 else ("yellow" if acc >= 0.5 else "red")
        click.echo(f"  {key:<{col_w}}  ", nl=False)
        click.secho(f"{acc:.1%}", fg=color)

    if report.mismatches:
        click.echo()
        shown = report.mismatches[:20]
        click.secho(
            f"Misclassified examples ({len(report.mismatches)} total, showing {len(shown)})",
            bold=True,
        )
        for m in shown:
            prompt_preview = m.prompt if len(m.prompt) <= 80 else m.prompt[:77] + "..."
            expected = f"{m.expected_complexity}/{m.expected_category}"
            got = f"{m.got_complexity}/{m.got_category}"
            click.echo(f"  expected={expected} got={got}")
            click.secho(f"    {prompt_preview}", fg="bright_black")
    else:
        click.echo()
        click.secho("No misclassifications.", fg="green")


@cli.command(name="compare")
@click.option(
    "--dataset",
    default="eval_data/test_set.yaml",
    show_default=True,
    help="Path to labeled eval dataset YAML.",
)
def compare_cmd(dataset):
    """Compare all available classifiers on eval dataset."""
    from mmrouter.eval.compare import run_comparison
    from mmrouter.eval.evaluate import load_eval_set

    try:
        eval_set = load_eval_set(dataset)
    except (FileNotFoundError, ValueError) as e:
        click.secho(f"Error loading dataset: {e}", fg="red", err=True)
        sys.exit(1)

    click.echo(f"Loaded {len(eval_set)} cases from {dataset}")
    click.echo()

    classifiers = {}

    from mmrouter.classifier.rules import RuleClassifier
    classifiers["rules"] = RuleClassifier()

    try:
        from mmrouter.classifier.embeddings import EmbeddingClassifier
        classifiers["embeddings"] = EmbeddingClassifier()
    except ImportError:
        click.secho("  embeddings: skipped (sentence-transformers not installed)", fg="bright_black")

    click.secho("  llm: skipped (requires API key, too slow for automated comparison)", fg="bright_black")

    if not classifiers:
        click.secho("No classifiers available.", fg="red", err=True)
        sys.exit(1)

    results = run_comparison(eval_set, classifiers)

    col_name = max(len(r.name) for r in results)
    col_name = max(col_name, len("Classifier"))

    header = f"{'Classifier':<{col_name}}  {'Overall':>7}  {'Complexity':>10}  {'Category':>8}  {'Time':>6}"
    sep = f"{'-' * col_name}  {'-' * 7}  {'-' * 10}  {'-' * 8}  {'-' * 6}"
    click.echo(header)
    click.echo(sep)

    for r in results:
        rep = r.report
        row = (
            f"{r.name:<{col_name}}"
            f"  {rep.overall_accuracy:>7.1%}"
            f"  {rep.complexity_accuracy:>10.1%}"
            f"  {rep.category_accuracy:>8.1%}"
            f"  {r.elapsed_seconds:>5.2f}s"
        )
        click.echo(row)


@cli.command()
@click.option("--dataset", default="eval_data/test_set.yaml", show_default=True)
@click.option("--judge-model", default="claude-sonnet-4-6", show_default=True)
@click.option("--baseline-model", default="claude-sonnet-4-6", show_default=True)
@click.option("--sample", default=10, type=int, show_default=True, help="Number of prompts to evaluate.")
@click.option("--db", default="mmrouter.db")
@click.pass_context
def quality(ctx, dataset, judge_model, baseline_model, sample, db):
    """Evaluate response quality using LLM-as-judge."""
    import random
    from mmrouter.eval.evaluate import load_eval_set
    from mmrouter.eval.quality import compare_quality
    from mmrouter.providers.litellm_provider import LiteLLMProvider, ProviderError
    from mmrouter.router.engine import Router

    try:
        eval_set = load_eval_set(dataset)
    except (FileNotFoundError, ValueError) as e:
        click.secho(f"Error loading dataset: {e}", fg="red", err=True)
        sys.exit(1)

    if sample < len(eval_set):
        eval_set = random.sample(eval_set, sample)

    click.echo(f"Evaluating {len(eval_set)} prompts...")
    click.echo(f"  Judge: {judge_model}")
    click.echo(f"  Baseline: {baseline_model}")
    click.echo()

    try:
        router = Router(ctx.obj["config"], db_path=db)
        provider = LiteLLMProvider()
    except Exception as e:
        click.secho(f"Error initializing: {e}", fg="red", err=True)
        sys.exit(1)

    router_responses = []
    baseline_responses = []

    for case in eval_set:
        try:
            routed = router.route(case.prompt)
            router_responses.append((case.prompt, routed.completion.content))
        except (ProviderError, RuntimeError) as e:
            click.secho(f"  Router error: {e}", fg="yellow", err=True)
            continue

        try:
            baseline = provider.complete(case.prompt, baseline_model)
            baseline_responses.append((case.prompt, baseline.content))
        except ProviderError as e:
            click.secho(f"  Baseline error: {e}", fg="yellow", err=True)
            router_responses.pop()  # keep lists aligned
            continue

    if not router_responses:
        click.secho("No successful responses to evaluate.", fg="red", err=True)
        sys.exit(1)

    click.echo(f"Got {len(router_responses)} response pairs. Judging...")

    try:
        result = compare_quality(provider, judge_model, router_responses, baseline_responses)
    except ProviderError as e:
        click.secho(f"Judge error: {e}", fg="red", err=True)
        sys.exit(1)

    rr = result["router"]
    br = result["baseline"]

    click.echo()
    click.secho("Quality comparison", bold=True)
    click.echo(f"  {'Dimension':<15} {'Routed':>8} {'Baseline':>8} {'Delta':>8}")
    click.echo(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8}")

    for dim in ("score", "relevance", "accuracy", "completeness"):
        rv = getattr(rr, f"avg_{dim}")
        bv = getattr(br, f"avg_{dim}")
        d = round(rv - bv, 2)
        color = "green" if d >= 0 else "red"
        click.echo(f"  {dim:<15} {rv:>8.2f} {bv:>8.2f} ", nl=False)
        click.secho(f"{d:>+8.2f}", fg=color)

    click.echo()
    delta_pct = result["delta_pct"]
    color = "green" if delta_pct >= 0 else "red"
    click.echo("  Overall delta: ", nl=False)
    click.secho(f"{delta_pct:+.1f}%", fg=color)

    router.close()


@cli.command()
@click.option("--db", default="mmrouter.db", help="Path to tracker database.")
@click.option("--port", default=8000, type=int, help="Port to serve on.")
@click.option("--host", default="0.0.0.0", help="Host to bind to.")
def dashboard(db, port, host):
    """Start the dashboard web server."""
    try:
        import uvicorn
        from mmrouter.dashboard.app import create_app
    except ImportError:
        click.secho(
            "Dashboard dependencies not installed. Run: pip install mmrouter[dashboard]",
            fg="red", err=True,
        )
        sys.exit(1)

    app = create_app(db)
    click.echo(f"Starting dashboard at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    cli()
