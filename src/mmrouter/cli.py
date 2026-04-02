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


@cli.command()
@click.argument("prompt")
@click.pass_context
def classify(ctx, prompt):
    """Classify a prompt without routing (debug)."""
    from mmrouter.classifier.rules import RuleClassifier

    classifier = RuleClassifier()
    result = classifier.classify(prompt)

    click.echo(json.dumps({
        "complexity": result.complexity.value,
        "category": result.category.value,
        "confidence": result.confidence,
    }, indent=2))


@cli.command()
@click.option("--db", default="mmrouter.db", help="Path to tracker database.")
@click.option("--json-output", "as_json", is_flag=True, help="Output as JSON.")
def stats(db, as_json):
    """Show routing stats from tracker database."""
    from mmrouter.tracker.logger import Tracker

    try:
        tracker = Tracker(db)
        data = tracker.get_stats()
        tracker.close()
    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)

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


@cli.command(name="eval")
@click.option(
    "--dataset",
    default="eval_data/test_set.yaml",
    show_default=True,
    help="Path to labeled eval dataset YAML.",
)
def eval_cmd(dataset):
    """Run classifier accuracy evaluation against labeled dataset."""
    from mmrouter.classifier.rules import RuleClassifier
    from mmrouter.eval.evaluate import load_eval_set, run_eval

    try:
        eval_set = load_eval_set(dataset)
    except (FileNotFoundError, ValueError) as e:
        click.secho(f"Error loading dataset: {e}", fg="red", err=True)
        sys.exit(1)

    click.echo(f"Loaded {len(eval_set)} cases from {dataset}")

    classifier = RuleClassifier()
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


if __name__ == "__main__":
    cli()
