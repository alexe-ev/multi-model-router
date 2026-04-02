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


if __name__ == "__main__":
    cli()
