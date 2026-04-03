"""CLI entry point for mmrouter."""

import json
import sys
from pathlib import Path

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

    if result.cascade_used:
        click.secho(f"  (cascade: tried {result.cascade_attempts} models)", fg="cyan")

    if result.budget_downgraded:
        click.secho("  (budget: model downgraded due to spend limit)", fg="yellow")

    if result.experiment_id is not None:
        click.secho(
            f"  (experiment: id={result.experiment_id} variant={result.variant})",
            fg="magenta",
        )

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
        # Check if config specifies a trained model path
        try:
            from mmrouter.router.config import load_config
            cfg = load_config(config_path)
            trained_path = cfg.classifier.trained_model
        except Exception:
            trained_path = None
        if trained_path and Path(trained_path).exists():
            return EmbeddingClassifier.load(trained_path)
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
        from mmrouter.models import BudgetConfig
        from mmrouter.router.budget import BudgetManager

        analytics = CostAnalytics(tracker.connection)
        data["daily_costs"] = analytics.daily_costs()
        data["savings"] = analytics.savings_vs_baseline()
        data["distribution"] = analytics.distribution()
        data["cascade"] = analytics.cascade_savings()

        # Budget status (uses default config if available)
        budget_mgr = BudgetManager(BudgetConfig(), tracker.connection)
        try:
            from mmrouter.router.config import load_config
            cfg = load_config("configs/default.yaml")
            budget_mgr = BudgetManager(cfg.budget, tracker.connection)
        except Exception:
            pass
        data["budget"] = budget_mgr.get_status()

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

        cascade = data["cascade"]
        if cascade["cascade_requests"] > 0:
            click.echo()
            click.secho("Cascade routing:", bold=True)
            click.echo(f"  Cascade requests: {cascade['cascade_requests']}")
            click.echo(f"  Total cost:       ${cascade['cascade_actual_cost']:.6f}")
            click.echo(f"  Total attempts:   {cascade['cascade_attempts_total']}")
            click.echo(f"  Avg attempts:     {cascade['avg_attempts']:.1f}")

        budget = data.get("budget", {})
        if budget.get("enabled"):
            click.echo()
            click.secho("Budget:", bold=True)
            click.echo(f"  Daily limit:  ${budget['daily_limit']:.2f}")
            click.echo(f"  Spent today:  ${budget['spent_today']:.6f}")
            click.echo(f"  Remaining:    ${budget['remaining']:.6f}")
            click.echo(f"  Usage:        {budget['usage_pct']:.1f}%")
            tier_color = {
                "normal": "green",
                "warn": "yellow",
                "downgrade": "red",
                "hard_limit": "red",
            }.get(budget["tier"], "white")
            click.echo("  Tier:         ", nl=False)
            click.secho(budget["tier"], fg=tier_color)


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


@cli.command()
@click.option(
    "--data",
    required=True,
    type=click.Path(exists=True),
    help="Path to labeled YAML training data.",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(),
    help="Output directory for trained model.",
)
@click.option("--k", default=5, type=int, show_default=True, help="Number of neighbors for kNN.")
@click.option(
    "--model-name",
    default="all-MiniLM-L6-v2",
    show_default=True,
    help="Sentence-transformers model name.",
)
@click.option(
    "--eval-split",
    default=0.0,
    type=float,
    help="Fraction of data to hold out for evaluation (0.0 to 0.5).",
)
def train(data, output, k, model_name, eval_split):
    """Train an embedding classifier from labeled YAML data."""
    import random

    try:
        from mmrouter.classifier.embeddings import EmbeddingClassifier
    except ImportError as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)

    if eval_split < 0 or eval_split > 0.5:
        click.secho("Error: --eval-split must be between 0.0 and 0.5", fg="red", err=True)
        sys.exit(1)

    if eval_split > 0:
        import yaml as _yaml
        from mmrouter.eval.evaluate import EvalCase, run_eval

        with open(data) as f:
            raw = _yaml.safe_load(f)

        if not isinstance(raw, list) or len(raw) < 2:
            click.secho("Error: need at least 2 examples for eval split", fg="red", err=True)
            sys.exit(1)

        shuffled = list(raw)
        random.shuffle(shuffled)
        split_idx = max(1, int(len(shuffled) * eval_split))
        eval_entries = shuffled[:split_idx]
        train_entries = shuffled[split_idx:]

        if len(train_entries) == 0:
            click.secho("Error: no training examples after split", fg="red", err=True)
            sys.exit(1)

        # Write temp training file
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            _yaml.dump(train_entries, tmp)
            train_path = tmp.name

        click.echo(f"Split: {len(train_entries)} train, {len(eval_entries)} eval")

        try:
            clf = EmbeddingClassifier(
                model_name=model_name, examples_path=train_path, k=k,
            )
        except (ValueError, FileNotFoundError) as e:
            click.secho(f"Error: {e}", fg="red", err=True)
            sys.exit(1)
        finally:
            Path(train_path).unlink(missing_ok=True)

        # Evaluate on held-out data
        eval_cases = []
        for entry in eval_entries:
            try:
                eval_cases.append(
                    EvalCase(
                        prompt=entry["prompt"],
                        expected_complexity=entry["complexity"],
                        expected_category=entry["category"],
                    )
                )
            except (KeyError, ValueError):
                continue

        if eval_cases:
            report = run_eval(clf, eval_cases)
            click.echo()
            click.secho("Eval on held-out split:", bold=True)
            click.echo(f"  Accuracy: {report.overall_accuracy:.1%} ({report.correct}/{report.total})")
            click.echo(f"  Complexity: {report.complexity_accuracy:.1%}")
            click.echo(f"  Category: {report.category_accuracy:.1%}")
    else:
        try:
            clf = EmbeddingClassifier(
                model_name=model_name, examples_path=data, k=k,
            )
        except (ValueError, FileNotFoundError) as e:
            click.secho(f"Error: {e}", fg="red", err=True)
            sys.exit(1)

    out_dir = clf.save(output)

    click.echo()
    click.secho("Training complete", bold=True)
    click.echo(f"  Examples: {len(clf._prompts)}")
    click.echo(f"  Model: {model_name}")
    click.echo(f"  k: {k}")
    click.echo(f"  Saved to: {out_dir}")
    click.echo()
    click.echo("To use this model, set in your config YAML:")
    click.echo(f"  classifier:")
    click.echo(f"    strategy: embeddings")
    click.echo(f"    trained_model: {out_dir}")


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
@click.argument("request_id", type=int)
@click.argument("direction", type=click.Choice(["up", "down"]))
@click.option("--db", default="mmrouter.db", help="Path to tracker database.")
def feedback(request_id, direction, db):
    """Submit feedback for a routed request. DIRECTION is 'up' or 'down'."""
    from mmrouter.tracker.logger import Tracker

    rating = 1 if direction == "up" else -1
    try:
        tracker = Tracker(db)
        tracker.submit_feedback(request_id, rating)
        tracker.close()
    except ValueError as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)

    click.secho(f"Feedback submitted: request {request_id} = {direction}", fg="green")


@cli.command()
@click.option("--port", default=8080, type=int, help="Port to serve on.")
@click.option("--host", default="0.0.0.0", help="Host to bind to.")
@click.option("--db", default="mmrouter.db", help="Path to tracker database.")
@click.option("--workers", default=1, type=int, help="Number of uvicorn workers.")
@click.pass_context
def serve(ctx, port, host, db, workers):
    """Start the OpenAI-compatible API server."""
    try:
        import uvicorn
        from mmrouter.server.app import create_app
    except ImportError:
        click.secho(
            "Server dependencies not installed. Run: pip install mmrouter[server]",
            fg="red", err=True,
        )
        sys.exit(1)

    app = create_app(config_path=ctx.obj["config"], db_path=db)
    click.echo(f"Starting mmrouter API at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, workers=workers)


@cli.group()
@click.option("--db", default="mmrouter.db", help="Path to tracker database.")
@click.pass_context
def experiment(ctx, db):
    """Manage A/B testing experiments."""
    ctx.ensure_object(dict)
    ctx.obj["experiment_db"] = db


@experiment.command(name="create")
@click.option("--name", required=True, help="Experiment name.")
@click.option("--control", required=True, type=click.Path(exists=True), help="Path to control config YAML.")
@click.option("--treatment", required=True, type=click.Path(exists=True), help="Path to treatment config YAML.")
@click.option("--split", default=0.5, type=float, help="Fraction of traffic to treatment (0.0-1.0).")
@click.pass_context
def experiment_create(ctx, name, control, treatment, split):
    """Create a new A/B experiment between two routing configs."""
    from mmrouter.experiments.store import ExperimentStore
    from mmrouter.models import Experiment
    from mmrouter.router.config import load_config
    from mmrouter.tracker.logger import Tracker

    if split < 0.0 or split > 1.0:
        click.secho("Error: --split must be between 0.0 and 1.0", fg="red", err=True)
        sys.exit(1)

    # Validate both configs parse
    try:
        load_config(control)
    except Exception as e:
        click.secho(f"Error loading control config: {e}", fg="red", err=True)
        sys.exit(1)

    try:
        load_config(treatment)
    except Exception as e:
        click.secho(f"Error loading treatment config: {e}", fg="red", err=True)
        sys.exit(1)

    db = ctx.obj["experiment_db"]
    tracker = Tracker(db)
    store = ExperimentStore(tracker.connection)

    try:
        exp = store.create(Experiment(
            name=name,
            control_config=str(Path(control).resolve()),
            treatment_config=str(Path(treatment).resolve()),
            traffic_split=split,
        ))
    except ValueError as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        tracker.close()
        sys.exit(1)

    click.secho(f"Experiment created: id={exp.id} name='{exp.name}'", fg="green")
    click.echo(f"  Control:   {exp.control_config}")
    click.echo(f"  Treatment: {exp.treatment_config}")
    click.echo(f"  Split:     {exp.traffic_split:.0%} treatment")
    tracker.close()


@experiment.command(name="status")
@click.pass_context
def experiment_status(ctx):
    """Show current experiment status."""
    from mmrouter.experiments.store import ExperimentStore
    from mmrouter.tracker.logger import Tracker

    db = ctx.obj["experiment_db"]
    tracker = Tracker(db)
    store = ExperimentStore(tracker.connection)

    experiments = store.list_all()
    if not experiments:
        click.echo("No experiments found.")
        tracker.close()
        return

    for exp in experiments:
        status_color = {
            "active": "green",
            "stopped": "yellow",
            "completed": "cyan",
        }.get(exp.status.value, "white")

        click.echo(f"  [{exp.id}] {exp.name}  ", nl=False)
        click.secho(exp.status.value, fg=status_color, nl=False)
        click.echo(f"  split={exp.traffic_split:.0%}")
        click.echo(f"       control={exp.control_config}")
        click.echo(f"       treatment={exp.treatment_config}")
        click.echo(f"       created={exp.created_at.strftime('%Y-%m-%d %H:%M')}")
        if exp.stopped_at:
            click.echo(f"       stopped={exp.stopped_at.strftime('%Y-%m-%d %H:%M')}")

        # Show request counts per variant
        cur = tracker.connection.execute(
            """SELECT variant, COUNT(*) as cnt, COALESCE(SUM(cost), 0) as cost
               FROM requests WHERE experiment_id = ? GROUP BY variant""",
            (exp.id,),
        )
        rows = cur.fetchall()
        if rows:
            for row in rows:
                v = row["variant"] or "none"
                click.echo(f"       {v}: {row['cnt']} requests, ${row['cost']:.6f}")
        click.echo()

    tracker.close()


@experiment.command(name="stop")
@click.option("--id", "experiment_id", type=int, default=None, help="Experiment ID to stop. Defaults to active.")
@click.pass_context
def experiment_stop(ctx, experiment_id):
    """Stop an active experiment."""
    from mmrouter.experiments.store import ExperimentStore
    from mmrouter.tracker.logger import Tracker

    db = ctx.obj["experiment_db"]
    tracker = Tracker(db)
    store = ExperimentStore(tracker.connection)

    try:
        if experiment_id is not None:
            exp = store.stop(experiment_id)
        else:
            exp = store.stop_active()
            if exp is None:
                click.echo("No active experiment to stop.")
                tracker.close()
                return
    except ValueError as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        tracker.close()
        sys.exit(1)

    click.secho(f"Experiment stopped: id={exp.id} name='{exp.name}'", fg="yellow")
    tracker.close()


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


@cli.group()
@click.pass_context
def alerts(ctx):
    """Manage alerting rules and webhooks."""
    pass


@alerts.command(name="status")
@click.option("--db", default="mmrouter.db", help="Path to tracker database.")
@click.pass_context
def alerts_status(ctx, db):
    """Show active alert rules, cooldown state, and webhook config."""
    from mmrouter.router.engine import Router

    try:
        router = Router(ctx.obj["config"], db_path=db)
        status = router.get_alerts_status()
        router.close()
    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)

    if not status.get("enabled"):
        click.echo("Alerting is disabled. Enable in config YAML:")
        click.echo("  alerts:")
        click.echo("    enabled: true")
        return

    click.secho("Alerting status", bold=True)
    webhook_url = status.get("webhook_url")
    if webhook_url:
        click.echo(f"  Webhook: {webhook_url}")
    else:
        click.echo("  Webhook: not configured")

    click.echo()
    click.secho("Rules:", bold=True)
    for rule in status.get("rules", []):
        state = "COOLDOWN" if rule["in_cooldown"] else "active"
        color = "yellow" if rule["in_cooldown"] else "green"
        click.echo(f"  {rule['name']}  ", nl=False)
        click.secho(state, fg=color, nl=False)
        if rule["in_cooldown"]:
            click.echo(f"  ({rule['cooldown_remaining']}s remaining)", nl=False)
        click.echo(f"  severity={rule['severity']}  cooldown={rule['cooldown_seconds']}s")


@alerts.command(name="test")
@click.option("--webhook-url", required=True, help="Webhook URL to send test alert to.")
def alerts_test(webhook_url):
    """Send a test alert to verify webhook delivery."""
    from mmrouter.alerts.channels import Alert, WebhookChannel

    channel = WebhookChannel(webhook_url)
    test_alert = Alert(
        rule_name="test",
        message="This is a test alert from mmrouter",
        severity="warning",
        details={"test": True},
    )

    click.echo(f"Sending test alert to {webhook_url}...")
    success = channel.send(test_alert)
    if success:
        click.secho("Test alert sent successfully.", fg="green")
    else:
        click.secho("Test alert delivery failed. Check the URL and try again.", fg="red")
        sys.exit(1)


if __name__ == "__main__":
    cli()
