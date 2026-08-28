"""Tests for the dashboard CLI command's bind address (AC15)."""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from mmrouter.cli import cli


def _run(args):
    with patch("uvicorn.run") as run, patch(
        "mmrouter.dashboard.app.create_app", return_value=MagicMock()
    ):
        result = CliRunner().invoke(cli, ["dashboard", *args])
    assert result.exit_code == 0, result.output
    return run


def test_default_host_is_loopback():
    run = _run([])
    assert run.call_args.kwargs["host"] == "127.0.0.1"


def test_host_flag_still_exposes_every_interface():
    run = _run(["--host", "0.0.0.0"])
    assert run.call_args.kwargs["host"] == "0.0.0.0"
