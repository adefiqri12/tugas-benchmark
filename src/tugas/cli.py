"""TUGAS command line: list tasks, run a model, print the score table."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .agent import LLMAgent
from .models import FailureType
from .runner import load_tasks, run_agent
from .score import summarize

app = typer.Typer(add_completion=False, help="Eval harness for Indonesian tool-calling agents.")
console = Console()

DEFAULT_TASKS = "eval/datasets/golden_id.json"


def _version_callback(value: bool) -> None:
    if value:
        console.print(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True,
        help="Show the TUGAS version and exit.",
    ),
) -> None:
    """Eval harness for Indonesian tool-calling agents."""


def default_tasks_path() -> str:
    """Locate the golden dataset regardless of the working directory.

    Installed wheels ship the dataset as ``tugas/data/golden_id.json`` (see the
    ``force-include`` entry in ``pyproject.toml``), so the CLI works from any
    cwd. In a source checkout we fall back to the canonical
    ``eval/datasets/golden_id.json`` next to the repo root.
    """
    try:
        from importlib.resources import files

        packaged = files("tugas").joinpath("data", "golden_id.json")
        if packaged.is_file():
            return str(packaged)
    except (ModuleNotFoundError, FileNotFoundError):
        pass

    source_tree = Path(__file__).resolve().parents[2] / DEFAULT_TASKS
    if source_tree.is_file():
        return str(source_tree)

    return DEFAULT_TASKS


def _load_or_exit(path: str | None):
    """Load tasks, turning a bad/missing dataset into a clean CLI error."""
    try:
        return load_tasks(path or default_tasks_path())
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc


@app.command("list")
def list_tasks(path: str | None = typer.Argument(None, help="path to golden tasks JSON")) -> None:
    """Show the golden tasks."""
    tasks = _load_or_exit(path)
    t = Table(title=f"tasks ({len(tasks)})")
    t.add_column("id")
    t.add_column("category")
    t.add_column("expected tools")
    t.add_column("prompt")
    for task in tasks:
        t.add_row(
            task.id,
            task.category,
            ", ".join(e.tool for e in task.expected),
            task.prompt[:60],
        )
    console.print(t)


@app.command("run")
def run(
    models: str = typer.Option(..., "--model", "-m", help="comma-separated model ids"),
    path: str | None = typer.Argument(None, help="path to golden tasks JSON"),
    out: str | None = typer.Option(None, "--out", help="write results JSON here"),
) -> None:
    """Run one or more models over the golden tasks and print a score table."""
    tasks = _load_or_exit(path)
    model_ids = [m.strip() for m in models.split(",") if m.strip()]
    if not model_ids:
        console.print("[red]--model needs at least one model id[/]")
        raise typer.Exit(code=2)
    all_results = []
    for mid in model_ids:
        agent = LLMAgent(model=mid)
        results = run_agent(agent, tasks)
        all_results.extend(results)
        s = summarize(results)
        console.print(f"\n[bold]{mid}[/]  exact_match={s['exact_match_rate']}  "
                      f"tool_sel={s['tool_selection']}  arg_acc={s['arg_accuracy']}  "
                      f"steps={s['avg_steps']}")
        console.print(f"  failures: {s['failures']}")

    t = Table(title="per-task results")
    t.add_column("model")
    t.add_column("task")
    t.add_column("failure")
    t.add_column("tool_sel", justify="right")
    t.add_column("arg_acc", justify="right")
    t.add_column("steps", justify="right")
    for r in all_results:
        style = "green" if r.failure.value == "OK" else "yellow"
        t.add_row(r.model, r.task_id, f"[{style}]{r.failure.value}[/]",
                  f"{r.tool_selection:.2f}", f"{r.arg_accuracy:.2f}", str(r.steps))
    console.print(t)

    if out:
        try:
            outp = Path(out)
            outp.parent.mkdir(parents=True, exist_ok=True)
            outp.write_text(
                json.dumps([r.model_dump() for r in all_results], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            console.print(f"[red]gagal menulis {out}: {exc}[/]")
            raise typer.Exit(code=1) from exc
        console.print(f"[green]wrote {out}[/]")


    failures = {r.failure for r in all_results}
    if all_results and failures == {FailureType.API_ERROR}:
        console.print(
            "[red]every task failed with API_ERROR[/] — check TUGAS_BASE_URL and "
            "TUGAS_API_KEY, then rerun.",
        )
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
