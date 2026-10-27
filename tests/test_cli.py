"""CLI data-location tests. Fully offline: no gateway, no network."""

from __future__ import annotations

import json

from pathlib import Path

import pytest
from typer.testing import CliRunner

from tugas.agent import ScriptedAgent
from tugas.cli import app, default_tasks_path
from tugas.models import FailureType, ToolCallRecord
from tugas.runner import load_tasks, run_agent


def test_default_tasks_path_resolves_to_existing_file() -> None:
    # Must work regardless of the current working directory.
    p = Path(default_tasks_path())
    assert p.is_file(), f"golden dataset not found at {p}"


def test_default_tasks_path_loads_golden_tasks() -> None:
    tasks = load_tasks(default_tasks_path())
    assert tasks, "golden dataset should not be empty"
    # A task is either a tool-call task or a no-tool abstain task.
    assert all(bool(t.expected) != bool(t.answer_any_of) for t in tasks), (
        "each task must define exactly one of expected calls / accepted answers"
    )


def test_golden_dataset_is_broad() -> None:
    tasks = load_tasks(default_tasks_path())
    assert len(tasks) >= 50, f"expected at least 50 golden tasks, got {len(tasks)}"
    tool_tasks = [t for t in tasks if t.expected]
    assert len(tool_tasks) >= 45, f"expected at least 45 tool tasks, got {len(tool_tasks)}"


def test_load_tasks_reports_malformed_dataset(tmp_path) -> None:
    # A typo or truncated file should raise a clear ValueError, not a raw
    # KeyError/JSONDecodeError traceback out of the public loader.
    bad = tmp_path / "bad.json"
    bad.write_text('{"meta": {}}', encoding="utf-8")
    with pytest.raises(ValueError, match="gagal memuat golden tasks"):
        load_tasks(bad)

def test_cli_reports_missing_dataset_without_traceback(tmp_path) -> None:
    result = CliRunner().invoke(app, ["list", str(tmp_path / "missing.json")])
    assert result.exit_code == 1
    assert "gagal memuat golden tasks" in result.output


def test_golden_dataset_includes_new_task_ids() -> None:
    ids = {t.id for t in load_tasks(default_tasks_path())}
    for expected_id in (
        "id-tanggal-lusa",
        "id-tanggal-ddmm",
        "id-satuan-kodi",
        "id-sapaan-malam",
        "id-alamat-kota",
    ):
        assert expected_id in ids, f"{expected_id} missing from golden set"


def test_every_golden_task_replays_to_ok() -> None:
    # Offline end-to-end: a ScriptedAgent replaying each task's expected calls
    # must score FailureType.OK (exercises the any-of list-arg replay path).
    tasks = load_tasks(default_tasks_path())

    plan = {}
    for task in tasks:
        plan[task.id] = [
            ToolCallRecord(tool=call.tool, args=dict(call.args)) for call in task.expected
        ]

    results = run_agent(ScriptedAgent("scripted", plan), tasks)
    assert len(results) == len(tasks)
    for result in results:
        assert result.failure == FailureType.OK, f"{result.task_id}: {result.failure}"

def test_cli_run_writes_results(monkeypatch, tmp_path) -> None:
    # Exercise the run command end to end with a scripted agent: no gateway,
    # no network. Covers summary printing and --out JSON writing.
    from tugas import cli

    tasks = load_tasks(default_tasks_path())
    plan = {
        t.id: [ToolCallRecord(tool=c.tool, args=dict(c.args)) for c in t.expected]
        for t in tasks
    }
    monkeypatch.setattr(cli, "LLMAgent", lambda model, timeout: ScriptedAgent(model, plan))
    out = tmp_path / "results" / "run.json"
    result = CliRunner().invoke(
        app, ["run", "-m", "fake", str(default_tasks_path()), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "exact_match=1.0" in result.output
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert len(rows) == len(tasks)
    assert {row["failure"] for row in rows} == {"OK"}


def test_cli_reports_version() -> None:
    from tugas import __version__, cli

    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0, result.output
    assert __version__ in result.output


def test_cli_exits_nonzero_when_every_task_is_api_error(monkeypatch, tmp_path) -> None:
    # A gateway that is down must not look like a successful zero score.
    from tugas import cli
    from tugas.models import TaskResult

    def fake_run(agent, tasks):
        return [
            TaskResult(
                task_id=t.id, model="fake", failure=FailureType.API_ERROR,
                tool_selection=0.0, arg_accuracy=0.0, exact_match=False, steps=0,
            )
            for t in tasks
        ]

    monkeypatch.setattr(cli, "LLMAgent", lambda model, timeout: ScriptedAgent(model, {}))
    monkeypatch.setattr(cli, "run_agent", fake_run)
    out = tmp_path / "api_error.json"
    result = CliRunner().invoke(
        app, ["run", "-m", "fake", str(default_tasks_path()), "--out", str(out)]
    )
    assert result.exit_code == 2, result.output
    assert "every task failed with API_ERROR" in result.output
    assert len(json.loads(out.read_text(encoding="utf-8"))) == len(load_tasks(default_tasks_path()))


def test_cli_rejects_empty_model_list(monkeypatch) -> None:
    # An empty -m must not look like a successful run.
    from tugas import cli

    monkeypatch.setattr(
        cli, "LLMAgent", lambda model, timeout: pytest.fail("agent must not be constructed")
    )
    result = CliRunner().invoke(app, ["run", "-m", " , "])
    assert result.exit_code == 2, result.output
    assert "--model needs at least one model id" in result.output


def test_cli_reports_unwritable_out(monkeypatch, tmp_path) -> None:
    from tugas import cli
    from tugas.models import TaskResult

    def fake_run(agent, tasks):
        return [
            TaskResult(
                task_id=t.id, model="fake", failure=FailureType.OK,
                tool_selection=1.0, arg_accuracy=1.0, exact_match=True, steps=1,
            )
            for t in tasks
        ]

    monkeypatch.setattr(cli, "LLMAgent", lambda model, timeout: ScriptedAgent(model, {}))
    monkeypatch.setattr(cli, "run_agent", fake_run)
    out_dir = tmp_path / "outdir"
    out_dir.mkdir()
    result = CliRunner().invoke(
        app, ["run", "-m", "fake", str(default_tasks_path()), "--out", str(out_dir)]
    )
    assert result.exit_code == 1, result.output
    assert "gagal menulis" in result.output


def test_cli_passes_timeout_to_agent(monkeypatch, tmp_path):
    from tugas import cli

    seen = {}
    monkeypatch.setattr(
        cli,
        "LLMAgent",
        lambda model, timeout: (
            seen.update(model=model, timeout=timeout) or ScriptedAgent(model, {})
        ),
    )
    monkeypatch.setattr(cli, "run_agent", lambda agent, tasks: [])

    result = CliRunner().invoke(
        app, ["run", "-m", "fake", "--timeout", "5", "--out", str(tmp_path / "o.json")]
    )

    assert result.exit_code == 0, result.output
    assert seen == {"model": "fake", "timeout": 5.0}
