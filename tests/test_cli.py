"""CLI data-location tests. Fully offline: no gateway, no network."""

from __future__ import annotations

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
