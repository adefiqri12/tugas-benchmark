"""CLI data-location tests. Fully offline: no gateway, no network."""

from __future__ import annotations

from pathlib import Path

from tugas.agent import ScriptedAgent
from tugas.cli import default_tasks_path
from tugas.models import FailureType, ToolCallRecord
from tugas.runner import load_tasks, run_agent


def test_default_tasks_path_resolves_to_existing_file() -> None:
    # Must work regardless of the current working directory.
    p = Path(default_tasks_path())
    assert p.is_file(), f"golden dataset not found at {p}"


def test_default_tasks_path_loads_golden_tasks() -> None:
    tasks = load_tasks(default_tasks_path())
    assert tasks, "golden dataset should not be empty"
    assert all(t.expected for t in tasks), "every task needs an expected call"


def test_golden_dataset_has_seven_tasks() -> None:
    tasks = load_tasks(default_tasks_path())
    assert len(tasks) == 7, f"expected 7 golden tasks, got {len(tasks)}"
    assert all(t.expected for t in tasks), "every task needs a non-empty expected trace"


def test_golden_dataset_includes_new_task_ids() -> None:
    ids = {t.id for t in load_tasks(default_tasks_path())}
    for expected_id in ("id-tanggal-lusa", "id-tanggal-ddmm", "id-satuan-kodi"):
        assert expected_id in ids, f"{expected_id} missing from golden set"


def test_new_golden_tasks_replay_to_ok() -> None:
    # Offline end-to-end: a ScriptedAgent replaying each new task's expected
    # calls must score FailureType.OK (exercises the any-of replay path).
    new_ids = {"id-tanggal-lusa", "id-tanggal-ddmm", "id-satuan-kodi"}
    tasks = [t for t in load_tasks(default_tasks_path()) if t.id in new_ids]
    assert len(tasks) == 3

    plan = {}
    for task in tasks:
        plan[task.id] = [
            ToolCallRecord(tool=call.tool, args=dict(call.args)) for call in task.expected
        ]

    results = run_agent(ScriptedAgent("scripted", plan), tasks)
    assert len(results) == 3
    for result in results:
        assert result.failure == FailureType.OK, f"{result.task_id}: {result.failure}"

