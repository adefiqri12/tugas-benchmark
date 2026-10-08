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


def test_golden_dataset_has_nine_tasks() -> None:
    tasks = load_tasks(default_tasks_path())
    assert len(tasks) == 9, f"expected 9 golden tasks, got {len(tasks)}"
    assert all(t.expected for t in tasks), "every task needs a non-empty expected trace"


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

