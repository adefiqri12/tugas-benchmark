"""Dataset integrity tests: the shipped golden set against the tool registry.

These tests are the guardrail for contributors: a task that points at a
missing tool, a misspelled argument, or a stale generated JSON fails here
before it can reach the leaderboard.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tugas.models import Task
from tugas.runner import load_tasks
from tugas.tools import TOOL_SPECS, execute, parse_tanggal

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "eval/datasets/golden_id.json"
SPECS = {spec.name: spec for spec in TOOL_SPECS}
ANCHOR = "2026-10-07"

@pytest.fixture(scope="module")
def tasks() -> list[Task]:
    return load_tasks(DATA)

@pytest.fixture(scope="module")
def raw_doc() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))

def test_dataset_is_broad(tasks: list[Task]) -> None:
    assert len(tasks) >= 50
    tool_tasks = [t for t in tasks if t.expected]
    no_tool_tasks = [t for t in tasks if not t.expected]
    assert len(tool_tasks) >= 45
    assert no_tool_tasks

def test_ids_are_unique(tasks: list[Task]) -> None:
    ids = [t.id for t in tasks]
    assert len(ids) == len(set(ids))

def test_prompts_are_unique(tasks: list[Task]) -> None:
    prompts = [t.prompt for t in tasks]
    assert len(prompts) == len(set(prompts))

def test_each_task_is_exactly_one_kind(tasks: list[Task]) -> None:
    for task in tasks:
        assert bool(task.expected) != bool(task.answer_any_of), task.id

def test_expected_tools_exist(tasks: list[Task]) -> None:
    for task in tasks:
        for call in task.expected:
            assert call.tool in SPECS, f"{task.id}: unknown tool {call.tool}"

def test_expected_args_are_declared(tasks: list[Task]) -> None:
    for task in tasks:
        for call in task.expected:
            declared = SPECS[call.tool].parameters
            for key in call.args:
                assert key in declared, f"{task.id}: {call.tool} has no arg {key!r}"

def test_all_tools_are_covered(tasks: list[Task]) -> None:
    used = {call.tool for task in tasks for call in task.expected}
    assert used == set(SPECS), f"uncovered tools: {sorted(set(SPECS) - used)}"

def test_anchor_date_is_pinned(raw_doc: dict) -> None:
    assert raw_doc["meta"]["anchor_date"] == ANCHOR
    assert parse_tanggal("hari ini")["tanggal"] == ANCHOR

def test_expected_calls_execute(tasks: list[Task]) -> None:
    """Every fully specified expected call must run without a ToolError.

    A golden call may intentionally list fewer args than the tool needs (it is
    a scoring spec, not a replayable call); those are skipped explicitly.
    """
    skipped: list[tuple[str, str]] = []
    for task in tasks:
        for call in task.expected:
            spec = SPECS[call.tool]
            if not set(spec.required) <= set(call.args):
                skipped.append((task.id, call.tool))
                continue
            args = {
                k: (v[0] if isinstance(v, list) else v)
                for k, v in call.args.items()
            }
            execute(call.tool, args)

    # The only known partial spec today: cek_cuaca scores the city, not dates.
    assert skipped == [("id-cuaca-relative", "cek_cuaca")]

def test_generated_json_is_in_sync() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/build_dataset.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def _write_dataset(tmp_path: Path, tasks: list[dict]) -> Path:
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"tasks": tasks}), encoding="utf-8")
    return path


def test_load_tasks_rejects_duplicate_ids(tmp_path: Path) -> None:
    task = {"id": "same", "prompt": "p", "expected": [], "answer_any_of": ["ok"]}
    path = _write_dataset(tmp_path, [task, task])
    with pytest.raises(ValueError, match="id duplikat"):
        load_tasks(path)


def test_load_tasks_rejects_task_with_both_kinds(tmp_path: Path) -> None:
    task = {
        "id": "both",
        "prompt": "p",
        "expected": [{"tool": "hitung", "args": {"ekspresi": "1+1"}}],
        "answer_any_of": ["2"],
    }
    path = _write_dataset(tmp_path, [task])
    with pytest.raises(ValueError, match="tepat satu"):
        load_tasks(path)


def test_load_tasks_rejects_task_with_no_expected_behaviour(tmp_path: Path) -> None:
    path = _write_dataset(tmp_path, [{"id": "empty", "prompt": "p", "expected": []}])
    with pytest.raises(ValueError, match="tepat satu"):
        load_tasks(path)


def test_load_tasks_rejects_empty_dataset(tmp_path: Path) -> None:
    path = _write_dataset(tmp_path, [])
    with pytest.raises(ValueError, match="dataset kosong"):
        load_tasks(path)
