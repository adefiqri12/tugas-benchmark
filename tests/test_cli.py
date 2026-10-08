"""CLI data-location tests. Fully offline: no gateway, no network."""

from __future__ import annotations

from pathlib import Path

from tugas.cli import default_tasks_path
from tugas.runner import load_tasks


def test_default_tasks_path_resolves_to_existing_file() -> None:
    # Must work regardless of the current working directory.
    p = Path(default_tasks_path())
    assert p.is_file(), f"golden dataset not found at {p}"


def test_default_tasks_path_loads_golden_tasks() -> None:
    tasks = load_tasks(default_tasks_path())
    assert tasks, "golden dataset should not be empty"
    assert all(t.expected for t in tasks), "every task needs an expected call"
