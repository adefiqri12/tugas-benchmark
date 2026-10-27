"""Load tasks and run an agent across them."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from .agent import Agent
from .models import Task, TaskResult
from .score import score_task


def _validate_dataset(tasks: list[Task]) -> None:
    """Reject datasets the scorer cannot attribute or score unambiguously."""
    if not tasks:
        raise ValueError("dataset kosong")
    seen: set[str] = set()
    dupes: set[str] = set()
    for task in tasks:
        if task.id in seen:
            dupes.add(task.id)
        seen.add(task.id)
    if dupes:
        raise ValueError(f"id duplikat: {sorted(dupes)}")
    for task in tasks:
        if bool(task.expected) == bool(task.answer_any_of):
            raise ValueError(
                f"task {task.id!r}: harus punya tepat satu dari 'expected' "
                "atau 'answer_any_of'"
            )


def load_tasks(path: str | Path) -> list[Task]:
    """Load golden tasks, turning any malformed-dataset error into a clear message."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        tasks = [Task(**t) for t in data["tasks"]]
        _validate_dataset(tasks)
        return tasks
    except (
        OSError, json.JSONDecodeError, KeyError, TypeError, ValidationError, ValueError
    ) as exc:
        raise ValueError(f"gagal memuat golden tasks dari {p}: {exc}") from exc


def run_agent(agent: Agent, tasks: list[Task]) -> list[TaskResult]:
    results: list[TaskResult] = []
    for t in tasks:
        traj = agent.run(t)
        results.append(score_task(t, agent.name, traj))
    return results
