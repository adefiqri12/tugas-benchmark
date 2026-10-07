"""Load tasks and run an agent across them."""

from __future__ import annotations

import json
from pathlib import Path

from .agent import Agent
from .models import Task, TaskResult
from .score import score_task


def load_tasks(path: str | Path) -> list[Task]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Task(**t) for t in data["tasks"]]


def run_agent(agent: Agent, tasks: list[Task]) -> list[TaskResult]:
    results: list[TaskResult] = []
    for t in tasks:
        traj = agent.run(t)
        results.append(score_task(t, agent.name, traj))
    return results
