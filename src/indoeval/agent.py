"""Agents: anything that turns a task prompt into a tool-call trajectory.

Two implementations:
- LLMAgent      — a real tool-calling model behind an OpenAI-compatible gateway.
- ScriptedAgent — deterministic offline baseline; used in tests and as a floor.

Both expose the same `.run(task) -> Trajectory` surface, so the runner and the
scorer never care which one is in play.
"""

from __future__ import annotations

import json
import os
from typing import Protocol

import httpx

from .models import Task, ToolCallRecord, Trajectory
from .tools import ToolError, execute, openai_schema

SYSTEM = (
    "Kamu adalah agent yang harus memanggil tool untuk menjawab. "
    "Jawab dalam Bahasa Indonesia. Jangan mengarang data — panggil tool yang sesuai. "
    "Jika perlu tanggal absolut, panggil parse_tanggal lebih dulu."
)


class Agent(Protocol):
    name: str

    def run(self, task: Task) -> Trajectory: ...


class LLMAgent:
    """Tool-calling loop against an OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        max_steps: int = 4,
        timeout: float = 60.0,
    ) -> None:
        self.name = model
        self.model = model
        self.base_url = (base_url or os.environ.get("INDOEVAL_BASE_URL")
                         or "http://127.0.0.1:20129/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("INDOEVAL_API_KEY", "")
        self.max_steps = max_steps
        self.timeout = timeout

    def run(self, task: Task) -> Trajectory:
        traj = Trajectory()
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": task.prompt},
        ]
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            for _ in range(self.max_steps):
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "tools": openai_schema(),
                    "tool_choice": "auto",
                    "temperature": 0,
                }
                r = httpx.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                r.raise_for_status()
                msg = r.json()["choices"][0]["message"]
                calls = msg.get("tool_calls") or []
                if not calls:
                    traj.final_text = msg.get("content") or ""
                    break

                messages.append(msg)
                for c in calls:
                    fn = c["function"]
                    name = fn["name"]
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    traj.calls.append(ToolCallRecord(tool=name, args=args))
                    try:
                        result = execute(name, args)
                        content = json.dumps(result, ensure_ascii=False)
                    except ToolError as exc:
                        content = json.dumps({"error": str(exc)}, ensure_ascii=False)
                    messages.append(
                        {"role": "tool", "tool_call_id": c["id"], "content": content}
                    )
            else:
                traj.final_text = "(max_steps tercapai)"
        except Exception as exc:  # noqa: BLE001 - transport/provider failure
            traj.error = f"{type(exc).__name__}: {exc}"
        return traj


class ScriptedAgent:
    """Offline baseline: emits a fixed trajectory per task id (or nothing)."""

    def __init__(self, name: str, plan: dict[str, list[ToolCallRecord]] | None = None) -> None:
        self.name = name
        self.plan = plan or {}

    def run(self, task: Task) -> Trajectory:
        calls = self.plan.get(task.id, [])
        return Trajectory(calls=list(calls), final_text="(scripted)")
