"""Offline tests for the LLMAgent tool-calling loop. No network: the gateway
POST is monkeypatched, so the loop's own logic (tool execution, result
feedback, error handling, step cap) is what gets exercised."""

from __future__ import annotations

import json

import tugas.agent as agent
from tugas.models import ExpectedCall, FailureType, Task
from tugas.score import score_task


class _Resp:
    def __init__(self, data: dict) -> None:
        self._data = data

    def raise_for_status(self) -> None:  # noqa: D401 - stub
        return None

    def json(self) -> dict:
        return self._data


def _tool_call(call_id: str, name: str, args: dict) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }
                    ],
                }
            }
        ]
    }


def _answer(text: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text, "tool_calls": []}}]}


def _install(monkeypatch, script: list[dict], seen: dict) -> None:
    """Serve `script` responses in order; record payloads and tool messages."""
    seen["payloads"] = []
    seen["tool_msgs"] = []

    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: A002 - match httpx
        seen["payloads"].append(json)
        msgs = json["messages"]
        if msgs and msgs[-1]["role"] == "tool":
            seen["tool_msgs"].append(msgs[-1]["content"])
        idx = min(len(seen["payloads"]) - 1, len(script) - 1)
        return _Resp(script[idx])

    monkeypatch.setattr(agent.httpx, "post", fake_post)


def test_llm_agent_executes_multi_step_loop(monkeypatch):
    seen: dict = {}
    _install(
        monkeypatch,
        [
            _tool_call("c1", "parse_tanggal", {"teks": "lusa"}),
            _tool_call("c2", "sapaan_waktu", {"jam": "19:30"}),
            _answer("Tanggal 2026-10-09, selamat malam."),
        ],
        seen,
    )
    task = Task(
        id="t",
        prompt="Meeting-ku lusa, sapaannya nanti malam apa?",
        expected=[
            ExpectedCall(tool="parse_tanggal", args={"teks": "lusa"}),
            ExpectedCall(tool="sapaan_waktu", args={"jam": "19:30"}),
        ],
    )

    traj = agent.LLMAgent("mock-model").run(task)

    assert traj.error is None
    assert [c.tool for c in traj.calls] == ["parse_tanggal", "sapaan_waktu"]
    assert traj.final_text == "Tanggal 2026-10-09, selamat malam."
    # tool results were executed and fed back into the conversation
    assert json.loads(seen["tool_msgs"][0])["tanggal"] == "2026-10-09"
    assert json.loads(seen["tool_msgs"][1])["sapaan"] == "Selamat malam"
    # all 6 tools advertised on the first request
    assert len(seen["payloads"][0]["tools"]) == 6

    result = score_task(task, "mock-model", traj)
    assert result.failure == FailureType.OK
    assert result.exact_match is True


def test_llm_agent_feeds_tool_error_back(monkeypatch):
    seen: dict = {}
    _install(
        monkeypatch,
        [
            _tool_call("e1", "parse_tanggal", {"teks": "7/10/26"}),
            _answer("tidak bisa."),
        ],
        seen,
    )
    task = Task(
        id="x",
        prompt="p",
        expected=[ExpectedCall(tool="parse_tanggal", args={"teks": "7/10/26"})],
    )

    traj = agent.LLMAgent("mock").run(task)

    assert traj.error is None
    assert "error" in json.loads(seen["tool_msgs"][0])
    assert traj.final_text == "tidak bisa."


def test_llm_agent_transport_failure_is_api_error(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(agent.httpx, "post", boom)
    task = Task(id="z", prompt="p", expected=[ExpectedCall(tool="parse_tanggal", args={"teks": "besok"})])

    traj = agent.LLMAgent("mock").run(task)

    assert traj.error is not None
    assert "connection refused" in traj.error
    assert score_task(task, "mock", traj).failure == FailureType.API_ERROR


def test_llm_agent_respects_max_steps(monkeypatch):
    seen: dict = {}
    # Gateway always asks for another tool call -> loop must stop at max_steps.
    _install(monkeypatch, [_tool_call("c", "parse_tanggal", {"teks": "besok"})], seen)
    task = Task(id="m", prompt="p", expected=[ExpectedCall(tool="parse_tanggal", args={"teks": "besok"})])

    traj = agent.LLMAgent("mock", max_steps=3).run(task)

    assert len(traj.calls) == 3
    assert traj.final_text == "(max_steps tercapai)"
