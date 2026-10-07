"""Scorer tests — no network. Each case builds a trajectory by hand and checks
the failure label the harness assigns."""

from __future__ import annotations

from tugas.agent import ScriptedAgent
from tugas.models import ExpectedCall, FailureType, Task, ToolCallRecord, Trajectory
from tugas.runner import run_agent
from tugas.score import score_task, summarize

TASK = Task(
    id="t1",
    prompt="Cuaca di Samarinda minggu depan?",
    expected=[
        ExpectedCall(tool="parse_tanggal", args={"teks": "minggu depan"}),
        ExpectedCall(tool="cek_cuaca", args={"kota": "Samarinda"}),
    ],
)


def test_perfect_trace_is_ok():
    traj = Trajectory(calls=[
        ToolCallRecord(tool="parse_tanggal", args={"teks": "minggu depan"}),
        ToolCallRecord(tool="cek_cuaca", args={"kota": "Samarinda"}),
    ])
    r = score_task(TASK, "m", traj)
    assert r.failure == FailureType.OK
    assert r.exact_match is True
    assert r.tool_selection == 1.0
    assert r.arg_accuracy == 1.0


def test_helper_tool_is_tolerated():
    # extra helper call before the expected ones -> EXTRA_TOOL, still exact
    traj = Trajectory(calls=[
        ToolCallRecord(tool="hitung", args={"ekspresi": "1+1"}),
        ToolCallRecord(tool="parse_tanggal", args={"teks": "minggu depan"}),
        ToolCallRecord(tool="cek_cuaca", args={"kota": "Samarinda"}),
    ])
    r = score_task(TASK, "m", traj)
    assert r.failure == FailureType.EXTRA_TOOL
    assert r.exact_match is True


def test_no_call():
    r = score_task(TASK, "m", Trajectory())
    assert r.failure == FailureType.NO_TOOL_CALL
    assert r.tool_selection == 0.0


def test_missing_second_tool():
    traj = Trajectory(calls=[ToolCallRecord(tool="parse_tanggal", args={"teks": "minggu depan"})])
    r = score_task(TASK, "m", traj)
    assert r.failure == FailureType.MISSING_TOOL
    assert r.tool_selection == 0.5


def test_bad_args():
    traj = Trajectory(calls=[
        ToolCallRecord(tool="parse_tanggal", args={"teks": "minggu depan"}),
        ToolCallRecord(tool="cek_cuaca", args={"kota": "Balikpapan"}),
    ])
    r = score_task(TASK, "m", traj)
    assert r.failure == FailureType.BAD_ARGS
    assert r.arg_accuracy == 0.5


def test_wrong_tool():
    traj = Trajectory(calls=[ToolCallRecord(tool="hitung", args={"ekspresi": "1+1"})])
    r = score_task(TASK, "m", traj)
    assert r.failure == FailureType.WRONG_TOOL


def test_order_error():
    traj = Trajectory(calls=[
        ToolCallRecord(tool="cek_cuaca", args={"kota": "Samarinda"}),
        ToolCallRecord(tool="parse_tanggal", args={"teks": "minggu depan"}),
    ])
    r = score_task(TASK, "m", traj)
    assert r.failure == FailureType.ORDER_ERROR


def test_rupiah_normalisation():
    task = Task(
        id="t2",
        prompt="Sisa?",
        expected=[ExpectedCall(tool="hitung", args={"ekspresi": "1200000 - 150000"})],
    )
    traj = Trajectory(calls=[ToolCallRecord(tool="hitung", args={"ekspresi": "1.200.000-150.000"})])
    assert score_task(task, "m", traj).failure == FailureType.OK


def test_api_error_path():
    r = score_task(TASK, "m", Trajectory(error="ConnectError: refused"))
    assert r.failure == FailureType.API_ERROR
    assert r.steps == 0


def test_summarize_and_scripted_agent():
    plan = {
        "t1": [
            ToolCallRecord(tool="parse_tanggal", args={"teks": "minggu depan"}),
            ToolCallRecord(tool="cek_cuaca", args={"kota": "Samarinda"}),
        ]
    }
    results = run_agent(ScriptedAgent("scripted", plan), [TASK])
    s = summarize(results)
    assert s["tasks"] == 1
    assert s["exact_match_rate"] == 1.0
    assert s["failures"] == {"OK": 1}
