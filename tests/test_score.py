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


def test_extra_args_are_ignored():
    # golden scores only `kota`; an agent that also supplies the required dates
    # must still score OK (extra args the agent adds are not penalised).
    traj = Trajectory(calls=[
        ToolCallRecord(tool="parse_tanggal", args={"teks": "minggu depan"}),
        ToolCallRecord(tool="cek_cuaca", args={
            "kota": "Samarinda",
            "tanggal_mulai": "2026-10-14",
            "tanggal_selesai": "2026-10-20",
        }),
    ])
    r = score_task(TASK, "m", traj)
    assert r.failure == FailureType.OK
    assert r.arg_accuracy == 1.0


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


def test_summarize_empty_keeps_all_keys():
    # An empty dataset must still expose every headline key, so consumers like
    # the CLI can index them without a KeyError.
    s = summarize([])
    assert s == {"tasks": 0, "exact_match_rate": 0.0, "tool_selection": 0.0,
                 "arg_accuracy": 0.0, "avg_steps": 0.0, "failures": {}}


def test_arg_matching_tolerates_case_and_whitespace():
    # Documented tolerance: a model that copies an address with different
    # spacing or case still matches (the whole point of `_norm`).
    task = Task(
        id="alamat",
        prompt="p",
        expected=[ExpectedCall(tool="parse_alamat", args={"teks": "Jl. A No. 12, Kel. B, Samarinda"})],
    )
    for variant in (
        "Jl. A No. 12, Kel. B, Samarinda",
        "jl. a no. 12,  kel. b, samarinda",
        "  Jl. A No. 12, Kel. B, Samarinda  ",
    ):
        traj = Trajectory(calls=[ToolCallRecord(tool="parse_alamat", args={"teks": variant})])
        assert score_task(task, "m", traj).failure == FailureType.OK, variant


def test_arg_matching_is_punctuation_sensitive():
    # Known limit (documented in tasks.md): `_norm` does not touch punctuation,
    # so "No.12" does not match "No. 12". Pinned so any change is deliberate.
    task = Task(
        id="alamat",
        prompt="p",
        expected=[ExpectedCall(tool="parse_alamat", args={"teks": "Jl. A No. 12, Samarinda"})],
    )
    traj = Trajectory(calls=[ToolCallRecord(tool="parse_alamat", args={"teks": "Jl. A No.12, Samarinda"})])
    assert score_task(task, "m", traj).failure == FailureType.BAD_ARGS


KODI = Task(
    id="kodi",
    prompt="Saya beli 2 kodi telur, pecah 3 butir. Sisa berapa butir?",
    expected=[ExpectedCall(tool="hitung", args={"ekspresi": ["2 * 20 - 3", "40 - 3"]})],
)


def test_any_of_arg_matches_each_acceptable_form():
    for form in ("40 - 3", "2 * 20 - 3"):
        traj = Trajectory(calls=[ToolCallRecord(tool="hitung", args={"ekspresi": form})])
        r = score_task(KODI, "m", traj)
        assert r.failure == FailureType.OK, form
        assert r.arg_accuracy == 1.0, form


def test_any_of_arg_rejects_non_member():
    traj = Trajectory(calls=[ToolCallRecord(tool="hitung", args={"ekspresi": "2 * 12 - 3"})])
    r = score_task(KODI, "m", traj)
    assert r.failure == FailureType.BAD_ARGS
    assert r.arg_accuracy == 0.0


def test_scripted_agent_replays_first_element_of_list_arg():
    plan = {"kodi": [ToolCallRecord(tool="hitung", args={"ekspresi": ["2 * 20 - 3", "40 - 3"]})]}
    results = run_agent(ScriptedAgent("scripted", plan), [KODI])
    assert results[0].failure == FailureType.OK
    assert results[0].arg_accuracy == 1.0
