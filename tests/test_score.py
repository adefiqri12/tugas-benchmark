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
    # No calls means no expected args matched, not vacuously perfect.
    assert r.arg_accuracy == 0.0


def test_missing_second_tool():
    traj = Trajectory(calls=[ToolCallRecord(tool="parse_tanggal", args={"teks": "minggu depan"})])
    r = score_task(TASK, "m", traj)
    assert r.failure == FailureType.MISSING_TOOL
    assert r.tool_selection == 0.5
    assert r.arg_accuracy == 0.5


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


DUP = Task(
    id="dup",
    prompt="Berapa 2+2, lalu 3*3?",
    expected=[
        ExpectedCall(tool="hitung", args={"ekspresi": "2+2"}),
        ExpectedCall(tool="hitung", args={"ekspresi": "3*3"}),
    ],
)


def test_duplicate_expected_tools_perfect_trace_is_ok():
    # The same tool twice in the expected trace: each occurrence matches its
    # own call (the old scorer mislabelled this ORDER_ERROR).
    traj = Trajectory(calls=[
        ToolCallRecord(tool="hitung", args={"ekspresi": "2+2"}),
        ToolCallRecord(tool="hitung", args={"ekspresi": "3*3"}),
    ])
    r = score_task(DUP, "m", traj)
    assert r.failure == FailureType.OK
    assert r.tool_selection == 1.0
    assert r.arg_accuracy == 1.0
    assert r.exact_match is True


def test_duplicate_expected_tools_partial_is_missing_tool():
    # Only the first call happened: MISSING_TOOL, not ORDER_ERROR.
    traj = Trajectory(calls=[ToolCallRecord(tool="hitung", args={"ekspresi": "2+2"})])
    r = score_task(DUP, "m", traj)
    assert r.failure == FailureType.MISSING_TOOL
    # tool_selection counts per-occurrence, not per-unique-name (old scorer
    # double-counted: 2 unique-name hits / 2 expected = 1.0).
    assert r.tool_selection == 0.5


def test_duplicate_expected_tools_args_scored_per_call():
    # Second occurrence has wrong args: BAD_ARGS with arg_accuracy 0.5 (old
    # scorer compared both expected args against the first call's args).
    traj = Trajectory(calls=[
        ToolCallRecord(tool="hitung", args={"ekspresi": "2+2"}),
        ToolCallRecord(tool="hitung", args={"ekspresi": "9"}),
    ])
    r = score_task(DUP, "m", traj)
    assert r.failure == FailureType.BAD_ARGS
    assert r.arg_accuracy == 0.5


NO_TOOL = Task(id="no_tool", prompt="Jawab tanpa tool.", expected=[])


def test_no_tool_task_abstaining_is_ok():
    # expected=[] means a correct agent calls nothing and still replies.
    r = score_task(NO_TOOL, "m", Trajectory(final_text="Baik."))
    assert r.failure == FailureType.OK
    assert r.tool_selection == 1.0
    assert r.arg_accuracy == 1.0
    assert r.exact_match is True

def test_no_tool_task_empty_answer_is_wrong_answer():
    # Abstention without any reply is not an answer.
    r = score_task(NO_TOOL, "m", Trajectory(final_text="   "))
    assert r.failure == FailureType.WRONG_ANSWER
    assert r.exact_match is False


def test_no_tool_task_calling_anyway_is_wrong_tool():
    # A no-tool task exists to catch the discipline failure (the Phase 1
    # lesson: answering "besok tanggal berapa?" from a guess). Calling any
    # tool is a failure, not a tolerated extra.
    traj = Trajectory(calls=[ToolCallRecord(tool="hitung", args={"ekspresi": "1+1"})])
    r = score_task(NO_TOOL, "m", traj)
    assert r.failure == FailureType.WRONG_TOOL
    assert r.exact_match is False


NO_TOOL_ANSWER = Task(
    id="no_tool_answer",
    prompt="Halo, apa kabar?",
    expected=[],
    answer_any_of=["Baik, terima kasih", "Baik. Ada yang bisa dibantu?"],
)


def test_no_tool_answer_match_is_ok():
    # Reference answers are not a required wording; any reply is fine.
    r = score_task(NO_TOOL_ANSWER, "m", Trajectory(final_text="baik,  terima kasih"))
    assert r.failure == FailureType.OK
    assert r.arg_accuracy == 1.0
    assert r.exact_match is True

def test_no_tool_answer_different_wording_is_ok():
    # The benchmark scores tool use, not prose: a different but non-empty
    # reply still passes the abstention task.
    r = score_task(NO_TOOL_ANSWER, "m", Trajectory(final_text="Halo juga"))
    assert r.failure == FailureType.OK
    assert r.exact_match is True


def test_no_tool_answer_but_called_tool_is_wrong_tool():
    # Discipline failure dominates the answer check.
    traj = Trajectory(
        calls=[ToolCallRecord(tool="hitung", args={"ekspresi": "1+1"})],
        final_text="Baik, terima kasih",
    )
    r = score_task(NO_TOOL_ANSWER, "m", traj)
    assert r.failure == FailureType.WRONG_TOOL

def test_extra_same_tool_call_before_expected_is_tolerated():
    # A helper call to the same tool with other args must not steal the slot
    # of a required call: extras are tolerated regardless of tool name.
    traj = Trajectory(calls=[
        ToolCallRecord(tool="hitung", args={"ekspresi": "1+1"}),
        ToolCallRecord(tool="hitung", args={"ekspresi": "2+2"}),
        ToolCallRecord(tool="hitung", args={"ekspresi": "3*3"}),
    ])
    r = score_task(DUP, "m", traj)
    assert r.failure == FailureType.EXTRA_TOOL
    assert r.exact_match is True
    assert r.arg_accuracy == 1.0

def test_duplicate_expected_tools_reversed_is_order_error():
    # Both required calls happened with the right args, only the order is
    # wrong: ORDER_ERROR, not BAD_ARGS, and the args are all correct.
    traj = Trajectory(calls=[
        ToolCallRecord(tool="hitung", args={"ekspresi": "3*3"}),
        ToolCallRecord(tool="hitung", args={"ekspresi": "2+2"}),
    ])
    r = score_task(DUP, "m", traj)
    assert r.failure == FailureType.ORDER_ERROR
    assert r.tool_selection == 1.0
    assert r.arg_accuracy == 1.0
    assert r.exact_match is False
