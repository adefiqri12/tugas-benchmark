"""Scoring: compare an agent trajectory against a task's expected trace.

Metrics are named for what they measure on *agents* — not borrowed from
retrieval. A RAG harness ranks chunks (precision@k, nDCG); an agent harness
checks which tool fired, with what arguments, in what order.

Design choices that matter:
- Expected calls must appear as an *ordered subsequence* of the actual calls.
  Extra helper calls (e.g. resolving a date before a search) are tolerated and
  flagged as EXTRA_TOOL, not treated as failure — real agents legitimately call
  helpers, and punishing that would measure rigidity, not correctness.
- Argument comparison normalises thousands separators and operator spacing, so
  "1.200.000 - 150.000" matches "1200000-150000".
"""

from __future__ import annotations

import re

from .models import (
    ExpectedCall,
    FailureType,
    Task,
    TaskResult,
    ToolCallRecord,
    Trajectory,
)

_SEP = re.compile(r"(?<=\d)[.,](?=\d{3}(?:\D|$))")  # 1.200.000 -> 1200000
_OPS = re.compile(r"\s*([+\-*/])\s*")                # "a - b" -> "a-b"
_WS = re.compile(r"\s+")


def _norm(v: object) -> str:
    s = str(v).strip().lower()
    s = _SEP.sub("", s)
    s = _OPS.sub(r"\1", s)
    s = _WS.sub(" ", s).strip()
    return s


def _args_ok(expected: dict, got: dict) -> bool:
    return all(k in got and _norm(got[k]) == _norm(v) for k, v in expected.items())


def _subsequence(expected: list[ExpectedCall], calls: list[ToolCallRecord]):
    """Walk `expected` as an ordered subsequence of `calls`.

    Returns (matched_pairs, all_args_ok).
    """
    j = 0
    matched: list[tuple[ExpectedCall, ToolCallRecord]] = []
    for e in expected:
        while j < len(calls) and calls[j].tool != e.tool:
            j += 1
        if j >= len(calls):
            return matched, False
        matched.append((e, calls[j]))
        j += 1
    all_args = all(_args_ok(e.args, g.args) for e, g in matched)
    return matched, all_args


def _classify(expected: list[ExpectedCall], calls: list[ToolCallRecord]) -> FailureType:
    if not calls:
        return FailureType.NO_TOOL_CALL
    exp_tools = {e.tool for e in expected}
    got_tools = {c.tool for c in calls}
    if not (exp_tools & got_tools):
        return FailureType.WRONG_TOOL
    if not exp_tools <= got_tools:
        return FailureType.MISSING_TOOL
    matched, args_ok = _subsequence(expected, calls)
    if len(matched) < len(expected):
        return FailureType.ORDER_ERROR
    if not args_ok:
        return FailureType.BAD_ARGS
    if len(calls) > len(expected):
        return FailureType.EXTRA_TOOL
    return FailureType.OK


def score_task(task: Task, model: str, traj: Trajectory) -> TaskResult:
    if traj.error:
        return TaskResult(
            task_id=task.id, model=model, failure=FailureType.API_ERROR,
            tool_selection=0.0, arg_accuracy=0.0, exact_match=False,
            steps=len(traj.calls), note=traj.error,
        )

    expected = task.expected
    calls = traj.calls
    failure = _classify(expected, calls)

    got_tools = [c.tool for c in calls]
    hit = sum(1 for e in expected if e.tool in got_tools)
    tool_selection = hit / len(expected) if expected else 0.0

    m = t = 0
    for e in expected:
        cand = next((c for c in calls if c.tool == e.tool), None)
        if cand:
            m += sum(1 for k, v in e.args.items()
                     if k in cand.args and _norm(cand.args[k]) == _norm(v))
        t += len(e.args)
    arg_accuracy = m / t if t else 1.0

    return TaskResult(
        task_id=task.id, model=model, failure=failure,
        tool_selection=round(tool_selection, 3),
        arg_accuracy=round(arg_accuracy, 3),
        # "exact" = the required behaviour happened (helper calls tolerated)
        exact_match=(failure in (FailureType.OK, FailureType.EXTRA_TOOL)),
        steps=len(calls),
    )


def summarize(results: list[TaskResult]) -> dict:
    """Aggregate a model's results into headline numbers + a failure histogram."""
    n = len(results)
    if n == 0:
        return {"tasks": 0}
    fails: dict[str, int] = {}
    for r in results:
        fails[r.failure.value] = fails.get(r.failure.value, 0) + 1
    return {
        "tasks": n,
        "exact_match_rate": round(sum(r.exact_match for r in results) / n, 3),
        "tool_selection": round(sum(r.tool_selection for r in results) / n, 3),
        "arg_accuracy": round(sum(r.arg_accuracy for r in results) / n, 3),
        "avg_steps": round(sum(r.steps for r in results) / n, 2),
        "failures": fails,
    }
