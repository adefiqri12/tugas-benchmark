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


def _arg_matches(expected_v: object, actual_v: object) -> bool:
    """One expected arg value against one actual value.

    A list/tuple expected value is *any-of*: the actual value matches when it
    equals (after `_norm`) any element. A scalar keeps the exact-match behavior.
    """
    actual = _norm(actual_v)
    if isinstance(expected_v, (list, tuple)):
        return any(actual == _norm(e) for e in expected_v)
    return actual == _norm(expected_v)


def _args_ok(expected: dict, got: dict) -> bool:
    return all(k in got and _arg_matches(v, got[k]) for k, v in expected.items())


def _match_subsequence(
    expected: list[ExpectedCall],
    calls: list[ToolCallRecord],
    require_args: bool = False,
):
    """Greedy left-to-right matching of `expected` as an ordered subsequence
    of `calls`.

    With `require_args=False` only tool names are compared (used for
    tool_selection); with `require_args=True` a call must also satisfy the
    expected args. Greedy leftmost assignment maximises the number of matched
    expected calls: each expected call takes the earliest usable slot, never
    blocking a later match. Returns the matched (expected, actual) pairs — a
    partial match is fine; the caller decides what an incomplete match means.
    """
    j = 0
    matched: list[tuple[ExpectedCall, ToolCallRecord]] = []
    for e in expected:
        while j < len(calls) and not (
            calls[j].tool == e.tool
            and (not require_args or _args_ok(e.args, calls[j].args))
        ):
            j += 1
        if j >= len(calls):
            break
        matched.append((e, calls[j]))
        j += 1
    return matched


def _unordered_arg_match(
    expected: list[ExpectedCall], calls: list[ToolCallRecord]
) -> list[tuple[ExpectedCall, ToolCallRecord]] | None:
    """Pair every expected call with a distinct actual call, in any order.

    Distinguishes "right calls, wrong order" (ORDER_ERROR) from "wrong args"
    (BAD_ARGS). Backtracking is needed because taking a call for one expected
    entry can remove the only candidate for another. Traces are tiny (<= 3
    expected calls), so the worst case is negligible.
    """
    used = [False] * len(calls)
    pairs: list[tuple[ExpectedCall, ToolCallRecord]] = []

    def assign(i: int) -> bool:
        if i == len(expected):
            return True
        e = expected[i]
        for j, c in enumerate(calls):
            if used[j] or c.tool != e.tool or not _args_ok(e.args, c.args):
                continue
            used[j] = True
            pairs.append((e, c))
            if assign(i + 1):
                return True
            pairs.pop()
            used[j] = False
        return False

    return list(pairs) if assign(0) else None


def _classify(expected: list[ExpectedCall], calls: list[ToolCallRecord]) -> FailureType:
    if not expected:
        # A task may require no tool at all ("answer without a tool"): no
        # calls is exactly right; any call is a discipline failure — the
        # exact lesson of Phase 1 ("besok tanggal berapa?" answered from a
        # guess). WRONG_TOOL, not the tolerated EXTRA_TOOL.
        return FailureType.OK if not calls else FailureType.WRONG_TOOL
    if not calls:
        return FailureType.NO_TOOL_CALL
    exp_tools = {e.tool for e in expected}
    got_tools = {c.tool for c in calls}
    if not (exp_tools & got_tools):
        return FailureType.WRONG_TOOL
    # A complete args-aware embedding wins even when extra calls appeared
    # first, including extra calls to the same tool with other arguments.
    # The extra call must not steal the slot of a required one.
    ordered = _match_subsequence(expected, calls, require_args=True)
    if len(ordered) == len(expected):
        return FailureType.EXTRA_TOOL if len(calls) > len(expected) else FailureType.OK
    matched = _match_subsequence(expected, calls)
    if len(matched) < len(expected):
        # Incomplete subsequence. MISSING_TOOL when some expected call never
        # happened (the tool fired fewer times than expected — includes the
        # duplicate-tools case, where the old scorer mislabelled this
        # ORDER_ERROR). ORDER_ERROR only when every expected call exists with
        # the right multiplicity but cannot be arranged as a subsequence.
        for e in expected:
            exp_n = sum(1 for x in expected if x.tool == e.tool)
            got_n = sum(1 for c in calls if c.tool == e.tool)
            if got_n < exp_n:
                return FailureType.MISSING_TOOL
        return FailureType.ORDER_ERROR
    if not all(_args_ok(e.args, g.args) for e, g in matched):
        if _unordered_arg_match(expected, calls) is not None:
            return FailureType.ORDER_ERROR
        return FailureType.BAD_ARGS


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

    # tool_selection: fraction of expected calls matched at all — derived from
    # the same greedy matching as the failure label, so duplicate expected
    # tools are counted per-occurrence, not per-unique-name.
    matched = _match_subsequence(expected, calls)
    if expected:
        tool_selection = len(matched) / len(expected)
    else:
        tool_selection = 1.0 if not calls else 0.0

    # No-tool task (expected == []): the signal is abstention, not prose.
    # `answer_any_of` is a reference reply (used by the scripted baseline);
    # any non-empty answer passes, an empty one is WRONG_ANSWER.
    if not expected and failure == FailureType.OK and not traj.final_text.strip():
        failure = FailureType.WRONG_ANSWER

    # arg_accuracy: matched args over all expected args for tool tasks; for
    # no-tool tasks it mirrors the abstention outcome.
    pairs = matched
    if expected:
        ordered = _match_subsequence(expected, calls, require_args=True)
        if len(ordered) == len(expected):
            pairs = ordered
        elif failure == FailureType.ORDER_ERROR:
            pairs = _unordered_arg_match(expected, calls) or matched
    m = 0
    for e, g in pairs:
        m += sum(1 for k, v in e.args.items()
                 if k in g.args and _arg_matches(v, g.args[k]))
    if expected:
        # Denominator covers every expected arg, not only the args of calls
        # that happened to match: no calls or a missing call must not look
        # like perfect argument accuracy.
        total = sum(len(e.args) for e in expected)
        arg_accuracy = m / total if total else 1.0
    else:
        arg_accuracy = 1.0 if failure == FailureType.OK else 0.0

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
        # Keep the shape stable so callers (e.g. the CLI) can index every key.
        return {"tasks": 0, "exact_match_rate": 0.0, "tool_selection": 0.0,
                "arg_accuracy": 0.0, "avg_steps": 0.0, "failures": {}}
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
