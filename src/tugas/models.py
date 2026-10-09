"""Core typed objects for the Indonesian agent-eval harness.

A *task* is one Indonesian prompt that requires the agent to call tools.
An *expected* trace is the tool call(s) a correct agent must produce.
The harness scores the agent's actual trajectory against that expectation.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class FailureType(str, Enum):
    """How a trajectory went wrong. Named so the report can count them."""

    OK = "OK"
    NO_TOOL_CALL = "NO_TOOL_CALL"          # answered without calling anything
    WRONG_TOOL = "WRONG_TOOL"              # called a tool, but not the expected one
    MISSING_TOOL = "MISSING_TOOL"          # a required call never happened
    BAD_ARGS = "BAD_ARGS"                  # right tool, wrong argument value
    WRONG_ANSWER = "WRONG_ANSWER"          # no-tool task: abstained but text wrong
    # Reserved: kept for schema stability, but not emitted by the scorer. Extra
    # or unknown args are deliberately ignored (a golden call may list fewer args
    # than the tool needs), so "hallucinated arg" is not a scored failure.
    HALLUCINATED_ARG = "HALLUCINATED_ARG"
    EXTRA_TOOL = "EXTRA_TOOL"              # called an unexpected extra tool
    ORDER_ERROR = "ORDER_ERROR"            # right calls, wrong order
    API_ERROR = "API_ERROR"                # provider/transport failure


class ToolSpec(BaseModel):
    """A tool the agent may call, described for the model + for execution."""

    name: str
    description: str
    parameters: dict[str, str] = Field(default_factory=dict)  # arg -> type
    required: list[str] = Field(default_factory=list)


class ExpectedCall(BaseModel):
    """One tool call a correct solution must contain."""

    tool: str
    args: dict[str, object] = Field(default_factory=dict)


class Task(BaseModel):
    """One Indonesian prompt with its expected tool-call trace."""

    id: str
    prompt: str
    expected: list[ExpectedCall]
    # For no-tool tasks (expected == []): accepted final answers, any-of,
    # matched after normalization. A non-matching answer is WRONG_ANSWER.
    answer_any_of: list[str] = Field(default_factory=list)
    category: str = "general"
    notes: str = ""


class ToolCallRecord(BaseModel):
    """A single tool call the agent actually produced (parsed)."""

    tool: str
    args: dict[str, object] = Field(default_factory=dict)


class Trajectory(BaseModel):
    """What the agent did on one task."""

    calls: list[ToolCallRecord] = Field(default_factory=list)
    final_text: str = ""
    error: str | None = None


class TaskResult(BaseModel):
    """Scored outcome for one task under one model."""

    task_id: str
    model: str
    failure: FailureType
    tool_selection: float   # fraction of expected tools that were called at all
    arg_accuracy: float     # fraction of expected args matched (over expected calls)
    exact_match: bool       # whole expected trace reproduced in order
    steps: int              # number of tool calls the agent made
    note: str = ""
