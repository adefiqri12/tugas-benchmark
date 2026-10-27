# TUGAS

TUGAS (Tool-Use and Grounded Agent Score) is an eval harness and planned
leaderboard for tool-calling AI agents in Bahasa Indonesia. It scores an agent's
tool-call *trajectory* (tool choice, arguments, order) against hand-written
golden traces, not just final text.

Current release: 51 tasks over 13 deterministic offline tools. Roadmap lives in
`planning/roadmap.md` (local, gitignored, not shipped). Task backlog:
`planning/tasks.md`; session specs: `planning/SPECS.md`. Personal notes live in
`private/` (gitignored; never publish or quote their contents).

## Dev environment

Python >=3.11, src layout (`src/tugas/`), hatchling. Toolchain is uv:

```bash
uv venv && uv pip install -e ".[dev]"
```

## Commands

```bash
.venv/bin/tugas list                       # show the 51 golden tasks
.venv/bin/tugas run -m "auto/best-fast"    # score a model (comma-sep ids ok)
.venv/bin/tugas run -m m1,m2 --out results/run.json
.venv/bin/python -m pytest -q              # offline tests only
.venv/bin/python scripts/build_dataset.py  # raw sources -> golden JSON
.venv/bin/python scripts/build_dataset.py --check  # CI drift check
```

`tugas run` needs a running OpenAI-compatible gateway with tool calling:

- `TUGAS_BASE_URL` (default `http://127.0.0.1:20129/v1`)
- `TUGAS_API_KEY`

Tests are fully offline (ScriptedAgent) and never touch the gateway.

CI (.github/workflows/ci.yml) runs the suite on Python 3.11/3.12/3.13, checks
that `golden_id.json` matches the raw sources, and smoke-tests the built wheel
from an arbitrary working directory. Local tests must stay offline: a
gateway-dependent test would hang CI.

## Layout and conventions

- `src/tugas/anchor.py`: the single fixed evaluation date (`2026-10-07`).
- `src/tugas/models.py`: pydantic objects (`Task`, `ExpectedCall`, `Trajectory`,
  `TaskResult`, `FailureType`, ...).
- `src/tugas/tools.py`: 13 `ToolSpec` entries plus deterministic implementations.
- `src/tugas/agent.py`: `LLMAgent` and `ScriptedAgent` behind one `Agent`
  protocol. The system prompt permits abstention, so no-tool tasks are fair.
- `src/tugas/raw_tasks.py`: parser for the hand-authored raw task format.
- `src/tugas/score.py`: scoring and failure taxonomy.
- `src/tugas/runner.py`, `src/tugas/cli.py`: loading and the typer CLI.
- `eval/datasets/raw/*.txt`: source of truth, hand-authored tasks.
- `eval/datasets/golden_id.json`: generated canonical dataset; force-included
  into the wheel as `tugas/data/golden_id.json`
  (`cli.default_tasks_path()` finds it from any cwd).
- `scripts/build_dataset.py`: compiles raw sources into the JSON.
- Data models are pydantic v2 `BaseModel`; enums are `str, Enum`.
- Code and docstrings are in English. Task prompts, tool descriptions, tool
  results, and the agent system prompt are in Indonesian.
- A new tool is a `ToolSpec` in `tools.py` plus an implementation, a registry
  entry, and a deterministic offline data table. No network calls.
- Commits: short lowercase prefix (`docs:`, `fix:`, `P1:`, `P2:`).

## Task format

Entries in `eval/datasets/raw/*.txt` are separated by a lone `---`:

```
id: wa-stok-kaos
category: wa/barang
Q: Halo kak, kaos polos warna hitam ukuran L masih ready?
T: cek_stok(barang="kaos polos"; warna="hitam"; ukuran="L")
R: Ready kak! Kaos polos hitam ukuran L masih ada stok.
N: Availability, with variant extraction from one casual line.
```

Use `T:` for expected tool calls or `A:` for reference no-tool answers, never
both. `R:` is the expected reply for human review and is not compiled. Add a
task, run the builder, then run the tests. `tests/test_dataset.py` enforces
unique ids, known tools, declared argument names, full tool coverage, and that
every fully specified expected call executes.

## Design rules: do not break

- Tools are deterministic and offline on purpose: runs must be reproducible;
  the score measures the agent, not an upstream API's mood.
- Ground truth is hand-written, never model-generated.
- Scoring matches the expected trace as an *ordered subsequence* of the actual
  calls: extra helper calls are `EXTRA_TOOL` (tolerated), not failures. Do not
  tighten this to exact-list matching.
- Argument matching normalizes case, thousands separators, and operator spacing
  (`"1.200.000 - 150.000"` equals `"1200000-150000"`). Only arguments listed in
  the golden trace are scored; extra arguments are ignored. A golden call may
  list fewer arguments than the tool needs to execute, so a golden trace is a
  scoring spec, not a replayable call.
- A no-tool task (`expected == []`) passes when no tool is called and the
  final text is non-empty. `answer_any_of` holds reference answers for the
  offline baseline; text quality is not scored.
- `golden_id.json` is generated. Never edit it by hand; edit
  `eval/datasets/raw/*.txt` and rerun the builder.

## Pitfalls

- `TUGAS_API_KEY` and `TUGAS_BASE_URL` are case-sensitive.
- Relative dates are pinned to `2026-10-07` in `src/tugas/anchor.py`. The
  dataset meta is generated from that constant; tools parse against it. Change
  it in one place, then rebuild and expect date tasks to shift.
- `hitung` rejects `**` on purpose: `9**9**9` builds a 369-million-digit int
  and hangs the process. Do not "fix" this to full Python eval.
- `results/` and `dist/` are gitignored; run outputs and build artifacts do not
  belong in the repo.
