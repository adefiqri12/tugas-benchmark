# TUGAS

TUGAS (Tool-Use and Grounded Agent Score) — eval harness + planned leaderboard
for tool-calling AI agents in Bahasa Indonesia. Scores an agent's tool-call
*trajectory* (tool choice, args, order) against hand-written golden traces, not
just final text. Phase 1 (scaffold) is done; roadmap lives in
`planning/roadmap.md` (local, gitignored — not shipped in the repo). Task
backlog: `planning/tasks.md`.

## Dev environment

Python >=3.11, src layout (`src/tugas/`), hatchling. Toolchain is uv:

```bash
uv venv && uv pip install -e ".[dev]"
```

## Commands

```bash
.venv/bin/tugas list                       # show golden tasks
.venv/bin/tugas run -m "auto/best-fast"    # score a model (comma-sep ids ok)
.venv/bin/tugas run -m m1,m2 --out results/run.json
.venv/bin/python -m pytest -q              # offline scorer tests
```

`tugas run` needs a running OpenAI-compatible gateway with tool calling:

- `TUGAS_BASE_URL` (default `http://127.0.0.1:20129/v1`)
- `TUGAS_API_KEY`

Tests are fully offline (ScriptedAgent) and never touch the gateway.

## Layout & conventions

- `src/tugas/`: `models.py` (pydantic: Task, ExpectedCall, Trajectory, ...),
  `tools.py` (4 deterministic mock tools), `agent.py` (LLMAgent + ScriptedAgent
  behind one `Agent` Protocol), `score.py` (scoring + failure taxonomy),
  `runner.py`, `cli.py` (typer).
- `eval/datasets/golden_id.json` — golden tasks; ground truth is hand-written.
  Canonical on disk; also force-included into the wheel as
  `tugas/data/golden_id.json` so an installed `tugas` finds it from any cwd
  (`cli.default_tasks_path()`).
- Data models are pydantic v2 `BaseModel`; enums are `str, Enum`.
- Code and docstrings in English; task prompts, tool descriptions, and the
  agent system prompt are in Indonesian.
- A new tool = `ToolSpec` in `tools.py` + implementation; it must stay
  deterministic and offline.
- Commits: short lowercase prefix (`docs:`, `rename:`, `P1:`).

## Design rules — do not break

- Tools are deterministic and offline on purpose: runs must be reproducible;
  the score measures the agent, not an upstream API's mood.
- Ground truth is hand-written, never model-generated.
- Scoring matches the expected trace as an *ordered subsequence* of the actual
  calls: extra helper calls are EXTRA_TOOL (tolerated), not failures. Don't
  tighten this to exact-list matching — it's deliberate.
- Arg matching normalizes thousands separators and operator spacing
  (`"1.200.000 - 150.000"` == `"1200000-150000"`); only args listed in the
  golden JSON are scored, and extra args the agent supplies are ignored. A
  golden call may therefore list fewer args than the tool needs to execute
  (e.g. `cek_cuaca` scores only `kota`, not the required dates), so a golden
  trace is a scoring spec, not a replayable call.

## Pitfalls

- `TUGAS_API_KEY` / `TUGAS_BASE_URL` are case-sensitive.
- Relative-date tasks ("minggu depan") are pinned to the anchor date
  2026-10-07, which lives in TWO places: `tools.py` `_DEFAULT_TODAY` and
  `golden_id.json` `meta.anchor_date`. Change them together or scoring breaks
  silently.
- `results/` is gitignored; that's where `--out` run outputs belong.
