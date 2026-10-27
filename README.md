# TUGAS

**Tool-Use and Grounded Agent Score**: a tool-calling agent benchmark for Bahasa Indonesia.

[![ci](https://github.com/adefiqri12/tugas-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/adefiqri12/tugas-benchmark/actions/workflows/ci.yml)

Most Indonesian-language benchmarks (SEA-HELM, IndoNLU, IndoMMLU, Cendol) measure
language *understanding*. Almost none measure whether an agent **picks the right
tool, with the right arguments, in the right order** when the user speaks
Indonesian. TUGAS scores the tool-call trajectory, not just the final text.

The tasks target conventions that transfer poorly from English:

- relative dates (`lusa` is +2 days, `minggu depan` is +7),
- DD/MM numeric dates (`7/10` is 7 October),
- local units (`kodi` is 20),
- reversed address order (RT/RW, kelurahan, kecamatan, kota),
- greeting time (after maghrib it is `malam`, not `sore`),
- WhatsApp admin flows: stock, opening hours, discounts, bookings,
- owner analytics: sales, revenue, leads, conversion, best seller.

## Status

- Harness and dataset: **v0.1.0**, usable now.
- Leaderboard: **not live yet**. Scores are reproducible locally from the
  commands below; a public leaderboard is the next milestone.
- 51 hand-written tasks over 13 deterministic mock tools.
- 149 offline tests, no network required.

## What it measures

Given an Indonesian prompt, a correct agent must produce a specific tool-call
trace. The harness compares the trajectory with a hand-written golden trace:

| Metric | Question it answers |
|---|---|
| `exact_match` | Was the whole expected trace reproduced, in order? |
| `tool_selection` | Of the tools that should fire, how many fired at all? |
| `arg_accuracy` | Of the expected arguments, how many matched? |
| `steps` | How many tool calls did the agent actually make? |

Every failure is labelled: `NO_TOOL_CALL`, `WRONG_TOOL`, `MISSING_TOOL`,
`BAD_ARGS`, `WRONG_ANSWER`, `EXTRA_TOOL`, `ORDER_ERROR`, `API_ERROR`.

## Quickstart

```bash
git clone https://github.com/adefiqri12/tugas-benchmark
cd tugas-benchmark
uv venv && uv pip install -e ".[dev]"

.venv/bin/tugas list                        # 51 tasks
.venv/bin/python -m pytest -q               # offline tests
```

Score a model against a running OpenAI-compatible gateway with tool calling:

```bash
export TUGAS_BASE_URL=http://127.0.0.1:20129/v1
export TUGAS_API_KEY=...

.venv/bin/tugas run -m "auto/best-fast"
.venv/bin/tugas run -m model-a,model-b --out results/run.json
```

Defaults: `TUGAS_BASE_URL=http://127.0.0.1:20129/v1`, `TUGAS_API_KEY=""`.
Environment variable names are case-sensitive.

## Dataset

`eval/datasets/golden_id.json` is the canonical, shipped dataset:

- 49 tool-call tasks and 2 no-tool abstain tasks,
- 13 tools, all deterministic and offline,
- one fixed anchor date (`2026-10-07`) so relative dates never drift.

| Tool group | Tools |
|---|---|
| Dates and places | `parse_tanggal`, `sapaan_waktu`, `parse_alamat` |
| Travel and weather | `cari_penerbangan`, `cek_cuaca` |
| Calculation | `hitung` |
| Commerce | `cek_stok`, `jam_buka_toko`, `cek_diskon`, `cek_booking` |
| Reporting | `laporan_penjualan`, `laporan_pendapatan`, `laporan_leads` |

Tools return fixed mock data on purpose. A run measures the agent, not an
upstream API's mood, so two runs of the same model and gateway are comparable.

### Where the tasks come from

Hand-authored raw text is the source of truth:

```
eval/datasets/raw/
  core.txt       9 original tasks
  customer.txt   20 consumer to admin WhatsApp flows
  no_tool.txt    2 abstain tasks
  owner.txt      20 owner analytics flows
```

`eval/datasets/golden_id.json` is generated from those files:

```bash
.venv/bin/python scripts/build_dataset.py          # regenerate
.venv/bin/python scripts/build_dataset.py --check  # fail if stale
```

Each raw entry has an `id`, a `category`, a one-line prompt `Q`, and either
`T:` expected tool calls or `A:` accepted answers. `R:` holds the expected reply
for review only and is never compiled. To add a task, edit a raw file, run the
builder, and run the tests.

## Scoring rules

- Expected calls are matched as an **ordered subsequence** of the actual calls.
  Extra helper calls are tolerated and reported as `EXTRA_TOOL`.
- Only arguments listed in the golden trace are scored. Extra arguments the
  agent supplies are ignored.
- Argument matching normalizes thousands separators and operator spacing, so
  `"1.200.000 - 150.000"` matches `"1200000-150000"`. A golden value may be a
  list of accepted forms (any-of).
- A no-tool task passes only when the agent calls nothing and the final text
  matches an accepted answer after case, whitespace, and trailing-punctuation
  normalization. Calling any tool is `WRONG_TOOL`.

A golden trace is a scoring spec, not a replayable call. It may list fewer
arguments than the tool needs to execute (for example, `cek_cuaca` scores only
`kota`).

## Repository layout

```
src/tugas/
  anchor.py     fixed evaluation date
  models.py     Task, ExpectedCall, Trajectory, TaskResult, FailureType
  tools.py      13 deterministic mock tools + OpenAI schemas
  agent.py      LLMAgent (gateway) + ScriptedAgent (offline baseline)
  raw_tasks.py  raw-text task parser
  score.py      trajectory scoring + failure taxonomy
  runner.py     load tasks, run an agent
  cli.py        tugas list | run
eval/datasets/  raw sources + generated golden JSON
scripts/        dataset builder
tests/          offline tests only
```

## Limitations

- The tools are mocks. TUGAS measures tool selection, arguments, and ordering,
  not real API reliability.
- One anchor date keeps runs reproducible; tasks written for a different date
  need the anchor updated in `src/tugas/anchor.py`.
- No-tool answers are matched exactly (any-of, normalized). Very free-form
  replies can be marked `WRONG_ANSWER` even when the intent is right.
- Results depend on the gateway's tool-calling implementation and model version.
  Record both when reporting scores.

## Contributing

Issues and pull requests are welcome. Before opening a PR:

```bash
.venv/bin/python scripts/build_dataset.py --check
.venv/bin/python -m pytest -q
```

Tests must stay offline. A test that needs a gateway will hang CI.

## Citation

```bibtex
@software{tugas2026,
  title  = {TUGAS: Tool-Use and Grounded Agent Score},
  author = {Tjikoa, Ade Fiqri},
  year   = {2026},
  url    = {https://github.com/adefiqri12/tugas-benchmark}
}
```

## License

MIT. See [LICENSE](LICENSE).
