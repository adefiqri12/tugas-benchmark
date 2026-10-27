# Golden dataset

`golden_id.json` is generated from the hand-authored files in `raw/`. Do not
edit the JSON by hand.

```bash
.venv/bin/python scripts/build_dataset.py
.venv/bin/python scripts/build_dataset.py --check
```

## Files

| File | Contents |
|---|---|
| `raw/core.txt` | 9 core tool-calling tasks (dates, units, address, greeting) |
| `raw/customer.txt` | 20 consumer to admin WhatsApp flows |
| `raw/owner.txt` | 20 owner analytics flows |
| `raw/no_tool.txt` | 2 tasks where the correct move is to call no tool |
| `golden_id.json` | generated canonical dataset, 51 tasks |

## JSON shape

```json
{
  "meta": {
    "name": "tugas golden",
    "language": "id",
    "anchor_date": "2026-10-07",
    "source": "eval/datasets/raw/*.txt",
    "note": "..."
  },
  "tasks": [
    {
      "id": "wa-stok-kaos",
      "category": "wa/barang",
      "prompt": "Halo kak, kaos polos warna hitam ukuran L masih ready?",
      "expected": [
        {"tool": "cek_stok", "args": {"barang": "kaos polos", "warna": "hitam", "ukuran": "L"}}
      ],
      "notes": "..."
    },
    {
      "id": "id-abstain-sapaan",
      "category": "no-tool/abstain",
      "prompt": "Halo, apa kabar?",
      "expected": [],
      "answer_any_of": ["Baik, terima kasih", "Baik. Ada yang bisa dibantu?"],
      "notes": "..."
    }
  ]
}
```

A task has exactly one of:

- `expected`: one or more tool calls that must appear, in order, in the agent's
  trajectory. Only the listed arguments are scored.
- `answer_any_of`: reference final answers for a no-tool task, replayed by the
  offline baseline. Scoring checks only that no tool was called and the reply
  is non-empty.

An expected argument value may be a list of accepted forms (any-of), for example
`"ekspresi": ["2 * 20 - 3", "40 - 3"]`.

## Raw format

Entries are separated by a line containing only `---`. Within an entry:

- `id:` unique slug (required)
- `category:` free-form category (optional, default `general`)
- `Q:` the prompt, one line (required)
- `T:` one expected tool call, repeatable (required for tool tasks)
- `A:` one reference answer, repeatable (required for no-tool tasks)
- `R:` expected reply, review only, never compiled
- `N:` notes (optional)

Tool calls use `tool(kunci="nilai"; kunci2="nilai2")`, or `tool()` for no
arguments. Values are double-quoted strings; a value may also be a list of
strings for any-of argument matching. An entry may not contain both `T` and `A`.
Blank lines and lines starting with `#` are ignored. A repeated key inside one
entry and a repeated `id` across the file are both hard errors.

All dates assume the anchor date `2026-10-07` from `src/tugas/anchor.py`.
