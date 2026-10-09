#!/usr/bin/env python3
"""Convert eval/datasets/no_tool_id.txt (raw text) into no_tool_id.json.

Author entries by hand in the .txt file, then run from the repo root:

    .venv/bin/python scripts/build_no_tool.py
    .venv/bin/python scripts/build_no_tool.py --src <in.txt> --out <out.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tugas.raw_tasks import RawTaskError, build_document, parse_raw

DEFAULT_SRC = "eval/datasets/no_tool_id.txt"
DEFAULT_OUT = "eval/datasets/no_tool_id.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    src = Path(args.src)
    try:
        text = src.read_text(encoding="utf-8")
        tasks = parse_raw(text)
    except (OSError, RawTaskError) as exc:
        print(f"error: {exc}")
        return 1

    doc = build_document(tasks, src.name)
    out = Path(args.out)
    out.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(tasks)} task(s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
