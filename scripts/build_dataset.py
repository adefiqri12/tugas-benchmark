"""Compile the hand-authored raw task sources into the golden dataset JSON.

Source of truth: ``eval/datasets/raw/*.txt`` (read in sorted filename order).
Output: ``eval/datasets/golden_id.json``, the canonical dataset that also gets
force-included into the wheel (see ``pyproject.toml``).

Usage::

    .venv/bin/python scripts/build_dataset.py            # regenerate
    .venv/bin/python scripts/build_dataset.py --check    # exit 1 if stale
    .venv/bin/python scripts/build_dataset.py --out /tmp/x.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # runnable straight from a checkout
    sys.path.insert(0, str(ROOT / "src"))

from tugas.raw_tasks import (  # noqa: E402
    RawTaskError,
    build_document,
    parse_raw,
)

RAW_DIR = ROOT / "eval/datasets/raw"
DEFAULT_OUT = ROOT / "eval/datasets/golden_id.json"


def source_paths(raw_dir: str | Path = RAW_DIR) -> list[Path]:
    """Every ``*.txt`` source under ``raw_dir``, in sorted filename order."""
    paths = sorted(Path(raw_dir).glob("*.txt"))
    if not paths:
        raise RawTaskError(f"tidak ada sumber *.txt di {raw_dir}")
    return paths


def source_label(raw_dir: str | Path = RAW_DIR) -> str:
    """Stable ``meta.source`` label: the source glob relative to the repo."""
    raw_path = Path(raw_dir)
    try:
        return raw_path.resolve().relative_to(ROOT).as_posix() + "/*.txt"
    except ValueError:
        return raw_path.as_posix() + "/*.txt"


def build_tasks(raw_dir: str | Path = RAW_DIR) -> list[dict]:
    """Parse every source, rejecting ids that collide across files."""
    tasks: list[dict] = []
    origin: dict[str, str] = {}
    for path in source_paths(raw_dir):
        for task in parse_raw(path.read_text(encoding="utf-8")):
            seen = origin.get(task["id"])
            if seen is not None:
                raise RawTaskError(
                    f"id duplikat lintas sumber: {task['id']!r} "
                    f"({seen} dan {path.name})"
                )
            origin[task["id"]] = path.name
            tasks.append(task)
    return tasks


def build_dataset(raw_dir: str | Path = RAW_DIR) -> dict:
    """Build the full golden document from raw sources."""
    return build_document(build_tasks(raw_dir), source_label(raw_dir))


def render(doc: dict) -> str:
    """Serialise a document exactly as the canonical file is stored."""
    return json.dumps(doc, ensure_ascii=False, indent=2) + "\n"


def check_dataset(raw_dir: str | Path = RAW_DIR, out: str | Path = DEFAULT_OUT) -> str | None:
    """Return ``None`` when ``out`` matches the sources, else why it is stale."""
    out_path = Path(out)
    expected = render(build_dataset(raw_dir))
    if not out_path.is_file():
        return f"stale: {out_path} belum ada; jalankan scripts/build_dataset.py"
    actual = out_path.read_text(encoding="utf-8")
    if actual != expected:
        return (
            f"stale: {out_path} tidak cocok dengan sumber di {source_label(raw_dir)}; "
            "jalankan .venv/bin/python scripts/build_dataset.py"
        )
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare instead of writing")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output JSON path")
    parser.add_argument("--raw-dir", default=str(RAW_DIR), help="directory of *.txt sources")
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    if args.check:
        problem = check_dataset(args.raw_dir, out_path)
        if problem:
            print(problem, file=sys.stderr)
            return 1
        count = len(build_dataset(args.raw_dir)["tasks"])
        print(f"ok: {out_path} cocok dengan sumber ({count} tasks)")
        return 0

    doc = build_dataset(args.raw_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(doc), encoding="utf-8")
    print(f"wrote {out_path} ({len(doc['tasks'])} tasks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())