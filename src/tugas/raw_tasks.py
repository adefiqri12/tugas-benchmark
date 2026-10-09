"""Parse hand-authored raw-text no-tool tasks into golden JSON.

The raw file (``eval/datasets/no_tool_id.txt``) is the human staging area: one
prompt per entry plus the accepted final answers. ``parse_raw`` turns it into
task dicts shaped like ``golden_id.json`` entries, adding ``answer_any_of`` (the
accepted answers). ``scripts/build_no_tool.py`` wraps this into a CLI.

Format (see the raw file's header for the authoring guide):

- entries separated by a line containing only ``---``
- ``id:``, ``category:``, ``Q:`` (prompt), ``A:`` (answer, repeatable),
  ``N:`` (notes)
- blank lines and lines starting with ``#`` are ignored
"""

from __future__ import annotations

ANCHOR_DATE = "2026-10-07"
_KEYS = {"id", "category", "Q", "A", "N"}


class RawTaskError(ValueError):
    """A malformed entry in the raw task file."""


def _iter_blocks(text: str):
    """Yield one list-of-lines per entry (split on a lone ``---``)."""
    block: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "---":
            if block:
                yield block
                block = []
            continue
        block.append(line)
    if block:
        yield block


def _parse_block(block: list[str]) -> dict:
    fields: dict = {
        "id": "", "category": "no-tool", "prompt": "", "answers": [], "notes": "",
    }
    for line in block:
        key, sep, val = line.partition(":")
        if not sep:
            raise RawTaskError(f"baris tanpa ':' -> {line!r}")
        key, val = key.strip(), val.strip()
        if key not in _KEYS:
            raise RawTaskError(f"key tidak dikenal {key!r} (baris: {line!r})")
        if key == "A":
            fields["answers"].append(val)
        elif key == "Q":
            fields["prompt"] = val
        elif key == "id":
            fields["id"] = val
        elif key == "category":
            fields["category"] = val
        elif key == "N":
            fields["notes"] = val

    if not fields["id"]:
        raise RawTaskError(f"entri tanpa 'id' (blok dimulai: {block[0]!r})")
    if not fields["prompt"]:
        raise RawTaskError(f"entri {fields['id']!r} tanpa 'Q'")
    if not fields["answers"]:
        raise RawTaskError(f"entri {fields['id']!r} tanpa 'A'")

    return {
        "id": fields["id"],
        "category": fields["category"],
        "prompt": fields["prompt"],
        "expected": [],
        "answer_any_of": fields["answers"],
        "notes": fields["notes"],
    }


def parse_raw(text: str) -> list[dict]:
    """Parse raw-text task entries into golden task dicts (no-tool)."""
    tasks = [_parse_block(b) for b in _iter_blocks(text)]
    ids = [t["id"] for t in tasks]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise RawTaskError(f"id duplikat: {dupes}")
    return tasks


def build_document(tasks: list[dict], source_name: str) -> dict:
    """Wrap parsed tasks in the golden-dataset envelope (matches golden_id.json)."""
    return {
        "meta": {
            "name": "tugas golden (no-tool)",
            "language": "id",
            "anchor_date": ANCHOR_DATE,
            "note": (
                f"Generated from {source_name} by scripts/build_no_tool.py. "
                "expected=[] means a correct agent calls no tool; answer_any_of "
                "lists the accepted final answers (any-of, normalized)."
            ),
        },
        "tasks": tasks,
    }
