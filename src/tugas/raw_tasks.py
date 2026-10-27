"""Parse hand-authored raw-text golden tasks into the canonical JSON document.

Raw sources live in ``eval/datasets/raw/*.txt`` and are compiled into
``eval/datasets/golden_id.json`` by ``scripts/build_dataset.py``. Keeping the
hand-authored text as the source of truth means prompt/notes edits never touch
JSON syntax.

Format
------
Entries are separated by a line containing only ``---``. Inside an entry:

- ``id:``       slug, unique across all sources (required)
- ``category:`` category string (optional, default ``"general"``)
- ``Q:``        the prompt, one line (required)
- ``T:``        one expected tool call, repeatable and ordered (tool entries)
- ``A:``        one accepted final answer, repeatable (no-tool entries)
- ``R:``        expected reply text, review only: kept in the raw file, never
                emitted into the JSON
- ``N:``        notes (optional)

An entry uses either ``T:`` lines (>=1) or ``A:`` lines (>=1), never both.

``T:`` syntax::

    tool()
    tool(kunci="nilai"; kunci2="nilai2")

Values are double-quoted strings with backslash escapes, decoded with
``ast.literal_eval``. A value may also be a list of such strings, which the
scorer treats as any-of accepted arguments::

    hitung(ekspresi=["2 * 20 - 3", "40 - 3"])

Blank lines and lines starting with ``#`` are ignored.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from typing import NoReturn

from .anchor import ANCHOR_DATE

_KEYS = {"id", "category", "Q", "T", "A", "R", "N"}
_TOOL_RE = re.compile(r"[a-z][a-z0-9_]*")
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class RawTaskError(ValueError):
    """A malformed entry in a raw task source."""


class _Scanner:
    """Cursor over one ``tool(...)`` expression."""

    def __init__(self, text: str, where: str) -> None:
        self.text = text
        self.where = where
        self.i = 0

    def error(self, message: str) -> NoReturn:
        raise RawTaskError(f"{self.where}: {message} (teks: {self.text!r})")

    def skip_ws(self) -> None:
        while self.i < len(self.text) and self.text[self.i].isspace():
            self.i += 1

    def peek(self) -> str:
        return self.text[self.i] if self.i < len(self.text) else ""

    def bump(self) -> None:
        self.i += 1

    def scan_string(self) -> str:
        """Scan one double-quoted string, returning its source slice."""
        start = self.i
        self.bump()  # opening quote
        while True:
            ch = self.peek()
            if ch == "":
                self.error("string tidak ditutup '\"'")
            if ch == "\\":
                self.bump()
                if self.peek() == "":
                    self.error("escape terpotong di akhir baris")
                self.bump()
                continue
            self.bump()
            if ch == '"':
                return self.text[start:self.i]

    def scan_list(self) -> str:
        """Scan a bracketed list of strings, returning its source slice."""
        start = self.i
        depth = 0
        while True:
            ch = self.peek()
            if ch == "":
                self.error("list nilai tidak ditutup ']'")
            if ch == '"':
                self.scan_string()
                continue
            self.bump()
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return self.text[start:self.i]

    def scan_value(self) -> str:
        ch = self.peek()
        if ch == '"':
            return self.scan_string()
        if ch == "[":
            return self.scan_list()
        self.error('nilai argumen harus diapit tanda kutip ganda, mis. kunci="nilai"')


def _decode_value(src: str, scanner: _Scanner, key: str) -> object:
    """Decode one argument value with ``ast.literal_eval``.

    Allowed shapes: a string, or a non-empty list of strings (any-of match).
    """
    try:
        value = ast.literal_eval(src)
    except (ValueError, SyntaxError):
        scanner.error(f"nilai untuk {key!r} tidak bisa dibaca: {src!r}")
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
        return value
    scanner.error(f"nilai untuk {key!r} harus string atau list string: {src!r}")


def _parse_call(text: str, where: str) -> dict:
    """Parse ``tool(kunci="nilai"; ...)`` into an expected-call dict."""
    stripped = text.strip()
    match = _TOOL_RE.match(stripped)
    if not match:
        raise RawTaskError(f"{where}: nama tool harus [a-z][a-z0-9_]* -> {text!r}")
    tool = match.group(0)
    sc = _Scanner(stripped, where)
    sc.i = match.end()
    sc.skip_ws()
    if sc.peek() != "(":
        sc.error("tool call harus diikuti '('")
    sc.bump()

    args: dict[str, object] = {}
    sc.skip_ws()
    if sc.peek() == ")":
        sc.bump()
    else:
        while True:
            sc.skip_ws()
            key_match = _NAME_RE.match(sc.text, sc.i)
            if not key_match:
                sc.error('argumen harus berbentuk kunci="nilai"')
            key = key_match.group(0)
            sc.i = key_match.end()
            sc.skip_ws()
            if sc.peek() != "=":
                sc.error(f"argumen {key!r} harus diikuti '='")
            sc.bump()
            sc.skip_ws()
            src = sc.scan_value()
            if key in args:
                sc.error(f"argumen {key!r} duplikat")
            args[key] = _decode_value(src, sc, key)
            sc.skip_ws()
            ch = sc.peek()
            if ch == ";":
                sc.bump()
                continue
            if ch == ")":
                sc.bump()
                break
            sc.error("argumen harus dipisah ';' dan ditutup ')'")

    sc.skip_ws()
    if sc.i != len(sc.text):
        sc.error("ada teks sisa setelah ')'")
    return {"tool": tool, "args": args}


def _iter_blocks(text: str):
    """Yield one list of ``(lineno, line)`` per entry (split on a lone ``---``)."""
    block: list[tuple[int, str]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "---":
            if block:
                yield block
                block = []
            continue
        block.append((lineno, line))
    if block:
        yield block


def _parse_block(block: list[tuple[int, str]]) -> dict:
    entry_id = ""
    category = "general"
    prompt = ""
    notes = ""
    calls: list[dict] = []
    answers: list[str] = []
    seen: set[str] = set()

    def where(lineno: int) -> str:
        if entry_id:
            return f"entri {entry_id!r} baris {lineno}"
        return f"baris {lineno}"

    for lineno, line in block:
        key, sep, val = line.partition(":")
        if not sep:
            raise RawTaskError(f"{where(lineno)}: baris tanpa ':' -> {line!r}")
        key, val = key.strip(), val.strip()
        if key not in _KEYS:
            raise RawTaskError(f"{where(lineno)}: key tidak dikenal {key!r}")

        if key in {"T", "A"}:
            if not val:
                raise RawTaskError(f"{where(lineno)}: '{key}:' kosong")
            if key == "T":
                calls.append(_parse_call(val, where(lineno)))
            else:
                answers.append(val)
            continue

        if key in seen:
            raise RawTaskError(f"{where(lineno)}: key {key!r} duplikat")
        seen.add(key)

        if key == "id":
            if not val:
                raise RawTaskError(f"{where(lineno)}: 'id:' kosong")
            entry_id = val
        elif key == "category":
            category = val or "general"
        elif key == "Q":
            prompt = val
        elif key == "N":
            notes = val
        # key == "R": review-only, deliberately dropped.

    if not entry_id:
        raise RawTaskError(
            f"baris {block[0][0]}: entri tanpa 'id' (blok dimulai: {block[0][1]!r})"
        )
    if not prompt:
        raise RawTaskError(f"entri {entry_id!r} tanpa 'Q'")
    if calls and answers:
        raise RawTaskError(
            f"entri {entry_id!r}: pakai 'T:' atau 'A:', tidak keduanya"
        )
    if not calls and not answers:
        raise RawTaskError(f"entri {entry_id!r}: butuh minimal satu 'T:' atau 'A:'")

    return {
        "id": entry_id,
        "category": category,
        "prompt": prompt,
        "expected": calls,
        "answer_any_of": answers,
        "notes": notes,
    }


def parse_raw(text: str) -> list[dict]:
    """Parse raw-text task entries into golden task dicts."""
    tasks = [_parse_block(block) for block in _iter_blocks(text)]
    dupes = sorted(i for i, n in Counter(t["id"] for t in tasks).items() if n > 1)
    if dupes:
        raise RawTaskError(f"id duplikat: {dupes}")
    return tasks


_META_NOTE = (
    "Generated from eval/datasets/raw/*.txt by scripts/build_dataset.py; do not "
    "edit by hand. expected = the tool call(s) a correct agent must produce, "
    "matched as an ordered subsequence; only the args listed are scored. For "
    "no-tool tasks expected == [] and answer_any_of lists the accepted final "
    "answers (any-of, normalized)."
)


def _wire_task(task: dict) -> dict:
    """Render one task in JSON shape (empty answer_any_of is dropped)."""
    out = {
        "id": task["id"],
        "category": task["category"],
        "prompt": task["prompt"],
        "expected": task["expected"],
    }
    if task["answer_any_of"]:
        out["answer_any_of"] = task["answer_any_of"]
    if task["notes"]:
        out["notes"] = task["notes"]
    return out


def build_document(tasks: list[dict], source: str = "eval/datasets/raw/*.txt") -> dict:
    """Wrap parsed tasks in the golden-dataset envelope (the shipped JSON)."""
    return {
        "meta": {
            "name": "tugas golden",
            "language": "id",
            "anchor_date": ANCHOR_DATE,
            "source": source,
            "note": _META_NOTE,
        },
        "tasks": [_wire_task(task) for task in tasks],
    }
