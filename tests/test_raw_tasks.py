"""Raw-text no-tool task parser tests — no network."""

from __future__ import annotations

from pathlib import Path

import pytest

from tugas.raw_tasks import RawTaskError, build_document, parse_raw

SAMPLE = """
# a comment line
id: a
category: no-tool/x
Q: Halo?
A: Baik
A: Hai

---

id: b
Q: Teh atau kopi?
A: Tergantung selera
N: subjective
"""


def test_parse_basic():
    tasks = parse_raw(SAMPLE)
    assert [t["id"] for t in tasks] == ["a", "b"]
    assert tasks[0]["expected"] == []
    assert tasks[0]["answer_any_of"] == ["Baik", "Hai"]
    assert tasks[0]["category"] == "no-tool/x"
    assert tasks[1]["category"] == "no-tool"  # default
    assert tasks[1]["notes"] == "subjective"


def test_duplicate_id_rejected():
    with pytest.raises(RawTaskError, match="duplikat"):
        parse_raw("id: a\nQ: x\nA: y\n---\nid: a\nQ: z\nA: w\n")


def test_missing_answer_rejected():
    with pytest.raises(RawTaskError, match="tanpa 'A'"):
        parse_raw("id: a\nQ: x\n")


def test_unknown_key_rejected():
    with pytest.raises(RawTaskError, match="key tidak dikenal"):
        parse_raw("id: a\nQ: x\nA: y\nBOGUS: z\n")


def test_build_document_envelope():
    doc = build_document(parse_raw(SAMPLE), "sample.txt")
    assert doc["meta"]["anchor_date"] == "2026-10-07"
    assert len(doc["tasks"]) == 2


def test_real_source_file_parses():
    path = Path(__file__).resolve().parents[1] / "eval/datasets/no_tool_id.txt"
    tasks = parse_raw(path.read_text(encoding="utf-8"))
    assert len(tasks) >= 1
    assert all(t["expected"] == [] for t in tasks)
    assert all(t["answer_any_of"] for t in tasks)
