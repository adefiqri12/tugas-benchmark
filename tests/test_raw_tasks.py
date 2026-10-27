"""Raw-text task parser and dataset builder tests — no network."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tugas.raw_tasks import ANCHOR_DATE, RawTaskError, build_document, parse_raw

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "eval/datasets/raw"
GOLDEN = ROOT / "eval/datasets/golden_id.json"


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_dataset", ROOT / "scripts/build_dataset.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _load_builder()

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
    assert tasks[1]["category"] == "general"  # default
    assert tasks[1]["notes"] == "subjective"


def test_tool_call_multiple_args_and_order():
    tasks = parse_raw(
        "id: t\n"
        'Q: cek stok\n'
        'T: cek_stok(barang="tas ransel"; warna="navy")\n'
        'T: jam_buka_toko(hari="besok")\n'
    )
    assert tasks[0]["answer_any_of"] == []
    assert tasks[0]["expected"] == [
        {"tool": "cek_stok", "args": {"barang": "tas ransel", "warna": "navy"}},
        {"tool": "jam_buka_toko", "args": {"hari": "besok"}},
    ]


def test_no_arg_tool_call():
    tasks = parse_raw('id: t\nQ: tutup jam berapa?\nT: jam_buka_toko()\n')
    assert tasks[0]["expected"] == [{"tool": "jam_buka_toko", "args": {}}]


def test_quoted_semicolon_and_escaped_quote():
    tasks = parse_raw(
        'id: t\nQ: x\nT: hitung(ekspresi="(1 + 2) * 3"; label="kata \\"kutip\\"")\n'
    )
    assert tasks[0]["expected"][0]["args"] == {
        "ekspresi": "(1 + 2) * 3",
        "label": 'kata "kutip"',
    }


def test_list_value_any_of():
    tasks = parse_raw('id: t\nQ: x\nT: hitung(ekspresi=["2 * 20 - 3", "40 - 3"])\n')
    assert tasks[0]["expected"][0]["args"] == {"ekspresi": ["2 * 20 - 3", "40 - 3"]}


@pytest.mark.parametrize(
    "call",
    [
        'cek_stok(barang=kaos)',  # unquoted value
        "cek_stok(barang='kaos')",  # single quotes
        'cek_stok(barang="kaos"',  # missing ')'
        'cek_stok(barang="kaos"); hitung()',  # trailing junk
        'cek_stok(barang="kaos" x="y")',  # missing separator
        'cek_stok(barang="kaos"; barang="lain")',  # duplicate key
        'cek_stok(barang=)',  # empty value
        'cek_stok(barang="kaos";)',  # dangling separator
        'Cek_Stok(barang="kaos")',  # bad tool name
        "cek_stok",  # not a call
    ],
)
def test_malformed_tool_call_rejected(call):
    with pytest.raises(RawTaskError, match="entri 't'"):
        parse_raw(f"id: t\nQ: x\nT: {call}\n")


def test_tool_and_answer_rejected():
    with pytest.raises(RawTaskError, match="tidak keduanya"):
        parse_raw('id: t\nQ: x\nT: hitung()\nA: 42\n')


def test_missing_tool_and_answer_rejected():
    with pytest.raises(RawTaskError, match="minimal satu"):
        parse_raw("id: a\nQ: x\n")


def test_duplicate_id_rejected():
    with pytest.raises(RawTaskError, match="duplikat"):
        parse_raw("id: a\nQ: x\nA: y\n---\nid: a\nQ: z\nA: w\n")


def test_unknown_key_rejected():
    with pytest.raises(RawTaskError, match="key tidak dikenal"):
        parse_raw("id: a\nQ: x\nA: y\nBOGUS: z\n")


def test_duplicate_scalar_key_rejected():
    with pytest.raises(RawTaskError, match="duplikat"):
        parse_raw("id: a\nQ: x\nQ: y\nA: z\n")


def test_review_line_not_emitted():
    tasks = parse_raw('id: t\nQ: x\nT: hitung()\nR: Selesai kak.\nN: n\n')
    assert "R" not in tasks[0]
    doc = build_document(tasks, "x.txt")
    assert "Selesai" not in json.dumps(doc, ensure_ascii=False)


def test_build_document_envelope():
    doc = build_document(parse_raw(SAMPLE), "sample.txt")
    assert doc["meta"]["anchor_date"] == ANCHOR_DATE == "2026-10-07"
    assert doc["meta"]["language"] == "id"
    assert doc["meta"]["source"] == "sample.txt"
    assert set(doc["meta"]) == {"name", "language", "anchor_date", "source", "note"}
    assert len(doc["tasks"]) == 2
    assert "expected" in doc["tasks"][1]


def test_real_sources_parse_and_ids_unique():
    paths = sorted(RAW_DIR.glob("*.txt"))
    assert [p.name for p in paths] == ["core.txt", "customer.txt", "no_tool.txt", "owner.txt"]
    tasks = builder.build_tasks(RAW_DIR)
    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids)) == 51
    assert {t["id"] for t in tasks} >= {"id-flight-abs", "wa-stok-kaos", "id-abstain-sapaan"}
    assert all(t["expected"] or t["answer_any_of"] for t in tasks)


def test_shipped_dataset_is_current():
    assert builder.check_dataset() is None
    shipped = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert [t["id"] for t in shipped["tasks"]] == [
        t["id"] for t in builder.build_tasks(RAW_DIR)
    ]


def test_check_detects_stale_dataset(tmp_path, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "core.txt").write_text("id: a\nQ: x\nT: hitung()\n", encoding="utf-8")
    out = tmp_path / "golden.json"
    args = ["--raw-dir", str(raw), "--out", str(out)]

    assert builder.main(args) == 0
    assert builder.check_dataset(raw, out) is None
    assert builder.main(["--check", *args]) == 0

    (raw / "core.txt").write_text("id: b\nQ: y\nT: hitung()\n", encoding="utf-8")
    problem = builder.check_dataset(raw, out)
    assert problem and "stale" in problem
    assert builder.main(["--check", *args]) == 1
    assert "stale" in capsys.readouterr().err


def test_duplicate_id_across_sources_rejected(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.txt").write_text("id: same\nQ: x\nT: hitung()\n", encoding="utf-8")
    (raw / "b.txt").write_text("id: same\nQ: y\nT: hitung()\n", encoding="utf-8")
    with pytest.raises(RawTaskError, match="lintas sumber"):
        builder.build_tasks(raw)

def test_parse_raw_rejects_non_string_input() -> None:
    with pytest.raises(RawTaskError):
        parse_raw(b"id: x")
