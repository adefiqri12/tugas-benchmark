"""Offline tests for the mock tools. No gateway, no network."""

from __future__ import annotations

import pytest

from tugas.tools import (
    ToolError,
    parse_alamat,
    parse_tanggal,
    sapaan_waktu,
)


# -- parse_tanggal: relative words ------------------------------------------


def test_lusa_default_anchor() -> None:
    assert parse_tanggal("lusa")["tanggal"] == "2026-10-09"


def test_lusa_honors_explicit_acuan() -> None:
    assert parse_tanggal("lusa", acuan="2026-01-01")["tanggal"] == "2026-01-03"


def test_lusa_case_insensitive() -> None:
    assert parse_tanggal("LUSA")["tanggal"] == "2026-10-09"


# -- parse_tanggal: numeric DD/MM -------------------------------------------


@pytest.mark.parametrize(
    "teks,expected",
    [
        ("7/10", "2026-10-07"),
        ("7/10/2026", "2026-10-07"),
        ("07/10", "2026-10-07"),
    ],
)
def test_numeric_ddmm(teks: str, expected: str) -> None:
    assert parse_tanggal(teks)["tanggal"] == expected


@pytest.mark.parametrize("teks", ["7/13", "31/2", "7/10/26"])
def test_numeric_ddmm_invalid(teks: str) -> None:
    with pytest.raises(ToolError):
        parse_tanggal(teks)


# -- parse_tanggal: word dates still work -----------------------------------


def test_word_date_unchanged() -> None:
    assert parse_tanggal("12 Oktober 2026")["tanggal"] == "2026-10-12"


# -- sapaan_waktu -----------------------------------------------------------


@pytest.mark.parametrize(
    "jam,waktu",
    [
        ("19:30", "malam"),
        ("05:00", "pagi"),
        ("12:00", "siang"),
        ("04:00", "pagi"),
        ("10:59", "pagi"),
        ("11:00", "siang"),
        ("14:59", "siang"),
        ("15:00", "sore"),
        ("17:59", "sore"),
        ("18:00", "malam"),
        ("03:59", "malam"),
    ],
)
def test_sapaan_waktu_boundaries(jam: str, waktu: str) -> None:
    assert sapaan_waktu(jam)["waktu"] == waktu


def test_sapaan_waktu_invalid() -> None:
    with pytest.raises(ToolError):
        sapaan_waktu("25:00")


# -- parse_alamat -----------------------------------------------------------


def test_parse_alamat_samarinda() -> None:
    alamat = (
        "Jl. Pahlawan No. 12, RT 03/RW 05, Kel. Sungai Pinang Dalam, "
        "Kec. Samarinda Utara, Samarinda"
    )
    assert parse_alamat(alamat)["kota"] == "Samarinda"


def test_parse_alamat_with_province() -> None:
    alamat = "Jl. Merdeka No. 1, Kel. A, Kec. B, Bandung, Jawa Barat"
    assert parse_alamat(alamat)["kota"] == "Bandung"


def test_parse_alamat_empty_raises() -> None:
    with pytest.raises(ToolError):
        parse_alamat("")


def test_parse_alamat_without_commas_raises() -> None:
    # No commas -> the whole string looks like one street component; there is
    # no city to return, so it must fail loudly, not echo the address back.
    with pytest.raises(ToolError):
        parse_alamat("Jl. Pahlawan No. 12 RT 03 Samarinda")


def test_parse_alamat_all_subcity_raises() -> None:
    with pytest.raises(ToolError):
        parse_alamat("Jl. A, Kel. B, Kec. C")
