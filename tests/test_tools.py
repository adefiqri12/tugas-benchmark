"""Offline tests for the mock tools. No gateway, no network."""

from __future__ import annotations

import pytest

from tugas.tools import (
    _STOK,
    ToolError,
    execute,
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


# -- execute: bad arguments must surface as ToolError ------------------------


@pytest.mark.parametrize(
    "name,args",
    [
        ("parse_tanggal", {}),  # missing required arg -> TypeError
        ("parse_tanggal", {"teks": "besok", "bogus": 1}),  # unexpected kwarg
        ("parse_tanggal", {"teks": 123}),  # non-string arg -> AttributeError
        ("sapaan_waktu", {"jam": 123}),  # non-string arg -> AttributeError
        ("cari_penerbangan", {"asal": "A", "tujuan": "B", "tanggal": "2026-13-40"}),
        ("cek_cuaca", {"kota": "X", "tanggal_mulai": "nope", "tanggal_selesai": "2026-10-10"}),
    ],
)
def test_execute_bad_args_raise_tool_error(name: str, args: dict) -> None:
    # Missing/extra kwargs (TypeError) and stdlib date errors (ValueError) are
    # bad arguments, not provider failures; they must be normalised to ToolError
    # so the agent loop feeds them back instead of scoring API_ERROR.
    with pytest.raises(ToolError):
        execute(name, args)


def test_execute_unknown_tool_raises_tool_error() -> None:
    with pytest.raises(ToolError):
        execute("tidak_ada", {})


def test_execute_valid_call_passes_through() -> None:
    assert execute("parse_tanggal", {"teks": "besok"})["tanggal"] == "2026-10-08"


# -- hitung: bounded, numeric-only results -----------------------------------


@pytest.mark.parametrize(
    "ekspresi",
    [
        "9**9**9",  # exponent bomb: would hang the harness
        "2**100000000",  # huge int: would overflow JSON serialisation
        "()",  # eval returns a tuple, not a number
    ],
)
def test_hitung_rejects_unbounded_or_non_numeric(ekspresi: str) -> None:
    with pytest.raises(ToolError):
        execute("hitung", {"ekspresi": ekspresi})


def test_hitung_rejects_overlong_expression() -> None:
    with pytest.raises(ToolError):
        execute("hitung", {"ekspresi": "1" * 201})


def test_hitung_still_computes_normally() -> None:
    assert execute("hitung", {"ekspresi": "1200000 - 150000"})["hasil"] == 1050000
    assert execute("hitung", {"ekspresi": "2 * 20 - 3"})["hasil"] == 37

# -- cek_stok ---------------------------------------------------------------

def test_cek_stok_variant_in_stock() -> None:
    r = execute("cek_stok", {"barang": "kaos polos", "warna": "hitam", "ukuran": "L"})
    assert r["tersedia"] is True
    assert r["stok"] == 24

def test_cek_stok_size_only() -> None:
    assert execute("cek_stok", {"barang": "sepatu sneakers", "ukuran": "42"})["stok"] == 11

def test_cek_stok_bulk_quantity() -> None:
    r = execute("cek_stok", {"barang": "gelas plastik", "jumlah": "50"})
    assert r["tersedia"] is True
    assert r["stok"] == 480

def test_cek_stok_quantity_over_stock() -> None:
    r = execute("cek_stok", {"barang": "gelas plastik", "jumlah": "500"})
    assert r["tersedia"] is False

def test_cek_stok_empty_variant_reports_restock() -> None:
    r = execute("cek_stok", {"barang": "hijab babydoll", "warna": "abu"})
    assert r["tersedia"] is False
    assert r["restock"] == "minggu depan"

def test_cek_stok_unknown_item_raises() -> None:
    with pytest.raises(ToolError):
        execute("cek_stok", {"barang": "sepeda gunung"})

def test_cek_stok_bad_quantity_raises() -> None:
    with pytest.raises(ToolError):
        execute("cek_stok", {"barang": "kaos polos", "jumlah": "banyak"})

# -- jam_buka_toko ----------------------------------------------------------

def test_jam_buka_default_is_anchor_day() -> None:
    r = execute("jam_buka_toko", {})
    assert (r["hari"], r["buka"], r["tutup"]) == ("rabu", "09:00", "21:00")

def test_jam_buka_sunday_differs() -> None:
    r = execute("jam_buka_toko", {"hari": "minggu"})
    assert r["tanggal"] == "2026-10-11"
    assert (r["buka"], r["tutup"]) == ("10:00", "20:00")

def test_jam_buka_branch_schedule() -> None:
    r = execute("jam_buka_toko", {"cabang": "samarinda"})
    assert r["tutup"] == "21:30"

def test_jam_buka_holiday_is_closed() -> None:
    r = execute("jam_buka_toko", {"hari": "2026-12-25"})
    assert r["libur"] is True
    assert r["buka"] is None

def test_jam_buka_invalid_day_raises() -> None:
    with pytest.raises(ToolError):
        execute("jam_buka_toko", {"hari": "hari kiamat"})

# -- cek_diskon -------------------------------------------------------------

def test_cek_diskon_all_active_promos() -> None:
    r = execute("cek_diskon", {})
    assert r["ditemukan"] is True
    assert len(r["promo"]) == 4

def test_cek_diskon_product_filter_excludes_vouchers() -> None:
    r = execute("cek_diskon", {"produk": "kaos"})
    assert [p["produk"] for p in r["promo"]] == ["kaos"]
    assert r["promo"][0]["diskon_persen"] == 20

def test_cek_diskon_voucher_code() -> None:
    r = execute("cek_diskon", {"kode": "hemat50"})
    assert r["promo"][0]["potongan"] == 50000

def test_cek_diskon_unknown_product_is_empty_not_error() -> None:
    r = execute("cek_diskon", {"produk": "sepeda gunung"})
    assert r["ditemukan"] is False
    assert r["promo"] == []

# -- cek_booking ------------------------------------------------------------

def test_cek_booking_available_slot() -> None:
    r = execute("cek_booking", {
        "tempat": "meja", "tanggal": "2026-10-08", "jam": "19:00", "jumlah_orang": "4",
    })
    assert r["tersedia"] is True
    assert r["kapasitas"] == 8

def test_cek_booking_booked_slot_is_unavailable() -> None:
    r = execute("cek_booking", {
        "tempat": "studio foto", "tanggal": "2026-10-12", "jam": "13:00",
    })
    assert r["tersedia"] is False
    assert r["catatan"] == "slot sudah dibooking"

def test_cek_booking_over_capacity() -> None:
    r = execute("cek_booking", {
        "tempat": "meja", "tanggal": "2026-10-08", "jumlah_orang": "20",
    })
    assert r["tersedia"] is False

def test_cek_booking_past_date_is_unavailable() -> None:
    r = execute("cek_booking", {"tempat": "meja", "tanggal": "2026-10-01"})
    assert r["tersedia"] is False
    assert r["catatan"] == "tanggal sudah lewat"

def test_cek_booking_normalises_indonesian_time() -> None:
    r = execute("cek_booking", {
        "tempat": "lapangan futsal", "tanggal": "2026-10-07", "jam": "9 malam",
    })
    assert r["jam"] == "21:00"

def test_cek_booking_invalid_date_raises() -> None:
    with pytest.raises(ToolError):
        execute("cek_booking", {"tempat": "meja", "tanggal": "besok"})

# -- reporting tools --------------------------------------------------------

@pytest.mark.parametrize(
    "periode,transaksi,total",
    [
        ("hari ini", 12, 2_450_000),
        ("bulan ini", 340, 78_900_000),
        ("bulan lalu", 298, 64_200_000),
        ("2026-10-05", 9, 1_780_000),
    ],
)
def test_laporan_penjualan(periode: str, transaksi: int, total: int) -> None:
    r = execute("laporan_penjualan", {"periode": periode})
    assert (r["transaksi"], r["total"]) == (transaksi, total)

def test_laporan_penjualan_best_seller() -> None:
    r = execute("laporan_penjualan", {"periode": "bulan ini", "metrik": "produk terlaris"})
    assert (r["produk"], r["unit"]) == ("Kaos Polos Hitam", 128)

def test_laporan_pendapatan_matches_sales_total() -> None:
    for periode in ("hari ini", "bulan ini", "bulan lalu", "2026-10-05"):
        penjualan = execute("laporan_penjualan", {"periode": periode})
        pendapatan = execute("laporan_pendapatan", {"periode": periode})
        assert pendapatan["pendapatan"] == penjualan["total"]

def test_laporan_pendapatan_branch() -> None:
    r = execute("laporan_pendapatan", {"periode": "bulan ini", "cabang": "samarinda"})
    assert r["pendapatan"] == 31_200_000

def test_laporan_pendapatan_branches_sum_to_total() -> None:
    total = execute("laporan_pendapatan", {"periode": "bulan ini"})["pendapatan"]
    parts = sum(
        execute("laporan_pendapatan", {"periode": "bulan ini", "cabang": cab})["pendapatan"]
        for cab in ("pusat", "samarinda")
    )
    assert parts == total

def test_laporan_leads_source() -> None:
    r = execute("laporan_leads", {"periode": "bulan ini", "sumber": "instagram"})
    assert r["leads"] == 173

def test_laporan_leads_sources_sum_to_total() -> None:
    total = execute("laporan_leads", {"periode": "bulan ini"})["leads"]
    parts = sum(
        execute("laporan_leads", {"periode": "bulan ini", "sumber": s})["leads"]
        for s in ("instagram", "whatsapp", "tiktok", "google")
    )
    assert parts == total

def test_stock_dimensions_sum_to_total() -> None:
    # Aggregates and per-dimension breakdowns must tell the same story: a
    # benchmark whose mock data contradicts itself teaches the wrong lesson.
    for name, item in _STOK.items():
        for dimension in ("ukuran", "warna"):
            if dimension in item:
                assert sum(item[dimension].values()) == item["stok"], (name, dimension)
        for variant, count in item.get("varian", {}).items():
            warna, ukuran = variant.split("|")
            assert count <= item["warna"][warna], (name, variant, "warna")
            assert count <= item["ukuran"][ukuran], (name, variant, "ukuran")

@pytest.mark.parametrize(
    "name,args",
    [
        ("laporan_penjualan", {"periode": "tahun lalu"}),
        ("laporan_penjualan", {"periode": "bulan ini", "metrik": "omzet"}),
        ("laporan_pendapatan", {"periode": "bulan ini", "cabang": "medan"}),
        ("laporan_leads", {"periode": "bulan ini", "sumber": "facebook"}),
    ],
)
def test_reporting_unknown_selection_raises(name: str, args: dict) -> None:
    with pytest.raises(ToolError):
        execute(name, args)

def test_execute_rejects_non_string_tool_name() -> None:
    with pytest.raises(ToolError):
        execute(["parse_tanggal"], {})

def test_cek_cuaca_rejects_range_over_two_weeks() -> None:
    with pytest.raises(ToolError):
        execute("cek_cuaca", {
            "kota": "Samarinda",
            "tanggal_mulai": "2026-10-01",
            "tanggal_selesai": "2026-10-15",
        })

def test_cek_stok_rejects_underscore_quantity() -> None:
    # int("1_0") is 10 in Python, but "1_0" is not a quantity a user typed.
    with pytest.raises(ToolError):
        execute("cek_stok", {"barang": "kaos polos", "jumlah": "1_0"})

@pytest.mark.parametrize(
    "jam,expected",
    [
        ("12 siang", "12:00"),
        ("12 malam", "00:00"),
        ("12 pagi", "00:00"),
        ("8 malam", "20:00"),
        ("8.30", "08:30"),
        ("1 sore", "13:00"),
    ],
)
def test_cek_booking_normalizes_indonesian_time(jam: str, expected: str) -> None:
    result = execute("cek_booking", {
        "tempat": "meja", "tanggal": "2026-10-08", "jam": jam,
    })
    assert result["jam"] == expected

def test_cek_cuaca_returns_the_requested_window() -> None:
    result = execute("cek_cuaca", {
        "kota": "Samarinda",
        "tanggal_mulai": "2026-10-07",
        "tanggal_selesai": "2026-10-09",
    })
    assert [d["tanggal"] for d in result["ramalan"]] == [
        "2026-10-07", "2026-10-08", "2026-10-09",
    ]
    assert all(d["prakiraan"] for d in result["ramalan"])
