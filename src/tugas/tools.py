"""Mock Indonesian tools the agent can call.

Deterministic, offline, no network — so a harness run is reproducible and the
score measures the *agent*, not an upstream API's mood. Each tool validates its
own arguments and returns a JSON-able dict.
"""

from __future__ import annotations

import datetime as _dt
import re as _re

from .anchor import ANCHOR
from .models import ToolSpec

# -- specs sent to the model ------------------------------------------------
TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="parse_tanggal",
        description=(
            "Ubah ungkapan tanggal Bahasa Indonesia (mis. 'besok', '12 Oktober 2026', "
            "'minggu depan') menjadi tanggal absolut format YYYY-MM-DD. Gunakan untuk "
            "mengetahui tanggal acuan sebelum memanggil tool lain yang butuh tanggal."
        ),
        parameters={"teks": "string", "acuan": "string (YYYY-MM-DD, opsional)"},
        required=["teks"],
    ),
    ToolSpec(
        name="cari_penerbangan",
        description="Cari penerbangan berdasarkan kota asal, tujuan, dan tanggal keberangkatan.",
        parameters={
            "asal": "string (nama kota)",
            "tujuan": "string (nama kota)",
            "tanggal": "string (YYYY-MM-DD)",
        },
        required=["asal", "tujuan", "tanggal"],
    ),
    ToolSpec(
        name="cek_cuaca",
        description="Cek prakiraan cuaca sebuah kota untuk rentang tanggal.",
        parameters={
            "kota": "string",
            "tanggal_mulai": "string (YYYY-MM-DD)",
            "tanggal_selesai": "string (YYYY-MM-DD)",
        },
        required=["kota", "tanggal_mulai", "tanggal_selesai"],
    ),
    ToolSpec(
        name="hitung",
        description="Hitung ekspresi aritmetika. Pakai untuk operasi angka apa pun.",
        parameters={"ekspresi": "string (mis. '1200000 - 150000')"},
        required=["ekspresi"],
    ),
    ToolSpec(
        name="sapaan_waktu",
        description=(
            "Tentukan sapaan waktu Bahasa Indonesia yang tepat untuk sebuah jam: "
            "pagi, siang, sore, atau malam. Pakai sebelum menyapa pengguna."
        ),
        parameters={"jam": "string (format HH:MM, mis. '19:30')"},
        required=["jam"],
    ),
    ToolSpec(
        name="parse_alamat",
        description=(
            "Ambil nama KOTA dari sebuah alamat Bahasa Indonesia. Gunakan saat perlu "
            "mengetahui kota pengiriman/tujuan dari teks alamat."
        ),
        parameters={"teks": "string (alamat lengkap)"},
        required=["teks"],
    ),
    ToolSpec(
        name="cek_stok",
        description=(
            "Cek ketersediaan stok barang, termasuk varian warna/ukuran dan jumlah "
            "pembelian grosir. Gunakan sebelum menjanjikan barang masih ready."
        ),
        parameters={
            "barang": "string (nama barang)",
            "ukuran": "string (opsional)",
            "warna": "string (opsional)",
            "jumlah": "string (opsional, jumlah yang diminta)",
        },
        required=["barang"],
    ),
    ToolSpec(
        name="jam_buka_toko",
        description=(
            "Cek jam buka dan tutup toko untuk hari/tanggal tertentu dan cabang "
            "tertentu. Tanpa argumen: jadwal cabang pusat hari ini."
        ),
        parameters={
            "hari": "string (opsional: nama hari, 'hari ini', 'besok', atau YYYY-MM-DD)",
            "cabang": "string (opsional: pusat, samarinda)",
        },
        required=[],
    ),
    ToolSpec(
        name="cek_diskon",
        description=(
            "Cek promo/diskon aktif untuk sebuah produk atau kode voucher. Tanpa "
            "argumen: tampilkan semua promo yang sedang aktif."
        ),
        parameters={
            "produk": "string (opsional, nama produk)",
            "kode": "string (opsional, kode voucher)",
        },
        required=[],
    ),
    ToolSpec(
        name="cek_booking",
        description=(
            "Cek ketersediaan booking tempat (meja, room karaoke, studio foto, "
            "lapangan futsal, reservasi acara) pada tanggal dan jam tertentu."
        ),
        parameters={
            "tempat": "string",
            "tanggal": "string (YYYY-MM-DD)",
            "jam": "string (opsional, HH:MM)",
            "jumlah_orang": "string (opsional)",
        },
        required=["tempat", "tanggal"],
    ),
    ToolSpec(
        name="laporan_penjualan",
        description=(
            "Laporan penjualan untuk satu periode: jumlah transaksi dan total rupiah. "
            "metrik='produk terlaris' mengembalikan produk paling laku pada periode itu."
        ),
        parameters={
            "periode": (
                "string: 'hari ini'|'kemarin'|'minggu ini'|'bulan ini'|"
                "'bulan lalu'|'tahun ini'|YYYY-MM-DD"
            ),
            "metrik": "string (opsional: 'produk terlaris')",
        },
        required=["periode"],
    ),
    ToolSpec(
        name="laporan_pendapatan",
        description="Laporan pendapatan (omzet) untuk satu periode, seluruh cabang atau satu cabang.",
        parameters={
            "periode": (
                "string: 'hari ini'|'kemarin'|'minggu ini'|'bulan ini'|"
                "'bulan lalu'|'tahun ini'|YYYY-MM-DD"
            ),
            "cabang": "string (opsional: pusat, samarinda)",
        },
        required=["periode"],
    ),
    ToolSpec(
        name="laporan_leads",
        description="Laporan jumlah leads untuk satu periode, seluruh sumber atau satu sumber.",
        parameters={
            "periode": (
                "string: 'hari ini'|'kemarin'|'minggu ini'|'bulan ini'|"
                "'bulan lalu'|'tahun ini'|YYYY-MM-DD"
            ),
            "sumber": "string (opsional: instagram, whatsapp, tiktok, google)",
        },
        required=["periode"],
    ),
]


class ToolError(Exception):
    """Raised when a tool is called with bad arguments."""


# -- implementations --------------------------------------------------------
_BULAN = {
    "januari": 1, "februari": 2, "maret": 3, "april": 4, "mei": 5, "juni": 6,
    "juli": 7, "agustus": 8, "september": 9, "oktober": 10, "november": 11,
    "desember": 12,
}

_DEFAULT_TODAY = ANCHOR  # fixed date -> reproducible evals

# Indonesian DD/MM numeric dates. Loose match detects "looks numeric"; strict
# fullmatch requires a 4-digit year (two-digit years are rejected on purpose).
_NUMERIC_DATE = _re.compile(r"(\d{1,2})/(\d{1,2})(?:/(\d{1,4}))?")
_NUMERIC_DATE_STRICT = _re.compile(r"(\d{1,2})/(\d{1,2})(?:/(\d{4}))?")


def parse_tanggal(teks: str, acuan: str | None = None) -> dict:
    base = _dt.date.fromisoformat(acuan) if acuan else _DEFAULT_TODAY
    t = (teks or "").strip().lower()

    if t in ("hari ini", "today"):
        d = base
    elif t in ("besok", "esok"):
        d = base + _dt.timedelta(days=1)
    elif t in ("lusa",):
        d = base + _dt.timedelta(days=2)
    elif t in ("kemarin",):
        d = base - _dt.timedelta(days=1)
    elif "minggu depan" in t or "pekan depan" in t:
        d = base + _dt.timedelta(days=7)
    elif _NUMERIC_DATE.search(t):
        # Indonesian DD/MM numeric dates, tried before the word-date branch.
        # "7/10" -> 7 Oct, "7/10/2026" -> explicit year, "07/10" allowed.
        m = _NUMERIC_DATE_STRICT.fullmatch(t)
        if m is None:
            # Looks numeric but the year is not four digits (e.g. "7/10/26").
            raise ToolError(f"tidak bisa memahami tanggal: {teks!r}")
        day = int(m.group(1))
        month = int(m.group(2))
        year = int(m.group(3)) if m.group(3) else base.year
        try:
            d = _dt.date(year, month, day)  # calendar validates 31/2, 7/13
        except ValueError as exc:
            raise ToolError(f"tidak bisa memahami tanggal: {teks!r}") from exc
    else:
        # "12 Oktober 2026" / "12 oktober 2026"
        parts = t.replace(",", "").split()
        try:
            day = int(parts[0])
            month = _BULAN[parts[1]]
            year = int(parts[2]) if len(parts) > 2 else base.year
            d = _dt.date(year, month, day)
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"tidak bisa memahami tanggal: {teks!r}") from exc
    return {"teks": teks, "tanggal": d.isoformat()}


def cari_penerbangan(asal: str, tujuan: str, tanggal: str) -> dict:
    _dt.date.fromisoformat(tanggal)  # validate
    # deterministic mock schedule
    flights = [
        {"kode": "ID-6120", "berangkat": "06:15", "harga_idr": 1450000},
        {"kode": "GA-410", "berangkat": "09:40", "harga_idr": 1780000},
        {"kode": "JT-780", "berangkat": "13:20", "harga_idr": 1290000},
        {"kode": "ID-6128", "berangkat": "17:55", "harga_idr": 1520000},
    ]
    return {"asal": asal, "tujuan": tujuan, "tanggal": tanggal, "penerbangan": flights}


def cek_cuaca(kota: str, tanggal_mulai: str, tanggal_selesai: str) -> dict:
    a = _dt.date.fromisoformat(tanggal_mulai)
    b = _dt.date.fromisoformat(tanggal_selesai)
    if b < a:
        raise ToolError("tanggal_selesai sebelum tanggal_mulai")
    if (b - a).days > 13:
        raise ToolError("rentang cuaca maksimal 14 hari")
    pola = ["cerah berawan", "hujan ringan", "berawan", "hujan petir", "cerah"]
    hari = []
    d = a
    i = 0
    while d <= b and i < 14:
        hari.append({"tanggal": d.isoformat(), "prakiraan": pola[i % len(pola)]})
        d += _dt.timedelta(days=1)
        i += 1
    return {"kota": kota, "ramalan": hari}


def hitung(ekspresi: str) -> dict:
    allowed = set("0123456789+-*/(). ")
    if not set(ekspresi) <= allowed:
        raise ToolError(f"ekspresi tidak diizinkan: {ekspresi!r}")
    # `**` is rejected: `9**9**9` builds a 369-million-digit integer and hangs
    # the process, and `2**100000000` overflows JSON serialisation. A mock
    # calculator for money/quantities does not need exponentiation.
    if "**" in ekspresi:
        raise ToolError(f"pangkat tidak diizinkan: {ekspresi!r}")
    if len(ekspresi) > 200:
        raise ToolError(f"ekspresi terlalu panjang ({len(ekspresi)} karakter)")
    try:
        val = eval(ekspresi, {"__builtins__": {}}, {})  # noqa: S307 - charset-guarded
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"gagal menghitung: {ekspresi!r}") from exc
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ToolError(f"hasil bukan angka: {ekspresi!r}")
    return {"ekspresi": ekspresi, "hasil": val}


def sapaan_waktu(jam: str) -> dict:
    """Indonesian time-of-day greeting for a clock time (HH:MM).

    Boundaries: pagi 04:00-10:59, siang 11:00-14:59, sore 15:00-17:59,
    malam 18:00-03:59 (after maghrib it is malam, not sore).
    """
    t = (jam or "").strip()
    m = _re.fullmatch(r"(\d{1,2})[:.](\d{2})", t)
    if m is None:
        raise ToolError(f"format jam tidak dikenal: {jam!r} (pakai HH:MM)")
    h, mi = int(m.group(1)), int(m.group(2))
    if h > 23 or mi > 59:
        raise ToolError(f"jam tidak valid: {jam!r}")
    total = h * 60 + mi
    if 4 * 60 <= total <= 10 * 60 + 59:
        sapaan, waktu = "Selamat pagi", "pagi"
    elif 11 * 60 <= total <= 14 * 60 + 59:
        sapaan, waktu = "Selamat siang", "siang"
    elif 15 * 60 <= total <= 17 * 60 + 59:
        sapaan, waktu = "Selamat sore", "sore"
    else:  # 18:00-23:59 and 00:00-03:59
        sapaan, waktu = "Selamat malam", "malam"
    return {"jam": jam, "waktu": waktu, "sapaan": sapaan}


# Prefixes that mark a non-city address component (street / RT-RW / kelurahan /
# kecamatan). The kota is the first comma-separated component that survives
# this filter (order: ... kecamatan, kota, provinsi).
_NON_KOTA_PREFIX = _re.compile(
    r"^(jl\.?|jalan|gg\.?|gang|no\.?|nomor|rt|rw|kel\.?|kelurahan|"
    r"kec\.?|kecamatan)\b",
    _re.IGNORECASE,
)


def parse_alamat(teks: str) -> dict:
    """Extract the KOTA (city) from an Indonesian address.

    Heuristic: split on commas, drop components that start with a street /
    RT-RW / Kel. / Kec. prefix, and return the first remaining component (the
    city precedes the province in the conventional order).

    Limits: it does not know province names, so it cannot tell a city from a
    province if a city field is missing; and it needs commas between the
    address parts. When no city component can be identified it raises
    ``ToolError`` rather than returning a non-city token. Kept intentionally
    simple and offline.
    """
    parts = [p.strip() for p in (teks or "").split(",") if p.strip()]
    if not parts:
        raise ToolError(f"alamat kosong: {teks!r}")
    for p in parts:
        if not _NON_KOTA_PREFIX.match(p):
            return {"teks": teks, "kota": p}
    # Every component looked like a street / RT-RW / kelurahan / kecamatan:
    # there is no city to return. Fail loudly instead of returning a non-city.
    raise ToolError(f"kota tidak ditemukan pada alamat: {teks!r}")


# -- commerce tools ---------------------------------------------------------

def _teks(value: object) -> str:
    return str(value).strip().lower()

def _bulat(value: object, nama: str) -> int:
    teks = str(value).strip()
    if not _re.fullmatch(r"\+?\d+", teks):
        raise ToolError(f"{nama} harus bilangan bulat: {value!r}")
    try:
        n = int(teks)
    except (TypeError, ValueError) as exc:
        raise ToolError(f"{nama} harus bilangan bulat: {value!r}") from exc
    if n < 1:
        raise ToolError(f"{nama} harus lebih dari 0: {n}")
    return n

def _pilih_kunci(query: object, table: dict, label: str) -> str:
    """Resolve a free-text name to one key of ``table`` (exact, then substring)."""
    q = _teks(query)
    if q in table:
        return q
    matches = [k for k in table if k in q or q in k]
    if not matches:
        raise ToolError(f"{label} tidak dikenal: {query!r}")
    return max(matches, key=len)  # most specific match

_STOK: dict[str, dict] = {
    "kaos polos": {
        "stok": 120,
        "ukuran": {"s": 20, "m": 40, "l": 35, "xl": 25},
        "warna": {"hitam": 60, "putih": 40, "navy": 20},
        "varian": {"hitam|l": 24, "hitam|m": 18, "putih|l": 12, "navy|l": 6},
    },
    "sepatu sneakers": {
        "stok": 30,
        "ukuran": {"40": 5, "41": 7, "42": 11, "43": 5, "44": 2},
        "warna": {"hitam": 18, "putih": 12},
        "varian": {"hitam|42": 7, "putih|42": 4},
    },
    "tas ransel": {
        "stok": 22,
        "warna": {"hitam": 10, "navy": 7, "abu": 5},
    },
    "gelas plastik": {"stok": 480, "satuan": "pcs"},
    "hijab babydoll": {
        "stok": 0,
        "warna": {"abu": 0, "hitam": 0},
        "restock": "minggu depan",
    },
}

def cek_stok(
    barang: str,
    ukuran: str | None = None,
    warna: str | None = None,
    jumlah: str | None = None,
) -> dict:
    key = _pilih_kunci(barang, _STOK, "barang")
    item = _STOK[key]
    minta = _bulat(jumlah, "jumlah") if jumlah not in (None, "") else 1
    uk = _teks(ukuran) if ukuran not in (None, "") else None
    wr = _teks(warna) if warna not in (None, "") else None

    if uk and wr:
        stok = item.get("varian", {}).get(f"{wr}|{uk}", 0)
    elif wr:
        stok = item.get("warna", {}).get(wr, 0)
    elif uk:
        stok = item.get("ukuran", {}).get(uk, 0)
    else:
        stok = item["stok"]

    hasil = {
        "barang": key,
        "ukuran": uk,
        "warna": wr,
        "jumlah": minta,
        "stok": stok,
        "tersedia": stok >= minta,
    }
    if item.get("restock"):
        hasil["restock"] = item["restock"]
    return hasil

_HARI_NAMA = ("senin", "selasa", "rabu", "kamis", "jumat", "sabtu", "minggu")

_JADWAL: dict[str, dict[str, tuple[str, str]]] = {
    "pusat": {
        "senin": ("09:00", "21:00"),
        "selasa": ("09:00", "21:00"),
        "rabu": ("09:00", "21:00"),
        "kamis": ("09:00", "21:00"),
        "jumat": ("09:00", "21:00"),
        "sabtu": ("09:00", "21:00"),
        "minggu": ("10:00", "20:00"),
    },
    "samarinda": {
        "senin": ("09:00", "21:30"),
        "selasa": ("09:00", "21:30"),
        "rabu": ("09:00", "21:30"),
        "kamis": ("09:00", "21:30"),
        "jumat": ("09:00", "21:30"),
        "sabtu": ("09:00", "21:30"),
        "minggu": ("10:00", "20:00"),
    },
}

# Fixed closures, aligned with the dataset anchor year.
_LIBUR = {
    "2026-12-25": "libur Natal",
    "2026-12-31": "tutup awal tahun",
    "2026-01-01": "libur Tahun Baru",
}

def jam_buka_toko(hari: str | None = None, cabang: str | None = None) -> dict:
    cab = _pilih_kunci(cabang or "pusat", _JADWAL, "cabang")
    h = _teks(hari) if hari not in (None, "") else "hari ini"

    if h in ("hari ini", "besok", "kemarin"):
        tanggal = _dt.date.fromisoformat(parse_tanggal(h)["tanggal"])
        nama_hari = _HARI_NAMA[tanggal.weekday()]
    elif h in _HARI_NAMA:
        nama_hari = h
        delta = (_HARI_NAMA.index(h) - _DEFAULT_TODAY.weekday()) % 7
        tanggal = _DEFAULT_TODAY + _dt.timedelta(days=delta)
    else:
        try:
            tanggal = _dt.date.fromisoformat(str(hari).strip())
        except ValueError as exc:
            raise ToolError(f"hari tidak dikenal: {hari!r}") from exc
        nama_hari = _HARI_NAMA[tanggal.weekday()]

    hasil = {"cabang": cab, "hari": nama_hari, "tanggal": tanggal.isoformat()}
    if tanggal.isoformat() in _LIBUR:
        hasil.update({"libur": True, "buka": None, "tutup": None,
                      "catatan": _LIBUR[tanggal.isoformat()]})
        return hasil

    buka, tutup = _JADWAL[cab][nama_hari]
    hasil.update({"libur": False, "buka": buka, "tutup": tutup, "catatan": ""})
    return hasil

_PROMO = [
    {"produk": "kaos", "diskon_persen": 20, "jenis": "diskon",
     "label": "Semua kaos", "berakhir": "2026-10-31"},
    {"produk": "kemeja flanel", "diskon_persen": 15, "jenis": "diskon",
     "label": "Kemeja flanel", "berakhir": "2026-10-20"},
    {"produk": "paket skincare", "diskon_persen": 30, "jenis": "bundling",
     "label": "Paket skincare bundling", "berakhir": "2026-10-31"},
    {"kode": "HEMAT50", "potongan": 50000, "jenis": "voucher",
     "label": "Voucher HEMAT50", "berakhir": "2026-10-31"},
]

def cek_diskon(produk: str | None = None, kode: str | None = None) -> dict:
    q_produk = _teks(produk) if produk not in (None, "") else None
    q_kode = _teks(kode) if kode not in (None, "") else None
    hasil = []
    for promo in _PROMO:
        p = promo.get("produk", "")
        if q_produk:
            if not p or not (p == q_produk or p in q_produk or q_produk in p):
                continue
        if q_kode and promo.get("kode", "").lower() != q_kode:
            continue
        hasil.append(dict(promo))
    return {
        "produk": q_produk,
        "kode": q_kode,
        "promo": hasil,
        "ditemukan": bool(hasil),
    }

_TEMPAT = {
    "meja": {"kapasitas": 8},
    "room karaoke": {"kapasitas": 12},
    "studio foto": {"kapasitas": 10},
    "lapangan futsal": {"kapasitas": 14},
    "reservasi acara": {"kapasitas": 50},
}

# Slots that are already taken, so availability is not a constant "yes".
_BOOKED = {
    ("meja", "2026-10-10", "19:00"),
    ("room karaoke", "2026-10-11", "20:00"),
    ("studio foto", "2026-10-12", "13:00"),
    ("lapangan futsal", "2026-10-07", "20:00"),
    ("reservasi acara", "2026-10-17", "10:00"),
}

def _jam(value: object) -> str:
    t = _teks(value).replace(".", ":")
    m = _re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(pagi|siang|sore|malam)?", t)
    if m is None:
        raise ToolError(f"jam tidak dikenal: {value!r}")
    jam, menit, bagian = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if bagian in ("sore", "malam") and jam < 12:
        jam += 12
    # "12 siang" is noon; "12 pagi"/"12 malam" are midnight.
    if bagian in ("pagi", "malam") and jam == 12:
        jam = 0
    if not (0 <= jam <= 23 and 0 <= menit <= 59):
        raise ToolError(f"jam tidak valid: {value!r}")
    return f"{jam:02d}:{menit:02d}"

def cek_booking(
    tempat: str,
    tanggal: str,
    jam: str | None = None,
    jumlah_orang: str | None = None,
) -> dict:
    key = _pilih_kunci(tempat, _TEMPAT, "tempat")
    try:
        d = _dt.date.fromisoformat(str(tanggal).strip())
    except ValueError as exc:
        raise ToolError(f"tanggal tidak valid: {tanggal!r}") from exc
    j = _jam(jam) if jam not in (None, "") else None
    orang = _bulat(jumlah_orang, "jumlah_orang") if jumlah_orang not in (None, "") else 1

    kapasitas = _TEMPAT[key]["kapasitas"]
    catatan = ""
    tersedia = True
    if d < _DEFAULT_TODAY:
        tersedia, catatan = False, "tanggal sudah lewat"
    elif j and (key, d.isoformat(), j) in _BOOKED:
        tersedia, catatan = False, "slot sudah dibooking"
    elif orang > kapasitas:
        tersedia, catatan = False, f"melebihi kapasitas {kapasitas} orang"

    return {
        "tempat": key,
        "tanggal": d.isoformat(),
        "jam": j,
        "jumlah_orang": orang,
        "kapasitas": kapasitas,
        "tersedia": tersedia,
        "catatan": catatan,
    }

# -- reporting tools --------------------------------------------------------

# One canonical label per period; ISO dates are looked up in the same tables.
_PERIODE_ALIAS = {
    "hari ini": "hari ini",
    "today": "hari ini",
    "kemarin": "kemarin",
    "minggu ini": "minggu ini",
    "pekan ini": "minggu ini",
    "bulan ini": "bulan ini",
    "bulan lalu": "bulan lalu",
    "bulan kemarin": "bulan lalu",
    "tahun ini": "tahun ini",
}

_PENJUALAN: dict[str, tuple[int, int]] = {  # periode -> (transaksi, total rupiah)
    "hari ini": (12, 2_450_000),
    "kemarin": (14, 2_900_000),
    "minggu ini": (88, 19_600_000),
    "bulan ini": (340, 78_900_000),
    "bulan lalu": (298, 64_200_000),
    "tahun ini": (2140, 486_000_000),
    "2026-10-05": (9, 1_780_000),
}

_PRODUK_TERLARIS: dict[str, tuple[str, int]] = {
    "hari ini": ("Kaos Polos Hitam", 5),
    "kemarin": ("Kaos Polos Hitam", 6),
    "minggu ini": ("Kaos Polos Hitam", 32),
    "bulan ini": ("Kaos Polos Hitam", 128),
    "bulan lalu": ("Kaos Polos Putih", 101),
    "tahun ini": ("Kaos Polos Hitam", 812),
    "2026-10-05": ("Kaos Polos Hitam", 3),
}

_PENDAPATAN_CABANG: dict[str, dict[str, int]] = {
    "pusat": {
        "hari ini": 980_000,
        "kemarin": 1_200_000,
        "minggu ini": 8_400_000,
        "bulan ini": 47_700_000,
        "bulan lalu": 25_400_000,
        "tahun ini": 190_000_000,
        "2026-10-05": 700_000,
    },
    "samarinda": {
        "hari ini": 1_470_000,
        "kemarin": 1_700_000,
        "minggu ini": 11_200_000,
        "bulan ini": 31_200_000,
        "bulan lalu": 38_800_000,
        "tahun ini": 296_000_000,
        "2026-10-05": 1_080_000,
    },
}

_LEADS: dict[str, int] = {
    "hari ini": 18,
    "kemarin": 25,
    "minggu ini": 96,
    "bulan ini": 412,
    "bulan lalu": 356,
    "tahun ini": 2840,
    "2026-10-05": 21,
}

_LEADS_SUMBER: dict[str, dict[str, int]] = {
    "instagram": {
        "hari ini": 7, "kemarin": 9, "minggu ini": 41,
        "bulan ini": 173, "bulan lalu": 149, "tahun ini": 1180, "2026-10-05": 8,
    },
    "whatsapp": {
        "hari ini": 6, "kemarin": 8, "minggu ini": 30,
        "bulan ini": 132, "bulan lalu": 116, "tahun ini": 900, "2026-10-05": 6,
    },
    "tiktok": {
        "hari ini": 3, "kemarin": 5, "minggu ini": 15,
        "bulan ini": 74, "bulan lalu": 61, "tahun ini": 480, "2026-10-05": 4,
    },
    "google": {
        "hari ini": 2, "kemarin": 3, "minggu ini": 10,
        "bulan ini": 33, "bulan lalu": 30, "tahun ini": 280, "2026-10-05": 3,
    },
}

def _periode(value: object) -> str:
    p = _teks(value)
    label = _PERIODE_ALIAS.get(p, p)
    if label not in _PENJUALAN:
        try:
            label = _dt.date.fromisoformat(label).isoformat()
        except ValueError as exc:
            raise ToolError(f"periode tidak dikenal: {value!r}") from exc
    if label not in _PENJUALAN:
        raise ToolError(f"data tidak tersedia untuk periode: {value!r}")
    return label

def laporan_penjualan(periode: str, metrik: str | None = None) -> dict:
    label = _periode(periode)
    if metrik not in (None, ""):
        if _teks(metrik) not in ("produk terlaris", "terlaris"):
            raise ToolError(f"metrik tidak dikenal: {metrik!r}")
        produk, unit = _PRODUK_TERLARIS[label]
        return {
            "periode": label,
            "metrik": "produk terlaris",
            "produk": produk,
            "unit": unit,
        }
    transaksi, total = _PENJUALAN[label]
    return {"periode": label, "transaksi": transaksi, "total": total}

def laporan_pendapatan(periode: str, cabang: str | None = None) -> dict:
    label = _periode(periode)
    if cabang in (None, ""):
        return {"periode": label, "cabang": "semua",
                "pendapatan": _PENJUALAN[label][1]}
    key = _pilih_kunci(cabang, _PENDAPATAN_CABANG, "cabang")
    nilai = _PENDAPATAN_CABANG[key].get(label)
    if nilai is None:
        raise ToolError(f"data cabang {key} tidak tersedia untuk periode {label}")
    return {"periode": label, "cabang": key, "pendapatan": nilai}

def laporan_leads(periode: str, sumber: str | None = None) -> dict:
    label = _periode(periode)
    if sumber in (None, ""):
        return {"periode": label, "sumber": "semua", "leads": _LEADS[label]}
    key = _pilih_kunci(sumber, _LEADS_SUMBER, "sumber")
    nilai = _LEADS_SUMBER[key].get(label)
    if nilai is None:
        raise ToolError(f"data sumber {key} tidak tersedia untuk periode {label}")
    return {"periode": label, "sumber": key, "leads": nilai}

_REGISTRY = {
    "parse_tanggal": parse_tanggal,
    "cari_penerbangan": cari_penerbangan,
    "cek_cuaca": cek_cuaca,
    "hitung": hitung,
    "sapaan_waktu": sapaan_waktu,
    "parse_alamat": parse_alamat,
    "cek_stok": cek_stok,
    "jam_buka_toko": jam_buka_toko,
    "cek_diskon": cek_diskon,
    "cek_booking": cek_booking,
    "laporan_penjualan": laporan_penjualan,
    "laporan_pendapatan": laporan_pendapatan,
    "laporan_leads": laporan_leads,
}


def execute(name: str, args: dict) -> dict:
    """Run a tool by name. Raises ToolError on bad name or arguments.

    Argument-validation failures are normalised to ToolError. The tools parse
    dates with the stdlib (which raises ValueError), call with missing or
    unexpected keyword arguments raises TypeError, and a non-string value where
    a string is expected raises AttributeError (`.strip()` on an int). All three
    are bad *arguments*, so the harness must surface a ToolError it can feed back
    to the model -- not an unhandled crash that the agent loop would mis-score as
    an API_ERROR.
    """
    if not isinstance(name, str):
        raise ToolError(f"nama tool harus string: {name!r}")
    fn = _REGISTRY.get(name)
    if fn is None:
        raise ToolError(f"tool tidak dikenal: {name!r}")
    try:
        return fn(**args)
    except ToolError:
        raise
    except (TypeError, ValueError, AttributeError) as exc:
        raise ToolError(f"argumen tidak valid untuk {name}: {exc}") from exc


def openai_schema() -> list[dict]:
    """Tool specs in OpenAI-compatible function-calling JSON."""
    out = []
    for s in TOOL_SPECS:
        props = {k: {"type": "string"} for k in s.parameters}
        out.append(
            {
                "type": "function",
                "function": {
                    "name": s.name,
                    "description": s.description,
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": s.required,
                    },
                },
            }
        )
    return out
