"""Mock Indonesian tools the agent can call.

Deterministic, offline, no network — so a harness run is reproducible and the
score measures the *agent*, not an upstream API's mood. Each tool validates its
own arguments and returns a JSON-able dict.
"""

from __future__ import annotations

import datetime as _dt
import re as _re

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
]


class ToolError(Exception):
    """Raised when a tool is called with bad arguments."""


# -- implementations --------------------------------------------------------
_BULAN = {
    "januari": 1, "februari": 2, "maret": 3, "april": 4, "mei": 5, "juni": 6,
    "juli": 7, "agustus": 8, "september": 9, "oktober": 10, "november": 11,
    "desember": 12,
}

_DEFAULT_TODAY = _dt.date(2026, 10, 7)  # fixed anchor -> reproducible evals

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
    try:
        val = eval(ekspresi, {"__builtins__": {}}, {})  # noqa: S307 - charset-guarded
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"gagal menghitung: {ekspresi!r}") from exc
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


_REGISTRY = {
    "parse_tanggal": parse_tanggal,
    "cari_penerbangan": cari_penerbangan,
    "cek_cuaca": cek_cuaca,
    "hitung": hitung,
    "sapaan_waktu": sapaan_waktu,
    "parse_alamat": parse_alamat,
}


def execute(name: str, args: dict) -> dict:
    """Run a tool by name. Raises ToolError on bad name or arguments."""
    fn = _REGISTRY.get(name)
    if fn is None:
        raise ToolError(f"tool tidak dikenal: {name!r}")
    return fn(**args)


def openai_schema() -> list[dict]:
    """Tool specs in [OI]-compatible function-calling JSON."""
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
