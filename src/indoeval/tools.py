"""Mock Indonesian tools the agent can call.

Deterministic, offline, no network — so a harness run is reproducible and the
score measures the *agent*, not an upstream API's mood. Each tool validates its
own arguments and returns a JSON-able dict.
"""

from __future__ import annotations

import datetime as _dt

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


def parse_tanggal(teks: str, acuan: str | None = None) -> dict:
    base = _dt.date.fromisoformat(acuan) if acuan else _DEFAULT_TODAY
    t = (teks or "").strip().lower()

    if t in ("hari ini", "today"):
        d = base
    elif t in ("besok", "esok"):
        d = base + _dt.timedelta(days=1)
    elif t in ("kemarin",):
        d = base - _dt.timedelta(days=1)
    elif "minggu depan" in t or "pekan depan" in t:
        d = base + _dt.timedelta(days=7)
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


_REGISTRY = {
    "parse_tanggal": parse_tanggal,
    "cari_penerbangan": cari_penerbangan,
    "cek_cuaca": cek_cuaca,
    "hitung": hitung,
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
