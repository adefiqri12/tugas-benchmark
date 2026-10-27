"""The fixed evaluation date.

Relative-date tasks ("besok", "minggu depan") must resolve to the same calendar
dates on every run, forever. One constant is the source of truth: tools parse
against it, the raw-task builder stamps it into the dataset meta, and tests
assert the three never drift.
"""

from __future__ import annotations

import datetime as dt

ANCHOR_DATE = "2026-10-07"
ANCHOR = dt.date.fromisoformat(ANCHOR_DATE)
