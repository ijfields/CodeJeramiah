from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd


def _fill_date(pattern: str, d: date) -> str:
    yyyy = f"{d.year:04d}"
    mm = f"{d.month:02d}"
    dd = f"{d.day:02d}"
    return (
        pattern.replace("{YYYY}", yyyy)
        .replace("{MM}", mm)
        .replace("{DD}", dd)
    )


def build_daily_paths(dates: Sequence[date], pattern: str) -> List[Path]:
    return [Path(_fill_date(pattern, d)) for d in dates]


def date_range(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def read_existing_parquet(paths: Sequence[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for p in paths:
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


