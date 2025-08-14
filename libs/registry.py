from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


DB_DEFAULT = Path("logs") / "runs.db"


@dataclass
class RunMeta:
    slug: str
    run_id: str
    path: str
    passed: Optional[bool]
    sharpe: Optional[float]
    max_drawdown: Optional[float]
    win_rate: Optional[float]
    trades: Optional[int]


def ensure_db(db_path: Path = DB_DEFAULT) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                slug TEXT NOT NULL,
                run_id TEXT NOT NULL,
                path TEXT NOT NULL,
                passed INTEGER,
                sharpe REAL,
                max_drawdown REAL,
                win_rate REAL,
                trades INTEGER,
                PRIMARY KEY (slug, run_id)
            )
            """
        )


def upsert_runs(records: Iterable[RunMeta], db_path: Path = DB_DEFAULT) -> None:
    ensure_db(db_path)
    with sqlite3.connect(db_path) as conn:
        for r in records:
            conn.execute(
                """
                INSERT INTO runs (slug, run_id, path, passed, sharpe, max_drawdown, win_rate, trades)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(slug, run_id) DO UPDATE SET
                    path=excluded.path,
                    passed=excluded.passed,
                    sharpe=excluded.sharpe,
                    max_drawdown=excluded.max_drawdown,
                    win_rate=excluded.win_rate,
                    trades=excluded.trades
                """,
                (
                    r.slug,
                    r.run_id,
                    r.path,
                    int(r.passed) if r.passed is not None else None,
                    r.sharpe,
                    r.max_drawdown,
                    r.win_rate,
                    r.trades,
                ),
            )


def read_evaluation(run_dir: Path):
    eval_path = run_dir / "evaluation.json"
    if not eval_path.exists():
        return None
    try:
        return json.loads(eval_path.read_text(encoding="utf-8"))
    except Exception:
        return None


