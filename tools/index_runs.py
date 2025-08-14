from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

from libs.registry import RunMeta, upsert_runs, read_evaluation




@dataclass
class RunRecord:
    slug: str
    run_id: str
    path: str


def find_runs(strategies_root: Path) -> List[RunRecord]:
    records: List[RunRecord] = []
    for strat_dir in strategies_root.iterdir():
        if not strat_dir.is_dir():
            continue
        results_dir = strat_dir / "results"
        if not results_dir.exists():
            continue
        for run_dir in results_dir.iterdir():
            if not run_dir.is_dir():
                continue
            records.append(
                RunRecord(slug=strat_dir.name, run_id=run_dir.name, path=str(run_dir))
            )
    return records


def main() -> int:
    strategies_root = Path("strategies")
    runs = find_runs(strategies_root)
    index = [asdict(r) for r in runs]
    out_path = Path("logs") / "runs_index.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    # Also update SQLite registry with metrics/pass state if available
    metas: List[RunMeta] = []
    for r in runs:
        eval_obj = read_evaluation(Path(r.path))
        passed = None
        sharpe = max_dd = win_rate = None
        trades = None
        if eval_obj:
            passed = bool(eval_obj.get("passed"))
            checks = eval_obj.get("checks", {})
            if isinstance(checks, dict):
                sharpe = checks.get("sharpe", {}).get("value") if checks.get("sharpe") else None
                max_dd = checks.get("max_drawdown", {}).get("value") if checks.get("max_drawdown") else None
                win_rate = checks.get("win_rate", {}).get("value") if checks.get("win_rate") else None
                trades = checks.get("trades", {}).get("value") if checks.get("trades") else None
        metas.append(
            RunMeta(
                slug=r.slug,
                run_id=r.run_id,
                path=r.path,
                passed=passed,
                sharpe=sharpe,
                max_drawdown=max_dd,
                win_rate=win_rate,
                trades=trades,
            )
        )
    upsert_runs(metas)
    print(f"Indexed {len(index)} runs -> {out_path} and logs/runs.db")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


