from __future__ import annotations
import argparse, json, subprocess, sys
from datetime import datetime, timedelta
from pathlib import Path

def load_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8-sig") as f:
        return json.load(f)

def write_json(p: Path, data: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def date_iter(start: datetime, end: datetime, step_days: int):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=step_days)

def main() -> int:
    ap = argparse.ArgumentParser(description="Run grid of backtests by date window")
    ap.add_argument("--spec", required=True, help="Path to base strategy_spec.json")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (window start)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--window-days", type=int, default=1, help="Days per window (default: 1)")
    ap.add_argument("--step-days", type=int, default=1, help="Days to step between windows (default: 1)")
    ap.add_argument("--symbol", default=None, help="Override symbol for demo")
    ap.add_argument("--demo", action="store_true", help="Use runner --demo mode")
    ap.add_argument("--keep-specs", action="store_true", help="Keep generated temp specs")
    args = ap.parse_args()

    base_spec_path = Path(args.spec).resolve()
    base = load_json(base_spec_path)
    slug = base.get("slug") or base_spec_path.parent.name

    d0 = datetime.strptime(args.start, "%Y-%m-%d")
    d1 = datetime.strptime(args.end, "%Y-%m-%d")
    win = max(1, args.window_days)

    tmp_dir = base_spec_path.parent / "tmp_specs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    ran = 0
    for s in date_iter(d0, d1 - timedelta(days=win - 1), args.step_days):
        e = s + timedelta(days=win - 1)
        spec = dict(base)
        spec.setdefault("backtest", {})
        spec["backtest"]["start"] = s.strftime("%Y-%m-%d")
        spec["backtest"]["end"] = e.strftime("%Y-%m-%d")

        tmp_spec = tmp_dir / f"spec_{s.strftime('%Y%m%d')}_{e.strftime('%Y%m%d')}.json"
        write_json(tmp_spec, spec)

        cmd = [sys.executable, "agents/runner.py", "--spec", str(tmp_spec)]
        if args.demo:
            cmd.append("--demo")
        if args.symbol:
            cmd += ["--symbol", args.symbol]

        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        if not args.keep_specs:
            try:
                tmp_spec.unlink(missing_ok=True)
            except Exception:
                pass
        ran += 1

    print(f"Completed {ran} runs.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())