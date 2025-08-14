from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from rich import print as rprint
import pandas as pd
import numpy as np

try:
    import yfinance as yf
except Exception:  # yfinance optional until demo mode used
    yf = None  # type: ignore

sys.path.append(str(Path(__file__).resolve().parents[1]))


def load_json_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json_file(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def build_run_id(now: datetime) -> str:
    # Use YYYYMMDD_HHMMSS format for readability
    return now.strftime("%Y%m%d_%H%M%S")


def create_output_placeholders(output_dir: Path, outputs: List[str]) -> None:
    for filename in outputs:
        file_path = output_dir / filename
        ensure_directory(file_path.parent)
        if filename.endswith(".json"):
            write_json_file(file_path, {"status": "placeholder"})
        else:
            # Create empty placeholder text/HTML/CSV files
            file_path.write_text("", encoding="utf-8")


def compute_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    rolling_max = equity_curve.cummax()
    drawdowns = (equity_curve - rolling_max) / rolling_max
    return float(-drawdowns.min())


def run_demo_backtest(spec: Dict[str, Any], run_dir: Path, symbol_override: str | None = None) -> Dict[str, Any]:
    if yf is None:
        raise RuntimeError(
            "yfinance is not installed. Install requirements and retry demo mode."
        )
    symbol = symbol_override or (spec.get("universe") or ["SPY"])[0]
    bt = spec.get("backtest", {})
    start = bt.get("start", "2020-01-01")
    end = bt.get("end", datetime.now().strftime("%Y-%m-%d"))
    # yfinance uses an exclusive end date for download; bump by +1 day to include the intended end date
    try:
        end_dt = pd.to_datetime(end, errors="coerce")
        if pd.notnull(end_dt):
            end = (end_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    except Exception:
        pass

    # Use daily data for reliability
    df = yf.download(
        symbol, start=start, end=end, interval="1d", auto_adjust=False, progress=False
    )
    if df.empty:
        raise RuntimeError(f"No data returned for {symbol} {start}..{end}")
    # Choose adjusted close if present, otherwise fall back to close
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    df = df.reset_index()
    # Normalize date column and keep a pre-formatted string to avoid per-row formatting issues
    if "Date" in df.columns:
        df["date"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        # Fallback if index name differs across versions
        df["date"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

    df["price"] = df[price_col]
    df["return"] = df["price"].pct_change().fillna(0.0)

    # Simple SMA crossover
    fast = 10
    slow = 20
    df["sma_fast"] = df["price"].rolling(fast).mean()
    df["sma_slow"] = df["price"].rolling(slow).mean()
    df["signal"] = (df["sma_fast"] > df["sma_slow"]).astype(int)
    df["position"] = df["signal"].shift(1).fillna(0)
    df["strategy_return"] = df["position"] * df["return"]
    equity = (1 + df["strategy_return"]).cumprod()

    # Generate naive trades: buy when position goes 0->1, sell when 1->0
    trades: List[Dict[str, Any]] = []
    prev_pos = 0
    entry_price = None
    entry_date = None
    for _, row in df.iterrows():
        pos_val = row["position"]
        try:
            pos = int(pos_val)
        except Exception:
            # Handle potential single-element Series
            try:
                pos = int(pos_val.iloc[0])  # type: ignore[attr-defined]
            except Exception:
                pos = 0
        if prev_pos == 0 and pos == 1:
            entry_price = row["price"]
            entry_date = row["date_str"]
            trades.append({
                "date": entry_date,
                "symbol": symbol,
                "side": "BUY",
                "price": float(entry_price),
            })
        elif prev_pos == 1 and pos == 0 and entry_price is not None:
            exit_price = row["price"]
            exit_date = row["date_str"]
            pnl = float(exit_price - entry_price)
            trades.append({
                "date": exit_date,
                "symbol": symbol,
                "side": "SELL",
                "price": float(exit_price),
                "pnl": pnl,
            })
            entry_price = None
            entry_date = None
        prev_pos = pos

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df.to_csv(run_dir / "trades.csv", index=False)

    # Stats
    daily_ret = df["strategy_return"].dropna()
    sharpe = float(np.sqrt(252) * (daily_ret.mean() / (daily_ret.std() + 1e-12)))
    max_dd = compute_drawdown(equity)
    num_trades = int((trades_df[trades_df["side"] == "SELL"].shape[0]) if not trades_df.empty else 0)
    wins = int(((trades_df["pnl"] > 0).sum()) if (not trades_df.empty and "pnl" in trades_df.columns) else 0)
    win_rate = float(wins / num_trades) if num_trades > 0 else 0.0

    stats_df = pd.DataFrame([
        {
            "symbol": symbol,
            "start": start,
            "end": end,
            "sharpe": round(sharpe, 3),
            "max_drawdown": round(max_dd, 3),
            "trades": num_trades,
            "win_rate": round(win_rate, 3),
            "final_equity": round(float(equity.iloc[-1]), 3),
        }
    ])
    stats_df.to_csv(run_dir / "stats.csv", index=False)
    stats_df.to_csv(run_dir / "tearsheet.csv", index=False)

    return {
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "trades": num_trades,
        "win_rate": win_rate,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a backtest for a strategy spec")
    parser.add_argument(
        "--spec",
        required=True,
        help="Path to strategy_spec.json (e.g. strategies/zrce_second_chance/strategy_spec.json)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a simple SMA crossover demo to produce real artifacts",
    )
    parser.add_argument(
        "--symbol",
        help="Override universe[0] symbol for demo mode (e.g., SPY, AAPL)",
        default=None,
    )

    args = parser.parse_args(argv)
    spec_path = Path(args.spec)
    if not spec_path.exists():
        rprint(f"[red]Spec not found:[/red] {spec_path}")
        return 1

    spec = load_json_file(spec_path)
    slug = spec.get("slug") or spec_path.parent.name
    outputs: List[str] = spec.get("outputs", [
        "settings.json",
        "tearsheet.csv",
        "tearsheet.html",
        "trades.csv",
        "trades.html",
        "stats.csv",
    ])

    # Prepare results directory
    results_root = spec_path.parent / "results"
    ensure_directory(results_root)
    run_id = build_run_id(datetime.now())
    run_dir = results_root / run_id
    ensure_directory(run_dir)

    # Save the resolved settings used for the run
    settings = {
        "spec_path": str(spec_path.resolve()),
        "slug": slug,
        "run_id": run_id,
        "started_at": datetime.now().isoformat(),
    }
    write_json_file(run_dir / "settings.json", settings)

    # Create placeholders for expected artifacts
    create_output_placeholders(run_dir, outputs)

    # Optionally run a simple demo backtest to produce real CSVs
    if args.demo:
        try:
            metrics = run_demo_backtest(spec, run_dir, symbol_override=args.symbol)
            # Evaluate against KPIs and persist
            try:
                from agents.evaluator import evaluate, get_kpis  # lazy import

                # Use a relaxed KPI profile for demo runs so you can get a green check
                evaluation = evaluate(metrics, get_kpis("demo"))
                write_json_file(run_dir / "evaluation.json", evaluation)
            except Exception as eval_exc:
                rprint(f"[yellow]Evaluation skipped:[/yellow] {eval_exc}")
            rprint(f"[cyan]Demo backtest metrics:[/cyan] {metrics}")
        except Exception as exc:
            rprint(f"[red]Demo backtest failed:[/red] {exc}")

    rprint(
        f"[green]Initialized run[/green] slug=[bold]{slug}[/bold] run_id=[bold]{run_id}[/bold] -> {run_dir}"
    )
    rprint(
        "This is a minimal placeholder runner. Next steps: integrate Lumibot strategy execution and metrics."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


