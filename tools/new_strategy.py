from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict


SPEC_SKELETON: Dict[str, Any] = {
    "version": "1.0",
    "name": "",
    "slug": "",
    "universe": ["SPY"],
    "timeframe": "1min",
    "entry": {"signal_logic": [], "enter_time": "09:45:00"},
    "legs": [],
    "risk_exits": {},
    "sizing": {},
    "data": {
        "source": "polygon_flat",
        "paths": {
            "equities_min": "data/processed/equities_min/{YYYY}/{MM}/{YYYY}-{MM}-{DD}.parquet",
            "equities_5min": "data/processed/equities_5min/{YYYY}/{MM}/{YYYY}-{MM}-{DD}.parquet",
            "options_min": "data/processed/options_min/{YYYY}/{MM}/{YYYY}-{MM}-{DD}.parquet",
        },
        "tz": "America/New_York",
    },
    "broker": {"name": "tradier", "account_type": "paper"},
    "backtest": {
        "start": "2023-01-01",
        "end": "2024-01-01",
        "cash": 100000,
        "fees": {"percent_fee": 0.001},
    },
    "outputs": [
        "settings.json",
        "tearsheet.csv",
        "tearsheet.html",
        "trades.csv",
        "trades.html",
        "stats.csv",
    ],
}


def slug_to_class_name(slug: str) -> str:
    words = re.split(r"[^a-zA-Z0-9]+", slug)
    return "".join(w.capitalize() for w in words if w)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a new strategy scaffold")
    parser.add_argument("--name", required=True, help="Human-readable strategy name")
    parser.add_argument("--slug", required=True, help="Folder slug (strategies/{slug})")
    args = parser.parse_args()

    name = args.name
    slug = args.slug

    target_dir = Path("strategies") / slug
    target_dir.mkdir(parents=True, exist_ok=True)

    spec = {**SPEC_SKELETON}
    spec["name"] = name
    spec["slug"] = slug
    spec_path = target_dir / "strategy_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")

    class_name = slug_to_class_name(slug)
    strategy_py = target_dir / "lumibot_strategy.py"
    code = f"""
try:
    from lumibot.strategies.strategy import Strategy
except Exception:
    Strategy = object  # type: ignore


class {class_name}(Strategy):
    """Auto-generated strategy stub."""

    def initialize(self):
        pass

    def on_trading_iteration(self):
        pass
"""
    strategy_py.write_text(code.strip() + "\n", encoding="utf-8")

    print(f"Created strategy in {target_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


