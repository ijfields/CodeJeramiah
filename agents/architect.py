from __future__ import annotations

"""Strategy Architect agent stub.

Converts research notes into a concrete Strategy Spec v1 structure.
"""

from typing import Dict, Any


def build_spec_skeleton(name: str, slug: str) -> Dict[str, Any]:
    return {
        "version": "1.0",
        "name": name,
        "slug": slug,
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


