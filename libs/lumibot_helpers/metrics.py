from __future__ import annotations

"""Basic metrics utilities (stub)."""

from typing import Dict, Any


def summarize_trades(trades) -> Dict[str, Any]:  # trades: DataFrame-like
    # Placeholder implementation
    return {
        "max_drawdown": None,
        "sharpe": None,
        "win_rate": None,
        "trades": getattr(trades, "shape", (0,))[0] if hasattr(trades, "shape") else 0,
    }


