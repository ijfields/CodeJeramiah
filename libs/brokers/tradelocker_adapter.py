from __future__ import annotations

from typing import Any, Dict

from credentials import TRADELOCKER_CONFIG


class TradeLockerAdapter:
    """Minimal TradeLocker/TrendMarkets adapter stub (forex indices)."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = {**TRADELOCKER_CONFIG, **(config or {})}

    def place_order(self, symbol: str, qty: float, side: str) -> Dict[str, Any]:
        # Placeholder only
        return {"status": "ok", "symbol": symbol, "qty": qty, "side": side}


