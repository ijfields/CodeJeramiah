from __future__ import annotations

from typing import Any, Dict

from credentials import TRADIER_CONFIG


class TradierAdapter:
    """Minimal Tradier adapter stub (options first)."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = {**TRADIER_CONFIG, **(config or {})}

    def place_order(self, symbol: str, qty: int, side: str) -> Dict[str, Any]:
        # Placeholder only
        return {"status": "ok", "symbol": symbol, "qty": qty, "side": side}


