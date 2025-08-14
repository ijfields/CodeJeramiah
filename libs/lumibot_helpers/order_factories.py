from __future__ import annotations

"""Factories for creating orders from legs/signals (stub)."""

from typing import Dict, Any


def create_order_from_leg(leg: Dict[str, Any]) -> Dict[str, Any]:
    return {"order": True, **leg}


