from __future__ import annotations

"""Helpers to construct option/future/forex legs from spec items (stub)."""

from typing import Dict, Any


def build_leg(leg_spec: Dict[str, Any]) -> Dict[str, Any]:
    return {"built": True, **leg_spec}


