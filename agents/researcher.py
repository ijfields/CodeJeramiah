from __future__ import annotations

"""Researcher agent stub.

Responsible for collecting market hypotheses, signals, and constraints
to inform a concrete strategy spec.
"""

from typing import Dict, Any


def propose_strategy_inputs(topic: str) -> Dict[str, Any]:
    return {
        "topic": topic,
        "notes": "Stub: gather data sources, hypotheses, and constraints",
    }


