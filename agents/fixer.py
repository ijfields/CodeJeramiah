from __future__ import annotations

"""Fixer agent stub.

Suggests adjustments to strategy parameters when KPIs fail.
"""

from typing import Dict, Any, List


def suggest_changes(failed_checks: Dict[str, Any]) -> List[str]:
    suggestions: List[str] = []
    if not failed_checks:
        return suggestions
    for k, info in failed_checks.items():
        if not info.get("passed"):
            suggestions.append(f"Adjust parameter related to {k} toward {info['op']} {info['threshold']}")
    return suggestions


