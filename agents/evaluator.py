from __future__ import annotations

"""Evaluator agent stub.

Evaluates backtest artifacts against KPIs.
"""

from typing import Dict, Any


# KPI profiles
DEFAULT_KPIS = {
    "max_drawdown": {"op": "<=", "value": 0.20},
    "sharpe": {"op": ">=", "value": 0.8},
    "win_rate": {"op": ">=", "value": 0.55},
    "trades": {"op": ">=", "value": 100},
}

DEMO_KPIS = {
    "max_drawdown": {"op": "<=", "value": 0.50},
    "sharpe": {"op": ">=", "value": 0.2},
    "win_rate": {"op": ">=", "value": 0.45},
    "trades": {"op": ">=", "value": 3},
}

KPI_PROFILES = {
    "strict": DEFAULT_KPIS,
    "demo": DEMO_KPIS,
}


def get_kpis(profile: str | None) -> dict:
    if not profile:
        return DEFAULT_KPIS
    return KPI_PROFILES.get(profile, DEFAULT_KPIS)


def evaluate(metrics: Dict[str, Any], kpis: Dict[str, Any] | None = None) -> Dict[str, Any]:
    kpis = kpis or DEFAULT_KPIS
    results: Dict[str, Any] = {"passed": True, "checks": {}}
    for k, rule in kpis.items():
        op = rule["op"]
        threshold = rule["value"]
        value = metrics.get(k)
        passed = False
        if value is None:
            passed = False
        elif op == "<=":
            passed = value <= threshold
        elif op == ">=":
            passed = value >= threshold
        results["checks"][k] = {"value": value, "op": op, "threshold": threshold, "passed": passed}
        results["passed"] = results["passed"] and passed
    return results


