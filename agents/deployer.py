from __future__ import annotations

"""Deployer agent stub.

Handles promotion to paper/demo and later to live with approvals.
"""

def deploy(slug: str, environment: str = "demo") -> str:
    return f"Stub deploy: {slug} → {environment}"


