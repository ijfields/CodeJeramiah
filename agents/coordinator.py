from __future__ import annotations

"""Coordinator agent stub.

Maintains a simple ops/tasks.json queue for subagents to coordinate work.
"""

from pathlib import Path
from typing import Any, Dict, List
import json


TASKS_PATH = Path("ops/tasks.json")


def _load() -> Dict[str, Any]:
    if not TASKS_PATH.exists():
        return {"tasks": []}
    with TASKS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save(data: Dict[str, Any]) -> None:
    TASKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TASKS_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def add_task(task: Dict[str, Any]) -> str:
    data = _load()
    tasks: List[Dict[str, Any]] = data.setdefault("tasks", [])
    tasks.append(task)
    _save(data)
    return task.get("id", "")


def set_state(task_id: str, state: str) -> bool:
    data = _load()
    for t in data.get("tasks", []):
        if t.get("id") == task_id:
            t["state"] = state
            _save(data)
            return True
    return False


def list_tasks(state: str | None = None) -> List[Dict[str, Any]]:
    data = _load()
    tasks: List[Dict[str, Any]] = data.get("tasks", [])
    if state:
        return [t for t in tasks if t.get("state") == state]
    return tasks


