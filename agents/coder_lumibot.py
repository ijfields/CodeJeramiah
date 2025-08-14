from __future__ import annotations

"""Lumibot Coder agent stub.

Translates a Strategy Spec into a Lumibot strategy class file.
"""

from pathlib import Path


def generate_strategy_file(target_path: Path, class_name: str) -> None:
    code = f"""
try:
    from lumibot.strategies.strategy import Strategy
except Exception:  # lumibot may not be installed yet
    Strategy = object  # type: ignore


class {class_name}(Strategy):
    '''Auto-generated strategy stub.'''

    def initialize(self):
        pass

    def on_trading_iteration(self):
        pass
"""
    target_path.write_text(code.strip() + "\n", encoding="utf-8")


