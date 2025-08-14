try:
    from lumibot.strategies.strategy import Strategy
except Exception:  # lumibot may not be installed yet
    Strategy = object  # type: ignore


class ZrceSecondChance(Strategy):
    """ZRCE Second Chance strategy stub for Lumibot."""

    def initialize(self):
        pass

    def on_trading_iteration(self):
        pass


