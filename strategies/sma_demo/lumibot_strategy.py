try:
    from lumibot.strategies.strategy import Strategy
except Exception:  # lumibot may not be installed yet
    Strategy = object  # type: ignore


class SmaDemo(Strategy):
    '''Auto-generated strategy stub.'''

    def initialize(self):
        pass

    def on_trading_iteration(self):
        pass
