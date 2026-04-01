# src/core/engine.py

import pandas as pd
from portfolio import Portfolio
from core.data import MultiDataLoader
from strategies.base import Strategy
from core.analytics import AnalyticsEngine
from rich.console import Console


class BacktestEngine:
    def __init__(
        self,
        portfolio: Portfolio,
        data_loaders: dict,
        analytics: AnalyticsEngine,
        risk_free_rate: float,
    ):
        self.portfolio = portfolio
        self.data_loaders = data_loaders
        self.analytics = analytics
        self.risk_free_rate = risk_free_rate
        self.console = Console()

    def run(
        self, strategy: Strategy, delta_hedging: bool, start_date=None, end_date=None
    ):

        multi_data_loader = MultiDataLoader(self.data_loaders, start_date, end_date)

        for date, market_data in multi_data_loader.daily_multi_stream():
            ohlc_data = market_data.get("ohlc")
            close = ohlc_data["close"]
            if close is None:
                self.analytics.record_day(date, self.portfolio)
                continue

            # Trade execution and portfolio update
            self.portfolio.handle_expired_options(date)
            current_options = self.portfolio.get_options()
            orders_df = strategy.process_data(market_data, current_options)

            if orders_df is not None and close is not None:
                option_orders = orders_df[orders_df["secid"].notna()]
                equity_orders = orders_df[orders_df["secid"].isna()]

                self.portfolio.update_options(option_orders)
                self.portfolio.update_equities(equity_orders)

                if delta_hedging:
                    self.portfolio.hedge_delta(
                        close,
                        commission_per_share=0.01,
                        base_spread=0.005,
                        spread_std=0.001,
                    )

            # Record daily performance
            self.analytics.record_day(date, self.portfolio)

        # Generate and print final performance report
        final_report = self.analytics.generate_report(self.risk_free_rate)
        print("Backtest Performance Report")
        self.console.print(final_report)
