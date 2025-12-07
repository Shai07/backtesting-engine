# src/core/analytics.py

import pandas as pd
import numpy as np
from src.core.portfolio import Portfolio


class AnalyticsEngine:
    def __init__(self, initial_cash: float):
        self.daily_metrics = []
        self.initial_cash = initial_cash

    def record_day(self, date: pd.Timestamp, portfolio: Portfolio):
        """
        Called daily by the BacktestEngine to record daily portfolio metrics
        """

        current_market_value = portfolio.get_market_value()
        current_delta_exposure = portfolio.get_delta_exposure()

        self.daily_metrics.append(
            {
                "date": date,
                "market_value": current_market_value,
                "delta_exposure": current_delta_exposure,
            }
        )

    def generate_report(
        self, risk_free_rate: float, annualized_days: int = 252
    ) -> pd.DataFrame:
        """
        Process the logged data and calculate final performance metrics.
        """
        if not self.daily_metrics:
            return pd.DataFrame()

        df = pd.DataFrame(self.daily_metrics).set_index("date")

        # Daily returns
        df["market_value"] = pd.to_numeric(df["market_value"], errors="coerce")
        df["daily_return"] = df["market_value"].pct_change()

        # First day returns
        if self.initial_cash > 0:
            first_day_return = (
                df["market_value"].iloc[0] - self.initial_cash
            ) / self.initial_cash
            df.loc[df.index[0], "daily_return"] = first_day_return

        # Clean daily returns - remove NaN and inf values
        daily_returns = df["daily_return"].replace([np.inf, -np.inf], np.nan).dropna()

        # Cumulative Returns
        cumulative_return = (daily_returns + 1).prod().item() - 1

        # Annualized Returns and Volatility
        annualized_return = daily_returns.mean() * annualized_days
        annualized_volatility = daily_returns.std() * np.sqrt(annualized_days)

        # Sharpe Ratio
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

        # Maximum Drawdown
        df["cumulative_peak"] = df["market_value"].cummax()
        df["drawdown"] = (df["market_value"] - df["cumulative_peak"]) / df[
            "cumulative_peak"
        ]
        max_drawdown = df["drawdown"].min()

        # Downside returns and Sortino Ratio
        downside_returns = daily_returns[
            daily_returns < risk_free_rate / annualized_days
        ]
        downside_volatility = downside_returns.std() * np.sqrt(annualized_days)

        if downside_volatility != 0:
            sortino_ratio = (annualized_return - risk_free_rate) / downside_volatility
        else:
            sortino_ratio = np.inf

        # Historical VaR at 99% confidence level
        historical_var = -daily_returns.quantile(0.01)

        # Final report
        report_data = {
            "Start Date": df.index[0].date(),
            "End Date": df.index[-1].date(),
            "Cumulative Return": f"{cumulative_return:.2%}",
            "Annualized Return": f"{annualized_return:.2%}",
            "Annualized Volatility": f"{annualized_volatility:.2%}",
            "Sharpe Ratio (Rf={risk_free_rate})": f"{sharpe_ratio:.2f}",
            "Sortino Ratio (Rf={risk_free_rate})": f"{sortino_ratio:.2f}",
            "Max Drawdown": f"{max_drawdown:.2%}",
            "Value-at-Risk (99%)": f"{historical_var:.2%}",
        }

        return pd.DataFrame([report_data]).T
