import pandas as pd
from pathlib import Path
from rich.console import Console
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

pd.set_option("display.max_rows", 1000)

console = Console()

from src.core.data import OptionsDataLoader, OhlcDataLoader, MultiDataLoader
from src.core.portfolio import Portfolio
from src.strategies.volatility_carry import VolatilityCarry

OPTIONS_DATA_PATH = Path("data/spx_options_samples_~1y.csv")
OHLC_DATA_PATH = Path("data/^SPX_history_2013-01-01_2015-01-1.csv")

END_DATE = "2014-01-01"

options_data = OptionsDataLoader(OPTIONS_DATA_PATH, "date", chunksize=10000)
ohlc_data = OhlcDataLoader(OHLC_DATA_PATH, "date")

volatility_carry = VolatilityCarry(
    30, 1.2, 1.8, min_dte=23, max_dte=30, max_positions=40
)

INTIAL_CASH = 1000000
portfolio = Portfolio(INTIAL_CASH)

multi_data_loader = MultiDataLoader(
    {"options": options_data, "ohlc": ohlc_data},
    end_date=END_DATE,
)

END_DATE = pd.to_datetime(END_DATE)
START_DATE = pd.to_datetime("2013-01-01")
total_days = (END_DATE - START_DATE).days

closes = []
portfolio_value = []
dates = []
day = 0

for date, market_data in multi_data_loader.daily_multi_stream():
    ohlc_data = market_data["ohlc"]
    spot = ohlc_data["close"]
    print(f"{(date - START_DATE).days / total_days * 100:.1f}%")
    orders = volatility_carry.process_data(market_data, portfolio.get_options())
    portfolio.update_options(orders)
    portfolio.handle_expired_options(date)
    if closes:
        portfolio.update_delta_pnl(spot, spot - closes[-1], 0.005, 0, 0)
    portfolio_value.append(portfolio.get_net_asset_value() + portfolio.get_cash())
    options = portfolio.get_options()
    if not options.empty:
        console.print(options)
    dates.append(date)

# Calculate portfolio percent returns
portfolio_pct_returns = [(value / INTIAL_CASH) * 10 - 1 for value in portfolio_value]
print(dates)

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.plot(dates, portfolio_pct_returns)
plt.show()
