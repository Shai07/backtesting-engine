# Options Backtesting Engine

Python-based modular and event-driven backtesting framework for the simulation and evaluation of systematic options trading strategies on historical datasets.

## Main Features:

* Event-Driven Architecture: The engine processes historical market data on a daily basis. Each trading day triggers a deterministic sequence to simulate live trading: Process concurrent data streams → Generate trading signals from strategy → Execute orders in portfolio. This simulates past performance by mutating the portfolio state according to daily data.

* Efficient Data Handling: Specialized multi-data loader classes enable concurrent streaming of OHLC and options data, leveraging chunk-loading for performance with large datasets.

* Performance Analytics: The analytics engine computes final performance metrics, including Sharpe Ratio, Sortino Ratio, Max Drawdown, and Historical Value-at-Risk (VaR 99%).

## Module Overview:

| Module     | Location                     | Summary                                                                 |
|------------|------------------------------|-------------------------------------------------------------------------|
| Engine     | `src/core/engine.py`          | Manages the primary simulation loop, controls data flow, executes portfolio updates |
| Portfolio  | `src/core/portfolio.py`       | Handles portfolio state, tracks current options and equity positions, calculates delta hedging PnL |
| Analytics  | `src/core/analytics.py`       | Records daily metrics and computes final performance and risk statistics |
| Strategies | `src/strategies/`             | Holds all trading logic, implemented via the `Strategy` base class       |

## Installation:

To install and import modules from the source code:

```
git clone https://github.com/liam-duke/backtesting-engine.git
cd backtesting-engine

# Optional virtual environment setup
python -m venv .venv
source .venv/bin/activate  # MacOS/Linux
.venv\scripts\activate     # Windows

pip install -r requirements.txt
```

## Project Structure:

```
backtesting-engine/
├── .gitignore
├── requirements.txt
└── src/
    ├── core/
    │   ├── analytics.py        # Performance metrics
    │   ├── constants.py        # Options/Equity column definitions
    │   ├── data.py             # Data Loaders (Ohlc, Options, MultiDataLoader)
    │   ├── engine.py           # Core backtesting loop
    │   └── portfolio.py        # Portfolio state, current positions, PnL
    ├── strategies/
    │   ├── base.py             # Strategy interface
    │   └── volatility_carry.py # Example volatility carry implementation
    └── utils/
        └── logger.py
```

## License

This project is licensed under the MIT License. See `backtesting-engine/LICENSE` for details.
