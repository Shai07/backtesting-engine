# src/core/portfolio.py

import pandas as pd
import numpy as np
from src.core.constants import OPTION_COLUMNS, EQUITY_COLUMNS


class Portfolio:
    def __init__(self, initial_cash: float):
        self.options = pd.DataFrame(columns=OPTION_COLUMNS)
        self.equities = pd.DataFrame(columns=EQUITY_COLUMNS)
        self.net_asset_value = 0
        self.cash = initial_cash

        self.shares_owned = 0

    def get_options(self):
        return self.options

    def get_equities(self):
        return self.equities

    def get_net_asset_value(self):
        return self.net_asset_value

    def get_cash(self):
        return self.cash

    def get_greek_exposure(self, greek: str) -> dict[str, float]:
        if self.options.empty:
            return {}

        greek_exposure_map = (
            self.options.assign(
                greek_exposure=self.options[greek] * self.options["quantity"] * 100
            )
            .groupby("symbol")["greek_exposure"]
            .sum()
            .to_dict()
        )

        return greek_exposure_map

    def update_equities(self, equity_orders: pd.DataFrame | None):
        """
        Update current equity positions to market
        """
        if equity_orders is None:
            return

        buy_orders = equity_orders[equity_orders["quantity"] > 0]
        sell_orders = equity_orders[equity_orders["quantity"] < 0]
        update_orders = equity_orders[equity_orders["quantity"] == 0]

        # Calculate and process net premium / allocation
        long_value = (buy_orders["spot"] * buy_orders["quantity"]).sum()
        short_value = (sell_orders["spot"] * sell_orders["quantity"]).sum()

        self.net_asset_value += long_value
        self.net_asset_value -= short_value

        self.cash -= long_value
        self.cash += short_value

        # Add positions to portfolio
        self.equities = pd.concat(
            [self.equities, buy_orders, sell_orders], ignore_index=True
        )

        # Update held equities to market data and process change in portfolio value
        if not update_orders.empty:
            merged = self.equities.merge(
                update_orders[["symbol", "spot"]],
                on="symbol",
                how="left",
                suffixes=("", "_new"),
            )
            mask = merged["spot_new"].notna()

            # Adjust portfolio market value according to changes in spot
            self.net_asset_value += (
                (merged.loc[mask, "spot_new"] - merged.loc[mask, "spot"])
                * merged.loc[mask, "quantity"]
            ).sum()

            merged.loc[mask, "spot"] = merged.loc[mask, "spot_new"]
            self.equities = merged.drop(columns="spot_new")

    def update_options(self, option_orders: pd.DataFrame | None):
        """
        Update current options positions to market
        """
        if option_orders is None:
            return

        buy_orders = option_orders[option_orders["quantity"] > 0]
        sell_orders = option_orders[option_orders["quantity"] < 0]
        update_orders = option_orders[option_orders["quantity"] == 0]

        # Calculate and process net premium / allocation
        buy_mid_prices = -(buy_orders["best_bid"] + buy_orders["best_offer"]) / 2
        sell_mid_prices = (sell_orders["best_bid"] + sell_orders["best_offer"]) / 2
        self.cash += 100 * (sell_mid_prices.sum() + buy_mid_prices.sum())

        # Add positions to portfolio
        self.options = pd.concat([self.options, sell_orders], ignore_index=True)

        # Update held options to market data
        if not update_orders.empty:
            self.options = self.options.set_index("optionid", drop=False)
            update_orders = update_orders.set_index("optionid", drop=False)

            cols_to_update = self.options.columns.difference(["exdate"])

            self.options.loc[update_orders.index, cols_to_update] = update_orders[
                cols_to_update
            ]

            update_long_orders = update_orders[update_orders["quantity"] > 0]
            update_short_orders = update_orders[update_orders["quantity"] < 0]

            update_long_mid_prices = (
                update_long_orders["best_bid"] + update_long_orders["best_offer"] / 2
            )
            update_short_mid_prices = (
                update_short_orders["best_bid"] + update_short_orders["best_offer"]
            ) / 2
            self.net_asset_value += 100 * (
                update_long_mid_prices.sum() + update_short_mid_prices.sum()
            )

    def handle_expired_options(self, current_date: pd.Timestamp):
        """
        Calculate PnL for expired options positions and remove from portfolio
        """
        if self.options.empty:
            return

        expired_mask = self.options["exdate"] <= current_date
        expired_options = self.options.loc[expired_mask]

        if not expired_options.empty:
            # Calculate intrinsic vlaues
            call_intrinsic = (
                expired_options["spot"] - expired_options["strike_price"]
            ).clip(lower=0)
            put_intrinsic = (
                expired_options["strike_price"] - expired_options["spot"]
            ).clip(lower=0)

            # Separate by long and short positions
            intrinsic_values = (
                np.where(
                    expired_options["cp_flag"] == "C", call_intrinsic, put_intrinsic
                )
                * 100
            )
            sign = np.where(expired_options["quantity"] > 0, 1, -1)

            # Adjust market value
            pnl_adjustments = intrinsic_values * sign
            self.cash += pnl_adjustments.sum()

            # Drop expired options
            self.options = self.options.loc[~expired_mask]
            self.options = self.options.reset_index(drop=True)

    def get_delta_exposure(self):
        return self.options["delta"].sum() + self.shares_owned

    def update_delta_pnl(
        self,
        spot: float,
        dS: float,
        commission_per_share: float,
        base_spread: float,
        spread_std: float,
    ):
        """
        Synthetic delta hedging PnL implementation
        """
        if self.options.empty:
            if self.shares_owned != 0:
                self.cash += self.shares_owned * dS
            return

        if self.shares_owned != 0:
            self.cash += self.shares_owned * dS

        net_delta = self.get_delta_exposure() - self.shares_owned
        trade_qty = round(net_delta, 0)

        trade_cashflow = net_delta * spot
        self.cash -= trade_cashflow

        spread = base_spread + np.random.normal(0, spread_std)
        spread = max(spread, 0)
        transaction_cost = trade_qty * (commission_per_share + spread)
        self.cash -= transaction_cost

        self.shares_owned += trade_qty
