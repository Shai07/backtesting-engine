# src/strategies/earnings_iv_edge.py

import math

import pandas as pd

from .base import Strategy


class EarningsIVEdgeStrategy(Strategy):
    def __init__(
        self,
        earnings_dates: list[pd.Timestamp | str],
        lookahead_days: int = 30,
        edge_threshold: float = 0.02,
        min_past_earnings: int = 1,
        max_positions: int = 2,
        max_straddles_per_event: int = 1,
        sizing_method: str = "kelly",
        kelly_fraction: float = 0.50,
        min_kelly_observations: int = 4,
        atm_tolerance: float = 0.01,
        annualized_days: int = 252,
    ):
        self.earnings_dates = sorted(
            pd.to_datetime(date).tz_localize(None).normalize()
            for date in earnings_dates
        )
        self.lookahead_days = lookahead_days
        self.edge_threshold = edge_threshold
        self.min_past_earnings = min_past_earnings
        self.max_positions = max_positions
        self.max_straddles_per_event = max_straddles_per_event
        self.sizing_method = sizing_method
        self.kelly_fraction = kelly_fraction
        self.min_kelly_observations = min_kelly_observations
        self.atm_tolerance = atm_tolerance
        self.annualized_days = annualized_days

        self.ohlc_history: dict[pd.Timestamp, pd.Series] = {}
        self.options_history: dict[pd.Timestamp, pd.DataFrame] = {}
        self.earnings_edge_cache: dict[
            tuple[pd.Timestamp, int], dict[str, float]
        ] = {}
        self.traded_earnings_dates: set[pd.Timestamp] = set()
        self.sizing_records: list[dict[str, object]] = []

    def process_data(
        self,
        market_data: dict[str, pd.DataFrame | pd.Series],
        options_positions: pd.DataFrame,
    ) -> pd.DataFrame | None:
        ohlc_data = market_data.get("ohlc")
        if ohlc_data is None:
            return None

        current_date = pd.to_datetime(ohlc_data["date"]).tz_localize(None).normalize()
        close = float(ohlc_data["close"])
        self.ohlc_history[current_date] = ohlc_data.copy()

        options_data = market_data.get("options")
        if options_data is not None:
            options_data = options_data.copy()
            options_data["date"] = pd.to_datetime(options_data["date"]).dt.normalize()
            options_data["exdate"] = pd.to_datetime(options_data["exdate"]).dt.normalize()
            self.options_history[current_date] = options_data
            option_spot = self._option_scaled_spot(options_data, close)
        else:
            option_spot = close

        orders = self._create_update_orders(options_data, options_positions, option_spot)

        upcoming_earnings = self._next_earnings_date(current_date)
        if upcoming_earnings is None or options_data is None:
            return pd.DataFrame(orders) if orders else None

        if upcoming_earnings in self.traded_earnings_dates:
            return pd.DataFrame(orders) if orders else None

        days_to_earnings = int((upcoming_earnings - current_date).days)
        self._update_past_earnings_edges(current_date, days_to_earnings)
        average_edge = self._average_edge(current_date, days_to_earnings)
        if average_edge is None or abs(average_edge) < self.edge_threshold:
            return pd.DataFrame(orders) if orders else None

        active_positions = len(options_positions)
        straddle_slots = min(
            self.max_straddles_per_event,
            max(0, (self.max_positions - active_positions) // 2),
        )
        if straddle_slots < 1:
            return pd.DataFrame(orders) if orders else None

        side = -1 if average_edge > 0 else 1
        past_edges = self._past_edges(current_date, days_to_earnings)
        straddle_count, kelly_size = self._straddle_count(
            past_edges=past_edges,
            side=side,
            straddle_slots=straddle_slots,
        )
        if straddle_count < 1:
            return pd.DataFrame(orders) if orders else None

        straddle_rows = self._find_live_straddle(
            options_data=options_data,
            spot=option_spot,
            earnings_date=upcoming_earnings,
            options_positions=options_positions,
        )
        if straddle_rows is None:
            return pd.DataFrame(orders) if orders else None

        for _ in range(straddle_count):
            for _, option_row in straddle_rows.iterrows():
                orders.append(self.create_option_order(side, option_spot, option_row))

        self.traded_earnings_dates.add(upcoming_earnings)
        self.sizing_records.append(
            {
                "date": current_date,
                "upcoming_earnings": upcoming_earnings,
                "days_to_earnings": days_to_earnings,
                "average_edge": average_edge,
                "side": "sell" if side < 0 else "buy",
                "past_edge_count": len(past_edges),
                "kelly_size": kelly_size,
                "straddle_slots": straddle_slots,
                "straddles_opened": straddle_count,
            }
        )

        return pd.DataFrame(orders) if orders else None

    def _create_update_orders(
        self,
        options_data: pd.DataFrame | None,
        options_positions: pd.DataFrame,
        spot: float,
    ) -> list[pd.Series]:
        if options_data is None or options_positions.empty:
            return []

        update_rows = options_data[
            options_data["optionid"].isin(options_positions["optionid"])
        ]
        return [
            self.create_option_order(0, spot, option_row)
            for _, option_row in update_rows.iterrows()
        ]

    def _next_earnings_date(self, current_date: pd.Timestamp) -> pd.Timestamp | None:
        window_end = current_date + pd.Timedelta(days=self.lookahead_days)
        for earnings_date in self.earnings_dates:
            if current_date <= earnings_date <= window_end:
                return earnings_date
        return None

    def _update_past_earnings_edges(
        self, current_date: pd.Timestamp, days_to_earnings: int
    ) -> None:
        for earnings_date in self.earnings_dates:
            if earnings_date >= current_date:
                break
            cache_key = (earnings_date, days_to_earnings)
            if cache_key in self.earnings_edge_cache:
                continue

            result = self._calculate_earnings_edge(earnings_date, days_to_earnings)
            if result is not None:
                self.earnings_edge_cache[cache_key] = result

    def _average_edge(
        self, current_date: pd.Timestamp, days_to_earnings: int
    ) -> float | None:
        past_edges = self._past_edges(current_date, days_to_earnings)
        if len(past_edges) < self.min_past_earnings:
            return None
        return float(sum(past_edges) / len(past_edges))

    def _past_edges(
        self, current_date: pd.Timestamp, days_to_earnings: int
    ) -> list[float]:
        return [
            result["edge"]
            for (earnings_date, edge_days), result in self.earnings_edge_cache.items()
            if earnings_date < current_date and edge_days == days_to_earnings
        ]

    def _straddle_count(
        self,
        past_edges: list[float],
        side: int,
        straddle_slots: int,
    ) -> tuple[int, float | None]:
        if straddle_slots < 1:
            return 0, None

        if self.sizing_method == "max_available":
            return straddle_slots, None

        if self.sizing_method != "kelly":
            return 1, None

        if len(past_edges) < self.min_kelly_observations:
            return straddle_slots, None

        directional_edges = [
            edge if side < 0 else -edge
            for edge in past_edges
        ]
        mean_edge = float(sum(directional_edges) / len(directional_edges))
        variance = float(
            sum((edge - mean_edge) ** 2 for edge in directional_edges)
            / (len(directional_edges) - 1)
        )
        if variance <= 0:
            return straddle_slots, None

        raw_kelly = mean_edge / variance
        capped_kelly = max(0.0, min(raw_kelly * self.kelly_fraction, 1.0))
        if capped_kelly <= 0:
            return 1, capped_kelly

        return max(1, math.floor(straddle_slots * capped_kelly)), capped_kelly

    def _calculate_earnings_edge(
        self, earnings_date: pd.Timestamp, days_to_earnings: int
    ) -> dict[str, float] | None:
        realized_move = self._realized_earnings_move(earnings_date)
        if realized_move is None:
            return None

        signal_date = earnings_date - pd.Timedelta(days=days_to_earnings)
        option_date = self._latest_option_date_on_or_before(signal_date)
        if option_date is None:
            return None

        options_data = self.options_history[option_date]
        close = float(self.ohlc_history[option_date]["close"])
        spot = self._option_scaled_spot(options_data, close)
        expected_move = self._expected_earnings_move(options_data, spot, earnings_date)
        if expected_move is None:
            return None

        return {
            "expected_move": expected_move,
            "realized_move": realized_move,
            "edge": expected_move - realized_move,
        }

    def _realized_earnings_move(self, earnings_date: pd.Timestamp) -> float | None:
        prior_date = self._latest_ohlc_date_before(earnings_date)
        event_date = self._first_ohlc_date_on_or_after(earnings_date)
        if prior_date is None or event_date is None:
            return None

        prior_close = float(self.ohlc_history[prior_date]["close"])
        event_close = float(self.ohlc_history[event_date]["close"])
        if prior_close <= 0:
            return None
        return abs(event_close / prior_close - 1)

    def _expected_earnings_move(
        self,
        options_data: pd.DataFrame,
        spot: float,
        earnings_date: pd.Timestamp,
    ) -> float | None:
        same_day_option = self._best_atm_option(
            options_data[options_data["exdate"] == earnings_date], spot
        )
        if same_day_option is not None:
            return float(same_day_option["impl_volatility"]) / math.sqrt(
                self.annualized_days
            )

        before_options = options_data[options_data["exdate"] < earnings_date]
        after_options = options_data[options_data["exdate"] > earnings_date]
        before_option = self._best_atm_option(before_options, spot)
        after_option = self._best_atm_option(after_options, spot)
        if before_option is None or after_option is None:
            return None

        after_dte = int((after_option["exdate"] - options_data["date"].iloc[0]).days)
        if after_dte <= 0:
            return None

        total_iv = float(after_option["impl_volatility"])
        baseline_iv = float(before_option["impl_volatility"])
        baseline_days = max(after_dte - 1, 0)

        total_variance = total_iv**2 * after_dte / self.annualized_days
        baseline_variance = baseline_iv**2 * baseline_days / self.annualized_days
        earnings_variance = max(total_variance - baseline_variance, 0)

        return math.sqrt(earnings_variance)

    def _find_live_straddle(
        self,
        options_data: pd.DataFrame,
        spot: float,
        earnings_date: pd.Timestamp,
        options_positions: pd.DataFrame,
    ) -> pd.DataFrame | None:
        option_columns = options_data.columns
        candidates = options_data[options_data["exdate"] > earnings_date].copy()
        candidates = candidates.dropna(
            subset=["best_bid", "best_offer", "impl_volatility"]
        )
        if candidates.empty:
            return None

        existing_optionids = (
            set(options_positions["optionid"]) if not options_positions.empty else set()
        )
        candidates = candidates[~candidates["optionid"].isin(existing_optionids)]
        if candidates.empty:
            return None

        candidates["strike"] = candidates["strike_price"] / 1000
        candidates["strike_distance"] = (candidates["strike"] - spot).abs()
        candidates["dte_after_earnings"] = (
            candidates["exdate"] - earnings_date
        ).dt.days

        nearest_expiry = candidates["dte_after_earnings"].min()
        candidates = candidates[candidates["dte_after_earnings"] == nearest_expiry]
        nearest_strike = candidates["strike_distance"].min()
        candidates = candidates[candidates["strike_distance"] == nearest_strike]

        call = candidates[candidates["cp_flag"] == "C"].head(1)
        put = candidates[candidates["cp_flag"] == "P"].head(1)
        if call.empty or put.empty:
            return None

        return pd.concat([call, put])[option_columns]

    def _best_atm_option(
        self, options_data: pd.DataFrame, spot: float
    ) -> pd.Series | None:
        if options_data.empty:
            return None

        candidates = options_data.dropna(
            subset=["strike_price", "exdate", "impl_volatility"]
        ).copy()
        if candidates.empty:
            return None

        candidates["strike"] = candidates["strike_price"] / 1000
        candidates["strike_distance"] = (candidates["strike"] - spot).abs()
        candidates = candidates[
            candidates["strike"].between(
                spot * (1 - self.atm_tolerance),
                spot * (1 + self.atm_tolerance),
            )
        ]
        if candidates.empty:
            return None

        candidates = candidates.sort_values(["strike_distance", "exdate"])
        return candidates.iloc[0]

    def _option_scaled_spot(self, options_data: pd.DataFrame, spot: float) -> float:
        if options_data.empty or spot <= 0:
            return spot

        strikes = options_data["strike_price"] / 1000
        possible_split_factors = [1, 2, 5, 10, 20, 50, 100]
        best_factor = min(
            possible_split_factors,
            key=lambda factor: (strikes - spot * factor).abs().min(),
        )
        return spot * best_factor

    def _latest_option_date_on_or_before(
        self, target_date: pd.Timestamp
    ) -> pd.Timestamp | None:
        dates = [date for date in self.options_history if date <= target_date]
        return max(dates) if dates else None

    def _latest_ohlc_date_before(self, target_date: pd.Timestamp) -> pd.Timestamp | None:
        dates = [date for date in self.ohlc_history if date < target_date]
        return max(dates) if dates else None

    def _first_ohlc_date_on_or_after(
        self, target_date: pd.Timestamp
    ) -> pd.Timestamp | None:
        dates = [date for date in self.ohlc_history if date >= target_date]
        return min(dates) if dates else None
