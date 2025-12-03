# src/core/data.py

import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Iterator, Tuple


class DataLoader(ABC):
    """
    Abstract base data loader class
    """

    def __init__(
        self,
        data_path: Path,
        date_col: str = "date",
        exdate_col="exdate",
        start_date=None,
        end_date=None,
    ):
        self.data_path = data_path
        self.date_col = date_col
        self.exdate_col = exdate_col
        self.start_date = (
            pd.to_datetime(start_date).tz_localize(None) if start_date else None
        )
        self.end_date = pd.to_datetime(end_date).tz_localize(None) if end_date else None

    @abstractmethod
    def daily_stream(self) -> Iterator[Tuple[pd.Timestamp, pd.DataFrame | pd.Series]]:
        pass


class OhlcDataLoader(DataLoader):
    """
    Reader for OHLC (open, high, low, close) data
    """

    def __init__(
        self,
        data_path: Path,
        date_col: str = "date",
        exdate_col: str = "exdate",
        start_date=None,
        end_date=None,
    ):
        super().__init__(data_path, date_col, exdate_col, start_date, end_date)

    def daily_stream(self) -> Iterator[Tuple[pd.Timestamp, pd.Series]]:
        """
        Reads and yields data from OHLC CSV by day (row)).
        """
        ohlc_data = pd.read_csv(self.data_path, parse_dates=[self.date_col])

        if self.start_date:
            ohlc_data = ohlc_data[ohlc_data[self.date_col] >= self.start_date]
        if self.end_date:
            ohlc_data = ohlc_data[ohlc_data[self.date_col] <= self.end_date]

        for _, row in ohlc_data.iterrows():
            yield row[self.date_col].tz_localize(None), row

    def monthly_stream(self):
        """
        Reads OHLC CSV by row and yields data by month.
        """
        pass

        # Would be good to implement for longer-term strategies that rebalance portfolio monthly


class OptionsDataLoader(DataLoader):
    """
    Reader for options data
    """

    def __init__(
        self,
        data_path: Path,
        date_col: str = "date",
        exdate_col: str = "exdate",
        start_date=None,
        end_date=None,
        chunksize: int = 100000,
    ):
        super().__init__(data_path, date_col, exdate_col, start_date, end_date)
        self.chunksize = chunksize

    def daily_stream(self):
        """
        Reads options CSV in chunks and yields daily data.
        """
        buffer = pd.DataFrame()
        reader = pd.read_csv(
            self.data_path, parse_dates=[self.date_col], chunksize=self.chunksize
        )

        for chunk in reader:
            # Convert date columns to datetime format
            chunk[self.date_col] = pd.to_datetime(chunk[self.date_col]).dt.tz_localize(
                None
            )
            chunk[self.exdate_col] = pd.to_datetime(
                chunk[self.exdate_col]
            ).dt.tz_localize(None)

            buffer = pd.concat([buffer, chunk], ignore_index=True)
            unique_dates = buffer[self.date_col].dt.date.unique()

            # Load only complete days into buffer
            for date in unique_dates[:-1]:
                if self.start_date:
                    buffer = buffer[buffer[self.date_col] >= self.start_date]
                if self.end_date and pd.to_datetime(date) > self.end_date:
                    return
                day_market_data = buffer[buffer[self.date_col].dt.date == date]

                yield pd.to_datetime(date), day_market_data

            buffer = buffer[buffer[self.date_col].dt.date == unique_dates[-1]]

        # Yield remaining last day
        if not buffer.empty:
            last_date = buffer[self.date_col].dt.date.iloc[0]
            if (
                not self.start_date or pd.to_datetime(last_date) >= self.start_date
            ) and (not self.end_date or pd.to_datetime(last_date) <= self.end_date):
                yield pd.to_datetime(last_date), buffer


class MultiDataLoader:
    """
    Reader for multiple data loaders - enables concurrent data streams
    """

    def __init__(self, loaders: dict[str, DataLoader], start_date=None, end_date=None):
        self.loaders = loaders
        self.start_date = (
            pd.to_datetime(start_date).tz_localize(None) if start_date else None
        )
        self.end_date = pd.to_datetime(end_date).tz_localize(None) if end_date else None

    def daily_multi_stream(
        self,
    ) -> Iterator:
        """
        Reads data from multiple streams and yields a tuple per day (date, {name: data})
        """
        streams = {name: loader.daily_stream() for name, loader in self.loaders.items()}

        next_items = {}
        for name, stream in streams.items():
            try:
                next_items[name] = next(stream)
            except StopIteration:
                next_items[name] = None

        while any(item is not None for item in next_items.values()):
            active_items = {k: v for k, v in next_items.items() if v is not None}
            current_date = min(date for date, _ in active_items.values())

            # Check that dates are within timeframe
            if self.start_date and current_date < self.start_date:
                for name, val in active_items.items():
                    date, data = val
                    if date < self.start_date:
                        try:
                            next_items[name] = next(streams[name])
                        except StopIteration:
                            next_items[name] = None
                continue
            self.start_date = None
            if self.end_date and current_date > self.end_date:
                return

            day_market_data = {}
            for name, val in active_items.items():
                date, data = val
                if date == current_date:
                    day_market_data[name] = data
                    try:
                        next_items[name] = next(streams[name])
                    except StopIteration:
                        next_items[name] = None

            yield current_date, day_market_data
