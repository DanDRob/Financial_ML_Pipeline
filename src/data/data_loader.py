"""
Data loading and management module for financial ML pipeline.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import pickle

import numpy as np
import pandas as pd
import yfinance as yf


class DataLoader:
    """Data loading and management class."""

    def __init__(
        self,
        config: Dict
    ):
        """
        Initialize DataLoader.

        Args:
            config: DataLoader configuration
        """
        self.config = config
        self.data_dir = Path(config.get('data_dir', 'data'))
        self.cache_dir = Path(
            config.get('cache_dir', 'data/cache'))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def fetch_data(
        self,
        target_symbol: str,
        start_date: str,
        end_date: str,
        additional_symbols: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for target and additional symbols.

        Args:
            target_symbol: Primary symbol to fetch
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            additional_symbols: Optional list of additional symbols

        Returns:
            Dictionary of DataFrames keyed by symbol
        """
        try:
            data = {}

            # Fetch target symbol
            data[target_symbol] = self._fetch_yahoo_data(
                target_symbol,
                start_date,
                end_date
            )

            # Fetch additional symbols if provided
            if additional_symbols:
                for symbol in additional_symbols:
                    if symbol != target_symbol:
                        data[symbol] = self._fetch_yahoo_data(
                            symbol,
                            start_date,
                            end_date
                        )

            return data

        except Exception as e:
            self.logger.error(f"Data fetching failed: {str(e)}")
            raise

    def _fetch_yahoo_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance.

        Args:
            symbol: Stock symbol
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Create Ticker object
            ticker = yf.Ticker(symbol)

            # Download data
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval="1d"
            )

            # Ensure all required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")

            return df

        except Exception as e:
            self.logger.error(f"Yahoo data fetch failed for {
                              symbol}: {str(e)}")
            raise

    def _validate_data(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> bool:
        """
        Validate fetched data.

        Args:
            df: DataFrame to validate
            symbol: Symbol for logging

        Returns:
            bool indicating if data is valid
        """
        try:
            # Check for empty DataFrame
            if df.empty:
                self.logger.error(f"Empty DataFrame for {symbol}")
                return False

            # Check for required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_columns):
                self.logger.error(f"Missing required columns for {symbol}")
                return False

            # Check for sufficient data points
            if len(df) < 30:  # Minimum required data points
                self.logger.error(f"Insufficient data points for {symbol}")
                return False

            # Check for missing values
            missing_pct = df[required_columns].isnull().mean()
            if any(missing_pct > 0.1):  # More than 10% missing values
                self.logger.error(f"Too many missing values for {symbol}")
                return False

            # Check for invalid values
            if any(df[['Open', 'High', 'Low', 'Close']] <= 0).any():
                self.logger.error(f"Invalid price values for {symbol}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Data validation failed for {symbol}: {str(e)}")
            return False

    def save_data(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        filename: str,
        format: str = 'parquet'
    ) -> None:
        """
        Save data to disk.

        Args:
            data: DataFrame or dict of DataFrames to save
            filename: Base filename
            format: File format ('parquet', 'csv', 'pickle')
        """
        try:
            if isinstance(data, pd.DataFrame):
                data = {'single': data}

            for name, df in data.items():
                if format == 'parquet':
                    path = self.data_dir / f"{filename}_{name}.parquet"
                    df.to_parquet(path)
                elif format == 'csv':
                    path = self.data_dir / f"{filename}_{name}.csv"
                    df.to_csv(path)
                elif format == 'pickle':
                    path = self.data_dir / f"{filename}_{name}.pkl"
                    with open(path, 'wb') as f:
                        pickle.dump(df, f)
                else:
                    raise ValueError(f"Unsupported format: {format}")

            self.logger.info(f"Data saved successfully in {format} format")

        except Exception as e:
            self.logger.error(f"Data saving failed: {str(e)}")
            raise

    def load_data(
        self,
        filename: str,
        format: str = 'parquet'
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Load data from disk.

        Args:
            filename: Base filename
            format: File format ('parquet', 'csv', 'pickle')

        Returns:
            DataFrame or dict of DataFrames
        """
        try:
            if format == 'parquet':
                pattern = f"{filename}_*.parquet"
            elif format == 'csv':
                pattern = f"{filename}_*.csv"
            elif format == 'pickle':
                pattern = f"{filename}_*.pkl"
            else:
                raise ValueError(f"Unsupported format: {format}")

            files = list(self.data_dir.glob(pattern))

            if not files:
                raise FileNotFoundError(
                    f"No files found matching pattern: {pattern}")

            data_dict = {}
            for file in files:
                name = file.stem.replace(f"{filename}_", "")

                if format == 'parquet':
                    df = pd.read_parquet(file)
                elif format == 'csv':
                    df = pd.read_csv(file)
                elif format == 'pickle':
                    with open(file, 'rb') as f:
                        df = pickle.load(f)

                data_dict[name] = df

            return data_dict if len(data_dict) > 1 else data_dict['single']

        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}")
            raise

    def merge_data(
        self,
        data_dict: Dict[str, pd.DataFrame],
        merge_on: str = 'Date',
        suffixes: Optional[Tuple[str, str]] = None
    ) -> pd.DataFrame:
        """
        Merge multiple DataFrames.

        Args:
            data_dict: Dict of DataFrames to merge
            merge_on: Column to merge on
            suffixes: Suffixes for overlapping columns

        Returns:
            Merged DataFrame
        """
        try:
            if len(data_dict) < 2:
                return next(iter(data_dict.values()))

            result = None
            for name, df in data_dict.items():
                if result is None:
                    result = df.copy()
                    continue

                current_suffixes = suffixes or (f'_{name}', '_right')
                result = pd.merge(
                    result,
                    df,
                    on=merge_on,
                    how='outer',
                    suffixes=current_suffixes
                )

            return result

        except Exception as e:
            self.logger.error(f"Data merging failed: {str(e)}")
            raise

    def resample_data(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        interval: str,
        agg_dict: Optional[Dict[str, str]] = None
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Resample data to different frequency.

        Args:
            data: Data to resample
            interval: Target interval ('1H', '1D', etc.)
            agg_dict: Aggregation dictionary

        Returns:
            Resampled data
        """
        try:
            if isinstance(data, pd.DataFrame):
                data = {'single': data}

            if agg_dict is None:
                agg_dict = {
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }

            resampled_data = {}
            for name, df in data.items():
                resampled = df.resample(interval).agg(agg_dict)
                resampled_data[name] = resampled

            return resampled_data if len(resampled_data) > 1 else resampled_data['single']

        except Exception as e:
            self.logger.error(f"Data resampling failed: {str(e)}")
            raise
