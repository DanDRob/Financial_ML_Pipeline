"""
Data processing and cleaning module for financial ML pipeline.
"""

from typing import Dict, List, Optional, Tuple, Union
import logging
import ta
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler


class DataProcessor:
    """Data processing and cleaning class."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize data processor.

        Args:
            config: Processing configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

    def process_data(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Process all data frames in the dictionary.

        Args:
            data: Dictionary of DataFrames keyed by symbol

        Returns:
            Dictionary of processed DataFrames
        """
        try:
            processed_data = {}

            for symbol, df in data.items():
                processed_df = self._process_single_dataframe(df)
                processed_data[symbol] = processed_df

            return processed_data

        except Exception as e:
            self.logger.error(f"Data processing failed: {str(e)}")
            raise

    def _process_single_dataframe(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Process a single DataFrame.

        Args:
        df: Input DataFrame

        Returns:
            Processed DataFrame
        """
        try:
            # Make a copy to avoid modifying original data
            df = df.copy()

            # Apply processing steps
            df = self.remove_outliers(df)
            df = self.handle_missing_values(df)

            # Add features before normalization
            df = self.add_returns(df)
            df = self.add_technical_indicators(df)
            df = self.add_custom_features(df)

            # Create target variable
            df, target = self.create_target_variable(df)
            df['target'] = target

            # Normalize after feature creation but exclude target
            columns_to_normalize = [
                col for col in df.columns if col != 'target']
            df[columns_to_normalize] = self.normalize_data(
                df[columns_to_normalize])

            return df

        except Exception as e:
            self.logger.error(f"DataFrame processing failed: {str(e)}")
            raise

    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        n_std: float = 3
    ) -> pd.DataFrame:
        """
        Remove outliers from specified columns.

        Args:
            df: Input DataFrame
            columns: List of columns to process (default: all numeric columns)
            n_std: Number of standard deviations for outlier detection

        Returns:
            DataFrame with outliers removed
        """
        try:
            df = df.copy()
            columns = columns or df.select_dtypes(include=[np.number]).columns

            for col in columns:
                mean = df[col].mean()
                std = df[col].std()

                # Create mask for values within n standard deviations
                mask = (df[col] - mean).abs() <= (n_std * std)
                df.loc[~mask, col] = np.nan

            return df

        except Exception as e:
            self.logger.error(f"Outlier removal failed: {str(e)}")
            raise

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        method: str = 'ffill'
    ) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame.

        Args:
            df: Input DataFrame
            method: Method to handle missing values ('ffill', 'bfill', or 'interpolate')

        Returns:
            DataFrame with missing values handled
        """
        try:
            df = df.copy()

            if method == 'ffill':
                df = df.ffill().bfill()  # Forward fill then backward fill
            elif method == 'bfill':
                df = df.bfill().ffill()  # Backward fill then forward fill
            elif method == 'interpolate':
                df = df.interpolate(method='linear')
                df = df.ffill().bfill()  # Handle leading/trailing NAs

            return df

        except Exception as e:
            self.logger.error(f"Missing value handling failed: {str(e)}")
            raise

    def normalize_data(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'zscore'
    ) -> pd.DataFrame:
        """
        Normalize specified columns.

        Args:
            df: Input DataFrame
            columns: List of columns to normalize (default: all numeric columns)
            method: Normalization method ('zscore', 'minmax', or 'robust')

        Returns:
            DataFrame with normalized columns
        """
        try:
            df = df.copy()
            columns = columns or df.select_dtypes(include=[np.number]).columns

            for col in columns:
                if method == 'zscore':
                    df[col] = stats.zscore(df[col])
                elif method == 'minmax':
                    df[col] = (df[col] - df[col].min()) / \
                        (df[col].max() - df[col].min())
                elif method == 'robust':
                    median = df[col].median()
                    iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
                    df[col] = (df[col] - median) / iqr

            return df

        except Exception as e:
            self.logger.error(f"Data normalization failed: {str(e)}")
            raise

    def add_returns(
        self,
        data: pd.DataFrame,
        price_col: str = 'Close',
        periods: List[int] = [1, 5, 10, 21, 63]
    ) -> pd.DataFrame:
        """
        Add return calculations to DataFrame.

        Args:
            data: Input DataFrame
            price_col: Price column name
            periods: List of periods for return calculation

        Returns:
            DataFrame with added returns
        """
        try:
            df = data.copy()

            # Simple returns
            for period in periods:
                df[f'return_{period}d'] = df[price_col].pct_change(period)

            # Log returns
            for period in periods:
                df[f'log_return_{period}d'] = np.log(
                    df[price_col] / df[price_col].shift(period))

            # Rolling statistics
            df['rolling_std_21d'] = df[f'return_1d'].rolling(window=21).std()
            df['rolling_var_21d'] = df[f'return_1d'].rolling(window=21).var()

            return df

        except Exception as e:
            self.logger.error(f"Return calculation failed: {str(e)}")
            raise

    def add_technical_indicators(
        self,
        data: pd.DataFrame,
        include_all: bool = False
    ) -> pd.DataFrame:
        """
        Add technical indicators to DataFrame.

        Args:
            data: Input DataFrame
            include_all: Whether to include all available indicators

        Returns:
            DataFrame with technical indicators
        """
        try:

            df = data.copy()

            # Trend indicators
            df['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df['ema_20'] = ta.trend.ema_indicator(df['Close'], window=20)
            df['macd_diff'] = ta.trend.macd_diff(df['Close'])

            # Momentum indicators
            df['rsi'] = ta.momentum.rsi(df['Close'])
            df['stoch'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
            df['cci'] = ta.trend.cci(df['High'], df['Low'], df['Close'])

            # Volatility indicators
            df['bb_high'] = ta.volatility.bollinger_hband(df['Close'])
            df['bb_low'] = ta.volatility.bollinger_lband(df['Close'])
            df['atr'] = ta.volatility.average_true_range(
                df['High'], df['Low'], df['Close'])

            # Volume indicators
            df['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
            df['vwap'] = (df['Volume'] * df['Close']).cumsum() / \
                df['Volume'].cumsum()

            if include_all:
                # Additional indicators
                df['trix'] = ta.trend.trix(df['Close'])
                df['mass_index'] = ta.trend.mass_index(df['High'], df['Low'])
                df['dpo'] = ta.trend.dpo(df['Close'])
                df['kst'] = ta.trend.kst(df['Close'])
                df['ichimoku_a'] = ta.trend.ichimoku_a(df['High'], df['Low'])
                df['ichimoku_b'] = ta.trend.ichimoku_b(df['High'], df['Low'])

            return df

        except Exception as e:
            self.logger.error(
                f"Technical indicator calculation failed: {str(e)}")
            raise

    def add_custom_features(
        self,
        data: pd.DataFrame,
        config: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Add custom features to DataFrame.

        Args:
            data: Input DataFrame
            config: Feature configuration

        Returns:
            DataFrame with custom features
        """
        try:
            df = data.copy()
            config = config or {}

            # Price gaps
            df['gap_open'] = df['Open'] / df['Close'].shift(1) - 1
            df['gap_close'] = df['Close'] / df['Open'] - 1

            # Price ranges
            df['daily_range'] = (df['High'] - df['Low']) / df['Close']
            df['weekly_range'] = df['High'].rolling(
                5).max() / df['Low'].rolling(5).min() - 1

            # Volume analysis
            df['volume_ma_ratio'] = df['Volume'] / \
                df['Volume'].rolling(20).mean()
            df['volume_ma_std'] = df['Volume'].rolling(
                20).std() / df['Volume'].rolling(20).mean()

            # Price momentum
            df['momentum_1d'] = df['Close'] / df['Close'].shift(1)
            df['momentum_5d'] = df['Close'] / df['Close'].shift(5)

            # Volatility features
            df['realized_volatility'] = df['return_1d'].rolling(
                21).std() * np.sqrt(252)
            df['high_low_ratio'] = df['High'] / df['Low']

            # Custom oscillators
            df['stoch_rsi'] = ta.momentum.stochrsi(df['Close'])
            df['williams_r'] = ta.momentum.williams_r(
                df['High'], df['Low'], df['Close'])

            return df

        except Exception as e:
            self.logger.error(f"Custom feature calculation failed: {str(e)}")
            raise

    def create_lagged_features(
        self,
        data: pd.DataFrame,
        columns: List[str],
        lags: List[int]
    ) -> pd.DataFrame:
        """
        Create lagged versions of features.

        Args:
            data: Input DataFrame
            columns: Columns to lag
            lags: List of lag periods

        Returns:
            DataFrame with lagged features
        """
        try:
            df = data.copy()

            for col in columns:
                for lag in lags:
                    df[f"{col}_lag_{lag}"] = df[col].shift(lag)

            return df

        except Exception as e:
            self.logger.error(f"Lagged feature creation failed: {str(e)}")
            raise

    def create_target_variable(
        self,
        data: pd.DataFrame,
        method: str = 'binary',
        horizon: int = 1,
        threshold: float = 0.0
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create target variable for machine learning.

        Args:
            data: Input DataFrame
            method: Target creation method ('binary', 'return', 'zscore')
            horizon: Prediction horizon
            threshold: Classification threshold

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        try:
            df = data.copy()

            if method == 'binary':
                # Binary classification target
                future_returns = df['Close'].pct_change(
                    horizon).shift(-horizon)
                target = (future_returns > threshold).astype(int)

            elif method == 'return':
                # Regression target
                target = df['Close'].pct_change(horizon).shift(-horizon)

            elif method == 'zscore':
                # Standardized return target
                returns = df['Close'].pct_change(horizon).shift(-horizon)
                target = (returns - returns.rolling(252).mean()) / \
                    returns.rolling(252).std()

            else:
                raise ValueError(f"Unknown target creation method: {method}")

            # Remove future data from features
            df = df.iloc[:-horizon]
            target = target.iloc[:-horizon]

            return df, target

        except Exception as e:
            self.logger.error(f"Target creation failed: {str(e)}")
            raise

    def save_processor_state(self, filepath: str) -> None:
        """
        Save processor state for later use.

        Args:
            filepath: Path to save state
        """
        try:

            state = {
                'scaler': self.scaler,
                'config': self.config
            }

            joblib.dump(state, filepath)
            self.logger.info(f"Processor state saved to {filepath}")

        except Exception as e:
            self.logger.error(f"State saving failed: {str(e)}")
            raise

    def load_processor_state(self, filepath: str) -> None:
        """
        Load processor state from file.

        Args:
            filepath: Path to load state from
        """
        try:

            state = joblib.load(filepath)
            self.scaler = state['scaler']
            self.config = state['config']

            self.logger.info(f"Processor state loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"State loading failed: {str(e)}")
            raise
