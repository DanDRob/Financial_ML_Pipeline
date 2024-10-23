"""
Custom financial indicators and feature creation module.
"""

from typing import Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd
from scipy import stats
import ta


class CustomFeatureCreator:
    """Custom financial feature creation class."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize CustomFeatureCreator.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

    def create_all_features(
        self,
        data: pd.DataFrame,
        include_groups: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create all custom features.

        Args:
            data: Input DataFrame
            include_groups: List of feature groups to include

        Returns:
            DataFrame with custom features
        """
        try:
            df = data.copy()
            groups = include_groups or [
                'price_derived',
                'volatility_derived',
                'volume_derived',
                'pattern_recognition',
                'market_microstructure'
            ]

            for group in groups:
                if group == 'price_derived':
                    df = self.add_price_derived_features(df)
                elif group == 'volatility_derived':
                    df = self.add_volatility_derived_features(df)
                elif group == 'volume_derived':
                    df = self.add_volume_derived_features(df)
                elif group == 'pattern_recognition':
                    df = self.add_pattern_recognition_features(df)
                elif group == 'market_microstructure':
                    df = self.add_microstructure_features(df)

            return df

        except Exception as e:
            self.logger.error(f"Custom feature creation failed: {str(e)}")
            raise

    def add_price_derived_features(
        self,
        data: pd.DataFrame,
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Add price-derived custom features.

        Args:
            data: Input DataFrame
            windows: List of rolling windows

        Returns:
            DataFrame with price features
        """
        try:
            df = data.copy()
            windows = windows or [5, 10, 21, 63]

            # Price gaps
            df['overnight_gap'] = (
                df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
            df['intraday_gap'] = (df['Close'] - df['Open']) / df['Open']

            # Price efficiency
            for window in windows:
                df[f'price_efficiency_{window}d'] = (
                    (df['Close'] - df['Close'].shift(window)).abs() /
                    (df['High'].rolling(window).max() -
                     df['Low'].rolling(window).min())
                )

            # Price acceleration
            df['price_acceleration'] = df['Close'].pct_change().diff()

            # Log returns distribution
            log_returns = np.log(df['Close'] / df['Close'].shift(1))

            for window in windows:
                roll_returns = log_returns.rolling(window)
                df[f'returns_skew_{window}d'] = roll_returns.skew()
                df[f'returns_kurt_{window}d'] = roll_returns.kurt()

            # Price levels
            for window in windows:
                df[f'price_level_{window}d'] = (
                    (df['Close'] - df['Close'].rolling(window).min()) /
                    (df['Close'].rolling(window).max() -
                     df['Close'].rolling(window).min())
                )

            # Relative strength
            for window in windows:
                df[f'relative_strength_{window}d'] = (
                    df['Close'] / df['Close'].rolling(window).mean()
                )

            return df

        except Exception as e:
            self.logger.error(f"Price feature creation failed: {str(e)}")
            raise

    def add_volatility_derived_features(
        self,
        data: pd.DataFrame,
        config: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Add volatility-derived custom features.

        Args:
            data: Input DataFrame
            config: Configuration dictionary

        Returns:
            DataFrame with volatility features
        """
        try:
            df = data.copy()
            windows = config.get('windows', [5, 10, 21, 63])

            # Parkinson volatility
            for window in windows:
                df[f'parkinson_vol_{window}d'] = np.sqrt(
                    (1 / (4 * np.log(2))) *
                    (np.log(df['High'] / df['Low']) **
                     2).rolling(window).mean() * 252
                )

            # Garman-Klass volatility
            for window in windows:
                df[f'garman_klass_vol_{window}d'] = np.sqrt(
                    (0.5 * np.log(df['High'] / df['Low'])**2) -
                    (2 * np.log(2) - 1) * (np.log(df['Close'] / df['Open'])**2)
                ).rolling(window).mean() * np.sqrt(252)

            # Rogers-Satchell volatility
            for window in windows:
                df[f'rogers_satchell_vol_{window}d'] = np.sqrt(
                    (np.log(df['High'] / df['Close']) * np.log(df['High'] / df['Open'])) +
                    (np.log(df['Low'] / df['Close'])
                     * np.log(df['Low'] / df['Open']))
                ).rolling(window).mean() * np.sqrt(252)

            # Volatility regime features
            returns = df['Close'].pct_change()
            for window in windows:
                # EWMA volatility
                df[f'ewma_vol_{window}d'] = returns.ewm(
                    span=window).std() * np.sqrt(252)

                # Volatility of volatility
                df[f'vol_of_vol_{window}d'] = (
                    df[f'ewma_vol_{window}d'].rolling(window).std()
                )

                # Volatility skew
                df[f'vol_skew_{window}d'] = (
                    (df['High'] - df['Close']) /
                    (df['High'] - df['Low'])
                ).rolling(window).mean()

            # Cross-sectional volatility features
            high_low_range = (df['High'] - df['Low']) / df['Close']
            for window in windows:
                df[f'range_expansion_{window}d'] = (
                    high_low_range / high_low_range.rolling(window).mean()
                )

            return df

        except Exception as e:
            self.logger.error(f"Volatility feature creation failed: {str(e)}")
            raise

    def add_volume_derived_features(
        self,
        data: pd.DataFrame,
        config: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Add volume-derived custom features.

        Args:
            data: Input DataFrame
            config: Configuration dictionary

        Returns:
            DataFrame with volume features
        """
        try:
            df = data.copy()
            windows = config.get('windows', [5, 10, 21, 63])

            # Volume pressure
            df['volume_pressure'] = (
                df['Volume'] * (df['Close'] - df['Open']) / df['Open']
            )

            # Volume force
            for window in windows:
                df[f'volume_force_{window}d'] = (
                    df['volume_pressure'].rolling(window).mean() *
                    df['Volume'] / df['Volume'].rolling(window).mean()
                )

            # Volume weighted price momentum
            for window in windows:
                df[f'vwap_momentum_{window}d'] = (
                    df['Close'] -
                    (df['Volume'] * df['Close']).rolling(window).sum() /
                    df['Volume'].rolling(window).sum()
                )

            # Volume distribution features
            for window in windows:
                roll_vol = df['Volume'].rolling(window)
                df[f'volume_skew_{window}d'] = roll_vol.skew()
                df[f'volume_kurt_{window}d'] = roll_vol.kurt()

                # Volume concentration
                df[f'volume_concentration_{window}d'] = (
                    df['Volume'] /
                    df['Volume'].rolling(window).sum()
                )

            # Volume price correlation
            for window in windows:
                df[f'volume_price_corr_{window}d'] = (
                    df['Volume'].rolling(window)
                    .corr(df['Close'].pct_change())
                )

            # Volume ratios
            df['up_down_volume_ratio'] = np.where(
                df['Close'] > df['Close'].shift(1),
                df['Volume'],
                -df['Volume']
            )

            for window in windows:
                df[f'up_down_volume_ratio_{window}d'] = (
                    df['up_down_volume_ratio']
                    .rolling(window)
                    .sum()
                )

            return df

        except Exception as e:
            self.logger.error(f"Volume feature creation failed: {str(e)}")
            raise

    def add_pattern_recognition_features(
        self,
        data: pd.DataFrame,
        config: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Add pattern recognition features.

        Args:
            data: Input DataFrame
            config: Configuration dictionary

        Returns:
            DataFrame with pattern features
        """
        try:
            df = data.copy()

            # Candlestick patterns
            pattern_functions = {
                'doji': ta.CDLDOJI,
                'hammer': ta.CDLHAMMER,
                'engulfing': ta.CDLENGULFING,
                'morning_star': ta.CDLMORNINGSTAR,
                'evening_star': ta.CDLEVENINGSTAR,
                'harami': ta.CDLHARAMI,
                'shooting_star': ta.CDLSHOOTINGSTAR
            }

            for name, func in pattern_functions.items():
                df[f'pattern_{name}'] = func(
                    df['Open'],
                    df['High'],
                    df['Low'],
                    df['Close']
                )

            # Price patterns
            windows = config.get('windows', [5, 10, 21])

            for window in windows:
                # Higher highs and lower lows
                df[f'higher_highs_{window}d'] = (
                    df['High'].rolling(window).max() >
                    df['High'].shift(window).rolling(window).max()
                ).astype(int)

                df[f'lower_lows_{window}d'] = (
                    df['Low'].rolling(window).min() <
                    df['Low'].shift(window).rolling(window).min()
                ).astype(int)

                # Inside/Outside bars
                df[f'inside_bar_{window}d'] = (
                    (df['High'] <= df['High'].shift(1)) &
                    (df['Low'] >= df['Low'].shift(1))
                ).astype(int)

                df[f'outside_bar_{window}d'] = (
                    (df['High'] > df['High'].shift(1)) &
                    (df['Low'] < df['Low'].shift(1))
                ).astype(int)

            return df

        except Exception as e:
            self.logger.error(f"Pattern feature creation failed: {str(e)}")
            raise

    def add_microstructure_features(
        self,
        data: pd.DataFrame,
        config: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Add market microstructure features.

        Args:
            data: Input DataFrame
            config: Configuration dictionary

        Returns:
            DataFrame with microstructure features
        """
        try:
            df = data.copy()
            windows = config.get('windows', [5, 10, 21])

            # Bid-ask bounce proxies
            df['price_reversal'] = (
                (df['Close'] - df['Open']) *
                (df['Open'] - df['Close'].shift(1))
            ).apply(lambda x: 1 if x < 0 else 0)

            # Trading intensity
            df['trading_intensity'] = df['Volume'] / (
                df['High'] - df['Low']
            ).replace(0, np.nan)

            # Roll spread estimator
            for window in windows:
                price_changes = df['Close'].diff()
                roll_cov = (
                    price_changes.rolling(window)
                    .apply(lambda x: np.cov(x[:-1], x[1:])[0, 1])
                )
                df[f'roll_spread_{window}d'] = 2 * np.sqrt(-roll_cov)

            # Kyle's lambda (price impact)
            for window in windows:
                df[f'kyle_lambda_{window}d'] = (
                    df['Close'].diff().abs().rolling(window).sum() /
                    (df['Volume'] * df['Close']).rolling(window).sum()
                )

            # Amihud illiquidity
            for window in windows:
                df[f'amihud_illiq_{window}d'] = (
                    df['Close'].pct_change().abs() /
                    (df['Volume'] * df['Close'])
                ).rolling(window).mean()

            # Probability of informed trading proxy
            for window in windows:
                daily_range = (df['High'] - df['Low']) / df['Close']
                df[f'informed_trading_{window}d'] = (
                    daily_range * df['Volume'] /
                    daily_range.rolling(window).mean()
                )

            return df

        except Exception as e:
            self.logger.error(
                f"Microstructure feature creation failed: {str(e)}")
            raise

    def create_interaction_features(
        self,
        data: pd.DataFrame,
        feature_pairs: List[Tuple[str, str]]
    ) -> pd.DataFrame:
        """
        Create interaction features between pairs of existing features.

        Args:
            data: Input DataFrame
            feature_pairs: List of feature name pairs

        Returns:
            DataFrame with interaction features
        """
        try:
            df = data.copy()

            for feat1, feat2 in feature_pairs:
                # Multiplication interaction
                df[f'interaction_mul_{feat1}_{feat2}'] = df[feat1] * df[feat2]

                # Ratio interaction (handling zeros)
                df[f'interaction_ratio_{feat1}_{feat2}'] = (
                    df[feat1] / df[feat2].replace(0, np.nan)
                )

                # Difference interaction
                df[f'interaction_diff_{feat1}_{feat2}'] = df[feat1] - df[feat2]

            return df

        except Exception as e:
            self.logger.error(f"Interaction feature creation failed: {str(e)}")
            raise

    def create_time_features(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create time-based features from datetime index.

        Args:
            data: Input DataFrame with datetime index

        Returns:
            DataFrame with time features
        """
        try:
            df = data.copy()

            # Basic time features
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['week_of_year'] = df.index.isocalendar().week
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            df['year'] = df.index.year

            # Market session indicators
            df['is_month_start'] = df.index.is_month_start.astype(int)
            df['is_month_end'] = df.index.is_month_end.astype(int)
            df['is_quarter_start'] = df.index.is_quarter_start.astype(int)
            df['is_quarter_end'] = df.index.is_quarter_end.astype(int)

            # Day of week effects
            df['is_monday'] = (df['day_of_week'] == 0).astype(int)
            df['is_friday'] = (df['day_of_week'] == 4).astype(int)

            # Seasonal decomposition
            if len(df) >= 2:
                from statsmodels.tsa.seasonal import seasonal_decompose

                # Decompose log prices
                log_prices = np.log(df['Close'])
                decomposition = seasonal_decompose(
                    log_prices,
                    period=252,  # Trading days in year
                    extrapolate_trend='freq'
                )

                df['seasonal_factor'] = decomposition.seasonal
                df['trend_factor'] = decomposition.trend
                df['residual_factor'] = decomposition.resid

            # Rolling seasonal patterns
            windows = [5, 10, 21, 63]
            for window in windows:
                df[f'seasonal_pattern_{window}d'] = (
                    df['Close'] / df['Close'].shift(window) - 1
                )

            return df

        except Exception as e:
            self.logger.error(f"Time feature creation failed: {str(e)}")
            raise
