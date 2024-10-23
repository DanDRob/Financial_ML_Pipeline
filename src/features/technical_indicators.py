"""
Technical indicators with fallback mechanism and comprehensive feature set.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple
import logging
import numpy as np
import pandas as pd
import ta


class IndicatorBackend(ABC):
    """Abstract base class for indicator backends."""

    @abstractmethod
    def sma(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        pass

    @abstractmethod
    def ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        pass

    @abstractmethod
    def macd(self, data: pd.Series, fast_period: int, slow_period: int, signal_period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD."""
        pass

    @abstractmethod
    def adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculate Average Directional Index."""
        pass

    @abstractmethod
    def rsi(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        pass

    @abstractmethod
    def bbands(self, data: pd.Series, period: int, nbdevup: int, nbdevdn: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        pass

    @abstractmethod
    def stoch(self, high: pd.Series, low: pd.Series, close: pd.Series,
              fastk_period: int, slowk_period: int, slowd_period: int) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        pass

    @abstractmethod
    def willr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculate Williams %R."""
        pass

    @abstractmethod
    def cci(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculate Commodity Channel Index."""
        pass

    @abstractmethod
    def atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculate Average True Range."""
        pass

    @abstractmethod
    def obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On Balance Volume."""
        pass

    @abstractmethod
    def mfi(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
        """Calculate Money Flow Index."""
        pass

    @abstractmethod
    def ht_trendline(self, close: pd.Series) -> pd.Series:
        """Calculate Hilbert Transform Trendline."""
        pass

    @abstractmethod
    def ht_sine(self, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate Hilbert Transform Sine Wave."""
        pass


class PythonIndicatorBackend(IndicatorBackend):
    """Pure Python implementation of technical indicators."""

    def sma(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average using pure Python."""
        return data.rolling(window=period).mean()

    def ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average using pure Python."""
        return data.ewm(span=period, adjust=False).mean()

    def macd(self, data: pd.Series, fast_period: int, slow_period: int, signal_period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD using pure Python."""
        fast_ema = self.ema(data, fast_period)
        slow_ema = self.ema(data, slow_period)
        macd_line = fast_ema - slow_ema
        signal_line = self.ema(macd_line, signal_period)
        hist = macd_line - signal_line
        return macd_line, signal_line, hist

    def adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculate ADX using pure Python."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.DataFrame({"TR1": tr1, "TR2": tr2, "TR3": tr3}).max(axis=1)
        atr = tr.rolling(window=period).mean()

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) &
                            (down_move > 0), down_move, 0)

        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx

    def rsi(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate RSI using pure Python."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def bbands(self, data: pd.Series, period: int, nbdevup: int, nbdevdn: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands using pure Python."""
        middle = self.sma(data, period)
        std = data.rolling(window=period).std()
        upper = middle + (std * nbdevup)
        lower = middle - (std * nbdevdn)
        return upper, middle, lower

    def stoch(self, high: pd.Series, low: pd.Series, close: pd.Series,
              fastk_period: int, slowk_period: int, slowd_period: int) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator using pure Python."""
        lowest_low = low.rolling(window=fastk_period).min()
        highest_high = high.rolling(window=fastk_period).max()
        fastk = 100 * (close - lowest_low) / (highest_high - lowest_low)
        slowk = fastk.rolling(window=slowk_period).mean()
        slowd = slowk.rolling(window=slowd_period).mean()
        return slowk, slowd

    def willr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculate Williams %R using pure Python."""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr

    def cci(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculate CCI using pure Python."""
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period).mean()
        mean_deviation = pd.Series(index=typical_price.index)

        for i in range(period - 1, len(typical_price)):
            mean_deviation.iloc[i] = sum(
                abs(typical_price.iloc[i-period+1:i+1] - sma.iloc[i])) / period

        cci = (typical_price - sma) / (0.015 * mean_deviation)
        return cci

    def atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculate ATR using pure Python."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.DataFrame({"TR1": tr1, "TR2": tr2, "TR3": tr3}).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        return atr

    def obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate OBV using pure Python."""
        obv = pd.Series(0, index=close.index)
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        return obv

    def mfi(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
        """Calculate MFI using pure Python."""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume

        positive_flow = pd.Series(0, index=money_flow.index)
        negative_flow = pd.Series(0, index=money_flow.index)

        for i in range(1, len(typical_price)):
            if typical_price[i] > typical_price[i-1]:
                positive_flow[i] = money_flow[i]
            elif typical_price[i] < typical_price[i-1]:
                negative_flow[i] = money_flow[i]

        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()

        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi

    def ht_trendline(self, close: pd.Series) -> pd.Series:
        """Calculate Hilbert Transform Trendline using pure Python approximation."""
        # This is a simplified approximation using double smoothing
        alpha = 0.07  # Smoothing factor
        smooth1 = close.ewm(alpha=alpha, adjust=False).mean()
        trendline = smooth1.ewm(alpha=alpha, adjust=False).mean()
        return trendline

    def ht_sine(self, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate Hilbert Transform Sine Wave using pure Python approximation."""
        # This is a simplified approximation using phase calculation
        hilbert = self.ht_trendline(close)
        phase = np.arctan2(close - hilbert, close.diff(2))
        sine = np.sin(phase)
        leadsine = np.sin(phase + np.pi/4)  # 45 degree phase shift
        return pd.Series(sine, index=close.index), pd.Series(leadsine, index=close.index)


class TAIndicatorBackend(IndicatorBackend):
    """ta implementation of technical indicators."""

    def __init__(self):
        """Initialize ta backend."""
        try:

            self.ta = ta
            self.available = True
        except ImportError:
            self.available = False
            logging.warning(
                "TA-Lib not available, indicators will fall back to Python implementation")

    def sma(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average using TA-Lib."""
        if self.available:
            return pd.Series(self.ta.SMA(data, timeperiod=period), index=data.index)
        raise NotImplementedError("TA-Lib not available")

    def ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average using TA-Lib."""
        if self.available:
            return pd.Series(self.ta.EMA(data, timeperiod=period), index=data.index)
        raise NotImplementedError("TA-Lib not available")

    def macd(self, data: pd.Series, fast_period: int, slow_period: int, signal_period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD using TA-Lib."""
        if self.available:
            macd, signal, hist = self.ta.MACD(
                data,
                fastperiod=fast_period,
                slowperiod=slow_period,
                signalperiod=signal_period
            )
            return (
                pd.Series(macd, index=data.index),
                pd.Series(signal, index=data.index),
                pd.Series(hist, index=data.index)
            )
        raise NotImplementedError("TA-Lib not available")

    def adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculate ADX using TA-Lib."""
        if self.available:
            return pd.Series(self.ta.ADX(high, low, close, timeperiod=period), index=high.index)
        raise NotImplementedError("TA-Lib not available")

    def rsi(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate RSI using TA-Lib."""
        if self.available:
            return pd.Series(self.ta.RSI(data, timeperiod=period), index=data.index)
        raise NotImplementedError("TA-Lib not available")

    def bbands(self, data: pd.Series, period: int, nbdevup: int, nbdevdn: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands using TA-Lib."""
        if self.available:
            upper, middle, lower = self.ta.BBANDS(
                data,
                timeperiod=period,
                nbdevup=nbdevup,
                nbdevdn=nbdevdn,
                matype=0
            )
            return (
                pd.Series(upper, index=data.index),
                pd.Series(middle, index=data.index),
                pd.Series(lower, index=data.index)
            )
        raise NotImplementedError("TA-Lib not available")

    def stoch(self, high: pd.Series, low: pd.Series, close: pd.Series,
              fastk_period: int, slowk_period: int, slowd_period: int) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator using TA-Lib."""
        if self.available:
            slowk, slowd = self.ta.STOCH(
                high,
                low,
                close,
                fastk_period=fastk_period,
                slowk_period=slowk_period,
                slowd_period=slowd_period
            )
            return (
                pd.Series(slowk, index=high.index),
                pd.Series(slowd, index=high.index)
            )
        raise NotImplementedError("TA-Lib not available")

    def willr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculate Williams %R using TA-Lib."""
        if self.available:
            return pd.Series(
                self.ta.WILLR(high, low, close, timeperiod=period),
                index=high.index
            )
        raise NotImplementedError("TA-Lib not available")

    def cci(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculate CCI using TA-Lib."""
        if self.available:
            return pd.Series(
                self.ta.CCI(high, low, close, timeperiod=period),
                index=high.index
            )
        raise NotImplementedError("TA-Lib not available")

    def atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculate ATR using TA-Lib."""
        if self.available:
            return pd.Series(
                self.ta.ATR(high, low, close, timeperiod=period),
                index=high.index
            )
        raise NotImplementedError("TA-Lib not available")

    def obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate OBV using TA-Lib."""
        if self.available:
            return pd.Series(
                self.ta.OBV(close, volume),
                index=close.index
            )
        raise NotImplementedError("TA-Lib not available")

    def mfi(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
        """Calculate MFI using TA-Lib."""
        if self.available:
            return pd.Series(
                self.ta.MFI(high, low, close, volume, timeperiod=period),
                index=high.index
            )
        raise NotImplementedError("TA-Lib not available")

    def ht_trendline(self, close: pd.Series) -> pd.Series:
        """Calculate Hilbert Transform Trendline using TA-Lib."""
        if self.available:
            return pd.Series(
                self.ta.HT_TRENDLINE(close),
                index=close.index
            )
        raise NotImplementedError("TA-Lib not available")

    def ht_sine(self, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate Hilbert Transform Sine Wave using TA-Lib."""
        if self.available:
            sine, leadsine = self.ta.HT_SINE(close)
            return (
                pd.Series(sine, name='sine', index=close.index),
                pd.Series(leadsine, name='leadsine', index=close.index)
            )
        raise NotImplementedError("TA-Lib not available")


class TechnicalIndicators:
    """Technical indicators with fallback mechanism."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize technical indicators with fallback mechanism."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Initialize backends
        self.ta_backend = TAIndicatorBackend()
        self.python_backend = PythonIndicatorBackend()

        # Set primary backend based on availability
        self.primary_backend = (
            self.ta_backend if self.ta_backend.available
            else self.python_backend
        )

        if not self.ta_backend.available:
            self.logger.info(
                "Using pure Python implementation for technical indicators")

    def generate_indicators(
        self,
        df: pd.DataFrame,
        indicators: List[str]
    ) -> pd.DataFrame:
        """
        Generate specified technical indicators.

        Args:
            df: Input DataFrame with OHLCV data
            indicators: List of indicator names to generate

        Returns:
            DataFrame with added technical indicators
        """
        try:
            df = df.copy()

            for indicator in indicators:
                if hasattr(self, f"add_{indicator}"):
                    df = getattr(self, f"add_{indicator}")(df)
                else:
                    self.logger.warning(
                        f"Indicator {indicator} not implemented")

            return df

        except Exception as e:
            self.logger.error(f"Indicator generation failed: {str(e)}")
            raise

    def add_sma(self, df: pd.DataFrame, periods: List[int] = [20, 50, 200]) -> pd.DataFrame:
        """Add Simple Moving Averages."""
        for period in periods:
            df[f'sma_{period}'] = ta.trend.sma_indicator(
                df['Close'], window=period)
        return df

    def add_ema(self, df: pd.DataFrame, periods: List[int] = [12, 26]) -> pd.DataFrame:
        """Add Exponential Moving Averages."""
        for period in periods:
            df[f'ema_{period}'] = ta.trend.ema_indicator(
                df['Close'], window=period)
        return df

    def add_macd(
        self,
        df: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> pd.DataFrame:
        """Add MACD indicator."""
        macd = ta.trend.MACD(
            df['Close'],
            window_slow=slow_period,
            window_fast=fast_period,
            window_sign=signal_period
        )
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        return df

    def add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add RSI indicator."""
        df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=period).rsi()
        return df

    def add_bbands(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0
    ) -> pd.DataFrame:
        """Add Bollinger Bands."""
        bb = ta.volatility.BollingerBands(
            df['Close'],
            window=period,
            window_dev=std_dev
        )
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        return df

    def add_stoch(
        self,
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3
    ) -> pd.DataFrame:
        """Add Stochastic Oscillator."""
        stoch = ta.momentum.StochasticOscillator(
            df['High'],
            df['Low'],
            df['Close'],
            window=k_period,
            smooth_window=d_period
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        return df

    def add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average True Range."""
        df['atr'] = ta.volatility.AverageTrueRange(
            df['High'],
            df['Low'],
            df['Close'],
            window=period
        ).average_true_range()
        return df

    def add_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average Directional Index."""
        adx = ta.trend.ADXIndicator(
            df['High'],
            df['Low'],
            df['Close'],
            window=period
        )
        df['adx'] = adx.adx()
        df['di_plus'] = adx.adx_pos()
        df['di_minus'] = adx.adx_neg()
        return df

    def add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add On Balance Volume."""
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(
            df['Close'],
            df['Volume']
        ).on_balance_volume()
        return df

    def add_cci(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add Commodity Channel Index."""
        df['cci'] = ta.trend.CCIIndicator(
            df['High'],
            df['Low'],
            df['Close'],
            window=period
        ).cci()
        return df

    def add_aroon(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Aroon indicator to the dataframe."""
        try:
            aroon = ta.trend.AroonIndicator(
                high=df['High'],  # Add High price
                low=df['Low'],    # Add Low price
                window=14         # Optional: specify window period
            )
            df['aroon_up'] = aroon.aroon_up()
            df['aroon_down'] = aroon.aroon_down()
            return df
        except Exception as e:
            self.logger.error(f"Aroon Indicator generation failed: {str(e)}")
            raise

    def add_williams(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Williams %R."""
        df['williams_r'] = ta.momentum.WilliamsRIndicator(
            df['High'],
            df['Low'],
            df['Close'],
            lbp=period
        ).williams_r()
        return df

    def add_ao(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Awesome Oscillator."""
        df['ao'] = ta.momentum.AwesomeOscillatorIndicator(
            df['High'],
            df['Low']
        ).awesome_oscillator()
        return df

    def add_kst(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Know Sure Thing (KST) oscillator."""
        df['kst'] = ta.trend.KSTIndicator(df['Close']).kst()
        df['kst_sig'] = ta.trend.KSTIndicator(df['Close']).kst_sig()
        return df

    def add_roc(self, df: pd.DataFrame, period: int = 12) -> pd.DataFrame:
        """Add Rate of Change."""
        df['roc'] = ta.momentum.ROCIndicator(
            df['Close'],
            window=period
        ).roc()
        return df

    def add_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Money Flow Index."""
        df['mfi'] = ta.volume.MFIIndicator(
            df['High'],
            df['Low'],
            df['Close'],
            df['Volume'],
            window=period
        ).money_flow_index()
        return df

    def add_cmf(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add Chaikin Money Flow."""
        df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(
            df['High'],
            df['Low'],
            df['Close'],
            df['Volume'],
            window=period
        ).chaikin_money_flow()
        return df

    def calculate_all_indicators(
        self,
        data: pd.DataFrame,
        include_groups: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Calculate all technical indicators."""
        try:
            df = data.copy()
            groups = include_groups or [
                'trend', 'momentum', 'volatility', 'volume', 'cycle']

            for group in groups:
                if group == 'trend':
                    df = self.add_trend_indicators(df)
                elif group == 'momentum':
                    df = self.add_momentum_indicators(df)
                elif group == 'volatility':
                    df = self.add_volatility_indicators(df)
                elif group == 'volume':
                    df = self.add_volume_indicators(df)
                elif group == 'cycle':
                    df = self.add_cycle_indicators(df)

            return df

        except Exception as e:
            self.logger.error(
                f"Technical indicator calculation failed: {str(e)}")
            raise

    def add_trend_indicators(
        self,
        data: pd.DataFrame,
        periods: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Add trend-based technical indicators."""
        try:
            df = data.copy()
            periods = periods or [10, 20, 50, 100, 200]

            # Moving Averages
            for period in periods:
                df[f'sma_{period}'] = self._calculate_indicator(
                    'sma', df['Close'], period)
                df[f'ema_{period}'] = self._calculate_indicator(
                    'ema', df['Close'], period)

            # MACD
            macd, signal, hist = self._calculate_indicator(
                'macd',
                df['Close'],
                12, 26, 9
            )
            df['macd'] = macd
            df['macd_signal'] = signal
            df['macd_hist'] = hist

            # ADX
            df['adx'] = self._calculate_indicator(
                'adx',
                df['High'],
                df['Low'],
                df['Close'],
                14
            )

            return df

        except Exception as e:
            self.logger.error(f"Trend indicator calculation failed: {str(e)}")
            raise

    def _calculate_indicator(
        self,
        func_name: str,
        *args,
        **kwargs
    ) -> Union[pd.Series, Tuple[pd.Series, ...]]:
        """Calculate indicator with fallback mechanism."""
        try:
            # Try primary backend first
            func = getattr(self.primary_backend, func_name)
            return func(*args, **kwargs)
        except Exception as e:
            self.logger.warning(
                f"Primary backend failed for {
                    func_name}, falling back to Python implementation: {str(e)}"
            )
            # Fall back to Python implementation
            func = getattr(self.python_backend, func_name)
            return func(*args, **kwargs)

    def add_momentum_indicators(
        self,
        data: pd.DataFrame,
        config: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Add momentum-based technical indicators."""
        try:
            df = data.copy()
            config = config or {}

            # RSI
            df['rsi'] = self._calculate_indicator('rsi', df['Close'], 14)

            # Stochastic
            slowk, slowd = self._calculate_indicator(
                'stoch',
                df['High'],
                df['Low'],
                df['Close'],
                14, 3, 3
            )
            df['stoch_k'] = slowk
            df['stoch_d'] = slowd

            # Williams %R
            df['williams_r'] = self._calculate_indicator(
                'willr',
                df['High'],
                df['Low'],
                df['Close'],
                14
            )

            # CCI
            df['cci'] = self._calculate_indicator(
                'cci',
                df['High'],
                df['Low'],
                df['Close'],
                14
            )

            # MFI
            df['mfi'] = self._calculate_indicator(
                'mfi',
                df['High'],
                df['Low'],
                df['Close'],
                df['Volume'],
                14
            )

            return df

        except Exception as e:
            self.logger.error(
                f"Momentum indicator calculation failed: {str(e)}")
            raise

    def add_volatility_indicators(
        self,
        data: pd.DataFrame,
        config: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Add volatility-based technical indicators."""
        try:
            df = data.copy()
            config = config or {}

            # Bollinger Bands
            upper, middle, lower = self._calculate_indicator(
                'bbands',
                df['Close'],
                20, 2, 2
            )
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower

            # ATR
            df['atr'] = self._calculate_indicator(
                'atr',
                df['High'],
                df['Low'],
                df['Close'],
                14
            )

            # Calculate additional volatility metrics
            df['volatility'] = df['Close'].pct_change().rolling(
                window=20).std() * np.sqrt(252)
            df['range'] = (df['High'] - df['Low']) / df['Close']

            return df

        except Exception as e:
            self.logger.error(
                f"Volatility indicator calculation failed: {str(e)}")
            raise

    def add_volume_indicators(
        self,
        data: pd.DataFrame,
        config: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Add volume-based technical indicators."""
        try:
            df = data.copy()
            config = config or {}

            # OBV
            df['obv'] = self._calculate_indicator(
                'obv', df['Close'], df['Volume'])

            # Volume ROC
            df['volume_roc'] = df['Volume'].pct_change()

            # VWAP
            df['vwap'] = (df['Volume'] * (df['High'] + df['Low'] +
                          df['Close']) / 3).cumsum() / df['Volume'].cumsum()

            # Chaikin Money Flow
            mfm = ((df['Close'] - df['Low']) - (df['High'] -
                   df['Close'])) / (df['High'] - df['Low'])
            mfv = mfm * df['Volume']
            df['cmf'] = mfv.rolling(window=20).sum(
            ) / df['Volume'].rolling(window=20).sum()

            return df

        except Exception as e:
            self.logger.error(f"Volume indicator calculation failed: {str(e)}")
            raise

    def add_cycle_indicators(
        self,
        data: pd.DataFrame,
        config: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Add cycle-based technical indicators."""
        try:
            df = data.copy()
            config = config or {}

            # Hilbert Transform Trendline
            df['ht_trendline'] = self._calculate_indicator(
                'ht_trendline', df['Close'])

            # Hilbert Transform Sine Wave
            sine, leadsine = self._calculate_indicator('ht_sine', df['Close'])
            df['ht_sine'] = sine
            df['ht_leadsine'] = leadsine

            return df

        except Exception as e:
            self.logger.error(f"Cycle indicator calculation failed: {str(e)}")
            raise

    def get_indicator_metadata(self) -> Dict[str, Dict[str, str]]:
        """Get metadata for all indicators."""
        return {
            'trend': {
                'sma': 'Simple Moving Average',
                'ema': 'Exponential Moving Average',
                'macd': 'Moving Average Convergence Divergence',
                'adx': 'Average Directional Index',
                'ht_trendline': 'Hilbert Transform Trendline',
                'ht_sine': 'Hilbert Transform Sine Wave',
            },
            'momentum': {
                'rsi': 'Relative Strength Index',
                'stoch': 'Stochastic Oscillator',
                'willr': 'Williams %R',
                'cci': 'Commodity Channel Index',
                'mfi': 'Money Flow Index'
            },
            'volatility': {
                'bbands': 'Bollinger Bands',
                'atr': 'Average True Range'
            },
            'volume': {
                'obv': 'On Balance Volume',
                'volume_roc': 'Volume Rate of Change',
                'vwap': 'Volume Weighted Average Price',
                'cmf': 'Chaikin Money Flow'
            },
            'cycle': {
                'ht_trendline': 'Hilbert Transform Trendline',
                'ht_sine': 'Hilbert Transform Sine Wave'
            }
        }
