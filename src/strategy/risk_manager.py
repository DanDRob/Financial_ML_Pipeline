"""
Risk management and position sizing module.
"""

from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class RiskConfig:
    """Risk management configuration."""
    # Portfolio risk limits
    max_portfolio_var: float = 0.15
    max_drawdown: float = 0.20
    var_confidence: float = 0.95
    max_leverage: float = 1.0

    # Position risk limits
    max_position_size: float = 0.10
    max_correlation: float = 0.70
    stop_loss: float = 0.02
    take_profit: float = 0.04

    # Risk factors
    risk_factors: List[str] = None
    factor_constraints: Dict[str, Tuple[float, float]] = None

    # Dynamic sizing
    volatility_lookback: int = 63
    correlation_lookback: int = 252
    use_dynamic_sizing: bool = True
    kelly_fraction: float = 0.5


class RiskManager:
    """Risk management and position sizing."""

    def __init__(self, config: RiskConfig):
        """
        Initialize risk manager.

        Args:
            config: Risk configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.position_limits = {}
        self.risk_metrics = {}

    def calculate_position_sizes(
        self,
        signals: pd.DataFrame,
        data: pd.DataFrame,
        portfolio_value: float
    ) -> pd.DataFrame:
        """
        Calculate position sizes based on risk limits.

        Args:
            signals: Trading signals DataFrame
            data: Market data DataFrame
            portfolio_value: Current portfolio value

        Returns:
            DataFrame of position sizes
        """
        try:
            position_sizes = pd.DataFrame(
                0, index=signals.index, columns=signals.columns)

            # Calculate volatilities and correlations
            volatilities = self._calculate_volatilities(data)
            correlations = self._calculate_correlations(data)

            # Calculate risk-adjusted position sizes
            for asset in signals.columns:
                if signals[asset].iloc[-1] != 0:
                    # Base position size
                    size = self._calculate_base_position_size(
                        asset,
                        signals[asset].iloc[-1],
                        volatilities[asset],
                        portfolio_value
                    )

                    # Adjust for correlations
                    size = self._adjust_for_correlations(
                        asset,
                        size,
                        correlations
                    )

                    position_sizes[asset].iloc[-1] = size

            # Apply portfolio-level constraints
            position_sizes = self._apply_portfolio_constraints(
                position_sizes,
                data,
                portfolio_value
            )

            return position_sizes

        except Exception as e:
            self.logger.error(f"Position size calculation failed: {str(e)}")
            raise

    def _calculate_volatilities(
        self,
        data: pd.DataFrame,
        window: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate asset volatilities.

        Args:
            data: Market data DataFrame
            window: Optional lookback window

        Returns:
            Series of annualized volatilities
        """
        window = window or self.config.volatility_lookback
        returns = data.pct_change()
        volatilities = returns.std() * np.sqrt(252)
        return volatilities

    def _calculate_correlations(
        self,
        data: pd.DataFrame,
        window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate asset correlations.

        Args:
            data: Market data DataFrame
            window: Optional lookback window

        Returns:
            Correlation matrix
        """
        window = window or self.config.correlation_lookback
        returns = data.pct_change()
        correlations = returns.rolling(window).corr()
        return correlations.iloc[-len(data.columns):]

    def _calculate_base_position_size(
        self,
        asset: str,
        signal: float,
        volatility: float,
        portfolio_value: float
    ) -> float:
        """
        Calculate base position size for asset.

        Args:
            asset: Asset symbol
            signal: Trading signal
            volatility: Asset volatility
            portfolio_value: Current portfolio value

        Returns:
            Base position size
        """
        try:
            if not self.config.use_dynamic_sizing:
                return signal * self.config.max_position_size

            # Calculate Kelly position size
            win_prob = 0.5 + abs(signal) / 2  # Convert signal to probability
            win_loss_ratio = self.config.take_profit / self.config.stop_loss
            kelly_size = (win_prob * win_loss_ratio -
                          (1 - win_prob)) / win_loss_ratio

            # Apply fractional Kelly
            kelly_size *= self.config.kelly_fraction

            # Adjust for volatility
            vol_adjustment = self.config.max_portfolio_var / volatility
            position_size = min(kelly_size, vol_adjustment)

            # Apply maximum position size constraint
            position_size = min(position_size, self.config.max_position_size)

            return position_size * np.sign(signal)

        except Exception as e:
            self.logger.error(
                f"Base position size calculation failed: {str(e)}")
            raise

    def _adjust_for_correlations(
        self,
        asset: str,
        size: float,
        correlations: pd.DataFrame
    ) -> float:
        """
        Adjust position size based on correlations.

        Args:
            asset: Asset symbol
            size: Base position size
            correlations: Correlation matrix

        Returns:
            Adjusted position size
        """
        try:
            # Find highly correlated assets
            corr_assets = correlations[asset][
                abs(correlations[asset]) > self.config.max_correlation
            ]

            if len(corr_assets) > 1:  # Exclude self-correlation
                # Reduce position size based on number of correlations
                adjustment = 1 / np.sqrt(len(corr_assets))
                return size * adjustment

            return size

        except Exception as e:
            self.logger.error(f"Correlation adjustment failed: {str(e)}")
            raise

    def _apply_portfolio_constraints(
        self,
        position_sizes: pd.DataFrame,
        data: pd.DataFrame,
        portfolio_value: float
    ) -> pd.DataFrame:
        """
        Apply portfolio-level risk constraints.

        Args:
            position_sizes: Position sizes DataFrame
            data: Market data DataFrame
            portfolio_value: Current portfolio value

        Returns:
            Constrained position sizes
        """
        try:
            # Calculate portfolio metrics
            weights = position_sizes.iloc[-1] / portfolio_value
            returns = data.pct_change()

            # Calculate portfolio variance
            port_var = self._calculate_portfolio_variance(weights, returns)

            # Scale positions if variance exceeds limit
            if port_var > self.config.max_portfolio_var:
                scale_factor = np.sqrt(
                    self.config.max_portfolio_var / port_var)
                position_sizes *= scale_factor

            # Apply leverage constraint
            total_exposure = abs(weights).sum()
            if total_exposure > self.config.max_leverage:
                scale_factor = self.config.max_leverage / total_exposure
                position_sizes *= scale_factor

            # Apply factor constraints
            if self.config.risk_factors and self.config.factor_constraints:
                position_sizes = self._apply_factor_constraints(
                    position_sizes,
                    data
                )

            return position_sizes

        except Exception as e:
            self.logger.error(
                f"Portfolio constraint application failed: {str(e)}")
            raise

    def _calculate_portfolio_variance(
        self,
        weights: pd.Series,
        returns: pd.DataFrame
    ) -> float:
        """Calculate portfolio variance."""
        cov_matrix = returns.cov()
        port_var = np.dot(weights, np.dot(cov_matrix, weights))
        return port_var

    def _apply_factor_constraints(
        self,
        position_sizes: pd.DataFrame,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply risk factor constraints."""
        try:
            # Calculate factor exposures
            factor_exposures = self._calculate_factor_exposures(
                position_sizes,
                data
            )

            # Check and adjust for factor constraints
            for factor, (min_exp, max_exp) in self.config.factor_constraints.items():
                exposure = factor_exposures.get(factor, 0)

                if exposure < min_exp:
                    scale = min_exp / exposure
                    position_sizes *= scale
                elif exposure > max_exp:
                    scale = max_exp / exposure
                    position_sizes *= scale

            return position_sizes

        except Exception as e:
            self.logger.error(
                f"Factor constraint application failed: {str(e)}")
            raise

    def _calculate_factor_exposures(
        self,
        position_sizes: pd.DataFrame,
        data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate risk factor exposures."""
        # Placeholder - implement based on specific risk factors
        return {}

    def calculate_risk_metrics(
        self,
        portfolio: pd.DataFrame,
        data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics.

        Args:
            portfolio: Portfolio positions DataFrame
            data: Market data DataFrame

        Returns:
            Dictionary of risk metrics
        """
        try:
            returns = self._calculate_portfolio_returns(portfolio, data)

            metrics = {
                'volatility': float(returns.std() * np.sqrt(252)),
                'var_95': float(self._calculate_var(returns)),
                'cvar_95': float(self._calculate_cvar(returns)),
                'max_drawdown': float(self._calculate_max_drawdown(returns)),
                'beta': float(self._calculate_beta(returns, data)),
                'sharpe_ratio': float(self._calculate_sharpe_ratio(returns)),
                'sortino_ratio': float(self._calculate_sortino_ratio(returns)),
                'tail_ratio': float(self._calculate_tail_ratio(returns)),
                'skewness': float(stats.skew(returns)),
                'kurtosis': float(stats.kurtosis(returns)),
                'leverage': float(abs(portfolio).sum().max())
            }

            self.risk_metrics = metrics
            return metrics

        except Exception as e:
            self.logger.error(f"Risk metric calculation failed: {str(e)}")
            raise

    def _calculate_portfolio_returns(
        self,
        portfolio: pd.DataFrame,
        data: pd.DataFrame
    ) -> pd.Series:
        """Calculate portfolio returns series."""
        weights = portfolio.div(portfolio.abs().sum(axis=1), axis=0)
        returns = (weights * data.pct_change()).sum(axis=1)
        return returns

    def _calculate_var(
        self,
        returns: pd.Series,
        confidence: Optional[float] = None
    ) -> float:
        """Calculate Value at Risk."""
        confidence = confidence or self.config.var_confidence
        return abs(np.percentile(returns, (1 - confidence) * 100))

    def _calculate_cvar(
        self,
        returns: pd.Series,
        confidence: Optional[float] = None
    ) -> float:
        """Calculate Conditional Value at Risk."""
        confidence = confidence or self.config.var_confidence
        var = self._calculate_var(returns, confidence)
        return abs(returns[returns <= -var].mean())

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = cumulative / running_max - 1
        return float(drawdowns.min())

    def _calculate_beta(
        self,
        returns: pd.Series,
        data: pd.DataFrame,
        market_col: str = 'SPY'
    ) -> float:
        """Calculate portfolio beta."""
        if market_col in data.columns:
            market_returns = data[market_col].pct_change()
            covariance = returns.cov(market_returns)
            market_variance = market_returns.var()
            return covariance / market_variance
        return 0.0

    def _calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        rf_rate: Optional[float] = None
    ) -> float:
        """Calculate Sharpe ratio."""
        rf_rate = rf_rate or self.config.risk_free_rate
        excess_returns = returns - rf_rate/252
        return np.sqrt(252) * excess_returns.mean() / returns.std()

    def _calculate_sortino_ratio(
        self,
        returns: pd.Series,
        rf_rate: Optional[float] = None
    ) -> float:
        """Calculate Sortino ratio."""
        rf_rate = rf_rate or self.config.risk_free_rate
        excess_returns = returns - rf_rate/252
        downside_returns = returns[returns < 0]
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()

    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """Calculate tail ratio."""
        top_tenth = np.percentile(returns, 90)
        bottom_tenth = np.percentile(returns, 10)
        return abs(top_tenth / bottom_tenth)
