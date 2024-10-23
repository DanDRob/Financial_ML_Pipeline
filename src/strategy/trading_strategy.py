"""
Trading strategy implementation with portfolio optimization and risk management.
"""

from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import cvxopt
from cvxopt import matrix, solvers
from src.models.model_wrapper import BaseModelWrapper


@dataclass
class StrategyConfig:
    """Configuration for trading strategy."""
    # Trading parameters
    position_size: float = 0.1
    max_positions: int = 10
    max_correlation: float = 0.7
    rebalance_frequency: str = 'monthly'

    # Risk parameters
    max_drawdown: float = 0.2
    stop_loss: float = 0.02
    take_profit: float = 0.04
    risk_free_rate: float = 0.02
    target_volatility: float = 0.15
    max_leverage: float = 1.0

    # Portfolio constraints
    sector_constraints: Dict[str, float] = None
    asset_constraints: Dict[str, Tuple[float, float]] = None

    # Transaction costs
    commission_rate: float = 0.001
    slippage_rate: float = 0.001


class TradingStrategy:
    """Trading strategy implementation with portfolio optimization."""

    def __init__(
        self,
        config: StrategyConfig,
        model_wrapper: 'BaseModelWrapper'
    ):
        """
        Initialize trading strategy.

        Args:
            config: Strategy configuration
            model_wrapper: Model wrapper instance
        """
        self.config = config
        self.model = model_wrapper
        self.logger = logging.getLogger(__name__)

        self.positions = {}
        self.portfolio_value = 1.0
        self.current_weights = None
        self.performance_history = []

    def generate_signals(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate trading signals.

        Args:
            data: Market data DataFrame
            features: Feature DataFrame

        Returns:
            DataFrame with trading signals
        """
        try:
            # Generate predictions
            predictions = self.model.predict_proba(features)

            # Convert to signal dataframe
            signals = pd.DataFrame(
                0,
                index=data.index,
                columns=data.columns
            )

            # Apply signal rules
            for asset in data.columns:
                prob = predictions[asset] if isinstance(
                    predictions, dict) else predictions

                # Long signal
                signals.loc[prob[:, 1] > 0.7, asset] = 1

                # Short signal (if allowed)
                signals.loc[prob[:, 1] < 0.3, asset] = -1

            return signals

        except Exception as e:
            self.logger.error(f"Signal generation failed: {str(e)}")
            raise

    def optimize_portfolio(
        self,
        signals: pd.DataFrame,
        data: pd.DataFrame,
        method: str = 'mean_variance'
    ) -> pd.Series:
        """
        Optimize portfolio weights.

        Args:
            signals: Trading signals DataFrame
            data: Market data DataFrame
            method: Optimization method

        Returns:
            Series of optimal weights
        """
        try:
            # Filter assets with active signals
            active_assets = signals.columns[signals.iloc[-1] != 0]

            if len(active_assets) == 0:
                return pd.Series(0, index=signals.columns)

            # Calculate returns and covariance
            returns = data[active_assets].pct_change()
            expected_returns = returns.mean()
            covariance = returns.cov()

            if method == 'mean_variance':
                weights = self._mean_variance_optimization(
                    expected_returns,
                    covariance
                )
            elif method == 'risk_parity':
                weights = self._risk_parity_optimization(covariance)
            elif method == 'min_variance':
                weights = self._minimum_variance_optimization(covariance)
            elif method == 'max_sharpe':
                weights = self._maximum_sharpe_optimization(
                    expected_returns,
                    covariance
                )
            else:
                raise ValueError(f"Unknown optimization method: {method}")

            # Apply position sizing and constraints
            weights = self._apply_constraints(weights, active_assets)

            # Expand weights to all assets
            full_weights = pd.Series(0, index=signals.columns)
            full_weights[active_assets] = weights

            return full_weights

        except Exception as e:
            self.logger.error(f"Portfolio optimization failed: {str(e)}")
            raise

    def _mean_variance_optimization(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
        target_return: Optional[float] = None
    ) -> pd.Series:
        """
        Perform mean-variance optimization.

        Args:
            expected_returns: Expected returns series
            covariance: Covariance matrix
            target_return: Optional target return

        Returns:
            Series of optimal weights
        """
        try:
            n_assets = len(expected_returns)

            # Convert to cvxopt matrices
            P = matrix(covariance.values)
            q = matrix(np.zeros(n_assets))

            # Constraints
            # Sum of weights = 1
            A = matrix(1.0, (1, n_assets))
            b = matrix(1.0)

            # Asset constraints
            if self.config.asset_constraints:
                constraints = np.array([
                    self.config.asset_constraints.get(asset, (0, 1))
                    for asset in expected_returns.index
                ])
                G = matrix(np.vstack((-np.eye(n_assets), np.eye(n_assets))))
                h = matrix(np.hstack((-constraints[:, 0], constraints[:, 1])))
            else:
                G = matrix(np.vstack((-np.eye(n_assets), np.eye(n_assets))))
                h = matrix(np.hstack((np.zeros(n_assets), np.ones(n_assets))))

            # Target return constraint
            if target_return is not None:
                A = matrix(np.vstack((A, expected_returns.values)))
                b = matrix([1.0, target_return])

            # Solve optimization problem
            solvers.options['show_progress'] = False
            sol = solvers.qp(P, q, G, h, A, b)

            if sol['status'] != 'optimal':
                raise ValueError("Optimization failed to converge")

            weights = pd.Series(
                np.array(sol['x']).flatten(),
                index=expected_returns.index
            )

            return weights

        except Exception as e:
            self.logger.error(f"Mean-variance optimization failed: {str(e)}")
            raise

    def _risk_parity_optimization(
        self,
        covariance: pd.DataFrame
    ) -> pd.Series:
        """
        Perform risk parity optimization.

        Args:
            covariance: Covariance matrix

        Returns:
            Series of optimal weights
        """
        try:
            n_assets = len(covariance)

            def risk_contribution(weights):
                portfolio_risk = np.sqrt(
                    np.dot(weights.T, np.dot(covariance, weights)))
                asset_contribution = np.dot(
                    covariance, weights) / portfolio_risk
                return asset_contribution * weights

            def objective(weights):
                risk_contrib = risk_contribution(weights)
                return np.sum((risk_contrib - risk_contrib.mean())**2)

            # Constraints
            constraints = [
                # weights sum to 1
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            ]

            bounds = tuple((0, 1) for _ in range(n_assets))

            # Initial guess: equal weights
            initial_weights = np.ones(n_assets) / n_assets

            # Optimize
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

            if not result.success:
                raise ValueError("Risk parity optimization failed to converge")

            weights = pd.Series(
                result.x,
                index=covariance.index
            )

            return weights

        except Exception as e:
            self.logger.error(f"Risk parity optimization failed: {str(e)}")
            raise

    def _minimum_variance_optimization(
        self,
        covariance: pd.DataFrame
    ) -> pd.Series:
        """
        Perform minimum variance optimization.

        Args:
            covariance: Covariance matrix

        Returns:
            Series of optimal weights
        """
        try:
            n_assets = len(covariance)

            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(covariance, weights))

            # Constraints
            constraints = [
                # weights sum to 1
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            ]

            bounds = tuple((0, 1) for _ in range(n_assets))

            # Initial guess: equal weights
            initial_weights = np.ones(n_assets) / n_assets

            # Optimize
            result = minimize(
                portfolio_variance,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

            if not result.success:
                raise ValueError(
                    "Minimum variance optimization failed to converge")

            weights = pd.Series(
                result.x,
                index=covariance.index
            )

            return weights

        except Exception as e:
            self.logger.error(
                f"Minimum variance optimization failed: {str(e)}")
            raise

    def _maximum_sharpe_optimization(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame
    ) -> pd.Series:
        """
        Perform maximum Sharpe ratio optimization.

        Args:
            expected_returns: Expected returns series
            covariance: Covariance matrix

        Returns:
            Series of optimal weights
        """
        try:
            n_assets = len(expected_returns)

            def negative_sharpe_ratio(weights):
                portfolio_return = np.sum(expected_returns * weights)
                portfolio_vol = np.sqrt(
                    np.dot(weights.T, np.dot(covariance, weights)))
                return -(portfolio_return - self.config.risk_free_rate) / portfolio_vol

            # Constraints
            constraints = [
                # weights sum to 1
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            ]

            bounds = tuple((0, 1) for _ in range(n_assets))

            # Initial guess: equal weights
            initial_weights = np.ones(n_assets) / n_assets

            # Optimize
            result = minimize(
                negative_sharpe_ratio,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

            if not result.success:
                raise ValueError(
                    "Maximum Sharpe optimization failed to converge")

            weights = pd.Series(
                result.x,
                index=expected_returns.index
            )

            return weights

        except Exception as e:
            self.logger.error(f"Maximum Sharpe optimization failed: {str(e)}")
            raise

    def _apply_constraints(
        self,
        weights: pd.Series,
        active_assets: pd.Index
    ) -> pd.Series:
        """
        Apply portfolio constraints to weights.

        Args:
            weights: Portfolio weights
            active_assets: Active asset indices

        Returns:
            Constrained weights
        """
        try:
            # Apply maximum position size
            weights = weights.clip(upper=self.config.position_size)

            # Apply maximum number of positions
            if len(active_assets) > self.config.max_positions:
                top_positions = weights.nlargest(self.config.max_positions)
                weights[~weights.index.isin(top_positions.index)] = 0

            # Apply sector constraints
            if self.config.sector_constraints:
                sector_weights = self._calculate_sector_weights(weights)
                for sector, max_weight in self.config.sector_constraints.items():
                    if sector_weights.get(sector, 0) > max_weight:
                        sector_assets = [
                            asset for asset in weights.index
                            if self._get_asset_sector(asset) == sector
                        ]
                        scale_factor = max_weight / sector_weights[sector]
                        weights[sector_assets] *= scale_factor

            # Normalize weights
            weights = weights / weights.sum() if weights.sum() > 0 else weights

            return weights

        except Exception as e:
            self.logger.error(f"Constraint application failed: {str(e)}")
            raise

    def _calculate_sector_weights(self, weights: pd.Series) -> Dict[str, float]:
        """Calculate sector weights from asset weights."""
        sector_weights = {}
        for asset in weights.index:
            sector = self._get_asset_sector(asset)
            sector_weights[sector] = sector_weights.get(
                sector, 0) + weights[asset]
        return sector_weights

    def _get_asset_sector(self, asset: str) -> str:
        """Get sector for asset (placeholder - implement based on data)."""
        return "Unknown"
