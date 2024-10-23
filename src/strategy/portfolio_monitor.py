"""
Portfolio monitoring and rebalancing system.
"""

from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from datetime import datetime, time
import pickle
import numpy as np
import pandas as pd
from src.strategy.execution_manager import ExecutionManager, OrderType


@dataclass
class MonitorConfig:
    """Portfolio monitoring configuration."""
    # Monitoring parameters
    update_frequency: int = 60  # seconds
    rebalance_threshold: float = 0.05
    drift_threshold: float = 0.02

    # Risk limits
    max_drawdown: float = 0.20
    max_leverage: float = 1.0
    var_limit: float = 0.02

    # Alert thresholds
    alert_drawdown: float = 0.15
    alert_var: float = 0.018
    alert_leverage: float = 0.9

    # Trading hours
    market_open: time = time(9, 30)
    market_close: time = time(16, 0)


class PortfolioMonitor:
    """Portfolio monitoring and rebalancing system."""

    def __init__(
        self,
        config: MonitorConfig,
        execution_manager: 'ExecutionManager'
    ):
        """
        Initialize portfolio monitor.

        Args:
            config: Monitor configuration
            execution_manager: Execution manager instance
        """
        self.config = config
        self.execution_manager = execution_manager
        self.logger = logging.getLogger(__name__)

        self.target_weights = {}
        self.current_weights = {}
        self.metrics_history = []
        self.alerts = []
        self.last_update = None

    def update(
        self,
        current_prices: pd.Series,
        target_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Update portfolio monitoring.

        Args:
            current_prices: Current asset prices
            target_weights: Optional new target weights

        Returns:
            Dictionary of current portfolio metrics
        """
        try:
            current_time = datetime.now().time()

            # Check if market is open
            if not self._is_market_open(current_time):
                return {}

            # Update target weights if provided
            if target_weights is not None:
                self.target_weights = target_weights

            # Calculate current weights and metrics
            metrics = self._calculate_portfolio_metrics(current_prices)

            # Check for rebalancing needs
            if self._check_rebalance_needed(metrics):
                self._initiate_rebalance(current_prices)

            # Check risk limits and generate alerts
            self._check_risk_limits(metrics)

            # Store metrics history
            self.metrics_history.append({
                'timestamp': datetime.now(),
                **metrics
            })

            self.last_update = datetime.now()
            return metrics

        except Exception as e:
            self.logger.error(f"Portfolio update failed: {str(e)}")
            raise

    def _calculate_portfolio_metrics(
        self,
        current_prices: pd.Series
    ) -> Dict[str, float]:
        """Calculate current portfolio metrics."""
        try:
            positions = self.execution_manager.get_all_positions()

            # Calculate portfolio value and weights
            total_value = 0
            current_weights = {}

            for symbol, position in positions.items():
                if symbol in current_prices:
                    value = position['quantity'] * current_prices[symbol]
                    total_value += value
                    current_weights[symbol] = value

            # Normalize weights
            if total_value > 0:
                current_weights = {
                    k: v/total_value for k, v in current_weights.items()
                }

            self.current_weights = current_weights

            # Calculate metrics
            metrics = {
                'total_value': total_value,
                'current_weights': current_weights,
                'tracking_error': self._calculate_tracking_error(),
                'leverage': self._calculate_leverage(),
                'var': self._calculate_var(),
                'drawdown': self._calculate_drawdown(),
                'drift': self._calculate_portfolio_drift()
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Metrics calculation failed: {str(e)}")
            raise

    def _check_rebalance_needed(self, metrics: Dict[str, float]) -> bool:
        """Check if portfolio rebalancing is needed."""
        try:
            # Check drift threshold
            if metrics['drift'] > self.config.drift_threshold:
                return True

            # Check tracking error
            if metrics['tracking_error'] > self.config.rebalance_threshold:
                return True

            # Check risk limits
            if (metrics['leverage'] > self.config.max_leverage or
                metrics['var'] > self.config.var_limit or
                    metrics['drawdown'] > self.config.max_drawdown):
                return True

            return False

        except Exception as e:
            self.logger.error(f"Rebalance check failed: {str(e)}")
            return False

    def _initiate_rebalance(self, current_prices: pd.Series) -> None:
        """Initiate portfolio rebalancing."""
        try:
            # Calculate target positions
            trades = self._calculate_rebalance_trades(current_prices)

            # Execute trades
            for symbol, quantity in trades.items():
                if abs(quantity) > 0:
                    direction = 1 if quantity > 0 else -1
                    self.execution_manager.submit_order(
                        symbol=symbol,
                        quantity=abs(quantity),
                        order_type=OrderType.MARKET,
                        direction=direction,
                        algo='VWAP'  # Use VWAP for rebalancing
                    )

            self.logger.info("Portfolio rebalancing initiated")

        except Exception as e:
            self.logger.error(f"Rebalance initiation failed: {str(e)}")
            raise

    def _calculate_rebalance_trades(
        self,
        current_prices: pd.Series
    ) -> Dict[str, float]:
        """Calculate required trades for rebalancing."""
        try:
            trades = {}
            positions = self.execution_manager.get_all_positions()
            portfolio_value = sum(
                pos['quantity'] * current_prices[sym]
                for sym, pos in positions.items()
                if sym in current_prices
            )

            for symbol, target_weight in self.target_weights.items():
                current_position = positions.get(symbol, {'quantity': 0})
                current_value = (
                    current_position['quantity'] * current_prices[symbol]
                )
                target_value = portfolio_value * target_weight

                # Calculate required trade
                trade_value = target_value - current_value
                trade_quantity = trade_value / current_prices[symbol]

                # Round to nearest tradeable quantity
                trade_quantity = round(trade_quantity, 8)

                if abs(trade_quantity) > 0:
                    trades[symbol] = trade_quantity

            return trades

        except Exception as e:
            self.logger.error(f"Trade calculation failed: {str(e)}")
            raise

    def _check_risk_limits(self, metrics: Dict[str, float]) -> None:
        """Check risk limits and generate alerts."""
        try:
            current_time = datetime.now()

            # Check drawdown
            if metrics['drawdown'] > self.config.alert_drawdown:
                self.alerts.append({
                    'timestamp': current_time,
                    'type': 'DRAWDOWN',
                    'message': f"Drawdown ({metrics['drawdown']:.2%}) exceeded alert threshold"
                })

            # Check VaR
            if metrics['var'] > self.config.alert_var:
                self.alerts.append({
                    'timestamp': current_time,
                    'type': 'VAR',
                    'message': f"VaR ({metrics['var']:.2%}) exceeded alert threshold"
                })

            # Check leverage
            if metrics['leverage'] > self.config.alert_leverage:
                self.alerts.append({
                    'timestamp': current_time,
                    'type': 'LEVERAGE',
                    'message': f"Leverage ({metrics['leverage']:.2f}) exceeded alert threshold"
                })

        except Exception as e:
            self.logger.error(f"Risk limit check failed: {str(e)}")
            raise

    def _calculate_tracking_error(self) -> float:
        """Calculate tracking error from target weights."""
        try:
            if not self.target_weights or not self.current_weights:
                return 0.0

            squared_diff = 0
            for symbol, target_weight in self.target_weights.items():
                current_weight = self.current_weights.get(symbol, 0)
                squared_diff += (target_weight - current_weight) ** 2

            return np.sqrt(squared_diff)

        except Exception as e:
            self.logger.error(f"Tracking error calculation failed: {str(e)}")
            return 0.0

    def _calculate_leverage(self) -> float:
        """Calculate current portfolio leverage."""
        try:
            return sum(abs(w) for w in self.current_weights.values())
        except Exception as e:
            self.logger.error(f"Leverage calculation failed: {str(e)}")
            return 0.0

    def _calculate_var(self) -> float:
        """Calculate Value at Risk."""
        try:
            if len(self.metrics_history) < 252:  # Need sufficient history
                return 0.0

            # Get historical returns
            returns = pd.Series([
                m['total_value'] for m in self.metrics_history
            ]).pct_change().dropna()

            # Calculate historical VaR
            return abs(np.percentile(returns, 5))

        except Exception as e:
            self.logger.error(f"VaR calculation failed: {str(e)}")
            return 0.0

    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown."""
        try:
            if len(self.metrics_history) < 2:
                return 0.0

            values = pd.Series([
                m['total_value'] for m in self.metrics_history
            ])

            peak = values.expanding().max()
            drawdown = (values - peak) / peak
            return abs(float(drawdown.iloc[-1]))
        except Exception as e:
            self.logger.error(f"Drawdown calculation failed: {str(e)}")
            return 0.0

    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown."""
        try:
            if len(self.metrics_history) < 2:
                return 0.0

            values = pd.Series([
                m['total_value'] for m in self.metrics_history
            ])

            peak = values.expanding().max()
            drawdown = (values - peak) / peak
            return abs(float(drawdown.iloc[-1]))

        except Exception as e:
            self.logger.error(f"Drawdown calculation failed: {str(e)}")
            return 0.0

    def _calculate_portfolio_drift(self) -> float:
        """Calculate portfolio drift from target weights."""
        try:
            if not self.target_weights or not self.current_weights:
                return 0.0

            drift = sum(
                abs(self.current_weights.get(symbol, 0) - target_weight)
                for symbol, target_weight in self.target_weights.items()
            )

            return drift / 2  # Divide by 2 to get proper drift measure

        except Exception as e:
            self.logger.error(f"Portfolio drift calculation failed: {str(e)}")
            return 0.0

    def _is_market_open(self, current_time: time) -> bool:
        """Check if market is currently open."""
        return self.config.market_open <= current_time <= self.config.market_close

    def get_portfolio_state(self) -> Dict:
        """Get current portfolio state and metrics."""
        try:
            if not self.last_update:
                return {}

            latest_metrics = self.metrics_history[-1] if self.metrics_history else {}

            return {
                'timestamp': self.last_update,
                'portfolio_value': latest_metrics.get('total_value', 0),
                'current_weights': self.current_weights,
                'target_weights': self.target_weights,
                'tracking_error': latest_metrics.get('tracking_error', 0),
                'leverage': latest_metrics.get('leverage', 0),
                'var': latest_metrics.get('var', 0),
                'drawdown': latest_metrics.get('drawdown', 0),
                'drift': latest_metrics.get('drift', 0),
                'active_alerts': [
                    alert for alert in self.alerts
                    if (datetime.now() - alert['timestamp']).seconds < 3600
                ]
            }

        except Exception as e:
            self.logger.error(f"Portfolio state retrieval failed: {str(e)}")
            return {}

    def generate_report(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict:
        """Generate comprehensive portfolio report."""
        try:
            # Filter metrics history
            metrics = pd.DataFrame(self.metrics_history)
            if start_time:
                metrics = metrics[metrics['timestamp'] >= start_time]
            if end_time:
                metrics = metrics[metrics['timestamp'] <= end_time]

            if len(metrics) == 0:
                return {}

            # Calculate performance metrics
            initial_value = metrics['total_value'].iloc[0]
            final_value = metrics['total_value'].iloc[-1]
            returns = metrics['total_value'].pct_change()

            report = {
                'period_start': metrics['timestamp'].iloc[0],
                'period_end': metrics['timestamp'].iloc[-1],
                'initial_value': initial_value,
                'final_value': final_value,
                'total_return': (final_value / initial_value - 1),
                'annualized_return': self._calculate_annualized_return(returns),
                'volatility': returns.std() * np.sqrt(252),
                'sharpe_ratio': self._calculate_sharpe_ratio(returns),
                'max_drawdown': self._calculate_max_drawdown(metrics['total_value']),
                'tracking_error_avg': metrics['tracking_error'].mean(),
                'tracking_error_max': metrics['tracking_error'].max(),
                'var_avg': metrics['var'].mean(),
                'var_max': metrics['var'].max(),
                'leverage_avg': metrics['leverage'].mean(),
                'leverage_max': metrics['leverage'].max(),
                'drift_avg': metrics['drift'].mean(),
                'drift_max': metrics['drift'].max(),
                'rebalance_events': self._count_rebalance_events(metrics),
                'risk_events': self._summarize_risk_events(),
                'position_summary': self._generate_position_summary(),
                'trading_summary': self._generate_trading_summary()
            }

            return report

        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            return {}

    def _calculate_annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return."""
        try:
            total_days = (returns.index[-1] - returns.index[0]).days
            if total_days == 0:
                return 0.0

            cumulative_return = (1 + returns).prod() - 1
            annualized_return = (
                1 + cumulative_return) ** (252 / total_days) - 1
            return float(annualized_return)

        except Exception as e:
            self.logger.error(
                f"Annualized return calculation failed: {str(e)}")
            return 0.0

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        try:
            if len(returns) < 2:
                return 0.0

            excess_returns = returns - self.config.risk_free_rate / 252
            sharpe = np.sqrt(252) * excess_returns.mean() / returns.std()
            return float(sharpe)

        except Exception as e:
            self.logger.error(f"Sharpe ratio calculation failed: {str(e)}")
            return 0.0

    def _calculate_max_drawdown(self, values: pd.Series) -> float:
        """Calculate maximum drawdown over period."""
        try:
            peak = values.expanding().max()
            drawdown = (values - peak) / peak
            return float(drawdown.min())

        except Exception as e:
            self.logger.error(f"Max drawdown calculation failed: {str(e)}")
            return 0.0

    def _count_rebalance_events(self, metrics: pd.DataFrame) -> int:
        """Count number of rebalancing events."""
        try:
            drift_exceeded = metrics['drift'] > self.config.drift_threshold
            tracking_exceeded = metrics['tracking_error'] > self.config.rebalance_threshold
            return int((drift_exceeded | tracking_exceeded).sum())

        except Exception as e:
            self.logger.error(f"Rebalance event counting failed: {str(e)}")
            return 0

    def _summarize_risk_events(self) -> Dict[str, int]:
        """Summarize risk events by type."""
        try:
            summary = {
                'DRAWDOWN': 0,
                'VAR': 0,
                'LEVERAGE': 0
            }

            for alert in self.alerts:
                if alert['type'] in summary:
                    summary[alert['type']] += 1

            return summary

        except Exception as e:
            self.logger.error(f"Risk event summarization failed: {str(e)}")
            return {}

    def _generate_position_summary(self) -> Dict[str, Dict]:
        """Generate summary of positions."""
        try:
            positions = self.execution_manager.get_all_positions()
            summary = {}

            for symbol, position in positions.items():
                if position['quantity'] != 0:
                    summary[symbol] = {
                        'quantity': position['quantity'],
                        'avg_price': position['average_price'],
                        'cost_basis': position['cost_basis'],
                        'target_weight': self.target_weights.get(symbol, 0),
                        'current_weight': self.current_weights.get(symbol, 0)
                    }

            return summary

        except Exception as e:
            self.logger.error(f"Position summary generation failed: {str(e)}")
            return {}

    def _generate_trading_summary(self) -> Dict[str, float]:
        """Generate trading activity summary."""
        try:
            fills = self.execution_manager.get_fills()

            if not fills:
                return {}

            total_volume = sum(abs(f['quantity']) for f in fills)
            total_commission = sum(f['commission'] for f in fills)

            return {
                'trade_count': len(fills),
                'total_volume': total_volume,
                'total_commission': total_commission,
                'avg_trade_size': total_volume / len(fills) if fills else 0,
                'commission_per_trade': total_commission / len(fills) if fills else 0
            }

        except Exception as e:
            self.logger.error(f"Trading summary generation failed: {str(e)}")
            return {}

    def get_alerts(self) -> List[Dict]:
        """Get active alerts."""
        return [
            alert for alert in self.alerts
            if (datetime.now() - alert['timestamp']).seconds < 3600
        ]

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self.alerts = []

    def save_state(self, filepath: str) -> None:
        """Save monitor state to file."""
        try:
            state = {
                'target_weights': self.target_weights,
                'current_weights': self.current_weights,
                'metrics_history': self.metrics_history,
                'alerts': self.alerts,
                'last_update': self.last_update
            }

            with open(filepath, 'wb') as f:
                pickle.dump(state, f)

            self.logger.info(f"Monitor state saved to {filepath}")

        except Exception as e:
            self.logger.error(f"State saving failed: {str(e)}")
            raise

    def load_state(self, filepath: str) -> None:
        """Load monitor state from file."""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)

            self.target_weights = state['target_weights']
            self.current_weights = state['current_weights']
            self.metrics_history = state['metrics_history']
            self.alerts = state['alerts']
            self.last_update = state['last_update']

            self.logger.info(f"Monitor state loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"State loading failed: {str(e)}")
            raise
