from scipy import stats
import pandas as pd
import numpy as np
from pathlib import Path
import multiprocessing as mp
from datetime import datetime
from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Union, Callable
from strategy.trading_strategy import TradingStrategy
from data.data_loader import DataLoader

"""
Backtesting engine for financial ML strategies.
"""


@dataclass
class BacktestConfig:
    """Configuration for backtesting engine."""
    # Data parameters
    start_date: str
    end_date: str
    universe: List[str]
    frequency: str = "daily"

    # Trading parameters
    initial_capital: float = 1000000
    commission_rate: float = 0.001
    slippage_rate: float = 0.001

    # Risk parameters
    max_position_size: float = 0.1
    max_leverage: float = 1.0
    stop_loss: float = 0.02

    # Test parameters
    use_walk_forward: bool = True
    window_size: int = 252
    step_size: int = 63
    n_walks: int = 5

    # Performance parameters
    benchmark: str = "SPY"
    risk_free_rate: float = 0.02


class BacktestEngine:
    """Backtesting engine for strategy evaluation."""

    def __init__(
        self,
        config: BacktestConfig,
        strategy: 'TradingStrategy',
        data_loader: 'DataLoader'
    ):
        """
        Initialize backtesting engine.

        Args:
            config: Backtest configuration
            strategy: Trading strategy instance
            data_loader: Data loader instance
        """
        self.config = config
        self.strategy = strategy
        self.data_loader = data_loader
        self.logger = logging.getLogger(__name__)

        self.results = {}
        self.positions = {}
        self.trades = []
        self.metrics = {}

        # Initialize performance tracking
        self.equity_curve = pd.Series()
        self.drawdown_curve = pd.Series()
        self.position_history = pd.DataFrame()

    def run_backtest(self) -> Dict:
        """
        Run backtest simulation.

        Returns:
            Dictionary of backtest results
        """
        try:
            # Load data
            data = self._load_backtest_data()

            if self.config.use_walk_forward:
                results = self._run_walk_forward_test(data)
            else:
                results = self._run_single_backtest(data)

            # Calculate performance metrics
            self.metrics = self._calculate_backtest_metrics(results)

            # Store results
            self.results = {
                'data': data,
                'trades': self.trades,
                'positions': self.position_history,
                'equity_curve': self.equity_curve,
                'drawdown_curve': self.drawdown_curve,
                'metrics': self.metrics
            }

            return self.results

        except Exception as e:
            self.logger.error(f"Backtest execution failed: {str(e)}")
            raise

    def _load_backtest_data(self) -> pd.DataFrame:
        """Load and prepare backtest data."""
        try:
            # Load market data
            market_data = self.data_loader.fetch_data(
                symbols=self.config.universe,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                frequency=self.config.frequency
            )

            # Load benchmark data
            benchmark_data = self.data_loader.fetch_data(
                symbols=[self.config.benchmark],
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                frequency=self.config.frequency
            )

            # Merge data
            data = self.data_loader.merge_data(
                {
                    'market': market_data,
                    'benchmark': benchmark_data[self.config.benchmark]
                }
            )

            return data

        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}")
            raise

    def _run_walk_forward_test(self, data: pd.DataFrame) -> Dict:
        """Run walk-forward optimization and testing."""
        try:
            results = []

            # Generate walk-forward windows
            windows = self._generate_walk_forward_windows(
                len(data),
                self.config.window_size,
                self.config.step_size,
                self.config.n_walks
            )

            # Run tests in parallel
            with mp.Pool() as pool:
                window_results = pool.starmap(
                    self._run_window_test,
                    [(data.iloc[train_idx], data.iloc[test_idx])
                     for train_idx, test_idx in windows]
                )

            # Combine results
            for train_results, test_results in window_results:
                results.append({
                    'train': train_results,
                    'test': test_results
                })

            return results

        except Exception as e:
            self.logger.error(f"Walk-forward testing failed: {str(e)}")
            raise

    def _run_single_backtest(self, data: pd.DataFrame) -> Dict:
        """Run single full-sample backtest."""
        try:
            # Initialize portfolio
            portfolio = self._initialize_portfolio()

            # Run simulation
            for timestamp, bar in data.iterrows():
                # Update strategy
                signals = self.strategy.generate_signals(bar)

                # Execute trades
                trades = self._execute_trades(
                    signals,
                    bar,
                    portfolio
                )

                # Update portfolio
                portfolio = self._update_portfolio(
                    portfolio,
                    trades,
                    bar
                )

                # Track performance
                self._update_performance_tracking(
                    portfolio,
                    timestamp
                )

            return {
                'portfolio': portfolio,
                'trades': self.trades,
                'equity_curve': self.equity_curve,
                'drawdown_curve': self.drawdown_curve
            }

        except Exception as e:
            self.logger.error(f"Single backtest failed: {str(e)}")
            raise

    def _execute_trades(
        self,
        signals: Dict[str, float],
        bar: pd.Series,
        portfolio: Dict
    ) -> List[Dict]:
        """Execute trading signals with transaction costs."""
        try:
            trades = []

            for symbol, signal in signals.items():
                if signal == 0:
                    continue

                # Calculate position size
                size = self._calculate_position_size(
                    signal,
                    symbol,
                    bar,
                    portfolio
                )

                if size == 0:
                    continue

                # Calculate transaction costs
                price = bar[f"{symbol}_Close"]
                commission = abs(size * price * self.config.commission_rate)
                slippage = abs(size * price * self.config.slippage_rate)

                # Record trade
                trade = {
                    'timestamp': bar.name,
                    'symbol': symbol,
                    'size': size,
                    'price': price,
                    'commission': commission,
                    'slippage': slippage,
                    'total_cost': commission + slippage
                }

                trades.append(trade)
                self.trades.append(trade)

            return trades

        except Exception as e:
            self.logger.error(f"Trade execution failed: {str(e)}")
            raise

    def _calculate_position_size(
        self,
        signal: float,
        symbol: str,
        bar: pd.Series,
        portfolio: Dict
    ) -> float:
        """Calculate position size with risk constraints."""
        try:
            # Get current portfolio value
            portfolio_value = self._get_portfolio_value(portfolio, bar)

            # Calculate base position size
            base_size = portfolio_value * self.config.max_position_size

            # Apply leverage constraint
            total_exposure = sum(abs(pos['size'] * bar[f"{s}_Close"])
                                 for s, pos in portfolio['positions'].items())
            available_leverage = (portfolio_value * self.config.max_leverage -
                                  total_exposure)

            max_size = min(base_size, available_leverage)

            # Calculate final size
            size = signal * max_size / bar[f"{symbol}_Close"]

            return size

        except Exception as e:
            self.logger.error(f"Position size calculation failed: {str(e)}")
            raise

    def _update_portfolio(
        self,
        portfolio: Dict,
        trades: List[Dict],
        bar: pd.Series
    ) -> Dict:
        """Update portfolio state with trades."""
        try:
            # Update positions
            for trade in trades:
                symbol = trade['symbol']

                if symbol not in portfolio['positions']:
                    portfolio['positions'][symbol] = {
                        'size': 0,
                        'cost_basis': 0
                    }

                # Update position
                position = portfolio['positions'][symbol]
                old_size = position['size']
                new_size = old_size + trade['size']

                if new_size == 0:
                    del portfolio['positions'][symbol]
                else:
                    # Update cost basis
                    old_cost = position['cost_basis'] * old_size
                    new_cost = trade['size'] * trade['price']
                    position['size'] = new_size
                    position['cost_basis'] = (old_cost + new_cost) / new_size

                # Update cash
                portfolio['cash'] -= (trade['size'] * trade['price'] +
                                      trade['total_cost'])

            # Update values
            portfolio['value'] = self._get_portfolio_value(portfolio, bar)

            return portfolio

        except Exception as e:
            self.logger.error(f"Portfolio update failed: {str(e)}")
            raise

    def _update_performance_tracking(
        self,
        portfolio: Dict,
        timestamp: datetime
    ) -> None:
        """Update performance tracking metrics."""
        try:
            # Update equity curve
            self.equity_curve[timestamp] = portfolio['value']

            # Update drawdown curve
            if len(self.equity_curve) > 0:
                rolling_max = self.equity_curve.expanding().max()
                self.drawdown_curve[timestamp] = (
                    self.equity_curve[timestamp] / rolling_max[timestamp] - 1
                )

            # Update position history
            positions = pd.Series({
                symbol: pos['size']
                for symbol, pos in portfolio['positions'].items()
            })
            self.position_history.loc[timestamp] = positions

        except Exception as e:
            self.logger.error(f"Performance tracking update failed: {str(e)}")
            raise

    def _calculate_backtest_metrics(self, results: Dict) -> Dict:
        """Calculate comprehensive backtest performance metrics."""
        try:
            returns = self.equity_curve.pct_change().dropna()

            metrics = {
                'total_return': float(self.equity_curve.iloc[-1] /
                                      self.equity_curve.iloc[0] - 1),
                'annualized_return': float(
                    (1 + returns.mean()) ** 252 - 1
                ),
                'volatility': float(returns.std() * np.sqrt(252)),
                'sharpe_ratio': float(
                    (returns.mean() - self.config.risk_free_rate/252) /
                    returns.std() * np.sqrt(252)
                ),
                'sortino_ratio': float(
                    (returns.mean() - self.config.risk_free_rate/252) /
                    returns[returns < 0].std() * np.sqrt(252)
                ),
                'max_drawdown': float(self.drawdown_curve.min()),
                'win_rate': float(np.mean(returns > 0)),
                'profit_factor': float(
                    abs(returns[returns > 0].sum() /
                        returns[returns < 0].sum())
                ),
                'trade_metrics': self._calculate_trade_metrics(),
                'risk_metrics': self._calculate_risk_metrics(returns)
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Metric calculation failed: {str(e)}")
            raise

    def _calculate_trade_metrics(self) -> Dict:
        """Calculate trade-specific metrics."""
        try:
            if not self.trades:
                return {}

            trade_returns = pd.Series([
                trade['size'] * (trade['price'] - trade['cost_basis'])
                for trade in self.trades
            ])

            metrics = {
                'total_trades': len(self.trades),
                'avg_trade_return': float(trade_returns.mean()),
                'avg_trade_size': float(
                    np.mean([abs(t['size']) for t in self.trades])
                ),
                'avg_commission': float(
                    np.mean([t['commission'] for t in self.trades])
                ),
                'avg_slippage': float(
                    np.mean([t['slippage'] for t in self.trades])
                )
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Trade metric calculation failed: {str(e)}")
            raise

    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict:
        """Calculate risk-related metrics."""
        try:
            metrics = {
                'var_95': float(np.percentile(returns, 5)),
                'cvar_95': float(
                    returns[returns <= np.percentile(returns, 5)].mean()
                ),
                'skewness': float(stats.skew(returns)),
                'kurtosis': float(stats.kurtosis(returns)),
                'leverage_utilization': float(
                    self.position_history.abs().sum(axis=1).mean() /
                    self.equity_curve
                ),
                'beta': self._calculate_portfolio_beta(returns)
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Risk metric calculation failed: {str(e)}")
            raise
