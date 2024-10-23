"""
Performance attribution and analysis module.
"""

from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from src.strategy.portfolio_monitor import PortfolioMonitor
from pathlib import Path


@dataclass
class AttributionConfig:
    """Configuration for performance attribution."""
    # Analysis parameters
    benchmark_symbol: str = 'SPY'
    risk_free_rate: float = 0.02
    analysis_frequency: str = 'daily'  # 'daily', 'weekly', 'monthly'
    rolling_window: int = 252

    # Risk factor analysis
    risk_factors: List[str] = None
    factor_windows: List[int] = None

    # Visualization
    plot_style: str = 'seaborn'
    figure_size: Tuple[int, int] = (12, 8)
    save_plots: bool = True
    plot_dir: str = 'reports/plots'


class PerformanceAttribution:
    """Performance attribution and analysis."""

    def __init__(
        self,
        config: AttributionConfig,
        portfolio_monitor: 'PortfolioMonitor'
    ):
        """
        Initialize performance attribution.

        Args:
            config: Attribution configuration
            portfolio_monitor: Portfolio monitor instance
        """
        self.config = config
        self.portfolio_monitor = portfolio_monitor
        self.logger = logging.getLogger(__name__)

        # Set plotting style
        plt.style.use(self.config.plot_style)

        self.benchmark_data = None
        self.factor_data = None
        self.attribution_results = {}

    def run_attribution_analysis(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """
        Run comprehensive performance attribution analysis.

        Args:
            start_date: Analysis start date
            end_date: Analysis end date

        Returns:
            Dictionary of attribution results
        """
        try:
            # Get portfolio data
            portfolio_data = self._get_portfolio_data(start_date, end_date)

            # Get benchmark data
            self.benchmark_data = self._get_benchmark_data(
                start_date, end_date)

            # Calculate returns
            portfolio_returns = self._calculate_returns(portfolio_data)
            benchmark_returns = self._calculate_returns(self.benchmark_data)

            # Run analyses
            results = {
                'return_attribution': self._analyze_return_attribution(
                    portfolio_returns,
                    benchmark_returns
                ),
                'risk_attribution': self._analyze_risk_attribution(
                    portfolio_returns,
                    benchmark_returns
                ),
                'factor_attribution': self._analyze_factor_attribution(
                    portfolio_returns
                ),
                'style_attribution': self._analyze_style_attribution(
                    portfolio_returns
                ),
                'sector_attribution': self._analyze_sector_attribution(
                    portfolio_returns
                ),
                'trading_attribution': self._analyze_trading_attribution()
            }

            # Generate plots
            if self.config.save_plots:
                self._generate_attribution_plots(results)

            self.attribution_results = results
            return results

        except Exception as e:
            self.logger.error(f"Attribution analysis failed: {str(e)}")
            raise

    def _analyze_return_attribution(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict:
        """Analyze return attribution components."""
        try:
            # Calculate excess returns
            excess_returns = portfolio_returns - benchmark_returns

            # Calculate components
            allocation_effect = self._calculate_allocation_effect()
            selection_effect = self._calculate_selection_effect()
            interaction_effect = self._calculate_interaction_effect()

            # Calculate attribution metrics
            metrics = {
                'total_return': float(portfolio_returns.sum()),
                'excess_return': float(excess_returns.sum()),
                'allocation_effect': allocation_effect,
                'selection_effect': selection_effect,
                'interaction_effect': interaction_effect,
                'active_return': float(excess_returns.mean() * 252),
                'tracking_error': float(excess_returns.std() * np.sqrt(252)),
                'information_ratio': float(
                    excess_returns.mean() / excess_returns.std() * np.sqrt(252)
                )
            }

            # Calculate rolling metrics
            rolling_metrics = self._calculate_rolling_attribution(
                portfolio_returns,
                benchmark_returns
            )

            return {
                'metrics': metrics,
                'rolling_metrics': rolling_metrics
            }

        except Exception as e:
            self.logger.error(f"Return attribution analysis failed: {str(e)}")
            raise

    def _analyze_risk_attribution(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict:
        """Analyze risk attribution components."""
        try:
            # Calculate risk metrics
            portfolio_risk = portfolio_returns.std() * np.sqrt(252)
            tracking_risk = (portfolio_returns -
                             benchmark_returns).std() * np.sqrt(252)

            # Decompose risk
            risk_decomposition = self._decompose_risk(portfolio_returns)

            # Calculate factor contributions
            factor_contributions = self._calculate_factor_contributions(
                portfolio_returns
            )

            # Calculate risk-adjusted metrics
            metrics = {
                'total_risk': float(portfolio_risk),
                'systematic_risk': float(risk_decomposition['systematic']),
                'idiosyncratic_risk': float(risk_decomposition['idiosyncratic']),
                'tracking_risk': float(tracking_risk),
                'diversification_ratio': float(
                    self._calculate_diversification_ratio()
                ),
                'factor_contributions': factor_contributions
            }

            # Calculate conditional metrics
            conditional_metrics = self._calculate_conditional_metrics(
                portfolio_returns,
                benchmark_returns
            )

            return {
                'metrics': metrics,
                'conditional_metrics': conditional_metrics,
                'risk_decomposition': risk_decomposition
            }

        except Exception as e:
            self.logger.error(f"Risk attribution analysis failed: {str(e)}")
            raise

    def _analyze_factor_attribution(
        self,
        portfolio_returns: pd.Series
    ) -> Dict:
        """Analyze factor attribution components."""
        try:
            if not self.config.risk_factors:
                return {}

            # Get factor data
            factor_data = self._get_factor_data()

            # Run factor regression
            factor_exposures, factor_returns = self._run_factor_regression(
                portfolio_returns,
                factor_data
            )

            # Calculate factor contributions
            factor_contributions = self._calculate_factor_contributions(
                factor_exposures,
                factor_returns
            )

            # Calculate style tilts
            style_tilts = self._calculate_style_tilts(factor_exposures)

            return {
                'factor_exposures': factor_exposures,
                'factor_returns': factor_returns,
                'factor_contributions': factor_contributions,
                'style_tilts': style_tilts,
                'r_squared': self._calculate_r_squared(
                    portfolio_returns,
                    factor_data,
                    factor_exposures
                )
            }

        except Exception as e:
            self.logger.error(f"Factor attribution analysis failed: {str(e)}")
            raise

    def _analyze_style_attribution(
        self,
        portfolio_returns: pd.Series
    ) -> Dict:
        """Analyze investment style attribution."""
        try:
            # Calculate style exposures
            style_exposures = self._calculate_style_exposures()

            # Calculate style returns
            style_returns = self._calculate_style_returns()

            # Calculate style contributions
            style_contributions = self._calculate_style_contributions(
                style_exposures,
                style_returns
            )

            return {
                'style_exposures': style_exposures,
                'style_returns': style_returns,
                'style_contributions': style_contributions,
                'active_style_bets': self._calculate_active_style_bets(
                    style_exposures
                )
            }

        except Exception as e:
            self.logger.error(f"Style attribution analysis failed: {str(e)}")
            raise

    def _analyze_sector_attribution(
        self,
        portfolio_returns: pd.Series
    ) -> Dict:
        """Analyze sector attribution."""
        try:
            # Get sector weights
            portfolio_weights = self.portfolio_monitor.current_weights
            sector_weights = self._calculate_sector_weights(portfolio_weights)

            # Calculate sector returns
            sector_returns = self._calculate_sector_returns()

            # Calculate attribution effects
            allocation_effect = self._calculate_sector_allocation_effect(
                sector_weights,
                sector_returns
            )

            selection_effect = self._calculate_sector_selection_effect(
                sector_weights,
                sector_returns
            )

            return {
                'sector_weights': sector_weights,
                'sector_returns': sector_returns,
                'allocation_effect': allocation_effect,
                'selection_effect': selection_effect,
                'total_effect': {
                    sector: allocation_effect[sector] +
                    selection_effect[sector]
                    for sector in allocation_effect
                }
            }

        except Exception as e:
            self.logger.error(f"Sector attribution analysis failed: {str(e)}")
            raise

    def _analyze_trading_attribution(self) -> Dict:
        """Analyze trading attribution and costs."""
        try:
            # Get trading data
            fills = self.portfolio_monitor.execution_manager.get_fills()

            if not fills:
                return {}

            # Calculate trading costs
            implementation_shortfall = self._calculate_implementation_shortfall(
                fills
            )

            market_impact = self._calculate_market_impact(fills)

            timing_cost = self._calculate_timing_cost(fills)

            return {
                'implementation_shortfall': implementation_shortfall,
                'market_impact': market_impact,
                'timing_cost': timing_cost,
                'total_cost': implementation_shortfall + market_impact + timing_cost,
                'cost_analysis': self._analyze_trading_costs(fills),
                'execution_quality': self._analyze_execution_quality(fills)
            }

        except Exception as e:
            self.logger.error(f"Trading attribution analysis failed: {str(e)}")
            raise

    def _generate_attribution_plots(self, results: Dict) -> None:
        """Generate attribution analysis plots."""
        try:
            # Create plot directory if it doesn't exist
            Path(self.config.plot_dir).mkdir(parents=True, exist_ok=True)

            # Return attribution plots
            self._plot_return_attribution(results['return_attribution'])

            # Risk attribution plots
            self._plot_risk_attribution(results['risk_attribution'])

            # Factor attribution plots
            if results['factor_attribution']:
                self._plot_factor_attribution(results['factor_attribution'])

            # Style attribution plots
            self._plot_style_attribution(results['style_attribution'])

            # Sector attribution plots
            self._plot_sector_attribution(results['sector_attribution'])

            # Trading attribution plots
            self._plot_trading_attribution(results['trading_attribution'])

        except Exception as e:
            self.logger.error(f"Plot generation failed: {str(e)}")
            raise

    def _plot_return_attribution(self, results: Dict) -> None:
        """Plot return attribution analysis."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)

            # Cumulative returns
            self._plot_cumulative_returns(axes[0, 0])

            # Attribution effects
            self._plot_attribution_effects(axes[0, 1])

            # Rolling metrics
            self._plot_rolling_metrics(axes[1, 0])

            # Performance statistics
            self._plot_performance_stats(axes[1, 1])

            plt.tight_layout()
            plt.savefig(f"{self.config.plot_dir}/return_attribution.png")
            plt.close()

        except Exception as e:
            self.logger.error(f"Return attribution plotting failed: {str(e)}")
            raise

    def generate_report(
        self,
        output_format: str = 'html'
    ) -> str:
        """
        Generate attribution analysis report.

        Args:
            output_format: Report format ('html' or 'pdf')

        Returns:
            Path to generated report
        """
        try:
            if not self.attribution_results:
                raise ValueError("No attribution results available")

            # Create report template
            template = self._create_report_template()

            # Add analysis results
            template = self._add_analysis_results(template)

            # Add plots
            template = self._add_plots(template)

            # Generate report
            output_path = f"{self.config.plot_dir}/attribution_report"
            if output_format == 'html':
                self._generate_html_report(template, output_path)
            elif output_format == 'pdf':
                self._generate_pdf_report(template, output_path)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")

            return f"{output_path}.{output_format}"

        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            raise
