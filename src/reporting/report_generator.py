import markdown2
import pdfkit
from jinja2 import Environment, FileSystemLoader
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Union

from src.analysis.trade_analysis import TradeAnalyzer
from src.strategy.portfolio_monitor import PortfolioMonitor

"""
Report generation module for financial ML pipeline.
"""


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    # Report settings
    template_dir: str = "templates"
    output_dir: str = "reports"
    report_format: str = "html"

    # Content settings
    include_plots: bool = True
    include_tables: bool = True
    include_metrics: bool = True
    include_recommendations: bool = True

    # Style settings
    theme: str = "light"
    logo_path: Optional[str] = None
    custom_css: Optional[str] = None


class ReportGenerator:
    """Report generation for trading strategy results."""

    def __init__(
        self,
        config: ReportConfig,
        trade_analyzer: 'TradeAnalyzer',
        portfolio_monitor: 'PortfolioMonitor'
    ):
        """
        Initialize report generator.

        Args:
            config: Report configuration
            trade_analyzer: Trade analyzer instance
            portfolio_monitor: Portfolio monitor instance
        """
        self.config = config
        self.trade_analyzer = trade_analyzer
        self.portfolio_monitor = portfolio_monitor
        self.logger = logging.getLogger(__name__)

        # Initialize template engine
        self.template_env = Environment(
            loader=FileSystemLoader(self.config.template_dir)
        )

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        report_type: str = "comprehensive",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> str:
        """
        Generate trading strategy report.

        Args:
            report_type: Type of report to generate
            start_date: Start date for report period
            end_date: End date for report period

        Returns:
            Path to generated report
        """
        try:
            # Get report data
            data = self._collect_report_data(
                report_type,
                start_date,
                end_date
            )

            # Generate report content
            content = self._generate_report_content(
                report_type,
                data
            )

            # Create report file
            output_path = self._create_report_file(
                content,
                report_type
            )

            self.logger.info(f"Report generated: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            raise

    def _collect_report_data(
        self,
        report_type: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> Dict:
        """Collect data for report generation."""
        try:
            data = {
                'metadata': {
                    'report_type': report_type,
                    'start_date': start_date,
                    'end_date': end_date,
                    'generation_time': datetime.now()
                }
            }

            # Get trade analysis results
            trade_results = self.trade_analyzer.analyze_trades(
                start_date,
                end_date
            )

            # Get portfolio state
            portfolio_state = self.portfolio_monitor.get_portfolio_state()

            # Combine results
            data.update({
                'trade_analysis': trade_results,
                'portfolio_state': portfolio_state,
                'performance_metrics': self._calculate_performance_metrics(),
                'risk_metrics': self._calculate_risk_metrics(),
                'recommendations': self._generate_recommendations()
            })

            return data

        except Exception as e:
            self.logger.error(f"Data collection failed: {str(e)}")
            raise

    def _generate_report_content(
        self,
        report_type: str,
        data: Dict
    ) -> Dict:
        """Generate report content from template."""
        try:
            # Load appropriate template
            template = self.template_env.get_template(
                f"{report_type}_report.html")

            # Generate plots if enabled
            plots = {}
            if self.config.include_plots:
                plots = self._generate_report_plots(data)

            # Generate tables if enabled
            tables = {}
            if self.config.include_tables:
                tables = self._generate_report_tables(data)

            # Render template
            content = template.render(
                data=data,
                plots=plots,
                tables=tables,
                theme=self.config.theme,
                logo_path=self.config.logo_path
            )

            return content

        except Exception as e:
            self.logger.error(f"Content generation failed: {str(e)}")
            raise

    def _create_report_file(
        self,
        content: str,
        report_type: str
    ) -> str:
        """Create report file in specified format."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{report_type}_report_{timestamp}"

            if self.config.report_format == "html":
                output_path = f"{self.config.output_dir}/{filename}.html"
                with open(output_path, 'w') as f:
                    f.write(content)

            elif self.config.report_format == "pdf":
                html_path = f"{self.config.output_dir}/{filename}.html"
                pdf_path = f"{self.config.output_dir}/{filename}.pdf"

                # Save HTML first
                with open(html_path, 'w') as f:
                    f.write(content)

                # Convert to PDF
                pdfkit.from_file(html_path, pdf_path)
                output_path = pdf_path

            else:
                raise ValueError(f"Unsupported report format: {
                                 self.config.report_format}")

            return output_path

        except Exception as e:
            self.logger.error(f"Report file creation failed: {str(e)}")
            raise

    def _generate_report_plots(self, data: Dict) -> Dict:
        """Generate plots for report."""
        try:
            plots = {}

            # Performance plots
            plots['performance'] = {
                'equity_curve': self._create_equity_curve_plot(data),
                'drawdown': self._create_drawdown_plot(data),
                'returns_distribution': self._create_returns_distribution_plot(data)
            }

            # Trade analysis plots
            plots['trades'] = {
                'execution_quality': self._create_execution_quality_plot(data),
                'cost_analysis': self._create_cost_analysis_plot(data),
                'timing_analysis': self._create_timing_analysis_plot(data)
            }

            # Risk analysis plots
            plots['risk'] = {
                'risk_metrics': self._create_risk_metrics_plot(data),
                'exposure_analysis': self._create_exposure_analysis_plot(data),
                'correlation_matrix': self._create_correlation_matrix_plot(data)
            }

            return plots

        except Exception as e:
            self.logger.error(f"Plot generation failed: {str(e)}")
            raise

    def _generate_report_tables(self, data: Dict) -> Dict:
        """Generate tables for report."""
        try:
            tables = {}

            # Performance tables
            tables['performance'] = {
                'summary_metrics': self._create_summary_metrics_table(data),
                'monthly_returns': self._create_monthly_returns_table(data),
                'risk_metrics': self._create_risk_metrics_table(data)
            }

            # Trade analysis tables
            tables['trades'] = {
                'execution_summary': self._create_execution_summary_table(data),
                'venue_analysis': self._create_venue_analysis_table(data),
                'cost_breakdown': self._create_cost_breakdown_table(data)
            }

            # Position tables
            tables['positions'] = {
                'current_positions': self._create_positions_table(data),
                'position_changes': self._create_position_changes_table(data),
                'exposure_analysis': self._create_exposure_analysis_table(data)
            }

            return tables

        except Exception as e:
            self.logger.error(f"Table generation failed: {str(e)}")
            raise

    def _calculate_performance_metrics(self) -> Dict:
        """Calculate performance metrics for report."""
        try:
            portfolio_state = self.portfolio_monitor.get_portfolio_state()
            trade_results = self.trade_analyzer.analysis_results

            metrics = {
                'total_return': self._calculate_total_return(portfolio_state),
                'annualized_return': self._calculate_annualized_return(portfolio_state),
                'sharpe_ratio': self._calculate_sharpe_ratio(portfolio_state),
                'sortino_ratio': self._calculate_sortino_ratio(portfolio_state),
                'max_drawdown': self._calculate_max_drawdown(portfolio_state),
                'win_rate': self._calculate_win_rate(trade_results),
                'profit_factor': self._calculate_profit_factor(trade_results),
                'avg_trade_return': self._calculate_avg_trade_return(trade_results)
            }

            return metrics

        except Exception as e:
            self.logger.error(
                f"Performance metric calculation failed: {str(e)}")
            raise
