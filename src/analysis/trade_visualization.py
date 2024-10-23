from dash.dependencies import Input, Output
from dash import dcc, html
import dash
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
from datetime import datetime
from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Tuple, Union
from src.analysis.trade_analysis import TradeAnalyzer

"""
Trade visualization and reporting components.
"""


@dataclass
class VisualizationConfig:
    """Configuration for trade visualization."""
    # Plot settings
    theme: str = 'plotly_white'
    color_palette: List[str] = None
    plot_height: int = 600
    plot_width: int = 800

    # Interactive features
    enable_zoom: bool = True
    enable_hover: bool = True
    enable_selection: bool = True

    # Export settings
    export_format: str = 'html'
    include_plotlyjs: bool = True

    # Dashboard settings
    update_interval: int = 60  # seconds
    max_points: int = 10000


class TradeVisualization:
    """Trade visualization and interactive dashboard."""

    def __init__(
        self,
        config: VisualizationConfig,
        trade_analyzer: 'TradeAnalyzer'
    ):
        """
        Initialize trade visualization.

        Args:
            config: Visualization configuration
            trade_analyzer: Trade analyzer instance
        """
        self.config = config
        self.trade_analyzer = trade_analyzer
        self.logger = logging.getLogger(__name__)

        # Set theme
        self.template = self.config.theme
        self.colors = self.config.color_palette or px.colors.qualitative.Set3

        self.figures = {}
        self.app = None

    def create_execution_dashboard(self) -> dash.Dash:
        """Create interactive execution analysis dashboard."""
        try:
            app = dash.Dash(__name__)

            app.layout = html.Div([
                # Header
                html.H1('Trade Execution Analysis Dashboard'),

                # Date range selector
                dcc.DatePickerRange(
                    id='date-range',
                    start_date=datetime.now().date() - pd.Timedelta(days=30),
                    end_date=datetime.now().date()
                ),

                # Metric selectors
                dcc.Dropdown(
                    id='metric-selector',
                    options=[
                        {'label': 'Implementation Shortfall', 'value': 'is'},
                        {'label': 'Market Impact', 'value': 'mi'},
                        {'label': 'VWAP Slippage', 'value': 'vwap'},
                        {'label': 'Fill Rate', 'value': 'fill'}
                    ],
                    value='is',
                    multi=False
                ),

                # Main charts
                html.Div([
                    dcc.Graph(id='execution-quality-chart'),
                    dcc.Graph(id='cost-analysis-chart'),
                    dcc.Graph(id='timing-analysis-chart')
                ], style={'display': 'flex', 'flexWrap': 'wrap'}),

                # Detail panels
                html.Div([
                    html.Div([
                        html.H3('Trade Details'),
                        dcc.Graph(id='trade-details-table')
                    ], className='six columns'),

                    html.Div([
                        html.H3('Venue Analysis'),
                        dcc.Graph(id='venue-analysis-chart')
                    ], className='six columns')
                ], className='row'),

                # Update interval
                dcc.Interval(
                    id='interval-component',
                    interval=self.config.update_interval * 1000,
                    n_intervals=0
                )
            ])

            self._setup_callbacks(app)
            self.app = app
            return app

        except Exception as e:
            self.logger.error(f"Dashboard creation failed: {str(e)}")
            raise

    def _setup_callbacks(self, app: dash.Dash) -> None:
        """Setup dashboard callbacks."""

        @app.callback(
            [Output('execution-quality-chart', 'figure'),
             Output('cost-analysis-chart', 'figure'),
             Output('timing-analysis-chart', 'figure'),
             Output('trade-details-table', 'figure'),
             Output('venue-analysis-chart', 'figure')],
            [Input('date-range', 'start_date'),
             Input('date-range', 'end_date'),
             Input('metric-selector', 'value'),
             Input('interval-component', 'n_intervals')]
        )
        def update_charts(start_date, end_date, metric, n_intervals):
            try:
                # Get updated analysis results
                results = self.trade_analyzer.analyze_trades(
                    start_date=pd.Timestamp(start_date),
                    end_date=pd.Timestamp(end_date)
                )

                # Generate charts
                execution_fig = self._create_execution_quality_chart(
                    results['execution_quality'],
                    metric
                )

                cost_fig = self._create_cost_analysis_chart(
                    results['cost_analysis']
                )

                timing_fig = self._create_timing_analysis_chart(
                    results['timing_analysis']
                )

                details_fig = self._create_trade_details_table(
                    results
                )

                venue_fig = self._create_venue_analysis_chart(
                    results['venue_analysis']
                )

                return execution_fig, cost_fig, timing_fig, details_fig, venue_fig

            except Exception as e:
                self.logger.error(f"Chart update failed: {str(e)}")
                raise

    def _create_execution_quality_chart(
        self,
        results: Dict,
        metric: str
    ) -> go.Figure:
        """Create execution quality visualization."""
        try:
            if metric == 'is':
                # Implementation shortfall distribution
                fig = go.Figure()

                fig.add_trace(go.Histogram(
                    x=results['implementation_shortfall'],
                    nbinsx=50,
                    name='Implementation Shortfall'
                ))

                fig.update_layout(
                    title='Implementation Shortfall Distribution',
                    xaxis_title='Shortfall (%)',
                    yaxis_title='Frequency',
                    template=self.template,
                    height=self.config.plot_height,
                    width=self.config.plot_width
                )

            elif metric == 'mi':
                # Market impact by trade size
                fig = px.scatter(
                    results['market_impact']['by_size'],
                    x='trade_size',
                    y='impact',
                    color='venue',
                    title='Market Impact vs Trade Size',
                    template=self.template,
                    height=self.config.plot_height,
                    width=self.config.plot_width
                )

            elif metric == 'vwap':
                # VWAP slippage over time
                fig = px.line(
                    results['vwap_slippage']['by_time'],
                    x='time',
                    y='slippage',
                    color='venue',
                    title='VWAP Slippage Over Time',
                    template=self.template,
                    height=self.config.plot_height,
                    width=self.config.plot_width
                )

            else:  # fill rate
                # Fill rates by venue
                fig = px.bar(
                    results['fill_rates'],
                    x='venue',
                    y='fill_rate',
                    title='Fill Rates by Venue',
                    template=self.template,
                    height=self.config.plot_height,
                    width=self.config.plot_width
                )

            return fig

        except Exception as e:
            self.logger.error(
                f"Execution quality chart creation failed: {str(e)}")
            raise

    def _create_cost_analysis_chart(self, results: Dict) -> go.Figure:
        """Create cost analysis visualization."""
        try:
            # Create subplots
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    'Cost Components',
                    'Cost by Venue',
                    'Cost by Size',
                    'Cost Over Time'
                )
            )

            # Cost components
            components = ['explicit', 'implicit', 'opportunity']
            values = [
                results[f'{c}_costs']['total']
                for c in components
            ]

            fig.add_trace(
                go.Pie(
                    labels=components,
                    values=values,
                    name='Cost Components'
                ),
                row=1,
                col=1
            )

            # Cost by venue
            venue_costs = results['cost_analysis']['by_venue']
            fig.add_trace(
                go.Bar(
                    x=venue_costs.index,
                    y=venue_costs.values,
                    name='Venue Costs'
                ),
                row=1,
                col=2
            )

            # Cost by size
            size_costs = results['cost_analysis']['by_size']
            fig.add_trace(
                go.Scatter(
                    x=size_costs['size'],
                    y=size_costs['cost'],
                    mode='markers',
                    name='Size Costs'
                ),
                row=2,
                col=1
            )

            # Cost over time
            time_costs = results['cost_analysis']['by_time']
            fig.add_trace(
                go.Line(
                    x=time_costs.index,
                    y=time_costs.values,
                    name='Time Costs'
                ),
                row=2,
                col=2
            )

            fig.update_layout(
                height=self.config.plot_height * 2,
                width=self.config.plot_width * 2,
                template=self.template,
                showlegend=True
            )

            return fig

        except Exception as e:
            self.logger.error(f"Cost analysis chart creation failed: {str(e)}")
            raise

    def _create_timing_analysis_chart(self, results: Dict) -> go.Figure:
        """Create timing analysis visualization."""
        try:
            # Create subplots
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    'Intraday Pattern',
                    'Seasonality',
                    'Volume Profile',
                    'Market Impact'
                )
            )

            # Intraday pattern
            intraday = results['intraday_patterns']
            fig.add_trace(
                go.Line(
                    x=intraday['time'],
                    y=intraday['volume'],
                    name='Volume'
                ),
                row=1,
                col=1
            )

            # Seasonality
            seasonality = results['seasonality']
            fig.add_trace(
                go.Heatmap(
                    z=seasonality['values'],
                    x=seasonality['months'],
                    y=seasonality['days'],
                    colorscale='Blues'
                ),
                row=1,
                col=2
            )

            # Volume profile
            volume = results['volume_profile']
            fig.add_trace(
                go.Bar(
                    x=volume['price'],
                    y=volume['volume'],
                    name='Volume Profile'
                ),
                row=2,
                col=1
            )

            # Market impact
            impact = results['market_conditions']
            fig.add_trace(
                go.Scatter(
                    x=impact['volatility'],
                    y=impact['impact'],
                    mode='markers',
                    marker=dict(
                        size=impact['volume'],
                        color=impact['spread'],
                        colorscale='Viridis',
                        showscale=True
                    ),
                    name='Market Impact'
                ),
                row=2,
                col=2
            )

            fig.update_layout(
                height=self.config.plot_height * 2,
                width=self.config.plot_width * 2,
                template=self.template,
                showlegend=True
            )

            return fig

        except Exception as e:
            self.logger.error(
                f"Timing analysis chart creation failed: {str(e)}")
            raise
