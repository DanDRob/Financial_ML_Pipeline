"""
Model evaluation and validation framework.
"""

from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
import optuna
from optuna.integration import SkoptSampler
from scipy import stats
from models.model_wrapper import BaseModelWrapper
from models.model_factory import ModelFactory


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    metrics: List[str]
    cv_folds: int = 5
    min_train_size: int = 252  # One year of daily data
    validation_size: int = 63  # Three months
    step_size: int = 21  # One month
    optimization_trials: int = 100
    early_stopping_rounds: int = 10
    random_state: int = 42


class ModelEvaluator:
    """Comprehensive model evaluation and validation."""

    def __init__(self, config: EvaluationConfig):
        """
        Initialize model evaluator.

        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize metric functions
        self.metric_functions = {
            'accuracy': self._calculate_accuracy,
            'precision': self._calculate_precision,
            'recall': self._calculate_recall,
            'f1': self._calculate_f1,
            'roc_auc': self._calculate_roc_auc,
            'pr_auc': self._calculate_pr_auc,
            'sharpe_ratio': self._calculate_sharpe_ratio,
            'sortino_ratio': self._calculate_sortino_ratio,
            'max_drawdown': self._calculate_max_drawdown,
            'win_rate': self._calculate_win_rate,
            'profit_factor': self._calculate_profit_factor,
            'calmar_ratio': self._calculate_calmar_ratio
        }

    def evaluate_model(
        self,
        model: 'BaseModelWrapper',
        X: pd.DataFrame,
        y: pd.Series,
        eval_type: str = 'walk_forward'
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Evaluate model performance.

        Args:
            model: Model wrapper instance
            X: Feature DataFrame
            y: Target series
            eval_type: Evaluation type ('walk_forward', 'backtest', 'cv')

        Returns:
            Dictionary of evaluation metrics
        """
        try:
            if eval_type == 'walk_forward':
                return self._walk_forward_validation(model, X, y)
            elif eval_type == 'backtest':
                return self._backtest_validation(model, X, y)
            elif eval_type == 'cv':
                return self._cross_validation(model, X, y)
            else:
                raise ValueError(f"Unknown evaluation type: {eval_type}")

        except Exception as e:
            self.logger.error(f"Model evaluation failed: {str(e)}")
            raise

    def _walk_forward_validation(
        self,
        model: 'BaseModelWrapper',
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, List[float]]:
        """
        Perform walk-forward validation.

        Args:
            model: Model wrapper instance
            X: Feature DataFrame
            y: Target series

        Returns:
            Dictionary of evaluation metrics
        """
        try:
            results = {metric: [] for metric in self.config.metrics}

            # Generate walk-forward splits
            splits = self._generate_time_splits(
                len(X),
                self.config.min_train_size,
                self.config.validation_size,
                self.config.step_size
            )

            for train_idx, val_idx in splits:
                # Split data
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_val = X.iloc[val_idx]
                y_val = y.iloc[val_idx]

                # Train model
                model.fit(X_train, y_train)

                # Make predictions
                predictions = model.predict(X_val)
                probabilities = model.predict_proba(X_val)

                # Calculate metrics
                for metric in self.config.metrics:
                    metric_value = self.metric_functions[metric](
                        y_val,
                        predictions,
                        probabilities
                    )
                    results[metric].append(metric_value)

            # Calculate summary statistics
            summary = {}
            for metric, values in results.items():
                summary[f'{metric}_mean'] = np.mean(values)
                summary[f'{metric}_std'] = np.std(values)
                summary[f'{metric}_min'] = np.min(values)
                summary[f'{metric}_max'] = np.max(values)

            return summary

        except Exception as e:
            self.logger.error(f"Walk-forward validation failed: {str(e)}")
            raise

    def _backtest_validation(
        self,
        model: 'BaseModelWrapper',
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """
        Perform backtesting validation.

        Args:
            model: Model wrapper instance
            X: Feature DataFrame
            y: Target series

        Returns:
            Dictionary of backtest metrics
        """
        try:
            # Split data into training and testing
            train_size = int(len(X) * 0.8)
            X_train = X.iloc[:train_size]
            y_train = y.iloc[:train_size]
            X_test = X.iloc[train_size:]
            y_test = y.iloc[train_size:]

            # Train model
            model.fit(X_train, y_train)

            # Make predictions
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)

            # Calculate metrics
            results = {}
            for metric in self.config.metrics:
                results[metric] = self.metric_functions[metric](
                    y_test,
                    predictions,
                    probabilities
                )

            # Add trading metrics
            results.update(self._calculate_trading_metrics(
                y_test,
                predictions,
                probabilities
            ))

            return results

        except Exception as e:
            self.logger.error(f"Backtest validation failed: {str(e)}")
            raise

    def _cross_validation(
        self,
        model: 'BaseModelWrapper',
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, List[float]]:
        """
        Perform cross-validation.

        Args:
            model: Model wrapper instance
            X: Feature DataFrame
            y: Target series

        Returns:
            Dictionary of cross-validation metrics
        """
        try:
            from sklearn.model_selection import TimeSeriesSplit

            results = {metric: [] for metric in self.config.metrics}

            # Create time series cross-validation splits
            tscv = TimeSeriesSplit(
                n_splits=self.config.cv_folds,
                min_train_size=self.config.min_train_size
            )

            for train_idx, val_idx in tscv.split(X):
                # Split data
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_val = X.iloc[val_idx]
                y_val = y.iloc[val_idx]

                # Train model
                model.fit(X_train, y_train)

                # Make predictions
                predictions = model.predict(X_val)
                probabilities = model.predict_proba(X_val)

                # Calculate metrics
                for metric in self.config.metrics:
                    metric_value = self.metric_functions[metric](
                        y_val,
                        predictions,
                        probabilities
                    )
                    results[metric].append(metric_value)

            # Calculate summary statistics
            summary = {}
            for metric, values in results.items():
                summary[f'{metric}_mean'] = np.mean(values)
                summary[f'{metric}_std'] = np.std(values)
                summary[f'{metric}_min'] = np.min(values)
                summary[f'{metric}_max'] = np.max(values)

            return summary

        except Exception as e:
            self.logger.error(f"Cross-validation failed: {str(e)}")
            raise

    def optimize_hyperparameters(
        self,
        model_factory: 'ModelFactory',
        model_type: str,
        X: pd.DataFrame,
        y: pd.Series,
        param_space: Dict
    ) -> Tuple[Dict, float]:
        """
        Optimize model hyperparameters.

        Args:
            model_factory: ModelFactory instance
            model_type: Type of model to optimize
            X: Feature DataFrame
            y: Target series
            param_space: Hyperparameter search space

        Returns:
            Tuple of (best parameters, best score)
        """
        try:
            # Create study
            study = optuna.create_study(
                direction='maximize',
                sampler=SkoptSampler()
            )

            # Define objective function
            def objective(trial):
                # Generate parameters
                params = {}
                for param_name, param_config in param_space.items():
                    if param_config['type'] == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name,
                            param_config['low'],
                            param_config['high']
                        )
                    elif param_config['type'] == 'float':
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config['low'],
                            param_config['high'],
                            log=param_config.get('log', False)
                        )
                    elif param_config['type'] == 'categorical':
                        params[param_name] = trial.suggest_categorical(
                            param_name,
                            param_config['choices']
                        )

                # Create and evaluate model
                model = model_factory.create_model(
                    model_type,
                    {'model_params': params}
                )

                metrics = self._cross_validation(model, X, y)
                return metrics.get('sharpe_ratio_mean', 0.0)

            # Optimize
            study.optimize(
                objective,
                n_trials=self.config.optimization_trials,
                n_jobs=-1
            )

            return study.best_params, study.best_value

        except Exception as e:
            self.logger.error(f"Hyperparameter optimization failed: {str(e)}")
            raise

    @staticmethod
    def _generate_time_splits(
        n_samples: int,
        min_train_size: int,
        validation_size: int,
        step_size: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate time series split indices."""
        splits = []
        start = 0
        end = min_train_size

        while end + validation_size <= n_samples:
            train_indices = np.arange(start, end)
            val_indices = np.arange(end, end + validation_size)
            splits.append((train_indices, val_indices))

            start += step_size
            end += step_size

        return splits

    # Metric calculation methods
    def _calculate_accuracy(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> float:
        """Calculate prediction accuracy."""
        return accuracy_score(y_true, y_pred)

    def _calculate_precision(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> float:
        """Calculate prediction precision."""
        return precision_score(y_true, y_pred)

    def _calculate_recall(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> float:
        """Calculate prediction recall."""
        return recall_score(y_true, y_pred)

    def _calculate_f1(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> float:
        """Calculate F1 score."""
        return f1_score(y_true, y_pred)

    def _calculate_roc_auc(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> float:
        """Calculate ROC AUC score."""
        return roc_auc_score(y_true, y_prob[:, 1])

    def _calculate_pr_auc(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> float:
        """Calculate PR AUC score."""
        return average_precision_score(y_true, y_prob[:, 1])

    def _calculate_sharpe_ratio(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        risk_free_rate: float = 0.02
    ) -> float:
        """Calculate annualized Sharpe ratio."""
        returns = y_true * y_pred
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def _calculate_sortino_ratio(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sortino ratio."""
        returns = y_true * y_pred
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()

    def _calculate_max_drawdown(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> float:
        """Calculate maximum drawdown."""
        returns = y_true * y_pred
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = cumulative / running_max - 1
        return float(drawdowns.min())

    def _calculate_win_rate(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> float:
        """Calculate win rate."""
        returns = y_true * y_pred
        return float(np.mean(returns > 0))

    def _calculate_profit_factor(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> float:
        """Calculate profit factor."""
        returns = y_true * y_pred
        positive_returns = returns[returns > 0].sum()
        negative_returns = abs(returns[returns < 0].sum())

        if negative_returns == 0:
            return float('inf')
        return float(positive_returns / negative_returns)

    def _calculate_calmar_ratio(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> float:
        """Calculate Calmar ratio."""
        returns = y_true * y_pred
        annual_return = returns.mean() * 252
        max_drawdown = self._calculate_max_drawdown(y_true, y_pred)

        if max_drawdown == 0:
            return float('inf')
        return float(annual_return / abs(max_drawdown))

    def _calculate_trading_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate comprehensive trading metrics."""
        try:
            returns = y_true * y_pred
            cumulative_returns = (1 + returns).cumprod()

            metrics = {
                'total_return': float(cumulative_returns.iloc[-1] - 1),
                'annual_return': float(returns.mean() * 252),
                'annual_volatility': float(returns.std() * np.sqrt(252)),
                'sharpe_ratio': self._calculate_sharpe_ratio(y_true, y_pred),
                'sortino_ratio': self._calculate_sortino_ratio(y_true, y_pred),
                'max_drawdown': self._calculate_max_drawdown(y_true, y_pred),
                'win_rate': self._calculate_win_rate(y_true, y_pred),
                'profit_factor': self._calculate_profit_factor(y_true, y_pred),
                'calmar_ratio': self._calculate_calmar_ratio(y_true, y_pred),

                # Additional metrics
                'avg_win': float(returns[returns > 0].mean()),
                'avg_loss': float(returns[returns < 0].mean()),
                'max_win': float(returns.max()),
                'max_loss': float(returns.min()),
                'skewness': float(stats.skew(returns)),
                'kurtosis': float(stats.kurtosis(returns)),

                # Risk metrics
                'value_at_risk_95': float(np.percentile(returns, 5)),
                'expected_shortfall_95': float(returns[returns <= np.percentile(returns, 5)].mean()),
                'downside_deviation': float(returns[returns < 0].std()),
                'omega_ratio': float((returns[returns > 0].mean() * len(returns[returns > 0])) /
                                     (abs(returns[returns < 0].mean()) * len(returns[returns < 0])))
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Trading metrics calculation failed: {str(e)}")
            raise


class ModelValidator:
    """Model validation and robustness testing."""

    def __init__(self, config: EvaluationConfig):
        """
        Initialize model validator.

        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def validate_model_robustness(
        self,
        model: 'BaseModelWrapper',
        X: pd.DataFrame,
        y: pd.Series,
        n_simulations: int = 1000
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform comprehensive model robustness testing.

        Args:
            model: Model wrapper instance
            X: Feature DataFrame
            y: Target series
            n_simulations: Number of Monte Carlo simulations

        Returns:
            Dictionary of robustness metrics
        """
        try:
            results = {}

            # Monte Carlo simulation
            results['monte_carlo'] = self._perform_monte_carlo(
                model, X, y, n_simulations
            )

            # Sensitivity analysis
            results['sensitivity'] = self._perform_sensitivity_analysis(
                model, X, y
            )

            # Feature importance stability
            results['feature_stability'] = self._analyze_feature_stability(
                model, X, y
            )

            # Out-of-sample testing
            results['out_of_sample'] = self._perform_oos_testing(
                model, X, y
            )

            return results

        except Exception as e:
            self.logger.error(f"Model robustness testing failed: {str(e)}")
            raise

    def _perform_monte_carlo(
        self,
        model: 'BaseModelWrapper',
        X: pd.DataFrame,
        y: pd.Series,
        n_simulations: int
    ) -> Dict[str, float]:
        """Perform Monte Carlo simulation testing."""
        try:
            base_predictions = model.predict(X)
            base_metrics = ModelEvaluator(self.config)._calculate_trading_metrics(
                y, base_predictions
            )

            simulated_metrics = []
            for _ in range(n_simulations):
                # Generate bootstrap sample
                bootstrap_idx = np.random.choice(
                    len(X),
                    size=len(X),
                    replace=True
                )
                X_boot = X.iloc[bootstrap_idx]
                y_boot = y.iloc[bootstrap_idx]

                # Train and evaluate model
                model.fit(X_boot, y_boot)
                predictions = model.predict(X)
                metrics = ModelEvaluator(self.config)._calculate_trading_metrics(
                    y, predictions
                )
                simulated_metrics.append(metrics)

            # Calculate p-values and confidence intervals
            results = {}
            for metric in base_metrics:
                simulated_values = [m[metric] for m in simulated_metrics]
                results[f'{metric}_pvalue'] = np.mean(
                    np.array(simulated_values) >= base_metrics[metric]
                )
                results[f'{metric}_ci_lower'] = np.percentile(
                    simulated_values, 2.5)
                results[f'{metric}_ci_upper'] = np.percentile(
                    simulated_values, 97.5)

            return results

        except Exception as e:
            self.logger.error(f"Monte Carlo simulation failed: {str(e)}")
            raise

    def _perform_sensitivity_analysis(
        self,
        model: 'BaseModelWrapper',
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """Perform sensitivity analysis on features."""
        try:
            base_predictions = model.predict(X)
            base_metrics = ModelEvaluator(self.config)._calculate_trading_metrics(
                y, base_predictions
            )

            sensitivity = {}
            for column in X.columns:
                # Perturb feature
                X_perturbed = X.copy()
                X_perturbed[column] = X_perturbed[column] * np.random.normal(
                    1, 0.1, size=len(X)
                )

                # Get predictions and metrics
                predictions = model.predict(X_perturbed)
                metrics = ModelEvaluator(self.config)._calculate_trading_metrics(
                    y, predictions
                )

                # Calculate sensitivity
                sensitivity[column] = {
                    metric: abs(metrics[metric] - base_metrics[metric])
                    for metric in base_metrics
                }

            return sensitivity

        except Exception as e:
            self.logger.error(f"Sensitivity analysis failed: {str(e)}")
            raise

    def _analyze_feature_stability(
        self,
        model: 'BaseModelWrapper',
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """Analyze feature importance stability."""
        try:
            base_importance = model.get_feature_importance()
            if base_importance is None:
                return {}

            importance_samples = []
            for _ in range(100):  # Number of bootstrap samples
                # Generate bootstrap sample
                bootstrap_idx = np.random.choice(
                    len(X),
                    size=len(X),
                    replace=True
                )
                X_boot = X.iloc[bootstrap_idx]
                y_boot = y.iloc[bootstrap_idx]

                # Train model and get feature importance
                model.fit(X_boot, y_boot)
                importance = model.get_feature_importance()
                if importance is not None:
                    importance_samples.append(importance)

            # Calculate stability metrics
            stability = {}
            for feature in base_importance:
                values = [sample[feature] for sample in importance_samples]
                stability[feature] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'cv': float(np.std(values) / np.mean(values))
                }

            return stability

        except Exception as e:
            self.logger.error(f"Feature stability analysis failed: {str(e)}")
            raise

    def _perform_oos_testing(
        self,
        model: 'BaseModelWrapper',
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """Perform out-of-sample testing."""
        try:
            # Split data into in-sample and out-of-sample
            train_size = int(len(X) * 0.8)
            X_train = X.iloc[:train_size]
            y_train = y.iloc[:train_size]
            X_test = X.iloc[train_size:]
            y_test = y.iloc[train_size:]

            # Train model on in-sample data
            model.fit(X_train, y_train)

            # Evaluate on both sets
            in_sample_pred = model.predict(X_train)
            out_sample_pred = model.predict(X_test)

            in_sample_metrics = ModelEvaluator(self.config)._calculate_trading_metrics(
                y_train, in_sample_pred
            )
            out_sample_metrics = ModelEvaluator(self.config)._calculate_trading_metrics(
                y_test, out_sample_pred
            )

            # Calculate degradation metrics
            results = {}
            for metric in in_sample_metrics:
                results[f'{metric}_degradation'] = (
                    out_sample_metrics[metric] - in_sample_metrics[metric]
                ) / abs(in_sample_metrics[metric])

            return results

        except Exception as e:
            self.logger.error(f"Out-of-sample testing failed: {str(e)}")
            raise
