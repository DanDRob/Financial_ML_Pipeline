"""
Feature selection module for the ML pipeline.
"""

from typing import Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest, SelectFromModel, RFE, RFECV,
    mutual_info_classif, f_classif, mutual_info_regression
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV
import xgboost as xgb
from scipy import stats


class FeatureSelector:
    """Feature selection and dimensionality reduction class."""

    def __init__(
        self,
        selection_method: str = 'combined',
        n_features: Optional[int] = None,
        random_state: int = 42
    ):
        """
        Initialize FeatureSelector.

        Args:
            selection_method: Method for feature selection
            n_features: Number of features to select
            random_state: Random state for reproducibility
        """
        self.selection_method = selection_method
        self.n_features = n_features
        self.random_state = random_state
        self.selected_features = None
        self.feature_importance = None
        self.logger = logging.getLogger(__name__)

    def fit_select(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Fit feature selector and select features.

        Args:
            X: Feature DataFrame
            y: Target series
            method: Optional override for selection method

        Returns:
            Tuple of (selected features DataFrame, feature importance dict)
        """
        try:
            method = method or self.selection_method

            if method == 'combined':
                return self._combined_selection(X, y)
            elif method == 'statistical':
                return self._statistical_selection(X, y)
            elif method == 'model_based':
                return self._model_based_selection(X, y)
            elif method == 'correlation':
                return self._correlation_based_selection(X, y)
            else:
                raise ValueError(f"Unknown selection method: {method}")

        except Exception as e:
            self.logger.error(f"Feature selection failed: {str(e)}")
            raise

    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select features from DataFrame.

        Args:
            df: Input DataFrame with features

        Returns:
            DataFrame with selected features
        """
        try:
            # Remove any non-numeric columns
            numeric_df = df.select_dtypes(include=[np.number])

            # Handle missing values for feature selection
            numeric_df = numeric_df.ffill().bfill()

            # Extract target variable (assuming it's in the DataFrame)
            y = df['target']  # Adjust column name if different

            # Select features based on method
            if self.selection_method == 'combined':
                selected_features = self._combined_selection(numeric_df, y)
            elif self.selection_method == 'mutual_info':
                selected_features = self._mutual_info_selection(numeric_df, y)
            elif self.selection_method == 'f_score':
                selected_features = self._f_score_selection(numeric_df, y)
            elif self.selection_method == 'random_forest':
                selected_features = self._random_forest_selection(
                    numeric_df, y)
            else:
                self.logger.warning(f"Unknown selection method: {
                                    self.selection_method}")
                return df

            # Store selected features
            self.selected_features = selected_features

            # Return DataFrame with selected features
            return df[selected_features]

        except Exception as e:
            self.logger.error(f"Feature selection failed: {str(e)}")
            raise

    def _mutual_info_selection(self, df: pd.DataFrame) -> List[str]:
        """Select features using mutual information."""
        try:
            # Use returns as target
            y = df['Close'].pct_change().fillna(0)
            X = df.drop('Close', axis=1)

            # Apply mutual information selection
            selector = SelectKBest(
                score_func=mutual_info_regression, k=self.n_features)
            selector.fit(X, y)

            # Get selected feature names
            mask = selector.get_support()
            return list(X.columns[mask])

        except Exception as e:
            self.logger.error(f"Mutual information selection failed: {str(e)}")
            raise

    def _f_score_selection(self, df: pd.DataFrame) -> List[str]:
        """Select features using F-score."""
        try:
            # Create binary target (up/down movement)
            y = (df['Close'].pct_change() > 0).astype(int).fillna(0)
            X = df.drop('Close', axis=1)

            # Apply F-score selection
            selector = SelectKBest(score_func=f_classif, k=self.n_features)
            selector.fit(X, y)

            # Get selected feature names
            mask = selector.get_support()
            return list(X.columns[mask])

        except Exception as e:
            self.logger.error(f"F-score selection failed: {str(e)}")
            raise

    def _random_forest_selection(self, df: pd.DataFrame) -> List[str]:
        """Select features using Random Forest importance."""
        try:
            # Use returns as target
            y = df['Close'].pct_change().fillna(0)
            X = df.drop('Close', axis=1)

            # Train Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)

            # Get feature importance rankings
            importance = pd.Series(rf.feature_importances_, index=X.columns)
            return list(importance.nlargest(self.n_features).index)

        except Exception as e:
            self.logger.error(f"Random Forest selection failed: {str(e)}")
            raise

    def _combined_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Combine multiple feature selection methods.

        Args:
            X: Feature DataFrame
            y: Target series

        Returns:
            Tuple of (selected features DataFrame, feature importance dict)
        """
        try:
            # Get features from each method
            statistical_features = self._statistical_selection(X, y)[0].columns
            model_features = self._model_based_selection(X, y)[0].columns
            correlation_features = self._correlation_based_selection(X, y)[
                0].columns

            # Combine feature sets
            feature_counts = pd.Series(
                list(statistical_features) +
                list(model_features) +
                list(correlation_features)
            ).value_counts()

            # Select features that appear in at least 2 methods
            selected_features = feature_counts[feature_counts >= 2].index

            if self.n_features:
                selected_features = selected_features[:self.n_features]

            # Calculate combined importance scores
            importance_dict = {}
            for feat in selected_features:
                # Average normalized importance from each method
                statistical_imp = self._get_statistical_importance(X[feat], y)
                model_imp = self._get_model_importance(X[feat], y)
                correlation_imp = self._get_correlation_importance(X[feat], y)

                importance_dict[feat] = np.mean([
                    statistical_imp,
                    model_imp,
                    correlation_imp
                ])

            self.selected_features = selected_features
            self.feature_importance = importance_dict

            return X[selected_features], importance_dict

        except Exception as e:
            self.logger.error(f"Combined selection failed: {str(e)}")
            raise

    def _statistical_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Perform statistical feature selection.

        Args:
            X: Feature DataFrame
            y: Target series

        Returns:
            Tuple of (selected features DataFrame, feature importance dict)
        """
        try:
            # Calculate various statistical metrics
            importance_dict = {}

            for column in X.columns:
                # Mutual information
                mi_score = mutual_info_classif(
                    X[[column]],
                    y,
                    random_state=self.random_state
                )[0]

                # F-score
                f_score = f_classif(X[[column]], y)[0][0]

                # Chi-square test for categorical features
                if X[column].dtype in ['object', 'category']:
                    chi2_score = stats.chi2_contingency(
                        pd.crosstab(X[column], y)
                    )[0]
                else:
                    chi2_score = 0

                # Combine scores
                importance_dict[column] = np.mean([
                    mi_score,
                    f_score,
                    chi2_score
                ])

            # Select top features
            selected_features = sorted(
                importance_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )

            if self.n_features:
                selected_features = selected_features[:self.n_features]

            selected_columns = [feat[0] for feat in selected_features]
            importance_dict = {feat[0]: feat[1] for feat in selected_features}

            self.selected_features = selected_columns
            self.feature_importance = importance_dict

            return X[selected_columns], importance_dict

        except Exception as e:
            self.logger.error(f"Statistical selection failed: {str(e)}")
            raise

    def _model_based_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Perform model-based feature selection.

        Args:
            X: Feature DataFrame
            y: Target series

        Returns:
            Tuple of (selected features DataFrame, feature importance dict)
        """
        try:
            # Random Forest based selection
            rf = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state
            )

            # Recursive feature elimination with cross-validation
            rfecv = RFECV(
                estimator=rf,
                step=1,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )

            rfecv.fit(X, y)
            rf_importance = dict(zip(X.columns, rfecv.support_))

            # Lasso based selection
            lasso = LassoCV(
                cv=5,
                random_state=self.random_state
            )

            lasso.fit(X, y)
            lasso_importance = dict(zip(X.columns, np.abs(lasso.coef_)))

            # XGBoost based selection
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                random_state=self.random_state
            )

            xgb_model.fit(X, y)
            xgb_importance = dict(
                zip(X.columns, xgb_model.feature_importances_))

            # Combine importance scores
            importance_dict = {}
            for column in X.columns:
                importance_dict[column] = np.mean([
                    rf_importance[column],
                    lasso_importance[column],
                    xgb_importance[column]
                ])

            # Select top features
            selected_features = sorted(
                importance_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )

            if self.n_features:
                selected_features = selected_features[:self.n_features]

            selected_columns = [feat[0] for feat in selected_features]
            importance_dict = {feat[0]: feat[1] for feat in selected_features}

            self.selected_features = selected_columns
            self.feature_importance = importance_dict

            return X[selected_columns], importance_dict

        except Exception as e:
            self.logger.error(f"Model-based selection failed: {str(e)}")
            raise

    def _correlation_based_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = 0.8
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Perform correlation-based feature selection.

        Args:
            X: Feature DataFrame
            y: Target series
            threshold: Correlation threshold

        Returns:
            Tuple of (selected features DataFrame, feature importance dict)
        """
        try:
            # Calculate correlation with target
            target_corr = X.apply(lambda x: abs(x.corr(y)))

            # Calculate feature correlations
            corr_matrix = X.corr().abs()

            # Remove highly correlated features
            features_to_remove = set()
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if corr_matrix.iloc[i, j] > threshold:
                        col_i = corr_matrix.columns[i]
                        col_j = corr_matrix.columns[j]

                        if target_corr[col_i] > target_corr[col_j]:
                            features_to_remove.add(col_j)
                        else:
                            features_to_remove.add(col_i)

            selected_columns = [
                col for col in X.columns
                if col not in features_to_remove
            ]

            # Calculate importance scores
            importance_dict = {}
            for col in selected_columns:
                # Combine correlation with target and uniqueness
                target_correlation = abs(X[col].corr(y))
                uniqueness = 1 - corr_matrix[col].mean()

                importance_dict[col] = np.mean([
                    target_correlation,
                    uniqueness
                ])

            if self.n_features:
                selected_features = sorted(
                    importance_dict.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:self.n_features]

                selected_columns = [feat[0] for feat in selected_features]
                importance_dict = {feat[0]: feat[1]
                                   for feat in selected_features}

            self.selected_features = selected_columns
            self.feature_importance = importance_dict

            return X[selected_columns], importance_dict

        except Exception as e:
            self.logger.error(f"Correlation-based selection failed: {str(e)}")
            raise

    def _get_statistical_importance(
        self,
        X: pd.Series,
        y: pd.Series
    ) -> float:
        """Calculate normalized statistical importance score."""
        try:
            mi_score = mutual_info_classif(
                X.values.reshape(-1, 1),
                y,
                random_state=self.random_state
            )[0]
            f_score = f_classif(X.values.reshape(-1, 1), y)[0][0]

            # Normalize and combine scores
            return np.mean([
                mi_score / (mi_score + 1),
                f_score / (f_score + 1)
            ])

        except Exception:
            return 0.0

    def _get_model_importance(
        self,
        X: pd.Series,
        y: pd.Series
    ) -> float:
        """Calculate normalized model-based importance score."""
        try:
            rf = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state
            )
            rf.fit(X.values.reshape(-1, 1), y)

            return rf.feature_importances_[0]

        except Exception:
            return 0.0

    def _get_correlation_importance(
        self,
        X: pd.Series,
        y: pd.Series
    ) -> float:
        """Calculate normalized correlation-based importance score."""
        try:
            return abs(X.corr(y))

        except Exception:
            return 0.0
