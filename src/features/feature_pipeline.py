"""
Feature engineering pipeline module.
"""

from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import dump, load
import json
import hashlib

from src.features.technical_indicators import TechnicalIndicators
from src.features.custom_indicators import CustomFeatureCreator
from src.features.feature_selector import FeatureSelector


@dataclass
class FeaturePipelineConfig:
    """Configuration for feature engineering pipeline."""
    feature_groups: List[str]
    technical_indicators: List[str]
    custom_features: List[str]
    selection_method: str
    n_features: Optional[int]
    use_cache: bool = True
    cache_dir: Optional[str] = None
    random_state: int = 42


class FeatureTransformer:
    """Feature transformation and engineering class."""

    def __init__(self, config: Dict):
        """
        Initialize feature transformer.

        Args:
            config: Feature configuration dictionary
        """
        self.config = config
        self.feature_selector = FeatureSelector(
            selection_method=config.get('selection_method', 'combined'),
            n_features=config.get('n_features', 20)
        )
        self.technical_indicators = TechnicalIndicators()
        self.custom_feature_creator = CustomFeatureCreator()
        self.scaler = None
        self.selected_features = None

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> 'FeatureTransformer':
        """
        Fit the feature transformer.

        Args:
            X: Input DataFrame
            y: Optional target series

        Returns:
            Fitted transformer instance
        """
        try:
            # Create features
            features_df = self._create_features(X)

            # Perform feature selection if target is provided
            if y is not None:
                features_df, importance_dict = self.feature_selector.fit_select(
                    features_df,
                    y
                )
                self.selected_features = features_df.columns.tolist()

            return self

        except Exception as e:
            logging.error(f"Feature transformer fitting failed: {str(e)}")
            raise

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform input data.

        Args:
            X: Input DataFrame

        Returns:
            Transformed DataFrame
        """
        try:
            # Create features
            features_df = self._create_features(X)

            # Select features if available
            if self.selected_features is not None:
                features_df = features_df[self.selected_features]

            return features_df

        except Exception as e:
            logging.error(f"Feature transformation failed: {str(e)}")
            raise

    def _create_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features based on configuration.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with created features
        """
        df = X.copy()

        # Add technical indicators
        if 'technical' in self.config.feature_groups:
            df = self.technical_indicators.calculate_all_indicators(
                df,
                include_groups=self.config.technical_indicators
            )

        # Add custom features
        if 'custom' in self.config.feature_groups:
            df = self.feature_creator.create_all_features(
                df,
                include_groups=self.config.custom_features
            )

        return df


class FeaturePipeline:
    """Feature engineering pipeline."""

    def __init__(self, config: Dict):
        """
        Initialize feature pipeline.

        Args:
            config: Feature configuration dictionary
        """
        self.config = config
        self.transformer = FeatureTransformer(config)
        self.feature_selector = None
        self.logger = logging.getLogger(__name__)

        # Set up cache directory
        self.cache_dir = Path(config.get('cache_dir', 'data/cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache configuration
        self.use_cache = config.get('use_cache', False)

    def fit_transform(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Fit pipeline and transform data.

        Args:
            data: Dictionary of DataFrames keyed by symbol

        Returns:
            Dictionary of transformed DataFrames with features
        """
        try:
            if self.use_cache:
                cache_key = self._generate_cache_key(data)
                cached_features = self._load_from_cache(cache_key)
                if cached_features is not None:
                    return cached_features

            transformed_data = {}
            for symbol, df in data.items():
                transformed_df = self._fit_transform_single(df)
                transformed_data[symbol] = transformed_df

            if self.use_cache:
                self._save_to_cache(cache_key, transformed_data)

            return transformed_data

        except Exception as e:
            self.logger.error(f"Pipeline fit_transform failed: {str(e)}")
            raise

    def _fit_transform_single(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform a single DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Transformed DataFrame with features
        """
        try:
            # Generate technical indicators
            df = self.transformer.technical_indicators.generate_indicators(
                df,
                self.config.get('technical_indicators', [])
            )

            # Generate custom features
            df = self.transformer.custom_feature_creator.create_all_features(
                df,
                self.config.get('custom_features', [])
            )

            # Select features if configured
            if self.config.get('selection_method'):
                df = self.transformer.feature_selector.select_features(df)

            return df

        except Exception as e:
            self.logger.error(f"Single DataFrame transform failed: {str(e)}")
            raise

    def _generate_cache_key(self, data: Dict[str, pd.DataFrame]) -> str:
        """
        Generate cache key for data dictionary.

        Args:
            data: Dictionary of DataFrames

        Returns:
            Cache key string
        """
        try:
            # Create a dictionary of DataFrame hashes
            data_hashes = {
                symbol: str(pd.util.hash_pandas_object(df).sum())
                for symbol, df in data.items()
            }

            # Combine with config hash
            config_str = json.dumps(self.config, sort_keys=True)
            combined_str = f"{config_str}_{
                json.dumps(data_hashes, sort_keys=True)}"

            return hashlib.md5(combined_str.encode()).hexdigest()

        except Exception as e:
            self.logger.error(f"Cache key generation failed: {str(e)}")
            raise

    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Load features from cache if available."""
        try:
            cache_file = self.cache_dir / f"features_{cache_key}.joblib"

            if cache_file.exists():
                return load(cache_file)
            return None

        except Exception as e:
            self.logger.warning(f"Cache load failed: {str(e)}")
            return None

    def _save_to_cache(
        self,
        cache_key: str,
        features: Dict[str, pd.DataFrame]
    ) -> None:
        """Save features to cache."""
        try:
            cache_dir = Path(self.config.get('cache_dir', 'cache'))
            cache_dir.mkdir(parents=True, exist_ok=True)

            cache_file = cache_dir / f"features_{cache_key}.joblib"
            dump(features, cache_file)

        except Exception as e:
            self.logger.warning(f"Cache save failed: {str(e)}")

        return None

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available."""
        return self.feature_selector.feature_importance

    def get_selected_features(self) -> Optional[List[str]]:
        """Get selected features if available."""
        return self.feature_selector.selected_features

    def save_pipeline(self, filepath: Union[str, Path]) -> None:
        """
        Save pipeline state to file.

        Args:
            filepath: Path to save file
        """
        try:
            state = {
                'config': self.config,
                'transformer': self.transformer,
                'feature_selector': self.feature_selector
            }
            dump(state, filepath)
            self.logger.info(f"Pipeline saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Pipeline saving failed: {str(e)}")
            raise

    @classmethod
    def load_pipeline(
        cls,
        filepath: Union[str, Path]
    ) -> 'FeaturePipeline':
        """
        Load pipeline from file.

        Args:
            filepath: Path to load file

        Returns:
            Loaded pipeline instance
        """
        try:
            state = load(filepath)
            pipeline = cls(state['config'])
            pipeline.transformer = state['transformer']
            pipeline.feature_selector = state['feature_selector']

            return pipeline

        except Exception as e:
            logging.error(f"Pipeline loading failed: {str(e)}")
            raise
