"""
Model factory for creating and configuring different types of models.
"""

from typing import Dict, Optional, Type, Union
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import xgboost as xgb
import lightgbm as lgb

from .model_wrapper import BaseModelWrapper, SklearnModelWrapper, TorchModelWrapper


class DeepLearningModel(nn.Module):
    """Base deep learning model for financial time series."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [64, 32],
        dropout_rate: float = 0.2
    ):
        """
        Initialize deep learning model.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Build hidden layers
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class LSTMModel(nn.Module):
    """LSTM model for financial time series."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout_rate: float = 0.2
    ):
        """
        Initialize LSTM model.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM layers
            dropout_rate: Dropout rate
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class AttentionModel(nn.Module):
    """Self-attention model for financial time series."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 4,
        hidden_dim: int = 64,
        dropout_rate: float = 0.2
    ):
        """
        Initialize attention model.

        Args:
            input_dim: Input feature dimension
            num_heads: Number of attention heads
            hidden_dim: Hidden layer dimension
            dropout_rate: Dropout rate
        """
        super().__init__()

        self.attention = nn.MultiheadAttention(
            input_dim,
            num_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim)
        )

        self.output = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with self-attention."""
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)

        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)

        # Output layer
        return self.output(x[:, -1, :])


class ModelFactory:
    """Factory class for creating and configuring models."""

    def __init__(self):
        """Initialize model factory."""
        self.logger = logging.getLogger(__name__)

        # Register available models
        self.available_models = {
            'random_forest': (RandomForestClassifier, SklearnModelWrapper),
            'gradient_boosting': (GradientBoostingClassifier, SklearnModelWrapper),
            'logistic_regression': (LogisticRegression, SklearnModelWrapper),
            'xgboost': (xgb.XGBClassifier, SklearnModelWrapper),
            'lightgbm': (lgb.LGBMClassifier, SklearnModelWrapper),
            'deep_learning': (DeepLearningModel, TorchModelWrapper),
            'lstm': (LSTMModel, TorchModelWrapper),
            'attention': (AttentionModel, TorchModelWrapper)
        }

    def create_model(
        self,
        model_type: str,
        config: Optional[Dict] = None
    ) -> BaseModelWrapper:
        """
        Create and configure a model.

        Args:
            model_type: Type of model to create
            config: Model configuration dictionary

        Returns:
            Configured model wrapper instance
        """
        try:
            if model_type not in self.available_models:
                raise ValueError(f"Unknown model type: {model_type}")

            model_class, wrapper_class = self.available_models[model_type]
            config = config or {}

            # Create model instance based on type
            if model_type in ['deep_learning', 'lstm', 'attention']:
                model = self._create_deep_learning_model(
                    model_class,
                    config
                )
            else:
                model = model_class(**config.get('model_params', {}))

            # Create and return wrapped model
            return wrapper_class(model, config)

        except Exception as e:
            self.logger.error(f"Model creation failed: {str(e)}")
            raise

    def _create_deep_learning_model(
        self,
        model_class: Type[nn.Module],
        config: Dict
    ) -> nn.Module:
        """
        Create deep learning model instance.

        Args:
            model_class: Deep learning model class
            config: Model configuration

        Returns:
            Configured PyTorch model instance
        """
        try:
            model_params = config.get('model_params', {})

            if 'input_dim' not in model_params:
                raise ValueError(
                    "input_dim must be specified for deep learning models")

            return model_class(**model_params)

        except Exception as e:
            self.logger.error(f"Deep learning model creation failed: {str(e)}")
            raise

    def get_default_config(
        self,
        model_type: str,
        input_dim: Optional[int] = None
    ) -> Dict:
        """
        Get default configuration for model type.

        Args:
            model_type: Type of model
            input_dim: Optional input dimension for deep learning models

        Returns:
            Default configuration dictionary
        """
        config = {
            'random_forest': {
                'model_params': {
                    'n_estimators': 100,
                    'max_depth': 5,
                    'min_samples_split': 50,
                    'random_state': 42
                }
            },
            'gradient_boosting': {
                'model_params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 3,
                    'random_state': 42
                }
            },
            'logistic_regression': {
                'model_params': {
                    'C': 1.0,
                    'max_iter': 1000,
                    'random_state': 42
                }
            },
            'xgboost': {
                'model_params': {
                    'n_estimators': 100,
                    'max_depth': 3,
                    'learning_rate': 0.1,
                    'objective': 'binary:logistic',
                    'random_state': 42
                }
            },
            'lightgbm': {
                'model_params': {
                    'n_estimators': 100,
                    'max_depth': -1,
                    'learning_rate': 0.1,
                    'objective': 'binary',
                    'random_state': 42
                }
            },
            'deep_learning': {
                'model_params': {
                    'input_dim': input_dim,
                    'hidden_dims': [64, 32],
                    'dropout_rate': 0.2
                },
                'training_params': {
                    'batch_size': 32,
                    'learning_rate': 0.001,
                    'epochs': 100
                }
            },
            'lstm': {
                'model_params': {
                    'input_dim': input_dim,
                    'hidden_dim': 64,
                    'num_layers': 2,
                    'dropout_rate': 0.2
                },
                'training_params': {
                    'batch_size': 32,
                    'learning_rate': 0.001,
                    'epochs': 100
                }
            },
            'attention': {
                'model_params': {
                    'input_dim': input_dim,
                    'num_heads': 4,
                    'hidden_dim': 64,
                    'dropout_rate': 0.2
                },
                'training_params': {
                    'batch_size': 32,
                    'learning_rate': 0.001,
                    'epochs': 100
                }
            }
        }

        return config.get(model_type, {})
