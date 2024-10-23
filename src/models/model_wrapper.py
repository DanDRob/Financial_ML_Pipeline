"""
Base model wrapper and interfaces for ML models.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import torch
import torch.nn as nn


class BaseModelWrapper(ABC):
    """Abstract base class for model wrappers."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize base model wrapper.

        Args:
            config: Model configuration dictionary
        """
        self.config = config or {}
        self.model = None
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None
    ) -> Dict[str, List[float]]:
        """
        Fit model to training data.

        Args:
            X: Training features
            y: Training targets
            validation_data: Optional validation data tuple

        Returns:
            Dictionary of training metrics
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using fitted model.

        Args:
            X: Feature DataFrame

        Returns:
            numpy array of predictions
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make probability predictions.

        Args:
            X: Feature DataFrame

        Returns:
            numpy array of probability predictions
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass

    def save_model(self, filepath: str) -> None:
        """
        Save model to file.

        Args:
            filepath: Path to save file
        """
        try:
            import joblib

            state = {
                'model': self.model,
                'config': self.config
            }

            joblib.dump(state, filepath)
            self.logger.info(f"Model saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Model saving failed: {str(e)}")
            raise

    def load_model(self, filepath: str) -> None:
        """
        Load model from file.

        Args:
            filepath: Path to load file
        """
        try:
            import joblib

            state = joblib.load(filepath)
            self.model = state['model']
            self.config = state['config']

            self.logger.info(f"Model loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
            raise


class SklearnModelWrapper(BaseModelWrapper):
    """Wrapper for scikit-learn models."""

    def __init__(
        self,
        model: BaseEstimator,
        config: Optional[Dict] = None
    ):
        """
        Initialize sklearn model wrapper.

        Args:
            model: Scikit-learn model instance
            config: Model configuration dictionary
        """
        super().__init__(config)
        self.model = model

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None
    ) -> Dict[str, List[float]]:
        """
        Fit sklearn model.

        Args:
            X: Training features
            y: Training targets
            validation_data: Optional validation data tuple

        Returns:
            Dictionary of training metrics
        """
        try:
            # Fit model
            self.model.fit(X, y)

            # Calculate training metrics
            train_pred = self.model.predict(X)
            metrics = {
                'train_accuracy': [self._calculate_accuracy(y, train_pred)],
                'train_loss': [self._calculate_loss(y, train_pred)]
            }

            # Calculate validation metrics if provided
            if validation_data is not None:
                X_val, y_val = validation_data
                val_pred = self.model.predict(X_val)
                metrics['val_accuracy'] = [
                    self._calculate_accuracy(y_val, val_pred)]
                metrics['val_loss'] = [self._calculate_loss(y_val, val_pred)]

            return metrics

        except Exception as e:
            self.logger.error(f"Model fitting failed: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using sklearn model."""
        try:
            return self.model.predict(X)
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions using sklearn model."""
        try:
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
            else:
                raise NotImplementedError(
                    "Model does not support probability predictions")
        except Exception as e:
            self.logger.error(f"Probability prediction failed: {str(e)}")
            raise

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from sklearn model."""
        try:
            if hasattr(self.model, 'feature_importances_'):
                return dict(zip(
                    self.config.get('feature_names', []),
                    self.model.feature_importances_
                ))
            elif hasattr(self.model, 'coef_'):
                return dict(zip(
                    self.config.get('feature_names', []),
                    np.abs(self.model.coef_)
                ))
            else:
                return None
        except Exception as e:
            self.logger.error(
                f"Feature importance calculation failed: {str(e)}")
            return None

    @staticmethod
    def _calculate_accuracy(y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate prediction accuracy."""
        return np.mean(y_true == y_pred)

    @staticmethod
    def _calculate_loss(y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate prediction loss."""
        return np.mean((y_true - y_pred) ** 2)


class TorchModelWrapper(BaseModelWrapper):
    """Wrapper for PyTorch models."""

    def __init__(
        self,
        model: nn.Module,
        config: Optional[Dict] = None
    ):
        """
        Initialize PyTorch model wrapper.

        Args:
            model: PyTorch model instance
            config: Model configuration dictionary
        """
        super().__init__(config)
        self.model = model
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None
    ) -> Dict[str, List[float]]:
        """
        Fit PyTorch model.

        Args:
            X: Training features
            y: Training targets
            validation_data: Optional validation data tuple

        Returns:
            Dictionary of training metrics
        """
        try:
            # Convert data to tensors
            X_tensor = torch.FloatTensor(X.values).to(self.device)
            y_tensor = torch.FloatTensor(y.values).to(self.device)

            if validation_data is not None:
                X_val, y_val = validation_data
                X_val_tensor = torch.FloatTensor(X_val.values).to(self.device)
                y_val_tensor = torch.FloatTensor(y_val.values).to(self.device)

            # Initialize optimizer and loss function
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.get('learning_rate', 0.001)
            )
            criterion = nn.BCEWithLogitsLoss()

            # Training loop
            epochs = self.config.get('epochs', 100)
            batch_size = self.config.get('batch_size', 32)

            metrics = {
                'train_loss': [],
                'train_accuracy': []
            }

            if validation_data is not None:
                metrics.update({
                    'val_loss': [],
                    'val_accuracy': []
                })

            for epoch in range(epochs):
                # Training
                self.model.train()
                train_losses = []
                train_accuracies = []

                for i in range(0, len(X_tensor), batch_size):
                    batch_X = X_tensor[i:i+batch_size]
                    batch_y = y_tensor[i:i+batch_size]

                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y.unsqueeze(1))

                    loss.backward()
                    optimizer.step()

                    preds = (outputs > 0).float()
                    accuracy = (preds == batch_y.unsqueeze(1)).float().mean()

                    train_losses.append(loss.item())
                    train_accuracies.append(accuracy.item())

                metrics['train_loss'].append(np.mean(train_losses))
                metrics['train_accuracy'].append(np.mean(train_accuracies))

                # Validation
                if validation_data is not None:
                    self.model.eval()
                    with torch.no_grad():
                        val_outputs = self.model(X_val_tensor)
                        val_loss = criterion(
                            val_outputs,
                            y_val_tensor.unsqueeze(1)
                        ).item()
                        val_preds = (val_outputs > 0).float()
                        val_accuracy = (
                            val_preds == y_val_tensor.unsqueeze(1)
                        ).float().mean().item()

                        metrics['val_loss'].append(val_loss)
                        metrics['val_accuracy'].append(val_accuracy)

            return metrics

        except Exception as e:
            self.logger.error(f"Model fitting failed: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using PyTorch model."""
        try:
            self.model.eval()
            X_tensor = torch.FloatTensor(X.values).to(self.device)

            with torch.no_grad():
                outputs = self.model(X_tensor)
                predictions = (outputs > 0).cpu().numpy()

            return predictions.squeeze()

        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions using PyTorch model."""
        try:
            self.model.eval()
            X_tensor = torch.FloatTensor(X.values).to(self.device)

            with torch.no_grad():
                outputs = torch.sigmoid(self.model(X_tensor))
                probabilities = outputs.cpu().numpy()

            return probabilities.squeeze()

        except Exception as e:
            self.logger.error(f"Probability prediction failed: {str(e)}")
            raise

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from PyTorch model."""
        try:
            # Use gradient-based feature importance
            importances = []
            X_tensor = torch.FloatTensor(
                self.config.get('feature_data', [])
            ).to(self.device)
            X_tensor.requires_grad = True

            outputs = self.model(X_tensor)
            outputs.sum().backward()

            importances = X_tensor.grad.abs().mean(0).cpu().numpy()

            return dict(zip(
                self.config.get('feature_names', []),
                importances
            ))

        except Exception as e:
            self.logger.error(
                f"Feature importance calculation failed: {str(e)}")
            return None
