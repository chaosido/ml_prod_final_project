"""
Model Trainer Module

Provides training logic for the ASR Quality Estimation model.
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class InsufficientDataError(Exception):
    """Raised when training data doesn't meet minimum requirements."""
    pass


class ModelPerformanceError(Exception):
    """Raised when model performance is below threshold."""
    pass


class TrainingResult:
    """Result of a training run."""
    
    def __init__(
        self,
        model_path: str,
        spearman_rho: float,
        num_samples: int,
        timestamp: str,
    ):
        self.model_path = model_path
        self.spearman_rho = spearman_rho
        self.num_samples = num_samples
        self.timestamp = timestamp


class XGBoostTrainer:
    """
    XGBoost-based model trainer for ASR Quality Estimation.
    
    Handles the full training pipeline:
    - Data validation
    - Train/test splitting
    - Model training
    - Performance evaluation
    - Model versioning and saving
    """
    
    def __init__(
        self,
        min_samples: int = 1000,
        min_spearman_rho: float = 0.4,
        model_dir: str = "models",
        test_size: float = 0.2,
        random_state: int = 42,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
    ):
        # Training validation config
        self.min_samples = min_samples
        self.min_spearman_rho = min_spearman_rho
        self.model_dir = model_dir
        self.test_size = test_size
        self.random_state = random_state
        
        # XGBoost hyperparameters
        self.xgb_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "random_state": random_state,
        }
        
        self._model = None

    def _validate_data(self, X: np.ndarray) -> None:
        """Validate training data meets minimum requirements."""
        num_samples = len(X)
        if num_samples < self.min_samples:
            raise InsufficientDataError(
                f"Insufficient training data: {num_samples} samples "
                f"(minimum: {self.min_samples})"
            )
        logger.info(f"Data validation passed: {num_samples} samples")

    def _calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        rho, p_value = spearmanr(y_true, y_pred)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        return {
            "spearman_rho": float(rho),
            "spearman_p_value": float(p_value),
            "rmse": float(rmse),
        }

    def _validate_metrics(self, metrics: Dict[str, float]) -> None:
        """Validate model performance meets threshold."""
        rho = metrics.get("spearman_rho", 0)
        if rho < self.min_spearman_rho:
            raise ModelPerformanceError(
                f"Model performance below threshold: Spearman Rho = {rho:.4f} "
                f"(minimum: {self.min_spearman_rho})"
            )
        logger.info(f"Metrics validation passed: Spearman Rho = {rho:.4f}")

    def _save_model(self, timestamp: str) -> str:
        """Save model to disk with versioning."""
        model_dir = Path(self.model_dir) / f"v{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / "model.joblib"
        joblib.dump(self._model, model_path)
        
        logger.info(f"Model saved to {model_path}")
        return str(model_path)

    def train(self, X: np.ndarray, y: np.ndarray) -> TrainingResult:
        """
        Train the model with full pipeline.
        
        Args:
            X: Feature matrix
            y: Target values (e.g., WER)
            
        Returns:
            TrainingResult with model path and metrics
            
        Raises:
            InsufficientDataError: If data doesn't meet minimum requirements
            ModelPerformanceError: If model Spearman Rho is below threshold
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Validate data
        self._validate_data(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        logger.info(f"Split: {len(X_train)} train, {len(X_test)} test")
        
        # Create and train model
        from xgboost import XGBRegressor
        self._model = XGBRegressor(**self.xgb_params)
        self._model.fit(X_train, y_train)
        logger.info("XGBoost model training complete")
        
        # Evaluate on test set
        y_pred = self._model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred)
        
        # Validate performance (raises if below threshold)
        self._validate_metrics(metrics)
        
        # Save model
        model_path = self._save_model(timestamp)
        
        return TrainingResult(
            model_path=model_path,
            spearman_rho=metrics["spearman_rho"],
            num_samples=len(X),
            timestamp=timestamp,
        )
