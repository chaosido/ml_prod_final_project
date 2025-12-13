import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from asr_qe.utils.diagnostics import log_diagnostics

logger = logging.getLogger(__name__)


class TrainingResult:
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

    def __init__(self, training_config, xgboost_config, output_dir: str = "models"):
        """
        Args:
            training_config: TrainingConfig object
            xgboost_config: XGBoostConfig object
            output_dir: Directory to save model artifacts
        """
        self.training_config = training_config
        self.xgboost_config = xgboost_config
        self.output_dir = output_dir
        self._model = None

    def _validate_data(self, X: np.ndarray) -> None:
        """just a check if data is actually suitable size&shape"""
        num_samples = len(X)
        if num_samples < self.training_config.min_samples:
            raise ValueError(
                f"Insufficient training data: {num_samples} samples "
                f"(minimum: {self.training_config.min_samples})"
            )
        logger.info(f"Data validation passed: {num_samples} samples")

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
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
        p_value = metrics.get("spearman_p_value", 1.0)

        logger.info(f"Spearman correlation: {rho:.4f} (p-value: {p_value:.4e})")

        if rho < self.training_config.min_spearman_rho:
            raise ValueError(
                f"Model performance below threshold: Spearman Rho = {rho:.4f} "
                f"(minimum: {self.training_config.min_spearman_rho})"
            )
        logger.info(f"Metrics validation passed: Spearman Rho = {rho:.4f}")

    def _save_model(self, timestamp: str) -> str:
        """Save model to disk with versioning."""
        model_dir = Path(self.output_dir) / f"v{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "model.joblib"
        joblib.dump(self._model, model_path)

        logger.info(f"Model saved to {model_path}")
        return str(model_path)

    def set_model(self, model) -> None:
        """Set an external model directly."""
        self._model = model

    def split_data(self, X: np.ndarray, y: np.ndarray):
        """
        Split data into train and test sets using config.
        Returns: (X_train, X_test, y_train, y_test)
        """
        return train_test_split(
            X,
            y,
            test_size=self.training_config.test_size,
            random_state=self.training_config.random_state,
        )

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the loaded model on the provided data.
        Returns dictionary of metrics.
        """
        if self._model is None:
            raise RuntimeError("No model loaded to evaluate.")

        y_pred = self._model.predict(X)
        metrics = self._calculate_metrics(y, y_pred)

        # Log diagnostics based on the prediction
        feature_names = (
            list(self.training_config.feature_columns)
            if hasattr(self.training_config, "feature_columns")
            else None
        )
        log_diagnostics(
            X_train=X,  # Note: using test data as 'train' for stats if only evaluating
            X_test=X,
            y_train=y,
            y_test=y,
            y_pred=y_pred,
            model=self._model,
            feature_names=feature_names,
        )
        return metrics

    def train(self, X: np.ndarray, y: np.ndarray) -> TrainingResult:
        """
        Train the model with full pipeline.

        Args:
            X: Feature matrix
            y: Target values (e.g., WER)

        Returns:
            TrainingResult with model path and metrics

        Raises:
            ValueError: If data insufficient or model performance below threshold
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Validate data
        self._validate_data(X)

        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        logger.info(f"Split: {len(X_train)} train, {len(X_test)} test")

        # Create and train model
        self._model = XGBRegressor(
            n_estimators=self.xgboost_config.n_estimators,
            max_depth=self.xgboost_config.max_depth,
            learning_rate=self.xgboost_config.learning_rate,
            random_state=self.xgboost_config.random_state,
        )
        self._model.fit(X_train, y_train)
        logger.info("XGBoost model training complete")

        # Evaluate on test set
        metrics = self.evaluate(X_test, y_test)

        # Validate performance (raises ValueError if below threshold)
        self._validate_metrics(metrics)

        # Save model
        model_path = self._save_model(timestamp)

        return TrainingResult(
            model_path=model_path,
            spearman_rho=metrics["spearman_rho"],
            num_samples=len(X),
            timestamp=timestamp,
        )
