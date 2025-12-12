import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

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
    
    def __init__(self, training_config, xgboost_config):
        """
        Args:
            training_config: TrainingConfig from conf/configs.py
            xgboost_config: XGBoostConfig from conf/configs.py
        """
        self.training_config = training_config
        self.xgboost_config = xgboost_config
        self._model = None

    def _validate_data(self, X: np.ndarray) -> None:
        ''' just a check if data is actually suitable size&shape'''
        num_samples = len(X)
        if num_samples < self.training_config.min_samples:
            raise ValueError(
                f"Insufficient training data: {num_samples} samples "
                f"(minimum: {self.training_config.min_samples})"
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

    def _log_diagnostics(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """
        Log diagnostic information about model performance and features.
        
        Args:
            X_train: Training feature matrix
            X_test: Test feature matrix
            y_train: Training target values
            y_test: Test target values
            y_pred: Model predictions on test set
            feature_names: Optional list of feature names for logging
        """
        logger.info("=" * 60)
        logger.info("MODEL DIAGNOSTICS")
        logger.info("=" * 60)
        
        # Feature names
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        elif len(feature_names) != X_train.shape[1]:
            logger.warning(f"Feature names length ({len(feature_names)}) doesn't match features ({X_train.shape[1]}), using defaults")
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        # Feature importance from XGBoost
        if self._model is not None:
            try:
                feature_importance = self._model.feature_importances_
                logger.info("Feature Importance (from XGBoost):")
                importance_pairs = list(zip(feature_names, feature_importance))
                importance_pairs.sort(key=lambda x: x[1], reverse=True)
                for feat_name, importance in importance_pairs:
                    logger.info(f"  {feat_name}: {importance:.6f}")
            except Exception as e:
                logger.warning(f"Could not extract feature importance: {e}")
        
        # Feature statistics
        logger.info("Feature Statistics (on training set):")
        for idx, feat_name in enumerate(feature_names):
            feat_values = X_train[:, idx]
            feat_mean = np.mean(feat_values)
            feat_std = np.std(feat_values)
            feat_min = np.min(feat_values)
            feat_max = np.max(feat_values)
            logger.info(f"  {feat_name}:")
            logger.info(f"    Mean: {feat_mean:.4f}, Std: {feat_std:.4f}, Min: {feat_min:.4f}, Max: {feat_max:.4f}")
        
        # Feature-target correlations
        logger.info("Feature-Target Correlations (on training set):")
        for idx, feat_name in enumerate(feature_names):
            feat_values = X_train[:, idx]
            corr_coef = np.corrcoef(feat_values, y_train)[0, 1]
            if not np.isnan(corr_coef):
                logger.info(f"  {feat_name}: {corr_coef:.4f}")
            else:
                logger.info(f"  {feat_name}: NaN (constant feature or invalid)")
        
        # Prediction statistics
        logger.info("Prediction Statistics (on test set):")
        logger.info(f"  Actual WER - Mean: {np.mean(y_test):.4f}, Std: {np.std(y_test):.4f}, Min: {np.min(y_test):.4f}, Max: {np.max(y_test):.4f}")
        logger.info(f"  Predicted WER - Mean: {np.mean(y_pred):.4f}, Std: {np.std(y_pred):.4f}, Min: {np.min(y_pred):.4f}, Max: {np.max(y_pred):.4f}")
        logger.info(f"  Prediction Error - Mean: {np.mean(np.abs(y_test - y_pred)):.4f}, RMSE: {np.sqrt(np.mean((y_test - y_pred) ** 2)):.4f}")
        
        logger.info("=" * 60)

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
        model_dir = Path(self.training_config.model_dir) / f"v{timestamp}"
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
            ValueError: If data insufficient or model performance below threshold
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Validate data
        self._validate_data(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.training_config.test_size, 
            random_state=self.training_config.random_state,
        )
        logger.info(f"Split: {len(X_train)} train, {len(X_test)} test")
        
        # Create and train model
        from xgboost import XGBRegressor
        self._model = XGBRegressor(
            n_estimators=self.xgboost_config.n_estimators,
            max_depth=self.xgboost_config.max_depth,
            learning_rate=self.xgboost_config.learning_rate,
            random_state=self.xgboost_config.random_state,
        )
        self._model.fit(X_train, y_train)
        logger.info("XGBoost model training complete")
        
        # Evaluate on test set
        y_pred = self._model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred)
        
        # Log diagnostics before validation
        feature_names = list(self.training_config.feature_columns) if hasattr(self.training_config, 'feature_columns') else None
        self._log_diagnostics(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            y_pred=y_pred,
            feature_names=feature_names,
        )
        
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
