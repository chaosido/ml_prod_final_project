import logging
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def log_diagnostics(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    model: Optional[Any] = None,
    feature_names: Optional[List[str]] = None,
) -> None:
    """
    Log diagnostic information about model performance and features, we can use this to know how features did in the model,  # noqa: E501
    and it automatically gets called in the trainer loop. 

    Args:
        X_train: Training feature matrix
        X_test: Test feature matrix
        y_train: Training target values
        y_test: Test target values
        y_pred: Model predictions on test set
        model: Trained model (optional, for feature importance)
        feature_names: Optional list of feature names for logging
    """
    logger.info("=" * 60)
    logger.info("MODEL DIAGNOSTICS")
    logger.info("=" * 60)

    # Feature names
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
    elif len(feature_names) != X_train.shape[1]:
        logger.warning(
            f"Feature names length ({len(feature_names)}) doesn't match features ({X_train.shape[1]}), using defaults"  # noqa: E501
        )
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

    # Feature importance from XGBoost (if available)
    if model is not None:
        try:
            # Handle XGBRegressor feature importance
            if hasattr(model, "feature_importances_"):
                feature_importance = model.feature_importances_
                logger.info("Feature Importance (from Model):")
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
        logger.info(
            f"    Mean: {feat_mean:.4f}, Std: {feat_std:.4f}, Min: {feat_min:.4f}, Max: {feat_max:.4f}"  # noqa: E501
        )

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
    logger.info(
        f"  Actual WER - Mean: {np.mean(y_test):.4f}, Std: {np.std(y_test):.4f}, Min: {np.min(y_test):.4f}, Max: {np.max(y_test):.4f}"  # noqa: E501
    )
    logger.info(
        f"  Predicted WER - Mean: {np.mean(y_pred):.4f}, Std: {np.std(y_pred):.4f}, Min: {np.min(y_pred):.4f}, Max: {np.max(y_pred):.4f}"  # noqa: E501
    )

    # Error stats
    error = np.abs(y_test - y_pred)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    logger.info(f"  Prediction Error - Mean: {np.mean(error):.4f}, RMSE: {rmse:.4f}")

    logger.info("=" * 60)
