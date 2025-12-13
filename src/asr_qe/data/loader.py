"""
Data loading and preparation utilities.

This module provides reusable functions for loading feature data
and preparing it for model training or inference.
"""

import logging
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_features(input_path: str) -> pd.DataFrame:
    """
    Load features from parquet file(s).

    Args:
        input_path: Path to parquet file or directory containing parquet files.

    Returns:
        DataFrame with loaded features.
    """
    path = Path(input_path)
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} samples from {input_path}")
    return df


def prepare_data(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare feature matrix and target vector from DataFrame.

    Args:
        df: DataFrame containing features and target.
        feature_columns: List of column names to use as features.
        target_column: Name of target column.

    Returns:
        Tuple of (X, y) arrays ready for training.

    Raises:
        ValueError: If required columns are missing.
    """
    feature_cols = list(feature_columns)

    # Validate feature columns exist
    missing_cols = set(feature_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing feature columns: {missing_cols}. Available columns: {list(df.columns)}"  # noqa: E501
        )

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")

    # Extract arrays
    X = df[feature_cols].values
    y = df[target_column].values

    # Filter NaN values
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[valid_mask]
    y = y[valid_mask]

    logger.info(f"Prepared {len(X)} valid samples for training")
    return X, y
