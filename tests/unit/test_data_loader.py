"""
Unit tests for the data loader module.
"""
import pytest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from asr_qe.data.loader import load_features, prepare_data


@pytest.fixture
def temp_parquet(tmp_path):
    """Create a temporary parquet file with sample data."""
    df = pd.DataFrame({
        "feat1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feat2": [10.0, 20.0, 30.0, 40.0, 50.0],
        "target": [0.1, 0.2, 0.3, 0.4, 0.5],
    })
    path = tmp_path / "test.parquet"
    df.to_parquet(path)
    return str(path)


def test_load_features(temp_parquet):
    """Test loading features from parquet."""
    df = load_features(temp_parquet)
    
    assert len(df) == 5
    assert "feat1" in df.columns
    assert "feat2" in df.columns
    assert "target" in df.columns


def test_load_features_missing_file():
    """Test loading from non-existent path raises error."""
    with pytest.raises(Exception):
        load_features("/nonexistent/path.parquet")


def test_prepare_data_valid():
    """Test preparing data with valid columns."""
    df = pd.DataFrame({
        "feat1": [1.0, 2.0, 3.0],
        "feat2": [10.0, 20.0, 30.0],
        "target": [0.1, 0.2, 0.3],
    })
    
    X, y = prepare_data(df, feature_columns=["feat1", "feat2"], target_column="target")
    
    assert X.shape == (3, 2)
    assert y.shape == (3,)
    assert np.allclose(X[:, 0], [1.0, 2.0, 3.0])
    assert np.allclose(y, [0.1, 0.2, 0.3])


def test_prepare_data_missing_feature_column():
    """Test that missing feature columns raise ValueError."""
    df = pd.DataFrame({
        "feat1": [1.0, 2.0],
        "target": [0.1, 0.2],
    })
    
    with pytest.raises(ValueError, match="Missing feature columns"):
        prepare_data(df, feature_columns=["feat1", "feat2"], target_column="target")


def test_prepare_data_missing_target_column():
    """Test that missing target column raises ValueError."""
    df = pd.DataFrame({
        "feat1": [1.0, 2.0],
        "feat2": [10.0, 20.0],
    })
    
    with pytest.raises(ValueError, match="not found"):
        prepare_data(df, feature_columns=["feat1", "feat2"], target_column="target")


def test_prepare_data_filters_nan():
    """Test that NaN values are filtered out."""
    df = pd.DataFrame({
        "feat1": [1.0, np.nan, 3.0, 4.0],
        "feat2": [10.0, 20.0, np.nan, 40.0],
        "target": [0.1, 0.2, 0.3, np.nan],
    })
    
    X, y = prepare_data(df, feature_columns=["feat1", "feat2"], target_column="target")
    
    # Only row 0 should remain (rows 1,2,3 have NaN somewhere)
    assert X.shape == (1, 2)
    assert y.shape == (1,)
    assert np.allclose(X[0], [1.0, 10.0])
    assert np.allclose(y, [0.1])
