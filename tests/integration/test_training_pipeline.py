"""
Integration tests for the model training pipeline.

These tests verify the full end-to-end training workflow:
1. Loading features from parquet
2. Training the model
3. Saving and loading the model
"""
import pytest
import tempfile
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from asr_qe.models.trainer import XGBoostTrainer
from asr_qe.models.loader import ModelLoader


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def sample_features_parquet(temp_dir):
    """Create sample parquet file with features."""
    np.random.seed(42)
    
    n_samples = 100
    df = pd.DataFrame({
        "rms_db": np.random.uniform(-60, -10, n_samples),
        "snr_db": np.random.uniform(0, 40, n_samples),
        "duration": np.random.uniform(1, 10, n_samples),
        "wer": np.random.uniform(0, 1, n_samples),
    })
    
    # Make target correlated with features
    df["wer"] = 0.5 - 0.01 * df["snr_db"] + 0.005 * df["rms_db"] + np.random.randn(n_samples) * 0.1
    df["wer"] = df["wer"].clip(0, 1)
    
    parquet_path = Path(temp_dir) / "features.parquet"
    df.to_parquet(parquet_path)
    
    return str(parquet_path)


def test_full_training_pipeline(temp_dir, sample_features_parquet):
    """Test complete training pipeline from parquet to saved model."""
    df = pd.read_parquet(sample_features_parquet)
    
    feature_columns = ["rms_db", "snr_db", "duration"]
    target_column = "wer"
    
    X = df[feature_columns].values
    y = df[target_column].values
    
    trainer = XGBoostTrainer(
        min_samples=50,
        min_spearman_rho=0.1,
        model_dir=temp_dir,
        n_estimators=50,
    )
    result = trainer.train(X, y)
    
    assert result.num_samples == 100
    assert Path(result.model_path).exists()
    
    # Verify model can be loaded and used
    ModelLoader._instance = None
    loader = ModelLoader()
    loader.load(result.model_path)
    
    assert loader.is_loaded
    
    predictions = loader.predict(X[:5])
    assert len(predictions) == 5


def test_training_with_missing_target_column(temp_dir):
    """Test that prepare_data fails gracefully with missing target column."""
    from asr_qe.data import prepare_data
    
    df = pd.DataFrame({
        "rms_db": [1, 2, 3],
        "snr_db": [10, 20, 30],
        "duration": [1.0, 2.0, 3.0],
    })
    
    with pytest.raises(ValueError, match="not found"):
        prepare_data(df, feature_columns=["rms_db", "snr_db"], target_column="wer")


def test_model_versioning(temp_dir, sample_features_parquet):
    """Test that multiple training runs create versioned models."""
    df = pd.read_parquet(sample_features_parquet)
    
    X = df[["rms_db", "snr_db", "duration"]].values
    y = df["wer"].values
    
    trainer = XGBoostTrainer(
        min_samples=10,
        min_spearman_rho=0.0,
        model_dir=temp_dir,
        n_estimators=10,
    )
    
    result1 = trainer.train(X, y)
    time.sleep(1.1)
    result2 = trainer.train(X, y)
    
    assert result1.model_path != result2.model_path
    assert Path(result1.model_path).exists()
    assert Path(result2.model_path).exists()
