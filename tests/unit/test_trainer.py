"""
Unit tests for the model trainer module.
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile

from asr_qe.config import TrainingConfig, XGBoostConfig
from asr_qe.models.trainer import XGBoostTrainer, TrainingResult


@pytest.fixture
def temp_model_dir():
    """Create a temporary directory for model outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def training_config(temp_model_dir):
    """Create a test training config."""
    return TrainingConfig(
        min_samples=10,
        min_spearman_rho=0.1,
        model_dir=temp_model_dir,
        test_size=0.2,
    )


@pytest.fixture
def xgboost_config():
    """Create a test XGBoost config."""
    return XGBoostConfig(n_estimators=10)


def test_validate_data_sufficient(training_config, xgboost_config):
    """Test that validation passes with sufficient samples."""
    trainer = XGBoostTrainer(training_config, xgboost_config)
    X = np.random.randn(20, 2)
    
    trainer._validate_data(X)


def test_validate_data_insufficient(training_config, xgboost_config):
    """Test that validation fails with insufficient samples."""
    trainer = XGBoostTrainer(training_config, xgboost_config)
    X = np.random.randn(5, 2)
    
    with pytest.raises(ValueError, match="Insufficient"):
        trainer._validate_data(X)


def test_calculate_metrics(training_config, xgboost_config):
    """Test metric calculation."""
    trainer = XGBoostTrainer(training_config, xgboost_config)
    
    cases = [
        {
            "name": "Perfect correlation",
            "y_true": np.array([1, 2, 3, 4, 5]),
            "y_pred": np.array([1, 2, 3, 4, 5]),
            "expected_rho": lambda x: x > 0.99,
        },
        {
            "name": "Some error",
            "y_true": np.array([1, 2, 3, 4, 5]),
            "y_pred": np.array([1.1, 2.2, 2.8, 4.1, 5.2]),
            "expected_rho": lambda x: x > 0.9,
        },
    ]
    
    for case in cases:
        metrics = trainer._calculate_metrics(case["y_true"], case["y_pred"])
        assert case["expected_rho"](metrics["spearman_rho"]), \
            f"{case['name']}: Unexpected rho {metrics['spearman_rho']}"


def test_validate_metrics_passes(training_config, xgboost_config):
    """Test that good metrics pass validation."""
    trainer = XGBoostTrainer(training_config, xgboost_config)
    
    trainer._validate_metrics({"spearman_rho": 0.5})


def test_validate_metrics_fails(temp_model_dir, xgboost_config):
    """Test that poor metrics fail validation."""
    config = TrainingConfig(min_spearman_rho=0.4, model_dir=temp_model_dir)
    trainer = XGBoostTrainer(config, xgboost_config)
    
    with pytest.raises(ValueError, match="below threshold"):
        trainer._validate_metrics({"spearman_rho": 0.1})


def test_full_training_pipeline(training_config, xgboost_config):
    """Integration test for full training pipeline."""
    np.random.seed(42)
    
    X = np.random.randn(100, 2)
    y = X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(100) * 0.1
    
    trainer = XGBoostTrainer(training_config, xgboost_config)
    result = trainer.train(X, y)
    
    assert result.spearman_rho > 0.1
    assert result.num_samples == 100
    assert Path(result.model_path).exists()


def test_training_raises_on_insufficient_data(temp_model_dir, xgboost_config):
    """Test that training raises exception with insufficient data."""
    config = TrainingConfig(min_samples=100, model_dir=temp_model_dir)
    trainer = XGBoostTrainer(config, xgboost_config)
    
    X = np.random.randn(10, 2)
    y = np.random.randn(10)
    
    with pytest.raises(ValueError, match="Insufficient"):
        trainer.train(X, y)
