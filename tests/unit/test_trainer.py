"""
Unit tests for the model trainer module.
"""
import pytest
import numpy as np
from unittest.mock import patch
from pathlib import Path
import tempfile

from asr_qe.models.trainer import (
    XGBoostTrainer,
    TrainingResult,
    InsufficientDataError,
    ModelPerformanceError,
)


@pytest.fixture
def temp_model_dir():
    """Create a temporary directory for model outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_validate_data_sufficient(temp_model_dir):
    """Test that validation passes with sufficient samples."""
    trainer = XGBoostTrainer(min_samples=10, model_dir=temp_model_dir)
    X = np.random.randn(20, 2)
    
    # Should not raise
    trainer._validate_data(X)


def test_validate_data_insufficient(temp_model_dir):
    """Test that validation fails with insufficient samples."""
    trainer = XGBoostTrainer(min_samples=10, model_dir=temp_model_dir)
    X = np.random.randn(5, 2)
    
    with pytest.raises(InsufficientDataError, match="Insufficient"):
        trainer._validate_data(X)


def test_calculate_metrics(temp_model_dir):
    """Test metric calculation."""
    trainer = XGBoostTrainer(model_dir=temp_model_dir)
    
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


def test_validate_metrics_passes(temp_model_dir):
    """Test that good metrics pass validation."""
    trainer = XGBoostTrainer(min_spearman_rho=0.3, model_dir=temp_model_dir)
    
    # Should not raise
    trainer._validate_metrics({"spearman_rho": 0.5})


def test_validate_metrics_fails(temp_model_dir):
    """Test that poor metrics fail validation."""
    trainer = XGBoostTrainer(min_spearman_rho=0.4, model_dir=temp_model_dir)
    
    with pytest.raises(ModelPerformanceError, match="below threshold"):
        trainer._validate_metrics({"spearman_rho": 0.1})


def test_full_training_pipeline(temp_model_dir):
    """Integration test for full training pipeline."""
    np.random.seed(42)
    
    # Generate synthetic data with clear correlation
    X = np.random.randn(100, 2)
    y = X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(100) * 0.1
    
    trainer = XGBoostTrainer(
        min_samples=10,
        min_spearman_rho=0.1,  # Low threshold for synthetic data
        model_dir=temp_model_dir,
        n_estimators=10,
    )
    
    result = trainer.train(X, y)
    
    assert result.spearman_rho > 0.1
    assert result.num_samples == 100
    assert Path(result.model_path).exists()


def test_training_raises_on_insufficient_data(temp_model_dir):
    """Test that training raises exception with insufficient data."""
    trainer = XGBoostTrainer(min_samples=100, model_dir=temp_model_dir)
    
    X = np.random.randn(10, 2)
    y = np.random.randn(10)
    
    with pytest.raises(InsufficientDataError):
        trainer.train(X, y)
