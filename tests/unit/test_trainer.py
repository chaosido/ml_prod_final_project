"""
Unit tests for the model trainer module.
Uses table-driven tests with Mock configuration.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile

from asr_qe.models.trainer import (
    TrainingConfig,
    TrainingResult,
    BaseTrainer,
    XGBoostTrainer,
)


@pytest.fixture
def config():
    """Returns a TrainingConfig for testing."""
    return TrainingConfig(
        min_samples=10,  # Lower for testing
        min_spearman_rho=0.3,
        model_dir=tempfile.mkdtemp(),
        feature_columns=("feat1", "feat2"),
        target_column="target",
    )


@pytest.fixture
def trainer(config):
    """Returns an XGBoostTrainer with test config."""
    return XGBoostTrainer(config=config, n_estimators=10)


def test_validate_data(trainer, config):
    """Test data validation logic."""
    cases = [
        {
            "name": "Sufficient samples",
            "X": np.random.randn(20, 2),
            "y": np.random.randn(20),
            "should_pass": True,
        },
        {
            "name": "Insufficient samples",
            "X": np.random.randn(5, 2),
            "y": np.random.randn(5),
            "should_pass": False,
        },
    ]
    
    for case in cases:
        if case["should_pass"]:
            trainer.validate_data(case["X"], case["y"])
        else:
            with pytest.raises(ValueError, match="Insufficient"):
                trainer.validate_data(case["X"], case["y"])


def test_calculate_metrics(trainer):
    """Test metric calculation."""
    cases = [
        {
            "name": "Perfect correlation",
            "y_true": np.array([1, 2, 3, 4, 5]),
            "y_pred": np.array([1, 2, 3, 4, 5]),
            "expected_rho": lambda x: x > 0.99,
            "expected_rmse": lambda x: x < 0.01,
        },
        {
            "name": "Some error",
            "y_true": np.array([1, 2, 3, 4, 5]),
            "y_pred": np.array([1.1, 2.2, 2.8, 4.1, 5.2]),
            "expected_rho": lambda x: x > 0.9,
            "expected_rmse": lambda x: 0.1 < x < 0.3,
        },
    ]
    
    for case in cases:
        metrics = trainer.calculate_metrics(case["y_true"], case["y_pred"])
        
        assert case["expected_rho"](metrics["spearman_rho"]), \
            f"{case['name']}: Unexpected rho {metrics['spearman_rho']}"
        assert case["expected_rmse"](metrics["rmse"]), \
            f"{case['name']}: Unexpected RMSE {metrics['rmse']}"


def test_validate_metrics(trainer, config):
    """Test metric validation against thresholds."""
    cases = [
        {
            "name": "Above threshold",
            "metrics": {"spearman_rho": 0.5},
            "should_pass": True,
        },
        {
            "name": "Below threshold",
            "metrics": {"spearman_rho": 0.1},
            "should_pass": False,
        },
        {
            "name": "At threshold",
            "metrics": {"spearman_rho": 0.3},
            "should_pass": True,  # At threshold is actually valid (>=)
        },
    ]
    
    for case in cases:
        if case["should_pass"]:
            trainer.validate_metrics(case["metrics"])
        else:
            with pytest.raises(ValueError, match="below threshold"):
                trainer.validate_metrics(case["metrics"])


def test_save_model(trainer, config):
    """Test model saving creates correct file structure."""
    mock_model = Mock()
    
    with patch("asr_qe.models.trainer.joblib.dump") as mock_dump:
        path = trainer.save_model(mock_model, timestamp="20231201_120000")
        
        assert "v20231201_120000" in path
        assert path.endswith("model.joblib")
        mock_dump.assert_called_once()


def test_full_training_pipeline(trainer, config):
    """Integration test for full training pipeline."""
    np.random.seed(42)
    
    # Generate synthetic data with clear correlation
    X = np.random.randn(50, 2)
    y = X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(50) * 0.1
    
    result = trainer.train(X, y)
    
    assert result.success
    assert result.spearman_rho > config.min_spearman_rho
    assert result.num_samples == 50
    assert Path(result.model_path).exists()


def test_training_fails_on_insufficient_data(config):
    """Test that training fails with insufficient data."""
    trainer = XGBoostTrainer(config=config)
    
    X = np.random.randn(5, 2)  # Less than min_samples=10
    y = np.random.randn(5)
    
    result = trainer.train(X, y)
    
    assert not result.success
    assert "Insufficient" in result.error_message
