import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from asr_qe.config import TrainingConfig, XGBoostConfig
from asr_qe.models.trainer import XGBoostTrainer


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


@pytest.mark.parametrize(
    "y_true, y_pred, check_rho, description",
    [
        (
            np.array([1, 2, 3, 4, 5]),
            np.array([1, 2, 3, 4, 5]),
            lambda x: x > 0.99,
            "Perfect correlation",
        ),
        (
            np.array([1, 2, 3, 4, 5]),
            np.array([1.1, 2.2, 2.8, 4.1, 5.2]),
            lambda x: x > 0.9,
            "Some error but high correlation",
        ),
    ],
)
def test_calculate_metrics(
    training_config, xgboost_config, y_true, y_pred, check_rho, description
):
    """Test metric calculation with various scenarios."""
    trainer = XGBoostTrainer(training_config, xgboost_config)

    metrics = trainer._calculate_metrics(y_true, y_pred)
    rho = metrics["spearman_rho"]

    assert check_rho(rho), f"{description}: Unexpected rho {rho}"


def test_validate_metrics_passes(training_config, xgboost_config):
    """Test that good metrics pass validation."""
    trainer = XGBoostTrainer(training_config, xgboost_config)

    trainer._validate_metrics({"spearman_rho": 0.5})


def test_validate_metrics_fails(temp_model_dir, xgboost_config):
    """Test that poor metrics fail validation."""
    config = TrainingConfig(min_spearman_rho=0.4)
    trainer = XGBoostTrainer(config, xgboost_config, output_dir=temp_model_dir)

    with pytest.raises(ValueError, match="below threshold"):
        trainer._validate_metrics({"spearman_rho": 0.1})


def test_split_data_determinism(training_config, xgboost_config):
    """Test that split_data produces deterministic splits logic."""
    trainer = XGBoostTrainer(training_config, xgboost_config)

    X = np.random.randn(20, 2)
    y = np.random.randn(20)

    # Run 1
    X_train1, X_test1, y_train1, y_test1 = trainer.split_data(X, y)

    # Run 2
    X_train2, X_test2, y_train2, y_test2 = trainer.split_data(X, y)

    # Assert Exact Equality
    np.testing.assert_array_equal(X_train1, X_train2)
    np.testing.assert_array_equal(y_test1, y_test2)


def test_set_model_and_evaluate(training_config, xgboost_config):
    """Test that set_model enables evaluate() without retraining."""
    trainer = XGBoostTrainer(training_config, xgboost_config)

    # Mock model
    mock_model = MagicMock()
    # Setup mock behavior: return perfect predictions
    X_test = np.array([[1, 2], [3, 4]])
    y_test = np.array([10, 20])
    mock_model.predict.return_value = np.array([10, 20])

    # Set model
    trainer.set_model(mock_model)

    # Mock log_diagnostics to avoid import/execution issues during test
    with patch("asr_qe.models.trainer.log_diagnostics") as mock_log:
        metrics = trainer.evaluate(X_test, y_test)

        # Verify call
        mock_model.predict.assert_called_once()
        # Verify metrics (perfect prediction = rho 1.0)
        assert metrics["spearman_rho"] == pytest.approx(1.0)
        # Verify diagnostics called
        mock_log.assert_called_once()


def test_evaluate_raises_without_model(training_config, xgboost_config):
    """Test that evaluate() raises error if no model is set."""
    trainer = XGBoostTrainer(training_config, xgboost_config)
    X = np.zeros((5, 2))
    y = np.zeros(5)

    with pytest.raises(RuntimeError, match="No model loaded"):
        trainer.evaluate(X, y)


def test_full_training_pipeline(training_config, xgboost_config):
    """Integration test for full training pipeline."""
    np.random.seed(42)

    X = np.random.randn(100, 2)
    y = X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(100) * 0.1

    # We don't care about output_dir for this test unless we check save location (we do)
    # But temp_model_dir is not passed to this test function.
    # Let's add it or rely on default "models".
    # Actually wait, verify usage of result.model_path below.
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = XGBoostTrainer(training_config, xgboost_config, output_dir=tmpdir)

        # Patch diagnostics to keep test output clean
        with patch("asr_qe.models.trainer.log_diagnostics"):
            result = trainer.train(X, y)

        assert result.spearman_rho > 0.1
        assert result.num_samples == 100
        assert Path(result.model_path).exists()


def test_training_raises_on_insufficient_data(temp_model_dir, xgboost_config):
    """Test that training raises exception with insufficient data."""
    config = TrainingConfig(min_samples=100)
    trainer = XGBoostTrainer(config, xgboost_config, output_dir=temp_model_dir)

    X = np.random.randn(10, 2)
    y = np.random.randn(10)

    with pytest.raises(ValueError, match="Insufficient"):
        trainer.train(X, y)
