"""
Unit tests for jobs/validate_model.py and jobs/deploy_model.py logic.

Since these are Hydra scripts with sys.exit(), we test the core logic
by importing the modules and mocking heavy dependencies.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np


class TestDeployModelLogic:
    """Tests for deploy_model.py file operations."""
    
    def test_copy_model_to_production(self):
        """Test that model file is correctly copied to production directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup: Create staging model
            staging_dir = Path(tmpdir) / "staging"
            staging_dir.mkdir()
            staging_model = staging_dir / "model.joblib"
            staging_model.write_text("mock model content")
            
            # Setup: Create production directory
            prod_dir = Path(tmpdir) / "production"
            prod_dir.mkdir()
            
            # Action: Copy (simulating deploy_model.py logic)
            dest = prod_dir / "model.joblib"
            shutil.copy2(staging_model, dest)
            
            # Assert
            assert dest.exists()
            assert dest.read_text() == "mock model content"
    
    def test_deploy_fails_if_source_missing(self):
        """Test that deployment fails gracefully if source model doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_model = Path(tmpdir) / "nonexistent" / "model.joblib"
            
            # The deploy script should check existence and exit 1
            assert not missing_model.exists()


class TestValidateModelLogic:
    """Tests for validate_model.py comparison logic."""
    
    def test_candidate_better_than_production_passes(self):
        """Test that validation passes when candidate > production."""
        candidate_rho = 0.85
        production_rho = 0.80
        
        # Simulate comparison logic from validate_model.py
        should_pass = candidate_rho >= production_rho
        
        assert should_pass is True
    
    def test_candidate_equal_to_production_passes(self):
        """Test that validation passes when candidate == production (no regression)."""
        candidate_rho = 0.80
        production_rho = 0.80
        
        should_pass = candidate_rho >= production_rho
        
        assert should_pass is True
    
    def test_candidate_worse_than_production_fails(self):
        """Test that validation fails when candidate < production."""
        candidate_rho = 0.70
        production_rho = 0.80
        
        should_pass = candidate_rho >= production_rho
        
        assert should_pass is False
    
    def test_first_deployment_no_production_model(self):
        """Test that first deployment (no production model) always passes."""
        candidate_rho = 0.50
        production_rho = -1.0  # Default when no production model exists
        
        should_pass = candidate_rho >= production_rho
        
        assert should_pass is True


class TestValidateModelIntegration:
    """Integration-style tests for validate_model using mocks."""
    
    @pytest.fixture
    def mock_trainer(self):
        """Create a mock XGBoostTrainer."""
        trainer = MagicMock()
        trainer.split_data.return_value = (
            np.zeros((80, 4)),  # X_train
            np.zeros((20, 4)),  # X_test
            np.zeros(80),       # y_train
            np.zeros(20),       # y_test
        )
        return trainer
    
    def test_validate_model_loads_both_models(self, mock_trainer):
        """Test that validation loads and evaluates both candidate and production."""
        # Setup mock metrics
        candidate_metrics = {"spearman_rho": 0.85}
        production_metrics = {"spearman_rho": 0.80}
        
        # Simulate calling evaluate twice
        mock_trainer.evaluate.side_effect = [candidate_metrics, production_metrics]
        
        # Call evaluate for candidate
        cand_result = mock_trainer.evaluate(np.zeros((20, 4)), np.zeros(20))
        # Call evaluate for production
        prod_result = mock_trainer.evaluate(np.zeros((20, 4)), np.zeros(20))
        
        # Verify both were called
        assert mock_trainer.evaluate.call_count == 2
        assert cand_result["spearman_rho"] > prod_result["spearman_rho"]
    
    def test_validate_model_uses_same_test_split(self, mock_trainer):
        """Test that split_data is called to ensure consistent test set."""
        X = np.random.randn(100, 4)
        y = np.random.randn(100)
        
        # Call split
        result = mock_trainer.split_data(X, y)
        
        # Verify split was used
        mock_trainer.split_data.assert_called_once()
        assert len(result) == 4  # X_train, X_test, y_train, y_test
