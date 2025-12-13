from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import ListConfig

from asr_qe.config import (
    Config,
    DeploymentConfig,
    TrainingConfig,
    ValidationConfig,
    XGBoostConfig,
    register_configs,
)


class TestStructuredConfigs:
    """Test Hydra Structured Config classes."""

    def test_training_config_defaults(self):
        """Verify TrainingConfig has correct default values from dataclass."""
        config = TrainingConfig()

        assert config.min_samples == 1000  
        assert config.min_spearman_rho == 0.4
        assert config.test_size == 0.2
        assert config.random_state == 42
        assert "rms_db" in config.feature_columns
        assert config.target_column == "wer"

    def test_xgboost_config_defaults(self):
        """Verify XGBoostConfig has correct defaults."""
        config = XGBoostConfig()

        assert config.n_estimators == 100
        assert config.max_depth == 6
        assert config.learning_rate == 0.1
        assert config.random_state == 42

    def test_top_level_config_composition(self):
        """Verify Config properly nests sub-configs."""
        config = Config()

        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.training.xgboost, XGBoostConfig)
        assert isinstance(config.deployment, DeploymentConfig)
        assert isinstance(config.validation, ValidationConfig)

    def test_deployment_config_has_candidate_model(self):
        """Verify DeploymentConfig includes candidate_model field (for DAG overrides)."""
        config = DeploymentConfig()

        assert hasattr(config, "candidate_model")
        assert config.candidate_model == "/models/staging/latest/model.joblib"

    def test_validation_config_structure(self):
        """Verify ValidationConfig has all required fields."""
        config = ValidationConfig()

        assert hasattr(config, "candidate_model")
        assert hasattr(config, "production_model")
        assert hasattr(config, "data_path")


class TestHydraIntegration:
    """Test Hydra config loading with the actual config.yaml file."""

    @pytest.fixture
    def config_dir(self):
        """Get absolute path to conf directory."""
        # Assumes tests run from project root
        project_root = Path(__file__).parent.parent.parent
        return str(project_root / "conf")

    def test_register_configs_without_error(self):
        """Verify register_configs() runs without error."""
        register_configs()  # Should not raise

    def test_hydra_loads_structured_config(self, config_dir):
        """Test that Hydra can load config.yaml with the schema."""
        register_configs()

        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config")

            # Verify structure
            assert "training" in cfg
            assert "xgboost" in cfg.training
            assert "deployment" in cfg
            assert "validation" in cfg

            # Verify values from config.yaml are loaded with correct types
        assert isinstance(cfg.training.min_samples, int)
        assert isinstance(cfg.training.xgboost.n_estimators, int)
        assert isinstance(cfg.training.min_spearman_rho, float)
        assert isinstance(cfg.training.test_size, float)
        assert isinstance(cfg.training.feature_columns, (list, ListConfig))
        assert isinstance(cfg.training.target_column, str) 
        assert isinstance(cfg.training.random_state, int)

    def test_hydra_validates_types(self, config_dir):
        """Test that Hydra raises error for invalid types."""
        register_configs()
        
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            # Attempting to override an int field with a string should fail
            # because of the schema in Config
            with pytest.raises(Exception): # Hydra/OmegaConf raises distinct errors
                compose(config_name="config", overrides=["training.min_samples=invalid_string"])

    def test_hydra_allows_overrides(self, config_dir):
        """Test that we can override values via command line."""
        register_configs()
        
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config", overrides=["training.min_samples=999"])
            assert cfg.training.min_samples == 999

    def test_hydra_loads_all_structured_config_sections(self, config_dir):
        """Ensure all major sections are present."""
        register_configs()
        
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config")
            
            assert "training" in cfg
            assert "data" in cfg
            assert "pipeline" in cfg
            assert "model" in cfg
            assert "deployment" in cfg
            assert "validation" in cfg
            assert "spark" in cfg
            assert "acoustic" in cfg
            assert "download" in cfg
            assert "ground_truth" in cfg
            assert "setup_verification" in cfg
            assert "paths" in cfg
            assert "nemo" in cfg
