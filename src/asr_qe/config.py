import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from hydra.core.config_store import ConfigStore


@dataclass
class AcousticConfig:
    """Configuration for Acoustic Feature Extraction."""
    frame_length: int = 2048
    hop_length: int = 512
    noise_ratio: float = 0.1
    epsilon: float = 1e-9
    silence_db: float = -100.0


@dataclass
class XGBoostConfig:
    """Configuration for XGBoost model hyperparameters."""
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    random_state: int = 42


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    min_samples: int = 1000
    min_spearman_rho: float = 0.4
    test_size: float = 0.2
    feature_columns: List[str] = field(
        default_factory=lambda: ["rms_db", "snr_db", "duration", "asr_confidence"]
    )
    target_column: str = "wer"
    random_state: int = 42
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)


@dataclass
class DataConfig:
    """Data input/output paths."""
    input_path: str = "/opt/data/features.parquet"
    output_path: str = "/opt/data/processed"


@dataclass
class DeploymentConfig:
    """Configuration for model deployment."""
    candidate_dir: str = "models/staging"
    production_dir: str = "models/production"
    output_path: str = "/opt/data/processed"
    candidate_model: str = "models/staging/latest/model.joblib"


@dataclass
class ValidationConfig:
    """Configuration for model validation."""
    candidate_model: str = "models/staging/latest/model.joblib"
    production_model: str = "models/production/model.joblib"
    data_path: str = "/opt/data/features/history.parquet"


@dataclass
class PipelineConfig:
    """Pipeline infrastructure paths."""
    incoming_data: str = "/opt/data/incoming"
    feature_history: str = "/opt/data/features/history.parquet"
    staging_models: str = "/models/staging"
    production_models: str = "/models/production"


@dataclass
class ModelConfig:
    """Model output configuration."""
    output_dir: str = "models"


@dataclass
class SparkConfig:
    """Spark job configuration."""
    app_name: str = "ASR-QE-FeatureExtraction"
    partitions_multiplier: int = 3


@dataclass
class Config:
    """Top-level configuration."""
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    spark: SparkConfig = field(default_factory=SparkConfig)
    acoustic: AcousticConfig = field(default_factory=AcousticConfig)


def register_configs():
    """Register configuration with Hydra ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
