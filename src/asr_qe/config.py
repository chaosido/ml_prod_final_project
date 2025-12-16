from dataclasses import dataclass, field
from typing import List, Optional

from hydra.core.config_store import ConfigStore

# hydra overwrites all configs but these are the constraints/placeholders
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

    min_samples: int = 50 
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
    """Data input/output paths.
    
    """

    input_path: str = "/opt/data/features/history.parquet"
    output_path: str = "/opt/data/features/history.parquet"  # Defaults to feature_history
    reference_lookup_path: str = "/opt/data/reference_lookup.json"


@dataclass
class PipelineConfig:
    """Pipeline infrastructure paths and settings.
    

    """

    incoming_data: str = "/opt/data/incoming"
    feature_history: str = "/opt/data/features/history.parquet"
    staging_models: str = "/models/staging"
    production_models: str = "/models/production"
    min_files_threshold: int = 50  # Minimum number of files in incoming/ we allow the pipeline to run 
    sensor_poke_interval: int = 60  # Seconds between sensor checks
    sensor_timeout: int = 3600  # Sensor timeout


@dataclass
class ModelConfig:
    """Model output configuration.
    
    Base directory for model storage. Specific paths are in PipelineConfig.
    """

    output_dir: str = "/models"


@dataclass
class DeploymentConfig:
    """Configuration for model deployment.
    
    """

    candidate_dir: str = "/models/staging"
    production_dir: str = "/models/production"
    candidate_model: str = "/models/staging/latest/model.joblib"


@dataclass
class ValidationConfig:
    """Configuration for model validation."""

    candidate_model: str = "/models/staging/latest/model.joblib"
    production_model: str = "/models/production/model.joblib"
    data_path: str = "/opt/data/features/history.parquet"


@dataclass
class SparkConfig:
    """Spark job configuration."""

    app_name: str = "ASR-QE-FeatureExtraction"
    master: str = "spark://spark-master:7077"
    partitions_multiplier: int = 3


@dataclass
class DownloadConfig:
    """Dataset download configuration."""

    output_dir: str = "/opt/data/voxpopuli_nl"
    split: str = "train"
    max_samples: int | None = None


@dataclass
class GroundTruthConfig:
    """Ground truth generation configuration."""

    manifest_path: str = "/opt/data/voxpopuli_nl/manifest_train.csv"
    output_path: str = "/opt/data/ground_truth.parquet"
    model_name: str = "nvidia/parakeet-tdt-0.6b-v3"
    batch_size: int = 16
    enable_confidence: bool = True


@dataclass
class SetupVerificationConfig:
    """Configuration for setting up verification data."""

    manifest_path: str = "data/voxpopuli_nl/manifest_train.csv"
    incoming_dir: str = "data/incoming"
    lookup_path: str = "data/reference_lookup.json"
    sample_size: int = 50
    random_state: int = 42


@dataclass
class NeMoConfig:
    """Configuration for NeMo model cache."""

    cache_dir: str = "/home/jesse-wonnink/Data/ml_store/models/nemo"


@dataclass
class PathsConfig:
    """Paths configuration."""

    data_dir: str = "/opt/data"
    models_dir: str = "/models"
    voxpopuli_dir: str = "${paths.data_dir}/voxpopuli_nl"
    manifest_path: str = "${paths.voxpopuli_dir}/manifest_train.csv"
    incoming_dir: str = "${paths.data_dir}/incoming"
    archive_dir: str = "${paths.data_dir}/archive"
    feature_history: str = "${paths.data_dir}/features/history.parquet"
    reference_lookup: str = "${paths.data_dir}/reference_lookup.json"
    staging_models: str = "${paths.models_dir}/staging"
    production_models: str = "${paths.models_dir}/production"



@dataclass
class Config:
    """Top-level configuration.
    
    This structured config provides type-safe access to configuration.
    All values are loaded from config.yaml via Hydra, with defaults
    matching the YAML file.
    """

    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    spark: SparkConfig = field(default_factory=SparkConfig)
    acoustic: AcousticConfig = field(default_factory=AcousticConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    ground_truth: GroundTruthConfig = field(default_factory=GroundTruthConfig)
    setup_verification: SetupVerificationConfig = field(
        default_factory=SetupVerificationConfig
    )
    paths: PathsConfig = field(default_factory=PathsConfig)
    nemo: NeMoConfig = field(default_factory=NeMoConfig)


def register_configs():
    """Register configuration with Hydra ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
