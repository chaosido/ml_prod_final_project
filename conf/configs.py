from dataclasses import dataclass
from typing import Tuple


@dataclass
class AcousticConfig:
    """Configuration for Acoustic Feature Extraction."""
    frame_length: int = 2048
    hop_length: int = 512
    noise_ratio: float = 0.1
    epsilon: float = 1e-9
    silence_db: float = -100.0


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    min_samples: int = 1000
    min_spearman_rho: float = 0.4
    model_dir: str = "models"
    feature_columns: Tuple[str, ...] = ("rms_db", "snr_db", "duration")
    target_column: str = "wer"
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class XGBoostConfig:
    """Configuration for XGBoost model."""
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    random_state: int = 42