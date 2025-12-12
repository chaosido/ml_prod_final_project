"""
Model Training Job

This script loads extracted features from Parquet, trains an XGBoost model,
validates performance, and saves the model with versioning.

Usage:
    python jobs/train_model.py
    python jobs/train_model.py training.min_samples=500 data.input_path=/custom/path
"""
import logging
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asr_qe.models.trainer import XGBoostTrainer, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_features(input_path: str) -> pd.DataFrame:
    """Load features from parquet file(s)."""
    path = Path(input_path)
    
    if path.is_dir():
        df = pd.read_parquet(path)
    else:
        df = pd.read_parquet(path)
    
    logger.info(f"Loaded {len(df)} samples from {input_path}")
    return df


def prepare_data(df: pd.DataFrame, config: TrainingConfig):
    """Prepare feature matrix and target vector."""
    feature_cols = list(config.feature_columns)
    
    missing_cols = set(feature_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")
    
    if config.target_column not in df.columns:
        raise ValueError(f"Target column '{config.target_column}' not found in data")
    
    X = df[feature_cols].values
    y = df[config.target_column].values
    
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[valid_mask]
    y = y[valid_mask]
    
    logger.info(f"Prepared {len(X)} valid samples for training")
    return X, y


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    logger.info("Starting model training job")
    logger.info(f"Input: {cfg.data.input_path}")
    logger.info(f"Output: {cfg.model.output_dir}")
    
    # Load data
    df = load_features(cfg.data.input_path)
    
    # Configure training from Hydra config
    config = TrainingConfig(
        min_samples=cfg.training.min_samples,
        min_spearman_rho=cfg.training.min_spearman_rho,
        model_dir=cfg.model.output_dir,
    )
    
    # Prepare data
    X, y = prepare_data(df, config)
    
    # Train model with XGBoost params from config
    trainer = XGBoostTrainer(
        config=config,
        n_estimators=cfg.training.n_estimators,
        max_depth=cfg.training.max_depth,
        learning_rate=cfg.training.learning_rate,
    )
    result = trainer.train(X, y)
    
    if result.success:
        logger.info(f"Training successful!")
        logger.info(f"  Model saved to: {result.model_path}")
        logger.info(f"  Spearman Rho: {result.spearman_rho:.4f}")
        logger.info(f"  Samples used: {result.num_samples}")
        return 0
    else:
        logger.error(f"Training failed: {result.error_message}")
        return 1


if __name__ == "__main__":
    main()
