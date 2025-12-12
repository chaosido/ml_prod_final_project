"""
Model Training Job

Orchestrates the training pipeline: load data → train model → save artifacts.
"""
import logging

import hydra
from omegaconf import DictConfig

from asr_qe.data import load_features, prepare_data
from asr_qe.models.trainer import XGBoostTrainer, TrainingConfig
from asr_qe.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> int:
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
    
    # Train model
    trainer = XGBoostTrainer(
        config=config,
        n_estimators=cfg.training.n_estimators,
        max_depth=cfg.training.max_depth,
        learning_rate=cfg.training.learning_rate,
    )
    result = trainer.train(X, y)
    
    if result.success:
        logger.info("Training successful!")
        logger.info(f"  Model saved to: {result.model_path}")
        logger.info(f"  Spearman Rho: {result.spearman_rho:.4f}")
        logger.info(f"  Samples used: {result.num_samples}")
        return 0
    else:
        logger.error(f"Training failed: {result.error_message}")
        return 1


if __name__ == "__main__":
    main()
