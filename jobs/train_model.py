import logging
import sys
from pathlib import Path

import hydra

from asr_qe.config import Config, register_configs
from asr_qe.data import load_features, prepare_data
from asr_qe.models.trainer import XGBoostTrainer
from asr_qe.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Register structured configs
register_configs()

# Determine config path: use Docker path if it exists, otherwise use local path
DOCKER_CONFIG_PATH = Path("/opt/airflow/conf")
LOCAL_CONFIG_PATH = Path(__file__).parent.parent / "conf"
CONFIG_PATH = str(DOCKER_CONFIG_PATH if DOCKER_CONFIG_PATH.exists() else LOCAL_CONFIG_PATH)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: Config) -> int:
    logger.info("Starting model training job")
    logger.info(f"Input: {cfg.data.input_path}")
    logger.info(f"Output: {cfg.model.output_dir}")

    # Load data
    df = load_features(cfg.data.input_path)

    # Prepare data
    X, y = prepare_data(
        df,
        feature_columns=cfg.training.feature_columns,
        target_column=cfg.training.target_column,
    )

    # Train model
    # cfg.training and cfg.training.xgboost are already typed objects populated by Hydra
    trainer = XGBoostTrainer(
        training_config=cfg.training,
        xgboost_config=cfg.training.xgboost,
        output_dir=cfg.model.output_dir,
    )
    result = trainer.train(X, y)

    logger.info("Training successful!")
    logger.info(f"  Model saved to: {result.model_path}")
    logger.info(f"  Spearman Rho: {result.spearman_rho:.4f}")
    logger.info(f"  Samples used: {result.num_samples}")

    # Explicitly print the model path to stdout for Airflow XCom to pick up
    print(result.model_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
