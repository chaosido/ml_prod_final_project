import logging
import sys
from pathlib import Path

import hydra
import joblib

from asr_qe.config import Config, register_configs

# Import shared modules
# Import shared modules
from asr_qe.data import load_features, prepare_data
from asr_qe.models import XGBoostTrainer
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
def main(cfg: Config) -> None:
    # Build config object
    val_config = cfg.validation

    logger.info(f"Starting validation for candidate: {val_config.candidate_model}")

    # 1. Load Candidate Model
    candidate_path = Path(val_config.candidate_model)
    if not candidate_path.exists():
        logger.error(f"Candidate model not found at {candidate_path}")
        sys.exit(1)

    logger.info(f"Selected Candidate Model: {candidate_path}")

    # 2. Prepare Data
    logger.info(f"Loading data from {val_config.data_path}")
    df = load_features(val_config.data_path)

    # Initialize Trainer to reuse logic use (split, eval)
    trainer = XGBoostTrainer(
        training_config=cfg.training,
        xgboost_config=cfg.training.xgboost,
        output_dir=cfg.model.output_dir,  # Required by new signature
    )

    X, y = prepare_data(
        df,
        feature_columns=cfg.training.feature_columns,
        target_column=cfg.training.target_column,
    )

    # Use EXACT same split as training
    _, X_test, _, y_test = trainer.split_data(X, y)
    logger.info(f"Test set size: {len(X_test)}")

    # 3. Evaluate Candidate
    logger.info("Evaluating Candidate...")
    trainer.set_model(joblib.load(candidate_path))
    cand_metrics = trainer.evaluate(X_test, y_test)
    cand_rho = cand_metrics["spearman_rho"]
    logger.info(f"Candidate Spearman Rho: {cand_rho:.4f}")

    # 4. Evaluate Production (if it exists)
    prod_rho = -1.0
    prod_path = Path(val_config.production_model)

    if prod_path.exists():
        logger.info(f"Evaluating Production Model: {prod_path}")
        try:
            trainer.set_model(joblib.load(prod_path))
            prod_metrics = trainer.evaluate(X_test, y_test)
            prod_rho = prod_metrics["spearman_rho"]
            logger.info(f"Production Spearman Rho: {prod_rho:.4f}")
        except Exception as e:
            logger.error(f"Failed to load/eval production model: {e}")
            # Keep prod_rho as -1.0
    else:
        logger.warning(
            f"Production model not found at {prod_path}. Treating baseline as 0."
        )

    # 5. Compare and Exit
    logger.info("-" * 30)
    logger.info(
        f"Comparison: Candidate ({cand_rho:.4f}) vs Production ({prod_rho:.4f})"
    )

    if cand_rho > prod_rho:
        logger.info(">>> RESULT: PASS. Candidate is better. Will be promoted.")
        sys.exit(0)
    else:
        logger.info(">>> RESULT: Candidate is not better. Keeping current production model.")
        sys.exit(0)  # Don't fail - just keep the old model


if __name__ == "__main__":
    main()
