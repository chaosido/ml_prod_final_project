import logging
import shutil
import sys
from pathlib import Path

import hydra

from asr_qe.config import Config, register_configs
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
    deploy_config = cfg.deployment

    logger.info(f"Starting deployment of {deploy_config.candidate_model}")

    # 1. Find Candidate
    source_model = Path(deploy_config.candidate_model)
    if not source_model.exists():
        logger.error(f"Candidate model not found: {source_model}")
        sys.exit(1)

    logger.info(f"Found candidate: {source_model}")

    # 2. Prepare Production Dir
    prod_dir = Path(deploy_config.production_dir)
    prod_dir.mkdir(parents=True, exist_ok=True)
    target_model = prod_dir / "model.joblib"

    # 3. Deploy (Copy)
    try:
        logger.info(f"Copying to {target_model}...")
        shutil.copy2(source_model, target_model)
        logger.info("Deployment successful!")

        # Optional: Archive/Tagging could go here

    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
