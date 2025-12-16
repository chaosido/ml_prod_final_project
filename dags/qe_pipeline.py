import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.sensors.python import PythonSensor
from airflow.utils.dates import days_ago
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

# Load Configuration (Single Source of Truth)
# We assume the DAG is running in the container where ./conf is mapped or copied
CONFIG_PATH = (
    "/opt/airflow/conf/config.yaml"
    if os.path.exists("/opt/airflow/conf/config.yaml")
    else "conf/config.yaml"
)
conf = OmegaConf.load(CONFIG_PATH)

# Define paths from config
INCOMING_DATA = conf.pipeline.incoming_data
HISTORY_DATA = conf.pipeline.feature_history
STAGING_MODELS = conf.pipeline.staging_models
PROD_MODELS = conf.pipeline.production_models

# Sensor configuration from config
MIN_FILES_THRESHOLD = conf.pipeline.get("min_files_threshold", 50)
SENSOR_POKE_INTERVAL = conf.pipeline.get("sensor_poke_interval", 60)
SENSOR_TIMEOUT = conf.pipeline.get("sensor_timeout", 3600)

# Define the DAG arguments
default_args = {
    "owner": "ml_engineer",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "asr_qe_pipeline",
    default_args=default_args,
    description="Cumulative Retraining Pipeline for ASR QE",
    # The sensor will check if 50+ files exist, and only proceed if threshold is met
    # Run daily in production (or manually triggered)
    schedule_interval="@daily",
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,  # Prevent overlapping runs
    is_paused_upon_creation=False,  # Start unpaused by default
    tags=["mlops", "asr-qe"],
) as dag:

    # TASK 1: Sensor
    # Watch for minimum number of new files in /opt/data/incoming
    def check_minimum_files(**context: Dict[str, Any]) -> bool:
        """Check if there are at least MIN_FILES_THRESHOLD .wav files in incoming directory.
        
        Args:
            **context: Airflow task context (unused but required by PythonSensor).
            
        Returns:
            True if file count >= threshold, False otherwise.
        """
        incoming_path = Path(INCOMING_DATA)
        
        if not incoming_path.exists():
            logger.warning(f"Incoming directory does not exist: {INCOMING_DATA}")
            return False
        
        try:
            wav_files = list(incoming_path.glob("*.wav"))
            file_count = len(wav_files)
            
            logger.info(
                f"Found {file_count} .wav files in {INCOMING_DATA} "
                f"(threshold: {MIN_FILES_THRESHOLD})"
            )
            
            return file_count >= MIN_FILES_THRESHOLD
        except Exception as e:
            logger.error(f"Error checking files in {INCOMING_DATA}: {e}")
            return False

    wait_for_data = PythonSensor(
        task_id="wait_for_incoming_data",
        python_callable=check_minimum_files,
        poke_interval=SENSOR_POKE_INTERVAL,
        timeout=SENSOR_TIMEOUT,
        mode="poke",
    )

    # TASK 2: Ingest (Spark)
    ingest_features = SparkSubmitOperator(
        task_id="ingest_data",
        application="/opt/airflow/jobs/feature_extract.py",
        conn_id="spark_default",  # Used for connection info, but master URL is set in Python code
        executor_memory="3g",  # Increased for ASR model
        driver_memory="2g",
        # Pass keys to override defaults in config.yaml
        application_args=[
            f"data.input_path={INCOMING_DATA}",
            f"data.output_path={HISTORY_DATA}",
        ],
        verbose=True,
    )

    # TASK 2.5: Archive Processed Files
    # Move processed files to archive to prevent re-triggering
    archive_processed_files = BashOperator(
        task_id="archive_processed_files",
        bash_command=(
            f"mkdir -p /opt/data/archive && "
            f"if ls {INCOMING_DATA}/*.wav 1> /dev/null 2>&1; then "
            f"  mv {INCOMING_DATA}/*.wav /opt/data/archive/; "
            f"  echo 'Archived files from {INCOMING_DATA} to /opt/data/archive'; "
            f"else "
            f"  echo 'No .wav files to archive in {INCOMING_DATA}'; "
            f"fi"
        ),
    )

    # TASK 3: Train (Bash -> Python + Hydra)
    # Writes to staging
    train_model = BashOperator(
        task_id="train_model",
        bash_command=f"python /opt/airflow/jobs/train_model.py model.output_dir={STAGING_MODELS} data.input_path={HISTORY_DATA}",  # noqa: E501
        env={"PYTHONPATH": "/opt/airflow/src"},
        do_xcom_push=True,
    )

    # TASK 4: Validate
    # Compares staging vs production
    validate_model = BashOperator(
        task_id="validate_model",
        bash_command=f'python /opt/airflow/jobs/validate_model.py validation.candidate_model={{{{ ti.xcom_pull(task_ids="train_model") }}}} validation.production_model={PROD_MODELS}/model.joblib validation.data_path={HISTORY_DATA}',  # noqa: E501
        env={"PYTHONPATH": "/opt/airflow/src"},
    )

    # TASK 5: Deploy
    # Moves the staging folder to production
    deploy_model = BashOperator(
        task_id="deploy_model",
        bash_command=f'python /opt/airflow/jobs/deploy_model.py deployment.candidate_model={{{{ ti.xcom_pull(task_ids="train_model") }}}} deployment.production_dir={PROD_MODELS}',  # noqa: E501
        env={"PYTHONPATH": "/opt/airflow/src"},
    )

    (
        wait_for_data
        >> ingest_features
        >> archive_processed_files
        >> train_model
        >> validate_model
        >> deploy_model
    )
