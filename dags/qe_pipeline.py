from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import os
from omegaconf import OmegaConf

# Load Configuration (Single Source of Truth)
# We assume the DAG is running in the container where ./conf is mapped or copied
CONFIG_PATH = '/opt/airflow/conf/config.yaml' if os.path.exists('/opt/airflow/conf/config.yaml') else 'conf/config.yaml'
conf = OmegaConf.load(CONFIG_PATH)

# Define paths from config
INCOMING_DATA = conf.pipeline.incoming_data
HISTORY_DATA = conf.pipeline.feature_history
STAGING_MODELS = conf.pipeline.staging_models
PROD_MODELS = conf.pipeline.production_models

# Define the DAG arguments
default_args = {
    'owner': 'ml_engineer',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'asr_qe_pipeline',
    default_args=default_args,
    description='Cumulative Retraining Pipeline for ASR QE',
    schedule_interval='@daily',
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 'asr-qe'],
) as dag:

    # TASK 1: Sensor
    # Watch for new files in /opt/data/incoming
    wait_for_data = FileSensor(
        task_id='wait_for_incoming_data',
        filepath=f"{INCOMING_DATA}/*.wav",  # Glob pattern to check for wav files
        fs_conn_id='fs_default',
        poke_interval=60,
        mode='poke',
        timeout=60 * 60,
    )

    # TASK 2: Ingest (Spark)
    # Uses Hydra args (key=value) for feature_extract.py
    ingest_features = SparkSubmitOperator(
        task_id='ingest_data',
        application='/opt/airflow/jobs/feature_extract.py',
        conn_id='spark_default',
        conf={'spark.master': 'spark://spark-master:7077'},
        # Pass keys to override defaults in config.yaml
        application_args=[
            f'data.input_path={INCOMING_DATA}',
            f'data.output_path={HISTORY_DATA}'
        ],
        verbose=True
    )

    # TASK 3: Train (Bash -> Python + Hydra)
    # Writes to staging
    train_model = BashOperator(
        task_id='train_model',
        bash_command=f'python /opt/airflow/jobs/train_model.py model.output_dir={STAGING_MODELS} data.input_path={HISTORY_DATA}',
        env={'PYTHONPATH': '/opt/airflow/src'},
        do_xcom_push=True
    )

    # TASK 4: Validate
    # Compares staging vs production
    validate_model = BashOperator(
        task_id='validate_model',
        # Hydra validation config overrides
        # We use XCom to get the exact path of the model trained in the previous step
        bash_command=f'python /opt/airflow/jobs/validate_model.py validation.candidate_model={{{{ ti.xcom_pull(task_ids="train_model") }}}} validation.production_model={PROD_MODELS}/model.joblib validation.data_path={HISTORY_DATA}',
        env={'PYTHONPATH': '/opt/airflow/src'}
    )

    # TASK 5: Deploy
    # Moves the staging folder (or specific model) to production
    deploy_model = BashOperator(
        task_id='deploy_model',
        bash_command=f'python /opt/airflow/jobs/deploy_model.py deployment.candidate_model={{{{ ti.xcom_pull(task_ids="train_model") }}}} deployment.production_dir={PROD_MODELS}',
        env={'PYTHONPATH': '/opt/airflow/src'}
    )

    wait_for_data >> ingest_features >> train_model >> validate_model >> deploy_model
