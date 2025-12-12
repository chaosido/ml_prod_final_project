FROM apache/airflow:2.9.1-python3.11

USER root

# Install system dependencies (build-essential for compiling wheels if needed)
# We also install Java because we might need it for SparkSubmitOperator local mode testing, 
# although typically SparkSubmitOperator talks to a remote cluster.
# But let's keep it lean for now.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    openjdk-17-jre-headless \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

USER airflow
# Install Airflow providers (pinned to match base image version to prevent upgrade)
RUN pip install --no-cache-dir "apache-airflow==2.9.1" apache-airflow-providers-apache-spark

# Copy project files
COPY --chown=airflow:root pyproject.toml uv.lock /opt/airflow/
COPY --chown=airflow:root jobs/ /opt/airflow/jobs/
COPY --chown=airflow:root src/ /opt/airflow/src/

# Install specific project dependencies (manual fallback since we removed uv)
# We need to install the dependencies defined in pyproject.toml explicitly or rely on pip install . handling it.
# To be safe and fast, let's install the heavy ones first directly.
RUN pip install --no-cache-dir \
    "numpy<2.0" \
    librosa \
    soundfile \
    datasets==2.20.0 \
    xgboost \
    scipy \
    joblib \
    pandas \
    hydra-core \
    omegaconf \
    transformers \
    jiwer \
    tqdm \
    prometheus-fastapi-instrumentator>=7.1.0 \
    prometheus-client>=0.23.1

# Install our package
RUN pip install -e /opt/airflow
