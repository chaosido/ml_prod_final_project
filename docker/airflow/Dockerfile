FROM apache/airflow:2.9.1-python3.11

USER root

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
RUN pip install --no-cache-dir "apache-airflow==2.9.1" apache-airflow-providers-apache-spark "pyspark==3.5.0"

# Copy project files
COPY --chown=airflow:root pyproject.toml uv.lock /opt/airflow/
COPY --chown=airflow:root jobs/ /opt/airflow/jobs/
COPY --chown=airflow:root src/ /opt/airflow/src/
COPY --chown=airflow:root conf/ /opt/airflow/conf/

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
    prometheus-client>=0.23.1 \
    torch \
    "nemo_toolkit[asr]"

RUN pip install -e /opt/airflow
