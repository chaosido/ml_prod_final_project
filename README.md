# ASR Quality Estimation System

An automated ML production pipeline for estimating the quality of Automatic Speech Recognition (ASR) transcripts. We attempt to predict the quality of a ASR transcript based on the audio file and the transcript, and by that we can potentially flag low quality transcripts for human review (outside of the scope of this project)

## Overview
This system provides an end-to-end workflow to:
1. Ingest audio data.
2. Store and manage data versioning.
3. Extract acoustic, linguistic, and signal features using **Apache Spark**.
4. Orchestrate workflows using **Apache Airflow**.
5. Train and deployment Quality Estimation (QE) models (XGBoost).
6. Serve predictions via a **FastAPI** service.
7. Monitor system health and performance with **Prometheus** and **Grafana**.

## Project Structure
- **`dags/`**: Airflow DAGs defining the orchestration logic (e.g., `qe_pipeline.py`).
- **`serving/`**: FastAPI application for serving model predictions.
- **`jobs/`**: Spark jobs for distributed feature extraction.
- **`scripts/`**: Utility scripts for setup, testing, and data management.
- **`tests/`**: Unit and Integration tests (run with `pytest`).
- **`conf/`**: Configuration files (managed via Hydra/OmegaConf).
- **`docker-compose.yml`**: Definition of all containerized services.

## Getting Started

### Prerequisites
- I build on Linux
- **Docker** and **Docker Compose**
- **uv**

### Installation
Install project dependencies with `uv`:
```bash
uv sync --extra research
uv sync --extra dev

```

## Running the End-to-End Pipeline
To run the full production pipeline simulation, use the provided E2E script. This script handles everything from spinning up Docker containers to verifying model deployment.

```bash
./scripts/test_pipeline_e2e.sh
```

**What this script does:**
1. Installs dependencies.
2. Downloads sample data (VoxPopuli).
3. specific verification data to the `incoming/` folder.
4. Starts all Docker services (Airflow, Spark, Postgres, API, Monitoring).
5. Waits for the Airflow DAG to trigger and process the data.
6. Verifies that features are extracted, models are trained, and services are healthy.

## Running Tests

```bash
pytest
```
or 
```bash
uv run --extra dev pytest
```
