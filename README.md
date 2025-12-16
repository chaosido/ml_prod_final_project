# ASR Quality Estimation System

An automated ML production pipeline for estimating the quality of Automatic Speech Recognition (ASR) transcripts. We predict the quality of ASR transcripts based on audio features and confidence scores, enabling flagging of low-quality transcripts for human review. Dataset used is VoxPopuli, only the dutch subset of ASR europarlament  transcripts, but its not super relevant to the final product.

## How It Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ASR Quality Estimation Pipeline                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────────┐    │
│   │  Audio   │───▶│   Spark     │───▶│   XGBoost   │───▶│   FastAPI    │    │
│   │  Files   │    │  Features   │    │   Model     │    │   Serving    │    │
│   └──────────┘    └─────────────┘    └─────────────┘    └──────────────┘    │
│        │                │                   │                   │           │
│        ▼                ▼                   ▼                   ▼           │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                        Airflow Orchestration                         │   │
│   │   File Sensor → Feature Extract → Train → Validate → Deploy          │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│        │                                                        │           │
│        ▼                                                        ▼           │
│   ┌──────────────┐                                    ┌──────────────────┐  │
│   │  Prometheus  │────────────────────────────────────│     Grafana      │  │
│   │   Metrics    │                                    │   Dashboards     │  │
│   └──────────────┘                                    └──────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Pipeline Steps:**
1. **Ingest**: Audio files land in `data/incoming/`
2. **Feature Extraction**: Spark extracts acoustic features (SNR, RMS, duration) + ASR confidence
3. **Training**: XGBoost model trained on features to predict WER
4. **Validation**: New model compared against production; promoted if better
5. **Serving**: FastAPI endpoint serves predictions in real-time
6. **Monitoring**: Prometheus collects metrics, Grafana visualizes

## Project Structure

```
.
├── conf/                  # Hydra/OmegaConf configuration
│   └── config.yaml        # Main config (paths, training params, etc.)
├── dags/                  # Airflow DAG definitions
│   └── qe_pipeline.py     # Main pipeline: ingest → train → deploy
├── docker/                # Dockerfiles for each service
│   ├── airflow/           # Airflow with Spark provider
│   ├── api/               # FastAPI serving container
│   └── spark/             # Spark with NeMo dependencies
├── jobs/                  # Standalone Python jobs
│   ├── feature_extract.py # Spark feature extraction
│   ├── train_model.py     # XGBoost training
│   └── validate_model.py  # Model validation
├── scripts/               # Utility scripts
│   ├── test_pipeline_e2e.sh  # Full E2E test
│   └── generate_traffic.sh   # API traffic generator
├── serving/               # FastAPI application
│   └── main.py            # Prediction endpoint
├── src/asr_qe/            # Core Python package
│   ├── features/          # Feature extractors (acoustic, ASR)
│   ├── models/            # XGBoost trainer
│   └── config.py          # Dataclass configs
├── tests/                 # Pytest test suite
├── docker-compose.yml     # All services orchestration
├── Makefile               # Common commands
└── pyproject.toml         # Python dependencies
```

## Getting Started

### Prerequisites
- **NVIDIA GPU** with CUDA support
- **Docker** with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- **Docker Compose** v2+
- **uv** ([Python package manager](https://github.com/astral-sh/uv))

### Quick Start
```bash
# Run the full E2E pipeline (downloads data, trains model, starts services)
./scripts/test_pipeline_e2e.sh
```
when docker is up you can test the dashboards in grafana using 
```bash
./scripts/generate_traffic.sh
```

## Available Commands

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make install` | Install all dependencies (`uv sync --all-extras`) |
| `make test` | Run pytest test suite |
| `make lint` | Run ruff linter with auto-fix |
| `make up` | Start all Docker services |
| `make down` | Stop all Docker services |
| `make e2e` | Run full E2E pipeline test |
| `make traffic` | Generate API traffic for Grafana |
| `make clean` | Full cleanup (Docker + data) |

## Services

| Service | URL | Credentials |
|---------|-----|-------------|
| Airflow | http://localhost:8081 | admin / admin |
| Spark UI | http://localhost:8080 | - |
| API Docs | http://localhost:8000/docs | - |
| Prometheus | http://localhost:9191 | - |
| Grafana | http://localhost:3000 | admin / admin, 2 dasbhoard set up for quality control and airflow  |

## Running Tests

```bash
# Using pytest directly (after make install)
pytest

# Or via uv
uv run --extra dev pytest
```
