# ASR Quality Estimation: Technical Specification & Implementation Plan

**Target Audience:** ML Engineers / Data Engineers / DevOps
**Objective:** Build a Production-Ready, Reference-Free ASR Quality Estimation System.
**Course Compliance:** Fully aligned with UvA ML Engineering curriculum (Weeks 1–6).

## 1. System Requirements

### 1.1 Functional Requirements
*   **Offline Training:** System must process the **facebook/voxpopuli** (Dutch) dataset (Audio + Intent/Text) to learn the correlation between Audio Features and WER (calculated vs ground truth).
*   **Online Serving:** System must expose an HTTP API that accepts audio and returns a WER prediction and a "Review Recommended" flag.
*   **Drift Detection:** System must statistically compare incoming production audio against training baselines (Audio Features + **Predicted WER**).

### 1.2 Non-Functional Requirements (Constraints)
*   **Latency:** API P95 latency < 500ms for 10s audio clips.
*   **Throughput:** Spark Batch job must process daily increments (approx. 50GB) within 2 hours.
*   **Reliability:** The pipeline must not crash on corrupt inputs (0-byte files, non-WAV headers).
*   **Test Coverage:** Strict >80% code coverage on `src/` logic.
*   **Testing Strategy (FIRST Principles):** All tests must be:
    *   **F**ast: Unit tests < 100ms.
    *   **I**solated/Independent: No shared state between tests.
    *   **R**epeatable: Deterministic results (seed random generators).
    *   **S**elf-Validating: Pass/Fail boolean output.
    *   **T**imely: Written alongside or before code.
    *   *Infrastructure Verification:* Docker builds must be verified with smoke tests (import checks) before deployment.

### 1.3 Tech Stack (Strict)
*   **Language:** Python 3.10+
*   **Distributed Engine:** PySpark 3.5.0 (No Pandas for data processing).
*   **Serving:** FastAPI + Uvicorn.
*   **Orchestration:** Airflow (Dockerized).
*   **Observability:** Prometheus + Grafana.

---

## 2. Software Architecture & Design Patterns

### 2.1 The Shared Core (`src` Package)
**Pattern:** *Shared Library / Code Reuse*
To prevent **Training-Serving Skew** (where the logic in production differs slightly from training), we implement a core Python package.

*   **Implementation:** A `pyproject.toml` based package.
*   **Usage:**
    *   **Spark:** Zipped and sent to workers via `--py-files` or installed in the Docker image.
    *   **API:** Installed via `pip install .` in the Dockerfile.

### 2.2 Feature Extraction Strategy
**Pattern:** *Strategy Pattern*
We define a standard interface for extracting features so algorithms can be hot-swapped.

```python
# src/features/interface.py
class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        pass
```

### 2.3 Model Loading
**Pattern:** *Singleton Pattern*
XGBoost models are heavy. Loading them inside the request handler kills performance.
*   **Implementation:** Use Python's `functools.lru_cache` or FastAPI's `Lifespan` events to load the model **once** at container startup.

---

## 3. Directory Structure & File Manifest

This structure separates concerns (Data vs. Logic vs. Ops).

```text
/asr-qe-system
├── pyproject.toml               # Define `src` package and dependencies
├── .pre-commit-config.yaml      # Gates: Ruff, Black, Mypy
├── docker-compose.yml           # Orchestration of all services
├── README.md
│
├── config/                      # DTAP Configuration
│   ├── dev.yaml
│   └── prod.yaml
│
├── src/                         # SHARED CORE (The "Business Logic")
│   ├── __init__.py
│   ├── config.py                # Pydantic Settings (Dependency Injection)
│   ├── validation.py            # Audio file integrity checks
│   ├── features/
│   │   ├── interface.py         # Abstract Base Class
│   │   └── acoustic.py          # Concrete implementation (SNR, RMS)
│   ├── models/
│   │   └── loader.py            # Singleton Model Loader
│   └── utils/
│       └── resiliency.py        # Error handling decorators
│
├── jobs/                        # OFFLINE (PySpark)
│   ├── spark_extract.py         # Main Driver Script
│   └── train_model.py           # Training Script
│
├── serving/                     # ONLINE (FastAPI)
│   └── main.py                  # API Entrypoint
│
├── dags/                        # ORCHESTRATION (Airflow)
│   └── qe_pipeline_dag.py
│
├── docker/                      # INFRASTRUCTURE
│   ├── spark/
│   │   └── Dockerfile           # Multi-stage build
│   ├── api/
│   │   └── Dockerfile           # Multi-stage build
│   └── monitoring/
│       ├── prometheus.yml
│       └── grafana_dashboards/
│
└── tests/                       # QUALITY ASSURANCE
    ├── unit/                    # Test logic in isolation
    └── integration/             # Test containers/services
```

---

## 4. Phase-by-Step Implementation Guide

### Phase 1: Core Logic & Quality Gates
**Goal:** Build the math layer with clean coding practices (Week 2).

1.  **Configure Quality Tools:**
    *   Setup `ruff` (Linter) to ban `print` statements (force logging).
    *   Setup `mypy` to enforce strict typing (`def func(x: int) -> float:`).
2.  **Develop `src/features/acoustic.py`:**
    *   Implement Signal-to-Noise Ratio (SNR).
    *   Implement RMS Energy.
    *   **Constraint:** Use only `numpy` and `librosa`. No `pandas`.
3.  **Develop `src/validation.py`:**
    *   Implement checks: `file_size > 1KB`, `duration > 0.5s`.
4.  **Unit Tests:**
    *   Test `extract()` with a generated Silent Array (expect SNR=0).
    *   Test `extract()` with White Noise (expect specific SNR).

### Phase 2: Docker Infrastructure
**Goal:** reproducible environments using Multi-Stage Builds (Week 4).

1.  **Spark Dockerfile:**
    *   **Stage 1 (Builder):** Install `gcc`, build wheels for dependencies.
    *   **Stage 2 (Runtime):** Copy wheels, install `libsndfile1`, install `src`.
    *   *Why?* Reduces image size and security surface.
2.  **API Dockerfile:**
    *   Similar multi-stage approach. Ensure `uvicorn` and `fastapi` are present.
3.  **Docker Compose:**
    *   Define the `shared-network`.
    *   Mount `./data` volume to `/opt/data` in all containers to simulate a shared Data Lake.

### Phase 3: The Distributed Batch Job
**Goal:** Fault-tolerant Data Processing (Week 3).

1.  **Script `jobs/spark_extract.py`:**
    *   **Read:** `spark.read.format("binaryFile")`.
    *   **Partitions:** `repartition(100)` (Tune based on file count to avoid small files).
    *   **UDF Wrapper:**
        ```python
        @udf(returnType=ArrayType(FloatType()))
        def safe_extract(audio_bytes):
            try:
                # Call pure python logic
                return src.features.acoustic.extract(audio_bytes)
            except Exception:
                return None # Soft failure
        ```
    *   **Write:** `df.write.partitionBy("date").mode("append").parquet(...)`.
    *   **Optimization:** Use `broadcast` join if checking against a small list of already processed files.

### Phase 4: Model Training Loop
**Goal:** Train and Version the Model.

1.  **Script `jobs/train_model.py`:**
    *   Load Parquet files.
    *   **Validation:** Assert `df.count() > 1000` before training (prevent training on empty batches).
    *   Train `XGBRegressor`.
    *   **Metric:** Calculate Spearman Rho.
    *   **Threshold:** If Rho < 0.4, raise Exception (fail the Airflow task).
    *   **Artifacts:** Save to `models/v{timestamp}/model.joblib`.

### Phase 5: Real-Time Serving API
**Goal:** Low-latency serving (Week 5).

1.  **Code `serving/main.py`:**
    *   **Startup:** Load `model.joblib`.
    *   **Endpoint:**
        ```python
        @app.post("/predict")
        async def predict(file: UploadFile):
            audio = await load_audio(file)
            if not validate(audio): raise HTTPException(400)
            feats = extract(audio)
            return model.predict(feats)
        ```
2.  **Concurrency:** Use `async def` for the controller, but run the CPU-bound `extract` logic in a `ThreadPoolExecutor` (FastAPI handles this if you declare the function as `def` instead of `async def` for blocking code, or explicitly offload it).

### Phase 6: Telemetry & Drift
**Goal:** Proactive Monitoring (Week 6).

1.  **Metrics Design:**
    *   **System:** `process_resident_memory_bytes` (Leak detection).
    *   **Data (Drift):** `histogram_input_rms_db`.
    *   **Business:** `counter_low_quality_transcripts`.
2.  **Prometheus Configuration:**
    *   Scrape API on port 8000 path `/metrics`.
    *   Scrape PushGateway on port 9091 (for Spark job metrics).
3.  **AlertManager Rules:**
    *   `alert: HighErrorRate` -> `expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05`.

### Phase 7: Orchestration & Automation
**Goal:** Deployment Patterns (Week 5).

1.  **Airflow DAG:**
    *   **Task 1:** `FileSensor` (Poll `/data/incoming`).
    *   **Task 2:** `SparkSubmitOperator` (Run Phase 3).
    *   **Task 3:** `PythonOperator` (Run Phase 4).
    *   **Task 4:** `BashOperator` (Move trained model to `models/production`).
2.  **CI/CD Simulation:**
    *   The DAG should pick up the code from the `src` package. In a real deployment, this would trigger a rebuild of the Docker images.

---

## 5. Engineering Checklist (The "Handover")

Engineers must verify these points before marking tickets "Done":

1.  **Idempotency:** Can I run the Spark job 3 times in a row without creating duplicate feature rows in Parquet?
2.  **Null Handling:** Does the pipeline survive if I feed it a `.txt` file renamed as `.wav`?
3.  **Metric Visibility:** Do I see the "Drift" histogram update in Grafana immediately after sending a request to the API?
4.  **Config Isolation:** Can I change the Model Threshold from `0.2` to `0.4` just by changing `prod.yaml`, without touching code?
5.  **Clean Code:** Does `ruff` pass without any ignores?

This plan restores the depth of the course concepts while keeping the steps actionable. It is ready for your engineers.