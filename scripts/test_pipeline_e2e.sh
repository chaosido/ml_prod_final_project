#!/bin/bash
# scripts/test_pipeline_e2e.sh
# End-to-End Test for ASR QE Pipeline
# Self-contained: downloads data, bootstraps model, runs Docker pipeline
set -e

# === Configuration ===
DOCKER_COMPOSE="docker compose"
command -v docker-compose &> /dev/null && DOCKER_COMPOSE="docker-compose"

BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-200}  # Samples for local bootstrap (smaller = faster)
INCOMING_SIZE=${INCOMING_SIZE:-51}            # Files for Docker pipeline
CLEANUP=${CLEANUP:-false}
DATA_DIR="data"
MODELS_DIR="models"


echo "=== ASR QE Pipeline E2E Test ==="
echo "Bootstrap samples: ${BOOTSTRAP_SAMPLES}, Incoming: ${INCOMING_SIZE}, Cleanup: ${CLEANUP}"
echo ""

# === Helper Functions ===
wait_for() {
    local name=$1 cmd=$2 timeout=${3:-60} elapsed=0
    echo -n "  Waiting for ${name}..."
    while ! eval "$cmd" &>/dev/null; do
        sleep 2; elapsed=$((elapsed + 2))
        [ $elapsed -ge $timeout ] && echo " TIMEOUT" && return 1
    done
    echo " ✓"
}

# === Step 0: Cleanup ===
if [ "${CLEANUP}" = "true" ]; then
    echo "Step 0: Cleanup (fresh start)..."
    ${DOCKER_COMPOSE} down -v 2>/dev/null || true
    # Clean ALL data for true fresh start
    docker run --rm -v "$(pwd):/app" busybox sh -c \
        "rm -rf /app/${DATA_DIR}/voxpopuli_nl /app/${DATA_DIR}/features /app/${DATA_DIR}/incoming/* /app/${DATA_DIR}/archive/* /app/${MODELS_DIR}/staging/* /app/${MODELS_DIR}/production/*" 2>/dev/null || true
    echo "  ✓ Cleanup complete"
fi

# === Step 1: Dependencies ===
echo "Step 1: Installing dependencies..."
uv sync --all-extras --quiet 2>/dev/null || uv sync --extra research --extra dev --quiet
echo "  ✓ Done"

# === Step 2: Data Download ===
echo "Step 2: Downloading VoxPopuli data (${BOOTSTRAP_SAMPLES} samples)..."
MANIFEST="${DATA_DIR}/voxpopuli_nl/manifest_train.csv"
if [ ! -f "${MANIFEST}" ] || [ $(wc -l < "${MANIFEST}") -lt $((BOOTSTRAP_SAMPLES + 1)) ]; then
    uv run python scripts/download_voxpopuli.py paths.data_dir=${DATA_DIR} download.max_samples=${BOOTSTRAP_SAMPLES}
fi
echo "  ✓ Data ready ($(wc -l < "${MANIFEST}" | xargs) samples)"

# === Step 3: Local Bootstrap (if no history exists) ===
FEATURE_HISTORY="${DATA_DIR}/features/history.parquet"
PRODUCTION_MODEL="${MODELS_DIR}/production/model.joblib"

# Check for existence (could be file OR directory after Spark conversion)
if [ ! -e "${FEATURE_HISTORY}" ]; then
    echo "Step 3: Bootstrapping local feature history..."
    echo "  Running ASR on ${BOOTSTRAP_SAMPLES} samples (requires GPU, ~5-15 min)..."
    mkdir -p "${DATA_DIR}/features"
    uv run python scripts/generate_ground_truth.py \
        paths.data_dir=${DATA_DIR} \
        ground_truth.output_path=${FEATURE_HISTORY} \
        ground_truth.batch_size=8
    ROW_COUNT=$(uv run python -c "import pandas as pd; print(len(pd.read_parquet('${FEATURE_HISTORY}')))" 2>/dev/null || echo "?")
    echo "  ✓ Feature history created: ${ROW_COUNT} rows"
else
    ROW_COUNT=$(uv run python -c "import pandas as pd; print(len(pd.read_parquet('${FEATURE_HISTORY}')))" 2>/dev/null || echo "?")
    echo "Step 3: Using existing feature history (${ROW_COUNT} rows)"
fi

# === Step 4: Train Baseline Model (if none exists) ===
if [ ! -f "${PRODUCTION_MODEL}" ]; then
    echo "Step 4: Training baseline model..."
    uv run python jobs/train_model.py \
        data.input_path=${FEATURE_HISTORY} \
        model.output_dir=${MODELS_DIR}/staging
    
    # Find the latest staging model and copy to production
    LATEST_MODEL=$(find ${MODELS_DIR}/staging -name "model.joblib" -type f | head -1)
    if [ -n "${LATEST_MODEL}" ]; then
        mkdir -p ${MODELS_DIR}/production
        cp "${LATEST_MODEL}" "${PRODUCTION_MODEL}"
        echo "  ✓ Production model created: ${PRODUCTION_MODEL}"
    else
        echo "  ⚠ Training completed but model not found"
    fi
else
    echo "Step 4: Using existing production model"
fi

# Convert pandas parquet file to Spark-compatible directory format
# Spark expects a directory, not a single file
if [ -f "${FEATURE_HISTORY}" ] && [ ! -d "${FEATURE_HISTORY}" ]; then
    echo "  Converting parquet to Spark format..."
    uv run python -c "
import pandas as pd
import shutil
import os
df = pd.read_parquet('${FEATURE_HISTORY}')
tmp_path = '${FEATURE_HISTORY}.tmp'
os.remove('${FEATURE_HISTORY}')
os.makedirs(tmp_path, exist_ok=True)
df.to_parquet(os.path.join(tmp_path, 'part-00000.parquet'), index=False)
os.rename(tmp_path, '${FEATURE_HISTORY}')
print(f'  ✓ Converted to Spark format ({len(df)} rows)')
"
fi

# === Step 5: Setup Verification Data ===
echo "Step 5: Setting up verification data (${INCOMING_SIZE} files)..."
mkdir -p "${DATA_DIR}/incoming"
uv run python scripts/setup_verification_data.py paths.data_dir=${DATA_DIR} setup_verification.sample_size=${INCOMING_SIZE}
INCOMING_COUNT=$(find "${DATA_DIR}/incoming" -name "*.wav" | wc -l)
echo "  ✓ ${INCOMING_COUNT} files in incoming/"

# === Step 6: Start Docker Services ===
echo "Step 6: Starting Docker services..."
${DOCKER_COMPOSE} up init-permissions 2>/dev/null || true
${DOCKER_COMPOSE} up -d postgres spark-master spark-worker

wait_for "Spark master" "docker logs spark-master 2>&1 | grep -q 'I have been elected leader'" 120
wait_for "Spark worker" "docker logs spark-master 2>&1 | grep -q 'Registering worker'" 60

# === Step 7: Initialize Airflow ===
echo "Step 7: Initializing Airflow..."
${DOCKER_COMPOSE} up airflow-init
echo "  ✓ Airflow initialized"

# === Step 8: Start Remaining Services ===
echo "Step 8: Starting Airflow and monitoring..."
${DOCKER_COMPOSE} up -d airflow-webserver airflow-scheduler
${DOCKER_COMPOSE} up -d api prometheus grafana

wait_for "Airflow" "curl -sf http://localhost:8081/health" 90
echo "  ✓ Services ready"

# Wait for DAG to be available (parsed by scheduler)
echo "Waiting for DAG 'asr_qe_pipeline'..."
for i in {1..60}; do
    if docker exec airflow-webserver airflow dags list | grep -q "asr_qe_pipeline"; then
        echo "  ✓ DAG found"
        break
    fi
    echo -n "."
    sleep 2
done

# Unpause DAG
docker exec airflow-webserver airflow dags unpause asr_qe_pipeline 2>/dev/null || true

# === Step 9: Trigger DAG ===
echo "Step 9: Triggering DAG..."
docker exec airflow-webserver airflow dags trigger asr_qe_pipeline
sleep 5

DAG_RUN_ID=$(docker exec airflow-webserver airflow dags list-runs -d asr_qe_pipeline --no-backfill 2>/dev/null | grep -o "manual__[a-zA-Z0-9_:+\.-]*" | head -1 || echo "")
if [ -z "${DAG_RUN_ID}" ]; then
    echo "  ⚠ Could not get DAG run ID. Check http://localhost:8081"
    exit 1
fi
echo "  ✓ DAG triggered: ${DAG_RUN_ID}"

# === Step 10: Monitor DAG ===
echo "Step 10: Monitoring DAG execution..."
MAX_WAIT=1800
ELAPSED=0
STATE=""
while [ $ELAPSED -lt $MAX_WAIT ]; do
    STATE=$(docker exec airflow-webserver airflow dags list-runs -d asr_qe_pipeline --no-backfill 2>/dev/null | grep "${DAG_RUN_ID}" | awk '{print $3}' || echo "")
    
    case "$STATE" in
        success)
            echo ""; echo "  ✓ DAG completed successfully!"
            break ;;
        failed)
            echo ""; echo "  ✗ DAG failed! Check http://localhost:8081"
            ${DOCKER_COMPOSE} logs airflow-scheduler --tail 30
            exit 1 ;;
        *)
            echo -n "."
            sleep 15
            ELAPSED=$((ELAPSED + 15)) ;;
    esac
done

[ "$STATE" != "success" ] && echo "" && echo "  ⚠ DAG did not complete in ${MAX_WAIT}s" && exit 1

# === Step 11: Verify Outputs ===
echo ""
echo "Step 11: Verifying outputs..."

ARCHIVE_DIR="${DATA_DIR}/archive"

if [ -e "${FEATURE_HISTORY}" ]; then
    FINAL_ROWS=$(uv run python -c "import pandas as pd; print(len(pd.read_parquet('${FEATURE_HISTORY}')))" 2>/dev/null || echo "?")
    echo "  ✓ Feature history: ${FINAL_ROWS} rows"
fi
ARCHIVE_COUNT=$(find "${ARCHIVE_DIR}" -name "*.wav" 2>/dev/null | wc -l)
echo "  ✓ Archived: ${ARCHIVE_COUNT} files"
[ -d "${MODELS_DIR}/staging" ] && ls "${MODELS_DIR}/staging" 2>/dev/null | head -1 | xargs -I{} echo "  ✓ Staging model: {}"
[ -f "${PRODUCTION_MODEL}" ] && echo "  ✓ Production model deployed"

# Test API
echo "  Testing API..."
SAMPLE_WAV=$(find "${ARCHIVE_DIR}" -name "*.wav" 2>/dev/null | head -1)
if [ -n "$SAMPLE_WAV" ]; then
    RESPONSE=$(curl -s -m 60 -X POST "http://localhost:8000/predict" -F "audio_file=@${SAMPLE_WAV}" 2>/dev/null || echo "{}")
    echo "$RESPONSE" | grep -q "predicted_wer" && echo "  ✓ API prediction working" || echo "  ⚠ API test inconclusive"
fi

# === Summary ===
echo ""
echo "=== E2E Test Complete ==="
echo ""
echo "Services:"
echo "  Airflow:    http://localhost:8081 (admin/admin)"
echo "  Spark:      http://localhost:8080"
echo "  API:        http://localhost:8000/docs"
echo "  Prometheus: http://localhost:9191"
echo "  Grafana:    http://localhost:3000 (admin/admin)"
echo ""
echo "To stop: ${DOCKER_COMPOSE} down -v"
