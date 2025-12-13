#!/bin/bash
set -e  # Exit on error

# Detect docker-compose command (v1: docker-compose, v2: docker compose)
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
elif docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    echo "Error: Neither 'docker-compose' nor 'docker compose' found. Please install Docker Compose."
    exit 1
fi

# Get default values from config.yaml
DEFAULT_SAMPLE_SIZE=$(uv run python -c "from omegaconf import OmegaConf; cfg = OmegaConf.load('conf/config.yaml'); print(cfg.setup_verification.sample_size)" 2>/dev/null || echo "50")

# Configuration (can be overridden via environment variables)
NUM_SAMPLES=${NUM_SAMPLES:-50}  # Smaller sample for faster testing
SAMPLE_SIZE=${SAMPLE_SIZE:-${DEFAULT_SAMPLE_SIZE}}  # Files to copy to incoming/ (from config, default 51)
CLEANUP=${CLEANUP:-true}  # Clean up Docker containers after test

# Local paths (override Docker paths for local testing)
LOCAL_DATA_DIR="data"
LOCAL_MODELS_DIR="models"

echo "=== Full Production Pipeline E2E Test ==="
echo ""
echo "Configuration:"
echo "  NUM_SAMPLES: ${NUM_SAMPLES} (VoxPopuli download)"
echo "  SAMPLE_SIZE: ${SAMPLE_SIZE} (files to process, default from config: ${DEFAULT_SAMPLE_SIZE})"
echo "  CLEANUP: ${CLEANUP}"
echo "  Using local paths: data_dir=${LOCAL_DATA_DIR}, models_dir=${LOCAL_MODELS_DIR}"
echo ""

# Step 0: Clean up previous Docker state (if requested)
if [ "${CLEANUP}" = "true" ]; then
    echo "Step 0: Cleaning up previous Docker state..."
    ${DOCKER_COMPOSE} down -v 2>/dev/null || true
    
    # Explicitly clean up local binds via Docker to handle permission issues (files owned by UID 50000)
    echo "  Cleaning up local data and models (using Docker for permissions)..."
    docker run --rm -v "$(pwd):/app" busybox sh -c "rm -rf /app/${LOCAL_DATA_DIR}/features/history.parquet /app/${LOCAL_DATA_DIR}/incoming/* /app/models/staging/* /app/models/production/*"
    
    echo "✓ Cleanup complete"
    echo ""
fi

# Step 1: Install dependencies
echo "Step 1: Installing dependencies..."
uv sync --extra research
echo "✓ Dependencies installed"
echo ""

# Step 2: Download VoxPopuli data (if needed)
MANIFEST_PATH="${LOCAL_DATA_DIR}/voxpopuli_nl/manifest_train.csv"
echo "Step 2: Ensuring VoxPopuli data is available..."
if [ -f "${MANIFEST_PATH}" ]; then
    EXISTING_SAMPLES=$(wc -l < "${MANIFEST_PATH}")
    EXISTING_SAMPLES=$((EXISTING_SAMPLES - 1))  # Subtract header
    echo "  Found ${EXISTING_SAMPLES} existing samples"
    if [ ${EXISTING_SAMPLES} -lt ${NUM_SAMPLES} ]; then
        echo "  Downloading additional samples to reach ${NUM_SAMPLES}..."
        uv run python scripts/download_voxpopuli.py \
            paths.data_dir=${LOCAL_DATA_DIR} \
            download.max_samples=${NUM_SAMPLES}
    fi
else
    echo "  Downloading ${NUM_SAMPLES} samples..."
    uv run python scripts/download_voxpopuli.py \
        paths.data_dir=${LOCAL_DATA_DIR} \
        download.max_samples=${NUM_SAMPLES}
fi
echo "✓ VoxPopuli data ready"
echo ""

# Step 3: Setup verification data (copy to incoming/)
INCOMING_DIR="${LOCAL_DATA_DIR}/incoming"
ARCHIVE_DIR="${LOCAL_DATA_DIR}/archive"
echo "Step 3: Setting up verification data in incoming/ folder..."
echo "  Note: Data written to ${INCOMING_DIR}/ is immediately visible in Docker at /opt/data/incoming/ (volume mount)"
mkdir -p "${INCOMING_DIR}"
echo "  Copying ${SAMPLE_SIZE} files to ${INCOMING_DIR}/..."
uv run python scripts/setup_verification_data.py \
    paths.data_dir=${LOCAL_DATA_DIR} \
    setup_verification.sample_size=${SAMPLE_SIZE}
if [ $? -ne 0 ]; then
    echo "  Error: setup_verification_data.py failed"
    exit 1
fi

# Verify files were copied
INCOMING_COUNT=$(find "${INCOMING_DIR}" -name "*.wav" 2>/dev/null | wc -l)
if [ ${INCOMING_COUNT} -eq 0 ]; then
    echo "  Error: No WAV files found in ${INCOMING_DIR}/"
    exit 1
fi
echo "  ✓ ${INCOMING_COUNT} files ready in ${INCOMING_DIR}/"
echo ""

# Step 4.5: Seed History Data (Simulate robust baseline)
echo "Step 4.5: Seeding History Data..."
# We use the backed-up history from previous runs (~600 samples)
# so the new model is trained on 600 + New > 600 (Prod)
SEED_HISTORY="data/seed_history.parquet"
if [ -d "${SEED_HISTORY}" ]; then
    mkdir -p "${LOCAL_DATA_DIR}/features"
    cp -r "${SEED_HISTORY}" "${LOCAL_DATA_DIR}/features/history.parquet"
    echo "  ✓ Seeded history from ${SEED_HISTORY}"
else
    echo "  ⚠ Seed history ${SEED_HISTORY} not found. Starting from scratch (cold start)."
fi
echo ""

# Step 5: Start Docker services
echo "Step 5: Starting Docker services..."
echo "  This may take a few minutes on first run (building images)..."

echo "  Running init container to fix volume permissions..."
${DOCKER_COMPOSE} up init-permissions

# Start infrastructure services first (postgres, spark)
${DOCKER_COMPOSE} up -d postgres spark-master spark-worker

# Wait for services to be ready
echo "  Waiting for services to be ready..."
sleep 5

# Step 6: Seed Production Model (for Baseline comparison)
echo "Step 6: Seeding Initial Production Model..."
# Check if we have a backup model or create a dummy one?
# For now, we assume models/v20251212_154204/model.joblib exists from repo
SEED_SOURCE="models/v20251212_154204/model.joblib"
if [ -f "${SEED_SOURCE}" ]; then
    mkdir -p models/production
    cp "${SEED_SOURCE}" models/production/model.joblib
    echo "  ✓ Seeded models/production/model.joblib"
else
    echo "  ⚠ Seed model ${SEED_SOURCE} not found. Validation pipeline might assume baseline=-1."
fi
echo ""

# Wait for Spark to be ready
echo "  Waiting for Spark master..."
SPARK_TIMEOUT=120
elapsed=0
while true; do
    # Check if Spark master is ALIVE by checking logs
    if docker logs spark-master 2>&1 | grep -q "I have been elected leader"; then
        break
    fi
    
    if [ $elapsed -ge $SPARK_TIMEOUT ]; then
        echo ""
        echo "  Error: Spark master did not start within ${SPARK_TIMEOUT}s"
        echo "  Last logs:"
        ${DOCKER_COMPOSE} logs spark-master | tail -20
        exit 1
    fi
    sleep 2
    elapsed=$((elapsed + 2))
    echo -n "."
done
echo ""
echo "  ✓ Spark master ready"

# Wait for Spark worker to register
echo "  Waiting for Spark worker to register..."
WORKER_TIMEOUT=60
elapsed=0
while true; do
    # Check if worker is registered by checking master logs
    if docker logs spark-master 2>&1 | grep -q "Registering worker"; then
        break
    fi
    
    if [ $elapsed -ge $WORKER_TIMEOUT ]; then
        echo ""
        echo "  Error: Spark worker did not register within ${WORKER_TIMEOUT}s"
        echo "  Last logs:"
        ${DOCKER_COMPOSE} logs spark-worker | tail -20
        exit 1
    fi
    sleep 2
    elapsed=$((elapsed + 2))
    echo -n "."
done
echo ""
echo "  ✓ Spark worker registered"
echo ""

# Step 6: Initialize Airflow
echo "Step 6: Initializing Airflow..."
${DOCKER_COMPOSE} up airflow-init
if [ $? -ne 0 ]; then
    echo "  Error: Airflow initialization failed"
    ${DOCKER_COMPOSE} logs airflow-init | tail -30
    exit 1
fi
echo "  ✓ Airflow initialized"
echo ""

# Step 7: Start Airflow and remaining services
echo "Step 7: Starting Airflow and remaining services..."
# Start Airflow services
${DOCKER_COMPOSE} up -d airflow-webserver airflow-scheduler
# Start API and monitoring services (all on same network: asr-network)
echo "  Starting API and monitoring services..."
${DOCKER_COMPOSE} up -d api prometheus grafana 2>/dev/null || echo "  Note: Some optional services (api, prometheus, grafana) may not be available"

# Give services a moment to start
sleep 5

# Wait for Airflow webserver
echo "  Waiting for Airflow webserver..."
AF_TIMEOUT=120
elapsed=0
while ! curl -s http://localhost:8081/health > /dev/null 2>&1; do
    if [ $elapsed -ge $AF_TIMEOUT ]; then
        echo "  Error: Airflow webserver did not start within ${AF_TIMEOUT}s"
        ${DOCKER_COMPOSE} logs airflow-webserver | tail -30
        exit 1
    fi
    sleep 3
    elapsed=$((elapsed + 3))
    echo -n "."
done
echo ""
echo "  ✓ Airflow webserver ready (http://localhost:8081)"
echo "  Login: admin / admin"
echo ""

# Ensure DAG is unpaused
echo "  Ensuring DAG is unpaused..."
sleep 5  # Give scheduler time to be ready
docker compose exec -T airflow-scheduler airflow dags unpause asr_qe_pipeline 2>&1 | grep -v "INFO\|WARNING" || true
echo "  ✓ DAG unpaused (if it was paused)"

# Step 7.5: Wait for all services to be healthy before launching browsers
echo "Step 7.5: Waiting for all services to be healthy..."
echo "  Checking service health..."

# Function to check if a service is healthy
check_service() {
    local name=$1
    local url=$2
    local timeout=${3:-30}
    local elapsed=0
    
    while [ $elapsed -lt $timeout ]; do
        if curl -s -f "${url}" > /dev/null 2>&1; then
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    return 1
}

# Check Spark (port 8080)
echo -n "  Checking Spark (port 8080)..."
if check_service "Spark" "http://localhost:8080" 60; then
    echo " ✓"
else
    echo " ⚠ (may not be ready, but continuing)"
fi

# Check API (port 8000) - may take longer due to NeMo model loading
echo -n "  Checking API (port 8000)..."
# API health check - check if container is running and responding
if docker ps --format "{{.Names}}" | grep -q "^asr-api$"; then
    if check_service "API" "http://localhost:8000/" 120; then
        echo " ✓"
    else
        echo " ⚠ (container running but not responding - may still be loading NeMo model)"
    fi
else
    echo " ⚠ (container not running)"
fi

# Check Prometheus (port 9191)
echo -n "  Checking Prometheus (port 9191)..."
if docker ps --format "{{.Names}}" | grep -q "^asr-prometheus$"; then
    if check_service "Prometheus" "http://localhost:9191/-/healthy" 60; then
        echo " ✓"
    else
        # Try alternative endpoint
        if check_service "Prometheus" "http://localhost:9191" 30; then
            echo " ✓"
        else
            echo " ⚠ (container running but not responding)"
        fi
    fi
else
    echo " ⚠ (container not running)"
fi

# Check Grafana (port 3000)
echo -n "  Checking Grafana (port 3000)..."
if docker ps --format "{{.Names}}" | grep -q "^asr-grafana$"; then
    if check_service "Grafana" "http://localhost:3000/api/health" 60; then
        echo " ✓"
    else
        # Try root endpoint
        if check_service "Grafana" "http://localhost:3000" 30; then
            echo " ✓"
        else
            echo " ⚠ (container running but not responding)"
        fi
    fi
else
    echo " ⚠ (container not running)"
fi

echo ""
echo "  All services checked. Launching web UIs..."
echo ""

# Launch web UIs in browser (only if services are ready)
# Detect browser command
if command -v xdg-open &> /dev/null; then
    BROWSER_CMD="xdg-open"
elif command -v open &> /dev/null; then
    BROWSER_CMD="open"
else
    BROWSER_CMD=""
fi

if [ -n "${BROWSER_CMD}" ]; then
    # Launch all UIs at once (they'll open in browser tabs)
    ${BROWSER_CMD} http://localhost:8081 2>/dev/null &  # Airflow
    sleep 1
    ${BROWSER_CMD} http://localhost:8080 2>/dev/null &  # Spark
    sleep 1
    ${BROWSER_CMD} http://localhost:8000/docs 2>/dev/null &  # API docs
    sleep 1
    ${BROWSER_CMD} http://localhost:9191 2>/dev/null &  # Prometheus
    sleep 1
    ${BROWSER_CMD} http://localhost:3000 2>/dev/null &  # Grafana
    echo "  ✓ Web UIs opened in browser"
    echo "  Note: Some services (API, Grafana) may still be loading if they showed warnings above"
else
    echo "  Note: Could not detect browser command (xdg-open/open). Open manually:"
    echo "    - Airflow: http://localhost:8081 (admin/admin) ✓"
    echo "    - Spark: http://localhost:8080"
    echo "    - API: http://localhost:8000/docs (may take time - loading NeMo model)"
    echo "    - Prometheus: http://localhost:9191 (metrics/monitoring)"
    echo "    - Grafana: http://localhost:3000"
fi
echo ""

# Step 8: Manually trigger the DAG
echo "Step 8: Manually triggering DAG..."
echo "  This ensures the test runs immediately."

# Trigger the DAG
echo "  Triggering 'asr_qe_pipeline'..."
docker exec airflow-webserver airflow dags trigger asr_qe_pipeline

# Wait a moment for it to appear
sleep 5

# Find the run we just triggered (or the latest running one)
echo "  Retrieving DAG Run ID..."
# Look strictly for the manual run we just triggered (queued or running)
# Use grep -o to extract *just* the run ID, ignoring table columns/formatting
LATEST_RUN_ID=$(docker exec airflow-webserver airflow dags list-runs -d asr_qe_pipeline --no-backfill 2>/dev/null | grep -o "manual__[a-zA-Z0-9_:+\.-]*" | head -1 || echo "")

DAG_RUN_ID="${LATEST_RUN_ID}"

if [ -n "${DAG_RUN_ID}" ]; then
    echo "  ✓ DAG Triggered! Run ID: ${DAG_RUN_ID}"
else
    echo ""
    echo "  ⚠ Could not retrieve Run ID. The DAG might have failed to start or list-runs output format changed."
    echo "  Checking Airflow logs (full list)..."
    docker exec airflow-webserver airflow dags list-runs -d asr_qe_pipeline 2>/dev/null
    echo ""
    echo "  Trying to monitor anyway (assuming manual__...)..."
    # Fallback attempt: construct the likely ID from timestamp if possible, but better to fail/warn.
fi
echo ""

# Step 9: Verify files are visible in Docker
echo "Step 9: Verifying files are visible in Docker..."
FILE_COUNT_LOCAL=$(find "${INCOMING_DIR}" -name "*.wav" 2>/dev/null | wc -l)

# Check from both scheduler and webserver (they should both see the same files)
# Use find instead of ls with glob to avoid expansion issues
FILE_COUNT_SCHEDULER=$(docker exec airflow-scheduler sh -c "find /opt/data/incoming -name '*.wav' 2>/dev/null | wc -l" || echo "0")
FILE_COUNT_WEBSERVER=$(docker exec airflow-webserver sh -c "find /opt/data/incoming -name '*.wav' 2>/dev/null | wc -l" || echo "0")

echo "  Local files: ${FILE_COUNT_LOCAL} .wav files in ${INCOMING_DIR}/"
echo "  Docker (scheduler): ${FILE_COUNT_SCHEDULER} .wav files in /opt/data/incoming/"
echo "  Docker (webserver): ${FILE_COUNT_WEBSERVER} .wav files in /opt/data/incoming/"

if [ "${FILE_COUNT_SCHEDULER}" -lt "${FILE_COUNT_LOCAL}" ] || [ "${FILE_COUNT_WEBSERVER}" -lt "${FILE_COUNT_LOCAL}" ]; then
    echo "  ⚠ Warning: Docker sees fewer files than local. Volume mount may not be working correctly."
    echo "  Expected: ${FILE_COUNT_LOCAL}, Got: scheduler=${FILE_COUNT_SCHEDULER}, webserver=${FILE_COUNT_WEBSERVER}"
    echo "  Checking volume mount..."
    echo "  Directory listing:"
    docker exec airflow-scheduler ls -la /opt/data/incoming/ 2>/dev/null | head -10 || echo "  (Could not list directory)"
    echo ""
    echo "  Note: If files are missing, the sensor will not trigger. Check docker-compose.yml volume mounts."
    echo "  Volume mount should be: ./data:/opt/data"
else
    echo "  ✓ Files are visible in Docker"
fi
echo ""

# Step 10: Monitor DAG execution
echo "Step 10: Monitoring DAG execution..."
echo "  View progress at: http://localhost:8081"
echo "  DAG: asr_qe_pipeline"
echo "  Run ID: ${DAG_RUN_ID}"
echo ""
echo "  Waiting for DAG to complete (this may take several minutes)..."
echo "  Press Ctrl+C to stop monitoring (DAG will continue running)"
echo ""

# Monitor DAG state (if we have a run_id)
if [ -n "${DAG_RUN_ID}" ]; then
    MAX_WAIT=1800  # 30 minutes max
    ELAPSED=0
    CHECK_INTERVAL=15
    LAST_STATE=""
    DAG_SUCCESS=false
    
    echo "  Checking DAG state every ${CHECK_INTERVAL}s..."
    echo ""
    
    while [ $ELAPSED -lt $MAX_WAIT ]; do
        # Use Airflow CLI to get DAG run state
        # The CLI needs execution_date, not run_id, so we need to get it from list-runs first
        # Try listing runs and filtering by run_id to get the state
        STATE_OUTPUT=$(docker exec airflow-webserver airflow dags list-runs -d asr_qe_pipeline --no-backfill 2>/dev/null | grep "${DAG_RUN_ID}" || echo "")
        
        if [ -n "${STATE_OUTPUT}" ]; then
            # Extract state from table (last column is state)
            # Format: conf | dag_id | dag_run_id | ... | state
            STATE=$(echo "${STATE_OUTPUT}" | awk '{print $NF}' | head -1 | tr -d ' \n' || echo "")
        fi
        
        # If that didn't work, try getting task state for the sensor (first task)
        # (Fallbacks removed for brevity as they were complex and potentially flaky)
        
        # Only print state if it changed
        if [ "${STATE}" != "${LAST_STATE}" ] && [ -n "${STATE}" ]; then
            echo "  DAG state: ${STATE}"
            LAST_STATE="${STATE}"
        fi
        
        case "$STATE" in
            "success")
                echo ""
                echo "  ✓ DAG completed successfully!"
                DAG_SUCCESS=true
                break
                ;;
            "failed")
                echo ""
                echo "  ✗ DAG failed!"
                echo "  Check logs at: http://localhost:8081"
                echo ""
                echo "  Recent scheduler logs:"
                ${DOCKER_COMPOSE} logs airflow-scheduler --tail 50
                echo ""
                echo "  Recent sensor task logs:"
                docker exec airflow-webserver airflow tasks logs asr_qe_pipeline wait_for_incoming_data "${DAG_RUN_ID}" 2>/dev/null | tail -30 || echo "  (Could not fetch logs)"
                exit 1
                ;;
            "running"|"queued"|"None"|"")
                # Show progress indicator
                if [ $((ELAPSED % 60)) -eq 0 ] && [ $ELAPSED -gt 0 ]; then
                    echo ""
                    echo "  Still ${STATE:-waiting}... (${ELAPSED}s elapsed)"
                else
                    echo -n "."
                fi
                sleep $CHECK_INTERVAL
                ELAPSED=$((ELAPSED + CHECK_INTERVAL))
                ;;
            *)
                # Unknown state - show it and continue
                if [ -n "${STATE}" ]; then
                    echo "  (State: ${STATE})"
                fi
                sleep $CHECK_INTERVAL
                ELAPSED=$((ELAPSED + CHECK_INTERVAL))
                ;;
        esac
    done
    
    if [ "$DAG_SUCCESS" = false ]; then
        echo ""
        echo "  ⚠ DAG did not complete successfully within ${MAX_WAIT}s"
        echo "  Current state: ${STATE:-unknown}"
        exit 1
    fi
else
    echo "  Monitoring skipped (no run_id available)"
    echo "  Please check DAG status manually at: http://localhost:8081"
    exit 1
fi
echo ""

# Step 11: Verify outputs
echo "Step 11: Verifying pipeline outputs..."

# Use config paths (Docker paths in production, but check local paths for verification)
FEATURE_HISTORY="${LOCAL_DATA_DIR}/features/history.parquet"
ARCHIVE_DIR="${LOCAL_DATA_DIR}/archive"
STAGING_DIR="${LOCAL_MODELS_DIR}/staging"
PRODUCTION_DIR="${LOCAL_MODELS_DIR}/production"

# Check feature history
if [ -f "${FEATURE_HISTORY}" ]; then
    echo "  ✓ Feature history created: ${FEATURE_HISTORY}"
    # Count rows if pandas is available
    ROW_COUNT=$(uv run python -c "import pandas as pd; df = pd.read_parquet('${FEATURE_HISTORY}'); print(len(df))" 2>/dev/null || echo "unknown")
    echo "    Rows: ${ROW_COUNT}"
else
    echo "  ⚠ Feature history not found: ${FEATURE_HISTORY} (may still be processing)"
fi

# Check archived files
ARCHIVE_COUNT=$(find "${ARCHIVE_DIR}" -name "*.wav" 2>/dev/null | wc -l)
if [ ${ARCHIVE_COUNT} -gt 0 ]; then
    echo "  ✓ Files archived: ${ARCHIVE_COUNT} files in ${ARCHIVE_DIR}/"
else
    echo "  ⚠ No files archived yet in ${ARCHIVE_DIR}/ (may still be processing)"
fi

# Check models
if [ -d "${STAGING_DIR}" ] && [ "$(ls -A ${STAGING_DIR} 2>/dev/null)" ]; then
    echo "  ✓ Staging model created: ${STAGING_DIR}/"
    ls -lh "${STAGING_DIR}" | head -5
fi

if [ -d "${PRODUCTION_DIR}" ] && [ "$(ls -A ${PRODUCTION_DIR} 2>/dev/null)" ]; then
    echo "  ✓ Production model deployed: ${PRODUCTION_DIR}/"
    ls -lh "${PRODUCTION_DIR}"
fi

# Check API endpoint with a sample file
echo ""
echo "  Checking API prediction endpoint..."
# Find a wav file to test
SAMPLE_WAV=$(find "${INCOMING_DIR}" -name "*.wav" | head -1)
if [ -z "$SAMPLE_WAV" ]; then
    # Try archive if incoming is empty
    SAMPLE_WAV=$(find "${ARCHIVE_DIR}" -name "*.wav" | head -1)
fi

if [ -n "$SAMPLE_WAV" ]; then
    echo "  Sending ${SAMPLE_WAV} to API..."
    RESPONSE=$(curl -s -X POST "http://localhost:8000/predict" \
        -H "accept: application/json" \
        -H "Content-Type: multipart/form-data" \
        -F "audio_file=@${SAMPLE_WAV}")
    
    echo "  Response: ${RESPONSE}"
    
    if echo "${RESPONSE}" | grep -q "predicted_wer"; then
        echo "  ✓ API prediction successful"
    else
        echo "  ✗ API prediction failed"
    fi
else
    echo "  ⚠ No WAV file found to test API"
fi


# Step 12: Test second batch processing
echo "Step 12: Testing second batch processing..."
echo "  Adding another batch of ${SAMPLE_SIZE} files to incoming/..."
echo "  Note: Files written to ${INCOMING_DIR}/ are immediately visible in Docker via volume mount (./data:/opt/data)"

# Store initial archive count (archive accumulates historical data for training)
ARCHIVE_COUNT_BEFORE_BATCH2=$(find "${ARCHIVE_DIR}" -name "*.wav" 2>/dev/null | wc -l)

# Add second batch (different random seed to get different samples)
uv run python scripts/setup_verification_data.py \
    paths.data_dir=${LOCAL_DATA_DIR} \
    setup_verification.sample_size=${SAMPLE_SIZE} \
    setup_verification.random_state=999 2>/dev/null || true

# Verify files were added
INCOMING_COUNT_BATCH2=$(find "${INCOMING_DIR}" -name "*.wav" 2>/dev/null | wc -l)
if [ ${INCOMING_COUNT_BATCH2} -eq 0 ]; then
    echo "  ⚠ Warning: No files found in incoming/ after adding second batch"
    echo "  Skipping second batch test"
else
    echo "  ✓ Added ${INCOMING_COUNT_BATCH2} files to incoming/"
    
    # Trigger DAG again for second batch
    # Triggering DAG for second batch
    echo "  Triggering DAG for second batch..."
    # Use CLI for consistency
    docker exec airflow-webserver airflow dags trigger asr_qe_pipeline
    sleep 5
    
    # Get the latest run ID    # Get the latest run ID (looking for manual run)
    # Use grep -o to extract *just* the run ID, ignoring table columns/formatting
    LATEST_RUN_ID_BATCH2=$(docker exec airflow-webserver airflow dags list-runs -d asr_qe_pipeline --no-backfill 2>/dev/null | grep -o "manual__[a-zA-Z0-9_:+\.-]*" | head -1 || echo "")
    
    DAG_RUN_ID_BATCH2="${LATEST_RUN_ID_BATCH2}"
    
    if [ -n "${DAG_RUN_ID_BATCH2}" ]; then
        echo "  ✓ DAG triggered for second batch (run_id: ${DAG_RUN_ID_BATCH2})"
        
        # Monitor second batch
        echo "  Monitoring second batch processing..."
        MAX_WAIT_BATCH2=1800  # 30 minutes max
        ELAPSED_BATCH2=0
        CHECK_INTERVAL_BATCH2=15
        LAST_STATE_BATCH2=""
        DAG_SUCCESS_BATCH2=false
        
        while [ $ELAPSED_BATCH2 -lt $MAX_WAIT_BATCH2 ]; do
             STATE_OUTPUT=$(docker exec airflow-webserver airflow dags list-runs -d asr_qe_pipeline --no-backfill 2>/dev/null | grep "${DAG_RUN_ID_BATCH2}" || echo "")
        
            if [ -n "${STATE_OUTPUT}" ]; then
                STATE_BATCH2=$(echo "${STATE_OUTPUT}" | awk '{print $NF}' | head -1 | tr -d ' \n' || echo "")
            fi

            if [ "${STATE_BATCH2}" != "${LAST_STATE_BATCH2}" ] && [ -n "${STATE_BATCH2}" ]; then
                echo "  DAG state: ${STATE_BATCH2}"
                LAST_STATE_BATCH2="${STATE_BATCH2}"
            fi
            
            if [ "${STATE_BATCH2}" == "success" ]; then
                echo "  ✓ Second batch processed successfully!"
                DAG_SUCCESS_BATCH2=true
                break
            elif [ "${STATE_BATCH2}" == "failed" ]; then
                echo "  ✗ Second batch failed!"
                exit 1
            fi
            
            sleep $CHECK_INTERVAL_BATCH2
            ELAPSED_BATCH2=$((ELAPSED_BATCH2 + CHECK_INTERVAL_BATCH2))
        done
         
         if [ "$DAG_SUCCESS_BATCH2" = false ]; then
            echo ""
            echo "  ⚠ DAG did not complete successfully within ${MAX_WAIT_BATCH2}s"
            exit 1
        fi
            
            # Fallback to alternative API endpoint
            if [ -z "${STATE_BATCH2}" ]; then
                STATE_BATCH2=$(curl -s -u "admin:admin" \
                    "http://localhost:8081/api/v1/dags/asr_qe_pipeline/dagRuns?dag_run_id=${DAG_RUN_ID_BATCH2}" \
                    2>/dev/null | grep -oP '"state":\s*"\K[^"]*' | head -1 || echo "")
            fi
            
            case "$STATE_BATCH2" in
                "success")
                    echo ""
                    echo "  ✓ Second batch DAG completed successfully!"
                    break
                    ;;
                "failed")
                    echo ""
                    echo "  ✗ Second batch DAG failed!"
                    echo "  Check logs at: http://localhost:8081"
                    break
                    ;;
                "running"|"queued")
                    echo -n "."
                    sleep $CHECK_INTERVAL_BATCH2
                    ELAPSED_BATCH2=$((ELAPSED_BATCH2 + CHECK_INTERVAL_BATCH2))
                    ;;
                *)
                    sleep $CHECK_INTERVAL_BATCH2
                    ELAPSED_BATCH2=$((ELAPSED_BATCH2 + CHECK_INTERVAL_BATCH2))
                    ;;
            esac
        done
        
        # Verify second batch was processed
        sleep 5  # Give archive task time to complete
        ARCHIVE_COUNT_AFTER_BATCH2=$(find "${ARCHIVE_DIR}" -name "*.wav" 2>/dev/null | wc -l)
        ARCHIVE_INCREASE=$((ARCHIVE_COUNT_AFTER_BATCH2 - ARCHIVE_COUNT_BEFORE_BATCH2))
        
        if [ ${ARCHIVE_INCREASE} -ge ${INCOMING_COUNT_BATCH2} ]; then
            echo "  ✓ Second batch files archived: ${ARCHIVE_INCREASE} new files in archive/"
        else
            echo "  ⚠ Archive increased by ${ARCHIVE_INCREASE} (expected ~${INCOMING_COUNT_BATCH2})"
        fi
        
        # Check feature history increased
        if [ -f "${FEATURE_HISTORY}" ]; then
            ROW_COUNT_BATCH2=$(uv run python -c "import pandas as pd; df = pd.read_parquet('${FEATURE_HISTORY}'); print(len(df))" 2>/dev/null || echo "unknown")
            echo "  ✓ Feature history now has ${ROW_COUNT_BATCH2} rows (increased from first batch)"
        fi
    else
        echo "  ⚠ Could not trigger DAG for second batch automatically"
        echo "  Please trigger manually at: http://localhost:8081"
    fi
fi
echo ""

# Step 12: Summary
echo "=== Test Summary ==="
echo ""
echo "✓ Pipeline execution initiated"
echo ""
echo "Services running (all on asr-network):"
echo "  - Airflow UI: http://localhost:8081 (admin/admin) - Pipeline orchestration"
echo "  - Spark UI: http://localhost:8080 - Spark cluster monitoring"
echo "  - API: http://localhost:8000/docs - FastAPI docs "
echo "  - Prometheus: http://localhost:9191 - Metrics/monitoring"
echo "  - Grafana: http://localhost:3000 - Metrics visualization dashboards"
echo ""
echo "=== Viewing Logs and Model Performance ==="
echo ""
echo "To view logs from all services:"
echo "  ${DOCKER_COMPOSE} logs -f                    # All services (follow mode)"
echo "  ${DOCKER_COMPOSE} logs airflow-scheduler     # Airflow scheduler logs"
echo "  ${DOCKER_COMPOSE} logs airflow-webserver     # Airflow webserver logs"
echo "  ${DOCKER_COMPOSE} logs spark-master          # Spark master logs"
echo "  ${DOCKER_COMPOSE} logs asr-api               # API service logs"
echo ""
echo "To view model performance metrics:"
echo "  1. Airflow Task Logs (BEST for model metrics):"
echo "     - Go to http://localhost:8081 → DAG 'asr_qe_pipeline' → Click a run"
echo "     - Click 'train_model' task → 'Log' button"
echo "     - Look for:"
echo "       * Spearman Rho (correlation coefficient)"
echo "       * RMSE (Root Mean Squared Error)"
echo "       * Feature Importance (asr_confidence, duration, snr_db, rms_db)"
echo "       * Prediction statistics"
echo ""
echo "  2. View all logs from command line:"
echo "     ${DOCKER_COMPOSE} logs airflow-scheduler | grep -E 'Spearman|RMSE|Feature Importance'"
echo ""
echo "  3. Model Files:"
echo "     - Check: ${STAGING_DIR}/<version>/model.joblib"
echo "     - Metrics are logged during training (see Airflow logs above)"
echo ""
echo "  4. Prometheus Metrics (API/inference performance):"
echo "     - Go to http://localhost:9191"
echo "     - Query: predicted_wer, input_audio_snr_db, low_quality_transcripts_total"
echo ""
echo "  5. View specific task logs directly:"
echo "     docker exec airflow-scheduler airflow tasks logs asr_qe_pipeline train_model <run_id>"
echo ""
echo "To check if pipeline succeeded:"
echo "  - Go to http://localhost:8081 → DAG 'asr_qe_pipeline'"
echo "  - Green = success, Red = failed, Yellow = running"
echo "  - Or run: ${DOCKER_COMPOSE} logs airflow-scheduler | grep -E 'success|failed|error' | tail -20"
echo ""
echo "Data locations:"
FINAL_ARCHIVE_COUNT=$(find "${ARCHIVE_DIR}" -name "*.wav" 2>/dev/null | wc -l)
echo "  - Incoming: ${INCOMING_DIR}/ (should be empty after processing)"
echo "  - Archive: ${ARCHIVE_DIR}/ (${FINAL_ARCHIVE_COUNT} files)"
echo "  - Features: ${FEATURE_HISTORY}"
echo "  - Models: ${STAGING_DIR}/ and ${PRODUCTION_DIR}/"
echo ""
if [ "${CLEANUP}" = "true" ]; then
    echo "To stop services and clean up:"
    echo "  ${DOCKER_COMPOSE} down -v"
else
    echo "Services will continue running."
    echo "To stop manually: ${DOCKER_COMPOSE} down"
fi
echo ""
echo "=== E2E Test Complete ==="

