#!/bin/bash
# scripts/generate_traffic.sh
# Unified traffic generator for ASR QE Pipeline
# Generates normal traffic, drift traffic (optional), and error traffic to populate Grafana and test Alerts.

set -e

API_URL="http://localhost:8000/predict"
INCOMING_DIR="data/incoming"
ARCHIVE_DIR="data/archive"
NOISE_FILE="data/noise.wav"
TEXT_FILE="README.md"
TIMEOUT=30 # Timeout for each curl request

# Defaults
NORMAL_REQUESTS=50
DRIFT_REQUESTS=20 
ERROR_REQUESTS=10
CONCURRENCY=10

echo "=== ASR QE Traffic Generator ==="
echo "Target: ${API_URL}"

# Helper function to send requests (sequential)
# Usage: send_request <file_path> <type>
send_request() {
    local file=$1
    local type=$2
    local method="POST"
    
    # Send request with timeout, silence output, print dot on completion
    if curl -s -m ${TIMEOUT} -o /dev/null -X "${method}" "${API_URL}" \
        -H "accept: application/json" \
        -H "Content-Type: multipart/form-data" \
        -F "audio_file=@${file}"; then
        echo -n "."
    else
        echo -n "x"
    fi
}

# --- Part 1: Normal Traffic ---
echo ""
echo "--- 1. Normal Traffic Generation ---"
FILES=$(find "${INCOMING_DIR}" "${ARCHIVE_DIR}" -name "*.wav" 2>/dev/null | head -50)

if [ -z "$FILES" ]; then
    echo "⚠ No .wav files found in ${INCOMING_DIR} or ${ARCHIVE_DIR}."
    echo "  Downloading minimal dataset (10 samples)..."
    uv run python scripts/download_voxpopuli.py paths.data_dir=data download.max_samples=10
    uv run python scripts/setup_verification_data.py paths.data_dir=data setup_verification.sample_size=10
    
    # Refresh file list
    FILES=$(find "${INCOMING_DIR}" "${ARCHIVE_DIR}" -name "*.wav" 2>/dev/null | head -50)
fi

if [ -z "$FILES" ]; then
    echo "⚠ No .wav files found in ${INCOMING_DIR} or ${ARCHIVE_DIR}. Skipping normal traffic."
else
    FILES_ARRAY=($FILES)
    NUM_FILES=${#FILES_ARRAY[@]}
    echo "Sending ${NORMAL_REQUESTS} normal requests (sequential with delay)..."

    for ((i=1; i<=NORMAL_REQUESTS; i++)); do
        INDEX=$(( (i - 1) % NUM_FILES ))
        FILE="${FILES_ARRAY[$INDEX]}"
        # Run sequentially with delay
        send_request "${FILE}" "normal"
        # Sleep 0.5s to 1.5s
        sleep 1
    done
    echo " Done!"
fi

# --- Part 2: Drift Traffic (High WER) ---
echo ""
echo "--- 2. Drift Traffic Generation (Testing 'ModelDriftHighWER') ---"
if command -v ffmpeg &> /dev/null; then
    echo "Generating white noise audio file..."
    ffmpeg -f lavfi -i "anoisesrc=a=0.5:c=white:d=5" -ar 16000 -ac 1 -y "${NOISE_FILE}" -loglevel error
    
    echo "Sending ${DRIFT_REQUESTS} noisy requests (sequential with delay)..."
    for ((i=1; i<=DRIFT_REQUESTS; i++)); do
        send_request "${NOISE_FILE}" "drift"
        sleep 1
    done
    echo " Done!"
    rm -f "${NOISE_FILE}"
else
    echo "⚠ ffmpeg not found. Skipping Drift Test generation."
fi

# --- Part 3: Error Traffic (5xx) ---
echo ""
echo "--- 3. Error Traffic Generation (Testing 'HighErrorRate') ---"
echo "Sending ${ERROR_REQUESTS} invalid requests (sequential with delay)..."

for ((i=1; i<=ERROR_REQUESTS; i++)); do
    send_request "${TEXT_FILE}" "error"
    sleep 1
done
echo " Done!"

echo ""
echo "=== Generation Complete ==="
echo "Check Grafana: http://localhost:3000"
echo "Check Alerts:  http://localhost:9191/alerts"
