#!/bin/bash
set -e

# 1. Create Network
echo "Creating network asr-network..."
docker network create asr-network || true

# 2. Build API Image
echo "Building API image..."
docker build -t asr-qe-api:latest -f docker/api/Dockerfile .

# 3. Run API
echo "Starting API..."
docker stop asr-api || true
docker rm asr-api || true
docker run -d \
  --name asr-api \
  --network asr-network \
  -p 8000:8000 \
  -v $(pwd)/models:/models \
  -e ASR_QE_MODEL_PATH=/models/v20251212_154204/model.joblib \
  -e LOG_LEVEL=INFO \
  asr-qe-api:latest

# 4. Run Prometheus
echo "Starting Prometheus..."
docker stop asr-prometheus || true
docker rm asr-prometheus || true
# Ensure volume exists
docker volume create prometheus_data
docker run -d \
  --name asr-prometheus \
  --network asr-network \
  -p 9091:9090 \
  -v $(pwd)/conf/monitoring/prometheus.yaml:/etc/prometheus/prometheus.yml \
  -v $(pwd)/conf/monitoring/alerts.yml:/etc/prometheus/alerts.yml \
  -v prometheus_data:/prometheus \
  prom/prometheus:v2.45.0 \
  --config.file=/etc/prometheus/prometheus.yml \
  --storage.tsdb.path=/prometheus

# 5. Run Grafana
echo "Starting Grafana..."
docker stop asr-grafana || true
docker rm asr-grafana || true
docker volume create grafana_data
docker run -d \
  --name asr-grafana \
  --network asr-network \
  -p 3000:3000 \
  -v grafana_data:/var/lib/grafana \
  grafana/grafana:10.0.0

echo "✅ Stack Started!"
echo "API: http://localhost:8000"
echo "Prometheus: http://localhost:9091"
echo "Grafana: http://localhost:3000"
