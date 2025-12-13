# Stage 1: Builder
FROM python:3.11-slim-bookworm AS builder

# Copy uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies FIRST (caching layer)
COPY pyproject.toml uv.lock ./
# Note: Spark requires pyspark [spark], and workers need torch/nemo for ASR processing [api]
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra spark --extra api --locked --no-install-project --no-editable --compile-bytecode

# Install the project
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra spark --extra api --locked --no-editable --compile-bytecode

# Stage 2: Runtime
FROM python:3.11-slim-bookworm

USER root

# Install Java 17 (LTS for Debian 12) and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    libsndfile1 \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Set Java environment
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Create Spark user with UID 50000 to match Airflow
# Create user first so we can use it for COPY
RUN useradd -m -u 50000 spark

WORKDIR /app

# Copy the environment from builder with correct ownership (fast!)
COPY --from=builder --chown=spark:spark /app/.venv /app/.venv

# Add venv to PATH (Crucial for Spark to find python)
ENV PATH="/app/.venv/bin:$PATH"
# Ensure PySpark uses the correct python
ENV PYSPARK_PYTHON=/app/.venv/bin/python
ENV PYSPARK_DRIVER_PYTHON=/app/.venv/bin/python

USER spark
