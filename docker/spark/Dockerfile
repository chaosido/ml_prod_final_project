# Stage 1: Builder
FROM python:3.12-slim-bookworm AS builder

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
# Note: Spark requires pyspark, which is in [spark] extra
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra spark --locked --no-install-project --no-editable --compile-bytecode

# Install the project
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra spark --locked --no-install-project --no-editable --compile-bytecode

# Stage 2: Runtime
FROM python:3.12-slim-bookworm

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

WORKDIR /app

# Copy the environment from builder
COPY --from=builder /app/.venv /app/.venv

# Add venv to PATH (Crucial for Spark to find python)
ENV PATH="/app/.venv/bin:$PATH"
# Ensure PySpark uses the correct python
ENV PYSPARK_PYTHON=/app/.venv/bin/python
ENV PYSPARK_DRIVER_PYTHON=/app/.venv/bin/python

# Create Spark user
RUN useradd -m -u 185 spark
USER spark
