# Stage 1: Builder
FROM python:3.12-slim-bookworm AS builder

# Copy uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install build dependencies (needed for some python packages that might compile extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies FIRST (caching layer)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra api --locked --no-install-project --no-editable --compile-bytecode

# Install the project
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra api --locked --no-editable --compile-bytecode

# Stage 2: Runtime
FROM python:3.12-slim-bookworm

WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the environment from builder
COPY --from=builder /app/.venv /app/.venv

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code and config
COPY serving/ serving/
COPY conf/ conf/

# Environment Variables
ENV LOG_LEVEL=INFO
ENV ASR_QE_MODEL_PATH=/models/model.joblib

# Expose the port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# Run the application
CMD ["/app/.venv/bin/uvicorn", "serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
