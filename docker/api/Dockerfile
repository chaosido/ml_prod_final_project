FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Copy uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies using uv
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra api --locked --no-install-project --no-editable --compile-bytecode

# Install the project
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra api --locked --no-editable --compile-bytecode

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code and config
COPY serving/ serving/
COPY conf/ conf/

# Environment Variables
ENV LOG_LEVEL=INFO
ENV ASR_QE_MODEL_PATH=/models/production/model.joblib

# Expose the port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["uvicorn", "serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
