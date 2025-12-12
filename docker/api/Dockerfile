# Stage 1: Builder
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y gcc python3-dev && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/

# Build wheels for api dependencies
RUN pip wheel .[api] --wheel-dir /wheels

# Stage 2: Runtime
FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels
COPY --from=builder /wheels /wheels

# Install
RUN pip install --no-cache-dir --no-index --find-links=/wheels asr_qe[api]

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "asr_qe.serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
