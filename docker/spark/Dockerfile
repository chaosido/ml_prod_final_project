# Use official PyTorch image - already has CUDA, Python, and Torch installed!
# This avoids slow builds and version mismatches.
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Copy uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install missing system dependencies
# add build-essential for texterrors compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    openjdk-17-jre-headless \
    libsndfile1 \
    ffmpeg \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Set Java environment
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Create Spark user with UID 50000 to match Airflow
RUN useradd -m -u 50000 spark
WORKDIR /app

# CHANGE USE
COPY pyproject.toml uv.lock ./

# Install packages into a venv
# using --system would install into the container's python, which is also fine 
# but venv is safer. We use the system python to create the venv.
RUN uv sync --extra spark --extra api --locked --no-install-project --no-editable --compile-bytecode

# Install the project spark:spark /app

USER spark

# Install dependencies using uv
# COPY with chown so spark user owns them
COPY --chown=spark:spark pyproject.toml uv.lock ./

# Install packages into a venv
# using --system would install into the container's python (owned by root), which uv might fail to write to as spark user.
# So we use venv. uv will create .venv owned by spark.
RUN uv sync --extra spark --extra api --locked --no-install-project --no-editable --compile-bytecode

# Install the project
COPY --chown=spark:spark src/ src/
RUN uv sync --extra spark --extra api --locked --no-editable --compile-bytecode

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYSPARK_PYTHON=/app/.venv/bin/python
ENV PYSPARK_DRIVER_PYTHON=/app/.venv/bin/python
