# Modern Python 3.12 base
FROM python:3.12-slim

USER root

# Install Java 21 and dependencies
RUN apt-get update && apt-get install -y \
    openjdk-21-jre-headless \
    libsndfile1 \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Set Java environment
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

WORKDIR /app

# Copy and install package (PySpark will be installed as a dependency)
COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir .[spark]

# Create Spark user
RUN useradd -m -u 185 spark
USER spark
