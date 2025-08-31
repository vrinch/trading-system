# Multi-stage build for minimal production image
FROM python:3.12.1-slim-bookworm AS builder

# Set build arguments for optimization
ARG BUILDPLATFORM
ARG TARGETPLATFORM

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    pkg-config \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib from source for technical indicators
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib*

# Set Python environment variables for optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# Create non-root user for security
RUN groupadd --gid 1000 trading && \
    useradd --uid 1000 --gid trading --shell /bin/bash --create-home trading

# Set working directory
WORKDIR /app

# Copy dependency files first for Docker layer caching
COPY pyproject.toml ./
COPY README.md ./

# Install Python dependencies with optimizations
RUN pip install --upgrade pip wheel setuptools && \
    pip install -e .[prod] && \
    pip cache purge

# Production image
FROM python:3.12.1-slim-bookworm AS production

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy TA-Lib from builder stage
COPY --from=builder /usr/lib/libta_lib* /usr/lib/
COPY --from=builder /usr/include/ta-lib/ /usr/include/ta-lib/

# Set runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app/src" \
    PATH="/home/trading/.local/bin:$PATH" \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random

# Create non-root user
RUN groupadd --gid 1000 trading && \
    useradd --uid 1000 --gid trading --shell /bin/bash --create-home trading

# Set working directory and ownership
WORKDIR /app
RUN chown -R trading:trading /app

# Switch to non-root user
USER trading

# Copy Python packages from builder
COPY --from=builder --chown=trading:trading /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder --chown=trading:trading /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=trading:trading src/ src/

# Health check configuration
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Set resource limits for container
ENV MAX_WORKERS=4 \
    WEB_CONCURRENCY=4 \
    WORKER_CONNECTIONS=1000

# Production startup command with optimizations
CMD ["uvicorn", "data_ingest.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--access-log", \
     "--log-config", "/app/src/data_ingest/logging.json"]

# Development stage for local development
FROM production AS development

USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
COPY --chown=trading:trading pyproject.toml ./
RUN pip install -e .[dev,test]

USER trading

# Development startup with auto-reload
CMD ["uvicorn", "data_ingest.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--reload", \
     "--log-level", "debug"]