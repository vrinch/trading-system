# Multi-stage build for strategy engine with ML and numerical libraries
FROM python:3.12.1-slim-bookworm AS builder

# Set build arguments
ARG BUILDPLATFORM
ARG TARGETPLATFORM
ARG BUILD_DATE
ARG VCS_REF

# Install system dependencies for numerical/ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    libhdf5-dev \
    pkg-config \
    git \
    wget \
    curl \
    cmake \
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

# Install QuantLib for quantitative finance
RUN wget https://github.com/lballabio/QuantLib/releases/download/QuantLib-v1.32/QuantLib-1.32.tar.gz && \
    tar -xzf QuantLib-1.32.tar.gz && \
    cd QuantLib-1.32 && \
    ./configure --disable-static --enable-shared && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd .. && \
    rm -rf QuantLib*

# Set Python environment variables for optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PYTHONPATH="/app/src"

# Create non-root user
RUN groupadd --gid 1000 trading && \
    useradd --uid 1000 --gid trading --shell /bin/bash --create-home trading

# Set working directory
WORKDIR /app

# Copy dependency files first for Docker layer caching
COPY pyproject.toml ./
COPY README.md ./

# Install Python dependencies with optimizations
RUN pip install --upgrade pip wheel setuptools && \
    # Install numpy/scipy first for better dependency resolution
    pip install numpy==1.24.3 scipy==1.11.4 && \
    # Install ML/numerical packages
    pip install torch==2.1.2+cpu torchvision==0.16.2+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    # Install other dependencies
    pip install -e .[prod] && \
    pip cache purge

# Production image
FROM python:3.12.1-slim-bookworm AS production

# Set build metadata
LABEL maintainer="Trading System Team" \
      version="1.0.0" \
      description="Strategy engine for algorithmic trading" \
      build-date=${BUILD_DATE} \
      vcs-ref=${VCS_REF}

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libblas3 \
    liblapack3 \
    libatlas3-base \
    libhdf5-103 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy compiled libraries from builder
COPY --from=builder /usr/lib/libta_lib* /usr/lib/
COPY --from=builder /usr/include/ta-lib/ /usr/include/ta-lib/
COPY --from=builder /usr/local/lib/libQuantLib* /usr/local/lib/
COPY --from=builder /usr/local/include/ql/ /usr/local/include/ql/

# Update library cache
RUN ldconfig

# Set runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app/src" \
    PATH="/home/trading/.local/bin:$PATH" \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    # ML/numerical optimizations
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    NUMEXPR_MAX_THREADS=4 \
    # Memory optimizations
    MALLOC_ARENA_MAX=2

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

# Create directories for model artifacts and data
RUN mkdir -p /app/models /app/data /app/logs /app/artifacts

# Copy application code
COPY --chown=trading:trading src/ src/
COPY --chown=trading:trading configs/ configs/

# Health check configuration
HEALTHCHECK --interval=30s --timeout=15s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Set resource limits and performance tuning
ENV MAX_WORKERS=4 \
    WEB_CONCURRENCY=4 \
    WORKER_CONNECTIONS=1000 \
    WORKER_CLASS="uvicorn.workers.UvicornWorker" \
    BACKTEST_WORKERS=2 \
    ML_WORKERS=2

# Production startup command with performance optimizations
CMD ["uvicorn", "strategy_engine.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--access-log", \
     "--log-config", "/app/configs/logging.yaml"]

# Development stage for local development with additional tools
FROM production AS development

USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    htop \
    tree \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Install Jupyter and development tools
RUN pip install jupyter ipykernel matplotlib seaborn plotly dash streamlit

# Install development Python packages
COPY --chown=trading:trading pyproject.toml ./
RUN pip install -e .[dev,research,test]

USER trading

# Create Jupyter config
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> ~/.jupyter/jupyter_notebook_config.py

# Expose additional ports for development tools
EXPOSE 8888 8501 8050

# Development startup with auto-reload
CMD ["uvicorn", "strategy_engine.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--reload", \
     "--reload-dir", "/app/src", \
     "--log-level", "debug"]

# Research stage with ML experiment tracking
FROM development AS research

USER root

# Install additional research tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    sqlite3 \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

USER trading

# Install MLflow server and experiment tracking tools
RUN pip install mlflow[extras] wandb optuna-dashboard

# Setup MLflow tracking
ENV MLFLOW_TRACKING_URI="sqlite:///app/mlruns.db" \
    MLFLOW_DEFAULT_ARTIFACT_ROOT="/app/mlruns" \
    WANDB_MODE="offline"

# Create MLflow experiment directory
RUN mkdir -p /app/mlruns /app/wandb

# Expose MLflow UI port
EXPOSE 5000

# Research startup script
COPY --chown=trading:trading tools/start_research.sh /app/start_research.sh
RUN chmod +x /app/start_research.sh

CMD ["/app/start_research.sh"]

# GPU-enabled stage for ML training
FROM nvidia/cuda:12.1-devel-ubuntu22.04 AS gpu

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3-pip \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN ln -s /usr/bin/python3.12 /usr/bin/python

# Copy application from production stage
COPY --from=production /app /app
COPY --from=builder /usr/lib/libta_lib* /usr/lib/
COPY --from=builder /usr/local/lib/libQuantLib* /usr/local/lib/

# Install GPU-specific packages
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Set CUDA environment variables
ENV CUDA_VISIBLE_DEVICES=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app
USER trading

# GPU training command
CMD ["python", "-m", "strategy_engine.ml.train", "--use-gpu"]

# Minimal production stage for deployment
FROM python:3.12.1-alpine AS minimal

# Install minimal runtime dependencies
RUN apk add --no-cache \
    libstdc++ \
    libgcc \
    curl

# Copy only essential files from production stage
COPY --from=production /app/src /app/src
COPY --from=production /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=production /usr/local/bin/uvicorn /usr/local/bin/

# Set minimal environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src"

WORKDIR /app
USER 1000

# Minimal startup
CMD ["uvicorn", "strategy_engine.main:app", "--host", "0.0.0.0", "--port", "8000"]