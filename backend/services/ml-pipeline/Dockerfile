FROM python:3.12.1-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc git wget curl cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY README.md ./
RUN pip install --upgrade pip wheel setuptools && pip install -e .[dev,test]

# Production image
FROM python:3.12.1-slim-bookworm AS production

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/ src/

EXPOSE 8000

CMD ["uvicorn", "ml_pipeline.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]