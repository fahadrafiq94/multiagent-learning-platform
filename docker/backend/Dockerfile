# ---------------------------------------------------
# Base Image
# ---------------------------------------------------
FROM python:3.12-slim

# ---------------------------------------------------
# Python Environment
# ---------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_PROJECT_ENVIRONMENT=/usr/local

# ---------------------------------------------------
# Working Directory
# ---------------------------------------------------
WORKDIR /app

# ---------------------------------------------------
# System dependencies
# ---------------------------------------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------
# Install uv
# ---------------------------------------------------
RUN pip install --no-cache-dir uv

# ---------------------------------------------------
# Dependency files
# ---------------------------------------------------
COPY backend/pyproject.toml ./pyproject.toml
COPY backend/uv.lock ./uv.lock

# ---------------------------------------------------
# Install dependencies
# ---------------------------------------------------
RUN uv sync --frozen

# ---------------------------------------------------
# Application source only
# ---------------------------------------------------
COPY backend/app ./app

# ---------------------------------------------------
# FastAPI
# ---------------------------------------------------
EXPOSE 8000

# ---------------------------------------------------
# Backend health
# ---------------------------------------------------
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=15s \
    --retries=3 \
    CMD curl --fail http://localhost:8000/model/health || exit 1

# ---------------------------------------------------
# Default command
# ---------------------------------------------------
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]