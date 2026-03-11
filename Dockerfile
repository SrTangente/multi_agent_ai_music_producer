# Multi-stage Dockerfile for AI Music Producer
# Stage 1: Builder with uv for fast dependency resolution
# Stage 2: Runtime with minimal footprint

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.12-slim as builder

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock* ./

# Create virtual environment and install dependencies
RUN uv venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Install dependencies (without dev dependencies for smaller image)
RUN uv pip install --no-cache -r pyproject.toml

# Copy source code
COPY src/ ./src/
COPY config/ ./config/

# =============================================================================
# Stage 2: Runtime (CPU)
# =============================================================================
FROM python:3.12-slim as runtime-cpu

WORKDIR /app

# Create non-root user for security
RUN groupadd -r musicprod && useradd -r -g musicprod musicprod

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/config /app/config

# Set environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# Create directories for audio I/O
RUN mkdir -p /app/output /app/references && \
    chown -R musicprod:musicprod /app

USER musicprod

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src; print('healthy')" || exit 1

# Default command (API server)
CMD ["python", "-m", "src.api.server"]

# =============================================================================
# Stage 3: Runtime with GPU (CUDA)
# =============================================================================
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 as runtime-gpu

WORKDIR /app

# Install Python 3.12
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r musicprod && useradd -r -g musicprod musicprod

# Copy from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/config /app/config

# Set environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Create directories
RUN mkdir -p /app/output /app/references && \
    chown -R musicprod:musicprod /app

USER musicprod

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3.12 -c "import src; print('healthy')" || exit 1

CMD ["python3.12", "-m", "src.api.server"]

# =============================================================================
# Stage 4: Job runner for batch processing
# =============================================================================
FROM runtime-gpu as job-runner

# Job runner doesn't need health check (ephemeral)
HEALTHCHECK NONE

# Override entrypoint for job execution
ENTRYPOINT ["python3.12", "-m", "src.jobs.runner"]
CMD ["--help"]
