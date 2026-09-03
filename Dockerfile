# ==============================================================================
# Multi-Stage Production Dockerfile for BioVision AI
# Stage 1: Builder (Compiles wheels and dependencies)
# Stage 2: Minimal Distroless/Slim Runtime (< 160MB footprint)
# ==============================================================================

# Stage 1: Builder
FROM python:3.10-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ==============================================================================
# Stage 2: Final Minimal Runtime
# ==============================================================================
FROM python:3.10-slim AS runner

WORKDIR /app

# Install runtime system libraries (libgomp for ONNX/OpenMP CPU acceleration)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-privileged system user for secure container execution
RUN groupadd -g 1001 appuser && \
    useradd -r -u 1001 -g appuser appuser

# Copy installed Python packages from builder stage
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy application source code
COPY --chown=appuser:appuser app/ /app/app/
COPY --chown=appuser:appuser scripts/ /app/scripts/

# Create models directory
RUN mkdir -p /app/models && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Health check configuration for orchestrator probes
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Launch production ASGI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--access-log"]
