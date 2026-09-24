# --- Stage 1: Builder ---
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies (only needed for compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt


# --- Stage 2: Runtime ---
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# curl is needed for the HEALTHCHECK below — python:3.11-slim doesn't
# include it (or wget) by default.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

# Install dependencies (no virtual environment needed in container)
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Non-root. The app writes nothing to disk, reads its GCP credential from a
# read-only mount and its database URL from the environment, so there is no
# reason for it to run as uid 0 with a writable copy of its own source. Created
# before COPY so the source lands already owned by it.
#
# A fixed uid (not just a name) so a bind-mounted volume in the local compose
# file has predictable ownership across machines.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/audit/db \
    && chown appuser:appuser /app/audit /app/audit/db

# Explicit runtime directories. Private seed payloads are excluded from the
# context; operator seed loading uses an explicit read-only data mount.
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser scripts/migrations/ ./scripts/migrations/
COPY --chown=appuser:appuser scripts/export_content.py ./scripts/export_content.py

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/api/health/ready || exit 1

# NO --reload. It is a development flag: it starts a supervisor process plus a
# worker and installs a filesystem watcher over /app, which in a deployed
# container walks a tree that never changes, forever, on an instance whose CPU
# credits are the scarce resource. It is also a correctness hazard -- any write
# into /app restarts the worker, and a restart drops every in-flight rate-limit
# slot and per-session lock RateControlService is holding.
#
# Local development gets it back as a `command:` override in
# docker-compose.yml, which is the file that actually bind-mounts the source.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
