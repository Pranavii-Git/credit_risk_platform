# ============================================================
# Credit Risk Intelligence Platform — Dockerfile
# ============================================================
# Base: Python 3.10 slim (smaller image, faster pull)
# Build: multi-stage avoided for simplicity — single container
# ============================================================

FROM python:3.10-slim

# Metadata
LABEL maintainer="credit-risk-platform"
LABEL description="AI-powered Credit Risk Intelligence Platform"

# Set working directory
WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
# build-essential: needed for LightGBM + some numpy builds
# git: needed by some pip packages at install time
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first to leverage Docker layer caching.
# If requirements.txt doesn't change, this layer is cached.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── App source code ───────────────────────────────────────────────────────────
COPY . .

# ── Create runtime directories ────────────────────────────────────────────────
# data/ and models/ are volume-mounted; create as fallback
RUN mkdir -p /app/data /app/models /app/sql /app/logs

# ── Streamlit config ──────────────────────────────────────────────────────────
RUN mkdir -p /root/.streamlit
RUN echo '\
[server]\n\
port = 8501\n\
address = "0.0.0.0"\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
' > /root/.streamlit/config.toml

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8501

# ── Health check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
