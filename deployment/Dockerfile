# Multi-stage build for smaller image size
FROM python:3.9-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.9-slim

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY pyproject.toml .

# Make sure scripts are in PATH
ENV PATH=/root/.local/bin:$PATH

# Create directories for data and outputs
RUN mkdir -p data/raw data/interim data/processed \
    datasets feature_store/offline feature_store/online \
    models outputs logs

# Expose port for API
EXPOSE 8000

# Default command (can be overridden)
CMD ["uvicorn", "src.inference.api:app", "--host", "0.0.0.0", "--port", "8000"]
