# Production Dockerfile for LLM Incident Response System
FROM python:3.11-slim

LABEL maintainer="LLM Incident Response Team"
LABEL description="LLM-powered Kubernetes incident response and auto-remediation"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies (faster - no extra tools)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY incident_analyzer.py .
COPY argocd_integration.py .

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port for health checks and metrics
EXPOSE 8080

# Run the application
CMD ["python", "incident_analyzer.py"]
