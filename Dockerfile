FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire dataIngest folder
COPY . .

# Create logs directory
RUN mkdir -p logs

# Set environment variables
ENV PYTHONPATH=/dataIngest
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "dataIngest/scripts/scheduler_service.py"]