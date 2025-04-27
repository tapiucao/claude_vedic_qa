FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for Chrome/Selenium
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create and set permissions for data directories
RUN mkdir -p /app/data/books /app/data/db
RUN chmod -R 777 /app/data

# Environment variables
ENV PDF_DIR=/app/data/books
ENV DB_DIR=/app/data/db
ENV PYTHONPATH=/app

# Expose API port
EXPOSE 8000

# Command to run
CMD ["python", "api.py"]