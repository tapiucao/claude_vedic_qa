# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies if needed (e.g., for certain libraries)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

# Install Poetry or pip-tools if you use them, otherwise skip
# Example for pip:
# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir reduces image size
RUN pip install --no-cache-dir --upgrade pip  
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on (defined in api.py with uvicorn)
EXPOSE 8000

# Define the command to run your application
# This assumes your FastAPI app instance is named 'app' in the 'api.py' file
# Adjust reload flag as needed (remove for production)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]