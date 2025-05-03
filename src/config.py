# src/config.py
"""
Configuration settings for the Vedic Knowledge AI system.
Loads environment variables and provides configuration for different components.
"""
import os
import warnings # Import warnings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR = os.getenv("PDF_DIR", os.path.join(BASE_DIR, "data", "books"))
DB_DIR = os.getenv("DB_DIR", os.path.join(BASE_DIR, "data", "db_new"))
TEMP_DIR = os.getenv("TEMP_DIR", os.path.join(BASE_DIR, "data", "temp"))

# Export directories
EXPORT_DIR = os.getenv("EXPORT_DIR", os.path.join(BASE_DIR, "data", "exports"))
QA_LOGS_DIR = os.path.join(EXPORT_DIR, "qa_logs")
REPORTS_DIR = os.path.join(EXPORT_DIR, "reports")
SUMMARIES_DIR = os.path.join(EXPORT_DIR, "summaries")

# Web cache directory
WEB_CACHE_DIR = os.getenv("WEB_CACHE_DIR", os.path.join(BASE_DIR, "data", "web_cache"))
# Default: 7 days in seconds
CACHE_EXPIRY = int(os.getenv("CACHE_EXPIRY", "604800"))

# Ensure directories exist
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(QA_LOGS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(SUMMARIES_DIR, exist_ok=True)
os.makedirs(WEB_CACHE_DIR, exist_ok=True)

# LLM Configuration
# Example: "gemini-1.5-flash", "gemini-1.5-pro" - Ensure the model is available for your API key
MODEL_NAME = "gemini-1.5-pro"
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))

# API Keys
GEMINI_API_KEY = 'AIzaSyDuLhEqJMWWtTseYm7V5KouXJ-605afKxY'
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# if not GEMINI_API_KEY:
    # Use warnings.warn for runtime warnings instead of print
   #  warnings.warn(
       #  "GEMINI_API_KEY not found in environment variables or .env file. "
       #  "LLM functionality will be unavailable. "
       #  "Please set the GEMINI_API_KEY environment variable.",
       #  UserWarning # Use UserWarning category
    # )
    # Consider if the application should exit or continue with limited functionality
    # For now, it will continue, but LLM calls will fail later.

# Vector Database Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))

# Embedding Model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")

# Web Scraping Configuration
# Default to daily (24 hours in seconds)
SCRAPING_INTERVAL = int(os.getenv("SCRAPING_INTERVAL", "86400"))
# Default: 5 seconds between requests per domain
REQUEST_DELAY = int(os.getenv("REQUEST_DELAY", "5"))

# Trusted websites for scraping (example list)
TRUSTED_WEBSITES_STR = os.getenv("TRUSTED_WEBSITES", "https://www.purebhakti.com,https://vedabase.io/en/,https://bhaktivedantavediclibrary.org/") # Example
TRUSTED_WEBSITES = [site.strip() for site in TRUSTED_WEBSITES_STR.split(',') if site.strip()]


# Cloud Storage Configuration
# AWS
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET")

# GCP
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_BUCKET_NAME = os.getenv("GCP_BUCKET_NAME")

# Azure
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper() # Ensure uppercase