"""
Configuration settings for the Vedic Knowledge AI system.
Loads environment variables and provides configuration for different components.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR = os.getenv("PDF_DIR", os.path.join(BASE_DIR, "data", "books"))
DB_DIR = os.getenv("DB_DIR", os.path.join(BASE_DIR, "data", "db"))

# Ensure directories exist
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# LLM Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    import warnings
    warnings.warn("OPENAI_API_KEY not found in environment variables. LLM functionality may be limited.")

# Vector Database Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))

# Embedding Model
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Web Scraping Configuration
SCRAPING_INTERVAL = int(os.getenv("SCRAPING_INTERVAL", "86400"))  # Default to daily
REQUEST_DELAY = int(os.getenv("REQUEST_DELAY", "5"))  # 5 seconds between requests

# Trusted websites for scraping
TRUSTED_WEBSITES = [
    # Add your trusted websites here
    "https://www.gaudiya.com",
    "https://www.vedabase.com",
    "https://www.sanskrit-lexicon.uni-koeln.de",
    # Add more as needed
]

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
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Vector Database Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))

# Export directories
EXPORT_DIR = os.getenv("EXPORT_DIR", os.path.join(BASE_DIR, "data", "exports"))
QA_LOGS_DIR = os.path.join(EXPORT_DIR, "qa_logs")
REPORTS_DIR = os.path.join(EXPORT_DIR, "reports")
SUMMARIES_DIR = os.path.join(EXPORT_DIR, "summaries")
# Web cache directory
WEB_CACHE_DIR = os.getenv("WEB_CACHE_DIR", os.path.join(BASE_DIR, "data", "web_cache"))
CACHE_EXPIRY = int(os.getenv("CACHE_EXPIRY", "604800"))  # 7 days in seconds


# Ensure directories exist
os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(QA_LOGS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(SUMMARIES_DIR, exist_ok=True)
# Ensure directory exists
os.makedirs(WEB_CACHE_DIR, exist_ok=True)