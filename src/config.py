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
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
PDF_DIR = os.getenv("PDF_DIR", os.path.join(DATA_DIR, "books"))
DB_DIR = os.getenv("DB_DIR", os.path.join(DATA_DIR, "db_new"))
TEMP_DIR = os.getenv("TEMP_DIR", os.path.join(DATA_DIR, "temp"))

EXPORT_DIR = os.getenv("EXPORT_DIR", os.path.join(DATA_DIR, "exports"))
QA_LOGS_DIR = os.path.join(EXPORT_DIR, "qa_logs")
REPORTS_DIR = os.path.join(EXPORT_DIR, "reports")
SUMMARIES_DIR = os.path.join(EXPORT_DIR, "summaries")

WEB_CACHE_DIR = os.getenv("WEB_CACHE_DIR", os.path.join(DATA_DIR, "web_cache"))
CACHE_EXPIRY = int(os.getenv("CACHE_EXPIRY", "604800")) # 7 days

# Ensure directories exist
for path in [DATA_DIR, PDF_DIR, DB_DIR, TEMP_DIR, EXPORT_DIR, QA_LOGS_DIR, REPORTS_DIR, SUMMARIES_DIR, WEB_CACHE_DIR]:
    os.makedirs(path, exist_ok=True)

# LLM Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-flash-latest") # Use gemini-1.5-flash-latest or check availability
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048")) # Gemini 1.5 Flash pode suportar mais

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", 'YOUR_GEMINI_API_KEY_HERE') # Get from .env or replace
if GEMINI_API_KEY == 'YOUR_GEMINI_API_KEY_HERE':
    warnings.warn("GEMINI_API_KEY is not set in environment or .env file. Using placeholder.")


# Vector Database Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5")) # Para busca local no VectorStore

# Embedding Model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")

# --- Web Scraping Configuration ---
SCRAPING_INTERVAL = int(os.getenv("SCRAPING_INTERVAL", "86400")) # Daily
REQUEST_DELAY = int(os.getenv("REQUEST_DELAY", "5")) # Seconds

# Trusted websites for scraping (URLs base)
TRUSTED_WEBSITES_STR = os.getenv("TRUSTED_WEBSITES", "https://www.purebhakti.com,https://vedabase.io/en/")
TRUSTED_WEBSITES = [site.strip() for site in TRUSTED_WEBSITES_STR.split(',') if site.strip() and site.startswith(('http://', 'https://'))]

# Site-specific scraping configurations for parsing individual article pages
SITE_SPECIFIC_SCRAPING_CONFIG = {
    "purebhakti.com": { # Note: domain key should not include 'www.' for consistency with urlparse().netloc.replace('www.','')
        "content_selectors": [
            "div.td-post-content",
            "article.post .entry-content",
            "div.entry-content",
        ],
        "elements_to_remove_selectors": [
            "div.td-post-source-tags", "div.td-post-sharing-bottom", "div.td-post-next-prev",
            "div.td-author-line", "div.td-module-meta-info",
            "div.td-post-views", "div.td-post-comments", "div#comments",
            ".jp-relatedposts", ".td-fix-index",
        ],
        "title_selector": "h1.entry-title",
        "metadata_selectors": {
            "article_date": {"selector": "time.entry-date.updated.td-module-date", "attribute": "datetime"},
            "author": {"selector": "div.td-module-meta-info span.td-post-author-name a"},
        }
    },
    "vedabase.io": {
        "content_selectors": ["div#content"],
        "elements_to_remove_selectors": [
            "div.r-lang-selection", "div.r-utils", "div.r-toc-title", "div.r-toc",
            "div.r-search", "div.noprint", "div.r-book-title",
            ".r-synonyms-title", ".r-translation-title", ".r-purport-title"
        ],
        "title_selector": "div.r-title > h1",
        "custom_content_assembly": True,
        "verse_selector": "div.r-verse",
        "synonyms_selector": "div.r-synonyms",
        "translation_selector": "div.r-translation",
        "purport_selector": "div.r-paragraph", # Selects all paragraphs for purport
        "metadata_selectors": {
            "text_reference": {"selector": "div.r-title > h1"},
            "chapter_info": {"selector": "div.r-title-small > a"},
        }
    },
    "bhaktivedantavediclibrary.org": {
        "content_selectors": ["article div.entry-content", "div.td-post-content"],
        "elements_to_remove_selectors": [
            ".td-post-sharing-bottom", ".td-post-next-prev", "div#comments",
            ".td-category", ".td-module-meta-info", "p > em > strong",
            "div.td-post-featured-image", "figure.wp-block-image",
        ],
        "title_selector": "h1.entry-title",
         "metadata_selectors": {
            "article_date": {"selector": "time.entry-date.updated", "attribute": "datetime"},
            "author": {"selector": ".td-post-author-name a"},
        }
    }
    # Adicione configurações para outros TRUSTED_WEBSITES aqui
}


# --- Cloud Storage Configuration (sem alterações) ---
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_BUCKET_NAME = os.getenv("GCP_BUCKET_NAME")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")

# --- Logging Configuration ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", os.path.join(DATA_DIR, "vedic_knowledge_ai.log"))