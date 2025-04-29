# Vedic Knowledge AI System

A comprehensive AI-powered knowledge base for Gaudiya Math bhakti and Vedic scriptures, designed to understand, retrieve, and provide answers about Sanskrit terminology, verses, and philosophical concepts from Vedic literature.

## Table of Contents

1. [System Overview](#system-overview)
2. [System Architecture](#system-architecture)
3. [Directory Structure](#directory-structure)
4. [Installation and Setup](#installation-and-setup)
5. [Usage Guide](#usage-guide)
   - [Command Line Interface](#command-line-interface)
   - [Interactive Mode](#interactive-mode)
   - [API Server](#api-server)
6. [Adding Content](#adding-content)
7. [Web Scraping](#web-scraping)
8. [Export Features](#export-features)
9. [Customization](#customization)
10. [Troubleshooting](#troubleshooting)
11. [Advanced Features](#advanced-features)
12. [Cloud Deployment](#cloud-deployment)

## System Overview

Vedic Knowledge AI is a specialized system for working with Vedic scriptures and Gaudiya Math texts. The system offers:

- **Natural Language Q&A**: Ask questions about Vedic concepts and receive detailed, sourced answers
- **Sanskrit Term Explanations**: Get comprehensive explanations of Sanskrit terms with etymologies
- **Verse Interpretations**: Explore the meanings and interpretations of verses from Vedic scriptures
- **Intelligent Web Scraping**: Ethically collect and integrate knowledge from trusted websites
- **Export Capabilities**: Generate dictionaries, reports, and Q&A logs for further study
- **Multiple Interfaces**: Command-line, interactive, and API access options

## System Architecture

The system is built around these core components:

- **Document Processing**: Handles PDF ingestion and text structuring
- **Knowledge Base**: Vector store for efficient retrieval of relevant content
- **QA System**: Combines retrieval and language models for accurate answers
- **Web Scraper**: Ethically collects data from trusted sources with caching
- **Export System**: Generates structured exports in multiple formats

## Directory Structure

```
vedic-knowledge-ai/
├── .env                        # Environment variables
├── requirements.txt            # Dependencies
├── README.md                   # Project documentation
├── app.py                      # Main application file
├── api.py                      # API interface
├── docker-compose.yml          # For containerization
├── Dockerfile                  # For containerization
├── data/
│   ├── books/                  # PDF storage
│   │   └── [your_pdfs_here]    # Place PDFs here
│   ├── db/                     # Vector database storage
│   ├── exports/                # Generated exports
│   │   ├── qa_logs/            # Question-answer records
│   │   ├── reports/            # System reports
│   │   └── summaries/          # Text summaries
│   ├── web_cache/              # Cached web content
│   │   └── metadata.json       # Cache metadata
│   └── temp/                   # Temporary processing
└── src/
    ├── config.py               # Configuration settings
    ├── document_processor/     # PDF and text processing
    │   ├── pdf_loader.py       # PDF loading
    │   ├── text_splitter.py    # Text chunking
    │   └── sanskrit_processor.py # Sanskrit handling
    ├── knowledge_base/         # Vector database
    │   ├── vector_store.py     # Vector database operations
    │   ├── embeddings.py       # Embedding models
    │   └── prompt_templates.py # Specialized prompts
    ├── qa_system/              # QA functionality
    │   ├── llm_interface.py    # LLM integration
    │   ├── retriever.py        # Document retrieval
    │   └── citation.py         # Source attribution
    ├── web_scraper/            # Web scraping
    │   ├── scraper.py          # Basic scraper
    │   ├── dynamic_scraper.py  # JavaScript handler
    │   ├── scheduler.py        # Scheduled scraping
    │   ├── cache_manager.py    # Web content caching
    │   └── ethics.py           # Ethical scraping rules
    └── utils/                  # Utilities
        ├── exporter.py         # Export functionality
        ├── logger.py           # Logging setup
        └── cloud_sync.py       # Cloud storage sync
```

## Installation and Setup

### Prerequisites

- Python 3.9+
- At least 8GB RAM recommended
- OpenAI API key

### Step 1: Clone or Create the Directory Structure

Create the project folder structure as shown above or clone the repository:

```bash
git clone https://github.com/yourusername/vedic-knowledge-ai.git
cd vedic-knowledge-ai
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables

Create a `.env` file in the project root and add your configuration:

```
# API Keys
GEMINI_API_KEY='AIzaSyDuLhEqJMWWtTseYm7V5KouXJ-605afKxY'

# Directories
PDF_DIR=./data/books
DB_DIR=./data/db_new
EXPORT_DIR=./data/exports
WEB_CACHE_DIR=./data/web_cache

# LLM Configuration
MODEL_NAME=gpt-4
TEMPERATURE=0.2
MAX_TOKENS=2048

# Web Scraper
SCRAPING_INTERVAL=86400  # 24 hours in seconds
REQUEST_DELAY=5  # 5 seconds between requests
```

### Step 4: Initialize the System

Run the initialization command to set up the directory structure:

```bash
python app.py init
```

### Step 5: Add Your PDF Documents

Place your Vedic and Gaudiya Math PDFs in the `data/books` directory. You can organize them in subdirectories if desired (e.g., by text, author, or tradition).

### Step 6: Load Documents into the System

```bash
python app.py load
```

## Usage Guide

### Command Line Interface

The system provides a comprehensive command-line interface:

#### Basic Commands

```bash
# Get help
python app.py -h

# Get information about database
python app.py info
```

#### Querying Knowledge

```bash
# Ask a question
python app.py answer "What is the meaning of dharma in Bhagavad Gita?"

# Ask and export the result
python app.py answer "What is the meaning of dharma in Bhagavad Gita?" --export

# Explain a Sanskrit term
python app.py explain-term "atma"

# Explain a verse
python app.py explain-verse "karmanye vadhikaraste ma phaleshu kadachana"

# Explain with reference
python app.py explain-verse "karmanye vadhikaraste ma phaleshu kadachana" --reference "Bhagavad Gita 2.47"
```

#### Web Scraping

```bash
# Scrape a website
python app.py scrape "https://www.vedabase.com/en/bg/2/47"

# Scrape a JavaScript-heavy website
python app.py scrape "https://example.com" --dynamic

# Force refresh of cached content
python app.py scrape "https://example.com" --bypass-cache

# Control the scraping scheduler
python app.py scheduler start
python app.py scheduler stop
python app.py scheduler status
```

#### Cache Management

```bash
# Show cache statistics
python app.py cache stats

# Clear expired cache entries
python app.py cache clear

# Clear all cache entries
python app.py cache clear --all
```

#### Export Functions

```bash
# Export Sanskrit terms dictionary
python app.py export terms

# Generate system report
python app.py export report
```

### Interactive Mode

The interactive mode provides an easy-to-use interface for exploring the system:

```bash
python app.py interactive
```

Available commands in interactive mode:

```
ask <question>         - Ask a question
term <sanskrit term>   - Explain a Sanskrit term
verse <verse text>     - Explain a verse
scrape <url>           - Scrape a website
dynamic <url>          - Scrape a JS-heavy website
cache stats            - Show cache statistics
cache clear            - Clear expired cache entries
export terms           - Export Sanskrit terms dictionary
export report          - Generate system report
info                   - Show database information
exit                   - Exit interactive mode
```

Example interactive session:

```
> ask What is the concept of bhakti in Gaudiya Vaishnavism?
[system provides answer with sources]

> term atma
[system explains the Sanskrit term]

> export terms
Exported Sanskrit terms dictionary to data/exports/summaries/sanskrit_terms_20250426123456.md
```

### API Server

For programmatic access or building web interfaces, use the API server:

```bash
python api.py
```

This starts a FastAPI server at http://localhost:8000 with interactive documentation at http://localhost:8000/docs.

Example API endpoints:

- `POST /answer` - Answer a question
- `POST /explain/term` - Explain a Sanskrit term
- `POST /explain/verse` - Explain a verse
- `POST /scrape` - Scrape a website
- `GET /database/info` - Get database information

Example request:

```json
POST /answer
{
  "question": "What is the meaning of dharma?",
  "filters": {"source_text": "Bhagavad Gita"}
}
```

## Adding Content

### Adding PDFs

1. Place PDF files in the `data/books` directory
2. Organize in subdirectories if desired
3. Run `python app.py load` to process and index the books

Best practices for PDFs:

- Use high-quality PDFs with proper text layers (not just scanned images)
- For scanned texts, run OCR before adding to the system
- Standardize naming conventions for easier reference

### Adding Web Sources

To add new sources for scraping:

1. Edit `src/config.py` to add trusted websites to `TRUSTED_WEBSITES`
2. Run `python app.py scheduler start --immediate` to start scraping

Alternatively, scrape individual websites:

```bash
python app.py scrape "https://www.example.com/vedic-content"
```

## Web Scraping

The system includes a sophisticated web scraping system with these features:

### Ethical Scraping

All web scraping follows strict ethical guidelines:

- Respects robots.txt directives
- Implements rate limiting to avoid overwhelming websites
- Checks for copyright notices and terms of service
- Avoids sensitive areas (login pages, admin sections, etc.)

### Caching System

The web cache system optimizes performance and reduces network load:

- Stores previously fetched web content by domain
- Automatically expires cache entries after configurable period
- Tracks cache hits and misses for performance monitoring
- Organizes cached content by domain for easy management

To view cache statistics:

```bash
python app.py cache stats
```

### Dynamic Website Handling

For JavaScript-heavy websites that render content dynamically:

```bash
python app.py scrape "https://example.com" --dynamic
```

This uses a headless browser to properly render the page before processing.

### Scheduled Scraping

To keep your knowledge base up-to-date:

```bash
# Start scheduled scraping
python app.py scheduler start

# Run immediately and then schedule
python app.py scheduler start --immediate

# Check status
python app.py scheduler status

# Stop scheduling
python app.py scheduler stop
```

## Export Features

The system can export various types of data:

### Q&A Logs

Records of questions asked and answers provided:

```bash
python app.py answer "What is dharma?" --export
```

Exports to `data/exports/qa_logs/` in both JSON and markdown formats.

### Sanskrit Terms Dictionary

Generate a comprehensive dictionary of Sanskrit terms:

```bash
python app.py export terms
```

Creates a searchable markdown file in `data/exports/summaries/` with:
- Devanagari script (if available)
- Transliteration
- Definition
- Etymology
- Examples
- Related terms

### System Reports

Generate reports about your system's status:

```bash
python app.py export report
```

Creates detailed reports in `data/exports/reports/` including:
- Database statistics
- Cache performance
- Scraper status
- System configuration

## Customization

### Vector Database Configuration

Key parameters in `.env`:

```
# Vector Database
CHUNK_SIZE=1000         # Size of text chunks
CHUNK_OVERLAP=200       # Overlap between chunks
TOP_K_RESULTS=5         # Number of results to retrieve
```

For Sanskrit and Vedic texts:
- Consider larger chunk sizes (1000-2000) to preserve context for philosophical concepts
- Use higher overlap (20-30%) to maintain context for verses and commentaries
- Adjust TOP_K_RESULTS based on query complexity (3-5 for specific questions, 5-10 for philosophical inquiries)

### LLM Configuration

```
# LLM Configuration
MODEL_NAME=gpt-4         # Which OpenAI model to use
TEMPERATURE=0.2          # Creativity vs. determinism (lower is more deterministic)
MAX_TOKENS=2048          # Maximum response length
```

### Prompt Templates

Customize prompt templates in `src/knowledge_base/prompt_templates.py` for different query types:

- General Vedic knowledge
- Sanskrit term definitions
- Verse explanations
- Concept comparisons
- Historical/biographical information
- Ritual/practice explanations

### Embedding Models

Change the embedding model in `src/config.py`:

```python
# Default model
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Alternative models to consider:
# - "sentence-transformers/multi-qa-mpnet-base-dot-v1" (better for Q&A)
# - "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" (may handle Sanskrit better)
```

## Troubleshooting

### Common Issues

#### PDF Processing Issues

**Symptom**: PDFs fail to load or have missing text
**Solutions**:
- Ensure PDFs are text-based, not just scanned images
- Run OCR on scanned PDFs first
- Check PDF permissions (some PDFs are locked)

#### Embedding Model Errors

**Symptom**: "Error initializing embeddings"
**Solutions**:
- Check internet connection
- Verify you have at least 8GB RAM
- Try a smaller embedding model in `src/config.py`

#### LLM API Errors

**Symptom**: "Error generating response"
**Solutions**: 
- Check your OpenAI API key in `.env`
- Verify you have API credits available
- Check your internet connection

#### Web Scraping Issues

**Symptom**: "Failed to scrape URL"
**Solutions**:
- Check if the website allows scraping (robots.txt)
- Try the dynamic scraper for JavaScript-heavy sites
- Increase request delay for rate-limited sites

### Logs

Logs are stored in:
- `vedic_knowledge_ai.log` for the main application
- `vedic_knowledge_api.log` for the API server

To check logs:

```bash
tail -f vedic_knowledge_ai.log
```

## Advanced Features

### Sanskrit Detection and Processing

The system includes specialized processing for Sanskrit:

- Automatic detection of Sanskrit content
- Devanagari and transliterated text handling
- Sanskrit term extraction and definition
- Construction of Sanskrit terminology dictionary

### Cloud Synchronization

For backup and multi-device deployment, use cloud sync:

```python
from src.utils import CloudSyncManager

# Sync to AWS S3
sync_manager = CloudSyncManager()
sync_manager.sync_to_s3()

# Sync to all configured clouds
sync_manager.sync_to_all()
```

Configure in `.env`:

```
# AWS
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
S3_BUCKET=your_bucket_name

# GCP
GCP_PROJECT_ID=your_gcp_project_id
GCP_BUCKET_NAME=your_gcp_bucket

# Azure
AZURE_STORAGE_CONNECTION_STRING=your_azure_connection_string
AZURE_CONTAINER_NAME=your_azure_container
```

### Custom Sanskrit Term Dictionary

Create a seed dictionary of Sanskrit terms:

```python
# In Python script or interactive shell
from src.utils.exporter import DataExporter

terms = {
    "dharma": {
        "devanagari": "धर्म",
        "transliteration": "dharma",
        "definition": "Righteous duty or natural law...",
        "etymology": "From the Sanskrit root 'dhṛ' meaning 'to hold or maintain'",
        "examples": ["Bhagavad Gita 1.40", "Bhagavata Purana 1.2.6"],
        "sources": ["Sanskrit-English Dictionary", "Vedabase.com"],
        "related_terms": ["adharma", "svadharma"]
    },
    # Add more terms...
}

DataExporter.export_sanskrit_terms(terms)
```

## Cloud Deployment

### Docker Deployment

Build and run with Docker:

```bash
# Build the Docker image
docker build -t vedic-knowledge-ai .

# Run the container
docker run -p 8000:8000 -v $(pwd)/data:/app/data vedic-knowledge-ai
```

Or use Docker Compose:

```bash
docker-compose up -d
```

### Serverless Deployment

For AWS Lambda:

1. Adjust `app.py` to create a Lambda handler
2. Use Lambda Layers for dependencies
3. Configure S3 for PDF storage
4. Use API Gateway for API access

### Multi-Server Setup

For large deployments:

1. Use shared cloud storage for data
2. Deploy API servers behind a load balancer
3. Use separate servers for web scraping and document processing
4. Implement authentication for API access

## Getting Started Tutorial

Here's a step-by-step guide to get up and running quickly:

### 1. Initial Setup

```bash
# Create project directory
mkdir vedic-knowledge-ai
cd vedic-knowledge-ai

# Copy all project files to this directory
# (Ensure the directory structure matches what's described in this README)

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env to add your OpenAI API key
```

### 2. Initialize the System

```bash
# Initialize the system with default directories
python app.py init
```

### 3. Add Sample Content

```bash
# Create a sample directory
mkdir -p data/books/bhagavad-gita

# Download a sample PDF (replace with your own PDFs)
# For example, add Bhagavad Gita with commentaries to data/books/bhagavad-gita/
```

### 4. Load Documents

```bash
# Process and index documents
python app.py load
```

### 5. Verify Setup

```bash
# Check database information
python app.py info
```

You should see something like:

```
Database Information:
Document count: 123
Collection name: vedic_knowledge
Directory: ./data/db
```

### 6. Try a Sample Query

```bash
# Ask a basic question
python app.py answer "What is the meaning of dharma in Bhagavad Gita?"
```

### 7. Add Web Content

```bash
# Scrape a Vedic knowledge website
python app.py scrape "https://www.vedabase.com/en/bg/2/47"
```

### 8. Start Interactive Mode

```bash
# Start the interactive interface
python app.py interactive
```

### 9. Export a Sanskrit Terms Dictionary

```
> export terms
Exported Sanskrit terms dictionary to data/exports/summaries/sanskrit_terms_20250426123456.md
```

### 10. Generate a System Report

```
> export report
Generated system report at data/exports/reports/system_report_20250426123456.md
```

Congratulations! You now have a fully functional Vedic Knowledge AI system. Continue adding your own PDFs and trusted websites to build a comprehensive knowledge base.

## Sanskrit Term Lookup Feature

The Vedic Knowledge AI system now includes direct Sanskrit term lookup from Vedabase. This feature allows you to:

1. Search for Sanskrit terms on Vedabase.io
2. Extract the Devanagari script, meaning, and occurrences
3. Add the information to your knowledge base
4. Optionally export the data to your Sanskrit terms dictionary

### Command Line Usage

```bash
# Basic term lookup
python app.py lookup-term "ahimsa"

# Lookup with fresh data (bypass cache)
python app.py lookup-term "ahimsa" --bypass-cache

# Lookup and export to Sanskrit terms dictionary
python app.py lookup-term "ahimsa" --export
```

### Interactive Mode Usage

In interactive mode, use the `lookup` command:

```
> lookup ahimsa
Looking up term: ahimsa...

Term: ahimsa
Devanagari: अहिंसा
Definition: nonviolence, not causing pain to any living being

Occurrences (15 found):
1. SB 1.17.24: ...The personality of religion said: Then, O chaste one, I, who am religion personified, shall tell you in the presence of these great sages about the principles...
2. SB 11.19.33: ...According to scriptural injunctions, one should perform sacrifices, study the Vedas and practice tolerance, truthfulness, mental and sensory control...
3. BG 13.8-12: ...Humility; pridelessness; nonviolence; tolerance; simplicity; approaching a bona fide spiritual master; cleanliness; steadiness; self-control...
...
```

### How It Works

1. The system searches for the term on Vedabase.io's synonym search
2. It extracts the Devanagari script and definition
3. It finds all occurrences in the scriptures
4. The data is stored in the knowledge base for future queries
5. Optionally, the information is exported to your Sanskrit terms dictionary

### Advanced Uses

#### Batch Processing Terms

You can process multiple terms with a simple script:

```python
# terms_to_lookup.py
from app import VedicKnowledgeAI

terms = [
    "ahimsa", "dharma", "karma", "yoga", "atma",
    "brahman", "bhakti", "jnana", "vairagya", "veda"
]

ai = VedicKnowledgeAI()

for term in terms:
    print(f"Looking up: {term}")
    term_data = ai.lookup_sanskrit_term(term)
    print(f"Found {len(term_data.get('occurrences', []))} occurrences")

# Export all terms at once
ai.export_sanskrit_terms_dictionary()
```

#### Integration with QA System

The term lookup data enhances the system's ability to answer Sanskrit-related questions:

```bash
python app.py answer "What is the definition and significance of ahimsa in Vedic texts?"
```

The system will now incorporate both LLM knowledge and the specific references and context found through the term lookup feature.

#### Building a Custom Sanskrit Dictionary

To create a comprehensive Sanskrit dictionary:

1. Identify key terms from your texts
2. Use the lookup-term command with --export for each term
3. The exported dictionary will build up over time
4. Access the compiled dictionary in `data/exports/summaries/`

This feature significantly enhances the system's ability to work with Sanskrit terminology by providing authoritative definitions and scriptural references directly from Vedabase.