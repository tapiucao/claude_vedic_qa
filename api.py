"""
API interface for Vedic Knowledge AI.
Provides a RESTful API for accessing the system's functionality.
"""
import os
import sys
import logging
from typing import List, Dict, Any, Optional

# FastAPI imports
from fastapi import FastAPI, HTTPException, Query, Depends, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel, Field

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import VedicKnowledgeAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('vedic_knowledge_api.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize the FastAPI app
app = FastAPI(
    title="Vedic Knowledge AI API",
    description="API for accessing Vedic and Gaudiya Vaishnava knowledge",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize the AI system
ai = VedicKnowledgeAI()

# Define request and response models
class QuestionRequest(BaseModel):
    question: str = Field(..., description="The question to answer")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional filters for document retrieval")

class TermRequest(BaseModel):
    term: str = Field(..., description="The Sanskrit term to explain")

class VerseRequest(BaseModel):
    verse: str = Field(..., description="The verse text to explain")
    reference: Optional[str] = Field(None, description="Optional reference (e.g., 'Bhagavad Gita 2.47')")

class ScrapeRequest(BaseModel):
    url: str = Field(..., description="The URL to scrape")

class WebsiteRequest(BaseModel):
    url: str = Field(..., description="The website URL to add")
    scrape_now: bool = Field(False, description="Whether to scrape the website immediately")

# API endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Vedic Knowledge AI API",
        "version": "1.0.0",
        "description": "API for accessing Vedic and Gaudiya Vaishnava knowledge"
    }

@app.post("/answer")
async def answer_question(request: QuestionRequest):
    """Answer a question using the knowledge base."""
    try:
        result = ai.answer_question(request.question, request.filters)
        
        # Clean up the response (remove document objects)
        if "documents" in result:
            del result["documents"]
        
        return result
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain/term")
async def explain_term(request: TermRequest):
    """Explain a Sanskrit term."""
    try:
        result = ai.explain_sanskrit_term(request.term)
        
        # Clean up the response
        if "documents" in result:
            del result["documents"]
        
        return result
    except Exception as e:
        logger.error(f"Error explaining term: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain/verse")
async def explain_verse(request: VerseRequest):
    """Explain a verse."""
    try:
        result = ai.explain_verse(request.verse, request.reference)
        
        # Clean up the response
        if "documents" in result:
            del result["documents"]
        
        return result
    except Exception as e:
        logger.error(f"Error explaining verse: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scrape")
async def scrape_website(request: ScrapeRequest):
    """Scrape a website and add to knowledge base."""
    try:
        success = ai.scrape_website(request.url)
        return {"success": success, "url": request.url}
    except Exception as e:
        logger.error(f"Error scraping website: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/websites/add")
async def add_website(request: WebsiteRequest):
    """Add a website to the scraping list."""
    try:
        success = ai.add_website(request.url, request.scrape_now)
        return {"success": success, "url": request.url}
    except Exception as e:
        logger.error(f"Error adding website: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/scheduler/status")
async def get_scheduler_status():
    """Get the status of the scraping scheduler."""
    try:
        status = ai.scraping_scheduler.get_status()
        return status
    except Exception as e:
        logger.error(f"Error getting scheduler status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scheduler/start")
async def start_scheduler(immediate: bool = False):
    """Start the scraping scheduler."""
    try:
        ai.start_scraping(immediate=immediate)
        return {"status": "started", "immediate": immediate}
    except Exception as e:
        logger.error(f"Error starting scheduler: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scheduler/stop")
async def stop_scheduler():
    """Stop the scraping scheduler."""
    try:
        ai.stop_scraping()
        return {"status": "stopped"}
    except Exception as e:
        logger.error(f"Error stopping scheduler: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/database/info")
async def get_database_info():
    """Get information about the vector database."""
    try:
        info = ai.get_database_info()
        return info
    except Exception as e:
        logger.error(f"Error getting database info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF file to the system."""
    try:
        # Ensure file is a PDF
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Create the PDF directory if it doesn't exist
        os.makedirs(ai.pdf_dir, exist_ok=True)
        
        # Save the file
        file_path = os.path.join(ai.pdf_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Process the file
        pdf_loader = ai.VedicPDFLoader(directory=ai.pdf_dir)
        documents = pdf_loader.load_single_pdf(file.filename)
        
        # Split documents into chunks
        text_splitter = ai.VedicTextSplitter()
        chunks = text_splitter.split_documents(documents)
        
        # Add to vector store
        ai.vector_store.add_documents(chunks)
        
        return {
            "success": True,
            "filename": file.filename,
            "chunks_added": len(chunks)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def start_api(host: str = "0.0.0.0", port: int = 8000):
    """Start the API server."""
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_api()