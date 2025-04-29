"""
PDF processing for Vedic Knowledge AI.
Handles loading, metadata extraction, and content processing for PDF files.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from tqdm import tqdm
import re

from ..config import PDF_DIR

# Configure logging
logger = logging.getLogger(__name__)

class VedicPDFLoader:
    """Enhanced PDF loader for Vedic texts with metadata extraction."""
    
    def __init__(self, directory: str = PDF_DIR):
        """Initialize with directory containing PDF files."""
        self.directory = directory
        logger.info(f"Initialized PDF loader with directory: {self.directory}")
    
    def list_available_pdfs(self) -> List[str]:
        """List all available PDF files in the directory."""
        pdfs = [f for f in os.listdir(self.directory) if f.lower().endswith('.pdf')]
        logger.info(f"Found {len(pdfs)} PDF files in {self.directory}")
        return pdfs
    
    def load_single_pdf(self, filename: str) -> List[Document]:
        """Load a single PDF file and enhance with metadata."""
        file_path = os.path.join(self.directory, filename)
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []
        
        try:
            # Use LangChain's PDF loader
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Extract metadata
            base_metadata = self.extract_metadata(file_path)
            
            # Enhance documents with additional metadata
            for i, doc in enumerate(documents):
                # Merge base metadata with page-specific metadata
                doc.metadata.update(base_metadata)
                # Add page number if not already present
                if "page" not in doc.metadata:
                    doc.metadata["page"] = i + 1
            
            logger.info(f"Successfully loaded {len(documents)} pages from {filename}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading PDF {filename}: {str(e)}")
            return []
    
    def load_all_pdfs(self, limit: Optional[int] = None) -> List[Document]:
        """Load all PDFs from the directory with optional limit."""
        all_documents = []
        pdf_files = self.list_available_pdfs()
        
        if limit:
            pdf_files = pdf_files[:limit]
        
        for pdf_file in tqdm(pdf_files, desc="Loading PDFs"):
            documents = self.load_single_pdf(pdf_file)
            all_documents.extend(documents)
            
        logger.info(f"Loaded {len(all_documents)} total pages from {len(pdf_files)} PDF files")
        return all_documents

    def get_pdf_statistics(self) -> Dict[str, Any]:
        """Get statistics about the PDF collection."""
        pdfs = self.list_available_pdfs()
        
        stats = {
            "total_files": len(pdfs),
            "total_pages": 0,
            "files": []
        }
        
        for pdf in pdfs:
            try:
                file_path = os.path.join(self.directory, pdf)
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                
                stats["total_pages"] += len(documents)
                stats["files"].append({
                    "filename": pdf,
                    "pages": len(documents)
                })
            except Exception as e:
                logger.error(f"Error analyzing PDF {pdf}: {str(e)}")
                stats["files"].append({
                    "filename": pdf,
                    "pages": "Error",
                    "error": str(e)
                })
        
        return stats

    def extract_metadata(self, file_path: str, page_number: int, page_content: str) -> Dict[str, Any]:
        """Extract metadata from PDF file name, properties, and content."""
        # Basic metadata from filename
        filename = os.path.basename(file_path)
        name_without_ext = filename.replace('.pdf', '')
        
        # Initialize metadata
        metadata = {
            "source": file_path,
            "title": name_without_ext,
            "file_type": "pdf",
            "page": page_number + 1  # Page numbers start from 1
        }
        
        # Try to detect chapter information
        chapter_info = self._detect_chapter(page_content)
        if chapter_info:
            metadata.update(chapter_info)
        
        return metadata

    def _detect_chapter(self, text: str) -> Dict[str, Any]:
        """Detect chapter information from text."""
        chapter_info = {}
        
        # Common chapter patterns in Vedic texts
        chapter_patterns = [
            r'Chapter\s+(\d+)',
            r'CHAPTER\s+(\d+)',
            r'अध्याय\s+(\d+)',  # Sanskrit "Adhyaya"
            r'Canto\s+(\d+)',
            r'Section\s+(\d+)',
            r'Verse\s+(\d+)'
        ]
        
        # Try to match chapter patterns
        for pattern in chapter_patterns:
            match = re.search(pattern, text)
            if match:
                chapter_num = match.group(1)
                chapter_info["chapter"] = int(chapter_num)
                break
        
        # Check for section/verse information
        verse_match = re.search(r'Verse\s+(\d+)', text)
        if verse_match:
            chapter_info["verse"] = int(verse_match.group(1))
        
        return chapter_info
    
    def load_single_pdf(self, filename: str) -> List[Document]:
        """Load a single PDF file and enhance with metadata."""
        file_path = os.path.join(self.directory, filename)
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []
        
        try:
            # Use LangChain's PDF loader
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Enhanced metadata processing
            for i, doc in enumerate(documents):
                # Extract metadata including chapter detection
                metadata = self.extract_metadata(file_path, i, doc.page_content)
                
                # Update document metadata
                doc.metadata.update(metadata)
            
            logger.info(f"Successfully loaded {len(documents)} pages from {filename}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading PDF {filename}: {str(e)}")
            return []