# src/document_processor/pdf_loader.py
# Applied changes: Item 2 (Remove duplicate load_single_pdf), Item 10 (Add docstring), Item 11 (Type Hinting)
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

# Assuming config is properly importable, otherwise adjust path
# from ..config import PDF_DIR 
PDF_DIR = os.getenv("PDF_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "books"))


# Configure logging
logger = logging.getLogger(__name__)

class VedicPDFLoader:
    """
    Enhanced PDF loader for Vedic texts with metadata extraction, 
    including attempts to detect chapter and verse information from page content.
    """
    
    def __init__(self, directory: str = PDF_DIR):
        """Initialize with directory containing PDF files."""
        self.directory = directory
        logger.info(f"Initialized PDF loader with directory: {self.directory}")
    
    def list_available_pdfs(self) -> List[str]:
        """List all available PDF files in the directory."""
        if not os.path.isdir(self.directory):
            logger.error(f"PDF directory not found: {self.directory}")
            return []
        try:
            pdfs = [f for f in os.listdir(self.directory) if f.lower().endswith('.pdf')]
            logger.info(f"Found {len(pdfs)} PDF files in {self.directory}")
            return pdfs
        except Exception as e:
            logger.error(f"Error listing PDF files in {self.directory}: {e}")
            return []
            
    def load_all_pdfs(self, limit: Optional[int] = None) -> List[Document]:
        """Load all PDFs from the directory with optional limit."""
        all_documents: List[Document] = []
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
        
        stats: Dict[str, Any] = {
            "total_files": len(pdfs),
            "total_pages": 0,
            "files": []
        }
        
        for pdf in pdfs:
            file_path = os.path.join(self.directory, pdf)
            try:
                # Use LangChain's PDF loader just to count pages quickly
                loader = PyPDFLoader(file_path)
                # load() can be slow, use lazy_load or just get page count if possible
                # PyPDFLoader loads all pages into memory, which might be inefficient here.
                # Consider using PyMuPDF directly for faster page count if performance is an issue.
                num_pages = len(loader.load()) # Simple but potentially slow for large PDFs
                
                stats["total_pages"] += num_pages
                stats["files"].append({
                    "filename": pdf,
                    "pages": num_pages
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
        """
        Extract metadata from PDF file name, properties, and content.

        Args:
            file_path (str): The full path to the PDF file.
            page_number (int): The zero-based index of the page.
            page_content (str): The extracted text content of the page.

        Returns:
            Dict[str, Any]: A dictionary containing extracted metadata.
        """
        # Basic metadata from filename
        filename = os.path.basename(file_path)
        name_without_ext = os.path.splitext(filename)[0] # Use os.path.splitext
        
        # Initialize metadata
        metadata: Dict[str, Any] = {
            "source": file_path,
            "title": name_without_ext, # Default title from filename
            "file_type": "pdf",
            "page": page_number + 1  # Page numbers usually start from 1 for users
        }
        
        # Try to detect chapter information from page content
        chapter_info = self._detect_chapter(page_content)
        if chapter_info:
            metadata.update(chapter_info)
        
        # TODO: Future enhancement: Extract metadata from PDF properties (Author, Title, etc.)
        # using a library like PyMuPDF or PyPDF2 if PyPDFLoader doesn't expose it sufficiently.
        # Example using PyPDFLoader's document metadata (if it contains useful info):
        # try:
        #     loader = PyPDFLoader(file_path) 
        #     # Load just the first page to potentially get document-level metadata faster
        #     # This depends on how PyPDFLoader populates metadata
        #     first_page_docs = loader.load() 
        #     if first_page_docs:
        #         doc_metadata = first_page_docs[0].metadata 
        #         if 'title' in doc_metadata and doc_metadata['title']:
        #             metadata['title'] = doc_metadata['title'] # Overwrite filename title if PDF title exists
        #         # Add other relevant metadata like 'author', 'subject' etc. if needed
        # except Exception as e:
        #      logger.warning(f"Could not extract document-level metadata for {filename}: {e}")

        return metadata

    def _detect_chapter(self, text: str) -> Dict[str, Any]:
        """
        Detect chapter and verse information from text using regex patterns.
        This is heuristic-based and might not be accurate for all PDF layouts.

        Args:
            text (str): The text content of a single PDF page.

        Returns:
            Dict[str, Any]: A dictionary containing detected 'chapter' or 'verse' numbers.
        """
        chapter_info: Dict[str, Any] = {}
        
        # Common chapter patterns in Vedic texts (case-insensitive matching)
        # Prioritize patterns that appear at the start of a line or with clear boundaries
        chapter_patterns = [
            r'(?i)^\s*Chapter\s+(\d+)',         # Chapter X at start of line
            r'(?i)\bCHAPTER\s+(\d+)\b',         # CHAPTER X as a whole word
            r'(?i)अध्याय\s+(\d+)',              # Sanskrit "Adhyaya"
            r'(?i)^\s*Canto\s+(\d+)',           # Canto X at start of line
            r'(?i)\bSection\s+(\d+)\b',         # Section X
        ]
        
        # Try to match chapter patterns - find first match on page
        for pattern in chapter_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    chapter_num = int(match.group(1))
                    chapter_info["chapter"] = chapter_num
                    # Break after finding the first chapter indicator on the page
                    break 
                except (ValueError, IndexError):
                    logger.warning(f"Regex pattern {pattern} matched but failed to extract integer.")
                    continue # Try next pattern

        # Check for verse information (can appear independently or within a chapter)
        # Look for patterns like "Verse X" or "X.Y.Z" which often denote verse numbers
        verse_patterns = [
            r'(?i)\bVerse\s+(\d+)\b',        # Verse X
            r'(?i)\bText\s+(\d+)\b',         # Text X (common in some translations)
            r'\b(\d{1,3}\.\d{1,3}\.\d{1,3})\b', # Match X.Y.Z format (e.g., SB 1.2.3)
            r'\b(\d{1,3}\.\d{1,3})\b',        # Match X.Y format (e.g., BG 2.13) - place after X.Y.Z
        ]
        for pattern in verse_patterns:
             match = re.search(pattern, text)
             if match:
                 verse_ref = match.group(1) # Capture the full match or the number
                 # Store the matched reference string - might be '15' or '2.13' etc.
                 chapter_info["verse_reference_detected"] = verse_ref 
                 # Try to get just the last number as a potential verse number if applicable
                 if '.' not in verse_ref:
                      try:
                           chapter_info["verse"] = int(verse_ref)
                      except ValueError:
                           pass # Keep the string reference
                 # Break after finding the first verse indicator
                 break
        
        return chapter_info
    
    # Removed the duplicate load_single_pdf method. This is the correct one.
    def load_single_pdf(self, filename: str) -> List[Document]:
        """
        Load a single PDF file, extract text page by page, and enhance with metadata.

        Args:
            filename (str): The name of the PDF file in the configured directory.

        Returns:
            List[Document]: A list of LangChain Document objects, one for each page.
        """
        file_path = os.path.join(self.directory, filename)
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []
        
        loaded_documents: List[Document] = []
        try:
            # Use LangChain's PDF loader
            logger.debug(f"Loading PDF: {file_path}")
            loader = PyPDFLoader(file_path)
            # load_and_split() might be better if you want splitter logic applied immediately
            # but load() gives page-level control for metadata extraction first.
            documents = loader.load() 
            
            # Enhanced metadata processing for each page (Document)
            for i, doc in enumerate(documents):
                # Extract metadata including chapter detection from this page's content
                page_content = doc.page_content if doc.page_content else ""
                metadata = self.extract_metadata(file_path, i, page_content)
                
                # Update the document's metadata dictionary
                # PyPDFLoader adds 'source' and 'page', we add/overwrite others
                doc.metadata.update(metadata) 
                loaded_documents.append(doc)

            logger.info(f"Successfully loaded {len(loaded_documents)} pages from {filename}")
            return loaded_documents
            
        except Exception as e:
            # Catch specific exceptions if possible (e.g., from PyPDF2)
            logger.error(f"Error loading PDF {filename}: {str(e)}", exc_info=True) # Log traceback
            return [] # Return empty list on failure