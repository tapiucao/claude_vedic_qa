
"""
PDF processing for Vedic Knowledge AI.
Handles loading, metadata extraction, and content processing for PDF files.
"""
import os
import logging
from typing import List, Dict, Any, Optional, Union 
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from tqdm import tqdm
import re

try:
    import fitz 
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    logging.info("PyMuPDF not found. Falling back to PyPDFLoader for page count (may be slower). Install with: pip install PyMuPDF")

# Assuming config is properly importable, otherwise adjust path
from ..config import PDF_DIR

# Configure logging
logger = logging.getLogger(__name__)

class VedicPDFLoader:
    """
    Enhanced PDF loader for Vedic texts with metadata extraction,
    including attempts to detect chapter and verse information from page content.
    Uses PyMuPDF for efficient page counting and potential metadata extraction if available.
    """

    def __init__(self, directory: str = PDF_DIR):
        """Initialize with directory containing PDF files."""
        if not os.path.isdir(directory):
             # Log error but allow initialization, list_available_pdfs will handle it
             logger.error(f"PDF directory not found during initialization: {directory}")
        self.directory = directory
        logger.info(f"Initialized PDF loader with directory: {self.directory}")

    def list_available_pdfs(self) -> List[str]:
        """List all available PDF files in the directory."""
        if not os.path.isdir(self.directory):
            # Logged error during init, return empty list here
            return []
        try:
            pdfs = [f for f in os.listdir(self.directory) if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(self.directory, f))]
            logger.info(f"Found {len(pdfs)} PDF files in {self.directory}")
            return pdfs
        except OSError as e: # Catch OS errors like permission denied
            logger.error(f"Error listing PDF files in {self.directory}: {e}")
            return []

    def _get_pdf_page_count(self, file_path: str) -> Optional[int]:
        """Efficiently get page count using PyMuPDF if available, else return None."""
        if HAS_PYMUPDF:
            try:
                with fitz.open(file_path) as doc:
                    return doc.page_count
            except Exception as e:
                logger.error(f"Error getting page count using PyMuPDF for {os.path.basename(file_path)}: {e}")
                return None # Fallback or indicate error
        else:
            # Cannot efficiently get page count without PyMuPDF here
            # The caller (get_pdf_statistics) will use PyPDFLoader as fallback
            return None


    def load_all_pdfs(self, limit: Optional[int] = None) -> List[Document]:
        """Load all PDFs from the directory with optional limit."""
        all_documents: List[Document] = []
        pdf_files = self.list_available_pdfs()

        if not pdf_files:
             logger.warning(f"No PDF files found or accessible in {self.directory}.")
             return []

        if limit is not None and limit > 0: # Ensure limit is positive
            pdf_files_to_load = pdf_files[:limit]
            logger.info(f"Loading limited to first {len(pdf_files_to_load)} PDFs.")
        else:
             pdf_files_to_load = pdf_files

        for pdf_file in tqdm(pdf_files_to_load, desc="Loading PDFs"):
            # Load single PDF handles its own errors and returns list
            documents = self.load_single_pdf(pdf_file)
            all_documents.extend(documents)

        logger.info(f"Loaded {len(all_documents)} total pages from {len(pdf_files_to_load)} PDF files")
        return all_documents

    def get_pdf_statistics(self) -> Dict[str, Any]:
        """Get statistics about the PDF collection, using efficient page counting if possible."""
        pdfs = self.list_available_pdfs()

        stats: Dict[str, Any] = {
            "total_files": len(pdfs),
            "total_pages": 0,
            "files_analyzed": 0,
            "files_with_errors": 0,
            "files": [] # List of dicts per file
        }

        if not pdfs:
            return stats # Return empty stats if no PDFs

        logger.info(f"Analyzing {len(pdfs)} PDF files for statistics...")
        for pdf in tqdm(pdfs, desc="Analyzing PDFs"):
            file_path = os.path.join(self.directory, pdf)
            num_pages: Union[int, str] = "Error" # Default to error state
            error_msg: Optional[str] = None

            # Try efficient page count first
            page_count = self._get_pdf_page_count(file_path)

            if page_count is not None:
                 num_pages = page_count
                 logger.debug(f"Got page count ({num_pages}) via PyMuPDF for {pdf}")
            else:
                # Fallback: Use PyPDFLoader (less efficient)
                logger.debug(f"Falling back to PyPDFLoader for page count of {pdf}")
                try:
                    loader = PyPDFLoader(file_path)
                    # load() is necessary to get page count via len() with PyPDFLoader
                    loaded_docs = loader.load() # This does the heavy lifting
                    num_pages = len(loaded_docs)
                except Exception as e:
                    error_msg = f"PyPDFLoader failed: {str(e)}"
                    logger.error(f"Error analyzing PDF {pdf} with PyPDFLoader: {error_msg}", exc_info=False) # Avoid traceback spam
                    stats["files_with_errors"] += 1


            # Update statistics
            if isinstance(num_pages, int):
                stats["total_pages"] += num_pages
            stats["files_analyzed"] += 1
            stats["files"].append({
                "filename": pdf,
                "pages": num_pages,
                "error": error_msg # Will be None if successful
            })

        logger.info(f"PDF Statistics: {stats['total_files']} files found, {stats['files_analyzed']} analyzed, "
                    f"{stats['total_pages']} total pages (approx. if errors), {stats['files_with_errors']} errors.")
        return stats

    def extract_metadata(self, file_path: str, page_number: int, page_content: str) -> Dict[str, Any]:
        """
        Extract metadata from PDF file name, properties (if available), and content.

        Args:
            file_path (str): The full path to the PDF file.
            page_number (int): The zero-based index of the page.
            page_content (str): The extracted text content of the page.

        Returns:
            Dict[str, Any]: A dictionary containing extracted metadata.
        """
        filename = os.path.basename(file_path)
        # Use os.path.splitext for cleaner extension removal
        name_without_ext = os.path.splitext(filename)[0]

        # Initialize basic metadata
        metadata: Dict[str, Any] = {
            "source": file_path, # Keep full path as unique source identifier
            "filename": filename, # Add filename separately
            "title": name_without_ext, # Default title from filename
            "page": page_number + 1,  # User-friendly 1-based page number
            "type": "pdf" # Document type
        }

        # Attempt to extract document-level metadata using PyMuPDF if available
        if HAS_PYMUPDF:
            try:
                 with fitz.open(file_path) as doc:
                      doc_meta = doc.metadata # PyMuPDF metadata dictionary
                      if doc_meta:
                           # Map PyMuPDF keys to our desired keys (adjust as needed)
                           pdf_title = doc_meta.get('title')
                           pdf_author = doc_meta.get('author')
                           pdf_subject = doc_meta.get('subject')
                           # ... potentially others like 'keywords', 'creationDate', 'modDate'

                           if pdf_title and pdf_title.strip():
                                metadata['title'] = pdf_title.strip() # Overwrite default title
                           if pdf_author and pdf_author.strip():
                                metadata['author'] = pdf_author.strip()
                           if pdf_subject and pdf_subject.strip():
                                metadata['subject'] = pdf_subject.strip()
                           # Add creation/mod date if useful
                           # metadata['creation_date'] = doc_meta.get('creationDate')
                           # metadata['modification_date'] = doc_meta.get('modDate')
            except Exception as e:
                 logger.warning(f"Could not extract document metadata using PyMuPDF for {filename}: {e}")


        # Try to detect chapter/verse information from page content
        content_metadata = self._detect_content_structure(page_content)
        if content_metadata:
            metadata.update(content_metadata) # Add chapter, verse_reference_detected, etc.

        # Remove None values before returning for cleaner storage
        return {k: v for k, v in metadata.items() if v is not None}


    def _detect_content_structure(self, text: str) -> Dict[str, Any]:
        """
        Detect chapter and potentially verse information from page text using regex.
        This is heuristic-based and depends on PDF text layout.

        Args:
            text (str): The text content of a single PDF page.

        Returns:
            Dict[str, Any]: A dictionary containing detected 'chapter' (int) or
                            'verse_reference_detected' (str).
        """
        structure_info: Dict[str, Any] = {}
        if not text: # Skip if no text
            return structure_info

        # Common chapter patterns (case-insensitive, look for boundaries)
        chapter_patterns = [
            # Novo padrão específico para o formato da sua imagem (corpo do texto)
            r'(?im)^\s*CHAPTER\s+(\d+)\s*$', # Tenta capturar "CHAPTER 15" em uma linha própria
            r'(?im)^\s*CHAPTER\s+(\d+)\s*\n\s*Prameya:', # Tenta capturar "CHAPTER 15" seguido por "Prameya:"

            # Padrões existentes (mantenha os que podem ser úteis para outros PDFs)
            r'(?im)^\s*(?:Chapter|CHAPTER|Canto|CANTO|Adhy[āa]ya|Adhyay)\s+(\d+)\b',
            r'\b(?:Chapter|Canto|Adhy[āa]ya|Adhyay)\s+(\d+)\b',
            r'\bSection\s+(\d+)\b',
        ]

        # Find the first chapter match on the page
        for pattern in chapter_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    chapter_num = int(match.group(1))
                    structure_info["chapter"] = chapter_num
                    logger.debug(f"Detected chapter {chapter_num} using pattern '{pattern}'")
                    # Break after finding the first, likely most prominent, chapter indicator
                    break
                except (ValueError, IndexError):
                    # Should not happen with \d+ but safety first
                    logger.warning(f"Regex pattern '{pattern}' matched but failed to extract integer.")
                    continue # Try next pattern if extraction failed

        # Look for patterns like "Verse X", "Text Y", "BG X.Y", "SB X.Y.Z"
        # Prioritize more specific patterns (like book references)
        verse_patterns = [
            # Specific book formats first
            r'\b(SB\s+\d{1,2}\.\d{1,2}\.\d{1,3})\b',    # Srimad Bhagavatam X.Y.Z
            r'\b(BG\s+\d{1,2}\.\d{1,2})\b',            # Bhagavad Gita X.Y
            r'\b(CC\s+[A-Za-z]+\s+\d{1,2}\.\d{1,3})\b', # Chaitanya Charitamrita Book X.Y
            # Generic numbered references
            r'\b(?:Verse|VERSE|Text|TEXT|Śloka|Sloka)\s+(\d+\.?\d*)\b', # Verse/Text X or X.Y
            r'\b(Chapter\s+\d+,\s*Verse\s+\d+)\b', # Chapter X, Verse Y format
            # Numeric patterns (less specific, place last)
            r'\b(\d{1,3}\.\d{1,3}\.\d{1,3})\b',         # X.Y.Z format
            r'\b(\d{1,2}\.\d{1,2})\b(?!\.\d)',          # X.Y format (ensure not part of X.Y.Z)
        ]

        # Find the first verse reference match on the page
        for pattern in verse_patterns:
             match = re.search(pattern, text, re.IGNORECASE) # Case-insensitive search
             if match:
                 # Capture the full matched reference string (e.g., "BG 2.13", "Verse 15", "1.2.3")
                 verse_ref = match.group(1).strip()
                 structure_info["verse_reference_detected"] = verse_ref
                 logger.debug(f"Detected verse reference '{verse_ref}' using pattern '{pattern}'")
                 # Break after finding the first, likely most prominent, reference
                 break

        return structure_info

    def load_single_pdf(self, filename: str) -> List[Document]:
        """
        Load a single PDF file, extract text page by page, and enhance with metadata.

        Args:
            filename (str): The name of the PDF file in the configured directory.

        Returns:
            List[Document]: A list of LangChain Document objects, one for each page,
                            or an empty list if loading fails.
        """
        file_path = os.path.join(self.directory, filename)
        if not os.path.isfile(file_path): # Check if it's a file
            logger.error(f"PDF file not found or is not a file: {file_path}")
            return []

        loaded_documents: List[Document] = []
        try:
            # Use LangChain's PDF loader for text extraction per page
            logger.debug(f"Loading PDF using PyPDFLoader: {file_path}")
            loader = PyPDFLoader(file_path, extract_images=False) # Disable image extraction if not needed

            # load() gives page-level documents which is suitable for metadata enrichment
            documents_from_loader: List[Document] = loader.load() # List of Documents

            if not documents_from_loader:
                 logger.warning(f"PyPDFLoader returned no documents for {filename}.")
                 return []

            # Process each page (Document) from the loader
            for i, doc in enumerate(documents_from_loader):
                page_content = doc.page_content if doc.page_content else ""

                # Extract metadata (basic file info + content structure)
                # Pass page index (i) and content
                page_metadata = self.extract_metadata(file_path, i, page_content)

                # Update the document's metadata dictionary.
                # PyPDFLoader usually adds 'source' and 'page'. We enrich/overwrite these.
                # Ensure our calculated 1-based page number is used if PyPDFLoader's differs.
                doc.metadata.update(page_metadata)
                # Make sure 'page' is correct (ours is 1-based)
                doc.metadata['page'] = page_metadata.get('page', i + 1)


                loaded_documents.append(doc)

            logger.info(f"Successfully loaded {len(loaded_documents)} pages from {filename}")
            return loaded_documents

        except Exception as e:
            # Catch potential errors from PyPDFLoader (e.g., encrypted PDFs, corrupt files)
            logger.error(f"Error loading PDF {filename} with PyPDFLoader: {str(e)}", exc_info=True)
            return [] # Return empty list on failure