"""
Text splitting utilities specialized for Vedic and Sanskrit content.
Handles chunking text while preserving important context.
"""
import re
import logging
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from ..config import CHUNK_SIZE, CHUNK_OVERLAP

# Configure logging
logger = logging.getLogger(__name__)

class VedicTextSplitter:
    """Text splitter optimized for Vedic texts and Sanskrit content."""
    
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP
    ):
        """Initialize with configurable chunk size and overlap."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Custom separators for Vedic texts
        # Priority order: verse markers, paragraphs, sentences, etc.
        self.separators = [
            # Verse markers (examples)
            "\n\n", # Double line breaks often separate verses or paragraphs
            "\nVerse ", # Common verse indicator
            "\nŚloka ", # Sanskrit verse indicator
            # Chapter markers
            "\nChapter ",
            "\nAdhyāya ",
            # Sentence endings
            "। ", # Sanskrit sentence ending (danda)
            "॥", # Sanskrit double danda
            ". ", # English period
            "? ", # Question mark
            "! ", # Exclamation mark
            # Word boundaries as last resort
            " ",
            ""
        ]
        
        logger.info(f"Initialized text splitter with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    def _is_sanskrit_verse(self, text: str) -> bool:
        """Check if text appears to be a Sanskrit verse."""
        # Simple heuristic: presence of Sanskrit-specific characters
        sanskrit_chars = r'[ऄ-औक-ह\u093Cा-ौ्ंःँॐॠऱऴ]'
        return bool(re.search(sanskrit_chars, text))
    
    def _add_verse_context(self, chunks: List[Document]) -> List[Document]:
        """Add context to chunks containing verses."""
        enhanced_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Check if chunk contains Sanskrit verses
            if self._is_sanskrit_verse(chunk.page_content):
                # Try to identify verse reference from content or previous chunk
                verse_ref = self._extract_verse_reference(chunk.page_content)
                
                if verse_ref and "verse_reference" not in chunk.metadata:
                    chunk.metadata["verse_reference"] = verse_ref
                    chunk.metadata["contains_sanskrit"] = True
            
            enhanced_chunks.append(chunk)
        
        return enhanced_chunks
    
    def _extract_verse_reference(self, text: str) -> str:
        """Extract verse reference information if present."""
        # Simple regex patterns for common verse reference formats
        patterns = [
            r'(?:Chapter|Adhyāya)\s+(\d+),?\s+(?:Verse|Śloka)\s+(\d+)',  # Chapter X, Verse Y
            r'(\d+)\.(\d+)\.(\d+)',  # X.Y.Z format
            r'BG\s+(\d+)\.(\d+)',    # BG X.Y (Bhagavad Gita)
            r'SB\s+(\d+)\.(\d+)\.(\d+)',  # SB X.Y.Z (Srimad Bhagavatam)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        
        return ""
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks while preserving context and metadata."""
        if not documents:
            logger.warning("No documents provided for splitting")
            return []
        
        # Create the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )
        
        # Split the documents
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
        
        # Enhance chunks with chapter context
        enhanced_chunks = self._enhance_chunks_with_context(chunks)
        
        return enhanced_chunks

    def _enhance_chunks_with_context(self, chunks: List[Document]) -> List[Document]:
        """Add context to chunks and ensure metadata preservation."""
        enhanced_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Ensure chapter metadata is preserved
            if "chapter" in chunk.metadata:
                # Add a readable chapter reference
                book_title = chunk.metadata.get("title", "Unknown")
                chapter_num = chunk.metadata.get("chapter")
                chunk.metadata["chapter_reference"] = f"{book_title} Chapter {chapter_num}"
            
            # Check if chunk contains Sanskrit verses
            if self._is_sanskrit_verse(chunk.page_content):
                # Try to identify verse reference
                verse_ref = self._extract_verse_reference(chunk.page_content)
                
                if verse_ref and "verse_reference" not in chunk.metadata:
                    chunk.metadata["verse_reference"] = verse_ref
                    chunk.metadata["contains_sanskrit"] = True
            
            enhanced_chunks.append(chunk)
        
        return enhanced_chunks
    
    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """Split a single text string into chunks."""
        if not metadata:
            metadata = {}
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )
        
        # Create a document and split it
        doc = Document(page_content=text, metadata=metadata)
        chunks = text_splitter.split_documents([doc])
        
        # Enhance chunks with verse context
        enhanced_chunks = self._add_verse_context(chunks)
        
        return enhanced_chunks