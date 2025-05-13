# src/document_processor/text_splitter.py
# Applied changes: Item 2 (Remove duplicate split_documents), Item 10 (Add comment), Item 11 (Type Hinting)
"""
Text splitting utilities specialized for Vedic and Sanskrit content.
Handles chunking text while preserving important context like chapter/verse references.
"""
import os
import re
import logging
from typing import List, Dict, Any, Optional, Sequence
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain_core.documents import Document # Use langchain_core

# Assuming config is properly importable
# from ..config import CHUNK_SIZE, CHUNK_OVERLAP
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


# Configure logging
logger = logging.getLogger(__name__)

class VedicTextSplitter:
    """
    A text splitter optimized for Vedic texts and Sanskrit content, aiming to split
    along meaningful boundaries like paragraphs, verses, or dandas, while preserving
    contextual metadata like chapter or verse references.
    """
    
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        separators: Optional[Sequence[str]] = None
    ):
        """
        Initialize the VedicTextSplitter.

        Args:
            chunk_size (int): The target size for each text chunk (in characters).
            chunk_overlap (int): The number of characters to overlap between chunks.
            separators (Optional[Sequence[str]]): Custom list of separators to split on,
                ordered by priority. If None, uses default Vedic/Sanskrit separators.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Custom separators optimized for Vedic/Sanskrit texts.
        # Order matters: more significant breaks first.
        self.separators = separators or [
            # Paragraphs / Major breaks
            "\n\n\n", # Triple newline
            "\n\n",   # Double newline (common paragraph/verse separator)
            
            # Verse/Sloka markers (examples, may need expansion)
            "\n॥",    # Double Danda often marks end of verse/section
            "।\n",    # Danda followed by newline
            "\nVerse ", # Common English verse indicator
            "\nŚloka ", # Sanskrit verse indicator
            "\nText ", # Another common indicator
            
            # Chapter/Section markers
            "\nChapter ",
            "\nAdhyāya ",
            "\nCanto ",
            
            # Sentence endings
            "। ",     # Danda followed by space (Sanskrit sentence end)
            ". ",     # English period followed by space
            "? ",     # Question mark
            "! ",     # Exclamation mark
            "\n",     # Single newline (less reliable sentence break)
            
            # Fallbacks
            " ",      # Space
            "",       # Empty string (split by character if all else fails)
        ]
        
        # Initialize the underlying LangChain splitter
        self._langchain_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            # keep_separator=True # Consider keeping separators for context if needed
        )
        
        logger.info(f"Initialized VedicTextSplitter with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        logger.debug(f"Using separators: {self.separators}")

    def _is_sanskrit_like(self, text: str) -> bool:
        """Rudimentary check if text snippet contains Sanskrit characters (Devanagari or IAST)."""
        # Using regex patterns similar to SanskritProcessor for consistency
        sanskrit_chars = r'[ऀ-ॿāīūṛṝḷḹṁḥśṣṭḍṇṅñ]' 
        return bool(re.search(sanskrit_chars, text))

    def _extract_contextual_reference(self, text: str) -> Optional[str]:
        """
        Attempt to extract a contextual reference (like Chapter/Verse number)
        from the text snippet using common patterns.

        Args:
            text (str): The text snippet (e.g., a chunk or surrounding text).

        Returns:
            Optional[str]: The extracted reference string (e.g., "BG 2.13", "SB 1.2.3"), or None.
        """
        # Patterns for common reference formats (expand as needed)
        # Prioritize more specific patterns first
        patterns = [
            r'\b(SB\s+\d{1,2}\.\d{1,2}\.\d{1,3})\b',    # SB X.Y.Z (Srimad Bhagavatam)
            r'\b(BG\s+\d{1,2}\.\d{1,2})\b',            # BG X.Y (Bhagavad Gita)
            r'\b(CC\s+[A-Za-z]+\s+\d{1,2}\.\d{1,3})\b', # CC Madhya X.Y (Chaitanya Charitamrita)
            r'\b(Chapter|Adhyāya)\s+(\d+),\s*(Verse|Śloka|Text)\s+(\d+)\b', # Chapter X, Verse Y
            r'\b(\d{1,3}\.\d{1,3}\.\d{1,3})\b',         # Generic X.Y.Z format 
            r'\b(\d{1,3}\.\d{1,3})\b',                 # Generic X.Y format
            r'\b(Verse|Śloka|Text)\s+(\d+)\b',         # Standalone Verse/Sloka/Text X
        ]
        
        # Look for these patterns anywhere in the text
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Return the matched reference string (e.g., "BG 2.13", "SB 1.2.3", "Verse 15")
                ref = match.group(0) # Get the full matched text
                logger.debug(f"Extracted reference '{ref}' using pattern '{pattern}'")
                return ref.strip()
                
        return None

    def _enhance_chunk_metadata(self, chunk: Document, original_doc: Optional[Document] = None, previous_chunk_text: Optional[str] = None) -> Document:
        """
        Enhances the metadata of a single chunk. Attempts to add chapter/verse context
        and flags if Sanskrit-like content is present.

        Args:
            chunk (Document): The chunk document generated by the splitter.
            original_doc (Optional[Document]): The original document the chunk came from.
            previous_chunk_text (Optional[str]): The text content of the previous chunk for context.

        Returns:
            Document: The chunk document with potentially enhanced metadata.
        """
        # Inherit metadata from the original document if available
        if original_doc:
            chunk.metadata.update(original_doc.metadata) 
            
        # Check if chunk content is Sanskrit-like
        chunk_content = chunk.page_content if chunk.page_content else ""
        if self._is_sanskrit_like(chunk_content):
            chunk.metadata["contains_sanskrit"] = True
            
            # Attempt to extract verse reference directly from the chunk
            verse_ref = self._extract_contextual_reference(chunk_content)
            if verse_ref:
                chunk.metadata["verse_reference"] = verse_ref
                
            # If no reference in chunk, check previous chunk for context (less reliable)
            elif previous_chunk_text:
                 prev_ref = self._extract_contextual_reference(previous_chunk_text)
                 if prev_ref:
                     chunk.metadata["contextual_reference_from_previous"] = prev_ref
                     logger.debug(f"Adding contextual reference '{prev_ref}' from previous chunk.")

        # Create a readable chapter reference if 'title' and 'chapter' metadata exist
        book_title = chunk.metadata.get("title")
        chapter_num = chunk.metadata.get("chapter")
        if book_title and chapter_num:
            chunk.metadata["chapter_reference"] = f"{book_title}, Chapter {chapter_num}"
            
        return chunk

    # Removed duplicate split_documents method. This is the correct one.
    def split_documents(self, documents: Sequence[Document]) -> List[Document]:
        """
        Split a sequence of LangChain Documents into smaller chunks using Vedic/Sanskrit-aware logic.

        Args:
            documents (Sequence[Document]): A list or other sequence of LangChain Documents.

        Returns:
            List[Document]: A list of smaller chunk Documents with potentially enhanced metadata.
        """
        if not documents:
            logger.warning("No documents provided for splitting.")
            return []

        all_chunks: List[Document] = []
        previous_chunk_text: Optional[str] = None 

        for doc in documents:
             # Use the LangChain splitter to get initial chunks for this document
             # split_documents inherently preserves metadata from the input doc
            doc_chunks = self._langchain_splitter.split_documents([doc])
            
            logger.debug(f"Split document '{doc.metadata.get('source', 'Unknown')}' into {len(doc_chunks)} initial chunks.")

            # Enhance metadata for each chunk from this document
            enhanced_doc_chunks: List[Document] = []
            for i, chunk in enumerate(doc_chunks):
                 # Pass original doc for metadata inheritance and previous chunk text for context
                 enhanced_chunk = self._enhance_chunk_metadata(chunk, original_doc=doc, previous_chunk_text=previous_chunk_text)
                 enhanced_doc_chunks.append(enhanced_chunk)
                 # Update previous chunk text for the next iteration
                 previous_chunk_text = chunk.page_content 

            all_chunks.extend(enhanced_doc_chunks)
            # Reset previous chunk text when moving to a new document
            previous_chunk_text = None 
        
        logger.info(f"Split {len(documents)} documents into a total of {len(all_chunks)} enhanced chunks.")
        return all_chunks

    def split_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Split a single text string into chunks.

        Args:
            text (str): The text to split.
            metadata (Optional[Dict[str, Any]]): Optional metadata to associate with the text.

        Returns:
            List[Document]: A list of chunk Documents derived from the text.
        """
        if not text:
             logger.warning("Empty text provided for splitting.")
             return []
             
        doc_metadata = metadata or {}
        
        # Create a temporary Document to use the split_documents method
        temp_doc = Document(page_content=text, metadata=doc_metadata)
        
        # Use the main splitting logic
        chunks = self.split_documents([temp_doc])
        
        logger.info(f"Split text into {len(chunks)} chunks.")
        return chunks