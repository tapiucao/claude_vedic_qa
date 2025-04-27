"""
Citation utilities for Vedic Knowledge AI.
Handles source attribution and citation formatting.
"""
import logging
import os
import re
from typing import List, Dict, Any, Optional
from langchain.docstore.document import Document

# Configure logging
logger = logging.getLogger(__name__)

class CitationManager:
    """Manager for handling citations and source attribution."""
    
    def __init__(self):
        """Initialize the citation manager."""
        logger.info("Initialized citation manager")
    
    def format_citation(self, doc: Document) -> str:
        """Format a citation for a document."""
        # Extract metadata
        source = doc.metadata.get("source", "Unknown source")
        doc_type = doc.metadata.get("type", "document")
        page = doc.metadata.get("page", "")
        title = doc.metadata.get("title", "")
        url = doc.metadata.get("url", "")
        
        # Format citation based on document type
        if doc_type == "website":
            # For website sources
            domain = self._extract_domain(url or source)
            if title:
                return f"{title} ({domain})"
            else:
                return domain
        
        elif doc_type == "pdf" or source.lower().endswith(".pdf"):
            # For PDF documents
            filename = os.path.basename(source)
            name_without_ext = filename.replace(".pdf", "")
            
            if title:
                citation = title
            else:
                citation = name_without_ext
            
            # Add page number if available
            if page:
                citation += f", page {page}"
            
            return citation
        
        else:
            # Default format for other types
            if title:
                return title
            return source
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain name from URL."""
        # Simple domain extraction
        match = re.search(r'https?://(?:www\.)?([^/]+)', url)
        if match:
            return match.group(1)
        return url
    
    def format_citations_for_docs(self, docs: List[Document]) -> List[str]:
        """Format citations for a list of documents."""
        return [self.format_citation(doc) for doc in docs]
    
    def format_citation_list(self, docs: List[Document]) -> str:
        """Format a list of citations as a string."""
        citations = self.format_citations_for_docs(docs)
        
        if not citations:
            return "No sources"
        
        if len(citations) == 1:
            return f"Source: {citations[0]}"
        
        # Format as numbered list
        citation_list = "\n".join(f"{i+1}. {citation}" for i, citation in enumerate(citations))
        return f"Sources:\n{citation_list}"
    
    def format_inline_citations(self, docs: List[Document]) -> Dict[str, str]:
        """Format inline citation keys for documents."""
        # Create a mapping of document ID to citation key
        citation_keys = {}
        
        for i, doc in enumerate(docs):
            # Generate a simple citation key
            key = f"[{i+1}]"
            doc_id = doc.metadata.get("source", str(i))
            citation_keys[doc_id] = key
        
        return citation_keys
    
    def add_citations_to_text(self, text: str, docs: List[Document]) -> str:
        """Add citation references to text based on content matching."""
        if not text or not docs:
            return text
        
        # Create citation keys
        citation_keys = self.format_inline_citations(docs)
        
        # Create a dictionary mapping content excerpts to citation keys
        content_citations = {}
        for doc in docs:
            doc_id = doc.metadata.get("source", "")
            if doc_id in citation_keys:
                # Get content excerpts (sentences or key phrases)
                excerpts = self._extract_key_excerpts(doc.page_content)
                for excerpt in excerpts:
                    if excerpt in content_citations:
                        # Append this citation to existing ones
                        if citation_keys[doc_id] not in content_citations[excerpt]:
                            content_citations[excerpt] += f" {citation_keys[doc_id]}"
                    else:
                        content_citations[excerpt] = citation_keys[doc_id]
        
        # Sort excerpts by length (descending) to match longer phrases first
        sorted_excerpts = sorted(content_citations.keys(), key=len, reverse=True)
        
        # Replace content with citations
        cited_text = text
        for excerpt in sorted_excerpts:
            if len(excerpt) > 10:  # Only replace substantial excerpts
                citation = content_citations[excerpt]
                # Only replace if the excerpt exists in the text
                if excerpt in cited_text:
                    cited_text = cited_text.replace(excerpt, f"{excerpt} {citation}")
        
        return cited_text
    
    def _extract_key_excerpts(self, text: str, max_excerpts: int = 5) -> List[str]:
        """Extract key excerpts (sentences or phrases) from text."""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?।॥])\s+', text)
        
        # Filter out very short sentences
        sentences = [s for s in sentences if len(s) > 20]
        
        # Limit the number of excerpts
        return sentences[:max_excerpts]
    
    def generate_bibliography(self, docs: List[Document]) -> str:
        """Generate a formal bibliography from documents."""
        if not docs:
            return "No references"
        
        bibliography = []
        
        for i, doc in enumerate(docs):
            # Extract metadata
            source = doc.metadata.get("source", "Unknown source")
            doc_type = doc.metadata.get("type", "document")
            title = doc.metadata.get("title", "")
            url = doc.metadata.get("url", "")
            domain = self._extract_domain(url or source)
            
            # Format based on document type
            if doc_type == "website":
                if title:
                    entry = f"{i+1}. {title}. Retrieved from {url or domain}"
                else:
                    entry = f"{i+1}. {url or domain}"
            
            elif doc_type == "pdf" or source.lower().endswith(".pdf"):
                filename = os.path.basename(source)
                name_without_ext = filename.replace(".pdf", "")
                
                if title:
                    entry = f"{i+1}. {title} ({name_without_ext})"
                else:
                    entry = f"{i+1}. {name_without_ext}"
            
            else:
                # Default format
                entry = f"{i+1}. {title or source}"
            
            bibliography.append(entry)
        
        return "References:\n" + "\n".join(bibliography)