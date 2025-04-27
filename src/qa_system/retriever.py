"""
Retrieval system for Vedic Knowledge AI.
Handles finding relevant documents and context for queries.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from ..knowledge_base.vector_store import VedicVectorStore
from ..knowledge_base.prompt_templates import select_prompt_template
from ..qa_system.gemini_interface import GeminiLLMInterface
from ..config import TOP_K_RESULTS

# Configure logging
logger = logging.getLogger(__name__)

class VedicRetriever:
    """Retrieval system for finding relevant Vedic knowledge."""
    
    def __init__(
        self,
        vector_store: VedicVectorStore,
        llm_interface: GeminiLLMInterface,
        top_k: int = TOP_K_RESULTS
    ):
        """Initialize the retriever with vector store and LLM interface."""
        self.vector_store = vector_store
        self.llm_interface = llm_interface
        self.top_k = top_k
        
        logger.info(f"Initialized Vedic retriever with top_k={top_k}")
    
    def retrieve_documents(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        k: Optional[int] = None
    ) -> List[Document]:
        """Retrieve relevant documents for a query."""
        k = k or self.top_k
        
        try:
            docs = self.vector_store.similarity_search(query=query, k=k, filter=filter_dict)
            logger.info(f"Retrieved {len(docs)} documents for query: {query}")
            return docs
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def retrieve_documents_with_scores(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """Retrieve relevant documents with similarity scores."""
        k = k or self.top_k
        
        try:
            docs_with_scores = self.vector_store.similarity_search_with_score(query=query, k=k, filter=filter_dict)
            logger.info(f"Retrieved {len(docs_with_scores)} scored documents for query: {query}")
            return docs_with_scores
        except Exception as e:
            logger.error(f"Error retrieving documents with scores: {str(e)}")
            return []
    
    def get_context_for_query(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        k: Optional[int] = None
    ) -> str:
        """Get concatenated context from relevant documents."""
        docs = self.retrieve_documents(query, filter_dict, k)
        
        if not docs:
            return ""
        
        # Concatenate document contents with separators
        context_parts = []
        for i, doc in enumerate(docs):
            # Extract source information for citation
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            doc_type = doc.metadata.get("type", "document")
            
            # Format citation based on document type
            if doc_type == "website":
                citation = f"[Source {i+1}: {source}]"
            elif page:
                citation = f"[Source {i+1}: {source}, page {page}]"
            else:
                citation = f"[Source {i+1}: {source}]"
            
            # Add document content with citation
            context_parts.append(f"{doc.page_content}\n{citation}")
        
        return "\n\n".join(context_parts)
    
    def _extract_relevant_content(self, docs: List[Document], query: str) -> str:
        """Extract the most relevant content from documents as a fallback."""
        if not docs:
            return "No relevant information found."
        
        # Create a summary from the documents
        summary_parts = []
        seen_sources = set()
        
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            
            # Avoid duplicate sources
            source_key = f"{source}_{page}"
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            
            # Add a brief excerpt (first 100-150 chars) from the document
            content = doc.page_content.strip()
            if len(content) > 150:
                content = content[:150] + "..."
                
            # Format with source
            if page:
                summary_parts.append(f"From {source} (page {page}): {content}")
            else:
                summary_parts.append(f"From {source}: {content}")
                
            # Limit to top 3 sources
            if len(summary_parts) >= 3:
                break
        
        return "\n\n".join(summary_parts)
    
    def answer_query(self, query: str, filter_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Answer a query using retrieved documents."""
        # Retrieve documents
        docs = self.retrieve_documents(query, filter_dict)
        
        if not docs:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "documents": []
            }
        
        # Prepare context
        context = self.get_context_for_query(query, filter_dict)
        
        # Select appropriate prompt template based on query
        prompt_template = select_prompt_template(query)
        
        # Generate answer
        answer = self.llm_interface.generate_response(
            query=query,
            context=context,
            system_prompt=prompt_template.template
        )
        
        # Check if there was an error with the LLM
        if answer.startswith("Error generating response:"):
            # Fall back to extracting relevant content directly from documents
            logger.warning(f"LLM error occurred, using fallback mechanism: {answer}")
            answer = self._create_fallback_answer(query, docs)
        
        # Extract source information
        sources = []
        for doc in docs:
            source = {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", ""),
                "type": doc.metadata.get("type", "document"),
                "title": doc.metadata.get("title", "")
            }
            sources.append(source)
        
        result = {
            "answer": answer,
            "sources": sources,
            "documents": docs
        }
        
        return result
    
    def _create_fallback_answer(self, query: str, docs: List[Document]) -> str:
        """Create a fallback answer when LLM fails."""
        # Extract information about the Sanskrit term from documents
        term = query.replace("What is the meaning of", "").replace("?", "").strip()
        
        fallback_answer = f"I found information about '{term}' but encountered an issue with the AI model. "
        fallback_answer += "Here's what I can tell you based on the retrieved documents:\n\n"
        
        # Check if there are definitions in the docs
        definition_found = False
        
        for doc in docs:
            content = doc.page_content.lower()
            term_lower = term.lower()
            
            # Look for phrases that might contain definitions
            definition_phrases = [
                f"{term_lower} means",
                f"{term_lower} is defined as",
                f"{term_lower} refers to",
                f"the meaning of {term_lower}",
                f"definition of {term_lower}"
            ]
            
            for phrase in definition_phrases:
                if phrase in content:
                    # Find the sentence containing the phrase
                    sentences = content.split('.')
                    for sentence in sentences:
                        if phrase in sentence:
                            cleaned = sentence.strip()
                            if cleaned:
                                fallback_answer += f"• {cleaned.capitalize()}.\n"
                                definition_found = True
                                break
            
            # If we found at least one definition, that's enough
            if definition_found:
                break
        
        # If no clear definition was found, extract relevant passages
        if not definition_found:
            fallback_answer += self._extract_relevant_content(docs, query)
        
        fallback_answer += "\n\nPlease try your query again later when the AI model is available."
        
        return fallback_answer