"""
Retrieval system for Vedic Knowledge AI.
Handles finding relevant documents and context for queries.
"""
import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from langchain.docstore.document import Document

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
    
    def parse_filter_dict(self, filter_dict: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse and enhance filter dictionary with chapter detection."""
        if not filter_dict:
            return {}
        
        enhanced_filter = filter_dict.copy()
        
        # Check for chapter references in text format
        if isinstance(filter_dict.get('chapter_reference'), str):
            chapter_ref = filter_dict['chapter_reference']
            
            # Try to extract chapter numbers from references like "Chapter 2" or "Bhagavad Gita Chapter 2"
            chapter_match = re.search(r'Chapter\s+(\d+)', chapter_ref)
            if chapter_match:
                enhanced_filter['chapter'] = int(chapter_match.group(1))
            
            # Try to extract canto numbers
            canto_match = re.search(r'Canto\s+(\d+)', chapter_ref)
            if canto_match:
                enhanced_filter['canto'] = int(canto_match.group(1))
        
        # Check for text queries about specific chapters
        if 'text_query' in filter_dict:
            text_query = filter_dict['text_query']
            
            # Extract chapter numbers from text queries like "tell me about chapter 2"
            chapter_match = re.search(r'chapter\s+(\d+)', text_query.lower())
            if chapter_match:
                enhanced_filter['chapter'] = int(chapter_match.group(1))
            
            # Look for book references
            book_patterns = {
                'bhagavad gita': 'bhagavad-gita',
                'bg': 'bhagavad-gita',
                'srimad bhagavatam': 'bhagavatam',
                'sb': 'bhagavatam'
            }
            
            for pattern, book in book_patterns.items():
                if pattern in text_query.lower():
                    enhanced_filter['title'] = book
            
            # Remove the text_query key as it's not used for actual filtering
            enhanced_filter.pop('text_query', None)
        
        return enhanced_filter
    
    def retrieve_documents(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        k: Optional[int] = None
    ) -> List[Document]:
        """Retrieve relevant documents for a query."""
        k = k or self.top_k
        enhanced_filter = self.parse_filter_dict(filter_dict)
        
        try:
            docs = self.vector_store.similarity_search(query=query, k=k, filter=enhanced_filter)
            logger.info(f"Retrieved {len(docs)} documents for query: {query}")
            logger.debug(f"Filter used: {enhanced_filter}")
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
        enhanced_filter = self.parse_filter_dict(filter_dict)
        
        try:
            docs_with_scores = self.vector_store.similarity_search_with_score(query=query, k=k, filter=enhanced_filter)
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
            chapter = doc.metadata.get("chapter", "")
            chapter_ref = doc.metadata.get("chapter_reference", "")
            
            # Format citation based on document type and available metadata
            if chapter:
                if chapter_ref:
                    citation = f"[Source {i+1}: {chapter_ref}, page {page}]"
                else:
                    citation = f"[Source {i+1}: {source}, Chapter {chapter}, page {page}]"
            elif doc_type == "website":
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
            chapter = doc.metadata.get("chapter", "")
            chapter_ref = doc.metadata.get("chapter_reference", "")
            
            # Create a unique source identifier
            if chapter_ref:
                source_key = f"{chapter_ref}_{page}"
            else:
                source_key = f"{source}_{chapter}_{page}"
            
            # Avoid duplicate sources
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            
            # Add a brief excerpt (first 100-150 chars) from the document
            content = doc.page_content.strip()
            if len(content) > 150:
                content = content[:150] + "..."
            
            # Format with source and chapter info
            if chapter_ref:
                summary_parts.append(f"From {chapter_ref} (page {page}): {content}")
            elif chapter:
                summary_parts.append(f"From {source} (Chapter {chapter}, page {page}): {content}")
            elif page:
                summary_parts.append(f"From {source} (page {page}): {content}")
            else:
                summary_parts.append(f"From {source}: {content}")
                
            # Limit to top 3 sources
            if len(summary_parts) >= 3:
                break
        
        return "\n\n".join(summary_parts)
    
    def answer_query(self, query: str, filter_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Answer a query using retrieved documents."""
        # Process any chapter-related filters
        enhanced_filter = self.parse_filter_dict(filter_dict)
        
        # Retrieve documents
        docs = self.retrieve_documents(query, enhanced_filter)
        
        if not docs:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "documents": []
            }
        
        # Prepare context
        context = self.get_context_for_query(query, enhanced_filter)
        
        # Select appropriate prompt template based on query
        prompt_template = select_prompt_template(query)
        
        # Generate answer
        answer = self.llm_interface.generate_response(
            query,
            context=context,
            system_prompt=prompt_template.template
        )
        
        # Check if there was an error with the LLM
        if answer.startswith("Error generating response:"):
            # Fall back to extracting relevant content directly from documents
            logger.warning(f"LLM error occurred, using fallback mechanism: {answer}")
            answer = self._create_fallback_answer(query, docs)
        
        # Extract source and chapter information
        sources = []
        for doc in docs:
            # Gather metadata for citation
            chapter = doc.metadata.get("chapter", None)
            chapter_ref = doc.metadata.get("chapter_reference", None)
            canto = doc.metadata.get("canto", None)
            verse = doc.metadata.get("verse", None)
            
            source_info = {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", ""),
                "type": doc.metadata.get("type", "document"),
                "title": doc.metadata.get("title", "")
            }
            
            # Add chapter information if available
            if chapter:
                source_info["chapter"] = chapter
            if chapter_ref:
                source_info["chapter_reference"] = chapter_ref
            if canto:
                source_info["canto"] = canto
            if verse:
                source_info["verse"] = verse
            
            sources.append(source_info)
        
        result = {
            "answer": answer,
            "sources": sources,
            "documents": docs
        }
        
        return result
    
    def _create_fallback_answer(self, query: str, docs: List[Document]) -> str:
        """Create a fallback answer when LLM fails."""
        # Extract search terms from query
        search_terms = query.lower().replace("?", "").replace(".", "").split()
        search_terms = [term for term in search_terms if len(term) > 3 and term not in ["what", "where", "when", "which", "who", "how", "why", "does", "about", "mean", "meaning"]]
        
        fallback_answer = f"I found information related to your query but encountered an issue with the AI model. "
        
        # If there are chapter references, mention them
        chapters_found = set()
        books_found = set()
        
        for doc in docs:
            if "chapter" in doc.metadata:
                chapter = doc.metadata.get("chapter")
                book = doc.metadata.get("title", "unknown").split("-")[0]
                chapter_ref = f"{book} Chapter {chapter}"
                chapters_found.add(chapter_ref)
                books_found.add(book)
        
        if chapters_found:
            fallback_answer += f"The information comes from {', '.join(chapters_found)}. "
        elif books_found:
            fallback_answer += f"The information comes from {', '.join(books_found)}. "
        
        fallback_answer += "Here's what I can tell you based on the retrieved documents:\n\n"
        
        # Check if there are definitions or direct answers in the docs
        definition_found = False
        
        # Look for sentences that contain the search terms
        for doc in docs:
            content = doc.page_content.lower()
            sentences = content.split('.')
            
            for sentence in sentences:
                # Check if sentence contains multiple search terms
                if all(term in sentence for term in search_terms) or any(term in sentence for term in search_terms):
                    cleaned = sentence.strip().capitalize()
                    if cleaned and len(cleaned) > 15:  # Ensure it's a meaningful sentence
                        fallback_answer += f"• {cleaned}.\n"
                        definition_found = True
                        break
            
            # Limit to 3 relevant sentences
            if definition_found and fallback_answer.count("•") >= 3:
                break
        
        # If no clear definition was found, extract relevant passages
        if not definition_found:
            fallback_answer += self._extract_relevant_content(docs, query)
        
        fallback_answer += "\n\nPlease try your query again later when the AI model is available."
        
        return fallback_answer
    
    def get_documents_by_chapter(self, chapter: int, book: Optional[str] = None, limit: int = 100) -> List[Document]:
        """Get documents from a specific chapter of a specific book."""
        filter_dict = {"chapter": chapter}
        if book:
            filter_dict["title"] = book
        
        try:
            docs = self.vector_store.filter_by_metadata(filter_dict, k=limit)
            logger.info(f"Retrieved {len(docs)} documents from chapter {chapter}")
            return docs
        except Exception as e:
            logger.error(f"Error retrieving documents by chapter: {str(e)}")
            return []
    
    def get_chapter_summary(self, chapter: int, book: Optional[str] = None) -> Dict[str, Any]:
        """Generate a summary for a specific chapter."""
        # Get documents from the chapter
        docs = self.get_documents_by_chapter(chapter, book)
        
        if not docs:
            return {
                "summary": f"No documents found for chapter {chapter}" + (f" of {book}" if book else ""),
                "sources": [],
                "documents": []
            }
        
        # Extract chapter reference if available
        chapter_ref = None
        for doc in docs:
            if "chapter_reference" in doc.metadata:
                chapter_ref = doc.metadata["chapter_reference"]
                break
        
        # Create context from first few documents
        context_docs = docs[:min(5, len(docs))]
        context = "\n\n".join(doc.page_content for doc in context_docs)
        
        # Generate summary
        system_prompt = f"""You are a scholarly expert on Vedic scriptures and Gaudiya Vaishnava texts.
        Provide a comprehensive summary of {chapter_ref or f'Chapter {chapter}'}.
        Include:
        1. The main themes and key teachings
        2. Important concepts introduced
        3. Significant verses and their meaning
        4. The chapter's place in the broader text
        
        Base your summary strictly on the context provided, without adding external information.
        """
        
        summary = self.llm_interface.generate_response(
            query=f"Summarize Chapter {chapter}" + (f" of {book}" if book else ""),
            context=context,
            system_prompt=system_prompt
        )
        
        # Check if there was an error with the LLM
        if summary.startswith("Error generating response:"):
            # Fall back to basic summary
            logger.warning(f"LLM error in chapter summary, using fallback mechanism")
            summary = f"Chapter {chapter}" + (f" of {book}" if book else "") + " contains information about: "
            
            # Extract main topics
            topics = set()
            for doc in docs[:10]:  # Use first 10 documents
                content = doc.page_content.lower()
                # Look for key terms in the content
                for para in content.split('\n\n'):
                    if len(para.split()) > 5:  # Ensure paragraph has enough words
                        first_sentence = para.split('.')[0]
                        if len(first_sentence) > 20:  # Ensure sentence is substantial
                            topics.add(first_sentence.strip().capitalize())
            
            # Add topics to summary
            for i, topic in enumerate(list(topics)[:5]):  # Limit to 5 topics
                summary += f"\n• {topic}"
        
        # Extract source information
        sources = []
        for doc in docs[:5]:  # Limit to first 5 documents
            source = {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", ""),
                "chapter": doc.metadata.get("chapter", ""),
                "chapter_reference": doc.metadata.get("chapter_reference", "")
            }
            sources.append(source)
        
        result = {
            "summary": summary,
            "sources": sources,
            "documents": docs[:5]  # Limit to first 5 documents
        }
        
        return result