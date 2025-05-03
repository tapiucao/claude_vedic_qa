# src/qa_system/retriever.py
# Applied changes: Item 12 (Update LLM interface import), Item 10 (Add comments/docstrings), Item 11 (Type Hinting)
"""
Retrieval system for Vedic Knowledge AI.
Handles finding relevant documents and context for queries using a vector store
and generating answers using an LLM interface.
"""
import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from langchain_core.documents import Document
import os

# Assuming components are importable from sibling/parent packages
from ..knowledge_base.vector_store import VedicVectorStore
from ..knowledge_base.prompt_templates import select_prompt_template, VEDIC_QA_PROMPT
# Updated import to use Gemini Interface
from .gemini_interface import GeminiLLMInterface 
# Assuming config is properly importable
# from ..config import TOP_K_RESULTS
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))


# Configure logging
logger = logging.getLogger(__name__)

class VedicRetriever:
    """
    Handles the retrieval augmented generation (RAG) process:
    1. Parses user queries and filters.
    2. Retrieves relevant documents from the VedicVectorStore.
    3. Prepares context for the LLM.
    4. Selects an appropriate prompt template.
    5. Calls the LLM (GeminiLLMInterface) to generate an answer.
    6. Formats the final response including sources.
    """
    
    def __init__(
        self,
        vector_store: VedicVectorStore,
        llm_interface: GeminiLLMInterface, # Updated type hint
        top_k: int = TOP_K_RESULTS
    ):
        """
        Initialize the retriever.

        Args:
            vector_store (VedicVectorStore): An initialized vector store instance.
            llm_interface (GeminiLLMInterface): An initialized LLM interface instance.
            top_k (int): Default number of documents to retrieve for context.
        """
        self.vector_store = vector_store
        self.llm_interface = llm_interface
        self.top_k = top_k
        
        logger.info(f"Initialized VedicRetriever with top_k={top_k}")
    
    def parse_filter_dict(self, filter_dict: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Parses and potentially enhances a filter dictionary for vector store queries.
        Attempts to extract structured information (like chapter number or book title)
        from less structured input within the filter dictionary (e.g., 'chapter_reference' string).

        Args:
            filter_dict (Optional[Dict[str, Any]]): The input filter dictionary. 
                                                    Can contain keys like 'chapter_reference', 'title', 'text_query'.

        Returns:
            Optional[Dict[str, Any]]: The parsed and potentially enhanced filter dictionary, 
                                      suitable for use in vector_store.similarity_search, or None if input was None.
                                      Returns None if the input is None or empty after parsing.
        """
        if not filter_dict:
            return None # Return None if input is None or empty
        
        # Copy to avoid modifying the original dictionary
        enhanced_filter = filter_dict.copy()
        
        # --- Enhance based on 'chapter_reference' string ---
        if isinstance(enhanced_filter.get('chapter_reference'), str):
            chapter_ref_str = enhanced_filter['chapter_reference']
            logger.debug(f"Parsing chapter_reference string: '{chapter_ref_str}'")
            
            # Try to extract chapter number
            # Match "Chapter <number>" or "Adhyāya <number>" etc. case-insensitively
            chapter_match = re.search(r'(?i)(?:Chapter|Adhyāya|Canto)\s+(\d+)', chapter_ref_str)
            if chapter_match:
                try:
                    chapter_num = int(chapter_match.group(1))
                    # Add the numeric chapter if not already present or different
                    if enhanced_filter.get('chapter') != chapter_num:
                         enhanced_filter['chapter'] = chapter_num
                         logger.debug(f"Extracted chapter number {chapter_num} from reference string.")
                except (ValueError, IndexError):
                     logger.warning(f"Could not parse chapter number from reference: {chapter_ref_str}")

            # Try to extract book title (simple cases)
            book_patterns = {
                'bhagavad gita': 'bhagavad-gita', # Assuming 'bhagavad-gita' is the title key used in metadata
                'bg': 'bhagavad-gita',
                'srimad bhagavatam': 'bhagavatam', # Assuming 'bhagavatam' is the title key
                'sb': 'bhagavatam'
                # Add more book patterns as needed
            }
            ref_lower = chapter_ref_str.lower()
            for pattern, book_key in book_patterns.items():
                 if pattern in ref_lower:
                      if enhanced_filter.get('title') != book_key:
                           enhanced_filter['title'] = book_key
                           logger.debug(f"Extracted book title '{book_key}' from reference string.")
                      break # Stop after first book match

        # --- Enhance based on 'text_query' (heuristic) ---
        # This is less reliable - prefer structured filters when possible.
        if isinstance(enhanced_filter.get('text_query'), str):
             text_query = enhanced_filter.pop('text_query').lower() # Remove after processing
             logger.debug(f"Parsing text_query for filter hints: '{text_query}'")

             # Extract chapter number like "tell me about chapter 2"
             chapter_match = re.search(r'chapter\s+(\d+)', text_query)
             if chapter_match:
                 try:
                    chapter_num = int(chapter_match.group(1))
                    if enhanced_filter.get('chapter') != chapter_num:
                         enhanced_filter['chapter'] = chapter_num
                         logger.debug(f"Extracted chapter number {chapter_num} from text_query.")
                 except (ValueError, IndexError):
                     logger.warning(f"Could not parse chapter number from text_query: {text_query}")

             # Extract book title hints
             book_patterns = {
                'bhagavad gita': 'bhagavad-gita', 
                'bg': 'bhagavad-gita',
                'srimad bhagavatam': 'bhagavatam', 
                'sb': 'bhagavatam'
                # Add more
            }
             for pattern, book_key in book_patterns.items():
                 if pattern in text_query:
                      if enhanced_filter.get('title') != book_key:
                           enhanced_filter['title'] = book_key
                           logger.debug(f"Extracted book title '{book_key}' from text_query.")
                      break

        # Return the enhanced filter, or None if it became empty
        return enhanced_filter if enhanced_filter else None
    
    def retrieve_documents(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        k: Optional[int] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents for a query using similarity search with optional filtering.

        Args:
            query (str): The user's query.
            filter_dict (Optional[Dict[str, Any]]): Metadata filters to apply.
            k (Optional[int]): Number of documents to retrieve (defaults to self.top_k).

        Returns:
            List[Document]: A list of retrieved documents, sorted by relevance.
        """
        num_results = k if k is not None and k > 0 else self.top_k
        # Parse the filter dictionary first
        effective_filter = self.parse_filter_dict(filter_dict)
        
        logger.info(f"Retrieving documents for query: '{query}' (k={num_results}, filter={effective_filter})")
        
        try:
            # Use the vector store's similarity search with the parsed filter
            docs = self.vector_store.similarity_search(
                query=query, 
                k=num_results, 
                filter=effective_filter # Pass the potentially enhanced filter
            )
            logger.info(f"Retrieved {len(docs)} documents.")
            return docs
        except Exception as e:
            # Catch specific ChromaDB or network errors if possible
            logger.error(f"Error retrieving documents from vector store: {str(e)}", exc_info=True)
            return [] # Return empty list on failure
    
    def retrieve_documents_with_scores(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant documents with similarity scores.

        Args:
            query (str): The user's query.
            filter_dict (Optional[Dict[str, Any]]): Metadata filters.
            k (Optional[int]): Number of documents to retrieve.

        Returns:
            List[Tuple[Document, float]]: List of (Document, score) tuples.
        """
        num_results = k if k is not None and k > 0 else self.top_k
        effective_filter = self.parse_filter_dict(filter_dict)
        
        logger.info(f"Retrieving documents with scores for query: '{query}' (k={num_results}, filter={effective_filter})")

        try:
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query=query, 
                k=num_results, 
                filter=effective_filter
            )
            logger.info(f"Retrieved {len(docs_with_scores)} scored documents.")
            return docs_with_scores
        except Exception as e:
            logger.error(f"Error retrieving documents with scores: {str(e)}", exc_info=True)
            return []
    
    def _format_context_with_citations(self, docs: List[Document]) -> str:
        """
        Formats the content of retrieved documents into a single string context block,
        including basic citations derived from metadata.

        Args:
            docs (List[Document]): The list of retrieved documents.

        Returns:
            str: A formatted string containing document contents and citations.
        """
        if not docs:
            return "No relevant context found." # Return informative message
        
        context_parts: List[str] = []
        for i, doc in enumerate(docs):
            # Extract metadata for citation
            metadata = doc.metadata
            source = metadata.get("source", "Unknown Source")
            page = metadata.get("page")
            doc_type = metadata.get("type", "document")
            chapter = metadata.get("chapter")
            # Use pre-formatted reference if available, otherwise try chapter/verse
            chapter_ref = metadata.get("chapter_reference", metadata.get("verse_reference")) 
            title = metadata.get("title", os.path.basename(source) if source != "Unknown Source" else "Unknown Title")

            # Construct citation string
            citation_parts = [f"Source {i+1}"]
            if chapter_ref:
                 citation_parts.append(chapter_ref)
            else:
                 # Fallback to title/chapter if no specific reference
                 if title and title != os.path.basename(source): # Avoid redundant filename if title is filename
                      citation_parts.append(title)
                 if chapter:
                      citation_parts.append(f"Chapter {chapter}")

            if page:
                citation_parts.append(f"Page {page}")
            
            # Add source path/URL if it provides useful unique info
            if source != "Unknown Source" and source != title and os.path.basename(source) != title:
                 if len(citation_parts) == 1: # Only add source if nothing else is available
                      citation_parts.append(f"({os.path.basename(source)})")


            citation = f"[{', '.join(citation_parts)}]"
            
            # Add document content with citation
            content = doc.page_content.strip() if doc.page_content else "[Empty Content]"
            context_parts.append(f"{content}\n{citation}")
        
        # Join parts with double newline for separation
        return "\n\n---\n\n".join(context_parts)

    def get_context_for_query(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        k: Optional[int] = None
    ) -> str:
        """
        Retrieves relevant documents and formats their content into a single context string for the LLM.

        Args:
            query (str): The user's query.
            filter_dict (Optional[Dict[str, Any]]): Metadata filters.
            k (Optional[int]): Number of documents to retrieve.

        Returns:
            str: A formatted context string with citations.
        """
        docs = self.retrieve_documents(query, filter_dict, k)
        return self._format_context_with_citations(docs)

    def _extract_relevant_content_fallback(self, docs: List[Document], query: str) -> str:
        """
        Fallback method: Extracts short snippets from documents when LLM fails.
        Tries to provide *some* relevant information directly from sources.

        Args:
            docs (List[Document]): Retrieved documents.
            query (str): Original user query (used for context).

        Returns:
            str: A string summarizing key snippets from the documents.
        """
        if not docs:
            return "No relevant information found in the retrieved documents."
        
        summary_parts: List[str] = []
        seen_sources: set[str] = set() # Track unique sources (e.g., "Book Title, Chapter X, Page Y")
        max_excerpts = 3 # Limit number of excerpts shown

        for doc in docs:
            if len(summary_parts) >= max_excerpts:
                break

            metadata = doc.metadata
            source = metadata.get("source", "Unknown")
            page = metadata.get("page", "")
            chapter = metadata.get("chapter", "")
            chapter_ref = metadata.get("chapter_reference", metadata.get("verse_reference"))
            title = metadata.get("title", os.path.basename(source) if source != "Unknown" else "")

            # Create a unique key for the source to avoid duplicates
            if chapter_ref:
                 source_key = f"{chapter_ref}_p{page}"
            else:
                 source_key = f"{title}_c{chapter}_p{page}"
            
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            
            # Format the source identifier for the excerpt
            source_identifier_parts = []
            if chapter_ref:
                 source_identifier_parts.append(chapter_ref)
            elif title:
                 source_identifier_parts.append(title)
                 if chapter: source_identifier_parts.append(f"Ch. {chapter}")
            if page: source_identifier_parts.append(f"p. {page}")
            source_identifier = ", ".join(source_identifier_parts) if source_identifier_parts else os.path.basename(source)

            # Extract a relevant excerpt (e.g., first ~200 chars)
            content = doc.page_content.strip() if doc.page_content else ""
            excerpt = content[:200] + "..." if len(content) > 200 else content
            
            if excerpt: # Only add if there's content
                summary_parts.append(f"From {source_identifier}:\n\"\"\"\n{excerpt}\n\"\"\"")
                
        return "\n\n".join(summary_parts)

    def answer_query(self, query: str, filter_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Answers a user query using the RAG process: retrieve, format context, generate answer.

        Args:
            query (str): The user's query.
            filter_dict (Optional[Dict[str, Any]]): Metadata filters.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'answer' (str): The generated answer.
                - 'sources' (List[Dict]): List of source document metadata.
                - 'documents' (List[Document]): The raw retrieved documents.
        """
        logger.info(f"Answering query: '{query}' with filter: {filter_dict}")
        
        # 1. Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(query, filter_dict)
        
        if not retrieved_docs:
            logger.warning("No relevant documents found for the query.")
            return {
                "answer": "I couldn't find specific information related to your query in the available documents.",
                "sources": [],
                "documents": []
            }
        
        # 2. Prepare context for the LLM
        context = self._format_context_with_citations(retrieved_docs)
        
        # 3. Select appropriate prompt template based on query type (heuristic)
        # Default to the general VEDIC_QA_PROMPT if specific type isn't detected
        prompt_template = select_prompt_template(query) or VEDIC_QA_PROMPT
        # Ensure the template expects 'context' and 'question' input variables
        # This assumes the template has structure like: "System instructions... Context: {context} Question: {question}"
        
        # 4. Generate answer using the LLM
        logger.debug("Generating answer using LLM...")
        try:
            # Use the LLM interface's generate_response method
            # Pass the context and query according to how the LLM interface expects them
            # (gemini_interface.py seems to combine them in the 'prompt' arg and uses 'context' separately)
            # Let's adapt to gemini_interface's current structure:
            
            # Construct the system prompt part from the template (everything before context/question)
            # This is a bit hacky; ideally, the prompt template itself would be structured better
            # or the LLM interface would handle PromptTemplate objects directly.
            system_prompt_base = prompt_template.template.split("Context:")[0].strip()
            
            llm_answer = self.llm_interface.generate_response(
                prompt=query, # Pass the raw query here
                context=context, # Pass the formatted context here
                system_prompt=system_prompt_base # Pass the instructional part
            )
            
            # Check for LLM errors indicated in the response string itself
            if llm_answer.startswith("Error generating response:"):
                 raise RuntimeError(f"LLM generation failed: {llm_answer}")

        except Exception as llm_error:
            logger.error(f"LLM generation failed: {llm_error}", exc_info=True)
            # Fallback mechanism
            logger.warning("LLM failed, attempting fallback answer generation.")
            answer = self._create_fallback_answer(query, retrieved_docs)
            llm_failed = True
        else:
             answer = llm_answer
             llm_failed = False

        # 5. Extract source metadata for the response
        sources_metadata: List[Dict[str, Any]] = []
        for doc in retrieved_docs:
             # Include key metadata used for citation or understanding source
             source_info = {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page"),
                "type": doc.metadata.get("type"),
                "title": doc.metadata.get("title"),
                "chapter": doc.metadata.get("chapter"),
                "chapter_reference": doc.metadata.get("chapter_reference"),
                "verse_reference": doc.metadata.get("verse_reference"),
                "contains_sanskrit": doc.metadata.get("contains_sanskrit"),
                # Add score if available (requires retrieve_documents_with_scores)
             }
             # Remove None values for cleaner output
             sources_metadata.append({k: v for k, v in source_info.items() if v is not None})
        
        result = {
            "answer": answer,
            "sources": sources_metadata,
            "documents": retrieved_docs, # Include raw documents for potential UI display or further processing
            "llm_fallback_used": llm_failed # Indicate if fallback was used
        }
        
        return result
    
    def _create_fallback_answer(self, query: str, docs: List[Document]) -> str:
        """Creates a simple fallback answer summarizing retrieved documents when the LLM fails."""
        logger.debug("Creating fallback answer.")
        
        fallback_intro = f"I encountered an issue generating a detailed answer with the AI model. However, I found the following information related to '{query}':\n\n"
        
        # Use the helper function to extract relevant content snippets
        relevant_content = self._extract_relevant_content_fallback(docs, query)
        
        fallback_outro = "\n\nPlease try your query again later or rephrase it."
        
        return fallback_intro + relevant_content + fallback_outro
    
    def get_documents_by_chapter(self, chapter: int, book_title: Optional[str] = None, limit: int = 100) -> List[Document]:
        """
        Retrieve all document chunks associated with a specific chapter (and optionally book).
        Uses metadata filtering, not semantic search.

        Args:
            chapter (int): The chapter number to retrieve.
            book_title (Optional[str]): The title of the book (must match 'title' metadata).
            limit (int): Maximum number of document chunks to return.

        Returns:
            List[Document]: List of documents matching the chapter/book criteria.
        """
        filter_dict: Dict[str, Any] = {"chapter": chapter}
        if book_title:
            # Ensure the book title matches the key used in metadata (e.g., 'bhagavad-gita')
            filter_dict["title"] = book_title 
        
        logger.info(f"Retrieving documents by metadata filter: {filter_dict} (limit={limit})")
        
        try:
            # Use the specific metadata filtering method
            results_dict = self.vector_store.get_documents_by_metadata_filter(
                filter_dict, 
                limit=limit, 
                include=["documents"] # We only need the documents here
            )
            docs = results_dict.get('documents', [])
            logger.info(f"Retrieved {len(docs)} documents for filter.")
            return docs
        except Exception as e:
            logger.error(f"Error retrieving documents by chapter/book filter: {str(e)}", exc_info=True)
            return []
    
    def get_chapter_summary(self, chapter: int, book_title: Optional[str] = None) -> Dict[str, Any]:
        """
        Generates a summary for a specific chapter by retrieving its documents and using the LLM.

        Args:
            chapter (int): The chapter number.
            book_title (Optional[str]): The title of the book (matching metadata).

        Returns:
            Dict[str, Any]: Dictionary containing 'summary', 'sources', and 'documents'.
        """
        logger.info(f"Generating summary for chapter {chapter}" + (f" of book '{book_title}'" if book_title else ""))
        
        # 1. Get documents for the chapter
        # Retrieve more documents than needed for context to ensure coverage
        chapter_docs = self.get_documents_by_chapter(chapter, book_title, limit=200) 
        
        if not chapter_docs:
            logger.warning(f"No documents found for chapter {chapter}" + (f" of {book_title}" if book_title else ""))
            return {
                "summary": f"Could not generate summary. No documents found for chapter {chapter}" + (f" of {book_title}." if book_title else "."),
                "sources": [],
                "documents": []
            }
            
        # 2. Prepare context for summary generation
        # Use a subset of documents for context to avoid exceeding LLM limits
        # Sort by page number if available? Or just take the first N?
        context_docs = sorted(chapter_docs, key=lambda d: d.metadata.get('page', 0))[:10] # Use first 10 pages/docs for context
        context = self._format_context_with_citations(context_docs) # Reuse formatting

        # Extract a representative chapter reference for the prompt
        chapter_ref_display = f"Chapter {chapter}"
        if book_title:
             chapter_ref_display = f"{book_title}, {chapter_ref_display}"
        # Try to get a more specific reference from metadata if available
        for doc in context_docs:
            if doc.metadata.get("chapter_reference"):
                chapter_ref_display = doc.metadata["chapter_reference"]
                break 

        # 3. Define the summarization prompt
        summarization_query = f"Provide a comprehensive summary of {chapter_ref_display}."
        system_prompt = f"""You are a scholarly expert specializing in summarizing Vedic texts like the Bhagavad Gita and Srimad Bhagavatam. 
Based *only* on the provided context documents for {chapter_ref_display}, generate a concise yet comprehensive summary. 
Your summary should include:
1. The main themes and narrative (if any).
2. Key teachings, concepts, or verses highlighted in the context.
3. The significance of this chapter/section based on the provided text.
Do not add external information or interpretations not present in the context.
"""

        # 4. Generate summary using the LLM
        logger.debug("Generating chapter summary using LLM...")
        try:
             summary = self.llm_interface.generate_response(
                  prompt=summarization_query, # The task description
                  context=context,           # The chapter content
                  system_prompt=system_prompt  # The role and instructions
             )
             if summary.startswith("Error generating response:"):
                 raise RuntimeError(f"LLM summarization failed: {summary}")
             llm_failed = False
        except Exception as e:
             logger.error(f"LLM chapter summarization failed: {e}", exc_info=True)
             summary = f"Could not automatically generate a summary for {chapter_ref_display} due to an AI model error. Please refer to the source documents."
             llm_failed = True
        
        # 5. Prepare result
        # Include metadata from the documents used for context generation
        sources_metadata = [
            {k: v for k, v in doc.metadata.items() if v is not None} 
            for doc in context_docs 
        ]
        
        result = {
            "summary": summary,
            "sources": sources_metadata, # Sources used for the summary context
            "documents": context_docs,    # Documents used for the summary context
            "llm_fallback_used": llm_failed
        }
        
        return result