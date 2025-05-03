# src/qa_system/retriever.py
# Applied changes: Item 12 (Update LLM interface import), Item 10 (Add comments/docstrings), Item 11 (Type Hinting)
"""
Retrieval system for Vedic Knowledge AI.
Handles finding relevant documents and context for queries using a vector store
and generating answers using an LLM interface.
"""
import logging
import re
import os
from typing import List, Dict, Any, Optional, Tuple, Union, Sequence # Added Sequence
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate # Import PromptTemplate


# Assuming components are importable from sibling/parent packages
from ..knowledge_base.vector_store import VedicVectorStore
from ..knowledge_base.prompt_templates import select_prompt_template, VEDIC_QA_PROMPT
# Updated import to use Gemini Interface
from .gemini_interface import GeminiLLMInterface
# Assuming config is properly importable
from ..config import TOP_K_RESULTS

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
            # These keys should match the 'title' metadata used during ingestion
            book_patterns = {
                'bhagavad gita': 'bhagavad-gita',
                'bg': 'bhagavad-gita',
                'srimad bhagavatam': 'srimad-bhagavatam', # Example adjusted key
                'sb': 'srimad-bhagavatam',
                'cc': 'chaitanya-charitamrita', # Example
                # Add more book patterns and their corresponding metadata keys as needed
            }
            ref_lower = chapter_ref_str.lower()
            for pattern, book_key in book_patterns.items():
                 # Use word boundaries for more specific matching
                 if re.search(r'\b' + re.escape(pattern) + r'\b', ref_lower):
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

             # Extract book title hints (reuse patterns from above)
             # These keys should match the 'title' metadata used during ingestion
             book_patterns = {
                'bhagavad gita': 'bhagavad-gita',
                'bg': 'bhagavad-gita',
                'srimad bhagavatam': 'srimad-bhagavatam',
                'sb': 'srimad-bhagavatam',
                'cc': 'chaitanya-charitamrita',
                # Add more
            }
             for pattern, book_key in book_patterns.items():
                 if re.search(r'\b' + re.escape(pattern) + r'\b', text_query):
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

    def _format_context_with_citations(self, docs: Sequence[Document]) -> str:
        """
        Formats the content of retrieved documents into a single string context block,
        including citations derived from metadata.

        Args:
            docs (Sequence[Document]): The list or sequence of retrieved documents.

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
            # Use pre-formatted reference if available (e.g., BG 2.13, SB 1.2.3)
            # Combine chapter_reference and verse_reference if both exist or prefer one
            verse_ref = metadata.get("verse_reference") # Specific verse ID like '2.13' or '1.2.3'
            chapter_ref_meta = metadata.get("chapter_reference") # General ref like 'Book, Chapter X'
            chapter_num = metadata.get("chapter") # Just the number

            # Try to build the best citation string
            citation_parts = [f"Source [{i+1}]"] # Start with index
            if verse_ref:
                 citation_parts.append(verse_ref) # Prefer specific verse ref
            elif chapter_ref_meta:
                 citation_parts.append(chapter_ref_meta) # Use chapter reference string
            else:
                 # Fallback to title/chapter number if available
                 title = metadata.get("title", os.path.basename(source) if source != "Unknown Source" else None)
                 if title: citation_parts.append(title)
                 if chapter_num: citation_parts.append(f"Ch. {chapter_num}")

            # Add page number if relevant and not redundant
            if page and (not verse_ref or '.' not in str(verse_ref)): # Add page if no verse ref or ref is simple number
                citation_parts.append(f"Page {page}")

            # Construct citation string
            citation = f"({' | '.join(citation_parts)})" # Use parentheses and pipe separators

            # Add document content with citation appended
            content = doc.page_content.strip() if doc.page_content else "[No Content]"
            # Add citation at the end of the content for this chunk
            context_parts.append(f"{content} {citation}")

        # Join parts with double newline for separation between source chunks
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


    def _extract_relevant_content_fallback(self, docs: Sequence[Document], query: str) -> str:
        """
        Fallback method: Extracts short snippets from documents when LLM fails.
        Tries to provide *some* relevant information directly from sources.

        Args:
            docs (Sequence[Document]): Retrieved documents.
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
            verse_ref = metadata.get("verse_reference") # Specific verse ID like '2.13' or '1.2.3'
            chapter_ref_meta = metadata.get("chapter_reference") # General ref like 'Book, Chapter X'
            chapter_num = metadata.get("chapter") # Just the number
            title = metadata.get("title", os.path.basename(source) if source != "Unknown" else "")

            # Create a unique key for the source chunk to avoid duplicates
            source_key_parts = [str(metadata.get(k)) for k in ['title', 'chapter', 'page', 'verse_reference'] if metadata.get(k)]
            source_key = "_".join(source_key_parts) if source_key_parts else str(doc.metadata) # Fallback key

            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)

            # Format the source identifier for the excerpt
            source_identifier_parts = []
            if verse_ref: source_identifier_parts.append(verse_ref)
            elif chapter_ref_meta: source_identifier_parts.append(chapter_ref_meta)
            else:
                 if title: source_identifier_parts.append(title)
                 if chapter_num: source_identifier_parts.append(f"Ch. {chapter_num}")
            if page: source_identifier_parts.append(f"Page {page}")
            source_identifier = ", ".join(filter(None, source_identifier_parts)) or os.path.basename(source)

            # Extract a relevant excerpt (e.g., first ~250 chars)
            content = doc.page_content.strip() if doc.page_content else ""
            excerpt = content[:250] + "..." if len(content) > 250 else content

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
                - 'llm_fallback_used' (bool): Indicates if the LLM failed and fallback was used.
        """
        logger.info(f"Answering query: '{query}' with filter: {filter_dict}")

        # 1. Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(query, filter_dict)

        if not retrieved_docs:
            logger.warning("No relevant documents found for the query.")
            return {
                "answer": "I couldn't find specific information related to your query in the available documents.",
                "sources": [],
                "documents": [],
                "llm_fallback_used": False # No LLM call attempted
            }

        # 2. Prepare context for the LLM
        context = self._format_context_with_citations(retrieved_docs)

        # 3. Select appropriate prompt template based on query type (heuristic)
        prompt_template: PromptTemplate = select_prompt_template(query) or VEDIC_QA_PROMPT

        # Extract system prompt from the selected template
        # This assumes the template string has a structure where system instructions
        # precede the context/question placeholders.
        # TODO: Future improvement - integrate PromptTemplate more directly with LLM interface if possible.
        try:
            # Attempt to format with dummy values to isolate the system part
            # This is less reliable, depends heavily on template structure
            # A simpler approach is often just to extract text before known markers
            template_str = prompt_template.template
            system_prompt_part = template_str.split("Context:", 1)[0].strip()
            if not system_prompt_part or "{context}" in system_prompt_part or "{question}" in system_prompt_part:
                 # Fallback if splitting fails or seems incorrect
                 system_prompt_part = "You are a knowledgeable scholar of Vedic philosophy. Use the provided context to answer the question."
                 logger.warning("Could not reliably extract system prompt from template, using default.")
        except Exception:
            logger.exception("Error extracting system prompt from template, using default.")
            system_prompt_part = "You are a knowledgeable scholar of Vedic philosophy. Use the provided context to answer the question."


        # 4. Generate answer using the LLM
        logger.debug("Generating answer using LLM...")
        llm_answer = None
        llm_failed = False
        try:
            # Use the LLM interface's generate_response method
            llm_answer = self.llm_interface.generate_response(
                prompt=query, # Pass the raw query here
                context=context, # Pass the formatted context here
                system_prompt=system_prompt_part # Pass the instructional part
            )

            # Check for error messages returned *in* the response string
            if llm_answer.startswith("Error:"):
                 logger.error(f"LLM generation returned an error message: {llm_answer}")
                 raise RuntimeError(f"LLM generation failed: {llm_answer}") # Treat as failure

        except Exception as llm_error:
            # Log the detailed error from the LLM interface or the RuntimeError
            logger.error(f"LLM generation failed: {llm_error}", exc_info=True)
            # Fallback mechanism
            logger.warning("LLM failed, attempting fallback answer generation.")
            answer = self._create_fallback_answer(query, retrieved_docs)
            llm_failed = True
        else:
             answer = llm_answer # Use the successful LLM response
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
                # "score": doc.metadata.get("score") # Example if score was added
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

    def _create_fallback_answer(self, query: str, docs: Sequence[Document]) -> str:
        """Creates a simple fallback answer summarizing retrieved documents when the LLM fails."""
        logger.debug("Creating fallback answer.")

        fallback_intro = f"I encountered an issue generating a detailed answer with the AI model. However, based on the retrieved documents related to '{query}', here is some potentially relevant information:\n\n"

        # Use the helper function to extract relevant content snippets
        relevant_content = self._extract_relevant_content_fallback(docs, query)

        fallback_outro = "\n\nPlease verify this information with the original sources. You might try rephrasing your query or asking again later."

        # Handle case where fallback extraction also yielded nothing
        if not relevant_content or relevant_content == "No relevant information found in the retrieved documents.":
            return "I encountered an issue generating an answer, and I couldn't extract relevant snippets from the retrieved documents either. Please try your query again later."

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
            filter_dict["title"] = book_title.lower().replace(" ", "-") # Example normalization
            logger.info(f"Filtering by book title: {filter_dict['title']}")

        logger.info(f"Retrieving documents by metadata filter: {filter_dict} (limit={limit})")

        try:
            # Use the specific metadata filtering method
            # Specify we want documents and metadatas included
            results_dict = self.vector_store.get_documents_by_metadata_filter(
                filter_dict,
                limit=limit,
                include=["documents", "metadatas"] # Request both for constructing Document objects
            )

            # Reconstruct Document objects if 'get' returns dicts
            docs = []
            retrieved_docs = results_dict.get('documents', [])
            retrieved_metadatas = results_dict.get('metadatas', [])
            if retrieved_docs and retrieved_metadatas and len(retrieved_docs) == len(retrieved_metadatas):
                for content, meta in zip(retrieved_docs, retrieved_metadatas):
                    docs.append(Document(page_content=content, metadata=meta))
            elif retrieved_docs: # If only documents returned somehow
                 docs = [Document(page_content=content) for content in retrieved_docs]


            logger.info(f"Retrieved {len(docs)} documents for filter.")
            # Sort documents by page number if available
            if docs and all('page' in d.metadata for d in docs):
                 docs.sort(key=lambda d: d.metadata.get('page', 0))

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
        book_title_log = f" of book '{book_title}'" if book_title else ""
        logger.info(f"Generating summary for chapter {chapter}{book_title_log}")

        # 1. Get documents for the chapter
        # Retrieve more documents than needed for context initially to ensure coverage
        chapter_docs = self.get_documents_by_chapter(chapter, book_title, limit=200)

        if not chapter_docs:
            logger.warning(f"No documents found for chapter {chapter}{book_title_log}")
            return {
                "summary": f"Could not generate summary. No documents found for Chapter {chapter}{book_title_log}.",
                "sources": [],
                "documents": [],
                "llm_fallback_used": False
            }

        # 2. Prepare context for summary generation
        # Select a subset for context (e.g., first 10-15 chunks after sorting)
        # Sorting ensures somewhat logical order if pages are numbered correctly
        context_docs = chapter_docs[:15] # Use first 15 chunks for context
        context = self._format_context_with_citations(context_docs) # Reuse formatting

        # Extract a representative chapter reference for the prompt
        # Try to get the best identifier from the context docs
        chapter_ref_display = f"Chapter {chapter}"
        found_title = book_title
        for doc in context_docs:
            meta = doc.metadata
            if meta.get("chapter_reference"): # e.g., "Book Title, Chapter X"
                chapter_ref_display = meta["chapter_reference"]
                found_title = meta.get("title", found_title)
                break
            elif meta.get("title") and not found_title:
                 found_title = meta.get("title")

        if found_title and f"Chapter {chapter}" in chapter_ref_display:
             chapter_ref_display = f"{found_title}, Chapter {chapter}"


        # 3. Define the summarization prompt
        summarization_query = f"Provide a comprehensive summary of {chapter_ref_display}."
        system_prompt = f"""You are a scholarly expert specializing in summarizing Vedic texts like the Bhagavad Gita and Srimad Bhagavatam.
Based *only* on the provided context documents for {chapter_ref_display}, generate a concise yet comprehensive summary.
Your summary should include:
1. The main themes and narrative presented in the context.
2. Key teachings, concepts, or verses highlighted in the context.
3. The overall significance or flow of this chapter section based *only* on the provided text.
Focus strictly on the information available in the context. Do not add external knowledge or interpretations.
"""

        # 4. Generate summary using the LLM
        logger.debug("Generating chapter summary using LLM...")
        summary = None
        llm_failed = False
        try:
             summary = self.llm_interface.generate_response(
                  prompt=summarization_query, # The task description
                  context=context,           # The chapter content chunks
                  system_prompt=system_prompt  # The role and instructions
             )
             if summary.startswith("Error:"):
                 logger.error(f"LLM summarization returned an error: {summary}")
                 raise RuntimeError(f"LLM summarization failed: {summary}")
             llm_failed = False
        except Exception as e:
             logger.error(f"LLM chapter summarization failed: {e}", exc_info=True)
             summary = f"Could not automatically generate a summary for {chapter_ref_display} due to an AI model error. Please refer to the source documents provided."
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
            "documents": context_docs,    # Document chunks used for the summary context
            "llm_fallback_used": llm_failed
        }

        return result