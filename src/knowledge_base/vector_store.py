# src/knowledge_base/vector_store.py
# Applied changes: Item 2 (Remove redundant methods), Item 9 (Clarify filtering), Item 11 (Type Hinting)
"""
Vector database operations for Vedic Knowledge AI.
Handles storing, retrieving, and managing vector embeddings using ChromaDB.
"""
import os
import logging
from typing import List, Dict, Any, Optional, Union, Tuple, Sequence
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings # Use base Embeddings type
from langchain_huggingface import HuggingFaceEmbeddings # Keep for type hint clarity if used directly

# Assuming config is properly importable
# from ..config import DB_DIR, TOP_K_RESULTS
DB_DIR = os.getenv("DB_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "db_new"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))

# Configure logging
logger = logging.getLogger(__name__)

class VedicVectorStore:
    """
    Manages interactions with a Chroma vector store for Vedic textual knowledge,
    including adding documents, performing similarity searches, and filtering.
    """
    
    def __init__(
        self,
        embedding_function: Embeddings, # Use the base Embeddings type hint
        persist_directory: str = DB_DIR,
        collection_name: str = "vedic_knowledge",
        chroma_host: Optional[str] = None,
        chroma_port: Optional[int] = None
    ):
        """
        Initialize the vector store.

        Args:
            embedding_function (Embeddings): An initialized LangChain Embeddings object.
            persist_directory (str): Path to the directory for local persistence.
            collection_name (str): Name of the Chroma collection.
            chroma_host (Optional[str]): Hostname for a remote Chroma server.
            chroma_port (Optional[int]): Port for a remote Chroma server.
        """
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Check for external Chroma settings from environment if not provided directly
        self.chroma_host = chroma_host or os.environ.get("CHROMA_HOST")
        chroma_port_str = os.environ.get("CHROMA_PORT")
        self.chroma_port = chroma_port or (int(chroma_port_str) if chroma_port_str and chroma_port_str.isdigit() else None)
        
        # Ensure local directory exists if used
        if not (self.chroma_host and self.chroma_port):
             os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize the vector store connection
        self.vector_store: Optional[Chroma] = None # Initialize as None
        try:
             self._initialize_vector_store()
             logger.info(f"Vector store initialized/connected for collection '{self.collection_name}'.")
        except Exception as e:
             logger.error(f"Failed to initialize vector store: {e}", exc_info=True)
             # Decide whether to raise the exception or handle it (e.g., allow retries later)
             raise # Re-raise the exception to signal failure
             
    def _initialize_vector_store(self):
        """Initializes the Chroma vector store client/connection."""
        # Check if already initialized
        if self.vector_store:
             return
             
        logger.info("Attempting to initialize Chroma vector store...")
        # Check if using external Chroma server
        if self.chroma_host and self.chroma_port:
            logger.info(f"Connecting to external Chroma server at {self.chroma_host}:{self.chroma_port}")
            try:
                # Use chromadb client for remote connection
                import chromadb
                from chromadb.config import Settings # Deprecated? Check ChromaDB docs
                
                # Recommended way to create HttpClient
                client = chromadb.HttpClient(
                    host=self.chroma_host,
                    port=self.chroma_port,
                    # settings=Settings(anonymized_telemetry=False) # Settings might be deprecated/changed
                )
                # Ping server to check connection early
                client.heartbeat() 
                logger.info("Successfully connected to Chroma server.")

                # Initialize LangChain Chroma wrapper with the client
                self.vector_store = Chroma(
                    client=client,
                    embedding_function=self.embedding_function,
                    collection_name=self.collection_name
                )
                
            except ImportError:
                 logger.error("chromadb library not found. Install with 'pip install chromadb' for remote connection.")
                 raise
            except Exception as e:
                 logger.error(f"Failed to connect to external Chroma server: {e}", exc_info=True)
                 raise # Re-raise connection error
        else:
            # Use local persistence
            logger.info(f"Using local Chroma persistence at {self.persist_directory}")
            try:
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding_function,
                    collection_name=self.collection_name
                )
            except Exception as e:
                 logger.error(f"Failed to initialize local Chroma database: {e}", exc_info=True)
                 raise # Re-raise initialization error
            
        # Log document count after successful initialization
        try:
             # Accessing _collection might be internal, use get() if possible or count() if available
             # doc_count = len(self.vector_store.get()['ids']) # Getting all IDs might be slow
             # Chroma has a count method now (check documentation)
             # For older versions or if count() isn't available:
             if hasattr(self.vector_store, '_collection'):
                 doc_count = self.vector_store._collection.count()
                 logger.info(f"Collection '{self.collection_name}' contains {doc_count} documents.")
             else:
                  logger.warning("Could not retrieve document count easily.")
        except Exception as e:
            logger.warning(f"Could not get document count: {e}")
            
    def add_documents(self, documents: Sequence[Document]) -> None:
        """
        Add documents to the vector store.

        Args:
            documents (Sequence[Document]): A list or sequence of LangChain Documents to add.
        """
        if not self.vector_store:
             logger.error("Vector store not initialized. Cannot add documents.")
             return
        if not documents:
            logger.warning("No documents provided to add to vector store.")
            return
        
        try:
            # Add documents using the LangChain Chroma wrapper
            # This handles embedding generation implicitly
            ids = self.vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to collection '{self.collection_name}'.")
            # Chroma returns the IDs of the added documents
            logger.debug(f"Added document IDs (first 10): {ids[:10]}") 
            # Persist changes if using local Chroma (usually handled automatically by Chroma client)
            # self.vector_store.persist() # Check if manual persist is needed
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}", exc_info=True)
            # Consider raising the exception depending on desired behavior
            # raise
    
    def similarity_search(
        self, 
        query: str, 
        k: int = TOP_K_RESULTS,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform a similarity search for documents relevant to the query, optionally filtering by metadata.

        Args:
            query (str): The query string to search for.
            k (int): The number of top similar documents to return.
            filter (Optional[Dict[str, Any]]): A dictionary specifying metadata filters. 
                                               Example: {'chapter': 2, 'title': 'bhagavad-gita'}

        Returns:
            List[Document]: A list of relevant documents sorted by similarity.
        """
        if not self.vector_store:
             logger.error("Vector store not initialized. Cannot perform search.")
             return []
             
        try:
            # Ensure filter is None if empty dict is passed, as Chroma might treat {} differently
            effective_filter = filter if filter else None
                
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=effective_filter # Pass the filter dictionary here
            )
            logger.info(f"Found {len(results)} results for query: '{query}' with k={k} and filter={effective_filter}")
            return results
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}", exc_info=True)
            return []

    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = TOP_K_RESULTS,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform a similarity search, returning documents along with their similarity scores.

        Args:
            query (str): The query string.
            k (int): The number of results to return.
            filter (Optional[Dict[str, Any]]): Metadata filter dictionary.

        Returns:
            List[Tuple[Document, float]]: List of (Document, score) tuples. Score typically represents distance (lower is better).
        """
        if not self.vector_store:
             logger.error("Vector store not initialized. Cannot perform search.")
             return []
             
        try:
            effective_filter = filter if filter else None
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=effective_filter
            )
            logger.info(f"Found {len(results)} scored results for query: '{query}' with k={k} and filter={effective_filter}")
            return results
        except Exception as e:
            logger.error(f"Error searching vector store with scores: {str(e)}", exc_info=True)
            return []
            
    # Removed redundant 'search' and 'retrieve_documents' methods. 
    # Use 'similarity_search' or 'similarity_search_with_score'.

    def get_documents_by_metadata_filter(
        self, 
        filter_dict: Dict[str, Any],
        limit: Optional[int] = None, # Use limit instead of k for clarity
        include: Optional[List[str]] = None # Optionally include embeddings, metadatas, documents
    ) -> Dict[str, Any]: 
        """
        Retrieve documents based *only* on metadata filtering (no semantic search).

        Args:
            filter_dict (Dict[str, Any]): The metadata filter dictionary (Chroma 'where' clause).
            limit (Optional[int]): Maximum number of documents to return.
            include (Optional[List[str]]): List of fields to include in the results 
                                           (e.g., ["metadatas", "documents"]). Defaults usually 
                                           include ids, metadatas, documents.

        Returns:
            Dict[str, Any]: A dictionary containing the retrieved data, typically including 
                           'ids', 'metadatas', 'documents' depending on the 'include' parameter.
                           Returns an empty dict on error or if not initialized.
        """
        if not self.vector_store:
             logger.error("Vector store not initialized. Cannot get documents by metadata.")
             return {}
        if not filter_dict:
             logger.warning("Empty filter dictionary provided to get_documents_by_metadata_filter.")
             # Depending on Chroma version, an empty filter might return all docs or raise error.
             # It's safer to return early or require a filter.
             return {}
             
        try:
            # Default include to get documents and metadata
            include_fields = include if include else ["metadatas", "documents"]
            
            # Use the underlying Chroma client's 'get' method via the wrapper
            results = self.vector_store.get(
                where=filter_dict,
                limit=limit,
                include=include_fields
            )
            count = len(results.get('ids', [])) # Get count from returned IDs
            logger.info(f"Found {count} documents matching metadata filter: {filter_dict} (limit={limit})")
            return results # Return the raw dictionary from Chroma client
        except Exception as e:
            logger.error(f"Error filtering by metadata: {str(e)}", exc_info=True)
            return {}

    def filter_by_source_type(
        self, 
        source_type: str,
        limit: int = 100
    ) -> List[Document]:
        """
        Convenience method to get documents matching a specific 'type' metadata field.

        Args:
            source_type (str): The type to filter by (e.g., 'pdf', 'website').
            limit (int): Maximum number of documents to return.

        Returns:
            List[Document]: A list of Document objects matching the type.
        """
        filter_dict = {"type": source_type}
        results_dict = self.get_documents_by_metadata_filter(filter_dict, limit=limit, include=["documents"])
        # Extract the Document objects from the results dictionary
        return results_dict.get('documents', [])


    def get_retriever(self, search_type: str = "similarity", search_kwargs: Optional[Dict[str, Any]] = None):
        """
        Get a LangChain Retriever interface to the vector store.

        Args:
            search_type (str): The type of search ('similarity', 'mmr', etc.).
            search_kwargs (Optional[Dict[str, Any]]): Arguments for the search (e.g., {'k': k, 'filter': filter}).

        Returns:
            A LangChain retriever object.
        """
        if not self.vector_store:
             logger.error("Vector store not initialized. Cannot get retriever.")
             # Optionally raise an error or return None
             raise ValueError("Vector store must be initialized before getting a retriever.")
             
        if search_kwargs is None:
            search_kwargs = {"k": TOP_K_RESULTS}
            
        # Ensure 'k' is present in search_kwargs if not provided
        if 'k' not in search_kwargs:
             search_kwargs['k'] = TOP_K_RESULTS
             
        return self.vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
    
    def delete_collection(self) -> None:
        """Delete the entire collection from the vector store."""
        if not self.vector_store:
             logger.error("Vector store not initialized. Cannot delete collection.")
             return
             
        logger.warning(f"Attempting to delete collection: {self.collection_name}")
        try:
            self.vector_store.delete_collection()
            logger.info(f"Successfully deleted collection '{self.collection_name}'.")
            # Reset the internal vector_store attribute as it's no longer valid
            self.vector_store = None 
            # Optionally, reinitialize an empty one immediately:
            # self._initialize_vector_store()
        except Exception as e:
            logger.error(f"Error deleting collection '{self.collection_name}': {str(e)}", exc_info=True)
            # Consider raising the exception
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection."""
        if not self.vector_store:
             logger.error("Vector store not initialized. Cannot get statistics.")
             return {"error": "Vector store not initialized."}
             
        stats: Dict[str, Any] = {
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory if not (self.chroma_host and self.chroma_port) else None,
            "chroma_host": self.chroma_host,
            "chroma_port": self.chroma_port,
            "document_count": "N/A", # Default value
            "embedding_function_type": type(self.embedding_function).__name__,
        }
        
        try:
             # Use the count method if available
             if hasattr(self.vector_store, '_collection'):
                 doc_count = self.vector_store._collection.count()
                 stats["document_count"] = doc_count
             else:
                  # Try getting count via get() as a fallback, might be slow
                  try:
                     all_ids = self.vector_store.get(include=[]) # Only get IDs
                     stats["document_count"] = len(all_ids.get('ids', []))
                  except Exception as count_err:
                     logger.warning(f"Could not retrieve document count: {count_err}")

        except Exception as e:
            logger.error(f"Error getting vector store statistics: {str(e)}", exc_info=True)
            stats["error"] = str(e)
            
        return stats