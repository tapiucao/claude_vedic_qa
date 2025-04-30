# src/knowledge_base/vector_store.py
import os
import logging
from typing import List, Dict, Any, Optional
from langchain_chroma import Chroma
import chromadb # Import the chromadb client library
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import DB_DIR, CHROMA_COLLECTION_NAME # Keep DB_DIR for potential local backups/metadata? Or remove if unused.
from src.utils.logger import setup_logger

logger = setup_logger(__name__, "vector_store.log")

# Get Chroma server details from environment variables (set in docker-compose)
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost") # Default to localhost if not running in Docker
CHROMA_PORT = os.getenv("CHROMA_PORT", "8001")      # Default port

class VedicVectorStore:
    """Handles interactions with the Chroma vector store."""

    def __init__(self, embedding_function: Embeddings, collection_name: str = CHROMA_COLLECTION_NAME):
        """Initialize the vector store client."""
        self.embedding_function = embedding_function
        self.collection_name = collection_name

        try:
            logger.info(f"Connecting to ChromaDB server at {CHROMA_HOST}:{CHROMA_PORT}")
            # Initialize the ChromaDB HTTP client
            self.client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

            # Initialize LangChain Chroma integration with the client and embedding function
            self.vector_store = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
                # Persistence is handled by the Chroma server now, remove persist_directory
            )
            # Check connection by trying to get the collection (raises exception if unavailable)
            self.client.get_collection(self.collection_name) # Creates if not exists by default with get_or_create
            count = self.vector_store._collection.count()
            logger.info(f"Connected to vector store. Collection '{self.collection_name}' has {count} documents.")

        except Exception as e:
            logger.error(f"Failed to connect or initialize ChromaDB client at {CHROMA_HOST}:{CHROMA_PORT}: {e}")
            # Handle specific connection errors if possible
            raise ConnectionError(f"Could not connect to ChromaDB server at {CHROMA_HOST}:{CHROMA_PORT}") from e

    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store."""
        if not documents:
            logger.warning("No documents provided to add.")
            return

        try:
            logger.info(f"Adding {len(documents)} documents to collection '{self.collection_name}'...")
            # Use LangChain's add_documents method
            ids = self.vector_store.add_documents(documents)
            logger.info(f"Successfully added {len(ids)} documents to vector store.")
            # No need to call self.vector_store.persist() anymore
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            # Consider adding retry logic here if needed

    def search(self, query: str, k: int = 5, filter_dict: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Search for similar documents."""
        try:
            logger.debug(f"Searching for '{query}' (k={k}, filter={filter_dict})")
            results = self.vector_store.similarity_search(query, k=k, filter=filter_dict)
            logger.info(f"Found {len(results)} results for query: {query}")
            return results
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []

    def get_retriever(self, k: int = 5, filter_dict: Optional[Dict[str, Any]] = None):
         """Get a LangChain retriever instance."""
         search_kwargs = {'k': k}
         if filter_dict:
             search_kwargs['filter'] = filter_dict
         return self.vector_store.as_retriever(search_kwargs=search_kwargs)


    def filter_by_metadata(self, filter_dict: Dict[str, Any], k: int = 100) -> List[Document]:
         """Filter documents by metadata (Note: Chroma filtering capabilities might vary)."""
         logger.warning("Metadata filtering effectiveness depends on Chroma version and implementation.")
         # This basic search might not be the most efficient way to filter large datasets in Chroma.
         # Consider using Chroma's direct query capabilities if complex filtering is needed.
         try:
             # Use similarity search with a dummy query and filter - less efficient
             # results = self.vector_store.similarity_search(" ", k=k, filter=filter_dict)

             # Better approach (if LangChain interface supports it well or use client directly):
             # Use the retriever with the filter
             retriever = self.get_retriever(k=k, filter_dict=filter_dict)
             results = retriever.get_relevant_documents(" ") # Use a neutral query string

             logger.info(f"Found {len(results)} documents matching filter: {filter_dict}")
             return results
         except Exception as e:
             logger.error(f"Error filtering by metadata: {e}")
             return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection."""
        try:
            count = self.vector_store._collection.count()
            metadata = self.vector_store._collection.metadata # Might be None
            return {
                "document_count": count,
                "collection_name": self.collection_name,
                "connection_host": CHROMA_HOST,
                "connection_port": CHROMA_PORT,
                "collection_metadata": metadata
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {
                "error": str(e),
                "collection_name": self.collection_name,
                "connection_host": CHROMA_HOST,
                 "connection_port": CHROMA_PORT,
            }
    
    def filter_by_source_type(
        self, 
        source_type: str,
        k: int = 100
    ) -> List[Document]:
        """Get documents by source type (pdf or website)."""
        filter_dict = {"type": source_type}
        return self.filter_by_metadata(filter_dict, k)