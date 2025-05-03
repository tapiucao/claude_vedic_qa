"""
Vector database operations for Vedic Knowledge AI.
Handles storing, retrieving, and managing vector embeddings.
"""
import os
import logging
from typing import List, Dict, Any, Optional, Union
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from ..config import DB_DIR, TOP_K_RESULTS

# Configure logging
logger = logging.getLogger(__name__)

class VedicVectorStore:
    """Vector store manager for Vedic textual knowledge."""
    
    def __init__(
        self,
        embedding_function: HuggingFaceEmbeddings,
        persist_directory: str = DB_DIR,
        collection_name: str = "vedic_knowledge",
        chroma_host: Optional[str] = None,
        chroma_port: Optional[int] = None
    ):
        """Initialize the vector store with an embedding function."""
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Check for external Chroma settings from environment
        self.chroma_host = chroma_host or os.environ.get("CHROMA_HOST")
        self.chroma_port = chroma_port or os.environ.get("CHROMA_PORT")
        
        # Create directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize the vector store
        self._initialize_vector_store()
        
        logger.info(f"Initialized vector store at {self.persist_directory}")
    
    def _initialize_vector_store(self):
        """Initialize the vector store."""
        try:
            # Check if using external Chroma server
            if self.chroma_host and self.chroma_port:
                # Connect to external Chroma server
                from chromadb.config import Settings
                import chromadb
                
                client = chromadb.HttpClient(
                    host=self.chroma_host,
                    port=self.chroma_port,
                    settings=Settings(anonymized_telemetry=False)
                )
                
                # Initialize with external client
                self.vector_store = Chroma(
                    client=client,
                    embedding_function=self.embedding_function,
                    collection_name=self.collection_name
                )
                logger.info(f"Connected to external Chroma server at {self.chroma_host}:{self.chroma_port}")
            else:
                # Use local persistence
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding_function,
                    collection_name=self.collection_name
                )
                logger.info(f"Using local Chroma at {self.persist_directory}")
            
            # Log document count
            doc_count = self.vector_store._collection.count()
            logger.info(f"Vector store has {doc_count} documents")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        if not documents:
            logger.warning("No documents provided to add to vector store")
            return
        
        try:
            # Add documents to the vector store
            self.vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def similarity_search(
        self, 
        query: str, 
        k: int = TOP_K_RESULTS,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Search for documents similar to the query."""
        try:
            # Make sure filter is properly formatted if provided but empty
            if filter is not None and not filter:
                filter = None
                
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter
            )
            logger.info(f"Found {len(results)} results for query: {query}")
            return results
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []

    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = TOP_K_RESULTS,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """Search for documents with similarity scores."""
        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter
            )
            logger.info(f"Found {len(results)} scored results for query: {query}")
            return results
        except Exception as e:
            logger.error(f"Error searching vector store with scores: {str(e)}")
            return []
    
    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """Get a retriever interface to the vector store."""
        if search_kwargs is None:
            search_kwargs = {"k": TOP_K_RESULTS}
        
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        try:
            self.vector_store.delete_collection()
            logger.info(f"Deleted collection {self.collection_name}")
            # Reinitialize an empty vector store
            self._initialize_vector_store()
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        try:
            doc_count = self.vector_store._collection.count()
            
            # Get metadata about the collection
            stats = {
                "document_count": doc_count,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory
            }
            
            return stats
        except Exception as e:
            logger.error(f"Error getting vector store statistics: {str(e)}")
            return {"error": str(e)}
    
    def filter_by_metadata(
        self, 
        filter_dict: Dict[str, Any],
        k: int = TOP_K_RESULTS
    ) -> List[Document]:
        """Get documents that match metadata filters."""
        try:
            results = self.vector_store.get(
                where=filter_dict,
                k=k
            )
            logger.info(f"Found {len(results)} documents matching filter: {filter_dict}")
            return results
        except Exception as e:
            logger.error(f"Error filtering by metadata: {str(e)}")
            return []
    
    def filter_by_source_type(
        self, 
        source_type: str,
        k: int = 100
    ) -> List[Document]:
        """Get documents by source type (pdf or website)."""
        filter_dict = {"type": source_type}
        return self.filter_by_metadata(filter_dict, k)
    
    def search(self, query: str, k: int = 10, filter_dict: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Search for similar documents."""
        try:
            logger.debug(f"Searching for '{query}' (k={k}, filter={filter_dict})")
            # This line passes the filter_dict to the LangChain Chroma wrapper
            results = self.vector_store.similarity_search(query, k=k, filter=filter_dict)
            logger.info(f"Found {len(results)} results for query: {query}")
            return results
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []
    
    def retrieve_documents(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        k: Optional[int] = None
    ) -> List[Document]:
        """Retrieve relevant documents for a query."""
        k = k or self.top_k
        
        try:
            # Make sure filter_dict is properly formatted if provided but empty
            if filter_dict is not None and not filter_dict:
                filter_dict = None
                
            docs = self.vector_store.similarity_search(query, k=k, filter=filter_dict)
            logger.info(f"Retrieved {len(docs)} documents for query: {query}")
            return docs
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []