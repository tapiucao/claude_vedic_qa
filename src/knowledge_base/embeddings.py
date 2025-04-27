"""
Embedding models for Vedic Knowledge AI.
Handles text embedding generation optimized for Sanskrit and Vedic content.
"""
import logging
from typing import List, Dict, Any
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings

from ..config import EMBEDDING_MODEL, OPENAI_API_KEY

# Configure logging
logger = logging.getLogger(__name__)

def get_huggingface_embeddings(
    model_name: str = EMBEDDING_MODEL,
    model_kwargs: Dict[str, Any] = {"device": "cpu"}
) -> HuggingFaceEmbeddings:
    """Initialize HuggingFace embeddings model."""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs
        )
        logger.info(f"Initialized HuggingFace embeddings with model: {model_name}")
        return embeddings
    except Exception as e:
        logger.error(f"Error initializing HuggingFace embeddings: {str(e)}")
        raise

def get_openai_embeddings() -> OpenAIEmbeddings:
    """Initialize OpenAI embeddings."""
    if not OPENAI_API_KEY:
        logger.error("OpenAI API key not found")
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    try:
        embeddings = OpenAIEmbeddings()
        logger.info("Initialized OpenAI embeddings")
        return embeddings
    except Exception as e:
        logger.error(f"Error initializing OpenAI embeddings: {str(e)}")
        raise

class VedicEmbeddingSelector:
    """Selector for choosing the appropriate embedding model."""
    
    def __init__(self, default_model: str = "huggingface"):
        """Initialize with default model type."""
        self.default_model = default_model
        self.embedding_models = {}
        
        logger.info(f"Initialized embedding selector with default model: {default_model}")
    
    def get_embeddings(self, model_type: str = None) -> Any:
        """Get embedding model based on type."""
        # Use default if not specified
        if model_type is None:
            model_type = self.default_model
        
        # Check if model is already initialized
        if model_type in self.embedding_models:
            return self.embedding_models[model_type]
        
        # Initialize the requested model
        if model_type == "huggingface":
            embeddings = get_huggingface_embeddings()
        elif model_type == "openai":
            embeddings = get_openai_embeddings()
        else:
            logger.error(f"Unknown embedding model type: {model_type}")
            raise ValueError(f"Unknown embedding model type: {model_type}")
        
        # Cache the model
        self.embedding_models[model_type] = embeddings
        
        return embeddings