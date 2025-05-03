# src/knowledge_base/embeddings.py
"""
Embedding models for Vedic Knowledge AI.
Handles text embedding generation optimized for Sanskrit and Vedic content.
"""
import logging
from typing import List, Dict, Any, Optional
from langchain_huggingface import HuggingFaceEmbeddings

# Import the configured model name from config
from ..config import EMBEDDING_MODEL

# Configure logging
logger = logging.getLogger(__name__)

def get_huggingface_embeddings(
    # Use the imported EMBEDDING_MODEL as the default
    model_name: str = EMBEDDING_MODEL,
    model_kwargs: Dict[str, Any] = {"device": "cpu"}, # Default to CPU, can be overridden
    encode_kwargs: Dict[str, Any] = {'normalize_embeddings': True} # Recommended for sentence-transformers
) -> HuggingFaceEmbeddings:
    """
    Initialize HuggingFace embeddings model.

    Args:
        model_name (str): The name of the HuggingFace model to use.
                          Defaults to the EMBEDDING_MODEL specified in config.py.
        model_kwargs (Dict[str, Any]): Arguments passed to the underlying model constructor
                                      (e.g., device mapping: {"device": "cuda"}).
        encode_kwargs (Dict[str, Any]): Arguments passed during the encoding process
                                       (e.g., {'normalize_embeddings': True}).

    Returns:
        HuggingFaceEmbeddings: An initialized LangChain embedding object.

    Raises:
        ImportError: If langchain_huggingface is not installed.
        Exception: For other initialization errors (e.g., model download failure).
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        logger.info(f"Initialized HuggingFace embeddings with model: {model_name} on device: {model_kwargs.get('device', 'cpu')}")
        return embeddings
    except ImportError as ie:
        logger.error("langchain_huggingface or sentence_transformers not found. Install with: pip install langchain-huggingface sentence-transformers")
        raise ie
    except Exception as e:
        logger.error(f"Error initializing HuggingFace embeddings model '{model_name}': {str(e)}")
        logger.error("Check model name, network connection, and disk space.")
        raise

class VedicEmbeddingSelector:
    """
    Selector for choosing and managing embedding models.
    Currently supports HuggingFace models.
    """

    def __init__(self, default_model_type: str = "huggingface"):
        """Initialize with default model type."""
        self.default_model_type = default_model_type
        # Cache for initialized embedding models
        self.embedding_models: Dict[str, Any] = {}
        logger.info(f"Initialized embedding selector with default type: {default_model_type}")

    def get_embeddings(self, model_type: Optional[str] = None, **kwargs) -> Any:
        """
        Get an initialized embedding model based on type, caching instances.

        Args:
            model_type (Optional[str]): The type of model to get (e.g., "huggingface").
                                       Defaults to self.default_model_type.
            **kwargs: Additional keyword arguments passed to the embedding model
                      initialization function (e.g., model_name, model_kwargs).

        Returns:
            An initialized LangChain Embeddings object.

        Raises:
            ValueError: If an unknown model_type is requested.
        """
        resolved_model_type = model_type or self.default_model_type

        # Create a cache key based on type and relevant kwargs (e.g., model_name if provided)
        # This prevents re-initializing the same HF model if called with the same name
        cache_key = resolved_model_type
        if resolved_model_type == "huggingface" and "model_name" in kwargs:
            cache_key += f":{kwargs['model_name']}"
        elif resolved_model_type == "huggingface":
             cache_key += f":{EMBEDDING_MODEL}" # Use default from config if no name specified


        # Check cache first
        if cache_key in self.embedding_models:
            logger.debug(f"Returning cached embedding model for key: {cache_key}")
            return self.embedding_models[cache_key]

        logger.info(f"Initializing new embedding model for type: {resolved_model_type} with args: {kwargs}")

        # Initialize the requested model
        if resolved_model_type == "huggingface":
            # Pass kwargs to the HuggingFace initialization function
            embeddings = get_huggingface_embeddings(**kwargs)
        # TODO: Add other types like OpenAI, Cohere etc. if needed
        # elif resolved_model_type == "openai":
        #     from langchain_openai import OpenAIEmbeddings # Example
        #     embeddings = OpenAIEmbeddings(**kwargs)
        else:
            logger.error(f"Unknown embedding model type requested: {resolved_model_type}")
            raise ValueError(f"Unknown embedding model type: {resolved_model_type}")

        # Cache the initialized model
        self.embedding_models[cache_key] = embeddings
        logger.info(f"Cached new embedding model instance for key: {cache_key}")

        return embeddings

# Optional: Example usage (if run directly)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        # Example 1: Get default embedding model
        selector = VedicEmbeddingSelector()
        default_embeddings = selector.get_embeddings()
        print(f"Default Embedding Model Type: {type(default_embeddings)}")
        vector = default_embeddings.embed_query("Test query")
        print(f"Vector dimension (default): {len(vector)}")

        # Example 2: Get a specific HuggingFace model
        specific_model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
        specific_embeddings = selector.get_embeddings(
            model_type="huggingface",
            model_name=specific_model_name,
            model_kwargs={"device": "cpu"} # Explicitly CPU
        )
        print(f"Specific Embedding Model Type: {type(specific_embeddings)}")
        vector_specific = specific_embeddings.embed_query("Another test query")
        print(f"Vector dimension (specific): {len(vector_specific)}")

        # Example 3: Get default again (should be cached)
        default_again = selector.get_embeddings()
        print(f"Default again is same instance: {default_again is default_embeddings}")

    except Exception as e:
        print(f"An error occurred during embedding test: {e}")