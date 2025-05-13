# src/knowledge_base/embeddings.py
"""
Embedding models for Vedic Knowledge AI.
Handles text embedding generation optimized for Sanskrit and Vedic content.
Includes capability for GPU acceleration via CUDA.
"""
import logging
import os
from typing import List, Dict, Any, Optional # Ensure Optional is imported
from langchain_huggingface import HuggingFaceEmbeddings
import torch # Import torch to check for CUDA

# Import the configured model name from config
from ..config import EMBEDDING_MODEL

# Configure logging
logger = logging.getLogger(__name__)

# --- Determine best available device ---
DEFAULT_DEVICE = "cpu"
if torch.cuda.is_available():
    DEFAULT_DEVICE = "cuda"
elif torch.backends.mps.is_available(): # Check for Apple Silicon MPS
     DEFAULT_DEVICE = "mps"

logger.info(f"Default embedding device determined: {DEFAULT_DEVICE}")
# ---

def get_huggingface_embeddings(
    # Use the imported EMBEDDING_MODEL as the default
    # Ensure this matches the model used when the vector database was created!
    model_name: str = EMBEDDING_MODEL,
    model_kwargs: Optional[Dict[str, Any]] = None,
    encode_kwargs: Dict[str, Any] = {'normalize_embeddings': True}
) -> HuggingFaceEmbeddings:
    """
    Initialize HuggingFace embeddings model, attempting to use GPU if available.

    Args:
        model_name (str): Name of the HuggingFace model. Defaults to config.EMBEDDING_MODEL.
        model_kwargs (Optional[Dict[str, Any]]): Arguments for the model constructor.
                                                If None, defaults to {"device": DEFAULT_DEVICE}.
        encode_kwargs (Dict[str, Any]): Arguments for the encoding process.

    Returns:
        HuggingFaceEmbeddings instance.
    """
    # Set default model_kwargs if not provided, using the detected best device
    if model_kwargs is None:
        model_kwargs = {"device": DEFAULT_DEVICE}
    # Ensure device is set if model_kwargs was provided but missing device
    elif "device" not in model_kwargs:
         model_kwargs["device"] = DEFAULT_DEVICE

    device = model_kwargs["device"] # Get the determined device for logging

    # Check if requested device is actually available if it's not CPU
    if device == "cuda" and not torch.cuda.is_available():
         logger.warning("CUDA requested but not available, falling back to CPU.")
         model_kwargs["device"] = "cpu"
         device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
         logger.warning("MPS requested but not available, falling back to CPU.")
         model_kwargs["device"] = "cpu"
         device = "cpu"

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        logger.info(f"Initialized HuggingFace embeddings with model: {model_name} on device: {device}")
        return embeddings
    except ImportError as ie:
        logger.error("langchain_huggingface or sentence_transformers not found. Install with: pip install langchain-huggingface sentence-transformers")
        logger.error("Ensure PyTorch is installed correctly: pip install torch torchvision torchaudio")
        if device == "cuda":
             logger.error("For GPU, ensure CUDA drivers and compatible PyTorch CUDA build are installed.")
        raise ie
    except Exception as e:
        logger.error(f"Error initializing HuggingFace embeddings model '{model_name}' on device '{device}': {str(e)}", exc_info=True)
        logger.error("Check model name, network connection, disk space, and CUDA/driver setup if using GPU.")
        raise

class VedicEmbeddingSelector:
    """Selector for choosing and managing embedding models."""
    def __init__(self, default_model_type: str = "huggingface"):
        """Initialize with default model type."""
        self.default_model_type = default_model_type
        self.embedding_models: Dict[str, Any] = {}
        logger.info(f"Initialized embedding selector with default type: {default_model_type}")

    def get_embeddings(self, model_type: Optional[str] = None, **kwargs) -> Any:
        """
        Get an initialized embedding model based on type, caching instances.

        Args:
            model_type (Optional[str]): The type of model (e.g., "huggingface"). Defaults to self.default_model_type.
            **kwargs: Arguments passed to the embedding model init function (e.g., model_name, model_kwargs).

        Returns:
            An initialized LangChain Embeddings object.

        Raises:
            ValueError: If an unknown model_type is requested.
        """
        resolved_model_type = model_type or self.default_model_type

        # Create a cache key based on type, model name, and device
        cache_key = resolved_model_type
        target_device = kwargs.get("model_kwargs", {}).get("device", DEFAULT_DEVICE)
        if resolved_model_type == "huggingface":
             model_name_key = kwargs.get('model_name', EMBEDDING_MODEL)
             cache_key += f":{model_name_key}:{target_device}"

        if cache_key in self.embedding_models:
            logger.debug(f"Returning cached embedding model for key: {cache_key}")
            return self.embedding_models[cache_key]

        logger.info(f"Initializing new embedding model for key: {cache_key} with args: {kwargs}")

        if resolved_model_type == "huggingface":
            # Pass all kwargs down to the initialization function
            embeddings = get_huggingface_embeddings(**kwargs)
        else:
            logger.error(f"Unknown embedding model type requested: {resolved_model_type}")
            raise ValueError(f"Unknown embedding model type: {resolved_model_type}")

        self.embedding_models[cache_key] = embeddings
        logger.info(f"Cached new embedding model instance for key: {cache_key}")
        return embeddings

# Optional: Example usage remains the same
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        selector = VedicEmbeddingSelector()
        # Get default (tries GPU first if available)
        print("\n--- Testing Default Embedding ---")
        default_embeddings = selector.get_embeddings()
        print(f"Default Embedding Device: {default_embeddings.client.device}")
        print(f"Default Model Name: {default_embeddings.model_name}")
        vector = default_embeddings.embed_query("Test query")
        print(f"Vector dimension (default): {len(vector)}")

        # Force CPU
        print("\n--- Testing CPU Embedding ---")
        cpu_embeddings = selector.get_embeddings(model_kwargs={"device": "cpu"})
        print(f"CPU Embedding Device: {cpu_embeddings.client.device}")
        print(f"CPU Model Name: {cpu_embeddings.model_name}")
        vector_cpu = cpu_embeddings.embed_query("Test query on CPU")
        print(f"Vector dimension (CPU): {len(vector_cpu)}")

        # Get default again (should be cached)
        print("\n--- Testing Cache ---")
        default_again = selector.get_embeddings()
        print(f"Default again is same instance: {default_again is default_embeddings}")
        cpu_again = selector.get_embeddings(model_kwargs={"device": "cpu"})
        print(f"CPU again is same instance: {cpu_again is cpu_embeddings}")

        # Get different model
        print("\n--- Testing Different Model ---")
        minilm_embeddings = selector.get_embeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
        print(f"MiniLM Embedding Device: {minilm_embeddings.client.device}")
        print(f"MiniLM Model Name: {minilm_embeddings.model_name}")
        vector_minilm = minilm_embeddings.embed_query("Test query MiniLM")
        print(f"Vector dimension (MiniLM): {len(vector_minilm)}")


    except Exception as e:
        print(f"An error occurred during embedding test: {e}")