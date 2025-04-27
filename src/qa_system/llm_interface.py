"""
LLM interface for Vedic Knowledge AI.
Manages communication with language models for generating responses.
"""
import logging
import os
from typing import Dict, List, Any, Optional, Union
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from ..config import GEMINI_API_KEY, MODEL_NAME, TEMPERATURE, MAX_TOKENS

# Configure logging
logger = logging.getLogger(__name__)

class VedicLLMInterface:
    """Interface for interacting with LLMs."""
    
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        streaming: bool = False,
        api_key: str = None
    ):
        """Initialize the LLM interface."""
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.streaming = streaming
        
        # Use provided API key or fall back to environment variable
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            logger.error("OpenAI API key not provided")
            raise ValueError("OpenAI API key is required")
        
        # Initialize the LLM
        self._initialize_llm()
        
        logger.info(f"Initialized LLM interface with model: {model_name}")
    
    def _initialize_llm(self):
        """Initialize the language model."""
        try:
            # Set up callbacks for streaming if enabled
            callbacks = None
            if self.streaming:
                callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
                callbacks = callback_manager
            
            # Initialize the model
            self.llm = ChatOpenAI(
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                openai_api_key=self.api_key,
                streaming=self.streaming,
                callback_manager=callbacks if self.streaming else None
            )
            
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise
    
    def generate_response(
        self,
        query: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate a response to a query."""
        try:
            messages = []
            
            # Add system prompt if provided
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            
            # Add context and query as human message
            message_content = query
            if context:
                message_content = f"Context:\n{context}\n\nQuestion: {query}"
            
            messages.append(HumanMessage(content=message_content))
            
            # Generate response
            response = self.llm.predict_messages(messages)
            
            # Extract content from response
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def generate_chat_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate a response in a chat conversation."""
        try:
            # Convert messages to LangChain format
            langchain_messages = []
            
            # Add system prompt if provided
            if system_prompt:
                langchain_messages.append(SystemMessage(content=system_prompt))
            
            # Add conversation messages
            for message in messages:
                if message.get("role") == "user":
                    langchain_messages.append(HumanMessage(content=message.get("content", "")))
                elif message.get("role") == "assistant":
                    langchain_messages.append(AIMessage(content=message.get("content", "")))
                elif message.get("role") == "system":
                    langchain_messages.append(SystemMessage(content=message.get("content", "")))
            
            # Generate response
            response = self.llm.predict_messages(langchain_messages)
            
            # Extract content from response
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"Error generating chat response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def change_model(
        self,
        model_name: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> bool:
        """Change the model configuration."""
        try:
            self.model_name = model_name
            
            # Update other parameters if provided
            if temperature is not None:
                self.temperature = temperature
            
            if max_tokens is not None:
                self.max_tokens = max_tokens
            
            # Reinitialize the LLM
            self._initialize_llm()
            
            logger.info(f"Changed model to {model_name} (temp={self.temperature}, max_tokens={self.max_tokens})")
            return True
        except Exception as e:
            logger.error(f"Error changing model: {str(e)}")
            return False
    
    def explain_sanskrit_term(self, term: str, context: Optional[str] = None) -> str:
        """Generate an explanation for a Sanskrit term."""
        system_prompt = """You are an expert in Sanskrit language and Vedic philosophy.
        Explain the given Sanskrit term comprehensively, including:
        1. The original Sanskrit in Devanagari if available
        2. IAST transliteration with diacritical marks
        3. Etymology and literal meaning
        4. Contextual meaning in Vedic or Gaudiya Vaishnava philosophy
        5. Related terms and concepts
        
        If there are multiple meanings or interpretations, include the most relevant ones.
        If you're unsure about any aspect, clearly state what you know and don't know.
        """
        
        return self.generate_response(term, context, system_prompt)
    
    def explain_verse(self, verse: str, reference: Optional[str] = None, context: Optional[str] = None) -> str:
        """Generate an explanation for a verse."""
        query = verse
        if reference:
            query = f"Verse: {verse}\nReference: {reference}"
        
        system_prompt = """You are a scholarly expert on Vedic scriptures and Gaudiya Vaishnava texts.
        Explain the given verse comprehensively, including:
        1. The verse in its original Sanskrit with Devanagari script (if available)
        2. IAST transliteration with proper diacritical marks
        3. Word-by-word meaning
        4. Clear translation of the full verse
        5. Explanation of the verse's meaning according to Gaudiya Vaishnava understanding
        6. References to important commentaries if available
        
        If there are multiple interpretations, include the most authoritative ones.
        If you're unsure about any aspect, clearly state what you know and don't know.
        """
        
        return self.generate_response(query, context, system_prompt)