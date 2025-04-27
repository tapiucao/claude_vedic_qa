"""
Gemini interface for Vedic Knowledge AI.
Manages communication with Google's Gemini models for generating responses.
"""
import logging
import os
from typing import Dict, List, Any, Optional

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage

from ..config import TEMPERATURE, MAX_TOKENS

# Configure logging
logger = logging.getLogger(__name__)

# Define the API key directly to avoid problems with environment variables
GEMINI_API_KEY = 'AIzaSyDuLhEqJMWWtTseYm7V5KouXJ-605afKxY'
MODEL_NAME = 'gemini-1.5-pro-latest'  # Updated to known working model

class GeminiLLMInterface:
    """Interface for interacting with Google's Gemini models."""
    
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        streaming: bool = False
    ):
        """Initialize the LLM interface."""
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.streaming = streaming
        self.api_key = GEMINI_API_KEY
        
        # Initialize the LLM
        self.initialize_llm()
        
        logger.info(f"Initialized Gemini LLM interface with model: {model_name}")
    
    def _list_available_models(self):
        """List available Gemini models for debugging."""
        try:
            genai.configure(api_key=self.api_key)
            models = genai.list_models()
            model_names = [model.name for model in models if "gemini" in model.name.lower()]
            logger.info(f"Available Gemini models: {model_names}")
            return model_names
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []
    
    def _extract_model_simple_name(self, full_name):
        """Extract the simple model name from a full model path."""
        # Handle cases where the name might be full path like 'models/gemini-1.5-pro-latest'
        if '/' in full_name:
            return full_name.split('/')[-1]
        return full_name
    
    def initialize_llm(self):
        """Initialize the language model."""
        try:
            # Configure the Gemini API
            genai.configure(api_key=self.api_key)
            
            # List available models for debugging
            available_models = self._list_available_models()
            
            # Check if our model is in the list of available models
            model_simple_name = self._extract_model_simple_name(self.model_name)
            available_model_names = [self._extract_model_simple_name(m) for m in available_models]
            
            # If model not found or not fully qualified with 'models/' prefix, find the full path
            if available_models and model_simple_name not in available_model_names:
                # Try to use the first available model instead
                if available_models:
                    logger.warning(f"Model {self.model_name} not found. Using {available_models[0]} instead.")
                    self.model_name = available_models[0]
            elif available_models and model_simple_name in available_model_names and not self.model_name.startswith('models/'):
                # Find the full model path if we only have the simple name
                for model in available_models:
                    if model.endswith(model_simple_name):
                        self.model_name = model
                        break
            
            # Initialize the model
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                convert_system_message_to_human=True,
                google_api_key=self.api_key,
                streaming=self.streaming
            )
            
            logger.info("Gemini LLM initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Gemini LLM: {str(e)}")
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
            
            # Try to generate response
            try:
                # Generate response using the updated invoke method
                response = self.llm.invoke(messages)
                
                # Extract content from response
                if hasattr(response, 'content'):
                    return response.content
                else:
                    return str(response)
            except Exception as e:
                logger.error(f"Error in LLM call: {str(e)}")
                
                # Try to reinitialize with a different model
                available_models = self._list_available_models()
                if available_models:
                    for model_name in available_models:
                        model_simple_name = self._extract_model_simple_name(model_name)
                        if model_simple_name != self._extract_model_simple_name(self.model_name):
                            logger.info(f"Trying alternative model: {model_simple_name}")
                            self.model_name = model_name
                            self.initialize_llm()
                            try:
                                response = self.llm.invoke(messages)
                                if hasattr(response, 'content'):
                                    return response.content
                                else:
                                    return str(response)
                            except Exception as retry_e:
                                logger.error(f"Error with alternative model: {str(retry_e)}")
                
                # If all retries fail, propagate the original error
                return f"Error generating response: {str(e)}"
                
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
            
            # Generate response using the updated invoke method
            try:
                response = self.llm.invoke(langchain_messages)
                
                # Extract content from response
                if hasattr(response, 'content'):
                    return response.content
                else:
                    return str(response)
            except Exception as e:
                logger.error(f"Error in chat response: {str(e)}")
                
                # Try alternative models
                available_models = self._list_available_models()
                if available_models:
                    for model_name in available_models:
                        model_simple_name = self._extract_model_simple_name(model_name)
                        if model_simple_name != self._extract_model_simple_name(self.model_name):
                            logger.info(f"Trying alternative model: {model_simple_name}")
                            self.model_name = model_name
                            self.initialize_llm()
                            try:
                                response = self.llm.invoke(langchain_messages)
                                if hasattr(response, 'content'):
                                    return response.content
                                else:
                                    return str(response)
                            except Exception as retry_e:
                                logger.error(f"Error with alternative model: {str(retry_e)}")
                
                # If all retries fail, propagate the original error
                return f"Error generating response: {str(e)}"
                
        except Exception as e:
            logger.error(f"Error generating chat response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
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