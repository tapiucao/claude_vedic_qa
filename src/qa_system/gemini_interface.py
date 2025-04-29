# src/qa_system/gemini_interface.py

import os
import logging
from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from google.api_core import exceptions as google_exceptions 
from src.config import MODEL_NAME, TEMPERATURE, MAX_TOKENS, GEMINI_API_KEY
from src.utils.logger import setup_logger

logger = setup_logger(__name__, "gemini_interface.log")

class GeminiLLMInterface:
    """Interface for interacting with Google Gemini models."""

    def __init__(self, model_name: str = MODEL_NAME, temperature: float = TEMPERATURE, api_key: str = GEMINI_API_KEY):
        """Initialize the Gemini LLM interface."""
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment variables.")
            raise ValueError("GEMINI_API_KEY is required.")

        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key

        try:
            # Use ChatGoogleGenerativeAI from langchain-google-genai
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                google_api_key=self.api_key,
                convert_system_message_to_human=True # Keep this for now, but monitor LangChain updates
            )
            # Define a basic chat chain with string output parser
            self.chain = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant knowledgeable about Vedic scriptures."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]) | self.llm | StrOutputParser()

            logger.info(f"Initialized Gemini LLM interface with model: {self.model_name}")

        except Exception as e:
            logger.error(f"Error initializing Gemini LLM: {str(e)}")
            # Provide more specific guidance on model errors
            if isinstance(e, (google_exceptions.NotFound, google_exceptions.InvalidArgument)):
                 logger.error(f"Check if model name '{self.model_name}' is correct and available for your API key.")
            raise

    def _prepare_messages(self, system_prompt: str, user_prompt: str, history: Optional[List[Dict[str, str]]] = None) -> List[BaseMessage]:
        """Prepare messages for the LLM."""
        messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
        if history:
            for msg in history:
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=msg.get("content", "")))
        messages.append(HumanMessage(content=user_prompt))
        return messages

    def generate_response(self, prompt: str, context: Optional[str] = None, 
                        history: Optional[List[Dict[str, str]]] = None,
                        system_prompt: Optional[str] = None) -> str:
        """Generate a response using the LLM, incorporating context, history, and system prompt."""
        # Use provided system prompt or default one
        system_template = system_prompt or """You are an AI assistant specializing in Vedic knowledge and Gaudiya Vaishnavism.
        Answer the user's question based *only* on the provided context.
        If the context doesn't contain the answer, state that clearly.
        Be concise, accurate, and cite sources if possible from the context metadata.
        Context:
        {context}
        """
        
        user_prompt = f"Question: {prompt}"

        # Construct the input dictionary for the chain
        chain_input = {"input": user_prompt}

        # Format history
        formatted_history = []
        if history:
            # Convert history format if needed for MessagesPlaceholder
            for msg in history:
                if msg.get("role") == "user":
                    formatted_history.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    formatted_history.append(AIMessage(content=msg.get("content", "")))
        
        # Add the formatted history to the input
        chain_input["history"] = formatted_history

        # Add context if provided
        if context:
            chain_input["input"] = f"Based on the following context:\n{context}\n\nQuestion: {prompt}"

        try:
            # Use invoke with the prepared input dictionary
            response = self.chain.invoke(chain_input)
            return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            # Error handling logic...
            return f"Error generating response: {str(e)}"


    def explain_sanskrit_term(self, term: str, context: Optional[str] = None) -> str:
        """Generate an explanation for a Sanskrit term."""
        prompt = f"Explain the Sanskrit term '{term}'. Provide its meaning, etymology (if known), and significance, using the provided context for examples or clarification."
        return self.generate_response(prompt, context)

    def explain_verse(self, verse: str, reference: Optional[str] = None, context: Optional[str] = None) -> str:
        """Generate an explanation for a verse."""
        ref_text = f" (Reference: {reference})" if reference else ""
        prompt = f"Explain the meaning and significance of the following verse{ref_text}: '{verse}'. Use the provided context for clarification or commentary."
        return self.generate_response(prompt, context)