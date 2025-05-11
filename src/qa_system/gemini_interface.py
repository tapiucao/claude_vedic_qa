# src/qa_system/gemini_interface.py
import os
import logging
from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, BasePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough
from google.api_core import exceptions as google_exceptions

# Import necessary config variables
# Use absolute import from project root perspective if running via app.py
from src.config import MODEL_NAME, TEMPERATURE, MAX_TOKENS, GEMINI_API_KEY
# Use relative import if running this file directly within qa_system? Unlikely.
# from ..config import MODEL_NAME, TEMPERATURE, MAX_TOKENS, GEMINI_API_KEY
from src.utils.logger import setup_logger # Assuming logger setup is handled elsewhere

logger = setup_logger(__name__, "gemini_interface.log") # Use setup_logger


class GeminiLLMInterface:
    """
    Interface for interacting with Google Gemini models via LangChain.
    Handles prompt formatting, context injection, history management, and response generation.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME, # Default from config
        temperature: float = TEMPERATURE, # Default from config
        api_key: Optional[str] = GEMINI_API_KEY, # Default from config
        max_tokens: Optional[int] = MAX_TOKENS # Default from config
    ):
        """
        Initialize the Gemini LLM interface.

        Args:
            model_name (str): The name of the Gemini model to use (e.g., "gemini-1.5-flash").
            temperature (float): The sampling temperature for generation (0.0 - 1.0).
            api_key (Optional[str]): The Google API key. Reads from config by default.
            max_tokens (Optional[int]): The maximum number of tokens to generate.

        Raises:
            ValueError: If the API key is not provided or found in config.
            Exception: For errors during LangChain/Google AI initialization.
        """
        if not api_key:
            logger.error("GEMINI_API_KEY not provided or found in environment/config.")
            raise ValueError("GEMINI_API_KEY is required for GeminiLLMInterface.")

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        try:
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                google_api_key=api_key,
                # convert_system_message_to_human=True # REMOVED - Deprecated and usually not needed
            )

            # Define a flexible chat chain structure
            self.prompt_template = ChatPromptTemplate.from_messages([
                MessagesPlaceholder(variable_name="system_message", optional=True),
                MessagesPlaceholder(variable_name="history", optional=True),
                MessagesPlaceholder(variable_name="context_message", optional=True), # For context if handled separately
                HumanMessage(content="{input}")
            ])

            self.chain: Runnable = self.prompt_template | self.llm | StrOutputParser()

            logger.info(f"Initialized Gemini LLM interface with model: {self.model_name}")

        except google_exceptions.PermissionDenied as e:
             logger.error(f"Permission denied initializing Gemini LLM: {e}. Check API key and project permissions (Generative Language API enabled?).")
             raise
        except (google_exceptions.NotFound, google_exceptions.InvalidArgument) as e:
             logger.error(f"Error initializing Gemini LLM (model '{self.model_name}' not found or invalid argument): {e}")
             logger.error(f"Ensure model name is correct and available for your API key. Check available models if needed.")
             raise
        except Exception as e:
            logger.error(f"Unexpected error initializing Gemini LLM: {str(e)}", exc_info=True)
            raise

    def _prepare_history_messages(self, history: Optional[List[Dict[str, str]]]) -> List[BaseMessage]:
        """Convert history dict list to LangChain BaseMessage list."""
        messages: List[BaseMessage] = []
        if history:
            for msg in history:
                role = msg.get("role", "").lower()
                content = msg.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant" or role == "ai":
                    messages.append(AIMessage(content=content))
                elif role == "system":
                     messages.append(SystemMessage(content=content))
        return messages

    def generate_response(
            self,
            prompt: str,
            context: Optional[str] = None,
            history: Optional[List[Dict[str, str]]] = None,
            system_prompt: Optional[str] = None
        ) -> str:
        """
        Generate a response using the LLM, incorporating context, history, and system prompt.

        Args:
            prompt (str): The main user question or instruction.
            context (Optional[str]): Relevant context to provide to the LLM.
            history (Optional[List[Dict[str, str]]]): Conversation history.
            system_prompt (Optional[str]): System instructions for the LLM.

        Returns:
            str: Generated response text or an error message string.
        """
        history_messages = self._prepare_history_messages(history)
        chain_input: Dict[str, Any] = {"input": prompt}

        if history_messages: chain_input["history"] = history_messages
        if system_prompt: chain_input["system_message"] = [SystemMessage(content=system_prompt)]
        if context: chain_input["context_message"] = [HumanMessage(content=f"Use the following context to answer:\n--- Context Start ---\n{context}\n--- Context End ---")] # Option 2: Context as part of a human-like instruction, potentially before the actual question.

        chain_input["input"] = prompt # The user's direct question

        logger.debug(f"Invoking LLM chain with input keys: {list(chain_input.keys())}")
        # For more detailed logging of the actual content being sent:
        for key, value in chain_input.items():
            if isinstance(value, list) and value and isinstance(value[0], BaseMessage): # Langchain message objects
                logger.debug(f"LLM Chain Input - {key} (Messages):")
                for msg_idx, msg_obj in enumerate(value):
                    logger.debug(f"  Msg {msg_idx} Role: {msg_obj.type}, Content: {str(msg_obj.content)[:200]}...") # Log message type and snippet
            else:
                logger.debug(f"LLM Chain Input - {key}: {str(value)[:300]}...") # Log snippet of other inputs

        try:
            response = self.chain.invoke(chain_input)
            logger.debug(f"LLM Raw Response: {str(response)[:500]}...") # Log raw response snippet

            if not response or not response.strip():
                logger.warning("LLM returned an empty response.")
                return "I apologize, but I received an empty response from the AI model."

            # Basic check for refusals (customize patterns as needed)
            refusal_patterns = ["i cannot fulfill", "i am unable to", "as a large language model"]
            if any(pattern in response.lower() for pattern in refusal_patterns):
                 logger.warning(f"LLM response may indicate refusal: '{response[:100]}...'")

            return response.strip()

        except google_exceptions.ResourceExhausted as e:
             logger.error(f"Rate limit or quota exceeded: {e}")
             return "Error: The request could not be processed due to high demand or quota limits. Please try again later."
        except google_exceptions.InvalidArgument as e:
             # Check for safety blocks
             if "response was blocked" in str(e).lower() or "safety settings" in str(e).lower():
                  logger.warning(f"LLM generation blocked due to safety settings: {e}")
                  return "Error: The response could not be generated due to safety filters."
             logger.error(f"Invalid argument during LLM generation: {e}")
             return f"Error: Invalid request sent to the LLM. Details: {e}"
        except google_exceptions.GoogleAPIError as e:
             logger.error(f"Google API error during LLM generation: {e}")
             return f"Error: An API error occurred while communicating with the AI model. Details: {e}"
        except Exception as e:
            logger.error(f"Unexpected error during LLM generation: {str(e)}", exc_info=True)
            return f"Error: An unexpected error occurred while generating the response."

    # --- Specialized methods ---
    def explain_sanskrit_term(self, term: str, context: Optional[str] = None) -> str:
        """Generate an explanation for a Sanskrit term."""
        system_prompt = """You are a Sanskrit language expert specializing in Vedic and Gaudiya Vaishnava terminology. Explain the provided Sanskrit term clearly and concisely based on the given context and your knowledge. Include: Devanagari script (if known), IAST transliteration, Literal meaning and etymology (if possible), Contextual meaning in Vedic/Gaudiya philosophy, Cite relevant examples from the context if provided. If the context doesn't help or you don't know, state that."""
        user_prompt = f"Explain the Sanskrit term: '{term}'"
        return self.generate_response(prompt=user_prompt, context=context, system_prompt=system_prompt)

    def explain_verse(self, verse: str, reference: Optional[str] = None, context: Optional[str] = None) -> str:
        """Generate an explanation for a verse."""
        system_prompt = """You are a scholarly expert on Vedic scriptures and Gaudiya Vaishnava texts. Explain the provided verse based on the context and your knowledge. Include: Clear translation, Explanation of meaning according to Gaudiya Vaishnava understanding (if applicable), Mention key terms/concepts, Refer to provided context for commentary. If context is unhelpful or you cannot explain, state that clearly."""
        ref_text = f" (Reference: {reference})" if reference else ""
        user_prompt = f"Explain the meaning and significance of the following verse{ref_text}:\n\n'{verse}'"
        return self.generate_response(prompt=user_prompt, context=context, system_prompt=system_prompt)

    def summarize_text_for_query(
        self,
        text_to_summarize: str,
        query_focus: str,
        max_summary_length: str = "concise, around 100-150 words" # Ou número de tokens
    ) -> str:
        """
        Gera um resumo do texto fornecido, focando na sua relevância para o query_focus.
        O texto a ser resumido é passado como 'context'.
        """
        system_prompt = f"""You are an expert summarizer. 
                            Your task is to read the provided text (which will be supplied in the context) and create a {max_summary_length} summary.
                            The summary MUST focus specifically on the parts of the text that are most relevant to the topic: '{query_focus}'.
                            Highlight key points, definitions, or explanations from the text related to this topic.
                            Do not add external information. Base your summary strictly on the provided text.
                            If the text (in the context) has very little or no relevant information about '{query_focus}', please state that clearly (e.g., "The provided text does not contain specific information regarding '{query_focus}'.").
                            It is crucial that you acknowledge the text is provided in the context; do not state that the text is missing."""

        # O prompt principal para generate_response será uma instrução curta,
        # e text_to_summarize irá para o argumento 'context'.
        user_instruction_for_summarization = f"Please summarize the provided text (found in the context given to you) with a specific focus on its relevance to the following topic: '{query_focus}'. Ensure your summary helps answer questions related to this topic."

        logger.debug(f"Requesting summarization. Query focus: '{query_focus}'. Text length for context: {len(text_to_summarize)}")
        
        summary = self.generate_response(
            prompt=user_instruction_for_summarization, # Instrução curta
            context=text_to_summarize,                # Texto longo vai aqui como contexto
            system_prompt=system_prompt
        )

        # Verifica se o LLM explicitamente disse que não há informação relevante
        # (Esta verificação pode ser mais robusta dependendo das respostas exatas do LLM)
        if summary.startswith("Error:"): # Erro retornado pelo generate_response
            logger.error(f"Summarization failed for query focus '{query_focus}'. LLM Error: {summary}")
            return f"[Summarization Error: {summary}]" 
        
        # Verifica frases comuns que indicam que o LLM não encontrou informação relevante no texto
        # (conforme instruído no system_prompt)
        no_info_phrases = [
            f"does not contain specific information regarding '{query_focus.lower()}'", # Convertendo query_focus para minúsculas para match
            f"does not provide significant details regarding '{query_focus.lower()}'"
        ]
        summary_lower = summary.lower()
        if any(phrase in summary_lower for phrase in no_info_phrases):
            logger.info(f"Text had little to no relevant information for query focus '{query_focus}', according to summarizer.")
            return summary # Retorna a própria afirmação do LLM
        
        # Checagem adicional para o caso do LLM insistir que o texto está faltando
        if "provided text is missing" in summary_lower:
            logger.error(f"Summarizer LLM claims text is missing, despite it being provided as context. Length was: {len(text_to_summarize)}")
            return "[Summarizer Error: LLM claims text was missing]"

        return summary