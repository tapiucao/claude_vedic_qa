"""
Retrieval system for Vedic Knowledge AI.
Handles finding relevant documents and context for queries.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from ..knowledge_base.vector_store import VedicVectorStore
from ..knowledge_base.prompt_templates import select_prompt_template
from .llm_interface import VedicLLMInterface
from ..config import TOP_K_RESULTS

# Configure logging
logger = logging.getLogger(__name__)

class VedicRetriever:
    """Retrieval system for finding relevant Vedic knowledge."""
    
    def __init__(
        self,
        vector_store: VedicVectorStore,
        llm_interface: VedicLLMInterface,
        top_k: int = TOP_K_RESULTS
    ):
        """Initialize the retriever with vector store and LLM interface."""
        self.vector_store = vector_store
        self.llm_interface = llm_interface
        self.top_k = top_k
        
        logger.info(f"Initialized Vedic retriever with top_k={top_k}")
    
    def retrieve_documents(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        k: Optional[int] = None
    ) -> List[Document]:
        """Retrieve relevant documents for a query."""
        k = k or self.top_k
        
        try:
            docs = self.vector_store.similarity_search(query, k=k, filter=filter_dict)
            logger.info(f"Retrieved {len(docs)} documents for query: {query}")
            return docs
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def retrieve_documents_with_scores(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """Retrieve relevant documents with similarity scores."""
        k = k or self.top_k
        
        try:
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k, filter=filter_dict)
            logger.info(f"Retrieved {len(docs_with_scores)} scored documents for query: {query}")
            return docs_with_scores
        except Exception as e:
            logger.error(f"Error retrieving documents with scores: {str(e)}")
            return []
    
    def get_context_for_query(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        k: Optional[int] = None
    ) -> str:
        """Get concatenated context from relevant documents."""
        docs = self.retrieve_documents(query, filter_dict, k)
        
        if not docs:
            return ""
        
        # Concatenate document contents with separators
        context_parts = []
        for i, doc in enumerate(docs):
            # Extract source information for citation
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            doc_type = doc.metadata.get("type", "document")
            
            # Format citation based on document type
            if doc_type == "website":
                citation = f"[Source {i+1}: {source}]"
            elif page:
                citation = f"[Source {i+1}: {source}, page {page}]"
            else:
                citation = f"[Source {i+1}: {source}]"
            
            # Add document content with citation
            context_parts.append(f"{doc.page_content}\n{citation}")
        
        return "\n\n".join(context_parts)
    
    def answer_query(self, query: str, filter_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Answer a query using retrieved documents."""
        # Retrieve documents
        docs = self.retrieve_documents(query, filter_dict)
        
        if not docs:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "documents": []
            }
        
        # Prepare context
        context = self.get_context_for_query(query, filter_dict)
        
        # Select appropriate prompt template based on query
        prompt_template = select_prompt_template(query)
        
        # Generate answer
        answer = self.llm_interface.generate_response(
            query=query,
            context=context,
            system_prompt=prompt_template.template
        )
        
        # Extract source information
        sources = []
        for doc in docs:
            source = {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", ""),
                "type": doc.metadata.get("type", "document"),
                "title": doc.metadata.get("title", "")
            }
            sources.append(source)
        
        result = {
            "answer": answer,
            "sources": sources,
            "documents": docs
        }
        
        return result
    
    def setup_retrieval_chain(self) -> RetrievalQA:
        """Set up a retrieval chain for more complex querying."""
        # Get the retriever interface
        retriever = self.vector_store.get_retriever(search_kwargs={"k": self.top_k})
        
        # Create the chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm_interface.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        return qa_chain
    
    def answer_with_chain(self, query: str, chain_type: str = "stuff") -> Dict[str, Any]:
        """Answer a query using a retrieval chain."""
        # Get the retriever interface
        retriever = self.vector_store.get_retriever(search_kwargs={"k": self.top_k})
        
        # Select appropriate prompt template based on query
        prompt_template = select_prompt_template(query)
        
        # Create the chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm_interface.llm,
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )
        
        # Run the chain
        try:
            result = qa_chain({"query": query})
            
            # Extract answer and sources
            answer = result.get("result", "")
            docs = result.get("source_documents", [])
            
            # Extract source information
            sources = []
            for doc in docs:
                source = {
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", ""),
                    "type": doc.metadata.get("type", "document"),
                    "title": doc.metadata.get("title", "")
                }
                sources.append(source)
            
            return {
                "answer": answer,
                "sources": sources,
                "documents": docs
            }
        except Exception as e:
            logger.error(f"Error in retrieval chain: {str(e)}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": [],
                "documents": []
            }