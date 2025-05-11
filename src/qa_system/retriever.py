# src/qa_system/retriever.py

import logging
import re
import os
from typing import List, Dict, Any, Optional, Sequence, Callable 
import concurrent.futures
from urllib.parse import urlparse

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from ..knowledge_base.vector_store import VedicVectorStore
from ..knowledge_base.prompt_templates import select_prompt_template, VEDIC_QA_PROMPT
from .gemini_interface import GeminiLLMInterface
from ..config import TOP_K_RESULTS, TRUSTED_WEBSITES
from ..web_scraper.scraper import VedicWebScraper
from ..web_scraper.ethics import is_blacklisted_url, respect_robots_txt

logger = logging.getLogger(__name__)

class VedicRetriever:
    def __init__(
        self,
        vector_store: VedicVectorStore,
        llm_interface: GeminiLLMInterface,
        top_k_local: int = TOP_K_RESULTS, # Para busca no VectorStore
        web_scraper: Optional[VedicWebScraper] = None
    ):
        self.vector_store = vector_store
        self.llm_interface = llm_interface
        self.top_k_local = top_k_local
        
        if web_scraper:
            self.web_scraper = web_scraper
        else:
            logger.info("VedicRetriever: Creating default VedicWebScraper instance.")
            self.web_scraper = VedicWebScraper() 
        
        self.site_search_handlers: Dict[str, str] = {
            "purebhakti.com": "scrape_search_results_purebhakti",
            "vedabase.io": "scrape_search_results_vedabase",
            "bhaktivedantavediclibrary.org": "scrape_search_results_bhaktivedantavediclibrary_org",
        }
        logger.info(f"Initialized VedicRetriever with top_k_local={top_k_local}. Web scraper {'configured' if web_scraper else 'default instance'}.")
        logger.debug(f"Trusted websites for hybrid search: {TRUSTED_WEBSITES}")
        logger.debug(f"Site search handlers: {self.site_search_handlers}")

    def parse_filter_dict(self, filter_dict: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not filter_dict: return None
        enhanced_filter = filter_dict.copy()
        if isinstance(enhanced_filter.get('chapter_reference'), str):
            chapter_ref_str = enhanced_filter['chapter_reference']
            chapter_match = re.search(r'(?i)(?:Chapter|Adhyāya|Canto)\s+(\d+)', chapter_ref_str)
            if chapter_match:
                try:
                    enhanced_filter['chapter'] = int(chapter_match.group(1))
                except (ValueError, IndexError):
                     logger.warning(f"Could not parse chapter number from reference: {chapter_ref_str}")
        return enhanced_filter if enhanced_filter else None


    def retrieve_documents_from_local_vs(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        k: Optional[int] = None
    ) -> List[Document]:
        num_results = k if k is not None and k > 0 else self.top_k_local
        effective_filter = self.parse_filter_dict(filter_dict)
        logger.info(f"Retrieving {num_results} documents from local VectorStore for query: '{query}' with filter: {effective_filter}")
        try:
            docs = self.vector_store.similarity_search(query=query, k=num_results, filter=effective_filter)
            logger.info(f"Retrieved {len(docs)} documents from local VectorStore for query '{query}'.")
            return docs
        except Exception as e:
            logger.error(f"Error retrieving documents from local VectorStore for query '{query}': {e}", exc_info=True)
            return []

    def _format_document_context_with_citations(self, docs: Sequence[Document], source_type_label: str = "Local Document") -> List[str]:
        context_parts: List[str] = []
        if not docs: return context_parts
        for i, doc in enumerate(docs):
            metadata = doc.metadata
            source_name = metadata.get("title") or metadata.get("filename") or os.path.basename(metadata.get("source", f"Unknown Source {i+1}"))
            page_num = metadata.get("page")
            verse_ref = metadata.get("verse_reference")
            chapter_ref = metadata.get("chapter_reference")
            url = metadata.get("url") 
            
            citation_details = [f"{source_type_label} '{source_name}'"]
            if url and not source_type_label.lower().startswith("summary from"): citation_details.append(f"URL: {url}")
            if verse_ref: citation_details.append(f"Ref: {verse_ref}")
            elif chapter_ref: citation_details.append(f"Ref: {chapter_ref}")
            if page_num: citation_details.append(f"Page: {page_num}")
            
            citation = f"(Source: {', '.join(citation_details)})"
            content = doc.page_content.strip() if doc.page_content else "[No Content]"
            context_parts.append(f"{content}\n{citation}")
        return context_parts

    def _format_summary_context(self, summaries_data: List[Dict[str, Any]]) -> List[str]:
        context_parts: List[str] = []
        if not summaries_data: return context_parts
        for summary_info in summaries_data:
            title = summary_info.get("title", "Unknown Article")
            url = summary_info.get("url", "N/A")
            summary_text = summary_info.get("summary", "[No summary available]")
            context_parts.append(f"Summary from web article '{title}' (URL: {url}):\n{summary_text}")
        return context_parts

    def _fetch_and_summarize_one_article(self, article_url: str, user_query: str, article_title: Optional[str]=None) -> Optional[Dict[str, Any]]:
        if is_blacklisted_url(article_url) or not respect_robots_txt(article_url):
            logger.info(f"Skipping blacklisted or disallowed URL: {article_url}")
            return None

        logger.info(f"Fetching full text for article: {article_url} for summarization relevant to '{user_query}'")
        article_html = self.web_scraper.fetch_url(article_url) 
        if not article_html:
            logger.warning(f"Failed to fetch HTML for article: {article_url}")
            return None

        parsed_article = self.web_scraper.parse_html(article_html, article_url)
        if not parsed_article.get("success") or not parsed_article.get("text"):
            logger.warning(f"Failed to parse or extract text from article: {article_url}")
            return None

        full_text = parsed_article["text"]
        title_to_use = article_title or parsed_article.get("title", article_url)

        logger.info(f"--- DEBUG: Texto extraído de '{title_to_use}' (URL: {article_url}) ---")
        logger.info(f"Comprimento do texto: {len(full_text)} caracteres")
        # Para ver uma amostra no log (cuidado com logs muito grandes):
        logger.debug(f"Amostra do texto extraído (primeiros 1500 caracteres):\n{full_text[:1500]}")
        logger.debug(f"Amostra do texto extraído (últimos 500 caracteres):\n{full_text[-500:]}")

        # Opcional: Salvar o texto completo em um arquivo para análise fácil
        try:
            import re
            safe_title = re.sub(r'[^\w\-. ]', '_', title_to_use).replace(' ', '_')[:100] # Nome de arquivo seguro
            debug_text_filename = f"debug_extracted_text_{safe_title}.txt"
            with open(debug_text_filename, "w", encoding="utf-8") as f_text:
                f_text.write(full_text)
            logger.info(f"Texto completo extraído salvo em: {debug_text_filename}")
        except Exception as e_save:
            logger.error(f"Não foi possível salvar o arquivo de texto de depuração: {e_save}")

        logger.debug(f"Summarizing article: {title_to_use} (length: {len(full_text)}) for query focus: '{user_query}'")
        if not hasattr(self.llm_interface, 'summarize_text_for_query'):
            logger.error("GeminiLLMInterface does not have 'summarize_text_for_query' method.")
            return {"url": article_url, "title": title_to_use, "summary": "[Summarization unavailable due to missing LLM method]"}

        summary = self.llm_interface.summarize_text_for_query(
            text_to_summarize=full_text,
            query_focus=user_query
        )

        logger.info(f"--- DEBUG: Sumário gerado para '{title_to_use}' ---")
        logger.info(f"Foco da query para sumarização: '{user_query}'")
        logger.info(f"TEXTO DO SUMÁRIO GERADO:\n{summary}") # Este log é o mais importante agora
        
        is_summary_useful = not (summary.startswith("Error:") or \
                             "contained little or no specific information" in summary.lower() or \
                             "does not provide significant details" in summary.lower() or \
                             "unable to find specific information" in summary.lower() or \
                             summary.startswith("[Summarization unavailable")) 

        if is_summary_useful:
            return {"url": article_url, "title": title_to_use, "summary": summary}
        else:
            logger.warning(f"Summarization of {article_url} (Title: {title_to_use}) was not useful or failed: {summary}")
            return {"url": article_url, "title": title_to_use, "summary": "[Summary not relevant or failed]"}


    def _get_web_summaries_for_query(
        self, 
        user_query: str, 
        num_articles_to_process_per_site: int = 3
    ) -> List[Dict[str, Any]]:
        all_summaries_data: List[Dict[str, Any]] = []
        search_term = user_query 

        tasks_for_executor = []

        for site_url_from_config in TRUSTED_WEBSITES: 
            parsed_site_url = urlparse(site_url_from_config)
            domain_key = parsed_site_url.netloc.replace("www.", "")

            handler_method_name = self.site_search_handlers.get(domain_key)
            if not handler_method_name:
                logger.warning(f"No search handler defined for domain: {domain_key}. Skipping web search for this site.")
                continue
            
            if not hasattr(self.web_scraper, handler_method_name):
                logger.error(f"Web scraper method '{handler_method_name}' for domain {domain_key} not found. Skipping.")
                continue
            
            search_handler_func = getattr(self.web_scraper, handler_method_name)
            logger.info(f"Preparing search on '{domain_key}' for term: '{search_term}' using {handler_method_name}")
            try:
                articles_info = search_handler_func(search_term, num_articles_to_process_per_site)
                for info in articles_info:
                    if info.get("url"):
                        tasks_for_executor.append(
                            (self._fetch_and_summarize_one_article, info["url"], user_query, info.get("title"))
                        )
                    else:
                        logger.warning(f"Search result from {domain_key} missing 'url' key: {info.get('title', 'N/A')}")
            except Exception as e:
                logger.error(f"Error during initial search (getting links) on site '{domain_key}': {e}", exc_info=True)

        if tasks_for_executor:
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor: 
                future_to_task_params = {executor.submit(func, *args): (func, args) for func, *args in tasks_for_executor}
                for future in concurrent.futures.as_completed(future_to_task_params):
                    try:
                        summary_data = future.result()
                        if summary_data: 
                            all_summaries_data.append(summary_data)
                    except Exception as e:
                        task_params = future_to_task_params[future]
                        logger.error(f"Error processing article summary future for task {task_params}: {e}", exc_info=True)
        
        useful_summaries = [s for s in all_summaries_data if s.get("summary") and not s["summary"].startswith("[")]
        logger.info(f"Generated {len(useful_summaries)} useful web summaries for query '{user_query}'.")
        return useful_summaries

    def answer_query_hybrid_rag(
        self,
        user_query: str,
        num_web_articles_per_site: int = 1, 
        num_local_docs: int = TOP_K_RESULTS
    ) -> Dict[str, Any]:
        logger.info(f"Starting Hybrid RAG for query: '{user_query}' (Web articles per site: {num_web_articles_per_site}, Local docs: {num_local_docs})")
        final_answer_text = "Could not generate an answer based on the available information." 
        llm_failed = True 
        all_source_details: List[Dict[str, Any]] = []
        
        web_summaries_data = self._get_web_summaries_for_query(user_query, num_web_articles_per_site)
        
        for summary_d in web_summaries_data:
            all_source_details.append({
                "type": "web_summary",
                "title": summary_d.get("title", "N/A"),
                "url": summary_d.get("url", "N/A"),
                "summary_content": summary_d.get("summary", "[No summary content]")
            })

        local_docs = self.retrieve_documents_from_local_vs(user_query, k=num_local_docs)
        logger.info(f"Retrieved {len(local_docs)} local documents for query '{user_query}'.")
        for i, doc_content in enumerate(local_docs): # Enhanced logging
            logger.debug(f"Local Doc {i+1} Metadata: {doc_content.metadata}") 
            logger.debug(f"Local Doc {i+1} Content (first 300 chars): {doc_content.page_content[:300]}")
        
        for doc in local_docs:
            source_file = doc.metadata.get("source", "N/A")
            if os.path.exists(source_file): 
                source_display = os.path.basename(source_file)
            else:
                source_display = source_file

            all_source_details.append({
                "type": "local_document_chunk",
                "source_file": source_display,
                "title": doc.metadata.get("title", "N/A"), 
                "page": doc.metadata.get("page", "N/A"),
                "content_preview": doc.page_content[:150] + "..." 
            })

        context_parts: List[str] = []
        if web_summaries_data:
            useful_web_summaries = [s for s in web_summaries_data if s.get("summary") and not s["summary"].startswith("[")]
            if useful_web_summaries:
                context_parts.extend(self._format_summary_context(useful_web_summaries))
        
        if local_docs:
            context_parts.extend(self._format_document_context_with_citations(local_docs, source_type_label="Excerpt from Local Document"))

        if not context_parts:
            logger.warning(f"No context generated from web or local search for query: '{user_query}'")
            return {
                "answer": "I could not find sufficient information from web searches or local documents to answer your question.",
                "sources": all_source_details, 
                "llm_fallback_used": False 
            }

        final_context = "\n\n---\n\n".join(context_parts)
        logger.info(f"--- DEBUG: CONTEXTO FINAL SENDO ENVIADO PARA O LLM (Query: '{user_query}') ---")

        # Alternativa: Salvar em arquivo para análise mais fácil
        try:
            import re
            safe_query = re.sub(r'[^\w\-. ]', '_', user_query).replace(' ', '_')[:50]
            debug_context_filename = f"debug_final_context_{safe_query}.txt"
            with open(debug_context_filename, "w", encoding="utf-8") as f_ctx:
                f_ctx.write(f"Query: {user_query}\n\n")
                f_ctx.write("System Prompt (início):\n")
                f_ctx.write(system_instruction[:500] + "...\n\n") # system_instruction é definido logo depois
                f_ctx.write("--- CONTEXTO COMBINADO ---\n")
                f_ctx.write(final_context)
            logger.info(f"Contexto final completo salvo em: {debug_context_filename}")
        except Exception as e_save_ctx:
            logger.error(f"Não foi possível salvar o arquivo de contexto final de depuração: {e_save_ctx}")

        logger.debug(f"Hybrid RAG final context (primeiros 500 caracteres para query '{user_query}'):\n{final_context[:100]}...") # Este log você já tinha, é bom.

        # REFINED system_instruction
        system_instruction = f"""You are a Vedic scholar and AI assistant.
Your task is to answer the user's question: '{user_query}'
You MUST base your answer *exclusively* on the following provided context. The context may include summaries from web articles and excerpts from local documents.
Directly synthesize the information from the context to answer the question.
When referencing information from the context:
- For web summaries, mention the article title and its URL.
- For local documents, mention the document title or filename and page/reference if available.
If the provided context does not contain sufficient information to answer the question comprehensively, clearly state that the information is not found in the provided texts or is insufficient.
Do not use any external knowledge. Do not provide generic statements about your capabilities or how you will answer. Focus *only* on answering the question using the given context.
Provide a clear, concise, and well-structured answer based *only* on the information given below.
"""
        try:
            # Enhanced logging for LLM call
            logger.debug(f"Calling LLM Interface. User Query: '{user_query}'. System Prompt (first 200 chars): '{system_instruction[:200]}...'. Context Length: {len(final_context)}")
            
            final_answer_text = self.llm_interface.generate_response(
                prompt=user_query, 
                context=final_context,
                system_prompt=system_instruction
            )
            llm_failed = final_answer_text.startswith("Error:") or "Could not summarize" in final_answer_text 
            logger.info(f"LLM response for query '{user_query}' (first 200 chars): {final_answer_text[:200]}...") # Log beginning of LLM response
        except Exception as e:
            logger.error(f"LLM generation failed for hybrid RAG query '{user_query}': {e}", exc_info=True)
            final_answer_text = "An error occurred while generating the final answer using combined information."
            llm_failed = True
            
        return {
            "answer": final_answer_text,
            "sources": all_source_details,
            "llm_fallback_used": llm_failed
        }

    def answer_query_from_local_rag(self, query: str, filter_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        logger.info(f"Answering query '{query}' using ONLY local VectorStore and filter: {filter_dict}")
        retrieved_docs = self.retrieve_documents_from_local_vs(query, filter_dict)
        if not retrieved_docs:
            return {"answer": "I couldn't find specific information in the local knowledge base.", "sources": [], "documents": [], "llm_fallback_used": False}
        
        context = "\n\n---\n\n".join(self._format_document_context_with_citations(retrieved_docs, source_type_label="Local Document"))
        prompt_template: PromptTemplate = select_prompt_template(query) or VEDIC_QA_PROMPT
        template_str = prompt_template.template
        system_prompt_part = template_str.split("Context:", 1)[0].strip()
        if not system_prompt_part or "{context}" in system_prompt_part or "{question}" in system_prompt_part:
            system_prompt_part = "You are a knowledgeable scholar of Vedic philosophy. Use the provided context to answer the question."

        answer = self.llm_interface.generate_response(prompt=query, context=context, system_prompt=system_prompt_part)
        llm_failed = answer.startswith("Error:")
        
        sources_metadata = []
        for doc in retrieved_docs:
            source_file = doc.metadata.get("source", "N/A")
            if os.path.exists(source_file): source_display = os.path.basename(source_file)
            else: source_display = source_file
            sources_metadata.append({
                "source_file": source_display, 
                "title": doc.metadata.get("title", "N/A"), 
                "page": doc.metadata.get("page", "N/A"),
                "url": doc.metadata.get("url") 
            })
        return {"answer": answer, "sources": sources_metadata, "documents": [], "llm_fallback_used": llm_failed}