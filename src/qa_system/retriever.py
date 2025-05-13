# src/qa_system/retriever.py

import logging
import re
import os
from typing import List, Dict, Any, Optional, Sequence, Tuple # Adicionado Tuple
import concurrent.futures
from urllib.parse import urlparse, quote_plus # Adicionado quote_plus

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

# Importações relativas para componentes internos
from ..knowledge_base.vector_store import VedicVectorStore
from ..knowledge_base.prompt_templates import select_prompt_template, VEDIC_QA_PROMPT
from .gemini_interface import GeminiLLMInterface
from ..config import TOP_K_RESULTS, TRUSTED_WEBSITES
from ..web_scraper.scraper import VedicWebScraper # Scraper base (requests)
from ..web_scraper.dynamic_scraper import DynamicVedicScraper # Scraper dinâmico (Selenium)
from ..web_scraper.ethics import is_blacklisted_url, respect_robots_txt
# from selenium.webdriver.common.by import By # Não é mais necessário aqui, pois é usado dentro dos scrapers

logger = logging.getLogger(__name__)

class VedicRetriever:
    def __init__(
        self,
        vector_store: VedicVectorStore,
        llm_interface: GeminiLLMInterface,
        top_k_local: int = TOP_K_RESULTS,
    ):
        self.vector_store = vector_store
        self.llm_interface = llm_interface
        self.top_k_local = top_k_local
        
        logger.info("VedicRetriever: Criando instância do VedicWebScraper (estático).")
        self.static_web_scraper = VedicWebScraper()

        # Instância para o scraper dinâmico, inicializada quando necessário (lazy loading)
        self._dynamic_web_scraper_instance: Optional[DynamicVedicScraper] = None
        
        # Mapeia o nome do método de scraping de resultados de busca.
        # Estes métodos serão chamados na instância de scraper apropriada (estática ou dinâmica).
        self.site_search_handlers: Dict[str, str] = {
            "purebhakti.com": "scrape_search_results_purebhakti", # Deve usar Selenium internamente via DynamicVedicScraper
            "vedabase.io": "scrape_search_results_vedabase",       # Deve usar Selenium internamente via DynamicVedicScraper
            "bhaktivedantavediclibrary.org": "scrape_search_results_bhaktivedantavediclibrary_org", # Atualmente estático
        }
        # Define quais sites sabidamente requerem scraper dinâmico para sua funcionalidade de BUSCA
        self.sites_requiring_dynamic_search = ["purebhakti.com", "vedabase.io"]
        # Define quais sites sabidamente requerem scraper dinâmico para buscar o CONTEÚDO DE ARTIGOS individuais
        self.sites_requiring_dynamic_article_fetch = ["purebhakti.com", "vedabase.io"]

        logger.info(f"Initialized VedicRetriever with top_k_local={top_k_local}.")
        logger.debug(f"Trusted websites for hybrid search: {TRUSTED_WEBSITES}")
        logger.debug(f"Site search handlers: {self.site_search_handlers}")
        logger.debug(f"Sites requiring dynamic search: {self.sites_requiring_dynamic_search}")
        logger.debug(f"Sites requiring dynamic article fetch: {self.sites_requiring_dynamic_article_fetch}")

    def _get_dynamic_scraper(self) -> DynamicVedicScraper:
        """Inicializa e retorna a instância do DynamicVedicScraper de forma preguiçosa."""
        if self._dynamic_web_scraper_instance is None:
            logger.info("VedicRetriever: Inicializando instância do DynamicVedicScraper.")
            cache_dir = self.static_web_scraper.cache_manager.cache_dir
            request_delay = self.static_web_scraper.request_delay # Dynamic scraper também tem seu próprio request_delay
            self._dynamic_web_scraper_instance = DynamicVedicScraper(cache_dir=cache_dir, request_delay=request_delay)
            # O driver do Selenium é inicializado dentro do DynamicVedicScraper quando necessário (ex: no primeiro fetch_url dele)
        return self._dynamic_web_scraper_instance

    def _shutdown_dynamic_scraper(self):
        """Fecha o driver do Selenium se a instância do DynamicVedicScraper foi utilizada."""
        if self._dynamic_web_scraper_instance:
            logger.info("VedicRetriever: Solicitando fechamento da instância do DynamicVedicScraper.")
            self._dynamic_web_scraper_instance._close_driver()
            self._dynamic_web_scraper_instance = None

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
        
        scraper_for_article_fetch = self.static_web_scraper 
        article_domain = urlparse(article_url).netloc.replace("www.", "")
        if article_domain in self.sites_requiring_dynamic_article_fetch: # Usa a lista definida no __init__
            scraper_for_article_fetch = self._get_dynamic_scraper()
            logger.info(f"Usando DYNAMIC scraper para buscar conteúdo do artigo de {article_domain} para URL: {article_url}")
        else:
            logger.info(f"Usando STATIC scraper para buscar conteúdo do artigo de {article_domain} para URL: {article_url}")

        # O método fetch_url será chamado na instância de scraper apropriada
        article_html = scraper_for_article_fetch.fetch_url(article_url) 

        if not article_html:
            logger.warning(f"Falha ao buscar HTML para o artigo: {article_url} usando {type(scraper_for_article_fetch).__name__}")
            return None

        # Para parse_html, você PODE usar o static_web_scraper, pois ele opera sobre a string HTML
        # A menos que parse_html também precise de estado do driver, o que não é comum.
        parsed_article = self.static_web_scraper.parse_html(article_html, article_url)
        if not parsed_article.get("success") or not parsed_article.get("text"):
            logger.warning(f"Failed to parse or extract text from article: {article_url}")
            return None

        full_text = parsed_article["text"]
        title_to_use = article_title or parsed_article.get("title", article_url)
        
        try:
            safe_title = re.sub(r'[^\w\-. ]', '_', title_to_use).replace(' ', '_')[:100] 
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

    def _extract_search_term_from_query(self, user_query: str) -> str:
        """
        Extrai um termo de busca mais curto de uma user_query mais longa,
        se a query parecer ser um pedido de definição. Caso contrário, retorna a query original.
        """
        extracted_term = None
        # Padrões da mais específica para a mais geral
        meaning_query_patterns = [
            r"what is the meaning of the term\s+['\"]?([\w\s-]+)['\"]?",
            r"what is the meaning of\s+['\"]?([\w\s-]+)['\"]?",
            r"define the term\s+['\"]?([\w\s-]+)['\"]?",
            r"define\s+['\"]?([\w\s-]+)['\"]?",
        ]

        for pattern in meaning_query_patterns:
            match = re.search(pattern, user_query, re.IGNORECASE)
            if match:
                # Pega o último grupo capturado, que deve ser o termo
                term_candidate = match.group(match.lastindex).strip()
                term_candidate = term_candidate.strip("'\"") # Remove aspas
                # Limita o número de palavras para evitar usar frases longas como termo
                if 0 < len(term_candidate.split()) <= 3:
                    extracted_term = term_candidate
                    logger.info(f"Query sobre significado de termo. Query original: '{user_query}', Termo extraído para busca nos sites: '{extracted_term}'")
                    return extracted_term
        
        logger.info(f"Nenhum termo específico extraído. Usando query original para busca nos sites: '{user_query}'")
        return user_query # Retorna a query original se nenhum termo for extraído


    def _get_web_summaries_for_query(
        self,
        user_query: str,
        num_articles_to_process_per_site: int = 3
    ) -> List[Dict[str, Any]]:
        all_summaries_data: List[Dict[str, Any]] = []
        
        search_term_for_sites = self._extract_search_term_from_query(user_query)

        tasks_for_executor = []

        for site_url_from_config in TRUSTED_WEBSITES:
            parsed_site_url = urlparse(site_url_from_config)
            domain_key = parsed_site_url.netloc.replace("www.", "")

            handler_method_name = self.site_search_handlers.get(domain_key)
            if not handler_method_name:
                logger.warning(f"Nenhum handler de busca definido para o domínio: {domain_key}. Pulando.")
                continue

            scraper_instance_to_use = self.static_web_scraper # Padrão
            if domain_key in self.sites_requiring_dynamic_search: # Usa a lista definida no __init__
                scraper_instance_to_use = self._get_dynamic_scraper()
                logger.info(f"Usando scraper DINÂMICO para busca de resultados em {domain_key}")
            else:
                logger.info(f"Usando scraper ESTÁTICO para busca de resultados em {domain_key}")

            if not hasattr(scraper_instance_to_use, handler_method_name): # Verifica na instância correta
                logger.error(f"Método '{handler_method_name}' não encontrado na instância de {type(scraper_instance_to_use).__name__} para {domain_key}. Pulando.")
                continue

            search_handler_func = getattr(scraper_instance_to_use, handler_method_name)

            logger.info(f"Preparando busca em '{domain_key}' para o termo: '{search_term_for_sites}' usando {handler_method_name} (scraper: {type(scraper_instance_to_use).__name__})")
            try:
                articles_info = search_handler_func(search_term_for_sites, num_articles_to_process_per_site)

                if articles_info is None:
                    logger.warning(f"Handler de busca para '{domain_key}' retornou None para o termo '{search_term_for_sites}'. Esperava uma lista, tratando como sem resultados.")
                    articles_info = [] 

                for info in articles_info:
                    if info.get("url"):
                        tasks_for_executor.append(
                            (self._fetch_and_summarize_one_article, info["url"], user_query, info.get("title"))
                        )
                    else:
                        logger.warning(f"Resultado da busca de {domain_key} sem chave 'url': {info.get('title', 'N/A')}")
            except Exception as e:
                logger.error(f"Erro durante busca inicial (obtenção de links) no site '{domain_key}': {e}", exc_info=True)

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
                        logger.error(f"Erro processando futuro de sumário de artigo para tarefa {task_params}: {e}", exc_info=True)
        
        useful_summaries = [s for s in all_summaries_data if s.get("summary") and not s["summary"].startswith("[")]
        logger.info(f"Gerados {len(useful_summaries)} sumários úteis da web para a query '{user_query}'.")
        return useful_summaries

    def answer_query_hybrid_rag(
        self,
        user_query: str,
        num_web_articles_per_site: int = 2, 
        num_local_docs: int = TOP_K_RESULTS
    ) -> Dict[str, Any]:
        logger.info(f"Starting Hybrid RAG for query: '{user_query}' (Web articles per site: {num_web_articles_per_site}, Local docs: {num_local_docs})")
        
        web_summaries_data = []
        try:
            web_summaries_data = self._get_web_summaries_for_query(user_query, num_web_articles_per_site)
        finally:
            # Garante que o driver do Selenium seja fechado após a busca na web, se foi usado.
            self._shutdown_dynamic_scraper()

        all_source_details: List[Dict[str, Any]] = []
        for summary_d in web_summaries_data:
            all_source_details.append({
                "type": "web_summary",
                "title": summary_d.get("title", "N/A"),
                "url": summary_d.get("url", "N/A"),
                "summary_content": summary_d.get("summary", "[No summary content]")
            })

        local_docs = self.retrieve_documents_from_local_vs(user_query, k=num_local_docs)
        logger.info(f"Retrieved {len(local_docs)} local documents for query '{user_query}'.")
        for i, doc_content in enumerate(local_docs):
            logger.debug(f"Local Doc {i+1} Metadata: {doc_content.metadata}") 
            logger.debug(f"Local Doc {i+1} Content (first 300 chars): {doc_content.page_content[:300]}")
        
        for doc in local_docs:
            source_file = doc.metadata.get("source", "N/A")
            source_display = os.path.basename(source_file) if os.path.exists(source_file) else source_file
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

        rag_answer_text = ""
        llm_fallback_used = False
        rag_sources = all_source_details 

        if not context_parts:
            logger.warning(f"No context generated from web or local search for query: '{user_query}'. Attempting direct LLM query.")
            rag_answer_text = "No information found in RAG context." 
        else:
            final_context = "\n\n---\n\n".join(context_parts)
            system_instruction_rag = f"""You are a Vedic scholar and AI assistant.
                                        Your task is to answer the user's question: '{user_query}'
                                        You MUST base your answer *exclusively* on the following provided context. The context may include summaries from web articles and excerpts from local documents.
                                        Directly synthesize the information from the context to answer the question.
                                        When referencing information from the context:
                                        - For web summaries, mention the article title and its URL.
                                        - For local documents, mention the document title or filename and page/reference if available.
                                        If the provided context does not contain sufficient information to answer the question comprehensively, CLEARLY STATE THIS (e.g., "Based on the provided texts, I cannot answer...", or "The documents do not contain specific information about...").
                                        Do not use any external knowledge. Do not provide generic statements. Focus *only* on answering the question using the given context.
                                        Provide a clear, concise, and well-structured answer based *only* on the information given below.
                                        If you cannot answer, explicitly say so."""
            
            try:
                logger.debug(f"Calling LLM Interface (RAG attempt). User Query: '{user_query}'. Context Length: {len(final_context)}")
                rag_answer_text = self.llm_interface.generate_response(
                    prompt=user_query, 
                    context=final_context,
                    system_prompt=system_instruction_rag
                )
                logger.info(f"RAG LLM response for query '{user_query}' (first 200 chars): {rag_answer_text[:200]}")
            except Exception as e:
                logger.error(f"LLM generation failed for hybrid RAG query '{user_query}': {e}", exc_info=True)
                rag_answer_text = "An error occurred while generating the answer using RAG."

        failure_indicators = [
            "i am sorry, but", "i cannot answer", "does not contain information",
            "no information found", "unable to find specific information",
            "do not contain sufficient information", "documents do not contain specific information",
            "no information found in rag context"
        ]
        rag_failed_to_answer = any(indicator in rag_answer_text.lower() for indicator in failure_indicators) or \
                               rag_answer_text.startswith("Error:") or \
                               not rag_answer_text.strip()

        final_answer_to_user = rag_answer_text

        if rag_failed_to_answer:
            logger.warning(f"RAG system could not answer or provided an insufficient answer for '{user_query}'. Attempting direct LLM query without RAG context.")
            system_instruction_direct = f"""You are a knowledgeable Gaudiya vaisnava scholar and AI assistant specialized in the Bhaktivinoda Thakur parivara teachings. 
                                            Answer the following question to the best of your ability based on your general knowledge.
                                            User's question: '{user_query}'
                                            Provide a comprehensive and informative answer. If you do not know the answer, simply state that you do not have that information."""
            try:
                direct_llm_answer = self.llm_interface.generate_response(
                    prompt=user_query,
                    context=None,
                    system_prompt=system_instruction_direct
                )
                logger.info(f"Direct LLM response for query '{user_query}' (first 200 chars): {direct_llm_answer[:200]}")
                final_answer_to_user = f"[Answer from general knowledge as RAG context was insufficient]:\n{direct_llm_answer}"
                llm_fallback_used = True
                rag_sources = [] 
            except Exception as e:
                logger.error(f"Direct LLM query also failed for query '{user_query}': {e}", exc_info=True)
                # final_answer_to_user remains rag_answer_text (which indicated failure)

        return {
            "answer": final_answer_to_user,
            "sources": rag_sources,
            "llm_fallback_used": llm_fallback_used 
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
            source_display = os.path.basename(source_file) if os.path.exists(source_file) else source_file
            sources_metadata.append({
                "source_file": source_display, 
                "title": doc.metadata.get("title", "N/A"), 
                "page": doc.metadata.get("page", "N/A"),
                "url": doc.metadata.get("url") 
            })
        return {"answer": answer, "sources": sources_metadata, "documents": [], "llm_fallback_used": llm_failed}

    def _get_dynamic_scraper(self) -> DynamicVedicScraper:
        """Inicializa e retorna a instância do DynamicVedicScraper de forma preguiçosa."""
        if self._dynamic_web_scraper_instance is None:
            logger.info("VedicRetriever: Inicializando instância do DynamicVedicScraper.")
            # Usa cache_dir e request_delay do static_scraper para consistência
            cache_dir = self.static_web_scraper.cache_manager.cache_dir 
            request_delay = self.static_web_scraper.request_delay 
            self._dynamic_web_scraper_instance = DynamicVedicScraper(cache_dir=cache_dir, request_delay=request_delay)
        return self._dynamic_web_scraper_instance

    def _shutdown_dynamic_scraper(self):
        """Fecha o driver do Selenium se a instância do DynamicVedicScraper foi utilizada."""
        if self._dynamic_web_scraper_instance:
            logger.info("VedicRetriever: Solicitando fechamento da instância do DynamicVedicScraper.")
            self._dynamic_web_scraper_instance._close_driver()
            self._dynamic_web_scraper_instance = None