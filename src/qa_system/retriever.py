# src/qa_system/retriever.py

import logging
import re
import os
from typing import List, Dict, Any, Optional, Sequence, Callable # Removido Tuple, Union não usados diretamente
import concurrent.futures
from urllib.parse import urlparse

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from ..knowledge_base.vector_store import VedicVectorStore
from ..knowledge_base.prompt_templates import select_prompt_template, VEDIC_QA_PROMPT
from .gemini_interface import GeminiLLMInterface
from ..config import TOP_K_RESULTS, TRUSTED_WEBSITES # TRUSTED_WEBSITES é usado aqui
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
            self.web_scraper = VedicWebScraper() # Garante que sempre haja um scraper
        
        # Mapeamento de domínios para seus respectivos métodos de busca no scraper
        # Chave: domínio (ex: "purebhakti.com"), Valor: nome do método no web_scraper
        self.site_search_handlers: Dict[str, str] = {
            "purebhakti.com": "scrape_search_results_purebhakti",
            "vedabase.io": "scrape_search_results_vedabase",
            "bhaktivedantavediclibrary.org": "scrape_search_results_bhaktivedantavediclibrary_org",
            # Adicione outros sites e seus métodos de busca aqui
        }
        logger.info(f"Initialized VedicRetriever with top_k_local={top_k_local}. Web scraper {'configured' if web_scraper else 'default instance'}.")
        logger.debug(f"Trusted websites for hybrid search: {TRUSTED_WEBSITES}")
        logger.debug(f"Site search handlers: {self.site_search_handlers}")

    def parse_filter_dict(self, filter_dict: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not filter_dict: return None
        # ... (implementação do parse_filter_dict como antes) ...
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
            logger.info(f"Retrieved {len(docs)} documents from local VectorStore.")
            return docs
        except Exception as e:
            logger.error(f"Error retrieving documents from local VectorStore: {e}", exc_info=True)
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
            url = metadata.get("url") # Pode vir de metadados de documentos web que foram indexados
            
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
            # Adicionar identificador para o LLM saber que é um resumo
            context_parts.append(f"Summary from web article '{title}' (URL: {url}):\n{summary_text}")
        return context_parts

    def _fetch_and_summarize_one_article(self, article_url: str, user_query: str, article_title: Optional[str]=None) -> Optional[Dict[str, Any]]:
        """Pega o texto completo de UM artigo e o resume, reutilizando o título se já conhecido."""
        if is_blacklisted_url(article_url) or not respect_robots_txt(article_url):
            logger.info(f"Skipping blacklisted or disallowed URL: {article_url}")
            return None

        logger.info(f"Fetching full text for article: {article_url} for summarization relevant to '{user_query}'")
        article_html = self.web_scraper.fetch_url(article_url) # fetch_url lida com cache
        if not article_html:
            logger.warning(f"Failed to fetch HTML for article: {article_url}")
            return None

        parsed_article = self.web_scraper.parse_html(article_html, article_url)
        if not parsed_article.get("success") or not parsed_article.get("text"):
            logger.warning(f"Failed to parse or extract text from article: {article_url}")
            return None

        full_text = parsed_article["text"]
        # Usar o título da busca se disponível, senão o título parseado, senão a URL
        title_to_use = article_title or parsed_article.get("title", article_url)


        logger.debug(f"Summarizing article: {title_to_use} (length: {len(full_text)}) for query focus: '{user_query}'")
        if not hasattr(self.llm_interface, 'summarize_text_for_query'):
            logger.error("GeminiLLMInterface does not have 'summarize_text_for_query' method.")
            return {"url": article_url, "title": title_to_use, "summary": "[Summarization unavailable due to missing LLM method]"}

        summary = self.llm_interface.summarize_text_for_query(
            text_to_summarize=full_text,
            query_focus=user_query
        )
        
        is_summary_useful = not (summary.startswith("Error:") or \
                             "contained little or no specific information" in summary.lower() or \
                             "does not provide significant details" in summary.lower() or \
                             "unable to find specific information" in summary.lower() or \
                             summary.startswith("[Summarization unavailable")) # Checar também o nosso fallback

        if is_summary_useful:
            return {"url": article_url, "title": title_to_use, "summary": summary}
        else:
            logger.warning(f"Summarization of {article_url} (Title: {title_to_use}) was not useful or failed: {summary}")
            # Retornar mesmo que não útil, para que a fonte seja listada, mas com um resumo indicando o problema
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
            # A tarefa agora é apenas obter os links e títulos dos resultados da busca
            try:
                # search_handler_func deve retornar List[Dict[str,str]] com {'url': ..., 'title': ...}
                articles_info = search_handler_func(search_term, num_articles_to_process_per_site)
                for info in articles_info:
                    if info.get("url"):
                        # Adicionar a tarefa de fetch E sumarização para o executor
                        tasks_for_executor.append(
                            (self._fetch_and_summarize_one_article, info["url"], user_query, info.get("title"))
                        )
                    else:
                        logger.warning(f"Search result from {domain_key} missing 'url' key: {info.get('title', 'N/A')}")
            except Exception as e:
                logger.error(f"Error during initial search (getting links) on site '{domain_key}': {e}", exc_info=True)

        # Agora processar todas as tarefas de fetch e sumarização em paralelo
        if tasks_for_executor:
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor: # Limitar workers para sumarização
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
        num_web_articles_per_site: int = 1, # Pegar X melhores artigos por site para sumarizar
        num_local_docs: int = 5
    ) -> Dict[str, Any]:
        logger.info(f"Starting Hybrid RAG for query: '{user_query}' (Web articles per site: {num_web_articles_per_site}, Local docs: {num_local_docs})")
        final_answer_text = "Could not generate an answer based on the available information." # Default mais informativo
        llm_failed = True # Assumir falha até sucesso
        all_source_details: List[Dict[str, Any]] = []
        
        web_summaries_data = self._get_web_summaries_for_query(user_query, num_web_articles_per_site)
        
        for summary_d in web_summaries_data:
            all_source_details.append({
                "type": "web_summary",
                "title": summary_d.get("title", "N/A"),
                "url": summary_d.get("url", "N/A"),
                "summary_content": summary_d.get("summary", "[No summary content]") # Incluir o resumo aqui
            })

        local_docs = self.retrieve_documents_from_local_vs(user_query, k=num_local_docs)
        
        for doc in local_docs:
            source_file = doc.metadata.get("source", "N/A")
            # Se a fonte for um caminho de arquivo, pegar apenas o nome do arquivo
            if os.path.exists(source_file): # Checa se é um caminho válido antes de tentar basename
                source_display = os.path.basename(source_file)
            else:
                source_display = source_file

            all_source_details.append({
                "type": "local_document_chunk",
                "source_file": source_display,
                "title": doc.metadata.get("title", "N/A"), # Título do livro/documento
                "page": doc.metadata.get("page", "N/A"),
                "content_preview": doc.page_content[:150] + "..." # Preview um pouco maior
            })

        context_parts: List[str] = []
        if web_summaries_data:
            # Filtrar resumos que não foram úteis antes de adicionar ao contexto
            useful_web_summaries = [s for s in web_summaries_data if s.get("summary") and not s["summary"].startswith("[")]
            if useful_web_summaries:
                context_parts.extend(self._format_summary_context(useful_web_summaries))
        
        if local_docs:
            context_parts.extend(self._format_document_context_with_citations(local_docs, source_type_label="Excerpt from Local Document"))

        if not context_parts:
            logger.warning(f"No context generated from web or local search for query: '{user_query}'")
            return {
                "answer": "I could not find sufficient information from web searches or local documents to answer your question.",
                "sources": all_source_details, # Ainda retorna as fontes tentadas
                "llm_fallback_used": False # Não houve falha do LLM, mas sim falta de contexto
            }

        final_context = "\n\n---\n\n".join(context_parts)
        logger.debug(f"Hybrid RAG final context (first 500 chars):\n{final_context[:500]}...")

        system_instruction = f"""You are a Vedic scholar and AI assistant. The user asked: '{user_query}'.
Base your answer *exclusively* on the following context, which includes summaries from web articles and excerpts from local documents.
When using information, clearly cite the source. For web summaries, mention the article title and URL. For local documents, mention the document title or filename and page/reference if available.
If the provided context is insufficient to answer the question thoroughly, state that you can only provide a partial answer based on the information or that the information is not present in the context.
Do not use any external knowledge. Your entire response must be derived from the provided text.
Strive for a comprehensive and well-structured answer if the context allows.
"""
        try:
            final_answer_text = self.llm_interface.generate_response(
                prompt=user_query, # A pergunta original do usuário
                context=final_context,
                system_prompt=system_instruction
            )
            llm_failed = final_answer_text.startswith("Error:") or "Could not summarize" in final_answer_text # Checagem mais ampla
        except Exception as e:
            logger.error(f"LLM generation failed for hybrid RAG: {e}", exc_info=True)
            final_answer_text = "An error occurred while generating the final answer using combined information."
            llm_failed = True
            
        return {
            "answer": final_answer_text,
            "sources": all_source_details,
            "llm_fallback_used": llm_failed
        }

    # --- Outros métodos como answer_query_from_local_rag, answer_query_from_single_site_summary, get_documents_by_chapter, get_chapter_summary ---
    # Mantenha-os ou ajuste-os conforme necessário.
    # answer_query_from_local_rag foi mantido como exemplo de uma estratégia alternativa.
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
                "url": doc.metadata.get("url") # Se a fonte local também for um URL
            })
        return {"answer": answer, "sources": sources_metadata, "documents": [], "llm_fallback_used": llm_failed} # Não retornar Documentos aqui para simplificar