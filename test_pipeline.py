# test_pipeline.py
import logging
import os
from typing import List, Dict, Any, Optional

# Configurar imports para acessar seus módulos src
# (Se test_pipeline.py estiver na raiz do projeto claude_vedic_qa)
from src.config import (
    LOG_LEVEL, LOG_FILE, TRUSTED_WEBSITES,
    GEMINI_API_KEY, MODEL_NAME, EMBEDDING_MODEL, DB_DIR
)
from src.utils.logger import setup_logger # Assumindo que você tem essa função
from src.web_scraper.scraper import VedicWebScraper
from src.qa_system.gemini_interface import GeminiLLMInterface
from src.knowledge_base.embeddings import get_huggingface_embeddings
from src.knowledge_base.vector_store import VedicVectorStore
from src.qa_system.retriever import VedicRetriever

# Configurar Logging (adapte conforme sua função setup_logger)
# setup_logger(name="vedic_ai_test", log_file="test_pipeline.log", level="DEBUG", console=True)
# Ou, para um teste rápido:
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(), # Para console
        # logging.FileHandler("test_pipeline.log") # Para arquivo
    ]
)
logger = logging.getLogger(__name__)


# --- Funções de Teste Individuais ---

def test_site_search_function(scraper_instance: VedicWebScraper, site_domain_key: str, query: str, num_articles: int):
    logger.info(f"\n--- Iniciando Teste: Busca no site '{site_domain_key}' por '{query}' ---")
    search_handler_method_name = scraper_instance.retriever.site_search_handlers.get(site_domain_key) # Acessar handlers do retriever
    if not search_handler_method_name:
        logger.error(f"Nenhum handler de busca definido para {site_domain_key} no retriever.")
        return

    if not hasattr(scraper_instance, search_handler_method_name):
        logger.error(f"Método '{search_handler_method_name}' não encontrado no scraper para {site_domain_key}.")
        return

    search_func = getattr(scraper_instance, search_handler_method_name)
    articles_info = search_func(query, num_articles)

    if articles_info:
        logger.info(f"Busca em '{site_domain_key}' por '{query}' encontrou {len(articles_info)} links:")
        for i, info in enumerate(articles_info):
            logger.info(f"  {i+1}. Título: {info.get('title')}, URL: {info.get('url')}")
    else:
        logger.warning(f"Nenhum link de artigo encontrado em '{site_domain_key}' para '{query}'.")
    logger.info(f"--- Fim do Teste: Busca no site '{site_domain_key}' ---")
    return articles_info

def test_fetch_and_parse_article(scraper_instance: VedicWebScraper, article_url: str):
    logger.info(f"\n--- Iniciando Teste: Fetch e Parse do Artigo '{article_url}' ---")
    html = scraper_instance.fetch_url(article_url, bypass_cache=True) # Bypass cache para garantir que estamos testando o parse
    if not html:
        logger.error(f"Falha ao buscar HTML de {article_url}")
        return None

    parsed_data = scraper_instance.parse_html(html, article_url)
    if parsed_data.get("success"):
        logger.info(f"Parse do artigo '{article_url}' bem-sucedido.")
        logger.debug(f"  Título Parseado: {parsed_data.get('title')}")
        logger.debug(f"  Metadados: {parsed_data.get('metadata')}")
        logger.info(f"  Texto (primeiros 200 chars): {parsed_data.get('text', '')[:200]}...")
    else:
        logger.error(f"Falha ao parsear HTML de {article_url}: {parsed_data.get('error')}")
    logger.info(f"--- Fim do Teste: Fetch e Parse do Artigo ---")
    return parsed_data

def test_summarization(retriever_instance: VedicRetriever, article_url: str, query_focus: str, article_title: Optional[str]=None):
    logger.info(f"\n--- Iniciando Teste: Sumarização do Artigo '{article_url}' com foco em '{query_focus}' ---")
    summary_data = retriever_instance._fetch_and_summarize_one_article(article_url, query_focus, article_title)
    if summary_data:
        logger.info("Dados da Sumarização:")
        logger.info(f"  URL: {summary_data.get('url')}")
        logger.info(f"  Título: {summary_data.get('title')}")
        logger.info(f"  Resumo: {summary_data.get('summary')}")
    else:
        logger.error("Não foi possível buscar e/ou sumarizar o artigo.")
    logger.info(f"--- Fim do Teste: Sumarização do Artigo ---")
    return summary_data

def test_get_all_web_summaries_function(retriever_instance: VedicRetriever, user_query: str, num_articles_per_site: int):
    logger.info(f"\n--- Iniciando Teste: Coleta de Todos os Resumos da Web para '{user_query}' ---")
    web_summaries = retriever_instance._get_web_summaries_for_query(user_query, num_articles_per_site)
    if web_summaries:
        logger.info(f"Total de resumos úteis coletados: {len(web_summaries)}")
        for i, summary_d in enumerate(web_summaries):
            logger.info(f"  Resumo {i+1}: Título: {summary_d.get('title')}, URL: {summary_d.get('url')}, Preview: {summary_d.get('summary', '')[:70]}...")
    else:
        logger.warning("Nenhum resumo útil coletado da web.")
    logger.info(f"--- Fim do Teste: Coleta de Todos os Resumos da Web ---")
    return web_summaries

def test_hybrid_rag_flow(retriever_instance: VedicRetriever, user_query: str, num_web_articles: int, num_local_docs: int):
    logger.info(f"\n--- Iniciando Teste: Fluxo Hybrid RAG Completo para '{user_query}' ---")
    response_data = retriever_instance.answer_query_hybrid_rag(
        user_query,
        num_web_articles_per_site=num_web_articles,
        num_local_docs=num_local_docs
    )
    logger.info("--- Resposta Final do Hybrid RAG ---")
    logger.info(f"Pergunta: {user_query}")
    logger.info(f"Resposta Gerada:\n{response_data.get('answer')}")
    logger.info("Fontes Consideradas:")
    if response_data.get("sources"):
        for i, source_info in enumerate(response_data["sources"]):
            logger.info(f"  {i+1}. Tipo: {source_info.get('type')}, Título/Arquivo: {source_info.get('title') or source_info.get('source_file')}, URL: {source_info.get('url', 'N/A')}, Page: {source_info.get('page', 'N/A')}")
    else:
        logger.info("  Nenhuma fonte listada.")
    logger.info(f"LLM Fallback Usado: {response_data.get('llm_fallback_used')}")
    logger.info(f"--- Fim do Teste: Fluxo Hybrid RAG Completo ---")


# --- Ponto de Entrada Principal do Teste ---
if __name__ == "__main__":
    logger.info(">>> INICIANDO SUÍTE DE TESTES PARA VEDIC KNOWLEDGE AI <<<")

    # Instanciar componentes principais uma vez
    try:
        test_scraper = VedicWebScraper() # Usa configs de REQUEST_DELAY, WEB_CACHE_DIR do config.py

        # Para o retriever, precisamos de todos os seus componentes
        test_embedding_func = get_huggingface_embeddings(model_name=EMBEDDING_MODEL)

        # Certifique-se que DB_DIR é um diretório válido, mesmo que o DB esteja vazio para alguns testes
        if not os.path.exists(DB_DIR):
            os.makedirs(DB_DIR)
            logger.info(f"Diretório DB_DIR criado em: {DB_DIR}")
        elif not os.path.isdir(DB_DIR):
            logger.error(f"DB_DIR ({DB_DIR}) existe mas não é um diretório! Testes do VectorStore podem falhar.")
            # Poderia sair aqui ou deixar os testes do VectorStore falharem e reportarem.

        test_vector_store = VedicVectorStore(embedding_function=test_embedding_func, persist_directory=DB_DIR)
        test_llm_interface = GeminiLLMInterface(api_key=GEMINI_API_KEY, model_name=MODEL_NAME)

        # Passar a instância do scraper para o retriever
        test_retriever = VedicRetriever(
            vector_store=test_vector_store,
            llm_interface=test_llm_interface,
            web_scraper=test_scraper # Importante!
        )
        # Adicionar o retriever ao scraper para que test_site_search_function possa acessar site_search_handlers
        test_scraper.retriever = test_retriever


        # --- EXECUTAR TESTES INDIVIDUAIS (descomente conforme necessário) ---

        # 1. Testar Busca em Sites Específicos
        # test_site_search_function(test_scraper, "purebhakti.com", "ekadasi", 2)
        # test_site_search_function(test_scraper, "vedabase.io", "consciousness", 2)
        # Adicione chamadas para outros sites quando implementar seus handlers de busca

        # 2. Testar Fetch e Parse de um Artigo Específico
        # (Pegue uma URL de um resultado da busca acima ou uma conhecida)
        # test_fetch_and_parse_article(test_scraper, "https://www.purebhakti.com/resources/vaisnava-calendar/ekadasi/196-yogini-ekadasi")
        # test_fetch_and_parse_article(test_scraper, "https://vedabase.io/en/library/sb/1/1/1/") # Teste com vedabase e custom assembly

        # 3. Testar Sumarização de um Artigo Específico
        # test_summarization(test_retriever, "https://www.purebhakti.com/resources/vaisnava-calendar/ekadasi/196-yogini-ekadasi", "significado de Yogini Ekadasi", "Yogini Ekadasi Article")
        # test_summarization(test_retriever, "https://vedabase.io/en/library/sb/1/1/1/", "qual a pergunta dos sábios", "SB 1.1.1")

        # 4. Testar Coleta de Resumos da Web para uma Pergunta
        # (Irá buscar e resumir de todos os sites configurados em TRUSTED_WEBSITES e site_search_handlers)
        # test_get_all_web_summaries_function(test_retriever, "what is bhakti?", num_articles_per_site=1)

        # 5. Testar o Fluxo Hybrid RAG Completo
        # Este é o teste mais abrangente. Certifique-se que seu VectorStore local (DB_DIR) está
        # populado se você quiser ver resultados da busca local também.
        # Se estiver vazio, a parte de "local_docs" do RAG híbrido não trará resultados, o que é esperado.
        logger.info("IMPORTANTE: Para o teste Hybrid RAG, o VectorStore local será usado. Popule-o se quiser resultados locais.")
        test_hybrid_rag_flow(test_retriever, user_query="What is the supreme goal of life according to vedic scriptures?", num_web_articles=1, num_local_docs=2)
        # test_hybrid_rag_flow(test_retriever, user_query="Tell me about Lord Chaitanya's teachings.", num_web_articles=1, num_local_docs=2)


        logger.info(">>> SUÍTE DE TESTES CONCLUÍDA <<<")

    except Exception as e:
        logger.error(f"Erro catastrófico durante a configuração da suíte de testes: {e}", exc_info=True)
        logger.info(">>> SUÍTE DE TESTES INTERROMPIDA DEVIDO A ERRO <<<")