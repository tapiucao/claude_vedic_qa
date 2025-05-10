# src/web_scraper/scraper.py
import logging
import requests
import time
from typing import Dict, List, Any, Optional, Sequence, Tuple
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, quote_plus # Adicionado quote_plus

from langchain.docstore.document import Document

from ..config import (
    REQUEST_DELAY, WEB_CACHE_DIR,
    SITE_SPECIFIC_SCRAPING_CONFIG # Importar a nova config
)
from ..document_processor.text_splitter import VedicTextSplitter
from ..document_processor.sanskrit_processor import SanskritProcessor
from .cache_manager import WebCacheManager
from .ethics import respect_robots_txt, is_scraping_allowed, is_blacklisted_url, EthicalScraper


logger = logging.getLogger(__name__)

class VedicWebScraper:
    def __init__(self, request_delay: int = REQUEST_DELAY, cache_dir: str = WEB_CACHE_DIR):
        self.request_delay = request_delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36 VedicKnowledgeBot/1.0 (+http://example.com/botinfo)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        })
        self.text_splitter = VedicTextSplitter()
        self.sanskrit_processor = SanskritProcessor()
        self.cache_manager = WebCacheManager(cache_dir=cache_dir)
        self.ethical_helper = EthicalScraper(base_delay=self.request_delay) # Re-adicionado se necessário
        logger.info(f"Initialized web scraper with request delay: {self.request_delay}s")

    def _respect_rate_limit(self): # Adicionado de volta para uso interno se não usar ethical_helper em todo lugar
        time.sleep(self.request_delay)

    def fetch_url(self, url: str, bypass_cache: bool = False, method: str = "GET", data: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Fetch content from a URL with caching, rate limiting, and ethical checks. Supports GET and POST."""
        effective_url = url
        if method.upper() == "POST" and data:
            # Para POST, a chave de cache pode incluir uma representação dos dados para distinguir requisições
            # Aqui, simplificamos usando a URL base para cache, mas o conteúdo pode variar.
            # Idealmente, para POST, a chave de cache deveria ser mais sofisticada ou o cache desabilitado para buscas.
            pass # A chave de cache será a URL base

        if not is_scraping_allowed(effective_url): # Checagem ética inicial
            logger.warning(f"Scraping disallowed by initial check for {effective_url}")
            return None

        if not bypass_cache:
            cached_content = self.cache_manager.get_cached_content(effective_url)
            if cached_content:
                logger.info(f"Using cached content for {effective_url}")
                return cached_content
        
        # Usar ethical_helper para o delay baseado no domínio
        self.ethical_helper.wait_for_domain(effective_url)

        try:
            if method.upper() == "GET":
                response = self.session.get(effective_url, timeout=30)
            elif method.upper() == "POST":
                response = self.session.post(effective_url, data=data, timeout=30)
            else:
                logger.error(f"Unsupported HTTP method: {method}")
                return None
                
            response.raise_for_status()
            response.encoding = response.apparent_encoding or 'utf-8'
            html_content = response.text
            
            # URL final após redirecionamentos (importante para POST)
            final_url = response.url 

            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type:
                logger.warning(f"URL {final_url} did not return HTML (Content-Type: {content_type}). Skipping parse.")
                return None

            if not is_scraping_allowed(final_url, html_content): # Checagem ética no conteúdo
                 logger.warning(f"Scraping disallowed by content check for {final_url}")
                 return None

            # Cachear usando a URL final, especialmente importante para POSTs que redirecionam
            self.cache_manager.cache_content(final_url if final_url != effective_url else effective_url, html_content)
            logger.info(f"Successfully fetched and cached URL: {final_url}")
            return html_content # Retornar o HTML, a URL final pode ser útil para quem chama

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching URL {effective_url} ({method}): {e}")
            return None
        except Exception as e:
             logger.error(f"Unexpected error during fetch for {effective_url} ({method}): {e}", exc_info=True)
             return None

    def parse_html(self, html: str, url: str) -> Dict[str, Any]:
        if not html:
            return {"success": False, "error": "No HTML content provided"}

        parsed_url_obj = urlparse(url)
        domain = parsed_url_obj.netloc.replace("www.", "") # Normalizar domínio para lookup
        site_config = SITE_SPECIFIC_SCRAPING_CONFIG.get(domain, {})

        logger.debug(f"Parsing HTML for {url} (domain: {domain}) using site config: {bool(site_config)}")

        try:
            soup = BeautifulSoup(html, 'lxml')
            title = "No title found"
            title_selector = site_config.get("title_selector")
            if title_selector:
                title_tag = soup.select_one(title_selector)
                if title_tag: title = title_tag.get_text(strip=True)
            if title == "No title found":
                html_title_tag = soup.find('title')
                if html_title_tag and html_title_tag.string: title = html_title_tag.string.strip()

            main_content_element = None
            content_selectors = site_config.get("content_selectors", ['article', 'main', '.content', '#content', '.main', '#main', 'div[role="main"]'])
            for selector in content_selectors:
                main_content_element = soup.select_one(selector)
                if main_content_element:
                    logger.debug(f"Found main content for {url} using selector: '{selector}'")
                    break
            
            target_clean_area = main_content_element if main_content_element else soup
            
            elements_to_remove_selectors = site_config.get("elements_to_remove_selectors", [])
            generic_elements_to_remove = ['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'button', 'iframe', 'noscript', '.noprint', '.advertisement', '.ad', '.popup', '.cookie-consent', '.related-links', '.sidebar', '.social-share', 'figure > figcaption']
            
            # Combinar seletores, garantindo que os genéricos sejam aplicados se os específicos não existirem, ou sempre
            combined_remove_selectors = list(set(elements_to_remove_selectors + generic_elements_to_remove))

            if target_clean_area:
                for selector in combined_remove_selectors:
                    try:
                        for element in target_clean_area.select(selector): element.decompose()
                    except Exception as decomp_err:
                        logger.warning(f"Error decomposing element with selector '{selector}' for {url}: {decomp_err}")
            
            text = ""
            if site_config.get("custom_content_assembly") and main_content_element:
                logger.info(f"Attempting custom content assembly for {domain}")
                parts = []
                verse_sel = site_config.get("verse_selector")
                syn_sel = site_config.get("synonyms_selector")
                trans_sel = site_config.get("translation_selector")
                purport_sel = site_config.get("purport_selector")

                if verse_sel and main_content_element.select_one(verse_sel): parts.append("Original Verse:\n" + main_content_element.select_one(verse_sel).get_text(separator="\n", strip=True))
                if syn_sel and main_content_element.select_one(syn_sel): parts.append("\nSynonyms:\n" + main_content_element.select_one(syn_sel).get_text(separator="\n", strip=True))
                if trans_sel and main_content_element.select_one(trans_sel): parts.append("\nTranslation:\n" + main_content_element.select_one(trans_sel).get_text(separator="\n", strip=True))
                if purport_sel:
                    purport_elements = main_content_element.select(purport_sel)
                    purport_text = "\n\n".join([p.get_text(separator="\n", strip=True) for p in purport_elements if p.get_text(strip=True)])
                    if purport_text: parts.append("\nPurport:\n" + purport_text)
                
                text = "\n\n".join(parts).strip()
                if not text: logger.warning(f"Custom assembly for {url} yielded no text.")
            elif main_content_element:
                text = main_content_element.get_text(separator='\n', strip=True)
            elif target_clean_area is soup.body : # Só extrair do body inteiro como último recurso
                logger.warning(f"No main content element for {url}, extracting from full body (might be noisy).")
                text = soup.body.get_text(separator='\n', strip=True) if soup.body else ""
            else:
                logger.warning(f"No main_content_element for text extraction on {url}. Text will be empty.")
            
            if text:
                lines = [line.strip() for line in text.splitlines()]
                text = '\n\n'.join(line for line in lines if line)

            metadata: Dict[str, Any] = {
                "url": url, "title": title, "source_domain": domain,
                "type": "website", "fetch_timestamp": time.time(),
                **self.sanskrit_processor.process_document_metadata(text)
            }
            
            site_metadata_selectors = site_config.get("metadata_selectors", {})
            for meta_key, meta_config in site_metadata_selectors.items():
                selector_str = meta_config.get("selector") if isinstance(meta_config, dict) else meta_config
                attribute_to_get = meta_config.get("attribute") if isinstance(meta_config, dict) else None
                if selector_str:
                    meta_element = soup.select_one(selector_str)
                    if meta_element:
                        meta_value = meta_element.get(attribute_to_get) if attribute_to_get else meta_element.get_text(strip=True)
                        if meta_value: metadata[meta_key] = meta_value.strip()
            
            if 'description' not in metadata:
                desc_tag = soup.find('meta', attrs={'name': 'description'})
                if desc_tag and desc_tag.has_attr('content'): metadata['description'] = desc_tag['content'].strip()
            if 'og_title' not in metadata and metadata.get("title", "").lower() == "no title found":
                og_title_tag = soup.find('meta', property='og:title')
                if og_title_tag and og_title_tag.has_attr('content'):
                    metadata['og_title'] = og_title_tag['content'].strip()
                    if metadata['title'].lower() == "no title found": metadata['title'] = title = metadata['og_title']
            
            metadata = {k: v for k, v in metadata.items() if v is not None}
            return {"success": True, "text": text, "metadata": metadata, "title": title }
        except Exception as e:
            logger.error(f"Error parsing HTML from {url}: {e}", exc_info=True)
            return {"success": False, "error": f"Parsing failed: {e}"}

    def scrape_url(self, url: str, bypass_cache: bool = False) -> List[Document]:
        html_content = self.fetch_url(url, bypass_cache=bypass_cache)
        if not html_content: return []
        parsed_result = self.parse_html(html_content, url)
        if not parsed_result.get("success"):
            logger.error(f"Failed to parse HTML for {url}: {parsed_result.get('error')}")
            return []
        text, metadata = parsed_result.get("text", ""), parsed_result.get("metadata", {})
        if not text.strip():
             logger.warning(f"No meaningful text extracted from {url}. Skipping document creation.")
             return []
        chunks = self.text_splitter.split_text(text, metadata=metadata)
        logger.info(f"Scraped URL '{url}', generated {len(chunks)} document chunks.")
        if chunks: logger.debug(f"First chunk metadata from '{url}': {chunks[0].metadata}")
        return chunks

    # --- MÉTODOS DE BUSCA POR SITE ---
    def search_purebhakti(self, query_term: str, page_number: int = 1) -> Tuple[Optional[str], Optional[str]]:
        """ Realiza busca POST no purebhakti.com. Retorna (html_content, final_url). """
        search_url = "https://purebhakti.com/resources/search"
        payload = {"q": query_term, "view": "search"}
        # Paginação para purebhakti via POST é complexa e não implementada aqui.
        # Esta função pegará apenas a primeira página de resultados.
        if page_number > 1:
            logger.warning("Pagination for purebhakti.com POST search is not implemented beyond page 1 by this method.")
            return None, None
            
        logger.info(f"Submitting search to {search_url} for term '{query_term}' (page {page_number})")
        # Reutilizar fetch_url para POST
        # A URL de cache para POSTs de busca pode ser problemática se o conteúdo variar muito
        # para a mesma URL base. Considere não cachear resultados de busca ou usar chaves de cache mais específicas.
        # Para simplificar, fetch_url usará a URL da busca como chave de cache.
        html_content = self.fetch_url(search_url, method="POST", data=payload, bypass_cache=True) # Bypass cache para buscas
        
        # O fetch_url não retorna a URL final diretamente. Precisamos modificar fetch_url ou obter de outra forma.
        # Por enquanto, assumimos que a URL de resultados é a mesma ou o conteúdo é o que importa.
        # Para urljoin funcionar corretamente, precisamos da URL da página que contém os links.
        # Se o POST redirecionar, self.session.post(...).url daria a URL final.
        # Vamos assumir que o scraper lida com isso internamente ou que a URL base é suficiente.
        # Idealmente, fetch_url deveria retornar (html, final_url)
        return html_content, search_url # Retornar a URL da busca como base para urljoin

    def scrape_search_results_purebhakti(self, query_term: str, num_articles: int) -> List[Dict[str, str]]:
        """ Extrai URLs e títulos dos resultados da busca no purebhakti.com. """
        articles_found: List[Dict[str, str]] = []
        html_results_page, base_search_url = self.search_purebhakti(query_term) # search_purebhakti retorna (html, url_usada_para_busca)

        if not html_results_page or not base_search_url:
            logger.warning(f"Could not retrieve search results page for '{query_term}' from purebhakti.com.")
            return articles_found

        soup = BeautifulSoup(html_results_page, 'lxml')
        results_container = soup.select_one("ul.com-finder-search-list")
        if not results_container:
            logger.warning(f"Could not find results container on purebhakti.com for '{query_term}'.")
            return articles_found

        for item in results_container.select("li.result_item")[:num_articles]:
            link_tag = item.select_one("h3.result_item_title > a.result_item_title_link")
            if link_tag and link_tag.has_attr('href'):
                href = link_tag['href']
                title = link_tag.get_text(strip=True)
                # Usar a URL da página de resultados (base_search_url) para resolver links relativos
                full_url = urljoin(base_search_url, href) 
                articles_found.append({"url": full_url, "title": title})
        
        logger.info(f"Extracted {len(articles_found)} article links from purebhakti.com search for '{query_term}'.")
        return articles_found

    def search_vedabase(self, query_term: str, page_number: int = 1) -> Tuple[Optional[str], Optional[str]]:
        """ Realiza busca GET no vedabase.io. Retorna (html_content, final_url). """
        # Vedabase usa GET: https://vedabase.io/en/search/?q=QUERY
        # A paginação é &start=10, &start=20 etc. (10 resultados por página)
        start_index = (page_number - 1) * 10
        search_url = f"https://vedabase.io/en/search/?q={quote_plus(query_term)}&start={start_index}"
        
        logger.info(f"Fetching search results from {search_url}")
        html_content = self.fetch_url(search_url, bypass_cache=True) # Bypass cache para buscas
        return html_content, search_url

    def scrape_search_results_vedabase(self, query_term: str, num_articles: int) -> List[Dict[str, str]]:
        """ Extrai URLs e títulos dos resultados da busca no vedabase.io. """
        articles_found: List[Dict[str, str]] = []
        # Vedabase pode ter muitas páginas, então vamos limitar o número de páginas de busca a serem verificadas.
        # Se num_articles = 3, geralmente estará na primeira página.
        # Se num_articles = 15, pode precisar de 2 páginas de busca.
        max_search_pages = (num_articles // 10) + 1 
        
        for page_num in range(1, max_search_pages + 1):
            if len(articles_found) >= num_articles: break

            html_results_page, search_page_url = self.search_vedabase(query_term, page_number=page_num)
            if not html_results_page or not search_page_url:
                logger.warning(f"Could not retrieve search results page {page_num} for '{query_term}' from vedabase.io.")
                break # Parar se uma página de busca falhar

            soup = BeautifulSoup(html_results_page, 'lxml')
            # Seletor para resultados no Vedabase: div.search-result-item h3 a
            results_items = soup.select("div.search-result-item")
            if not results_items and page_num == 1:
                logger.warning(f"No search result items found on vedabase.io for '{query_term}'.")
                break
            if not results_items: # Sem mais resultados nesta página de busca
                break

            for item in results_items:
                if len(articles_found) >= num_articles: break
                link_tag = item.select_one("h3 a")
                if link_tag and link_tag.has_attr('href'):
                    href = link_tag['href']
                    title = link_tag.get_text(strip=True)
                    full_url = urljoin(search_page_url, href) # Usar a URL da página de busca para resolver
                    articles_found.append({"url": full_url, "title": title})
            
            if len(articles_found) < num_articles and len(results_items) < 10 : # Menos de 10 resultados na página, provavelmente a última
                break
        
        logger.info(f"Extracted {len(articles_found)} article links from vedabase.io search for '{query_term}'.")
        return articles_found[:num_articles] # Garantir que não exceda num_articles


    # Placeholder para outros sites - VOCÊ PRECISA IMPLEMENTAR ESTES
    def scrape_search_results_bhaktivedantavediclibrary_org(self, query_term: str, num_articles: int) -> List[Dict[str, str]]:
        logger.warning(f"Search for 'bhaktivedantavediclibrary.org' not implemented yet. Query: '{query_term}'")
        # 1. Descobrir como o search funciona (GET/POST, URL, parâmetros)
        # 2. Implementar a lógica para buscar e parsear a página de resultados (similar aos acima)
        # Exemplo (hipotético):
        # search_url = f"https://bhaktivedantavediclibrary.org/?s={quote_plus(query_term)}"
        # html, _ = self.fetch_url(search_url)
        # ... parsear soup para links ...
        return []

    # ... (outros métodos como extract_links, crawl, clear_cache, get_cache_stats, lookup_sanskrit_term)
    # Mantenha os métodos _respect_rate_limit e outros helpers se forem usados.
    def extract_links(self, html: str, base_url: str, same_domain_only: bool = True) -> List[str]:
        if not html: return []
        try:
            soup = BeautifulSoup(html, 'lxml')
            base_domain = urlparse(base_url).netloc
            links = set()
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href'].strip()
                if not href or href.startswith(('#', 'javascript:', 'mailto:')): continue
                try:
                    absolute_url = urljoin(base_url, href)
                    parsed_absolute_url = urlparse(absolute_url)
                    if parsed_absolute_url.scheme not in ['http', 'https'] or not parsed_absolute_url.netloc: continue
                    if same_domain_only and parsed_absolute_url.netloc != base_domain: continue
                    if is_blacklisted_url(absolute_url): continue # Reutiliza a função de ética
                    links.add(absolute_url)
                except ValueError: continue
            return sorted(list(links))
        except Exception as e:
            logger.error(f"Error extracting links from {base_url}: {e}", exc_info=True)
            return []

    def crawl(self, start_url: str, max_pages: int = 10, same_domain_only: bool = True, bypass_cache: bool = False) -> List[Document]:
        visited_urls, to_visit_queue, all_documents = set(), [start_url], []
        logger.info(f"Starting crawl: {start_url}, max_pages={max_pages}, same_domain={same_domain_only}")
        while to_visit_queue and len(visited_urls) < max_pages:
            current_url = to_visit_queue.pop(0)
            if current_url in visited_urls: continue
            logger.info(f"Visiting ({len(visited_urls)+1}/{max_pages}): {current_url}")
            visited_urls.add(current_url)
            documents = self.scrape_url(current_url, bypass_cache=bypass_cache)
            if documents:
                all_documents.extend(documents)
                # É ineficiente buscar o HTML novamente para extrair links.
                # parse_html poderia retornar o soup, ou scrape_url o html.
                # Para simplificar, vamos manter a busca novamente se necessário, mas com cache.
                html_for_links = self.fetch_url(current_url, bypass_cache=False) # Usar cache aqui
                if html_for_links:
                    new_links = self.extract_links(html_for_links, current_url, same_domain_only)
                    for link in new_links:
                        if link not in visited_urls and link not in to_visit_queue:
                            to_visit_queue.append(link)
            self.ethical_helper.wait_for_domain(current_url) # Esperar após processar uma página
        logger.info(f"Crawl finished. Visited {len(visited_urls)} pages, extracted {len(all_documents)} chunks.")
        return all_documents

    def clear_cache(self) -> int: return self.cache_manager.clear_expired()
    def get_cache_stats(self) -> Dict[str, Any]: return self.cache_manager.get_stats()