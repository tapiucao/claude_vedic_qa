import logging
import requests
import time
from typing import Dict, List, Any, Optional, Tuple
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, quote_plus 
from langchain.docstore.document import Document
from ..config import (
    REQUEST_DELAY, WEB_CACHE_DIR,
    SITE_SPECIFIC_SCRAPING_CONFIG 
)
from ..document_processor.text_splitter import VedicTextSplitter
from ..document_processor.sanskrit_processor import SanskritProcessor
from .cache_manager import WebCacheManager
from .ethics import is_scraping_allowed, is_blacklisted_url, EthicalScraper
import gzip

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
        effective_url = url
        
        # Lógica de cache (pode ser adaptada ou simplificada se necessário para teste)
        if not bypass_cache:
            cached_content = self.cache_manager.get_cached_content(effective_url)
            if cached_content:
                # Heurística simples para verificar se o cache está possivelmente corrompido/gzipped
                # Isso é uma tentativa de evitar usar um cache problemático.
                try:
                    if isinstance(cached_content, str) and len(cached_content) > 2 and cached_content.startswith('\x1f\x8b'):
                        logger.warning(f"Cached content for {effective_url} looks like raw gzip string. Bypassing cache.")
                    elif isinstance(cached_content, bytes) and cached_content.startswith(b'\x1f\x8b'):
                        logger.warning(f"Cached content for {effective_url} looks like raw gzip bytes. Bypassing cache.")
                    else:
                        logger.info(f"Using valid cached content for {effective_url}")
                        return str(cached_content) # Garante que é string
                except Exception: # Se qualquer verificação no cache falhar, melhor buscar de novo
                    logger.warning(f"Error inspecting cached content for {effective_url}. Bypassing cache.")
            
        if not is_scraping_allowed(effective_url): # Supondo que esta função esteja definida e acessível
            logger.warning(f"Scraping disallowed by initial check for {effective_url}")
            return None
        
        self.ethical_helper.wait_for_domain(effective_url) # Supondo que ethical_helper esteja inicializado

        try:
            if method.upper() == "GET":
                response = self.session.get(effective_url, timeout=30)
            elif method.upper() == "POST":
                response = self.session.post(effective_url, data=data, timeout=30)
            else:
                logger.error(f"Unsupported HTTP method: {method}")
                return None
                
            response.raise_for_status()
            final_url = response.url
            
            # Logar os cabeçalhos é MUITO IMPORTANTE para este diagnóstico
            logger.info(f"Response headers for {final_url}: {response.headers}")
            
            content_type_header = response.headers.get('Content-Type', '').lower()
            content_encoding_header = response.headers.get('Content-Encoding', '').lower()
            raw_content = response.content # Bytes brutos

            html_content_str = None

            if raw_content.startswith(b'\x1f\x8b'): # Magic numbers para GZIP
                logger.info(f"Content from {final_url} starts with GZIP magic numbers.")
                try:
                    decompressed_bytes = gzip.decompress(raw_content)
                    logger.info(f"Successfully decompressed gzipped content from {final_url} (length: {len(decompressed_bytes)} bytes).")
                    # Agora, decodificar os bytes descomprimidos para string
                    try:
                        html_content_str = decompressed_bytes.decode('utf-8')
                        logger.info(f"Decoded decompressed content from {final_url} using UTF-8.")
                    except UnicodeDecodeError:
                        logger.warning(f"UTF-8 decoding failed for decompressed content from {final_url}. Trying other encodings.")
                        # Tentar com o charset do Content-Type header, se disponível
                        charset = None
                        if 'charset=' in content_type_header:
                            charset = content_type_header.split('charset=')[-1].split(';')[0].strip()
                        
                        # Tentar com o response.apparent_encoding (que o 'requests' adivinha)
                        fallback_encodings = [charset, response.apparent_encoding, 'latin-1', 'iso-8859-1']
                        for enc in fallback_encodings:
                            if enc: # Se o encoding não for None
                                try:
                                    logger.info(f"Attempting to decode with: {enc}")
                                    html_content_str = decompressed_bytes.decode(enc)
                                    logger.info(f"Successfully decoded decompressed content with {enc}.")
                                    break # Sai do loop se a decodificação for bem-sucedida
                                except UnicodeDecodeError:
                                    logger.warning(f"Decoding with {enc} failed.")
                                except Exception as e_dec: # Outros erros de decodificação
                                    logger.warning(f"Error decoding with {enc}: {e_dec}")
                        if html_content_str is None:
                            logger.error(f"Could not decode decompressed content from {final_url} with any attempted encoding. Content might be corrupted.")
                            html_content_str = decompressed_bytes.decode('latin-1', errors='replace') # Última tentativa com substituição
                            logger.warning(f"Using latin-1 with error replacement as last resort for {final_url}.")

                except gzip.BadGzipFile:
                    logger.error(f"Content from {final_url} has gzip magic numbers but is a BadGzipFile. Using raw content as text (might be garbled).")
                    html_content_str = raw_content.decode('latin-1', errors='replace') # Tenta decodificar os bytes originais
                except Exception as e_gz:
                    logger.error(f"Error during manual gzip decompression for {final_url}: {e_gz}. Using raw content as text (might be garbled).")
                    html_content_str = raw_content.decode('latin-1', errors='replace')
            else:
                # Se não começar com magic numbers de gzip, assume que 'requests' lidou com qualquer 'Content-Encoding' (como 'br')
                # ou que não está comprimido de forma que 'requests' não detecte.
                logger.info(f"Content from {final_url} does not start with GZIP magic numbers. Relying on requests.text for decoding.")
                # Define o encoding para 'response.text' se não foi pego dos headers
                if response.encoding is None:
                    response.encoding = response.apparent_encoding or 'utf-8'
                html_content_str = response.text
                logger.info(f"Used requests.text with encoding {response.encoding} for {final_url}.")

            if not html_content_str: # Se, após tudo, ainda for None ou vazio
                logger.error(f"Failed to obtain any processable HTML string for {final_url}.")
                return None

            # Checagem do Content-Type para garantir que é HTML (mesmo que já tenha sido checado antes)
            if not ('text/html' in content_type_header or \
                    'application/xhtml+xml' in content_type_header or \
                    'application/xml' in content_type_header):
                logger.warning(f"Final content for {final_url} does not have a standard HTML/XML Content-Type ('{content_type_header}'), but proceeding to cache and parse.")

            # (A checagem ética is_scraping_allowed(final_url, html_content_str) pode vir aqui)

            self.cache_manager.cache_content(final_url, html_content_str)
            logger.info(f"Successfully processed and cached URL: {final_url}. Length: {len(html_content_str)}")
            return html_content_str

        except requests.exceptions.RequestException as e_req:
            logger.error(f"RequestException for {effective_url} ({method}): {e_req}")
            return None
        except Exception as e_gen:
             logger.error(f"Unexpected error in fetch_url for {effective_url} ({method}): {e_gen}", exc_info=True)
             return None

    def parse_html(self, html: str, url: str) -> Dict[str, Any]:
        if not html:
            return {"success": False, "error": "No HTML content provided"}

        parsed_url_obj = urlparse(url)
        domain = parsed_url_obj.netloc.replace("www.", "")
        site_config = SITE_SPECIFIC_SCRAPING_CONFIG.get(domain, {})

        logger.debug(f"Parsing HTML for {url} (domain: {domain}) using site config: {bool(site_config)}")

        try:
            soup = BeautifulSoup(html, 'lxml')
            title = "No title found" # Default title
            # ... (your existing title extraction logic) ...
            # (Make sure title extraction is robust here, e.g., from soup.find('title') if specific selectors fail)
            title_tag_from_html = soup.find('title')
            if title_tag_from_html and title_tag_from_html.string:
                title = title_tag_from_html.string.strip()

            if site_config.get("title_selector"):
                title_element = soup.select_one(site_config.get("title_selector"))
                if title_element:
                    title = title_element.get_text(strip=True) or title # Use found title or fallback


            main_content_element = None
            content_selectors = site_config.get("content_selectors", ['article', 'main', '.content', '#content', '.main', '#main', 'div[role="main"]'])
            for selector in content_selectors:
                main_content_element = soup.select_one(selector)
                if main_content_element:
                    logger.debug(f"Found main content for {url} using selector: '{selector}'") 
                    # logger.info(f"ARTICLE_CONTENT_DEBUG ({url}): Initial main_content_element HTML (BEFORE ANY .decompose() calls):\n{main_content_element.prettify(formatter='html5')}")
                    break
            
            target_clean_area = main_content_element if main_content_element else soup.body # Fallback to soup.body if no main_content_element
            if not target_clean_area: # If even soup.body is None (highly unlikely for valid HTML)
                logger.error(f"No target_clean_area (not even soup.body) for {url}. HTML might be malformed.")
                return {"success": False, "error": "HTML structure issue, no body or main content found."}


            elements_to_remove_selectors = site_config.get("elements_to_remove_selectors", [])
            generic_elements_to_remove = ['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'button', 'iframe', 'noscript', '.noprint', '.advertisement', '.ad', '.popup', '.cookie-consent', '.related-links', '.sidebar', '.social-share', 'figure > figcaption']
            combined_remove_selectors = list(set(elements_to_remove_selectors + generic_elements_to_remove))

            if target_clean_area:
                for selector_str_to_remove in combined_remove_selectors: # Renamed to avoid conflict
                    try:
                        for element_to_decompose in target_clean_area.select(selector_str_to_remove): # Renamed
                            element_to_decompose.decompose()
                    except Exception as decomp_err:
                        logger.warning(f"Error decomposing element with selector '{selector_str_to_remove}' for {url}: {decomp_err}")
            
            # Initialize text variable
            text = ""

            if main_content_element: # Only proceed if a main content element was identified
                if site_config.get("custom_content_assembly"):
                    logger.info(f"Attempting custom content assembly for {domain} from URL: {url}")
                    
                    parts = [] # <<<< MAKE SURE THIS LINE IS PRESENT AND CORRECTLY INDENTED

                    verse_sel = site_config.get("verse_selector")
                    syn_sel = site_config.get("synonyms_selector")
                    trans_sel = site_config.get("translation_selector")
                    purport_sel = site_config.get("purport_selector")

                    if verse_sel:
                        verse_element = main_content_element.select_one(verse_sel)
                        if verse_element: parts.append("Original Verse:\n" + verse_element.get_text(separator="\n", strip=True))
                    if syn_sel:
                        syn_element = main_content_element.select_one(syn_sel)
                        if syn_element: parts.append("\nSynonyms:\n" + syn_element.get_text(separator="\n", strip=True))
                    if trans_sel:
                        trans_element = main_content_element.select_one(trans_sel)
                        if trans_element: parts.append("\nTranslation:\n" + trans_element.get_text(separator="\n", strip=True))
                    if purport_sel:
                        purport_elements = main_content_element.select(purport_sel)
                        purport_text_parts = [p.get_text(separator="\n", strip=True) for p in purport_elements if p.get_text(strip=True)]
                        if purport_text_parts:
                            parts.append("\nPurport:\n" + "\n\n".join(purport_text_parts))
                    
                    text = "\n\n".join(parts).strip() # This is where the NameError occurred

            elif target_clean_area is soup.body: # Fallback if main_content_element was not found at all
                logger.warning(f"No main content element for {url}, extracting from full body (after generic removals). This might be noisy.")
                text = target_clean_area.get_text(separator='\n', strip=True)
            
            else: # Should not happen if target_clean_area is soup.body as a fallback
                logger.warning(f"No text could be extracted for {url} as no main_content_element or soup.body was processed.")


            # Final text processing
            if text:
                lines = [line.strip() for line in text.splitlines()]
                text = '\n\n'.join(line for line in lines if line)

            metadata: Dict[str, Any] = {
                "url": url, "title": title, "source_domain": domain,
                "type": "website", "fetch_timestamp": time.time(), # Use time.time() for float timestamp
                **self.sanskrit_processor.process_document_metadata(text)
            }
            
            # ... (your existing metadata_selectors logic) ...
            site_metadata_selectors = site_config.get("metadata_selectors", {})
            for meta_key, meta_config in site_metadata_selectors.items():
                selector_str_meta = meta_config.get("selector") if isinstance(meta_config, dict) else meta_config # Renamed
                attribute_to_get = meta_config.get("attribute") if isinstance(meta_config, dict) else None
                if selector_str_meta:
                    meta_element = soup.select_one(selector_str_meta)
                    if meta_element:
                        meta_value = meta_element.get(attribute_to_get) if attribute_to_get else meta_element.get_text(strip=True)
                        if meta_value: metadata[meta_key] = meta_value.strip()
            
            if 'description' not in metadata:
                desc_tag = soup.find('meta', attrs={'name': 'description'})
                if desc_tag and desc_tag.has_attr('content'): metadata['description'] = desc_tag['content'].strip()
            # (Keep your og_title logic if it was working)

            metadata = {k: v for k, v in metadata.items() if v is not None} # Clean None values
            return {"success": True, "text": text, "metadata": metadata, "title": title }

        except Exception as e:
            logger.error(f"Error parsing HTML from {url}: {e}", exc_info=True)
            # Ensure it still returns the required dict structure on error
            return {"success": False, "error": f"Parsing failed: {e}", "text": "", "metadata": {"url": url, "title": "Error during parsing"}, "title": "Error during parsing"}

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

    # Dentro da classe VedicWebScraper em scraper.py
    def search_purebhakti(self, query_term: str, page_number: int = 1) -> Tuple[Optional[str], Optional[str]]:

        base_site_url = "https://www.purebhakti.com/"
        search_path_with_query = f"resources/search?q={quote_plus(query_term)}" # quote_plus de urllib.parse
        search_url = urljoin(base_site_url, search_path_with_query) # urljoin de urllib.parse
        logger.info(f"VedicWebScraper (estático): Tentando buscar página de busca do purebhakti.com (via requests): {search_url}")
        # self.fetch_url aqui é o da classe VedicWebScraper (baseado em requests)
        html_content = self.fetch_url(search_url, bypass_cache=True)
        return html_content, search_url

    # Dentro da classe VedicWebScraper em scraper.py
    def search_vedabase(self, query_term: str, page_number: int = 1) -> Tuple[Optional[str], Optional[str]]:
        """
        Realiza busca no vedabase.io usando o fetch_url padrão (requests) desta classe.
        Isso obterá o HTML inicial, que para o vedabase.io, é um esqueleto.
        """
        start_index = (page_number - 1) * 10
        search_url = f"https://vedabase.io/en/search/?query={quote_plus(query_term)}&start={start_index}"

        logger.info(f"VedicWebScraper (estático): Buscando resultados para vedabase.io (via requests): {search_url}")
        # self.fetch_url aqui é o da classe VedicWebScraper (baseado em requests)
        html_content = self.fetch_url(search_url, bypass_cache=True) 
        return html_content, search_url

    def scrape_search_results_vedabase(self, query_term: str, num_articles: int) -> List[Dict[str, str]]:
        articles_found: List[Dict[str, str]] = []
        max_search_pages = (num_articles + 9) // 10 # Ensures enough pages for num_articles items (10 per page)
        
        logger.info(f"Starting to scrape vedabase.io for '{query_term}', targeting {num_articles} articles across max {max_search_pages} pages.")

        try:
            for page_num in range(1, max_search_pages + 1):
                if len(articles_found) >= num_articles:
                    logger.info(f"Already found {len(articles_found)} articles, reaching target of {num_articles}. Stopping pagination.")
                    break

                # This call now correctly invokes DynamicVedicScraper.search_vedabase
                # which successfully returns HTML with results.
                html_results_page, search_page_url = self.search_vedabase(query_term, page_number=page_num) 
                
                if not html_results_page: # search_page_url might still be valid if html_results_page is None after an error
                    logger.warning(f"Could not retrieve HTML for search results page {page_num} for '{query_term}' from vedabase.io. Base URL for this attempt: {search_page_url or 'Unknown'}")
                    # Decide if you want to break or continue to next page if one page fails
                    # For now, let's break if a page fetch fails critically (no HTML)
                    break 

                if not search_page_url: # Should ideally not happen if html_results_page is present
                    logger.error(f"Search page URL is missing for vedabase.io query '{query_term}' page {page_num}, cannot resolve relative links. Skipping page.")
                    continue


                soup = BeautifulSoup(html_results_page, 'lxml')
                
                # CSS selector for each search result item, from your screenshot
                # Escaping the colon for CSS: div.search-result.em\:mb-4
                results_items_selector = "div.search-result.em\\:mb-4" 
                results_items = soup.select(results_items_selector)

                logger.info(f"Page {page_num} for '{query_term}': Found {len(results_items)} item(s) using selector '{results_items_selector}'.")

                if not results_items: # No items found on this page
                    if page_num == 1:
                        logger.warning(f"No search result items found using '{results_items_selector}' on vedabase.io for '{query_term}' on the first page.")
                        # Save debug HTML if no results on the first page (even if Selenium found the container)
                        # This helps see what BeautifulSoup is working with.
                        debug_html_bs_fail_path = f"vedabase_BS_FAIL_debug_output_{quote_plus(query_term)}_page{page_num}.html"
                        try:
                            with open(debug_html_bs_fail_path, "w", encoding="utf-8") as f_debug_bs:
                                f_debug_bs.write(html_results_page)
                            logger.info(f"Saved BeautifulSoup FAIL debug HTML to '{debug_html_bs_fail_path}'")
                        except Exception as e_save_bs_debug:
                            logger.error(f"Error saving BeautifulSoup FAIL debug HTML: {e_save_bs_debug}")
                    else:
                        logger.info(f"No more search results found on page {page_num} for '{query_term}'.")
                    break # Stop paginating if no results found

                for item_idx, item in enumerate(results_items):
                    if len(articles_found) >= num_articles:
                        break
                    
                    # Selector for the link tag based on your screenshot: <a> with classes "text-vb-link" and "em:text-lg"
                    link_tag_selector = "a.text-vb-link.em\\:text-lg" # Escaping the colon
                    link_tag = item.select_one(link_tag_selector)
                    
                    if link_tag and link_tag.has_attr('href'):
                        href = link_tag['href']
                        title = link_tag.get_text(strip=True)
                        
                        # Ensure search_page_url (base for resolving relative URLs) is valid
                        if not search_page_url:
                            logger.error(f"Cannot resolve relative href '{href}' because search_page_url is missing for query '{query_term}'. Skipping item.")
                            continue
                            
                        full_url = urljoin(search_page_url, href)
                        
                        articles_found.append({"url": full_url, "title": title})
                        logger.debug(f"Found result #{len(articles_found)} (Item {item_idx+1}/Page {page_num}): '{title}' - {full_url}")
                    else:
                        logger.warning(f"Item {item_idx+1}/Page {page_num} for '{query_term}': Could not find a valid link_tag using '{link_tag_selector}'. Item HTML snippet: {str(item)[:250]}")
                
                if len(articles_found) >= num_articles: # Check after processing items on the current page
                    logger.info(f"Reached target num_articles ({num_articles}) after page {page_num}. Stopping pagination.")
                    break
            
            logger.info(f"Finished scraping vedabase.io for '{query_term}'. Found {len(articles_found)} articles in total.")
            return articles_found

        except Exception as e_main:
            logger.error(f"Critical exception in scrape_search_results_vedabase for '{query_term}': {e_main}", exc_info=True)
            return [] 

    def scrape_search_results_bhaktivedantavediclibrary_org(self, query_term: str, num_articles: int) -> List[Dict[str, str]]:
        logger.warning(f"Search for 'bhaktivedantavediclibrary.org' not implemented yet. Query: '{query_term}'")
        return []

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