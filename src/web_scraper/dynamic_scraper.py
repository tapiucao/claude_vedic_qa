# Updated: src/web_scraper/dynamic_scraper.py
"""
Dynamic web scraping for JavaScript-heavy sites.
Uses Selenium to handle JavaScript-rendered content for Vedic Knowledge AI.
"""
import logging
import time
from typing import List, Any, Optional, Tuple, Dict
from bs4 import BeautifulSoup 
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions 
from selenium.webdriver.chrome.service import Service as ChromeService 
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, WebDriverException, NoSuchElementException,
    ElementNotInteractableException, StaleElementReferenceException
)
from webdriver_manager.chrome import ChromeDriverManager
from langchain.docstore.document import Document

from ..config import REQUEST_DELAY, WEB_CACHE_DIR 
from .scraper import VedicWebScraper
from .ethics import is_scraping_allowed
from urllib.parse import urljoin, urlparse, quote_plus

logger = logging.getLogger(__name__)

class DynamicVedicScraper(VedicWebScraper):
    """Dynamic scraper for JavaScript-heavy websites using Selenium."""

    # Keep cache_dir consistent with superclass if needed, add type hints
    def __init__(self, cache_dir: str = WEB_CACHE_DIR, request_delay: int = REQUEST_DELAY):
        """Initialize the dynamic scraper."""
        # Initialize superclass (which initializes cache_manager, etc.)
        super().__init__(request_delay=request_delay, cache_dir=cache_dir)
        self.driver: Optional[webdriver.Chrome] = None # Type hint for driver

        logger.info("Initialized dynamic web scraper")

    def _initialize_driver(self) -> None:
        """Initialize the Selenium WebDriver."""
        if self.driver:
            logger.debug("WebDriver already initialized.")
            return

        logger.info("Initializing WebDriver...")
        try:
            # Configure Chrome options
            chrome_options = ChromeOptions()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            if hasattr(self, 'session') and self.session.headers.get('User-Agent'):
                chrome_options.add_argument(f"user-agent={self.session.headers['User-Agent']}")
            else:
                # Fallback para um User-Agent genérico se o da sessão não estiver disponível por algum motivo
                logger.warning("User-Agent from session headers not found, using a default User-Agent for WebDriver.")
                chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36 VedicKnowledgeBot/1.0_Dynamic")

            service = ChromeService(ChromeDriverManager().install())

            # Initialize the driver
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            logger.info("WebDriver initialized successfully")
        except WebDriverException as e:
            logger.error(f"WebDriver initialization failed: {str(e)}. Check ChromeDriver compatibility and installation.")
            self.driver = None # Ensure driver is None on failure
            raise # Re-raise the exception to signal failure
        except Exception as e: # Catch other potential errors
             logger.error(f"Unexpected error initializing WebDriver: {str(e)}")
             self.driver = None
             raise


    def _close_driver(self) -> None:
        """Close the WebDriver if it's open."""
        if self.driver:
            logger.info("Closing WebDriver...")
            try:
                self.driver.quit()
                logger.info("WebDriver closed successfully")
            except WebDriverException as e:
                logger.error(f"Error closing WebDriver: {str(e)}")
            finally:
                self.driver = None # Ensure driver is set to None


    # Overriding fetch_url to use Selenium
    def fetch_url(self, url: str, bypass_cache: bool = False, wait_time: int = 10, scroll: bool = True) -> Optional[str]:
        """Fetch content from a URL using Selenium with JavaScript support, includes caching."""

        # Check cache first (using superclass's cache manager) unless bypassing
        if not bypass_cache:
            cached_content = self.cache_manager.get_cached_content(url)
            if cached_content:
                logger.info(f"Using cached content for dynamic URL: {url}")
                return cached_content

        # Respect rate limiting (from superclass)
        self._respect_rate_limit()

        # Initialize driver if needed
        if not self.driver:
            try:
                self._initialize_driver()
            except Exception: # Handle initialization failure
                 return None # Cannot fetch without driver

        # --- Use try...finally to ensure driver cleanup ---
        html: Optional[str] = None
        try:
            logger.debug(f"Dynamically fetching URL: {url}")
            # Navigate to URL
            self.driver.get(url) # type: ignore # Add ignore if type checker complains about None driver

            # Wait for page body to be present
            WebDriverWait(self.driver, wait_time).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            # Optional: Scroll to load lazy content
            if scroll:
                self._scroll_page()

            # Get the page source after JavaScript execution
            html = self.driver.page_source # type: ignore

            # Cache the fetched content
            if html:
                self.cache_manager.cache_content(url, html)

            logger.info(f"Successfully fetched dynamic URL: {url}")
            return html

        except TimeoutException:
            logger.warning(f"Timeout while loading dynamic URL: {url}")
            # Attempt to return whatever loaded, cache it if possible
            try:
                html = self.driver.page_source # type: ignore
                if html: self.cache_manager.cache_content(url, html)
                return html
            except WebDriverException:
                 return None # Driver might be unusable after timeout
        except WebDriverException as e:
            logger.error(f"WebDriver error fetching dynamic URL {url}: {str(e)}")
            # Consider closing and reopening the driver on severe errors
            # self._close_driver()
            return None
        except Exception as e: # Catch other unexpected errors
             logger.error(f"Unexpected error fetching dynamic URL {url}: {str(e)}")
             return None
        # No finally block needed here as driver closing is handled elsewhere or by external context manager

    def _scroll_page(self, scroll_pause_time: float = 1.0, max_scrolls: int = 5) -> None:
        """Scroll the page to trigger lazy loading."""
        if not self.driver:
            logger.warning("Cannot scroll page, WebDriver not available.")
            return
        try:
            # Get initial scroll height
            last_height = self.driver.execute_script("return document.body.scrollHeight")

            logger.debug("Scrolling page to load dynamic content...")
            for i in range(max_scrolls):
                # Scroll down
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

                # Wait to load the page
                time.sleep(scroll_pause_time)

                # Calculate new scroll height and compare with last scroll height
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    logger.debug(f"Scrolling stopped after {i+1} scrolls, height didn't change.")
                    break  # No more content to load
                last_height = new_height
            else: # If loop finished without break
                 logger.debug(f"Scrolling completed after max {max_scrolls} scrolls.")


            # Optional: Scroll back to top (might not always be necessary)
            # self.driver.execute_script("window.scrollTo(0, 0);")
        except WebDriverException as e:
            logger.error(f"Error during page scrolling: {str(e)}")
        except Exception as e: # Catch other unexpected errors
             logger.error(f"Unexpected error during page scrolling: {str(e)}")

    def wait_for_element(self, selector: str, by: By = By.CSS_SELECTOR, wait_time: int = 10) -> bool:
        """Wait for a specific element to appear on the page."""
        if not self.driver:
            logger.error("WebDriver not initialized, cannot wait for element.")
            return False

        logger.debug(f"Waiting for element '{selector}' by {by} for {wait_time}s...")
        try:
            WebDriverWait(self.driver, wait_time).until(
                EC.presence_of_element_located((by, selector))
            )
            logger.debug(f"Element '{selector}' found.")
            return True
        except TimeoutException:
            logger.warning(f"Element '{selector}' not found within {wait_time} seconds.")
            return False
        except (WebDriverException, NoSuchElementException) as e: # Catch specific Selenium errors
            logger.error(f"Error waiting for element '{selector}': {str(e)}")
            return False
        except Exception as e: # Catch other unexpected errors
             logger.error(f"Unexpected error waiting for element '{selector}': {str(e)}")
             return False


    def click_element(self, selector: str, by: By = By.CSS_SELECTOR, wait_time: int = 10) -> bool:
        """Click an element on the page."""
        if not self.driver:
            logger.error("WebDriver not initialized, cannot click element.")
            return False

        logger.debug(f"Attempting to click element '{selector}' by {by}...")
        try:
            # Wait for element to be clickable
            element = WebDriverWait(self.driver, wait_time).until(
                EC.element_to_be_clickable((by, selector))
            )
            # Click the element
            element.click()
            logger.debug(f"Clicked element '{selector}' successfully.")
            return True
        except (TimeoutException, ElementNotInteractableException, StaleElementReferenceException) as e: # Specific exceptions
            logger.error(f"Error clicking element '{selector}': {str(e)}")
            return False
        except WebDriverException as e: # General WebDriver errors
            logger.error(f"WebDriver error clicking element '{selector}': {str(e)}")
            return False
        except Exception as e: # Catch other unexpected errors
             logger.error(f"Unexpected error clicking element '{selector}': {str(e)}")
             return False

    # Override scrape_url to use dynamic fetching
    def scrape_url(self, url: str, bypass_cache: bool = False, wait_selectors: Optional[List[str]] = None) -> List[Document]:
        """Scrape a dynamic URL and convert content to Document objects."""
        # Check ethics first (using superclass method)
        if not is_scraping_allowed(url): # is_scraping_allowed needs import if not already present
            logger.warning(f"Scraping not ethically allowed for dynamic URL: {url}")
            return []

        # Use the overridden dynamic fetch_url
        # Pass wait_time, scroll options if needed, or use defaults
        html = self.fetch_url(url, bypass_cache=bypass_cache) # This now uses Selenium
        if not html:
            # Fetch_url already logs errors
            # logger.error(f"Failed to fetch dynamic URL: {url}")
            return []

        # Check ethics again with fetched HTML content
        if not is_scraping_allowed(url, html):
             logger.warning(f"Content prohibits scraping after fetch for dynamic URL: {url}")
             return []

        # Parse the HTML (using superclass method)
        parsed_result = self.parse_html(html, url)
        if not parsed_result.get("success", False):
            # Parse_html already logs errors
            # logger.error(f"Failed to parse HTML from dynamic URL: {url}")
            return []

        # Create a Document from the parsed content
        text = parsed_result.get("text", "")
        metadata = parsed_result.get("metadata", {})

        # Split the text into chunks (using superclass's text_splitter)
        chunks = self.text_splitter.split_text(text, metadata)

        logger.info(f"Successfully scraped dynamic URL {url} into {len(chunks)} chunks")
        return chunks

    def extract_dynamic_links(self, url: str, link_selector: str = "a", same_domain_only: bool = True) -> List[str]:
        """Extract links from a dynamic page after ensuring it's loaded."""
        if not self.driver:
             # Attempt to fetch the URL which initializes the driver
             logger.info(f"Driver not active, fetching URL {url} first to extract dynamic links.")
             if not self.fetch_url(url): # Fetch might fail
                  logger.error(f"Could not fetch {url} to extract links.")
                  return []
             # If fetch was successful, driver should now be initialized
             if not self.driver:
                  logger.error("Driver still not available after fetch attempt.")
                  return []
        else:
             # If driver exists, ensure we are on the right page or fetch it
             try:
                 if self.driver.current_url != url:
                     logger.info(f"Driver on different URL ({self.driver.current_url}), fetching {url} for link extraction.")
                     if not self.fetch_url(url):
                          logger.error(f"Could not fetch {url} to extract links.")
                          return []
             except WebDriverException: # Handle case where driver might be closed or invalid
                 logger.error("WebDriver error checking current URL. Fetching URL again.")
                 if not self.fetch_url(url):
                      logger.error(f"Could not fetch {url} to extract links.")
                      return []


        logger.debug(f"Extracting links matching '{link_selector}' from dynamic page {url}...")
        links: List[str] = []
        try:
            # Find all link elements matching the selector (usually 'a')
            link_elements = self.driver.find_elements(By.TAG_NAME, link_selector)

            # Extract href attributes
            for element in link_elements:
                try:
                    href = element.get_attribute("href")
                    # Basic validation of href
                    if href and isinstance(href, str) and not href.startswith("#") and not href.startswith("javascript:"):
                         # Resolve relative URLs using the current page URL
                         absolute_url = urljoin(self.driver.current_url, href) # Use current URL as base
                         links.append(absolute_url)
                except StaleElementReferenceException:
                     logger.warning("Stale element reference encountered while extracting links, skipping element.")
                     continue # Skip this element

            # Filter by domain if requested
            if same_domain_only:
                base_domain = urlparse(url).netloc
                links = [link for link in links if urlparse(link).netloc == base_domain]

            # Remove duplicates while preserving order (if important) or just use set for efficiency
            unique_links = list(dict.fromkeys(links)) # Preserves order

            logger.info(f"Extracted {len(unique_links)} unique links from dynamic page {url}")
            return unique_links
        except WebDriverException as e:
            logger.error(f"WebDriver error extracting links from dynamic page {url}: {str(e)}")
            return []
        except Exception as e: # Catch other unexpected errors
             logger.error(f"Unexpected error extracting links from {url}: {str(e)}")
             return []

    def search_vedabase(self, query_term: str, page_number: int = 1) -> Tuple[Optional[str], Optional[str]]:
        start_index = (page_number - 1) * 10
        search_url = f"https://vedabase.io/en/search/?query={quote_plus(query_term)}&start={start_index}"
        logger.info(f"DynamicVedicScraper: Iniciando busca DINÂMICA em vedabase.io: {search_url}")

        self._initialize_driver() # Garante que o driver está pronto
        if not self.driver:
            logger.error("DynamicVedicScraper: Driver não inicializado para busca no vedabase.io.")
            return None, search_url

        html_content = None
        actual_search_url_after_load = search_url
        try:
            # A linha abaixo usa o self.fetch_url DESTA CLASSE (DynamicVedicScraper), que é baseado em Selenium.
            html_content = self.fetch_url(search_url, bypass_cache=True, wait_time=15, scroll=True)
            actual_search_url_after_load = self.driver.current_url if self.driver else search_url

            if html_content or self.driver: # Added self.driver check for safety
                # The selector for an INDIVIDUAL search result item.
                # From your screenshot, it's a div with class "search-result" and "em:mb-4"
                # Using a CSS selector is safer for classes with special characters or multiple parts.
                individual_result_selector = "div.search-result.em\\:mb-4" # Escaping the colon for CSS selector
                # Or, more simply, if "search-result" is unique enough for these items:
                # individual_result_selector = "div.search-result"

                logger.info(f"DynamicVedicScraper: Aguardando por elementos com SELETOR CSS '{individual_result_selector}' no vedabase.io...")
                results_loaded = self.wait_for_element(selector=individual_result_selector, by=By.CSS_SELECTOR, wait_time=25) # Increased wait time

                if results_loaded:
                    logger.info(f"DynamicVedicScraper: Elemento(s) '{individual_result_selector}' encontrado(s). Obtendo HTML final.")
                    html_content = self.driver.page_source # Get the page source again AFTER elements are loaded
                else:
                    logger.warning(f"DynamicVedicScraper: Elemento(s) '{individual_result_selector}' NÃO encontrado(s) em {actual_search_url_after_load} para vedabase.io.")
                    # Save debug HTML if results still not found
                    if self.driver:
                        failed_html_path = f"vedabase_DYNAMIC_FAIL_debug_output_{quote_plus(query_term)}_page{page_number}.html"
                        try:
                            with open(failed_html_path, "w", encoding="utf-8") as f_debug:
                                f_debug.write(self.driver.page_source)
                            logger.info(f"Saved DYNAMIC FAIL debug HTML to '{failed_html_path}'")
                        except Exception as e_save_debug:
                            logger.error(f"Error saving DYNAMIC FAIL debug HTML: {e_save_debug}")
        except Exception as e:
            logger.error(f"DynamicVedicScraper: Erro durante busca dinâmica no vedabase.io: {e}", exc_info=True)
            if self.driver:
                try: html_content = self.driver.page_source
                except: pass
        # Não feche o driver aqui, o VedicRetriever gerenciará
        return html_content, actual_search_url_after_load

    def _fetch_purebhakti_search_page_html(self, query_term: str, page_number: int = 1) -> Tuple[Optional[str], Optional[str]]:

        if page_number > 1:
            logger.warning(f"Pagination for purebhakti.com search (page {page_number}) is not explicitly handled in the current URL structure. Fetching first page for query: {query_term}")
            search_url = f"https://www.purebhakti.com/resources/search?q={quote_plus(query_term)}"
        else:
            search_url = f"https://www.purebhakti.com/resources/search?q={quote_plus(query_term)}"
        
        logger.info(f"DynamicVedicScraper: Initiating dynamic search on purebhakti.com: {search_url}")

        self._initialize_driver() 
        if not self.driver:
            logger.error("DynamicVedicScraper: WebDriver not initialized for purebhakti.com search.")
            return None, search_url

        html_content = None
        actual_search_url_after_load = search_url
        try:
            # Fetch the URL using Selenium
            html_content = self.fetch_url(search_url, bypass_cache=True, wait_time=20, scroll=True)
            actual_search_url_after_load = self.driver.current_url if self.driver else search_url

            if html_content:
                # Wait for an individual result item to ensure content is loaded
                results_readiness_selector = "ul#search-result-list li.result__item" 
                logger.info(f"DynamicVedicScraper: Waiting for elements with selector '{results_readiness_selector}' on purebhakti.com...")
                results_loaded = self.wait_for_element(selector=results_readiness_selector, by=By.CSS_SELECTOR, wait_time=25)
                
                if results_loaded:
                    logger.info(f"DynamicVedicScraper: Elements matching '{results_readiness_selector}' found. Getting final HTML for purebhakti.")
                    html_content = self.driver.page_source 
                else:
                    logger.warning(f"DynamicVedicScraper: Readiness selector '{results_readiness_selector}' NOT found on {actual_search_url_after_load} for purebhakti.com.")
                    if self.driver: # Save debug HTML if results readiness selector not found
                        failed_html_path = f"purebhakti_DYNAMIC_FAIL_debug_{quote_plus(query_term)}_page{page_number}.html"
                        try:
                            with open(failed_html_path, "w", encoding="utf-8") as f_debug:
                                f_debug.write(self.driver.page_source)
                            logger.info(f"Saved purebhakti DYNAMIC FAIL debug HTML to '{failed_html_path}'")
                        except Exception as e_save_debug:
                            logger.error(f"Error saving purebhakti DYNAMIC FAIL debug HTML: {e_save_debug}")
            else:
                logger.warning(f"DynamicVedicScraper: fetch_url returned no HTML for purebhakti.com search URL {search_url}.")
        except Exception as e:
            logger.error(f"DynamicVedicScraper: Error during dynamic fetch for purebhakti.com search: {e}", exc_info=True)
            if self.driver:
                try: html_content = self.driver.page_source
                except: pass # Ignore error if driver is already in a bad state
        return html_content, actual_search_url_after_load

    def scrape_search_results_purebhakti(self, query_term: str, num_articles: int) -> List[Dict[str, str]]:
        articles_found: List[Dict[str, str]] = []
        # Assuming purebhakti.com search shows enough results on the first page for `num_articles`
        # or that pagination needs more specific handling if required.
        max_search_pages = 1 
        logger.info(f"Scraping purebhakti.com for '{query_term}', targeting {num_articles} articles across max {max_search_pages} pages.")

        try:
            for page_num in range(1, max_search_pages + 1):
                if len(articles_found) >= num_articles:
                    break

                html_results_page, search_page_url = self._fetch_purebhakti_search_page_html(query_term, page_number=page_num)

                if not html_results_page:
                    logger.warning(f"Could not retrieve HTML for search results page {page_num} for '{query_term}' from purebhakti.com.")
                    break 
                
                if not search_page_url:
                    logger.error(f"Search page URL is missing for purebhakti.com query '{query_term}' page {page_num}, cannot resolve relative links.")
                    break

                soup = BeautifulSoup(html_results_page, 'lxml')
                
                # Corrected selector for each search result item based on debug HTML
                results_items_selector = "ul#search-result-list li.result__item" 
                results_items = soup.select(results_items_selector)
                
                logger.info(f"Purebhakti Page {page_num} for '{query_term}': Found {len(results_items)} item(s) using selector '{results_items_selector}'.")

                if not results_items:
                    if page_num == 1:
                        logger.warning(f"No search result items found using '{results_items_selector}' on purebhakti.com for '{query_term}' on the first page.")
                        # Optional: Save debug HTML if needed
                        # debug_html_bs_fail_path = f"purebhakti_BS_FAIL_debug_{quote_plus(query_term)}_page{page_num}.html"
                        # try:
                        #     with open(debug_html_bs_fail_path, "w", encoding="utf-8") as f_debug_bs:
                        #         f_debug_bs.write(html_results_page)
                        #     logger.info(f"Saved purebhakti BeautifulSoup FAIL debug HTML to '{debug_html_bs_fail_path}'")
                        # except Exception as e_save_bs_debug:
                        #     logger.error(f"Error saving purebhakti BeautifulSoup FAIL debug HTML: {e_save_bs_debug}")
                    else:
                        logger.info(f"No more search results found on page {page_num} for '{query_term}' on purebhakti.com.")
                    break 

                for item_idx, item in enumerate(results_items):
                    if len(articles_found) >= num_articles:
                        break
                    
                    # Corrected selector for link and title within each item
                    link_tag = item.select_one("p.result__title > a.result__title-link") 
                    
                    if link_tag and link_tag.has_attr('href'):
                        href = link_tag['href']
                        title_text = ""
                        title_span = link_tag.select_one("span.result__title-text")
                        if title_span:
                            title_text = title_span.get_text(strip=True)
                        else: # Fallback if span isn't there for some reason
                            title_text = link_tag.get_text(strip=True)
                        
                        # Remove leading number like "1. " from title if it exists
                        if '.' in title_text[:4] and title_text.split('.',1)[0].isdigit():
                             title = title_text.split('.', 1)[-1].strip()
                        else:
                             title = title_text

                        full_url = urljoin(search_page_url, href) 
                        
                        articles_found.append({"url": full_url, "title": title})
                        logger.debug(f"Found purebhakti result #{len(articles_found)}: '{title}' - {full_url}")
                    else:
                        logger.warning(f"Purebhakti Item {item_idx+1}/Page {page_num} for '{query_term}': Could not find link/title using 'p.result__title > a.result__title-link'. Item HTML snippet: {str(item)[:250]}")
                
                if len(articles_found) >= num_articles:
                    logger.info(f"Reached target num_articles ({num_articles}) for purebhakti.com.")
                    # Since purebhakti pagination is not handled, we break after the first page anyway if target met or not.
                    break 
            
            logger.info(f"Finished scraping purebhakti.com for '{query_term}'. Found {len(articles_found)} articles.")
            return articles_found

        except Exception as e_main:
            logger.error(f"Critical exception in scrape_search_results_purebhakti for '{query_term}': {e_main}", exc_info=True)
            return []