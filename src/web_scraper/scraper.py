# src/web_scraper/scraper.py
"""
Basic web scraping functionality for Vedic Knowledge AI.
Handles fetching and parsing web content from trusted sources.
"""
import logging
import requests
import time
from typing import Dict, List, Any, Optional, Union, Sequence # Added Sequence
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from langchain.docstore.document import Document

# Use relative imports assuming standard package structure
from ..config import REQUEST_DELAY, WEB_CACHE_DIR
from ..document_processor.text_splitter import VedicTextSplitter
from ..document_processor.sanskrit_processor import SanskritProcessor
from .cache_manager import WebCacheManager
from .ethics import respect_robots_txt, is_scraping_allowed, EthicalScraper

# Configure logging
logger = logging.getLogger(__name__)

class VedicWebScraper:
    """Web scraper for Vedic and Gaudiya Vaishnava content."""

    def __init__(self, request_delay: int = REQUEST_DELAY, cache_dir: str = WEB_CACHE_DIR):
        """Initialize the scraper with configurable request delay."""
        # Use the request_delay from config (which is now int)
        self.request_delay = request_delay
        # Use a shared requests session for potential connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36 VedicKnowledgeBot/1.0 (+http://example.com/botinfo)', # More descriptive User-Agent
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br', # Request compressed content
            'Connection': 'keep-alive',
        })

        # Initialize processors
        self.text_splitter = VedicTextSplitter()
        self.sanskrit_processor = SanskritProcessor()

        # Initialize cache manager
        self.cache_manager = WebCacheManager(cache_dir=cache_dir)

        # Initialize ethical scraper helper for rate limiting per domain
        self.ethical_helper = EthicalScraper(base_delay=self.request_delay)

        logger.info(f"Initialized web scraper with request delay: {self.request_delay}s")

    # Removed _respect_rate_limit, using ethical_helper.wait_for_domain now

    def fetch_url(self, url: str, bypass_cache: bool = False) -> Optional[str]:
        """Fetch content from a URL with caching, rate limiting, and ethical checks."""
        # Check if scraping is allowed by ethics rules before any network call
        if not is_scraping_allowed(url): # Basic check without HTML content
            logger.warning(f"Scraping disallowed by initial check for {url}")
            return None

        # Check cache first if not bypassing
        if not bypass_cache:
            cached_content = self.cache_manager.get_cached_content(url)
            if cached_content:
                logger.info(f"Using cached content for {url}")
                # Even if cached, we might want to re-check ethics if policies change
                # For simplicity here, assume cached content was obtained ethically
                return cached_content

        # Wait based on domain-specific rate limits
        self.ethical_helper.wait_for_domain(url)

        try:
            # Use the session for the request
            response = self.session.get(url, timeout=30)
            response.raise_for_status()  # Raise an exception for error status codes (4xx, 5xx)

            # Decode content correctly (requests usually handles this well with response.text)
            # Ensure UTF-8 if possible
            response.encoding = response.apparent_encoding or 'utf-8'
            html_content = response.text

            # Check content type AFTER fetching
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type:
                logger.warning(f"URL {url} did not return HTML (Content-Type: {content_type}). Skipping parse.")
                # Cache non-HTML content? Maybe not useful for text extraction.
                # self.cache_manager.cache_content(url, f"Non-HTML content: {content_type}")
                return None # Indicate failure to get HTML

            # Perform ethical check on fetched HTML content
            if not is_scraping_allowed(url, html_content):
                 logger.warning(f"Scraping disallowed by content check (e.g., copyright notice) for {url}")
                 # Cache the fact that it's disallowed? Optional.
                 return None

            # Cache the valid HTML content
            self.cache_manager.cache_content(url, html_content)

            logger.info(f"Successfully fetched and cached URL: {url}")
            return html_content

        except requests.exceptions.Timeout:
             logger.error(f"Timeout error fetching URL {url}")
             return None
        except requests.exceptions.ConnectionError as e:
             logger.error(f"Connection error fetching URL {url}: {e}")
             return None
        except requests.exceptions.HTTPError as e:
             logger.error(f"HTTP error fetching URL {url}: {e.response.status_code} {e.response.reason}")
             # Cache the error status? Useful for avoiding repeated fetches of broken links.
             # self.cache_manager.cache_content(url, f"HTTP Error: {e.response.status_code}")
             return None
        except requests.exceptions.RequestException as e:
            # Catch any other requests-related errors
            logger.error(f"Generic error fetching URL {url}: {str(e)}")
            return None
        except Exception as e:
             # Catch other potential errors (e.g., during encoding)
             logger.error(f"Unexpected error during fetch for {url}: {str(e)}", exc_info=True)
             return None

    def parse_html(self, html: str, url: str) -> Dict[str, Any]:
        """Parse HTML content and extract useful information."""
        if not html:
            return {"success": False, "error": "No HTML content provided"}

        try:
            soup = BeautifulSoup(html, 'lxml') # Use lxml for potentially better performance/robustness

            # Extract title
            title_tag = soup.find('title')
            title = title_tag.string.strip() if title_tag and title_tag.string else "No title found"

            # Extract main content - Refined approach
            main_content_tags = ['main', 'article', 'section[role="main"]', 'div[role="main"]',
                                 'div.content', 'div.main-content', 'div#content', 'div#main']
            main_content = None
            for tag_selector in main_content_tags:
                main_content = soup.select_one(tag_selector)
                if main_content:
                    logger.debug(f"Found main content using selector: {tag_selector}")
                    break

            # Remove unwanted elements (more comprehensive list)
            elements_to_remove = ['script', 'style', 'nav', 'footer', 'header', 'aside',
                                  'form', 'button', 'iframe', 'noscript', 'figure > figcaption',
                                  '.advertisement', '.ad', '.popup', '.cookie-consent',
                                  '.related-links', '.sidebar', '.social-share'] # Common class/id patterns

            if main_content: # If specific main content found, clean within it
                target_clean_area = main_content
            else: # Otherwise, clean the whole body but be cautious
                 target_clean_area = soup.body
                 logger.debug("No specific main content tag found, attempting cleanup on body.")


            if target_clean_area:
                for selector in elements_to_remove:
                    try:
                        for element in target_clean_area.select(selector):
                             element.decompose()
                    except Exception as decomp_err:
                         logger.warning(f"Error decomposing element with selector '{selector}': {decomp_err}")


                # Extract text using cleaner method
                text = target_clean_area.get_text(separator='\n', strip=True)

                # Further cleanup - remove excessive blank lines
                lines = [line.strip() for line in text.splitlines()]
                cleaned_lines = [line for line in lines if line] # Remove empty lines
                # Optionally, join with single newline, or double for paragraphs
                text = '\n\n'.join(cleaned_lines)
            else:
                text = "Could not extract main content text."
                logger.warning(f"Failed to find or process target clean area for {url}")


            # Analyze text for Sanskrit
            sanskrit_analysis = self.sanskrit_processor.process_document_metadata(text)

            # Extract other metadata (OpenGraph, etc. - optional)
            meta_description = soup.find('meta', attrs={'name': 'description'})
            og_title = soup.find('meta', property='og:title')
            # ... add more metadata extraction as needed

            metadata = {
                "url": url,
                "title": title,
                "source": urlparse(url).netloc, # Use domain as default source identifier
                "type": "website", # Indicate source type
                "description": meta_description['content'].strip() if meta_description else None,
                "og_title": og_title['content'].strip() if og_title else None,
                "fetch_time": time.time(),
                **sanskrit_analysis # Merge Sanskrit analysis results
            }
            # Remove None values from metadata for cleaner storage
            metadata = {k: v for k, v in metadata.items() if v is not None}


            return {
                "success": True,
                "text": text,
                "metadata": metadata,
                "title": title # Keep title accessible at top level too
            }
        except Exception as e:
            logger.error(f"Error parsing HTML from {url}: {str(e)}", exc_info=True)
            return {"success": False, "error": f"Parsing failed: {str(e)}"}

    def scrape_url(self, url: str, bypass_cache: bool = False) -> List[Document]:
        """Scrape a URL, parse HTML, and convert content to Document objects."""

        # Fetch the URL using the instance's fetch_url method
        html = self.fetch_url(url, bypass_cache=bypass_cache)
        if not html:
            # Fetch_url already logs errors
            return []

        # Parse the HTML
        parsed_result = self.parse_html(html, url)
        if not parsed_result.get("success", False):
            # Parse_html logs errors
            return []

        # Get text and metadata
        text = parsed_result.get("text", "")
        metadata = parsed_result.get("metadata", {})

        if not text or text == "Could not extract main content text.":
             logger.warning(f"No meaningful text extracted from {url}. Skipping document creation.")
             return []

        # Split the text into chunks using the instance's text_splitter
        # Pass the extracted metadata to be included in each chunk
        chunks = self.text_splitter.split_text(text, metadata=metadata)

        logger.info(f"Successfully scraped URL {url} into {len(chunks)} chunks")
        return chunks

    def extract_links(self, html: str, base_url: str, same_domain_only: bool = True) -> List[str]:
        """Extract valid, absolute links from HTML content with optional domain filtering."""
        if not html:
            return []

        try:
            soup = BeautifulSoup(html, 'lxml') # Use lxml
            base_domain = urlparse(base_url).netloc
            links = set() # Use a set for automatic deduplication

            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href'].strip()

                # Basic filtering of unwanted links
                if not href or href.startswith('#') or href.startswith('javascript:') or href.startswith('mailto:'):
                    continue

                try:
                    # Construct absolute URL robustly
                    absolute_url = urljoin(base_url, href)
                    parsed_absolute_url = urlparse(absolute_url)

                    # Check scheme and netloc validity
                    if not parsed_absolute_url.scheme in ['http', 'https'] or not parsed_absolute_url.netloc:
                         logger.debug(f"Skipping invalid or non-HTTP(S) URL: {absolute_url}")
                         continue

                    # Filter by domain if requested
                    if same_domain_only:
                        if parsed_absolute_url.netloc != base_domain:
                            continue

                    # Further ethical check: avoid blacklisted patterns
                    if is_blacklisted_url(absolute_url):
                         logger.debug(f"Skipping blacklisted URL pattern: {absolute_url}")
                         continue

                    links.add(absolute_url)

                except ValueError as ve: # Catch errors from urlparse/urljoin
                     logger.debug(f"Could not process href '{href}' relative to {base_url}: {ve}")
                     continue


            unique_links = sorted(list(links)) # Convert back to list and sort

            logger.info(f"Extracted {len(unique_links)} unique, valid links from {base_url}")
            return unique_links

        except Exception as e:
            logger.error(f"Error extracting links from HTML of {base_url}: {str(e)}", exc_info=True)
            return []

    def crawl(
        self,
        start_url: str,
        max_pages: int = 10,
        same_domain_only: bool = True,
        bypass_cache: bool = False
    ) -> List[Document]:
        """Crawl a website starting from a URL, respecting ethics and limits."""
        visited_urls = set()
        # Use a list as a queue for BFS (Breadth-First Search)
        to_visit_queue: List[str] = [start_url]
        all_documents: List[Document] = []

        logger.info(f"Starting crawl from {start_url} (max_pages={max_pages}, same_domain={same_domain_only})")

        while to_visit_queue and len(visited_urls) < max_pages:
            # Get next URL from the front of the queue
            current_url = to_visit_queue.pop(0)

            # Skip if already visited or queued excessively long ago (optional timeout)
            if current_url in visited_urls:
                continue

            logger.info(f"Visiting ({len(visited_urls) + 1}/{max_pages}): {current_url}")
            visited_urls.add(current_url) # Mark as visited early to handle redirects

            # Fetch, parse, and split (scrape_url handles ethics, caching, etc.)
            # Pass bypass_cache setting
            documents = self.scrape_url(current_url, bypass_cache=bypass_cache)

            if documents: # Only add if scraping was successful and yielded content
                all_documents.extend(documents)

                # Re-fetch HTML for link extraction ONLY if scrape_url didn't return it
                # (scrape_url currently fetches internally, so we need the HTML again)
                # This is slightly inefficient - could refactor scrape_url to return HTML too.
                html_for_links = self.fetch_url(current_url, bypass_cache=True) # Use cache if available now

                if html_for_links:
                    # Extract new links to visit
                    new_links = self.extract_links(html_for_links, current_url, same_domain_only)

                    # Add new, unvisited links to the queue
                    for link in new_links:
                        if link not in visited_urls and link not in to_visit_queue:
                            # Limit queue size? Optional.
                            to_visit_queue.append(link)
            else:
                 logger.warning(f"Scraping failed or yielded no documents for {current_url}, cannot extract links.")

        logger.info(f"Crawling finished. Visited {len(visited_urls)} pages, extracted {len(all_documents)} document chunks.")
        return all_documents

    def clear_cache(self) -> int:
        """Clear expired cache entries and return count of cleared entries."""
        return self.cache_manager.clear_expired()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache."""
        return self.cache_manager.get_stats()

    def lookup_sanskrit_term(self, term: str, bypass_cache: bool = False) -> Optional[Dict[str, Any]]:
        """Looks up a Sanskrit term on Vedabase.io and parses the result page."""
        logger.info(f"Looking up Sanskrit term: '{term}' on Vedabase.io")
        search_url = f"https://vedabase.io/en/search/synonyms/?original={quote_plus(term)}" # URL Encode term

        # Use the scraper's own fetch_url method correctly
        html = self.fetch_url(search_url, bypass_cache=bypass_cache)
        if not html:
            logger.error(f"Failed to fetch search results page for term: {term} from {search_url}")
            return None

        try:
            soup = BeautifulSoup(html, 'lxml')
            term_data: Dict[str, Any] = {
                "term": term, "url": search_url, "devanagari": None,
                "transliteration": None, "definition": None,
                "occurrences": [], "sources": set()
            }

            # Selectors might need updating if Vedabase changes layout
            term_header = soup.select_one('h1.r-title') # More specific selector
            main_content = soup.select_one('div.r-synonyms-details') # More specific

            if not term_header or not main_content: # Check if main elements are present
                 # Check for "not found" message specifically
                 not_found_el = soup.select_one("div.synonym-not-found, p.no-results") # Example selectors
                 if not_found_el or "not found" in html.lower():
                     logger.warning(f"Term '{term}' not found on Vedabase ({search_url}).")
                     term_data["sources"] = [] # Ensure sources is list even if empty
                     return term_data # Return data indicating not found
                 else:
                     logger.warning(f"Could not find expected content structure on Vedabase page for term: {term} ({search_url}). Page structure may have changed.")
                     term_data["sources"] = []
                     return term_data # Return empty data, but log warning


            # Extract Devanagari from header or specific span
            devanagari_el = term_header.select_one('span.sa') if term_header else None
            if devanagari_el: term_data["devanagari"] = devanagari_el.get_text(strip=True)
            # Transliteration might be part of the header text itself
            # term_data["transliteration"] = term_header.get_text(strip=True).replace(term_data["devanagari"] or '', '').strip()

            definition_el = main_content.select_one('div.meaning')
            if definition_el: term_data["definition"] = definition_el.get_text(separator="\n", strip=True)

            occurrence_items = main_content.select('div.hit-item') # Selector for occurrences
            for item in occurrence_items:
                ref_el = item.select_one('a.hit-link')
                text_el = item.select_one('div.hit-text')
                if ref_el and text_el:
                    reference = ref_el.get_text(strip=True)
                    verse_text = text_el.get_text(strip=True)
                    term_data["occurrences"].append({"reference": reference, "text": verse_text})
                    term_data["sources"].add(reference.split(',')[0].strip())

            term_data["sources"] = sorted(list(term_data["sources"]))

            # --- Define or remove the call to _add_term_to_knowledge_base ---
            # This method should likely exist in a higher-level application/manager class,
            # not within the scraper itself.
            # if term_data["occurrences"]:
            #     logger.debug(f"Term '{term}' found with occurrences. TODO: Implement addition to knowledge base.")
                # Example: self.knowledge_base_manager.add_term(term_data)

            logger.info(f"Successfully looked up and parsed term: {term}")
            return term_data

        except Exception as e:
             logger.error(f"Error parsing Vedabase page for term '{term}': {e}", exc_info=True)
             return None # Indicate failure