"""
Basic web scraping functionality for Vedic Knowledge AI.
Handles fetching and parsing web content from trusted sources.
"""
import logging
import requests
import time
from typing import Dict, List, Any, Optional, Union
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from langchain.docstore.document import Document

from ..config import REQUEST_DELAY, WEB_CACHE_DIR
from ..document_processor.text_splitter import VedicTextSplitter
from ..document_processor.sanskrit_processor import SanskritProcessor
from .cache_manager import WebCacheManager
from .ethics import respect_robots_txt, is_scraping_allowed

# Configure logging
logger = logging.getLogger(__name__)

class VedicWebScraper:
    """Web scraper for Vedic and Gaudiya Vaishnava content."""
    
    def __init__(self, request_delay: int = REQUEST_DELAY, cache_dir: str = WEB_CACHE_DIR):
        """Initialize the scraper with configurable request delay."""
        self.request_delay = request_delay
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5'
        }
        
        # Initialize processors
        self.text_splitter = VedicTextSplitter()
        self.sanskrit_processor = SanskritProcessor()
        
        # Initialize cache manager
        self.cache_manager = WebCacheManager(cache_dir=cache_dir)
        
        # Track last request time for rate limiting
        self.last_request_time = 0
        
        logger.info(f"Initialized web scraper with request delay: {request_delay}s")
    
    def _respect_rate_limit(self):
        """Respect rate limiting by waiting if needed."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.request_delay:
            # Wait the remaining time
            wait_time = self.request_delay - time_since_last_request
            logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
            time.sleep(wait_time)
        
        # Update last request time
        self.last_request_time = time.time()
    
    def fetch_url(self, url: str, bypass_cache: bool = False) -> Optional[str]:
        """Fetch content from a URL with caching and rate limiting."""
        # Check if scraping is allowed
        if not respect_robots_txt(url):
            logger.warning(f"Scraping not allowed by robots.txt for {url}")
            return None
        
        # Check cache first if not bypassing
        if not bypass_cache:
            cached_content = self.cache_manager.get_cached_content(url)
            if cached_content:
                logger.info(f"Using cached content for {url}")
                return cached_content
        
        # Respect rate limiting
        self._respect_rate_limit()
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()  # Raise an exception for error status codes
            
            # Check if the content is HTML
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type:
                logger.warning(f"URL {url} is not HTML (Content-Type: {content_type})")
                return None
            
            # Get the content
            content = response.text
            
            # Cache the content
            self.cache_manager.cache_content(url, content)
            
            logger.info(f"Successfully fetched URL: {url}")
            return content
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching URL {url}: {str(e)}")
            return None
    
    def parse_html(self, html: str, url: str) -> Dict[str, Any]:
        """Parse HTML content and extract useful information."""
        if not html:
            return {"success": False, "error": "No HTML content"}
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract title
            title = soup.title.string if soup.title else "No title"
            
            # Extract main content - this is a simplified approach
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
            
            # Get main content (adjust selectors based on target websites)
            main_content = soup.find('main') or soup.find('article') or soup.find('div', {'class': ['content', 'main-content']})
            
            # If no specific content area found, use body with some exclusions
            if not main_content:
                main_content = soup.body
            
            # Extract text
            if main_content:
                text = main_content.get_text(separator='\n', strip=True)
                
                # Clean up text - remove excessive whitespace
                text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())
            else:
                text = "No content extracted"
            
            # Check if the content contains Sanskrit
            contains_sanskrit = self.sanskrit_processor.contains_sanskrit(text)
            
            # Process Sanskrit content if detected
            if contains_sanskrit:
                sanskrit_info = self.sanskrit_processor.process_document(text)
                sanskrit_terms = sanskrit_info.get("sanskrit_terms", [])
                term_definitions = sanskrit_info.get("term_definitions", {})
            else:
                sanskrit_terms = []
                term_definitions = {}
            
            # Extract metadata
            metadata = {
                "url": url,
                "title": title,
                "domain": urlparse(url).netloc,
                "type": "website",
                "contains_sanskrit": contains_sanskrit,
                "sanskrit_terms": sanskrit_terms,
                "term_definitions": term_definitions,
                "fetch_time": time.time()
            }
            
            return {
                "success": True,
                "text": text,
                "metadata": metadata,
                "title": title
            }
        except Exception as e:
            logger.error(f"Error parsing HTML from {url}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def scrape_url(self, url: str, bypass_cache: bool = False) -> List[Document]:
        """Scrape a URL and convert content to Document objects."""
        # Check if scraping is allowed by ethics
        if not is_scraping_allowed(url):
            logger.warning(f"Scraping not ethically allowed for {url}")
            return []
        
        # Fetch the URL
        html = self.fetch_url(url, bypass_cache=bypass_cache)
        if not html:
            logger.error(f"Failed to fetch URL: {url}")
            return []
        
        # Check for copyright notices that might prohibit scraping
        if not is_scraping_allowed(url, html):
            logger.warning(f"Content contains copyright notices prohibiting scraping: {url}")
            return []
        
        # Parse the HTML
        parsed_result = self.parse_html(html, url)
        if not parsed_result.get("success", False):
            logger.error(f"Failed to parse HTML from URL: {url}")
            return []
        
        # Create a Document from the parsed content
        text = parsed_result.get("text", "")
        metadata = parsed_result.get("metadata", {})
        
        # Split the text into chunks
        chunks = self.text_splitter.split_text(text, metadata)
        
        logger.info(f"Successfully scraped URL {url} into {len(chunks)} chunks")
        return chunks
    
    def extract_links(self, html: str, base_url: str, same_domain_only: bool = True) -> List[str]:
        """Extract links from HTML content with optional domain filtering."""
        if not html:
            return []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            base_domain = urlparse(base_url).netloc
            
            links = []
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                
                # Skip empty links, anchors, and javascript
                if not href or href.startswith('#') or href.startswith('javascript:'):
                    continue
                
                # Construct absolute URL
                absolute_url = urljoin(base_url, href)
                
                # Filter by domain if requested
                if same_domain_only:
                    url_domain = urlparse(absolute_url).netloc
                    if url_domain != base_domain:
                        continue
                
                links.append(absolute_url)
            
            # Remove duplicates
            unique_links = list(set(links))
            
            logger.info(f"Extracted {len(unique_links)} unique links from {base_url}")
            return unique_links
        except Exception as e:
            logger.error(f"Error extracting links from {base_url}: {str(e)}")
            return []
    
    def crawl(
        self, 
        start_url: str, 
        max_pages: int = 10, 
        same_domain_only: bool = True,
        bypass_cache: bool = False
    ) -> List[Document]:
        """Crawl a website starting from a URL, up to max_pages."""
        visited_urls = set()
        to_visit = [start_url]
        all_documents = []
        
        while to_visit and len(visited_urls) < max_pages:
            # Get next URL to visit
            current_url = to_visit.pop(0)
            
            # Skip if already visited
            if current_url in visited_urls:
                continue
            
            logger.info(f"Crawling URL: {current_url}")
            
            # Check if scraping is allowed
            if not is_scraping_allowed(current_url):
                logger.warning(f"Skipping {current_url} - scraping not allowed")
                visited_urls.add(current_url)
                continue
            
            # Fetch and process the URL
            html = self.fetch_url(current_url, bypass_cache=bypass_cache)
            if not html:
                visited_urls.add(current_url)
                continue
            
            # Check content for copyright notices
            if not is_scraping_allowed(current_url, html):
                logger.warning(f"Skipping {current_url} - content prohibits scraping")
                visited_urls.add(current_url)
                continue
            
            # Extract documents
            documents = self.scrape_url(current_url, bypass_cache=True)  # Already fetched, bypass cache
            all_documents.extend(documents)
            
            # Mark as visited
            visited_urls.add(current_url)
            
            # Extract more links to visit
            new_links = self.extract_links(html, current_url, same_domain_only)
            
            # Add new links to visit
            for link in new_links:
                if link not in visited_urls and link not in to_visit:
                    to_visit.append(link)
        
        logger.info(f"Crawling completed. Visited {len(visited_urls)} pages, extracted {len(all_documents)} document chunks")
        return all_documents
    
    def clear_cache(self) -> int:
        """Clear expired cache entries and return count of cleared entries."""
        return self.cache_manager.clear_expired()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache."""
        return self.cache_manager.get_stats()
    
    def lookup_sanskrit_term(self, term: str, bypass_cache: bool = False) -> Dict[str, Any]:
        """Look up a Sanskrit term on Vedabase and return the data."""
        logger.info(f"Looking up Sanskrit term: {term}")
        
        # Construct the search URL
        search_url = f"https://vedabase.io/en/search/synonyms/?original={term}"
        
        # Use the already initialized web scraper to fetch the page
        html = self.web_scraper.fetch_url(search_url, bypass_cache=bypass_cache)
        if not html:
            logger.error(f"Failed to fetch search results for term: {term}")
            return None
        
        # Parse the HTML using BeautifulSoup
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract term data
        term_data = {
            "term": term,
            "devanagari": None,
            "transliteration": None,
            "definition": None,
            "occurrences": [],
            "sources": []
        }
        
        # Find the main content
        main_content = soup.find('div', class_='content-area')
        if not main_content:
            logger.warning(f"Could not find main content for term: {term}")
            return term_data
        
        # Try to find the Sanskrit term in Devanagari
        devanagari_el = main_content.find('span', class_='sa')
        if devanagari_el:
            term_data["devanagari"] = devanagari_el.text.strip()
        
        # Find the definition/meaning
        definition_el = main_content.find('div', class_='meaning')
        if definition_el:
            term_data["definition"] = definition_el.text.strip()
        
        # Find occurrences
        occurrences = main_content.find_all('div', class_='hit-item')
        for occurrence in occurrences:
            # Get the verse reference
            ref_el = occurrence.find('a', class_='hit-link')
            if ref_el:
                reference = ref_el.text.strip()
                # Get the verse text
                verse_el = occurrence.find('div', class_='hit-text')
                if verse_el:
                    verse_text = verse_el.text.strip()
                    term_data["occurrences"].append({
                        "reference": reference,
                        "text": verse_text
                    })
                    # Add to sources if not already there
                    if reference not in term_data["sources"]:
                        term_data["sources"].append(reference)
        
        # Add to knowledge base if we have occurrences
        if term_data and term_data.get("occurrences"):
            self._add_term_to_knowledge_base(term_data)
        
        logger.info(f"Successfully looked up term: {term}")
        return term_data