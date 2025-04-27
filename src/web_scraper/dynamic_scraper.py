"""
Dynamic web scraping for JavaScript-heavy sites.
Uses Selenium to handle JavaScript-rendered content for Vedic Knowledge AI.
"""
import logging
import time
from typing import Dict, List, Any, Optional
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from langchain.docstore.document import Document

from ..config import REQUEST_DELAY
from ..document_processor.text_splitter import VedicTextSplitter
from ..document_processor.sanskrit_processor import SanskritProcessor
from .scraper import VedicWebScraper

# Configure logging
logger = logging.getLogger(__name__)

class DynamicVedicScraper(VedicWebScraper):
    """Dynamic scraper for JavaScript-heavy websites using Selenium."""
    
    def __init__(self, cache_dir: str, request_delay: int = REQUEST_DELAY):
        """Initialize the dynamic scraper."""
        super().__init__(request_delay=request_delay)
        self.driver = None
        self.text_splitter = VedicTextSplitter()
        self.sanskrit_processor = SanskritProcessor()
        
        logger.info("Initialized dynamic web scraper")
    
    def _initialize_driver(self):
        """Initialize the Selenium WebDriver."""
        if self.driver:
            return
        
        try:
            # Configure Chrome options
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument(f"user-agent={self.headers['User-Agent']}")
            
            # Install or update ChromeDriver automatically
            service = Service(ChromeDriverManager().install())
            
            # Initialize the driver
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            logger.info("WebDriver initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing WebDriver: {str(e)}")
            raise
    
    def _close_driver(self):
        """Close the WebDriver if it's open."""
        if self.driver:
            try:
                self.driver.quit()
                self.driver = None
                logger.info("WebDriver closed")
            except Exception as e:
                logger.error(f"Error closing WebDriver: {str(e)}")
    
    def __del__(self):
        """Destructor to ensure WebDriver is closed."""
        self._close_driver()
    
    def fetch_url(self, url: str, wait_time: int = 10, scroll: bool = True) -> Optional[str]:
        """Fetch content from a URL using Selenium with JavaScript support."""
        # Respect rate limiting
        self._respect_rate_limit()
        
        # Initialize driver if needed
        if not self.driver:
            self._initialize_driver()
        
        try:
            # Navigate to URL
            self.driver.get(url)
            
            # Wait for page to load
            WebDriverWait(self.driver, wait_time).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Optional: Scroll to load lazy content
            if scroll:
                self._scroll_page()
            
            # Get the page source after JavaScript execution
            html = self.driver.page_source
            
            logger.info(f"Successfully fetched dynamic URL: {url}")
            return html
        except TimeoutException:
            logger.warning(f"Timeout while loading URL: {url}")
            return self.driver.page_source  # Return what we got so far
        except Exception as e:
            logger.error(f"Error fetching dynamic URL {url}: {str(e)}")
            return None
    
    def _scroll_page(self, scroll_pause_time: float = 1.0, max_scrolls: int = 5):
        """Scroll the page to trigger lazy loading."""
        try:
            # Get scroll height
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            
            for i in range(max_scrolls):
                # Scroll down
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                
                # Wait to load the page
                time.sleep(scroll_pause_time)
                
                # Calculate new scroll height and compare with last scroll height
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break  # No more content to load
                last_height = new_height
            
            # Scroll back to top
            self.driver.execute_script("window.scrollTo(0, 0);")
        except Exception as e:
            logger.error(f"Error during page scrolling: {str(e)}")
    
    def wait_for_element(self, selector: str, by: By = By.CSS_SELECTOR, wait_time: int = 10) -> bool:
        """Wait for a specific element to appear on the page."""
        if not self.driver:
            logger.error("WebDriver not initialized")
            return False
        
        try:
            WebDriverWait(self.driver, wait_time).until(
                EC.presence_of_element_located((by, selector))
            )
            return True
        except TimeoutException:
            logger.warning(f"Element '{selector}' not found within {wait_time} seconds")
            return False
        except Exception as e:
            logger.error(f"Error waiting for element: {str(e)}")
            return False
    
    def click_element(self, selector: str, by: By = By.CSS_SELECTOR, wait_time: int = 10) -> bool:
        """Click an element on the page."""
        if not self.driver:
            logger.error("WebDriver not initialized")
            return False
        
        try:
            # Wait for element to be clickable
            element = WebDriverWait(self.driver, wait_time).until(
                EC.element_to_be_clickable((by, selector))
            )
            # Click the element
            element.click()
            return True
        except Exception as e:
            logger.error(f"Error clicking element '{selector}': {str(e)}")
            return False
    
    def scrape_dynamic_url(self, url: str, wait_selectors: List[str] = None) -> List[Document]:
        """Scrape a dynamic URL and convert content to Document objects."""
        # Fetch the URL using the dynamic method
        html = self.fetch_url(url)
        if not html:
            logger.error(f"Failed to fetch dynamic URL: {url}")
            return []
        
        # Wait for specific selectors if provided
        if wait_selectors:
            for selector in wait_selectors:
                self.wait_for_element(selector)
        
        # Get the final HTML after all JavaScript has executed
        final_html = self.driver.page_source
        
        # Parse the HTML using the superclass method
        parsed_result = self.parse_html(final_html, url)
        if not parsed_result.get("success", False):
            logger.error(f"Failed to parse HTML from dynamic URL: {url}")
            return []
        
        # Create Documents from the parsed content
        text = parsed_result.get("text", "")
        metadata = parsed_result.get("metadata", {})
        
        # Split the text into chunks
        chunks = self.text_splitter.split_text(text, metadata)
        
        logger.info(f"Successfully scraped dynamic URL {url} into {len(chunks)} chunks")
        return chunks
    
    def extract_dynamic_links(self, url: str, link_selector: str = "a", same_domain_only: bool = True) -> List[str]:
        """Extract links from a dynamic page."""
        # Fetch the URL
        self.fetch_url(url)
        
        try:
            # Find all link elements
            link_elements = self.driver.find_elements(By.TAG_NAME, link_selector)
            
            # Extract href attributes
            links = []
            for element in link_elements:
                href = element.get_attribute("href")
                if href and not href.startswith("#") and not href.startswith("javascript:"):
                    links.append(href)
            
            # Filter by domain if requested
            if same_domain_only:
                domain = url.split("//", 1)[1].split("/", 1)[0]
                links = [link for link in links if domain in link]
            
            # Remove duplicates
            unique_links = list(set(links))
            
            logger.info(f"Extracted {len(unique_links)} unique links from dynamic page {url}")
            return unique_links
        except Exception as e:
            logger.error(f"Error extracting links from dynamic page {url}: {str(e)}")
            return []