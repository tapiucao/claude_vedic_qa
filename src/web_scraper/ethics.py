"""
Ethical web scraping utilities for Vedic Knowledge AI.
Handles robots.txt parsing, rate limiting, and ethical scraping decisions.
"""
import logging
import re
import time
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse, urljoin
import requests
from typing import Dict, Optional, List

from ..config import REQUEST_DELAY

# Configure logging
logger = logging.getLogger(__name__)

# Cache for robots.txt parsers to avoid repeated fetches
ROBOTS_CACHE = {}

# Cache for scraping permissions
SCRAPING_PERMISSIONS_CACHE = {}

# Blacklisted patterns for ethical reasons
BLACKLISTED_PATTERNS = [
    r'\/private\/',
    r'\/login\/',
    r'\/auth\/',
    r'\/account\/',
    r'\/admin\/',
    r'\?password=',
    r'\?token=',
    r'\/logout\/',
]

# Patterns indicating content that should not be scraped
COPYRIGHT_PATTERNS = [
    r'(?i)prohibits?\s+(?:automated\s+)?(?:scraping|crawling|harvesting)',
    r'(?i)no\s+(?:automated\s+)?(?:scraping|crawling|harvesting)',
    r'(?i)(?:content|material)\s+may\s+not\s+be\s+(?:scraped|crawled|harvested|copied)',
    r'(?i)all\s+rights\s+reserved.*no\s+reproduction',
    r'(?i)not\s+permitted\s+to\s+copy\s+or\s+reproduce',
    r'(?i)copyright.*?all\s+rights\s+reserved'
]

def get_robots_parser(url: str) -> Optional[RobotFileParser]:
    """Get a robots.txt parser for a given URL, with caching."""
    # Parse the URL
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    robots_url = f"{base_url}/robots.txt"
    
    # Check cache first
    if robots_url in ROBOTS_CACHE:
        return ROBOTS_CACHE[robots_url]
    
    # Create a new parser
    rp = RobotFileParser()
    rp.set_url(robots_url)
    
    try:
        # Fetch and parse robots.txt
        rp.read()
        
        # Cache the parser
        ROBOTS_CACHE[robots_url] = rp
        logger.info(f"Fetched and cached robots.txt for {base_url}")
        
        return rp
    except Exception as e:
        logger.error(f"Error fetching robots.txt for {base_url}: {str(e)}")
        # Cache a permissive parser to avoid repeated failures
        empty_parser = RobotFileParser()
        ROBOTS_CACHE[robots_url] = empty_parser
        return empty_parser

def respect_robots_txt(url: str, user_agent: str = "*") -> bool:
    """Check if scraping is allowed by robots.txt."""
    # Check cache first
    cache_key = f"{url}|{user_agent}"
    if cache_key in SCRAPING_PERMISSIONS_CACHE:
        return SCRAPING_PERMISSIONS_CACHE[cache_key]
    
    try:
        # Get the parser
        rp = get_robots_parser(url)
        
        # Check if scraping is allowed
        allowed = rp.can_fetch(user_agent, url)
        
        # Cache the result
        SCRAPING_PERMISSIONS_CACHE[cache_key] = allowed
        
        if not allowed:
            logger.warning(f"Scraping not allowed by robots.txt for {url}")
        
        return allowed
    except Exception as e:
        logger.error(f"Error checking robots.txt for {url}: {str(e)}")
        # Default to allowed in case of errors
        return True

def is_blacklisted_url(url: str) -> bool:
    """Check if a URL matches blacklisted patterns for ethical reasons."""
    return any(re.search(pattern, url) for pattern in BLACKLISTED_PATTERNS)

def check_copyright_notices(html: str) -> bool:
    """Check if a page has explicit copyright notices that prohibit scraping."""
    return any(re.search(pattern, html) for pattern in COPYRIGHT_PATTERNS)

def is_scraping_allowed(url: str, html: str = None) -> bool:
    """Determine if scraping is ethically allowed for a given URL."""
    # Check robots.txt first
    if not respect_robots_txt(url):
        return False
    
    # Check if URL matches blacklisted patterns
    if is_blacklisted_url(url):
        logger.warning(f"URL {url} matches blacklisted pattern")
        return False
    
    # Check for copyright notices if HTML is provided
    if html and check_copyright_notices(html):
        logger.warning(f"Found copyright notice prohibiting scraping on {url}")
        return False
    
    return True

def extract_site_policies(html: str) -> Dict[str, bool]:
    """Extract site policies from HTML content."""
    policies = {
        "allows_scraping": True,
        "allows_republishing": True,
        "requires_attribution": False,
        "prohibits_commercial_use": False
    }
    
    # Check for common policy phrases
    if re.search(r'(?i)prohibits?\s+(?:automated\s+)?(?:scraping|crawling)', html):
        policies["allows_scraping"] = False
    
    if re.search(r'(?i)prohibits?\s+republish', html):
        policies["allows_republishing"] = False
    
    if re.search(r'(?i)requires?\s+attribution', html) or re.search(r'(?i)must\s+cite', html):
        policies["requires_attribution"] = True
    
    if re.search(r'(?i)non[\s-]commercial', html) or re.search(r'(?i)prohibits?\s+commercial', html):
        policies["prohibits_commercial_use"] = True
    
    return policies

def get_crawl_delay(url: str, user_agent: str = "*") -> Optional[float]:
    """Get the crawl delay specified in robots.txt."""
    try:
        rp = get_robots_parser(url)
        
        # Get the crawl delay
        crawl_delay = rp.crawl_delay(user_agent)
        
        if crawl_delay:
            logger.info(f"Crawl delay for {url}: {crawl_delay} seconds")
            return float(crawl_delay)
        
        return None
    except Exception as e:
        logger.error(f"Error getting crawl delay for {url}: {str(e)}")
        return None

def rate_limit(delay: float = REQUEST_DELAY) -> None:
    """Simple rate limiting function."""
    time.sleep(delay)

def check_domain_courtesy_pages(domain: str) -> bool:
    """Check for 'courtesy pages' like terms of service and about pages."""
    courtesy_paths = [
        "/robots.txt",
        "/terms",
        "/terms-of-service",
        "/terms-and-conditions",
        "/about",
        "/privacy",
        "/legal"
    ]
    
    parsed_domain = urlparse(domain)
    if not parsed_domain.scheme:
        domain = f"https://{domain}"
    
    for path in courtesy_paths:
        url = urljoin(domain, path)
        try:
            response = requests.head(url, timeout=5)
            if response.status_code == 200:
                logger.info(f"Found courtesy page at {url}")
                return True
        except Exception:
            continue
    
    return False

class EthicalScraper:
    """Helper class for tracking and maintaining ethical scraping behavior."""
    
    def __init__(self, base_delay: float = REQUEST_DELAY):
        """Initialize the ethical scraper."""
        self.base_delay = base_delay
        self.domain_last_request = {}  # Track last request time per domain
        self.domain_delays = {}  # Custom delays per domain
        
        # Initialize domain policies
        self.domain_policies = {}
    
    def set_domain_delay(self, domain: str, delay: float) -> None:
        """Set a custom delay for a specific domain."""
        self.domain_delays[domain] = delay
        logger.info(f"Set custom delay of {delay}s for domain {domain}")
    
    def wait_for_domain(self, url: str) -> None:
        """Wait an appropriate amount of time before making a request to a domain."""
        # Extract domain from URL
        domain = urlparse(url).netloc
        
        # Try to get crawl delay from robots.txt
        crawl_delay = get_crawl_delay(url)
        
        # Get the appropriate delay for this domain
        if crawl_delay:
            delay = crawl_delay
        else:
            delay = self.domain_delays.get(domain, self.base_delay)
        
        # Check if we need to wait
        current_time = time.time()
        if domain in self.domain_last_request:
            elapsed = current_time - self.domain_last_request[domain]
            if elapsed < delay:
                wait_time = delay - elapsed
                logger.debug(f"Rate limiting: waiting {wait_time:.2f}s for domain {domain}")
                time.sleep(wait_time)
        
        # Update last request time
        self.domain_last_request[domain] = time.time()
    
    def can_scrape(self, url: str, html: str = None) -> bool:
        """Check if scraping is allowed for this URL."""
        # Extract domain
        domain = urlparse(url).netloc
        
        # Check domain policies if we have scraped this domain before
        if domain in self.domain_policies:
            if not self.domain_policies[domain].get("allows_scraping", True):
                logger.warning(f"Domain policy prohibits scraping for {domain}")
                return False
        
        # Check standard ethical guidelines
        if not is_scraping_allowed(url, html):
            return False
        
        # If we have HTML content, extract and store domain policies
        if html and domain not in self.domain_policies:
            policies = extract_site_policies(html)
            self.domain_policies[domain] = policies
            
            if not policies.get("allows_scraping", True):
                logger.warning(f"Domain policy prohibits scraping for {domain}")
                return False
        
        return True
    
    def scrape_url(self, url: str, scrape_function, html: str = None):
        """Scrape a URL ethically."""
        if not self.can_scrape(url, html):
            logger.warning(f"Scraping not allowed for {url}")
            return None
        
        # Wait before scraping
        self.wait_for_domain(url)
        
        # Perform the scraping
        return scrape_function(url)
    
    def get_domain_policies(self) -> Dict[str, Dict[str, bool]]:
        """Get all domain policies."""
        return self.domain_policies
    
    def check_attribution_requirements(self, url: str, html: str = None) -> bool:
        """Check if attribution is required for a domain."""
        domain = urlparse(url).netloc
        
        # Check if we already have policies for this domain
        if domain in self.domain_policies:
            return self.domain_policies[domain].get("requires_attribution", False)
        
        # If we have HTML content, extract policies
        if html:
            policies = extract_site_policies(html)
            self.domain_policies[domain] = policies
            return policies.get("requires_attribution", False)
        
        # Default to requiring attribution to be safe
        return True
    
    def format_attribution(self, url: str, title: str = None) -> str:
        """Format attribution for a URL."""
        domain = urlparse(url).netloc
        
        if title:
            return f"{title} - {domain}"
        
        return domain
    
    def add_courtesy_header(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Add courtesy headers to identify the scraper."""
        courtesy_headers = headers.copy()
        
        # Add identifying information
        courtesy_headers["From"] = "bot@example.com"  # Replace with your contact
        courtesy_headers["User-Agent"] = "VedicKnowledgeAI/1.0 (+https://example.com/bot)"  # Replace with your bot info
        
        return courtesy_headers