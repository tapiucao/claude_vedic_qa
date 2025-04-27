# Save this as src/web_scraper/scheduler.py

"""
Scheduled web scraping functionality for Vedic Knowledge AI.
Handles periodic scraping of trusted websites to keep knowledge base updated.
"""
import logging
import time
import threading
import schedule
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

from ..config import TRUSTED_WEBSITES, SCRAPING_INTERVAL
from .scraper import VedicWebScraper
from .ethics import respect_robots_txt, is_scraping_allowed

# Configure logging
logger = logging.getLogger(__name__)

class ScrapingScheduler:
    """Scheduler for periodic web scraping of trusted sources."""
    
    def __init__(
        self,
        vector_store,
        websites: List[str] = None,
        scraping_interval: int = SCRAPING_INTERVAL,
        dynamic_site_patterns: List[str] = None
    ):
        """Initialize the scraping scheduler."""
        self.vector_store = vector_store
        self.websites = websites or TRUSTED_WEBSITES
        self.scraping_interval = scraping_interval  # in seconds
        
        # Patterns for identifying dynamic sites (requiring JavaScript)
        self.dynamic_site_patterns = dynamic_site_patterns or [
            "react", "angular", "vue", "javascript", "js", 
            "spa", "app", "portal", "interactive"
        ]
        
        # Initialize scrapers
        self.web_scraper = VedicWebScraper()
        
        # Scheduling thread
        self.scheduler_thread = None
        self.is_running = False
        
        # Last scrape times
        self.last_scrape_times = {}
        
        logger.info(f"Initialized scraping scheduler with {len(self.websites)} websites")
    
    def _is_dynamic_site(self, url: str) -> bool:
        """Determine if a site is likely dynamic and requires JavaScript."""
        url_lower = url.lower()
        return any(pattern in url_lower for pattern in self.dynamic_site_patterns)
    
    def scrape_website(self, url: str) -> bool:
        """Scrape a single website and add to vector store."""
        logger.info(f"Starting scrape of website: {url}")
        
        # Check if scraping is allowed by robots.txt
        if not respect_robots_txt(url):
            logger.warning(f"Scraping not allowed by robots.txt for {url}")
            return False
        
        # Check if scraping is allowed by other ethics rules
        if not is_scraping_allowed(url):
            logger.warning(f"Scraping not allowed by ethics rules for {url}")
            return False
        
        try:
            # Choose the appropriate scraper
            documents = self.web_scraper.scrape_url(url)
            
            if not documents:
                logger.warning(f"No content scraped from {url}")
                return False
            
            # Add documents to vector store
            self.vector_store.add_documents(documents)
            
            # Update last scrape time
            self.last_scrape_times[url] = datetime.now()
            
            logger.info(f"Successfully scraped {len(documents)} chunks from {url}")
            return True
            
        except Exception as e:
            logger.error(f"Error scraping website {url}: {str(e)}")
            return False
    
    def scrape_all_websites(self) -> Dict[str, bool]:
        """Scrape all configured websites."""
        results = {}
        
        for url in self.websites:
            logger.info(f"Scheduling scrape of {url}")
            success = self.scrape_website(url)
            results[url] = success
        
        return results
    
    def _run_scheduler(self):
        """Run the scheduler in a loop."""
        while self.is_running:
            schedule.run_pending()
            time.sleep(1)
    
    def start(self, immediate: bool = False):
        """Start the scheduled scraping."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        # Schedule the scraping job
        schedule.every(self.scraping_interval).seconds.do(self.scrape_all_websites)
        
        # Flag as running
        self.is_running = True
        
        # Start the scheduler thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler)
        self.scheduler_thread.daemon = True  # Allow the thread to exit when the main program exits
        self.scheduler_thread.start()
        
        logger.info(f"Started scraping scheduler (interval: {self.scraping_interval}s)")
        
        # Run immediately if requested
        if immediate:
            logger.info("Running initial scrape immediately")
            self.scrape_all_websites()
    
    def stop(self):
        """Stop the scheduled scraping."""
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return
        
        # Clear all scheduled jobs
        schedule.clear()
        
        # Flag as not running
        self.is_running = False
        
        # Wait for the thread to finish
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
            self.scheduler_thread = None
        
        logger.info("Stopped scraping scheduler")
    
    def add_website(self, url: str, scrape_now: bool = False) -> bool:
        """Add a new website to the scraping list."""
        if url in self.websites:
            logger.warning(f"Website {url} is already in the scraping list")
            return False
        
        # Add to the list
        self.websites.append(url)
        
        logger.info(f"Added website {url} to scraping list")
        
        # Scrape now if requested
        if scrape_now:
            return self.scrape_website(url)
        
        return True
    
    def remove_website(self, url: str) -> bool:
        """Remove a website from the scraping list."""
        if url not in self.websites:
            logger.warning(f"Website {url} is not in the scraping list")
            return False
        
        # Remove from the list
        self.websites.remove(url)
        
        # Remove from last scrape times
        if url in self.last_scrape_times:
            del self.last_scrape_times[url]
        
        logger.info(f"Removed website {url} from scraping list")
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get the status of the scraping scheduler."""
        return {
            "is_running": self.is_running,
            "websites": self.websites,
            "scraping_interval": self.scraping_interval,
            "last_scrape_times": {
                url: str(time) for url, time in self.last_scrape_times.items()
            }
        }