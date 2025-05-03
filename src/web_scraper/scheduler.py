# Updated: src/web_scraper/scheduler.py
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
from .dynamic_scraper import DynamicVedicScraper # Import Dynamic Scraper
from .ethics import respect_robots_txt, is_scraping_allowed
from ..knowledge_base.vector_store import VedicVectorStore # Ensure correct import path

# Configure logging
logger = logging.getLogger(__name__)

class ScrapingScheduler:
    """Scheduler for periodic web scraping of trusted sources."""

    def __init__(
        self,
        vector_store: VedicVectorStore, # Add type hint
        websites: Optional[List[str]] = None, # Use Optional
        scraping_interval: int = int(SCRAPING_INTERVAL), # Convert interval to int
        dynamic_site_patterns: Optional[List[str]] = None # Use Optional
    ):
        """Initialize the scraping scheduler."""
        self.vector_store: VedicVectorStore = vector_store
        self.websites: List[str] = websites if websites is not None else TRUSTED_WEBSITES
        self.scraping_interval: int = scraping_interval  # in seconds

        # Patterns for identifying dynamic sites (requiring JavaScript)
        self.dynamic_site_patterns: List[str] = dynamic_site_patterns or [
            "react", "angular", "vue", "javascript", "js",
            "spa", "app", "portal", "interactive", "aspx" # Added aspx as potential indicator
        ]

        # Initialize scrapers (consider initializing dynamically scraper lazily)
        self.web_scraper = VedicWebScraper()
        # Don't initialize dynamic scraper here to avoid starting browser immediately
        self._dynamic_scraper_instance: Optional[DynamicVedicScraper] = None

        # Scheduling thread
        self.scheduler_thread: Optional[threading.Thread] = None
        self.is_running: bool = False

        # Last scrape times
        self.last_scrape_times: Dict[str, datetime] = {}

        logger.info(f"Initialized scraping scheduler with {len(self.websites)} websites, interval {self.scraping_interval}s")

    # Lazy initialization of the dynamic scraper
    def _get_dynamic_scraper(self) -> DynamicVedicScraper:
        if self._dynamic_scraper_instance is None:
             logger.info("Initializing DynamicVedicScraper instance...")
             self._dynamic_scraper_instance = DynamicVedicScraper()
        return self._dynamic_scraper_instance

    def _shutdown_dynamic_scraper(self) -> None:
        """Safely closes the dynamic scraper's driver if it exists."""
        if self._dynamic_scraper_instance:
             logger.info("Shutting down DynamicVedicScraper instance...")
             self._dynamic_scraper_instance._close_driver() # Call the close method
             self._dynamic_scraper_instance = None


    def _is_dynamic_site(self, url: str) -> bool:
        """Determine if a site is likely dynamic and requires JavaScript based on URL patterns."""
        url_lower = url.lower()
        # Check patterns in URL path and query parameters
        return any(pattern in url_lower for pattern in self.dynamic_site_patterns)

    def scrape_website(self, url: str) -> bool:
        """Scrape a single website and add to vector store, choosing the appropriate scraper."""
        logger.info(f"Starting scrape task for website: {url}")

        # Check if scraping is allowed by robots.txt and other ethics rules
        # (These checks are also done inside the scraper's scrape_url, but checking early saves resources)
        if not respect_robots_txt(url) or not is_scraping_allowed(url):
            logger.warning(f"Scraping not allowed by ethics rules/robots.txt for {url}. Skipping.")
            return False

        documents = None
        scraper_used = "static"
        try:
            # Choose the appropriate scraper
            if self._is_dynamic_site(url):
                logger.info(f"URL {url} matches dynamic pattern, using DynamicVedicScraper.")
                scraper_used = "dynamic"
                dynamic_scraper = self._get_dynamic_scraper()
                # Dynamic scraper's scrape_url handles fetching, parsing, splitting
                documents = dynamic_scraper.scrape_url(url)
            else:
                logger.info(f"URL {url} does not match dynamic pattern, using VedicWebScraper.")
                scraper_used = "static"
                # Static scraper's scrape_url handles fetching, parsing, splitting
                documents = self.web_scraper.scrape_url(url)


            if not documents:
                logger.warning(f"No content scraped from {url} using {scraper_used} scraper.")
                return False

            # Add documents to vector store
            if self.vector_store:
                 logger.info(f"Adding {len(documents)} document chunks from {url} to vector store...")
                 self.vector_store.add_documents(documents) # Assuming add_documents handles errors
                 logger.info(f"Successfully added documents from {url} to vector store.")
            else:
                 logger.warning("Vector store not available, scraped documents are not stored.")


            # Update last scrape time
            self.last_scrape_times[url] = datetime.now()

            logger.info(f"Successfully completed scrape task for {url} ({len(documents)} chunks)")
            return True

        except Exception as e:
            # Log detailed error including which scraper was used
            logger.error(f"Error scraping website {url} using {scraper_used} scraper: {str(e)}", exc_info=True) # Log traceback
            return False
        # No finally block needed here to close dynamic scraper, manage its lifecycle elsewhere (e.g., on stop)

    def scrape_all_websites(self) -> Dict[str, bool]:
        """Scrape all configured websites."""
        logger.info("Starting scheduled run: scrape_all_websites")
        results: Dict[str, bool] = {}
        start_time = time.time()

        for url in self.websites:
            # Check if enough time has passed since the last scrape for this specific URL
            # This provides more granular control than the main schedule interval alone
            last_scraped = self.last_scrape_times.get(url)
            if last_scraped and (datetime.now() - last_scraped).total_seconds() < self.scraping_interval:
                 logger.debug(f"Skipping {url}, scraped too recently ({last_scraped.strftime('%Y-%m-%d %H:%M:%S')}).")
                 continue

            logger.info(f"Initiating scrape for {url}")
            success = self.scrape_website(url)
            results[url] = success
            # Add a small delay between scraping different websites, even if rate limits handled per domain
            time.sleep(1) # Small courtesy delay

        end_time = time.time()
        logger.info(f"Finished scheduled run: scrape_all_websites. Duration: {end_time - start_time:.2f}s. Results: {results}")
        # Consider shutting down the dynamic scraper driver after a full run if memory is a concern
        # self._shutdown_dynamic_scraper()
        return results

    def _run_scheduler(self):
        """Run the scheduler in a loop."""
        logger.info("Scheduler thread started.")
        while self.is_running:
            try:
                schedule.run_pending()
            except Exception as e:
                 logger.error(f"Error during schedule run: {e}", exc_info=True)
            # Sleep for a short interval to avoid busy-waiting
            # Use a longer sleep if schedule interval is large
            sleep_interval = min(60, self.scraping_interval / 10) if self.scraping_interval > 0 else 60
            time.sleep(max(1, sleep_interval)) # Sleep at least 1 second
        logger.info("Scheduler thread finished.")


    def start(self, immediate: bool = False):
        """Start the scheduled scraping."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return

        # Clear any previous schedule just in case
        schedule.clear()

        # Schedule the scraping job
        if self.scraping_interval > 0:
             schedule.every(self.scraping_interval).seconds.do(self.scrape_all_websites)
             logger.info(f"Scheduled 'scrape_all_websites' to run every {self.scraping_interval} seconds.")
        else:
             logger.warning("Scraping interval is 0 or negative. Scheduling disabled.")


        # Flag as running
        self.is_running = True

        # Start the scheduler thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, name="ScrapingSchedulerThread")
        self.scheduler_thread.daemon = True  # Allow the thread to exit when the main program exits
        self.scheduler_thread.start()

        logger.info("Started scraping scheduler thread.")

        # Run immediately if requested
        if immediate:
            logger.info("Running initial scrape immediately...")
            # Run in a separate thread to avoid blocking the caller
            initial_scrape_thread = threading.Thread(target=self.scrape_all_websites, name="InitialScrapeThread")
            initial_scrape_thread.start()


    def stop(self):
        """Stop the scheduled scraping."""
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return

        logger.info("Stopping scraping scheduler...")

        # Flag as not running to stop the scheduler loop
        self.is_running = False

        # Clear all scheduled jobs
        schedule.clear()
        logger.info("Cleared scheduled jobs.")

        # Wait for the scheduler thread to finish
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            logger.info("Waiting for scheduler thread to finish...")
            self.scheduler_thread.join(timeout=10) # Wait for up to 10 seconds
            if self.scheduler_thread.is_alive():
                 logger.warning("Scheduler thread did not finish within timeout.")
            else:
                 logger.info("Scheduler thread finished.")
            self.scheduler_thread = None

        # Shutdown the dynamic scraper's browser if it's running
        self._shutdown_dynamic_scraper()

        logger.info("Scraping scheduler stopped.")

    def add_website(self, url: str, scrape_now: bool = False) -> bool:
        """Add a new website to the scraping list."""
        if url in self.websites:
            logger.warning(f"Website {url} is already in the scraping list")
            return False

        # Add to the list
        self.websites.append(url)
        logger.info(f"Added website {url} to scraping list (Total: {len(self.websites)})")

        # Scrape now if requested (run in a separate thread)
        if scrape_now:
            logger.info(f"Initiating immediate scrape for newly added website: {url}")
            scrape_thread = threading.Thread(target=self.scrape_website, args=(url,), name=f"ScrapeNow_{url[:20]}")
            scrape_thread.start()

        return True

    def remove_website(self, url: str) -> bool:
        """Remove a website from the scraping list."""
        if url not in self.websites:
            logger.warning(f"Website {url} is not in the scraping list")
            return False

        try:
            # Remove from the list
            self.websites.remove(url)

            # Remove from last scrape times
            if url in self.last_scrape_times:
                del self.last_scrape_times[url]

            logger.info(f"Removed website {url} from scraping list (Remaining: {len(self.websites)})")
            return True
        except ValueError: # Should not happen if check passed, but for safety
             logger.warning(f"Could not remove {url}, possibly already removed.")
             return False


    def get_status(self) -> Dict[str, Any]:
        """Get the status of the scraping scheduler."""
        return {
            "is_running": self.is_running,
            "website_count": len(self.websites),
            "websites": self.websites, # Maybe truncate if list is very long for status display
            "scraping_interval_seconds": self.scraping_interval,
            "last_scrape_times": {
                url: time.strftime('%Y-%m-%d %H:%M:%S') for url, time in self.last_scrape_times.items()
            },
             "dynamic_scraper_active": self._dynamic_scraper_instance is not None and self._dynamic_scraper_instance.driver is not None
        }