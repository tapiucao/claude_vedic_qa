# Updated: src/web_scraper/cache_manager.py
"""
Web cache management for Vedic Knowledge AI.
Handles caching of scraped web content.
"""
import os
import json
import time
import hashlib
import logging
from typing import Optional, Dict, Any, List, Tuple # Added Tuple
from urllib.parse import urlparse
import io # For potential exception type

from ..config import WEB_CACHE_DIR, CACHE_EXPIRY

# Configure logging
logger = logging.getLogger(__name__)

class WebCacheManager:
    """Manager for cached web content."""

    def __init__(self, cache_dir: str = WEB_CACHE_DIR, expiry: int = CACHE_EXPIRY):
        """Initialize the cache manager."""
        self.cache_dir: str = cache_dir
        self.expiry: int = expiry
        self.metadata_file: str = os.path.join(cache_dir, "metadata.json")
        self.metadata: Dict[str, Any] = {"urls": {}, "stats": {"hits": 0, "misses": 0}} # Initialize metadata structure

        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

        # Load or initialize metadata
        self._load_metadata()

        logger.info(f"Initialized web cache at {cache_dir} (expiry: {expiry}s)")

    def _load_metadata(self) -> None:
        """Load cache metadata from file or initialize if not exists."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                    # Basic validation
                    if isinstance(loaded_data, dict) and "urls" in loaded_data and "stats" in loaded_data:
                         self.metadata = loaded_data
                    else:
                         logger.warning(f"Invalid metadata structure in {self.metadata_file}. Reinitializing.")
                         self._initialize_metadata() # Reinitialize if structure is wrong
            except (json.JSONDecodeError, IOError, UnicodeDecodeError) as e:
                logger.error(f"Error loading cache metadata from {self.metadata_file}: {str(e)}. Reinitializing.")
                self._initialize_metadata() # Reinitialize on error
        else:
            self._initialize_metadata()

    def _initialize_metadata(self) -> None:
        """Initialize metadata structure."""
        self.metadata = {"urls": {}, "stats": {"hits": 0, "misses": 0}}
        self._save_metadata() # Save the initial structure

    def _save_metadata(self) -> None:
        """Save cache metadata to file."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except (IOError, TypeError) as e: # Added TypeError for non-serializable data
            logger.error(f"Error saving cache metadata to {self.metadata_file}: {str(e)}")

    def _get_cache_path(self, url: str) -> str:
        """Get the cache file path for a URL."""
        # Generate a hash of the URL
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest() # Specify encoding

        # Get domain for organization
        try:
             domain = urlparse(url).netloc
             if not domain: # Handle cases like relative URLs if they ever sneak in
                  domain = "_nohost_"
        except ValueError: # Handle potential URL parsing errors
             domain = "_invalidurl_"


        # Create domain directory if it doesn't exist
        domain_dir = os.path.join(self.cache_dir, domain)
        try:
            os.makedirs(domain_dir, exist_ok=True)
        except OSError as e:
             logger.error(f"Failed to create cache subdirectory {domain_dir}: {e}. Using main cache directory.")
             domain_dir = self.cache_dir # Fallback to main cache dir

        # Return the full path
        return os.path.join(domain_dir, f"{url_hash}.html")

    def get_cached_content(self, url: str) -> Optional[str]:
        """Get cached content for a URL if it exists and is not expired."""
        cache_path = self._get_cache_path(url)

        # Check if URL is in metadata
        if url not in self.metadata.get("urls", {}):
            self.metadata.setdefault("stats", {"hits": 0, "misses": 0})["misses"] += 1
            # No need to save metadata on every miss, can be done less frequently if needed
            # self._save_metadata()
            return None

        # Check if cache file exists
        if not os.path.exists(cache_path):
            logger.warning(f"Cache metadata exists for {url} but file {cache_path} is missing. Cleaning up metadata.")
            del self.metadata["urls"][url] # Clean up inconsistent state
            self.metadata.setdefault("stats", {"hits": 0, "misses": 0})["misses"] += 1
            self._save_metadata()
            return None

        # Check if cache is expired
        cached_time = self.metadata["urls"][url].get("timestamp", 0)
        if time.time() - cached_time > self.expiry:
            logger.debug(f"Cache expired for {url}")
            self.metadata.setdefault("stats", {"hits": 0, "misses": 0})["misses"] += 1
            # No need to save metadata on every miss
            # self._save_metadata()
            # Consider removing the expired file here or rely on clear_expired
            return None

        # Read the cached content
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Retrieved cached content for {url}")
            self.metadata.setdefault("stats", {"hits": 0, "misses": 0})["hits"] += 1
            # No need to save metadata on every hit
            # self._save_metadata()
            return content
        except (IOError, UnicodeDecodeError) as e:
            logger.error(f"Error reading cache file for {url} ({cache_path}): {str(e)}")
            self.metadata.setdefault("stats", {"hits": 0, "misses": 0})["misses"] += 1
            # self._save_metadata()
            return None

    def cache_content(self, url: str, content: str) -> bool:
        """Cache content for a URL."""
        if not content:
            logger.warning(f"Not caching empty content for {url}")
            return False

        cache_path = self._get_cache_path(url)

        try:
            # Write content to cache file
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # Update metadata
            self.metadata.setdefault("urls", {})[url] = {
                "timestamp": time.time(),
                "cache_path": cache_path,
                "size": len(content.encode('utf-8')) # Get size in bytes
            }

            # Save metadata
            self._save_metadata()

            logger.info(f"Cached content for {url}")
            return True
        except (IOError, UnicodeEncodeError) as e:
            logger.error(f"Error caching content for {url} to {cache_path}: {str(e)}")
            # Attempt to clean up potentially corrupted file
            if os.path.exists(cache_path):
                 try: os.remove(cache_path)
                 except OSError: pass
            return False

    def clear_expired(self) -> int:
        """Clear expired cache entries and return count of cleared entries."""
        cleared_count = 0
        current_time = time.time()
        urls_to_remove: List[str] = []

        for url, info in self.metadata.get("urls", {}).items():
            cached_time = info.get("timestamp", 0)
            if current_time - cached_time > self.expiry:
                urls_to_remove.append(url)
                cache_path = info.get("cache_path")
                if cache_path and os.path.exists(cache_path):
                    try:
                        os.remove(cache_path)
                        cleared_count += 1
                    except OSError as e: # Catch OSError for file removal issues
                        logger.error(f"Error removing expired cache file {cache_path}: {str(e)}")

        # Remove from metadata after iterating
        if urls_to_remove:
             for url in urls_to_remove:
                 if url in self.metadata["urls"]:
                     del self.metadata["urls"][url]
             self._save_metadata() # Save changes after removal

        if cleared_count > 0:
             logger.info(f"Cleared {cleared_count} expired cache entries")
        return cleared_count

    def clear_all(self) -> int:
        """Clear all cache entries and return count of cleared entries."""
        cleared_count = 0

        # First, clear metadata
        urls_to_clear = list(self.metadata.get("urls", {}).keys())

        for url in urls_to_clear:
            info = self.metadata["urls"].get(url, {})
            cache_path = info.get("cache_path")

            if cache_path and os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                    cleared_count += 1
                except OSError as e:
                    logger.error(f"Error removing cache file {cache_path}: {str(e)}")

        # Reset metadata completely
        self._initialize_metadata() # This also saves the empty metadata

        logger.info(f"Cleared all cache entries ({cleared_count} files)")
        return cleared_count

    # Removed duplicate get_stats method here
    # Kept the second, more detailed one

    def _get_oldest_entry_age(self) -> Optional[float]:
        """Get the age in days of the oldest cache entry."""
        urls_metadata = self.metadata.get("urls", {})
        if not urls_metadata:
            return None

        current_time = time.time()
        try:
            oldest_time = min(
                info.get("timestamp", current_time)
                for info in urls_metadata.values() if isinstance(info, dict) # Ensure info is dict
            )
            return round((current_time - oldest_time) / (24 * 3600), 2)  # Convert to days
        except ValueError: # Handles case where timestamps might be missing or invalid
            return None

    def _get_newest_entry_age(self) -> Optional[float]:
        """Get the age in days of the newest cache entry."""
        urls_metadata = self.metadata.get("urls", {})
        if not urls_metadata:
            return None

        current_time = time.time()
        try:
            newest_time = max(
                info.get("timestamp", 0)
                for info in urls_metadata.values() if isinstance(info, dict) # Ensure info is dict
            )
            # Ensure newest_time is not 0 before calculating age
            return round((current_time - newest_time) / (24 * 3600), 2) if newest_time > 0 else 0.0
        except ValueError: # Handles case where timestamps might be missing or invalid
            return None


    def get_cached_urls_by_domain(self, domain: str) -> List[str]:
        """Get all cached URLs for a specific domain."""
        return [
            url for url in self.metadata.get("urls", {})
            if urlparse(url).netloc == domain
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache."""
        urls_metadata = self.metadata.get("urls", {})
        cache_stats = self.metadata.get("stats", {"hits": 0, "misses": 0})
        # Count total entries and size
        total_entries = len(urls_metadata)
        total_size = sum(info.get("size", 0) for info in urls_metadata.values() if isinstance(info, dict))

        # Count entries by domain
        domains: Dict[str, int] = {}
        for url in urls_metadata:
             try:
                 domain = urlparse(url).netloc
                 if domain:
                     domains[domain] = domains.get(domain, 0) + 1
             except ValueError:
                 logger.warning(f"Could not parse domain for cached URL: {url}")


        # Get hit rate
        hits = cache_stats.get("hits", 0)
        misses = cache_stats.get("misses", 0)
        hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0.0

        return {
            "total_entries": total_entries,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "domains_count": len(domains),
            "domain_entries": domains, # Renamed for clarity
            "hits": hits,
            "misses": misses,
            "hit_rate_percent": round(hit_rate * 100, 2), # Renamed for clarity
            "oldest_entry_days": self._get_oldest_entry_age(), # Renamed for clarity
            "newest_entry_days": self._get_newest_entry_age(), # Renamed for clarity
            "vedabase_entries": self._count_domain_entries("vedabase.io")
        }

    def _count_domain_entries(self, domain_to_count: str) -> int:
        """Count the number of cache entries for a specific domain."""
        count = 0
        for url in self.metadata.get("urls", {}):
            try:
                parsed_domain = urlparse(url).netloc
                if parsed_domain == domain_to_count:
                     count += 1
            except ValueError:
                continue # Skip unparseable URLs
        return count