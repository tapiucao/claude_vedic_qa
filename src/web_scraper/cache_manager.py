# src/web_scraper/cache_manager.py
"""
Web cache management for Vedic Knowledge AI.
Handles caching of scraped web content with expiration and statistics.
"""
import os
import json
import time
import hashlib
import logging
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urlparse
import io

# Import config variables
from ..config import WEB_CACHE_DIR, CACHE_EXPIRY

# Configure logging
logger = logging.getLogger(__name__)

class WebCacheManager:
    """Manager for cached web content."""

    def __init__(self, cache_dir: str = WEB_CACHE_DIR, expiry: int = CACHE_EXPIRY):
        """Initialize the cache manager."""
        self.cache_dir: str = cache_dir
        # Ensure expiry is positive
        self.expiry: int = max(0, expiry) # Use max(0, expiry)
        self.metadata_file: str = os.path.join(cache_dir, "metadata.json")
        # Initialize metadata structure robustly
        self.metadata: Dict[str, Any] = {"urls": {}, "stats": {"hits": 0, "misses": 0}}

        # Create cache directory if it doesn't exist
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except OSError as e:
             logger.error(f"Failed to create cache directory {cache_dir}: {e}. Cache may not function correctly.")
             # Application might need to decide whether to proceed without cache

        # Load or initialize metadata
        self._load_metadata()

        logger.info(f"Initialized web cache at {cache_dir} (expiry: {self.expiry}s)")

    def _load_metadata(self) -> None:
        """Load cache metadata from file or initialize if not exists/corrupt."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                    # Validate basic structure
                    if (isinstance(loaded_data, dict) and
                        isinstance(loaded_data.get("urls"), dict) and
                        isinstance(loaded_data.get("stats"), dict) and
                        isinstance(loaded_data["stats"].get("hits"), int) and
                        isinstance(loaded_data["stats"].get("misses"), int)):
                        self.metadata = loaded_data
                        # Optional: Validate timestamps/sizes within urls dict if needed
                    else:
                         logger.warning(f"Invalid metadata structure in {self.metadata_file}. Reinitializing.")
                         self._initialize_metadata()
            except (json.JSONDecodeError, IOError, UnicodeDecodeError, TypeError) as e:
                logger.error(f"Error loading cache metadata from {self.metadata_file}: {str(e)}. Reinitializing.")
                self._initialize_metadata()
        else:
            self._initialize_metadata() # File doesn't exist, initialize

    def _initialize_metadata(self) -> None:
        """Initialize or reset metadata structure."""
        self.metadata = {"urls": {}, "stats": {"hits": 0, "misses": 0}}
        self._save_metadata() # Save the initial/reset structure

    def _save_metadata(self) -> None:
        """Save cache metadata to file atomically (if possible)."""
        temp_filepath = self.metadata_file + ".tmp"
        try:
            # Write to temporary file first
            with open(temp_filepath, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            # Atomically replace the old file with the new one
            os.replace(temp_filepath, self.metadata_file) # os.replace is atomic on most systems
        except (IOError, TypeError, OSError) as e:
            logger.error(f"Error saving cache metadata to {self.metadata_file}: {str(e)}")
            # Clean up temp file if it exists
            if os.path.exists(temp_filepath):
                 try: os.remove(temp_filepath)
                 except OSError: pass


    def _get_cache_path(self, url: str) -> Optional[str]:
        """Get the cache file path for a URL, returns None if URL is invalid."""
        try:
            # Generate a hash of the URL (use sha256 for less collision chance)
            url_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()

            # Get domain for organization
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            if not domain:
                # Handle relative URLs or invalid schemes if they occur
                logger.warning(f"Could not determine domain for URL: {url}")
                domain = "_invalid_or_relative_"

             # Sanitize domain for use as directory name (replace invalid chars)
            safe_domain = "".join(c if c.isalnum() or c in '-.' else '_' for c in domain)

            # Create domain directory if it doesn't exist
            domain_dir = os.path.join(self.cache_dir, safe_domain)
            # Check if cache_dir itself exists before creating subdirs
            if not os.path.isdir(self.cache_dir):
                 logger.error(f"Cache directory {self.cache_dir} does not exist. Cannot create cache path.")
                 return None
            os.makedirs(domain_dir, exist_ok=True)

            # Return the full path using the hash as filename
            return os.path.join(domain_dir, f"{url_hash}.html")

        except (ValueError, TypeError) as e: # Catch potential errors during hashing/parsing
             logger.error(f"Error generating cache path for URL '{url}': {e}")
             return None
        except OSError as e:
             logger.error(f"OS error creating cache directory for domain '{domain}': {e}")
             # Fallback to main cache dir? Or return None? Returning None is safer.
             return None


    def get_cached_content(self, url: str) -> Optional[str]:
        """Get cached content for a URL if it exists and is not expired."""
        # Safely get cache path
        cache_path = self._get_cache_path(url)
        if not cache_path:
            return None # Invalid URL or path generation error

        url_metadata = self.metadata.get("urls", {}).get(url)

        # Check if URL metadata exists
        if not url_metadata:
            self.metadata.setdefault("stats", {"hits": 0, "misses": 0})["misses"] += 1
            # Don't save metadata on every miss
            return None

        # Check if cache file exists physically
        if not os.path.exists(cache_path):
            logger.warning(f"Cache metadata exists for {url} but file {cache_path} is missing. Cleaning up metadata.")
            # Clean up inconsistent state from metadata
            if url in self.metadata.get("urls", {}):
                 del self.metadata["urls"][url]
                 self._save_metadata() # Save after cleanup
            self.metadata.setdefault("stats", {"hits": 0, "misses": 0})["misses"] += 1
            return None

        # Check if cache is expired
        cached_time = url_metadata.get("timestamp", 0)
        if self.expiry > 0 and (time.time() - cached_time > self.expiry): # Check expiry > 0
            logger.debug(f"Cache expired for {url} (timestamp: {cached_time}, expiry: {self.expiry}s)")
            self.metadata.setdefault("stats", {"hits": 0, "misses": 0})["misses"] += 1
            # Expired entry will be cleaned by clear_expired, no need to delete here
            return None

        # Read the cached content
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Check if content size matches metadata (optional integrity check)
            # expected_size = url_metadata.get("size")
            # actual_size = len(content.encode('utf-8'))
            # if expected_size is not None and expected_size != actual_size:
            #     logger.warning(f"Cache size mismatch for {url}. Expected {expected_size}, got {actual_size}. Returning content anyway.")

            logger.info(f"Retrieved valid cached content for {url}")
            self.metadata.setdefault("stats", {"hits": 0, "misses": 0})["hits"] += 1
            # Don't save metadata on every hit
            return content
        except (IOError, UnicodeDecodeError) as e:
            logger.error(f"Error reading cache file for {url} ({cache_path}): {str(e)}")
            self.metadata.setdefault("stats", {"hits": 0, "misses": 0})["misses"] += 1
            return None

    def cache_content(self, url: str, content: str) -> bool:
        """Cache content for a URL."""
        if not content:
            logger.warning(f"Attempted to cache empty content for {url}. Skipping.")
            return False

        # Safely get cache path
        cache_path = self._get_cache_path(url)
        if not cache_path:
            return False # Invalid URL or path generation error

        try:
            # Write content to cache file atomically (write to temp then rename)
            temp_cache_path = cache_path + ".tmp"
            with open(temp_cache_path, 'w', encoding='utf-8') as f:
                f.write(content)
            os.replace(temp_cache_path, cache_path) # Atomic rename

            # Update metadata
            current_time = time.time()
            content_size = len(content.encode('utf-8')) # Get size in bytes
            self.metadata.setdefault("urls", {})[url] = {
                "timestamp": current_time,
                "cache_path": cache_path, # Store relative or absolute? Absolute stored here.
                "size": content_size
            }

            # Save metadata after successful cache write
            self._save_metadata()

            logger.info(f"Cached content ({content_size} bytes) for {url}")
            return True
        except (IOError, UnicodeEncodeError, OSError) as e:
            logger.error(f"Error caching content for {url} to {cache_path}: {str(e)}")
            # Clean up potentially corrupted temp file
            if os.path.exists(temp_cache_path):
                 try: os.remove(temp_cache_path)
                 except OSError: pass
            return False

    def clear_expired(self) -> int:
        """Clear expired cache entries and return count of cleared entries."""
        if self.expiry <= 0:
            logger.info("Cache expiry is disabled (<= 0). No entries cleared.")
            return 0

        cleared_count = 0
        current_time = time.time()
        urls_to_remove: List[str] = []
        files_to_remove: List[str] = []

        urls_metadata = self.metadata.get("urls", {})
        # Iterate over a copy of items to allow deletion during iteration
        for url, info in list(urls_metadata.items()):
            # Ensure info is a dictionary and has a timestamp
            if not isinstance(info, dict): continue
            cached_time = info.get("timestamp", 0)

            if current_time - cached_time > self.expiry:
                urls_to_remove.append(url)
                # Use stored cache_path if available, otherwise regenerate it
                cache_path = info.get("cache_path") or self._get_cache_path(url)
                if cache_path: # Ensure path is valid
                     files_to_remove.append(cache_path)

        # Remove files first
        for file_path in files_to_remove:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    cleared_count += 1
                    logger.debug(f"Removed expired cache file: {file_path}")
                except OSError as e:
                    logger.error(f"Error removing expired cache file {file_path}: {str(e)}")

        # Remove entries from metadata if files were successfully targeted for removal
        if urls_to_remove:
             metadata_changed = False
             for url in urls_to_remove:
                 if url in self.metadata.get("urls", {}):
                     del self.metadata["urls"][url]
                     metadata_changed = True
             if metadata_changed:
                 self._save_metadata() # Save changes after removal

        if cleared_count > 0:
             logger.info(f"Cleared {cleared_count} expired cache entries.")
        else:
             logger.info("No expired cache entries found to clear.")

        return cleared_count

    def clear_all(self) -> int:
        """Clear ALL cache entries and metadata, returning count of files deleted."""
        cleared_count = 0
        logger.warning(f"Clearing ALL cache entries from {self.cache_dir}...")

        # Get list of all known cache files from metadata
        files_to_remove: List[str] = []
        for url, info in self.metadata.get("urls", {}).items():
             if isinstance(info, dict):
                  cache_path = info.get("cache_path") or self._get_cache_path(url)
                  if cache_path:
                      files_to_remove.append(cache_path)

        # Additionally, walk the cache directory to find potentially orphaned files
        # Be cautious with recursive delete, ensure it only targets expected files/dirs
        try:
            for root, dirs, files in os.walk(self.cache_dir, topdown=False): # topdown=False allows deleting dirs after files
                # Delete cache files (.html)
                for name in files:
                    if name.endswith('.html') or name.endswith('.tmp'): # Target cache files and temps
                        file_path = os.path.join(root, name)
                        if file_path not in files_to_remove: # Avoid double counting
                             files_to_remove.append(file_path)

                # Optionally delete empty domain directories (be careful!)
                # if root != self.cache_dir and not os.listdir(root): # Check if dir is empty
                #     try:
                #         os.rmdir(root)
                #         logger.debug(f"Removed empty cache directory: {root}")
                #     except OSError as e:
                #         logger.error(f"Error removing directory {root}: {e}")
        except Exception as e:
             logger.error(f"Error walking cache directory {self.cache_dir} during clear_all: {e}")

        # Remove identified files
        for file_path in set(files_to_remove): # Use set to ensure unique paths
             if os.path.exists(file_path):
                 try:
                     os.remove(file_path)
                     cleared_count += 1
                 except OSError as e:
                     logger.error(f"Error removing cache file {file_path} during clear_all: {str(e)}")

        # Reset metadata completely (this saves the empty metadata file)
        self._initialize_metadata()

        logger.info(f"Cleared all cache entries. Removed approximately {cleared_count} files.")
        return cleared_count

    def get_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the web cache."""
        urls_metadata = self.metadata.get("urls", {})
        cache_stats = self.metadata.get("stats", {"hits": 0, "misses": 0})

        total_entries = len(urls_metadata)
        total_size = 0
        domains: Dict[str, int] = {}
        valid_timestamps: List[float] = []

        for url, info in urls_metadata.items():
            if isinstance(info, dict):
                 total_size += info.get("size", 0)
                 timestamp = info.get("timestamp")
                 if timestamp: valid_timestamps.append(timestamp)
                 try:
                     domain = urlparse(url).netloc
                     if domain:
                         domains[domain] = domains.get(domain, 0) + 1
                 except ValueError:
                     logger.warning(f"Could not parse domain for cached URL: {url}")

        hits = cache_stats.get("hits", 0)
        misses = cache_stats.get("misses", 0)
        total_lookups = hits + misses
        hit_rate = hits / total_lookups if total_lookups > 0 else 0.0

        current_time = time.time()
        oldest_age_seconds = (current_time - min(valid_timestamps)) if valid_timestamps else None
        newest_age_seconds = (current_time - max(valid_timestamps)) if valid_timestamps else None

        return {
            "cache_directory": self.cache_dir,
            "metadata_file": self.metadata_file,
            "cache_expiry_seconds": self.expiry,
            "total_entries": total_entries,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "domains_count": len(domains),
            "domain_entries": domains, # Dict of domain: count
            "cache_hits": hits,
            "cache_misses": misses,
            "cache_total_lookups": total_lookups,
            "cache_hit_rate_percent": round(hit_rate * 100, 2),
            "oldest_entry_age_days": round(oldest_age_seconds / 86400, 2) if oldest_age_seconds is not None else None,
            "newest_entry_age_days": round(newest_age_seconds / 86400, 2) if newest_age_seconds is not None else None,
            # Make domain count generic
            # "vedabase_entries": self.count_entries_for_domain("vedabase.io")
        }

    def count_entries_for_domain(self, domain_to_count: str) -> int:
        """Counts the number of cache entries for a specific domain."""
        count = 0
        # Use the pre-calculated domain counts if available and accurate
        # Or iterate if needed
        for url in self.metadata.get("urls", {}):
            try:
                if urlparse(url).netloc == domain_to_count:
                     count += 1
            except ValueError:
                continue # Skip unparseable URLs
        return count

    def get_cached_urls_by_domain(self, domain: str) -> List[str]:
        """Get all cached URLs for a specific domain."""
        urls = []
        for url in self.metadata.get("urls", {}):
            try:
                if urlparse(url).netloc == domain:
                     urls.append(url)
            except ValueError:
                continue
        return urls