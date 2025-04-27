"""
Web cache management for Vedic Knowledge AI.
Handles caching of scraped web content.
"""
import os
import json
import time
import hashlib
import logging
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse

from ..config import WEB_CACHE_DIR, CACHE_EXPIRY

# Configure logging
logger = logging.getLogger(__name__)

class WebCacheManager:
    """Manager for cached web content."""
    
    def __init__(self, cache_dir: str = WEB_CACHE_DIR, expiry: int = CACHE_EXPIRY):
        """Initialize the cache manager."""
        self.cache_dir = cache_dir
        self.expiry = expiry
        self.metadata_file = os.path.join(cache_dir, "metadata.json")
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load or initialize metadata
        self._load_metadata()
        
        logger.info(f"Initialized web cache at {cache_dir} (expiry: {expiry}s)")
    
    def _load_metadata(self):
        """Load cache metadata from file or initialize if not exists."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.error(f"Error loading cache metadata: {str(e)}")
                self.metadata = {"urls": {}, "stats": {"hits": 0, "misses": 0}}
        else:
            self.metadata = {"urls": {}, "stats": {"hits": 0, "misses": 0}}
    
    def _save_metadata(self):
        """Save cache metadata to file."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache metadata: {str(e)}")
    
    def _get_cache_path(self, url: str) -> str:
        """Get the cache file path for a URL."""
        # Generate a hash of the URL
        url_hash = hashlib.md5(url.encode()).hexdigest()
        
        # Get domain for organization
        domain = urlparse(url).netloc
        
        # Create domain directory if it doesn't exist
        domain_dir = os.path.join(self.cache_dir, domain)
        os.makedirs(domain_dir, exist_ok=True)
        
        # Return the full path
        return os.path.join(domain_dir, f"{url_hash}.html")
    
    def get_cached_content(self, url: str) -> Optional[str]:
        """Get cached content for a URL if it exists and is not expired."""
        cache_path = self._get_cache_path(url)
        
        # Check if URL is in metadata
        if url not in self.metadata["urls"]:
            self.metadata["stats"]["misses"] += 1
            self._save_metadata()
            return None
        
        # Check if cache file exists
        if not os.path.exists(cache_path):
            self.metadata["stats"]["misses"] += 1
            self._save_metadata()
            return None
        
        # Check if cache is expired
        cached_time = self.metadata["urls"][url].get("timestamp", 0)
        if time.time() - cached_time > self.expiry:
            logger.debug(f"Cache expired for {url}")
            self.metadata["stats"]["misses"] += 1
            self._save_metadata()
            return None
        
        # Read the cached content
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Retrieved cached content for {url}")
            self.metadata["stats"]["hits"] += 1
            self._save_metadata()
            return content
        except Exception as e:
            logger.error(f"Error reading cache file for {url}: {str(e)}")
            self.metadata["stats"]["misses"] += 1
            self._save_metadata()
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
            self.metadata["urls"][url] = {
                "timestamp": time.time(),
                "cache_path": cache_path,
                "size": len(content)
            }
            
            # Save metadata
            self._save_metadata()
            
            logger.info(f"Cached content for {url}")
            return True
        except Exception as e:
            logger.error(f"Error caching content for {url}: {str(e)}")
            return False
    
    def clear_expired(self) -> int:
        """Clear expired cache entries and return count of cleared entries."""
        cleared_count = 0
        current_time = time.time()
        
        for url, info in list(self.metadata["urls"].items()):
            cached_time = info.get("timestamp", 0)
            if current_time - cached_time > self.expiry:
                cache_path = info.get("cache_path")
                if cache_path and os.path.exists(cache_path):
                    try:
                        os.remove(cache_path)
                        cleared_count += 1
                    except Exception as e:
                        logger.error(f"Error removing cache file {cache_path}: {str(e)}")
                
                # Remove from metadata
                del self.metadata["urls"][url]
        
        # Save updated metadata
        self._save_metadata()
        
        logger.info(f"Cleared {cleared_count} expired cache entries")
        return cleared_count
    
    def clear_all(self) -> int:
        """Clear all cache entries and return count of cleared entries."""
        cleared_count = 0
        
        # First, clear metadata
        urls_to_clear = list(self.metadata["urls"].keys())
        
        for url in urls_to_clear:
            info = self.metadata["urls"][url]
            cache_path = info.get("cache_path")
            
            if cache_path and os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                    cleared_count += 1
                except Exception as e:
                    logger.error(f"Error removing cache file {cache_path}: {str(e)}")
            
            # Remove from metadata
            del self.metadata["urls"][url]
        
        # Reset stats
        self.metadata["stats"] = {"hits": 0, "misses": 0}
        
        # Save updated metadata
        self._save_metadata()
        
        logger.info(f"Cleared all cache entries ({cleared_count} files)")
        return cleared_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache."""
        # Count total entries and size
        total_entries = len(self.metadata["urls"])
        total_size = sum(info.get("size", 0) for info in self.metadata["urls"].values())
        
        # Count entries by domain
        domains = {}
        for url in self.metadata["urls"]:
            domain = urlparse(url).netloc
            if domain in domains:
                domains[domain] += 1
            else:
                domains[domain] = 1
        
        # Get hit rate
        hits = self.metadata["stats"].get("hits", 0)
        misses = self.metadata["stats"].get("misses", 0)
        hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0
        
        return {
            "total_entries": total_entries,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "domains": domains,
            "hits": hits,
            "misses": misses,
            "hit_rate": round(hit_rate * 100, 2),
            "oldest_entry": self._get_oldest_entry_age(),
            "newest_entry": self._get_newest_entry_age()
        }
    
    def _get_oldest_entry_age(self) -> Optional[float]:
        """Get the age in days of the oldest cache entry."""
        if not self.metadata["urls"]:
            return None
        
        current_time = time.time()
        oldest_time = min(
            info.get("timestamp", current_time) 
            for info in self.metadata["urls"].values()
        )
        
        return round((current_time - oldest_time) / (24 * 3600), 2)  # Convert to days
    
    def _get_newest_entry_age(self) -> Optional[float]:
        """Get the age in days of the newest cache entry."""
        if not self.metadata["urls"]:
            return None
        
        current_time = time.time()
        newest_time = max(
            info.get("timestamp", 0) 
            for info in self.metadata["urls"].values()
        )
        
        return round((current_time - newest_time) / (24 * 3600), 2)  # Convert to days
    
    def get_cached_urls_by_domain(self, domain: str) -> List[str]:
        """Get all cached URLs for a specific domain."""
        return [
            url for url in self.metadata["urls"] 
            if urlparse(url).netloc == domain
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache."""
        # Count total entries and size
        total_entries = len(self.metadata["urls"])
        total_size = sum(info.get("size", 0) for info in self.metadata["urls"].values())
        
        # Count entries by domain
        domains = {}
        for url in self.metadata["urls"]:
            domain = urlparse(url).netloc
            if domain in domains:
                domains[domain] += 1
            else:
                domains[domain] = 1
        
        # Get hit rate
        hits = self.metadata["stats"].get("hits", 0)
        misses = self.metadata["stats"].get("misses", 0)
        hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0
        
        return {
            "total_entries": total_entries,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "domains": domains,
            "hits": hits,
            "misses": misses,
            "hit_rate": round(hit_rate * 100, 2),
            "oldest_entry": self._get_oldest_entry_age(),
            "newest_entry": self._get_newest_entry_age(),
            "vedabase_entries": self._count_domain_entries("vedabase.io")
        }
    
    def _count_domain_entries(self, domain: str) -> int:
        """Count the number of cache entries for a specific domain."""
        return sum(1 for url in self.metadata["urls"] if domain in url)