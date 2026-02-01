"""
LLM caching system using LangChain's caching mechanisms.

Uses SQLite for persistent caching of LLM responses to:
- Reduce API costs
- Speed up repeated experiments
- Enable reproducible results
"""

import logging
from pathlib import Path
from typing import Optional

from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache


logger = logging.getLogger("insideout.cache")


_cache_initialized = False


def setup_llm_cache(
    cache_dir: Path,
    cache_filename: str = "llm_cache.db",
    enable: bool = True,
    force_reinit: bool = False
) -> None:
    """
    Set up LangChain's global LLM cache using SQLite.
    
    This function should be called once at the start of your application.
    After calling this, all LLM calls will automatically be cached.
    
    Args:
        cache_dir: Directory to store the cache database
        cache_filename: Name of the SQLite database file
        enable: Whether to enable caching (useful for debugging)
        force_reinit: Force reinitialization even if already initialized
    
    Examples:
        >>> from pathlib import Path
        >>> setup_llm_cache(Path("cache"))
        >>> # Now all LLM calls will be cached automatically
    """
    global _cache_initialized
    
    if not enable:
        logger.info("LLM caching is disabled")
        _cache_initialized = False
        return
    
    if _cache_initialized and not force_reinit:
        logger.warning("LLM cache already initialized, skipping")
        return
    
    # Create cache directory if it doesn't exist
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_path = cache_dir / cache_filename
    
    try:
        # Set up SQLite cache
        cache = SQLiteCache(database_path=str(cache_path))
        set_llm_cache(cache)
        
        _cache_initialized = True
        logger.info(f"LLM cache initialized at: {cache_path}")
        
    except Exception as e:
        logger.error(f"Failed to initialize LLM cache: {e}")
        raise RuntimeError(f"Failed to set up LLM cache: {e}") from e


def clear_llm_cache(
    cache_dir: Path,
    cache_filename: str = "llm_cache.db"
) -> None:
    """
    Clear the LLM cache by deleting the cache database file.
    
    Warning: This will delete all cached LLM responses!
    
    Args:
        cache_dir: Directory containing the cache database
        cache_filename: Name of the SQLite database file
    
    Examples:
        >>> clear_llm_cache(Path("cache"))
    """
    cache_path = Path(cache_dir) / cache_filename
    
    if cache_path.exists():
        try:
            cache_path.unlink()
            logger.info(f"Cleared LLM cache: {cache_path}")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            raise RuntimeError(f"Failed to clear cache: {e}") from e
    else:
        logger.warning(f"Cache file not found: {cache_path}")


def get_cache_stats(
    cache_dir: Path,
    cache_filename: str = "llm_cache.db"
) -> dict:
    """
    Get statistics about the cache database.
    
    Args:
        cache_dir: Directory containing the cache database
        cache_filename: Name of the SQLite database file
    
    Returns:
        Dictionary with cache statistics:
        - exists: bool - whether cache file exists
        - size_mb: float - size of cache file in MB
        - path: str - full path to cache file
    
    Examples:
        >>> stats = get_cache_stats(Path("cache"))
        >>> print(f"Cache size: {stats['size_mb']:.2f} MB")
    """
    cache_path = Path(cache_dir) / cache_filename
    
    stats = {
        "exists": cache_path.exists(),
        "path": str(cache_path),
        "size_mb": 0.0
    }
    
    if cache_path.exists():
        size_bytes = cache_path.stat().st_size
        stats["size_mb"] = size_bytes / (1024 * 1024)
    
    return stats


def is_cache_initialized() -> bool:
    """
    Check if the LLM cache has been initialized.
    
    Returns:
        True if cache is initialized, False otherwise
    """
    return _cache_initialized
