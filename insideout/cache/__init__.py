"""Caching infrastructure for LLM calls and results."""

from insideout.cache.llm_cache import (
    setup_llm_cache,
    clear_llm_cache,
    get_cache_stats,
    is_cache_initialized
)

from insideout.cache.results_db import (
    ResultsDatabase,
    create_results_db
)

__all__ = [
    # LLM cache
    "setup_llm_cache",
    "clear_llm_cache",
    "get_cache_stats",
    "is_cache_initialized",
    # Results DB
    "ResultsDatabase",
    "create_results_db"
]
