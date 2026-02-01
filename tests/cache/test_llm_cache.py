"""
Tests for LLM caching system.
"""

import pytest
from pathlib import Path

from insideout.cache import (
    setup_llm_cache,
    clear_llm_cache,
    get_cache_stats,
    is_cache_initialized
)


class TestLLMCache:
    """Tests for LLM caching functions."""
    
    def test_setup_llm_cache(self, temp_cache_dir):
        """Test setting up LLM cache."""
        setup_llm_cache(temp_cache_dir, force_reinit=True)
        
        # Check that cache file was created
        cache_file = temp_cache_dir / "llm_cache.db"
        assert cache_file.exists()
        assert is_cache_initialized()
    
    def test_setup_llm_cache_disabled(self, temp_cache_dir):
        """Test that cache can be disabled."""
        setup_llm_cache(temp_cache_dir, enable=False, force_reinit=True)
        
        # Cache file should not be created
        cache_file = temp_cache_dir / "llm_cache.db"
        assert not cache_file.exists()
    
    def test_setup_llm_cache_custom_filename(self, temp_cache_dir):
        """Test cache with custom filename."""
        custom_name = "custom_cache.db"
        setup_llm_cache(temp_cache_dir, cache_filename=custom_name, force_reinit=True)
        
        cache_file = temp_cache_dir / custom_name
        assert cache_file.exists()
    
    def test_setup_llm_cache_creates_directory(self, temp_dir):
        """Test that cache setup creates directory if not exists."""
        cache_dir = temp_dir / "new_cache_dir"
        assert not cache_dir.exists()
        
        setup_llm_cache(cache_dir, force_reinit=True)
        
        assert cache_dir.exists()
        assert (cache_dir / "llm_cache.db").exists()
    
    def test_setup_llm_cache_nested_directory(self, temp_dir):
        """Test cache setup with nested directory."""
        nested_dir = temp_dir / "level1" / "level2" / "cache"
        
        setup_llm_cache(nested_dir, force_reinit=True)
        
        assert nested_dir.exists()
        assert (nested_dir / "llm_cache.db").exists()
    
    def test_clear_llm_cache(self, temp_cache_dir):
        """Test clearing cache."""
        # Create cache
        setup_llm_cache(temp_cache_dir, force_reinit=True)
        cache_file = temp_cache_dir / "llm_cache.db"
        assert cache_file.exists()
        
        # Clear cache
        clear_llm_cache(temp_cache_dir)
        assert not cache_file.exists()
    
    def test_clear_llm_cache_nonexistent(self, temp_cache_dir):
        """Test clearing cache that doesn't exist."""
        # Should not raise error
        clear_llm_cache(temp_cache_dir)
    
    def test_clear_llm_cache_custom_filename(self, temp_cache_dir):
        """Test clearing cache with custom filename."""
        custom_name = "my_cache.db"
        setup_llm_cache(temp_cache_dir, cache_filename=custom_name, force_reinit=True)
        
        cache_file = temp_cache_dir / custom_name
        assert cache_file.exists()
        
        clear_llm_cache(temp_cache_dir, cache_filename=custom_name)
        assert not cache_file.exists()
    
    def test_get_cache_stats_nonexistent(self, temp_cache_dir):
        """Test getting stats for nonexistent cache."""
        stats = get_cache_stats(temp_cache_dir)
        assert not stats["exists"]
        assert stats["size_mb"] == 0.0
        assert "llm_cache.db" in stats["path"]
    
    def test_get_cache_stats_exists(self, temp_cache_dir):
        """Test getting stats for existing cache."""
        setup_llm_cache(temp_cache_dir, force_reinit=True)
        
        stats = get_cache_stats(temp_cache_dir)
        assert stats["exists"]
        assert stats["size_mb"] >= 0.0
        assert "llm_cache.db" in stats["path"]
    
    def test_get_cache_stats_custom_filename(self, temp_cache_dir):
        """Test stats with custom filename."""
        custom_name = "test_cache.db"
        setup_llm_cache(temp_cache_dir, cache_filename=custom_name, force_reinit=True)
        
        stats = get_cache_stats(temp_cache_dir, cache_filename=custom_name)
        assert stats["exists"]
        assert custom_name in stats["path"]
    
    def test_is_cache_initialized_false(self):
        """Test cache initialization check when false."""
        # Reset state
        import insideout.cache.llm_cache as cache_module
        cache_module._cache_initialized = False
        
        assert not is_cache_initialized()
    
    def test_is_cache_initialized_true(self, temp_cache_dir):
        """Test cache initialization check when true."""
        import insideout.cache.llm_cache as cache_module
        cache_module._cache_initialized = False
        
        setup_llm_cache(temp_cache_dir, force_reinit=True)
        assert is_cache_initialized()


class TestLLMCacheEdgeCases:
    """Edge case tests for LLM cache."""
    
    def test_setup_cache_readonly_directory(self, temp_cache_dir):
        """Test setup in readonly directory (should fail gracefully)."""
        # Make directory readonly
        import os
        os.chmod(temp_cache_dir, 0o444)
        
        try:
            with pytest.raises(RuntimeError):
                setup_llm_cache(temp_cache_dir, force_reinit=True)
        finally:
            # Restore permissions for cleanup
            os.chmod(temp_cache_dir, 0o755)
    
    def test_cache_path_with_spaces(self, temp_dir):
        """Test cache path with spaces in name."""
        cache_dir = temp_dir / "cache with spaces"
        setup_llm_cache(cache_dir, force_reinit=True)
        
        assert cache_dir.exists()
        assert (cache_dir / "llm_cache.db").exists()
    
    def test_cache_multiple_setups_without_force(self, temp_cache_dir):
        """Test multiple setups without force reinit."""
        setup_llm_cache(temp_cache_dir, force_reinit=True)
        
        # Second setup should log warning but not fail
        setup_llm_cache(temp_cache_dir, force_reinit=False)
        
        # Cache should still exist
        assert (temp_cache_dir / "llm_cache.db").exists()
    
    def test_cache_disable_after_enable(self, temp_cache_dir):
        """Test disabling cache after enabling."""
        setup_llm_cache(temp_cache_dir, enable=True, force_reinit=True)
        assert is_cache_initialized()
        
        setup_llm_cache(temp_cache_dir, enable=False, force_reinit=True)
        assert not is_cache_initialized()
