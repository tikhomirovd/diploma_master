"""
Integration tests for LLM Factory with real API calls.

These tests require actual API credentials and will be skipped if not available.
Run with: pytest tests/llm/test_factory_integration.py -v
"""

import os
import pytest
from pydantic import BaseModel, Field

from insideout.llm import LLMFactory


# Check if API keys are available
HAS_GIGACHAT = bool(os.getenv("GIGACHAT_CREDENTIALS"))
HAS_OPENAI = bool(os.getenv("OPENAI_API_KEY"))


class SimpleOutput(BaseModel):
    """Simple structured output for testing."""
    emotion: str = Field(..., description="Detected emotion")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


@pytest.mark.skipif(not HAS_OPENAI, reason="OPENAI_API_KEY not set")
@pytest.mark.integration
class TestOpenAIIntegration:
    """Integration tests for OpenAI provider."""
    
    def test_create_and_invoke_openai(self):
        """Test creating OpenAI LLM and making a simple request."""
        llm = LLMFactory.create_llm(
            provider="openai",
            model_name="gpt-3.5-turbo",
            temperature=0.0
        )
        
        # Simple test message
        response = llm.invoke("Say 'test'")
        
        # Verify we got a response
        assert response is not None
        assert hasattr(response, 'content')
        assert len(response.content) > 0
    
    def test_openai_with_structured_output(self):
        """Test OpenAI with structured output."""
        llm = LLMFactory.create_llm(
            provider="openai",
            model_name="gpt-4-turbo-preview",  # Supports structured output
            temperature=0.0,
            structured_output=SimpleOutput
        )
        
        # Request emotion detection
        prompt = "The person said: 'I'm so happy today!'. Detect the emotion and confidence."
        response = llm.invoke(prompt)
        
        # Verify structured output
        assert isinstance(response, SimpleOutput)
        assert response.emotion is not None
        assert 0.0 <= response.confidence <= 1.0
    
    def test_openai_caching(self):
        """Test that caching works with OpenAI."""
        from insideout.cache import setup_llm_cache
        import tempfile
        from pathlib import Path
        
        cache_dir = Path(tempfile.mkdtemp())
        setup_llm_cache(cache_dir, force_reinit=True)
        
        try:
            llm = LLMFactory.create_llm(
                provider="openai",
                model_name="gpt-3.5-turbo",
                temperature=0.0
            )
            
            prompt = "Count to 3"
            
            # First call - should hit API
            response1 = llm.invoke(prompt)
            
            # Second call - should use cache
            response2 = llm.invoke(prompt)
            
            # Responses should be identical (cached)
            assert response1.content == response2.content
        finally:
            import shutil
            shutil.rmtree(cache_dir, ignore_errors=True)


@pytest.mark.skipif(not HAS_GIGACHAT, reason="GIGACHAT_CREDENTIALS not set")
@pytest.mark.integration
class TestGigaChatIntegration:
    """Integration tests for GigaChat provider."""
    
    def test_create_and_invoke_gigachat(self):
        """Test creating GigaChat LLM and making a simple request."""
        llm = LLMFactory.create_llm(
            provider="gigachat",
            model_name="GigaChat",
            temperature=0.0
        )
        
        # Simple test message in Russian
        response = llm.invoke("Скажи 'тест'")
        
        # Verify we got a response
        assert response is not None
        assert hasattr(response, 'content')
        assert len(response.content) > 0
    
    def test_gigachat_caching(self):
        """Test that caching works with GigaChat."""
        from insideout.cache import setup_llm_cache
        import tempfile
        from pathlib import Path
        
        cache_dir = Path(tempfile.mkdtemp())
        setup_llm_cache(cache_dir, force_reinit=True)
        
        try:
            llm = LLMFactory.create_llm(
                provider="gigachat",
                model_name="GigaChat",
                temperature=0.0
            )
            
            prompt = "Посчитай до трёх"
            
            # First call - should hit API
            response1 = llm.invoke(prompt)
            
            # Second call - should use cache
            response2 = llm.invoke(prompt)
            
            # Responses should be identical (cached)
            assert response1.content == response2.content
        finally:
            import shutil
            shutil.rmtree(cache_dir, ignore_errors=True)


@pytest.mark.skipif(not (HAS_OPENAI or HAS_GIGACHAT), reason="No API keys available")
@pytest.mark.integration
class TestFactoryIntegration:
    """General integration tests for factory."""
    
    def test_factory_from_config(self):
        """Test factory creation from config."""
        if HAS_OPENAI:
            config = {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "temperature": 0.0
            }
            
            llm = LLMFactory.from_config(config)
            response = llm.invoke("Say hello")
            
            assert response is not None
            assert len(response.content) > 0
        elif HAS_GIGACHAT:
            config = {
                "provider": "gigachat",
                "model": "GigaChat",
                "temperature": 0.0
            }
            
            llm = LLMFactory.from_config(config)
            response = llm.invoke("Скажи привет")
            
            assert response is not None
            assert len(response.content) > 0
