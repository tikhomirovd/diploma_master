"""
Unit tests for LLM Factory.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from pydantic import BaseModel, Field

from insideout.llm import LLMFactory


class TestStructuredOutput(BaseModel):
    """Test Pydantic model for structured output."""
    predicted_label: str = Field(..., description="Predicted label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence")


class TestLLMFactory:
    """Tests for LLMFactory class."""
    
    def test_unsupported_provider_raises_error(self):
        """Test that unsupported provider raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            LLMFactory.create_llm("unsupported_provider", "model-name")
    
    def test_empty_provider_raises_error(self):
        """Test that empty provider string raises error."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            LLMFactory.create_llm("", "model-name")
    
    def test_empty_model_name(self):
        """Test behavior with empty model name."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            with patch("insideout.llm.factory.ChatOpenAI") as mock_openai:
                mock_instance = MagicMock()
                mock_openai.return_value = mock_instance
                
                # Should still create LLM, provider might handle empty model
                llm = LLMFactory.create_llm("openai", "")
                assert llm is not None
    
    @patch("insideout.llm.factory.GigaChat")
    def test_create_gigachat_success(self, mock_gigachat):
        """Test successful GigaChat creation."""
        with patch.dict(os.environ, {"GIGACHAT_CREDENTIALS": "test_credentials"}):
            mock_instance = MagicMock()
            mock_gigachat.return_value = mock_instance
            
            llm = LLMFactory.create_llm(
                provider="gigachat",
                model_name="GigaChat:latest",
                temperature=0.0
            )
            
            mock_gigachat.assert_called_once()
            call_kwargs = mock_gigachat.call_args[1]
            assert call_kwargs["model"] == "GigaChat:latest"
            assert call_kwargs["temperature"] == 0.0
            assert call_kwargs["credentials"] == "test_credentials"
            assert llm == mock_instance
    
    @patch("insideout.llm.factory.GigaChat")
    def test_create_gigachat_with_custom_params(self, mock_gigachat):
        """Test GigaChat with custom parameters."""
        with patch.dict(os.environ, {"GIGACHAT_CREDENTIALS": "test_creds"}):
            mock_instance = MagicMock()
            mock_gigachat.return_value = mock_instance
            
            llm = LLMFactory.create_llm(
                provider="gigachat",
                model_name="GigaChat-Pro",
                temperature=0.7,
                max_retries=5,
                request_timeout=60.0,
                verify_ssl_certs=False,
                scope="GIGACHAT_API_CORP"
            )
            
            call_kwargs = mock_gigachat.call_args[1]
            assert call_kwargs["temperature"] == 0.7
            assert call_kwargs["max_retries"] == 5
            assert call_kwargs["timeout"] == 60.0
            assert call_kwargs["verify_ssl_certs"] is False
            assert call_kwargs["scope"] == "GIGACHAT_API_CORP"
    
    def test_create_gigachat_missing_credentials(self):
        """Test that missing credentials raises RuntimeError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="GIGACHAT_CREDENTIALS"):
                LLMFactory.create_llm("gigachat", "GigaChat:latest")
    
    @patch("insideout.llm.factory.GigaChat")
    def test_create_gigachat_initialization_error(self, mock_gigachat):
        """Test handling of GigaChat initialization errors."""
        with patch.dict(os.environ, {"GIGACHAT_CREDENTIALS": "test"}):
            mock_gigachat.side_effect = Exception("Connection failed")
            
            with pytest.raises(RuntimeError, match="Failed to initialize GigaChat"):
                LLMFactory.create_llm("gigachat", "GigaChat:latest")
    
    @patch("insideout.llm.factory.ChatOpenAI")
    def test_create_openai_success(self, mock_openai):
        """Test successful OpenAI creation."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance
            
            llm = LLMFactory.create_llm(
                provider="openai",
                model_name="gpt-4",
                temperature=0.5
            )
            
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["model"] == "gpt-4"
            assert call_kwargs["temperature"] == 0.5
            assert call_kwargs["api_key"] == "test_key"
            assert llm == mock_instance
    
    @patch("insideout.llm.factory.ChatOpenAI")
    def test_create_openai_with_max_tokens(self, mock_openai):
        """Test OpenAI with max_tokens parameter."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance
            
            llm = LLMFactory.create_llm(
                provider="openai",
                model_name="gpt-3.5-turbo",
                temperature=0.0,
                max_tokens=500
            )
            
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["max_tokens"] == 500
    
    @patch("insideout.llm.factory.ChatOpenAI")
    def test_create_openai_extreme_temperature(self, mock_openai):
        """Test OpenAI with extreme temperature values."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance
            
            # Very low temperature
            llm = LLMFactory.create_llm("openai", "gpt-4", temperature=0.0)
            assert mock_openai.call_args[1]["temperature"] == 0.0
            
            # Very high temperature
            llm = LLMFactory.create_llm("openai", "gpt-4", temperature=2.0)
            assert mock_openai.call_args[1]["temperature"] == 2.0
    
    def test_create_openai_missing_api_key(self):
        """Test that missing API key raises RuntimeError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
                LLMFactory.create_llm("openai", "gpt-4")
    
    @patch("insideout.llm.factory.ChatOpenAI")
    def test_create_openai_initialization_error(self, mock_openai):
        """Test handling of OpenAI initialization errors."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            mock_openai.side_effect = Exception("API error")
            
            with pytest.raises(RuntimeError, match="Failed to initialize OpenAI"):
                LLMFactory.create_llm("openai", "gpt-4")
    
    @patch("insideout.llm.factory.ChatOpenAI")
    def test_structured_output(self, mock_openai):
        """Test LLM creation with structured output."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            mock_instance = MagicMock()
            mock_structured = MagicMock()
            mock_instance.with_structured_output.return_value = mock_structured
            mock_openai.return_value = mock_instance
            
            llm = LLMFactory.create_llm(
                provider="openai",
                model_name="gpt-4",
                structured_output=TestStructuredOutput
            )
            
            # Verify with_structured_output was called
            mock_instance.with_structured_output.assert_called_once_with(TestStructuredOutput)
            assert llm == mock_structured
    
    @patch("insideout.llm.factory.ChatOpenAI")
    def test_from_config(self, mock_openai):
        """Test creating LLM from config dict."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance
            
            config = {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_retries": 5
            }
            
            llm = LLMFactory.from_config(config)
            
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["model"] == "gpt-3.5-turbo"
            assert call_kwargs["temperature"] == 0.7
            assert call_kwargs["max_retries"] == 5
    
    def test_from_config_missing_provider(self):
        """Test that missing provider raises ValueError."""
        config = {"model": "gpt-4"}
        
        with pytest.raises(ValueError, match="Provider must be specified"):
            LLMFactory.from_config(config)
    
    def test_from_config_missing_model(self):
        """Test that missing model raises ValueError."""
        config = {"provider": "openai"}
        
        with pytest.raises(ValueError, match="Model name must be specified"):
            LLMFactory.from_config(config)
    
    def test_from_config_empty_config(self):
        """Test from_config with empty config dict."""
        with pytest.raises(ValueError):
            LLMFactory.from_config({})
    
    @patch("insideout.llm.factory.ChatOpenAI")
    def test_from_config_with_provider_override(self, mock_openai):
        """Test overriding provider from config."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance
            
            config = {
                "provider": "gigachat",  # This should be overridden
                "model": "gpt-4"
            }
            
            llm = LLMFactory.from_config(config, provider="openai")
            
            # Verify OpenAI was used, not GigaChat
            mock_openai.assert_called_once()
    
    @patch("insideout.llm.factory.ChatOpenAI")
    def test_from_config_with_extra_params(self, mock_openai):
        """Test from_config with extra parameters."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance
            
            config = {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.5,
                "max_tokens": 1000,
                "extra_param": "value"
            }
            
            llm = LLMFactory.from_config(config)
            
            # Extra params should be passed through
            call_kwargs = mock_openai.call_args[1]
            assert "max_tokens" in call_kwargs


class TestLLMFactoryEdgeCases:
    """Edge case tests for LLM Factory."""
    
    @patch("insideout.llm.factory.ChatOpenAI")
    def test_zero_max_retries(self, mock_openai):
        """Test with zero max retries."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance
            
            llm = LLMFactory.create_llm(
                "openai",
                "gpt-4",
                max_retries=0
            )
            
            assert mock_openai.call_args[1]["max_retries"] == 0
    
    @patch("insideout.llm.factory.ChatOpenAI")
    def test_negative_temperature(self, mock_openai):
        """Test with negative temperature (provider should handle validation)."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance
            
            # Factory should pass through, provider validates
            llm = LLMFactory.create_llm("openai", "gpt-4", temperature=-0.5)
            assert mock_openai.call_args[1]["temperature"] == -0.5
    
    @patch("insideout.llm.factory.ChatOpenAI")
    def test_very_long_model_name(self, mock_openai):
        """Test with very long model name."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance
            
            long_name = "a" * 1000
            llm = LLMFactory.create_llm("openai", long_name)
            assert mock_openai.call_args[1]["model"] == long_name
    
    @patch("insideout.llm.factory.ChatOpenAI")
    def test_special_characters_in_model_name(self, mock_openai):
        """Test model name with special characters."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance
            
            special_name = "model-name_v1.2.3:latest"
            llm = LLMFactory.create_llm("openai", special_name)
            assert mock_openai.call_args[1]["model"] == special_name
