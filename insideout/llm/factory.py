"""
LLM Factory for creating and configuring language models.

Supports multiple providers: GigaChat, OpenAI, and others.
Includes caching, retry logic, and error handling.
"""

import os
from typing import Optional, Dict, Any, Literal, Type
from pathlib import Path
import logging

from langchain_core.language_models import BaseChatModel
from langchain_gigachat import GigaChat
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


logger = logging.getLogger("insideout.llm")


LLMProvider = Literal["gigachat", "openai"]


class LLMFactory:
    """Factory for creating LLM instances with caching and configuration."""
    
    @staticmethod
    def create_llm(
        provider: LLMProvider,
        model_name: str,
        temperature: float = 0.0,
        max_retries: int = 3,
        request_timeout: Optional[float] = None,
        structured_output: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> BaseChatModel:
        """
        Create an LLM instance with the specified configuration.
        
        Args:
            provider: LLM provider name ("gigachat" or "openai")
            model_name: Specific model name/version
            temperature: Sampling temperature (default: 0.0 for deterministic output)
            max_retries: Maximum number of retries on API failures
            request_timeout: Request timeout in seconds (None for no timeout)
            structured_output: Pydantic model for structured output (recommended)
            **kwargs: Additional provider-specific arguments
        
        Returns:
            Configured LLM instance with optional structured output
        
        Raises:
            ValueError: If provider is not supported
            RuntimeError: If required API keys are missing
        
        Examples:
            >>> llm = LLMFactory.create_llm("gigachat", "GigaChat:latest")
            >>> llm = LLMFactory.create_llm("openai", "gpt-4", temperature=0.0)
        """
        logger.info(f"Creating LLM: provider={provider}, model={model_name}, temp={temperature}")
        
        if provider == "gigachat":
            llm = LLMFactory._create_gigachat(
                model_name=model_name,
                temperature=temperature,
                max_retries=max_retries,
                request_timeout=request_timeout,
                **kwargs
            )
        elif provider == "openai":
            llm = LLMFactory._create_openai(
                model_name=model_name,
                temperature=temperature,
                max_retries=max_retries,
                request_timeout=request_timeout,
                **kwargs
            )
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Supported providers: 'gigachat', 'openai'"
            )
        
        # Apply structured output if specified
        if structured_output is not None:
            logger.info(f"Applying structured output: {structured_output.__name__}")
            llm = llm.with_structured_output(structured_output)
        
        return llm
    
    @staticmethod
    def _create_gigachat(
        model_name: str,
        temperature: float = 0.0,
        max_retries: int = 3,
        request_timeout: Optional[float] = None,
        **kwargs
    ) -> GigaChat:
        """
        Create a GigaChat LLM instance.
        
        Requires GIGACHAT_CREDENTIALS environment variable.
        """
        # Check for API credentials
        credentials = os.getenv("GIGACHAT_CREDENTIALS")
        if not credentials:
            raise RuntimeError(
                "GIGACHAT_CREDENTIALS environment variable not found. "
                "Please set it in your .env file."
            )
        
        # Extract additional parameters
        verify_ssl_certs = kwargs.get("verify_ssl_certs", True)
        scope = kwargs.get("scope", "GIGACHAT_API_PERS")
        
        try:
            llm = GigaChat(
                model=model_name,
                credentials=credentials,
                temperature=temperature,
                max_retries=max_retries,
                timeout=request_timeout,
                verify_ssl_certs=verify_ssl_certs,
                scope=scope,
            )
            logger.info(f"Successfully created GigaChat LLM: {model_name}")
            return llm
        
        except Exception as e:
            logger.error(f"Failed to create GigaChat LLM: {e}")
            raise RuntimeError(f"Failed to initialize GigaChat: {e}") from e
    
    @staticmethod
    def _create_openai(
        model_name: str,
        temperature: float = 0.0,
        max_retries: int = 3,
        request_timeout: Optional[float] = None,
        **kwargs
    ) -> ChatOpenAI:
        """
        Create an OpenAI LLM instance.
        
        Requires OPENAI_API_KEY environment variable.
        """
        # Check for API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable not found. "
                "Please set it in your .env file."
            )
        
        # Extract additional parameters
        max_tokens = kwargs.get("max_tokens", None)
        
        try:
            llm = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                temperature=temperature,
                max_retries=max_retries,
                request_timeout=request_timeout,
                max_tokens=max_tokens,
            )
            logger.info(f"Successfully created OpenAI LLM: {model_name}")
            return llm
        
        except Exception as e:
            logger.error(f"Failed to create OpenAI LLM: {e}")
            raise RuntimeError(f"Failed to initialize OpenAI: {e}") from e
    
    @staticmethod
    def from_config(
        config: Dict[str, Any],
        provider: Optional[LLMProvider] = None
    ) -> BaseChatModel:
        """
        Create an LLM from a configuration dictionary.
        
        Args:
            config: Configuration dict with keys:
                - provider: str (required)
                - model: str (required)
                - temperature: float (optional, default: 0.0)
                - max_retries: int (optional, default: 3)
                - request_timeout: float (optional)
            provider: Override provider from config (optional)
        
        Returns:
            Configured LLM instance
        
        Examples:
            >>> config = {"provider": "openai", "model": "gpt-4", "temperature": 0.0}
            >>> llm = LLMFactory.from_config(config)
        """
        provider_name = provider or config.get("provider")
        model_name = config.get("model")
        
        if not provider_name:
            raise ValueError("Provider must be specified in config or as argument")
        if not model_name:
            raise ValueError("Model name must be specified in config")
        
        return LLMFactory.create_llm(
            provider=provider_name,
            model_name=model_name,
            temperature=config.get("temperature", 0.0),
            max_retries=config.get("max_retries", 3),
            request_timeout=config.get("request_timeout"),
            **{k: v for k, v in config.items() 
               if k not in ["provider", "model", "temperature", "max_retries", "request_timeout"]}
        )


def create_llm_from_settings(
    provider: LLMProvider,
    settings_dict: Dict[str, Any]
) -> BaseChatModel:
    """
    Convenience function to create LLM from settings.yaml structure.
    
    Args:
        provider: LLM provider name
        settings_dict: Settings dictionary (typically from YAML file)
    
    Returns:
        Configured LLM instance
    
    Examples:
        >>> import yaml
        >>> with open("config/settings.yaml") as f:
        ...     settings = yaml.safe_load(f)
        >>> llm = create_llm_from_settings("gigachat", settings["llm"]["gigachat"])
    """
    if "llm" in settings_dict and provider in settings_dict["llm"]:
        llm_config = settings_dict["llm"][provider]
    elif "model" in settings_dict:
        llm_config = settings_dict
    else:
        raise ValueError(
            f"Invalid settings structure. Expected 'llm.{provider}' or model config"
        )
    
    return LLMFactory.create_llm(
        provider=provider,
        model_name=llm_config["model"],
        temperature=llm_config.get("temperature", 0.0),
    )
