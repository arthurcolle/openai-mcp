#!/usr/bin/env python3
# claude_code/lib/providers/__init__.py
"""LLM provider module."""

import logging
import os
from typing import Dict, Type, Optional

from .base import BaseProvider
from .openai import OpenAIProvider

logger = logging.getLogger(__name__)

# Registry of provider classes
PROVIDER_REGISTRY: Dict[str, Type[BaseProvider]] = {
    "openai": OpenAIProvider,
}

# Try to import other providers if available
try:
    from .anthropic import AnthropicProvider
    PROVIDER_REGISTRY["anthropic"] = AnthropicProvider
except ImportError:
    logger.debug("Anthropic provider not available")

try:
    from .local import LocalProvider
    PROVIDER_REGISTRY["local"] = LocalProvider
except ImportError:
    logger.debug("Local provider not available")


def get_provider(name: Optional[str] = None, **kwargs) -> BaseProvider:
    """Get a provider instance by name.
    
    Args:
        name: Provider name, or None to use default provider
        **kwargs: Additional arguments to pass to the provider constructor
        
    Returns:
        Provider instance
        
    Raises:
        ValueError: If provider is not found
    """
    # If name is not specified, try to infer from environment
    if name is None:
        if os.environ.get("OPENAI_API_KEY"):
            name = "openai"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            name = "anthropic"
        else:
            # Default to OpenAI if nothing else is available
            name = "openai"
    
    if name.lower() not in PROVIDER_REGISTRY:
        raise ValueError(f"Provider {name} not found. Available providers: {', '.join(PROVIDER_REGISTRY.keys())}")
    
    provider_class = PROVIDER_REGISTRY[name.lower()]
    return provider_class(**kwargs)


def list_available_providers() -> Dict[str, Dict]:
    """List all available providers and their models.
    
    Returns:
        Dictionary mapping provider names to information about them
    """
    result = {}
    
    for name, provider_class in PROVIDER_REGISTRY.items():
        try:
            # Create a temporary instance to get model information
            # This might fail if API keys are not available
            instance = provider_class()
            result[name] = {
                "name": instance.name,
                "available": True,
                "models": instance.available_models,
                "current_model": instance.current_model
            }
        except Exception as e:
            # Provider is available but not configured correctly
            result[name] = {
                "name": name.capitalize(),
                "available": False,
                "error": str(e),
                "models": [],
                "current_model": None
            }
    
    return result