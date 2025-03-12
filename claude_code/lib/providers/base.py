#!/usr/bin/env python3
# claude_code/lib/providers/base.py
"""Base provider interface for LLM integration."""

import abc
from typing import Dict, List, Generator, Optional, Any, Union


class BaseProvider(abc.ABC):
    """Abstract base class for LLM providers.
    
    This class defines the interface that all LLM providers must implement.
    Providers are responsible for:
    - Generating completions from LLMs
    - Counting tokens
    - Managing rate limits
    - Tracking costs
    """
    
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Get the name of the provider."""
        pass
    
    @property
    @abc.abstractmethod
    def available_models(self) -> List[str]:
        """Get a list of available models from this provider."""
        pass
    
    @property
    @abc.abstractmethod
    def current_model(self) -> str:
        """Get the currently selected model."""
        pass
    
    @abc.abstractmethod
    def set_model(self, model_name: str) -> None:
        """Set the current model.
        
        Args:
            model_name: The name of the model to use
            
        Raises:
            ValueError: If the model is not available
        """
        pass
    
    @abc.abstractmethod
    def generate_completion(self, 
                           messages: List[Dict[str, Any]], 
                           tools: Optional[List[Dict[str, Any]]] = None,
                           temperature: float = 0.0,
                           stream: bool = True,
                           reasoning_effort: Optional[float] = None,
                           web_search: bool = False) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """Generate a completion from the provider.
        
        Args:
            messages: List of message dictionaries
            tools: Optional list of tool dictionaries
            temperature: Model temperature (0-1)
            stream: Whether to stream the response
            reasoning_effort: Optional float (0-1) for controlling reasoning depth in supported models
            web_search: Whether to enable web search capability for models that support it
            
        Returns:
            If stream=True, returns a generator of response chunks
            If stream=False, returns the complete response
        """
        pass
    
    @abc.abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            The number of tokens in the text
        """
        pass
    
    @abc.abstractmethod
    def count_message_tokens(self, messages: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count tokens in a message list.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Dictionary with 'input' and 'output' token counts
        """
        pass
    
    @abc.abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.
        
        Returns:
            Dictionary with model information including:
            - context_window: Maximum context window size
            - input_cost_per_1k: Cost per 1K input tokens
            - output_cost_per_1k: Cost per 1K output tokens
            - capabilities: List of model capabilities
        """
        pass
    
    @property
    @abc.abstractmethod
    def cost_per_1k_tokens(self) -> Dict[str, float]:
        """Get cost per 1K tokens for input and output.
        
        Returns:
            Dictionary with 'input' and 'output' costs
        """
        pass
    
    @abc.abstractmethod
    def validate_api_key(self) -> bool:
        """Validate the API key.
        
        Returns:
            True if the API key is valid, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get rate limit information.
        
        Returns:
            Dictionary with rate limit information
        """
        pass