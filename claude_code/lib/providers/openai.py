#!/usr/bin/env python3
# claude_code/lib/providers/openai.py
"""OpenAI provider implementation."""

import os
from typing import Dict, List, Generator, Optional, Any, Union
import time
import logging
import json

import tiktoken
from openai import OpenAI, RateLimitError, APIError

from .base import BaseProvider

logger = logging.getLogger(__name__)

# Model information including context window and pricing
MODEL_INFO = {
    "gpt-3.5-turbo": {
        "context_window": 16385,
        "input_cost_per_1k": 0.0015,
        "output_cost_per_1k": 0.002,
        "capabilities": ["function_calling", "json_mode"],
    },
    "gpt-4o": {
        "context_window": 128000,
        "input_cost_per_1k": 0.005,
        "output_cost_per_1k": 0.015,
        "capabilities": ["function_calling", "json_mode", "vision"],
    },
    "gpt-4-turbo": {
        "context_window": 128000, 
        "input_cost_per_1k": 0.01,
        "output_cost_per_1k": 0.03,
        "capabilities": ["function_calling", "json_mode", "vision"],
    },
    "gpt-4": {
        "context_window": 8192,
        "input_cost_per_1k": 0.03,
        "output_cost_per_1k": 0.06,
        "capabilities": ["function_calling", "json_mode"],
    },
}

DEFAULT_MODEL = "gpt-4o"


class OpenAIProvider(BaseProvider):
    """OpenAI API provider implementation."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize the OpenAI provider.
        
        Args:
            api_key: OpenAI API key. If None, will use OPENAI_API_KEY environment variable
            model: Model to use. If None, will use DEFAULT_MODEL
        """
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key.")
        
        self._client = OpenAI(api_key=self._api_key)
        self._model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
        
        if self._model not in MODEL_INFO:
            logger.warning(f"Unknown model: {self._model}. Using {DEFAULT_MODEL} instead.")
            self._model = DEFAULT_MODEL
            
        # Cache for tokenizers
        self._tokenizers = {}
    
    @property
    def name(self) -> str:
        return "OpenAI"
    
    @property
    def available_models(self) -> List[str]:
        return list(MODEL_INFO.keys())
    
    @property
    def current_model(self) -> str:
        return self._model
    
    def set_model(self, model_name: str) -> None:
        if model_name not in MODEL_INFO:
            raise ValueError(f"Unknown model: {model_name}. Available models: {', '.join(self.available_models)}")
        self._model = model_name
    
    def generate_completion(self, 
                           messages: List[Dict[str, Any]], 
                           tools: Optional[List[Dict[str, Any]]] = None,
                           temperature: float = 0.0,
                           stream: bool = True) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """Generate a completion from OpenAI.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            tools: Optional list of tool dictionaries
            temperature: Model temperature (0-1)
            stream: Whether to stream the response
            
        Returns:
            If stream=True, returns a generator of response chunks
            If stream=False, returns the complete response
        """
        try:
            # Convert tools to OpenAI format if provided
            api_tools = None
            if tools:
                api_tools = []
                for tool in tools:
                    api_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool["description"],
                            "parameters": tool["parameters"]
                        }
                    })
            
            # Make the API call
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=api_tools,
                temperature=temperature,
                stream=stream
            )
            
            # Handle streaming and non-streaming responses
            if stream:
                return self._process_streaming_response(response)
            else:
                return {
                    "content": response.choices[0].message.content,
                    "tool_calls": response.choices[0].message.tool_calls,
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }
                
        except RateLimitError as e:
            logger.error(f"Rate limit exceeded: {str(e)}")
            raise
        except APIError as e:
            logger.error(f"API error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}")
            raise
    
    def _process_streaming_response(self, response):
        """Process a streaming response from OpenAI."""
        current_tool_calls = []
        tool_call_chunks = {}
        
        for chunk in response:
            # Create a result chunk to yield
            result_chunk = {
                "content": None,
                "tool_calls": None,
                "delta": True
            }
            
            # Process content
            delta = chunk.choices[0].delta
            if delta.content:
                result_chunk["content"] = delta.content
            
            # Process tool calls
            if delta.tool_calls:
                result_chunk["tool_calls"] = []
                
                for tool_call_delta in delta.tool_calls:
                    # Initialize tool call in chunks dictionary if new
                    idx = tool_call_delta.index
                    if idx not in tool_call_chunks:
                        tool_call_chunks[idx] = {
                            "id": "",
                            "function": {"name": "", "arguments": ""}
                        }
                    
                    # Update tool call data
                    if tool_call_delta.id:
                        tool_call_chunks[idx]["id"] = tool_call_delta.id
                    
                    if tool_call_delta.function:
                        if tool_call_delta.function.name:
                            tool_call_chunks[idx]["function"]["name"] = tool_call_delta.function.name
                        
                        if tool_call_delta.function.arguments:
                            tool_call_chunks[idx]["function"]["arguments"] += tool_call_delta.function.arguments
                    
                    # Add current state to result
                    result_chunk["tool_calls"].append(tool_call_chunks[idx])
            
            # Yield the chunk
            yield result_chunk
        
        # Final yield with complete tool calls
        if tool_call_chunks:
            complete_calls = list(tool_call_chunks.values())
            yield {
                "content": None,
                "tool_calls": complete_calls,
                "delta": False,
                "finish_reason": "tool_calls"
            }
    
    def _get_tokenizer(self, model: str = None) -> Any:
        """Get a tokenizer for the specified model."""
        model = model or self._model
        
        if model not in self._tokenizers:
            try:
                encoder_name = "cl100k_base" if model.startswith("gpt-4") or model.startswith("gpt-3.5") else "p50k_base"
                self._tokenizers[model] = tiktoken.get_encoding(encoder_name)
            except Exception as e:
                logger.error(f"Error loading tokenizer for {model}: {str(e)}")
                raise
        
        return self._tokenizers[model]
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        tokenizer = self._get_tokenizer()
        return len(tokenizer.encode(text))
    
    def count_message_tokens(self, messages: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count tokens in a message list."""
        # Simple approximation - in production, would need to match OpenAI's tokenization exactly
        prompt_tokens = 0
        
        for message in messages:
            # Add tokens for message role
            prompt_tokens += 4  # ~4 tokens for role
            
            # Count content tokens
            if "content" in message and message["content"]:
                prompt_tokens += self.count_tokens(message["content"])
            
            # Count tokens from any tool calls or tool results
            if "tool_calls" in message and message["tool_calls"]:
                for tool_call in message["tool_calls"]:
                    prompt_tokens += 4  # ~4 tokens for tool call overhead
                    prompt_tokens += self.count_tokens(tool_call.get("function", {}).get("name", ""))
                    prompt_tokens += self.count_tokens(tool_call.get("function", {}).get("arguments", ""))
            
            if "name" in message and message["name"]:
                prompt_tokens += self.count_tokens(message["name"])
                
            if "tool_call_id" in message and message["tool_call_id"]:
                prompt_tokens += 10  # ~10 tokens for tool_call_id and overhead
        
        # Add ~3 tokens for message formatting
        prompt_tokens += 3
        
        return {
            "input": prompt_tokens,
            "output": 0  # We don't know output tokens yet
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return MODEL_INFO[self._model]
    
    @property
    def cost_per_1k_tokens(self) -> Dict[str, float]:
        """Get cost per 1K tokens for input and output."""
        info = self.get_model_info()
        return {
            "input": info["input_cost_per_1k"],
            "output": info["output_cost_per_1k"]
        }
    
    def validate_api_key(self) -> bool:
        """Validate the API key."""
        try:
            # Make a minimal API call to test the key
            self._client.models.list(limit=1)
            return True
        except Exception as e:
            logger.error(f"API key validation failed: {str(e)}")
            return False
    
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get rate limit information."""
        # OpenAI doesn't provide direct rate limit info via API
        # This is a placeholder implementation
        return {
            "requests_per_minute": 3500,
            "tokens_per_minute": 90000,
            "reset_time": None
        }