#!/usr/bin/env python3
# claude_code/lib/tools/ai_tools.py
"""AI-powered tools for generation and analysis."""

import os
import logging
import json
import base64
import requests
import tempfile
from typing import Dict, List, Optional, Any, Union
import time

from .base import tool, ToolRegistry

logger = logging.getLogger(__name__)


@tool(
    name="GenerateImage",
    description="Generate an image using AI based on a text prompt",
    parameters={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Text description of the image to generate"
            },
            "style": {
                "type": "string",
                "description": "Style of the image (realistic, cartoon, sketch, etc.)",
                "enum": ["realistic", "cartoon", "sketch", "painting", "3d", "pixel-art", "abstract"],
                "default": "realistic"
            },
            "size": {
                "type": "string",
                "description": "Size of the image",
                "enum": ["small", "medium", "large"],
                "default": "medium"
            },
            "save_path": {
                "type": "string",
                "description": "Absolute path where the image should be saved (optional)"
            }
        },
        "required": ["prompt"]
    },
    needs_permission=True,
    category="ai"
)
def generate_image(prompt: str, style: str = "realistic", size: str = "medium", save_path: Optional[str] = None) -> str:
    """Generate an image using AI based on a text prompt.
    
    Args:
        prompt: Text description of the image to generate
        style: Style of the image
        size: Size of the image
        save_path: Path where to save the image
        
    Returns:
        Path to the generated image or error message
    """
    logger.info(f"Generating image with prompt: {prompt} (style: {style}, size: {size})")
    
    # Map size to actual dimensions
    size_map = {
        "small": "512x512",
        "medium": "1024x1024",
        "large": "1792x1024"
    }
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
    
    # Prepare the prompt based on style
    full_prompt = prompt
    if style != "realistic":
        style_prompts = {
            "cartoon": f"A cartoon-style image of {prompt}",
            "sketch": f"A pencil sketch of {prompt}",
            "painting": f"An oil painting of {prompt}",
            "3d": f"A 3D rendered image of {prompt}",
            "pixel-art": f"A pixel art image of {prompt}",
            "abstract": f"An abstract representation of {prompt}"
        }
        full_prompt = style_prompts.get(style, prompt)
    
    try:
        # Call OpenAI API to generate image
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "dall-e-3",
            "prompt": full_prompt,
            "size": size_map.get(size, "1024x1024"),
            "quality": "standard",
            "n": 1
        }
        
        response = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            return f"Error: API request failed with status code {response.status_code}: {response.text}"
        
        data = response.json()
        
        if "data" not in data or not data["data"]:
            return "Error: No image data in response"
        
        image_url = data["data"][0]["url"]
        
        # Download the image
        image_response = requests.get(image_url)
        if image_response.status_code != 200:
            return f"Error: Failed to download image: {image_response.status_code}"
        
        # Save the image
        if save_path:
            # Ensure the path is absolute
            if not os.path.isabs(save_path):
                return f"Error: Save path must be absolute: {save_path}"
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save the image
            with open(save_path, "wb") as f:
                f.write(image_response.content)
            
            return f"Image generated and saved to: {save_path}"
        else:
            # Save to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(image_response.content)
                return f"Image generated and saved to temporary file: {tmp.name}"
    
    except Exception as e:
        logger.exception(f"Error generating image: {str(e)}")
        return f"Error generating image: {str(e)}"


@tool(
    name="TextToSpeech",
    description="Convert text to speech using AI",
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to convert to speech"
            },
            "voice": {
                "type": "string",
                "description": "Voice to use",
                "enum": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                "default": "nova"
            },
            "save_path": {
                "type": "string",
                "description": "Absolute path where the audio file should be saved (optional)"
            }
        },
        "required": ["text"]
    },
    needs_permission=True,
    category="ai"
)
def text_to_speech(text: str, voice: str = "nova", save_path: Optional[str] = None) -> str:
    """Convert text to speech using AI.
    
    Args:
        text: Text to convert to speech
        voice: Voice to use
        save_path: Path where to save the audio file
        
    Returns:
        Path to the generated audio file or error message
    """
    logger.info(f"Converting text to speech: {text[:50]}... (voice: {voice})")
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
    
    try:
        # Call OpenAI API to generate speech
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "tts-1",
            "input": text,
            "voice": voice
        }
        
        response = requests.post(
            "https://api.openai.com/v1/audio/speech",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            return f"Error: API request failed with status code {response.status_code}: {response.text}"
        
        # Save the audio
        if save_path:
            # Ensure the path is absolute
            if not os.path.isabs(save_path):
                return f"Error: Save path must be absolute: {save_path}"
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save the audio
            with open(save_path, "wb") as f:
                f.write(response.content)
            
            return f"Speech generated and saved to: {save_path}"
        else:
            # Save to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(response.content)
                return f"Speech generated and saved to temporary file: {tmp.name}"
    
    except Exception as e:
        logger.exception(f"Error generating speech: {str(e)}")
        return f"Error generating speech: {str(e)}"


def register_ai_tools(registry: ToolRegistry) -> None:
    """Register all AI tools with the registry.
    
    Args:
        registry: Tool registry to register with
    """
    from .base import create_tools_from_functions
    
    ai_tools = [
        generate_image,
        text_to_speech
    ]
    
    create_tools_from_functions(registry, ai_tools)
