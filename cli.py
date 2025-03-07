#!/usr/bin/env python3
# TODO: Refactor into modular structure similar to Claude Code (lib/, commands/, tools/ directories)
# TODO: Add support for multiple LLM providers (Azure OpenAI, Anthropic, etc.)
# TODO: Implement telemetry and usage tracking (optional, with consent)
import os
import sys
import json
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.panel import Panel
from rich.progress import Progress
from rich.syntax import Syntax
from rich.live import Live
from rich.layout import Layout
from rich.table import Table
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Callable
import asyncio
import concurrent.futures
from dotenv import load_dotenv
import time
import re
import traceback
import requests
import urllib.parse
from uuid import uuid4
import socket
import threading
import multiprocessing
import pickle
import hashlib
import logging
import fastapi
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Jina.ai client for search, fact-checking, and web reading
class JinaClient:
    """Client for interacting with Jina.ai endpoints"""
    
    def __init__(self, token: Optional[str] = None):
        """Initialize with your Jina token"""
        self.token = token or os.getenv("JINA_API_KEY", "")
        
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    
    def search(self, query: str) -> dict:
        """
        Search using s.jina.ai endpoint
        Args:
            query: Search term
        Returns:
            API response as dict
        """
        encoded_query = urllib.parse.quote(query)
        url = f"https://s.jina.ai/{encoded_query}"
        response = requests.get(url, headers=self.headers)
        return response.json()
    
    def fact_check(self, query: str) -> dict:
        """
        Get grounding info using g.jina.ai endpoint
        Args:
            query: Query to ground
        Returns:
            API response as dict
        """
        encoded_query = urllib.parse.quote(query)
        url = f"https://g.jina.ai/{encoded_query}"
        response = requests.get(url, headers=self.headers)
        return response.json()
        
    def reader(self, url: str) -> dict:
        """
        Get ranking using r.jina.ai endpoint
        Args:
            url: URL to rank
        Returns:
            API response as dict
        """
        encoded_url = urllib.parse.quote(url)
        url = f"https://r.jina.ai/{encoded_url}"
        response = requests.get(url, headers=self.headers)
        return response.json()

# Check if RL tools are available
HAVE_RL_TOOLS = False
try:
    # This is a placeholder for the actual import that would be used
    from tool_optimizer import ToolSelectionManager
    # If the import succeeds, set HAVE_RL_TOOLS to True
    HAVE_RL_TOOLS = True
except ImportError:
    # RL tools not available
    # Define a dummy ToolSelectionManager to avoid NameError
    class ToolSelectionManager:
        def __init__(self, **kwargs):
            self.optimizer = None
            self.data_dir = kwargs.get('data_dir', '')
            
        def record_tool_usage(self, **kwargs):
            pass

# Load environment variables
load_dotenv()

# TODO: Add update checking similar to Claude Code's auto-update functionality
# TODO: Add configuration file support to store settings beyond environment variables

app = typer.Typer(help="OpenAI Code Assistant CLI")
console = Console()

# Global Constants
# TODO: Move these to a config file
DEFAULT_MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 0
MAX_TOKENS = 4096
TOKEN_LIMIT_WARNING = 0.8  # Warn when 80% of token limit is reached

# Models
# TODO: Implement more sophisticated schema validation similar to Zod in the original
# TODO: Add permission system for tools that require user approval

class ToolParameter(BaseModel):
    name: str
    description: str
    type: str
    required: bool = False

class Tool(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    # TODO: Add needs_permission flag for sensitive operations
    # TODO: Add category for organizing tools (file, search, etc.)

class Message(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    # TODO: Add timestamp for message tracking
    # TODO: Add token count for better context management

class Conversation:
    def __init__(self):
        self.messages = []
        # TODO: Implement retry logic with exponential backoff for API calls
        # TODO: Add support for multiple LLM providers
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", DEFAULT_TEMPERATURE))
        self.tools = self._register_tools()
        self.tool_map = {tool.name: tool.function for tool in self.tools}
        self.conversation_id = str(uuid4())
        self.session_start_time = time.time()
        self.token_usage = {"prompt": 0, "completion": 0, "total": 0}
        self.verbose = False
        self.max_tool_iterations = int(os.getenv("MAX_TOOL_ITERATIONS", "10"))
        
        # Initialize tool selection optimizer if available
        self.tool_optimizer = None
        if HAVE_RL_TOOLS:
            try:
                # Create a simple tool registry adapter for the optimizer
                class ToolRegistryAdapter:
                    def __init__(self, tools):
                        self.tools = tools
                
                    def get_all_tools(self):
                        return self.tools
                
                    def get_all_tool_names(self):
                        return [tool.name for tool in self.tools]
            
                # Initialize the tool selection manager
                self.tool_optimizer = ToolSelectionManager(
                    tool_registry=ToolRegistryAdapter(self.tools),
                    enable_optimization=os.getenv("ENABLE_TOOL_OPTIMIZATION", "1") == "1",
                    data_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/rl")
                )
                if self.verbose:
                    print("Tool selection optimization enabled")
            except Exception as e:
                print(f"Warning: Failed to initialize tool optimizer: {e}")
        # TODO: Implement context window management
        
    # Jina.ai client for search, fact-checking, and web reading
    def _init_jina_client(self):
        """Initialize the Jina.ai client"""
        token = os.getenv("JINA_API_KEY", "")
        return JinaClient(token)
    
    def _jina_search(self, query: str) -> str:
        """Search the web using Jina.ai"""
        try:
            client = self._init_jina_client()
            results = client.search(query)
            
            if not results or not isinstance(results, dict):
                return f"No search results found for '{query}'"
            
            # Format the results
            formatted_results = "Search Results:\n\n"
            
            if "results" in results and isinstance(results["results"], list):
                for i, result in enumerate(results["results"], 1):
                    title = result.get("title", "No title")
                    url = result.get("url", "No URL")
                    snippet = result.get("snippet", "No snippet")
                    
                    formatted_results += f"{i}. {title}\n"
                    formatted_results += f"   URL: {url}\n"
                    formatted_results += f"   {snippet}\n\n"
            else:
                formatted_results += "Unexpected response format. Raw data:\n"
                formatted_results += json.dumps(results, indent=2)[:1000]
                
            return formatted_results
        except Exception as e:
            return f"Error performing search: {str(e)}"
    
    def _jina_fact_check(self, statement: str) -> str:
        """Fact check a statement using Jina.ai"""
        try:
            client = self._init_jina_client()
            results = client.fact_check(statement)
            
            if not results or not isinstance(results, dict):
                return f"No fact-checking results for '{statement}'"
            
            # Format the results
            formatted_results = "Fact Check Results:\n\n"
            formatted_results += f"Statement: {statement}\n\n"
            
            if "grounding" in results:
                grounding = results["grounding"]
                verdict = grounding.get("verdict", "Unknown")
                confidence = grounding.get("confidence", 0)
                
                formatted_results += f"Verdict: {verdict}\n"
                formatted_results += f"Confidence: {confidence:.2f}\n\n"
                
                if "sources" in grounding and isinstance(grounding["sources"], list):
                    formatted_results += "Sources:\n"
                    for i, source in enumerate(grounding["sources"], 1):
                        title = source.get("title", "No title")
                        url = source.get("url", "No URL")
                        formatted_results += f"{i}. {title}\n   {url}\n\n"
            else:
                formatted_results += "Unexpected response format. Raw data:\n"
                formatted_results += json.dumps(results, indent=2)[:1000]
                
            return formatted_results
        except Exception as e:
            return f"Error performing fact check: {str(e)}"
    
    def _jina_read_url(self, url: str) -> str:
        """Read and summarize a webpage using Jina.ai"""
        try:
            client = self._init_jina_client()
            results = client.reader(url)
            
            if not results or not isinstance(results, dict):
                return f"No reading results for URL '{url}'"
            
            # Format the results
            formatted_results = f"Web Page Summary: {url}\n\n"
            
            if "content" in results:
                content = results["content"]
                title = content.get("title", "No title")
                summary = content.get("summary", "No summary available")
                
                formatted_results += f"Title: {title}\n\n"
                formatted_results += f"Summary:\n{summary}\n\n"
                
                if "keyPoints" in content and isinstance(content["keyPoints"], list):
                    formatted_results += "Key Points:\n"
                    for i, point in enumerate(content["keyPoints"], 1):
                        formatted_results += f"{i}. {point}\n"
            else:
                formatted_results += "Unexpected response format. Raw data:\n"
                formatted_results += json.dumps(results, indent=2)[:1000]
                
            return formatted_results
        except Exception as e:
            return f"Error reading URL: {str(e)}"
    
    def _register_tools(self) -> List[Tool]:
        # TODO: Modularize tools into separate files
        # TODO: Implement Tool decorators for easier registration
        # TODO: Add more tools similar to Claude Code (ReadNotebook, NotebookEditCell, etc.)
        
        # Define and register all tools
        tools = [
            Tool(
                name="Weather",
                description="Gets the current weather for a location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and optional state/country (e.g., 'San Francisco, CA' or 'London, UK')"
                        }
                    },
                    "required": ["location"]
                },
                function=self._get_weather
            ),
            Tool(
                name="View",
                description="Reads a file from the local filesystem. The file_path parameter must be an absolute path, not a relative path.",
                parameters={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The absolute path to the file to read"
                        },
                        "limit": {
                            "type": "number",
                            "description": "The number of lines to read. Only provide if the file is too large to read at once."
                        },
                        "offset": {
                            "type": "number",
                            "description": "The line number to start reading from. Only provide if the file is too large to read at once"
                        }
                    },
                    "required": ["file_path"]
                },
                function=self._view_file
            ),
            Tool(
                name="Edit",
                description="This is a tool for editing files.",
                parameters={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The absolute path to the file to modify"
                        },
                        "old_string": {
                            "type": "string",
                            "description": "The text to replace"
                        },
                        "new_string": {
                            "type": "string",
                            "description": "The text to replace it with"
                        }
                    },
                    "required": ["file_path", "old_string", "new_string"]
                },
                function=self._edit_file
            ),
            Tool(
                name="Replace",
                description="Write a file to the local filesystem. Overwrites the existing file if there is one.",
                parameters={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The absolute path to the file to write"
                        },
                        "content": {
                            "type": "string",
                            "description": "The content to write to the file"
                        }
                    },
                    "required": ["file_path", "content"]
                },
                function=self._replace_file
            ),
            Tool(
                name="Bash",
                description="Executes a given bash command in a persistent shell session.",
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command to execute"
                        },
                        "timeout": {
                            "type": "number",
                            "description": "Optional timeout in milliseconds (max 600000)"
                        }
                    },
                    "required": ["command"]
                },
                function=self._execute_bash
            ),
            Tool(
                name="GlobTool",
                description="Fast file pattern matching tool that works with any codebase size.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The directory to search in. Defaults to the current working directory."
                        },
                        "pattern": {
                            "type": "string",
                            "description": "The glob pattern to match files against"
                        }
                    },
                    "required": ["pattern"]
                },
                function=self._glob_tool
            ),
            Tool(
                name="GrepTool",
                description="Fast content search tool that works with any codebase size.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The directory to search in. Defaults to the current working directory."
                        },
                        "pattern": {
                            "type": "string",
                            "description": "The regular expression pattern to search for in file contents"
                        },
                        "include": {
                            "type": "string",
                            "description": "File pattern to include in the search (e.g. \"*.js\", \"*.{ts,tsx}\")"
                        }
                    },
                    "required": ["pattern"]
                },
                function=self._grep_tool
            ),
            Tool(
                name="LS",
                description="Lists files and directories in a given path.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The absolute path to the directory to list"
                        },
                        "ignore": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "List of glob patterns to ignore"
                        }
                    },
                    "required": ["path"]
                },
                function=self._list_directory
            ),
            Tool(
                name="JinaSearch",
                description="Search the web for information using Jina.ai",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                },
                function=self._jina_search
            ),
            Tool(
                name="JinaFactCheck",
                description="Fact check a statement using Jina.ai",
                parameters={
                    "type": "object",
                    "properties": {
                        "statement": {
                            "type": "string",
                            "description": "The statement to fact check"
                        }
                    },
                    "required": ["statement"]
                },
                function=self._jina_fact_check
            ),
            Tool(
                name="JinaReadURL",
                description="Read and summarize a webpage using Jina.ai",
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL of the webpage to read"
                        }
                    },
                    "required": ["url"]
                },
                function=self._jina_read_url
            )
        ]
        return tools
    
    # Tool implementations
    # TODO: Add better error handling and user feedback
    # TODO: Implement tool usage tracking and metrics
    
    def _get_weather(self, location: str) -> str:
        """Get current weather for a location using OpenWeatherMap API"""
        try:
            # Get API key from environment or use a default for testing
            api_key = os.getenv("OPENWEATHER_API_KEY", "")
            if not api_key:
                return "Error: OpenWeatherMap API key not found. Please set the OPENWEATHER_API_KEY environment variable."
            
            # Prepare the API request
            base_url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                "q": location,
                "appid": api_key,
                "units": "metric"  # Use metric units (Celsius)
            }
            
            # Make the API request
            response = requests.get(base_url, params=params)
            
            # Check if the request was successful
            if response.status_code == 200:
                data = response.text
                # Try to parse as JSON
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    return f"Error: Unable to parse weather data. Raw response: {data[:200]}..."
                
                # Extract relevant weather information
                weather_desc = data["weather"][0]["description"]
                temp = data["main"]["temp"]
                feels_like = data["main"]["feels_like"]
                humidity = data["main"]["humidity"]
                wind_speed = data["wind"]["speed"]
                
                # Format the response
                weather_info = (
                    f"Current weather in {location}:\n"
                    f"• Condition: {weather_desc.capitalize()}\n"
                    f"• Temperature: {temp}°C ({(temp * 9/5) + 32:.1f}°F)\n"
                    f"• Feels like: {feels_like}°C ({(feels_like * 9/5) + 32:.1f}°F)\n"
                    f"• Humidity: {humidity}%\n"
                    f"• Wind speed: {wind_speed} m/s ({wind_speed * 2.237:.1f} mph)"
                )
                return weather_info
            else:
                # Handle API errors
                if response.status_code == 404:
                    return f"Error: Location '{location}' not found. Please check the spelling or try a different location."
                elif response.status_code == 401:
                    return "Error: Invalid API key. Please check your OpenWeatherMap API key."
                else:
                    return f"Error: Unable to fetch weather data. Status code: {response.status_code}"
        
        except requests.exceptions.RequestException as e:
            return f"Error: Network error when fetching weather data: {str(e)}"
        except Exception as e:
            return f"Error: Failed to get weather information: {str(e)}"
    
    def _view_file(self, file_path: str, limit: Optional[int] = None, offset: Optional[int] = 0) -> str:
        # TODO: Add special handling for binary files and images
        # TODO: Add syntax highlighting for code files
        try:
            if not os.path.exists(file_path):
                return f"Error: File not found: {file_path}"
            
            # TODO: Handle file size limits better
            
            with open(file_path, 'r') as f:
                if limit is not None and offset is not None:
                    # Skip to offset
                    for _ in range(offset):
                        next(f, None)
                    
                    # Read limited lines
                    lines = []
                    for _ in range(limit):
                        line = next(f, None)
                        if line is None:
                            break
                        lines.append(line)
                    content = ''.join(lines)
                else:
                    content = f.read()
            
            # TODO: Add file metadata like size, permissions, etc.
            return content
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def _edit_file(self, file_path: str, old_string: str, new_string: str) -> str:
        try:
            # Create directory if creating new file
            if not os.path.exists(os.path.dirname(file_path)) and old_string == "":
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
            if old_string == "" and not os.path.exists(file_path):
                # Creating new file
                with open(file_path, 'w') as f:
                    f.write(new_string)
                return f"Created new file: {file_path}"
            
            # Reading existing file
            if not os.path.exists(file_path):
                return f"Error: File not found: {file_path}"
            
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Replace string
            if old_string not in content:
                return f"Error: Could not find the specified text in {file_path}"
            
            # Count occurrences to ensure uniqueness
            occurrences = content.count(old_string)
            if occurrences > 1:
                return f"Error: Found {occurrences} occurrences of the specified text in {file_path}. Please provide more context to uniquely identify the text to replace."
            
            new_content = content.replace(old_string, new_string)
            
            # Write back to file
            with open(file_path, 'w') as f:
                f.write(new_content)
            
            return f"Successfully edited {file_path}"
        
        except Exception as e:
            return f"Error editing file: {str(e)}"
    
    def _replace_file(self, file_path: str, content: str) -> str:
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            # Write content to file
            with open(file_path, 'w') as f:
                f.write(content)
            
            return f"Successfully wrote to {file_path}"
        
        except Exception as e:
            return f"Error writing file: {str(e)}"
    
    def _execute_bash(self, command: str, timeout: Optional[int] = None) -> str:
        try:
            import subprocess
            import shlex
            
            # Security check for banned commands
            banned_commands = [
                'alias', 'curl', 'curlie', 'wget', 'axel', 'aria2c', 'nc', 
                'telnet', 'lynx', 'w3m', 'links', 'httpie', 'xh', 'http-prompt', 
                'chrome', 'firefox', 'safari'
            ]
            
            for banned in banned_commands:
                if banned in command.split():
                    return f"Error: The command '{banned}' is not allowed for security reasons."
            
            # Execute command
            if timeout:
                timeout_seconds = timeout / 1000  # Convert to seconds
            else:
                timeout_seconds = 1800  # 30 minutes default
            
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_seconds
            )
            
            output = result.stdout
            if result.stderr:
                output += f"\nErrors:\n{result.stderr}"
            
            # Truncate if too long
            if len(output) > 30000:
                output = output[:30000] + "\n... (output truncated)"
            
            return output
        
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout_seconds} seconds"
        except Exception as e:
            return f"Error executing command: {str(e)}"
    
    def _glob_tool(self, pattern: str, path: Optional[str] = None) -> str:
        try:
            import glob
            import os
            
            if path is None:
                path = os.getcwd()
            
            # Build the full pattern path
            if not os.path.isabs(path):
                path = os.path.abspath(path)
            
            full_pattern = os.path.join(path, pattern)
            
            # Get matching files
            matches = glob.glob(full_pattern, recursive=True)
            
            # Sort by modification time (newest first)
            matches.sort(key=os.path.getmtime, reverse=True)
            
            if not matches:
                return f"No files matching pattern '{pattern}' in {path}"
            
            return "\n".join(matches)
        
        except Exception as e:
            return f"Error in glob search: {str(e)}"
    
    def _grep_tool(self, pattern: str, path: Optional[str] = None, include: Optional[str] = None) -> str:
        try:
            import re
            import os
            import fnmatch
            from concurrent.futures import ThreadPoolExecutor
            
            if path is None:
                path = os.getcwd()
            
            if not os.path.isabs(path):
                path = os.path.abspath(path)
            
            # Compile regex pattern
            regex = re.compile(pattern)
            
            # Get all files
            all_files = []
            for root, _, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Apply include filter if provided
                    if include:
                        if not fnmatch.fnmatch(file, include):
                            continue
                    
                    all_files.append(file_path)
            
            # Sort by modification time (newest first)
            all_files.sort(key=os.path.getmtime, reverse=True)
            
            matches = []
            
            def search_file(file_path):
                try:
                    with open(file_path, 'r', errors='ignore') as f:
                        content = f.read()
                        if regex.search(content):
                            return file_path
                except:
                    # Skip files that can't be read
                    pass
                return None
            
            # Search files in parallel
            with ThreadPoolExecutor(max_workers=10) as executor:
                results = executor.map(search_file, all_files)
                
                for result in results:
                    if result:
                        matches.append(result)
            
            if not matches:
                return f"No matches found for pattern '{pattern}' in {path}"
            
            return "\n".join(matches)
        
        except Exception as e:
            return f"Error in grep search: {str(e)}"
    
    def _list_directory(self, path: str, ignore: Optional[List[str]] = None) -> str:
        try:
            import os
            import fnmatch
            
            # If path is not absolute, make it absolute from current directory
            if not os.path.isabs(path):
                path = os.path.abspath(os.path.join(os.getcwd(), path))
            
            if not os.path.exists(path):
                return f"Error: Directory not found: {path}"
            
            if not os.path.isdir(path):
                return f"Error: Path is not a directory: {path}"
            
            # List directory contents
            items = os.listdir(path)
            
            # Apply ignore patterns
            if ignore:
                for pattern in ignore:
                    items = [item for item in items if not fnmatch.fnmatch(item, pattern)]
            
            # Sort items
            items.sort()
            
            # Format output
            result = []
            for item in items:
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    result.append(f"{item}/")
                else:
                    result.append(item)
            
            if not result:
                return f"Directory {path} is empty"
            
            return "\n".join(result)
        
        except Exception as e:
            return f"Error listing directory: {str(e)}"
    
    def add_message(self, role: str, content: str):
        """Legacy method to add messages - use direct append now"""
        self.messages.append({"role": role, "content": content})
    
    def process_tool_calls(self, tool_calls, query=None):
        # TODO: Add tool call validation
        # TODO: Add permission system for sensitive tools
        # TODO: Add progress visualization for long-running tools
        responses = []
        
        # Process tool calls in parallel
        from concurrent.futures import ThreadPoolExecutor
        
        def process_single_tool(tool_call):
            # Handle both object-style and dict-style tool calls
            if isinstance(tool_call, dict):
                function_name = tool_call["function"]["name"]
                function_args = json.loads(tool_call["function"]["arguments"])
                tool_call_id = tool_call["id"]
            else:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                tool_call_id = tool_call.id
            
            # Get the tool function
            if function_name in self.tool_map:
                # TODO: Add pre-execution validation
                # TODO: Add permission check here
                
                # Track start time for metrics
                start_time = time.time()
                
                try:
                    function = self.tool_map[function_name]
                    result = function(**function_args)
                    success = True
                except Exception as e:
                    result = f"Error executing tool {function_name}: {str(e)}\n{traceback.format_exc()}"
                    success = False
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Record tool usage for optimization if optimizer is available
                if self.tool_optimizer is not None and query is not None:
                    try:
                        # Create current context snapshot
                        context = {
                            "messages": self.messages.copy(),
                            "conversation_id": self.conversation_id,
                        }
                        
                        # Record tool usage
                        self.tool_optimizer.record_tool_usage(
                            query=query,
                            tool_name=function_name,
                            execution_time=execution_time,
                            token_usage=self.token_usage.copy(),
                            success=success,
                            context=context,
                            result=result
                        )
                    except Exception as e:
                        if self.verbose:
                            print(f"Warning: Failed to record tool usage: {e}")
                
                return {
                    "tool_call_id": tool_call_id,
                    "function_name": function_name,
                    "result": result,
                    "name": function_name,
                    "execution_time": execution_time,  # For metrics
                    "success": success
                }
            return None
        
        # Process all tool calls in parallel
        with ThreadPoolExecutor(max_workers=min(10, len(tool_calls))) as executor:
            futures = [executor.submit(process_single_tool, tool_call) for tool_call in tool_calls]
            for future in futures:
                result = future.result()
                if result:
                    # Add tool response to messages
                    self.messages.append({
                        "tool_call_id": result["tool_call_id"],
                        "role": "tool",
                        "name": result["name"],
                        "content": result["result"]
                    })
                    
                    responses.append({
                        "tool_call_id": result["tool_call_id"],
                        "function_name": result["function_name"],
                        "result": result["result"]
                    })
                    
                    # Log tool execution metrics if verbose
                    if self.verbose:
                        print(f"Tool {result['function_name']} executed in {result['execution_time']:.2f}s (success: {result['success']})")
        
        # Return tool responses
        return responses
    
    def compact(self):
        # TODO: Add more sophisticated compaction with token counting
        # TODO: Implement selective retention of critical information
        # TODO: Add option to save conversation history before compacting
        
        system_prompt = next((m for m in self.messages if m["role"] == "system"), None)
        user_messages = [m for m in self.messages if m["role"] == "user"]
        
        if not user_messages:
            return "No user messages to compact."
        
        last_user_message = user_messages[-1]
        
        # Create a compaction prompt
        # TODO: Improve the compaction prompt with more guidance on what to retain
        compact_prompt = (
            "Summarize the conversation so far, focusing on the key points, decisions, and context. "
            "Keep important details about the code and tasks. Retain critical file paths, commands, "
            "and code snippets. The summary should be concise but complete enough to continue the "
            "conversation effectively."
        )
        
        # Add compaction message
        self.messages.append({"role": "user", "content": compact_prompt})
        
        # Get compaction summary
        # TODO: Add error handling for compaction API call
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            stream=False
        )
        
        summary = response.choices[0].message.content
        
        # Reset conversation with summary
        if system_prompt:
            self.messages = [system_prompt]
        else:
            self.messages = []
        
        self.messages.append({"role": "system", "content": f"This is a compacted conversation. Previous context: {summary}"})
        self.messages.append({"role": "user", "content": last_user_message["content"]})
        
        # TODO: Add metrics for compaction (tokens before/after)
        
        return "Conversation compacted successfully."
    
    def get_response(self, user_input: str, stream: bool = True):
        # TODO: Add more special commands similar to Claude Code (e.g., /version, /status)
        # TODO: Implement binary feedback mechanism for comparing responses
        
        # Special commands
        if user_input.strip() == "/compact":
            return self.compact()
        
        # Add a debug command to help diagnose issues
        if user_input.strip() == "/debug":
            debug_info = {
                "model": self.model,
                "temperature": self.temperature,
                "message_count": len(self.messages),
                "token_usage": self.token_usage,
                "conversation_id": self.conversation_id,
                "session_duration": time.time() - self.session_start_time,
                "tools_count": len(self.tools),
                "python_version": sys.version,
                "openai_version": OpenAI.__version__ if hasattr(OpenAI, "__version__") else "Unknown"
            }
            return "Debug Information:\n" + json.dumps(debug_info, indent=2)
        
        if user_input.strip() == "/help":
            # Standard commands
            commands = [
                "/help - Show this help message",
                "/compact - Compact the conversation to reduce token usage",
                "/status - Show token usage and session information",
                "/config - Show current configuration settings",
            ]
            
            # RL-specific commands if available
            if self.tool_optimizer is not None:
                commands.extend([
                    "/rl-status - Show RL tool optimizer status",
                    "/rl-update - Update the RL model manually",
                    "/rl-stats - Show tool usage statistics",
                ])
            
            return "Available commands:\n" + "\n".join(commands)
        
        # Token usage and session stats
        if user_input.strip() == "/status":
            # Calculate session duration
            session_duration = time.time() - self.session_start_time
            hours, remainder = divmod(session_duration, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            # Format message
            status = (
                f"Session ID: {self.conversation_id}\n"
                f"Model: {self.model} (Temperature: {self.temperature})\n"
                f"Session duration: {int(hours)}h {int(minutes)}m {int(seconds)}s\n\n"
                f"Token usage:\n"
                f"  Prompt tokens: {self.token_usage['prompt']}\n"
                f"  Completion tokens: {self.token_usage['completion']}\n"
                f"  Total tokens: {self.token_usage['total']}\n"
            )
            return status
            
        # Configuration settings
        if user_input.strip() == "/config":
            config_info = (
                f"Current Configuration:\n"
                f"  Model: {self.model}\n"
                f"  Temperature: {self.temperature}\n"
                f"  Max tool iterations: {self.max_tool_iterations}\n"
                f"  Verbose mode: {self.verbose}\n"
                f"  RL optimization: {self.tool_optimizer is not None}\n"
            )
            
            # Provide instructions for changing settings
            config_info += "\nTo change settings, use:\n"
            config_info += "  /config set <setting> <value>\n"
            config_info += "Example: /config set max_tool_iterations 15"
            
            return config_info
            
        # Handle configuration changes
        if user_input.strip().startswith("/config set "):
            parts = user_input.strip().split(" ", 3)
            if len(parts) != 4:
                return "Invalid format. Use: /config set <setting> <value>"
                
            setting = parts[2]
            value = parts[3]
            
            if setting == "max_tool_iterations":
                try:
                    self.max_tool_iterations = int(value)
                    return f"Max tool iterations set to {self.max_tool_iterations}"
                except ValueError:
                    return "Invalid value. Please provide a number."
            elif setting == "temperature":
                try:
                    self.temperature = float(value)
                    return f"Temperature set to {self.temperature}"
                except ValueError:
                    return "Invalid value. Please provide a number."
            elif setting == "verbose":
                if value.lower() in ("true", "yes", "1", "on"):
                    self.verbose = True
                    return "Verbose mode enabled"
                elif value.lower() in ("false", "no", "0", "off"):
                    self.verbose = False
                    return "Verbose mode disabled"
                else:
                    return "Invalid value. Use 'true' or 'false'."
            elif setting == "model":
                self.model = value
                return f"Model set to {self.model}"
            else:
                return f"Unknown setting: {setting}"
        
        # RL-specific commands
        if self.tool_optimizer is not None:
            # RL status command
            if user_input.strip() == "/rl-status":
                return (
                    f"RL tool optimization is active\n"
                    f"Optimizer type: {type(self.tool_optimizer).__name__}\n"
                    f"Number of tools: {len(self.tools)}\n"
                    f"Data directory: {self.tool_optimizer.optimizer.data_dir if hasattr(self.tool_optimizer, 'optimizer') else 'N/A'}\n"
                )
            
            # RL update command
            if user_input.strip() == "/rl-update":
                try:
                    result = self.tool_optimizer.optimizer.update_model()
                    status = f"RL model update status: {result['status']}\n"
                    if 'metrics' in result:
                        status += "Metrics:\n" + "\n".join([f"  {k}: {v}" for k, v in result['metrics'].items()])
                    return status
                except Exception as e:
                    return f"Error updating RL model: {str(e)}"
            
            # RL stats command
            if user_input.strip() == "/rl-stats":
                try:
                    if hasattr(self.tool_optimizer, 'optimizer') and hasattr(self.tool_optimizer.optimizer, 'tracker'):
                        stats = self.tool_optimizer.optimizer.tracker.get_tool_stats()
                        if not stats:
                            return "No tool usage data available yet."
                        
                        result = "Tool Usage Statistics:\n\n"
                        for tool_name, tool_stats in stats.items():
                            result += f"{tool_name}:\n"
                            result += f"  Count: {tool_stats['count']}\n"
                            result += f"  Success rate: {tool_stats['success_rate']:.2f}\n"
                            result += f"  Avg time: {tool_stats['avg_time']:.2f}s\n"
                            result += f"  Avg tokens: {tool_stats['avg_total_tokens']:.1f}\n"
                            result += "\n"
                        return result
                    return "Tool usage statistics not available."
                except Exception as e:
                    return f"Error getting tool statistics: {str(e)}"
        
        # TODO: Add /version command to show version information
        
        # Add user message
        self.messages.append({"role": "user", "content": user_input})
        
        # Initialize empty response
        response_text = ""
        
        # Create tools list for API
        # TODO: Add dynamic tool availability based on context
        api_tools = []
        for tool in self.tools:
            api_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            })
        
        if stream:
            # TODO: Add retry mechanism for API failures
            # TODO: Add token tracking for response
            # TODO: Implement cancellation support
            
            # Stream response
            try:
                # Add retry logic for API calls
                max_retries = 3
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        stream = self.client.chat.completions.create(
                            model=self.model,
                            messages=self.messages,
                            tools=api_tools,
                            temperature=self.temperature,
                            stream=True
                        )
                        break  # Success, exit retry loop
                    except Exception as e:
                        retry_count += 1
                        if retry_count >= max_retries:
                            raise  # Re-raise if we've exhausted retries
                        
                        # Exponential backoff
                        wait_time = 2 ** retry_count
                        if self.verbose:
                            console.print(f"[yellow]API call failed, retrying in {wait_time}s... ({retry_count}/{max_retries})[/yellow]")
                        time.sleep(wait_time)
                
                current_tool_calls = []
                tool_call_chunks = {}
                
                # Process streaming response outside of the status context
                with Live("", refresh_per_second=10) as live:
                    for chunk in stream:
                        # If there's content, print it
                        if chunk.choices[0].delta.content:
                            content_piece = chunk.choices[0].delta.content
                            response_text += content_piece
                            # Update the live display with the accumulated response
                            live.update(response_text)
                        
                        # Process tool calls
                        delta = chunk.choices[0].delta
                        if delta.tool_calls:
                            for tool_call_delta in delta.tool_calls:
                                # Initialize tool call in chunks dictionary if new
                                if tool_call_delta.index not in tool_call_chunks:
                                    tool_call_chunks[tool_call_delta.index] = {
                                        "id": "",
                                        "function": {"name": "", "arguments": ""}
                                    }
                                
                                # Update tool call data
                                if tool_call_delta.id:
                                    tool_call_chunks[tool_call_delta.index]["id"] = tool_call_delta.id
                                
                                if tool_call_delta.function:
                                    if tool_call_delta.function.name:
                                        tool_call_chunks[tool_call_delta.index]["function"]["name"] = tool_call_delta.function.name
                                    
                                    if tool_call_delta.function.arguments:
                                        tool_call_chunks[tool_call_delta.index]["function"]["arguments"] += tool_call_delta.function.arguments
                
                # No need to print the response again as it was already streamed in the Live context
                
            except Exception as e:
                # TODO: Add better error handling and user feedback
                console.print(f"[bold red]Error during API call:[/bold red] {str(e)}")
                return f"Error during API call: {str(e)}"
            
            # Convert tool call chunks to actual tool calls
            for index, tool_call_data in tool_call_chunks.items():
                current_tool_calls.append({
                    "id": tool_call_data["id"],
                    "function": {
                        "name": tool_call_data["function"]["name"],
                        "arguments": tool_call_data["function"]["arguments"]
                    }
                })
            
            # Process tool calls if any
            if current_tool_calls:
                try:
                    # Add assistant message with tool_calls to messages first
                    # Ensure each tool call has a "type" field set to "function"
                    processed_tool_calls = []
                    for tool_call in current_tool_calls:
                        processed_tool_call = tool_call.copy()
                        processed_tool_call["type"] = "function"
                        processed_tool_calls.append(processed_tool_call)
                        
                    # Make sure we add the assistant message with tool calls before processing them
                    self.messages.append({
                        "role": "assistant", 
                        "content": response_text,
                        "tool_calls": processed_tool_calls
                    })
                        
                    # Now process the tool calls
                    with console.status("[bold green]Running tools..."):
                        tool_responses = self.process_tool_calls(current_tool_calls, query=user_input)
                except Exception as e:
                    console.print(f"[bold red]Error:[/bold red] {str(e)}")
                    console.print(traceback.format_exc())
                    return f"Error processing tool calls: {str(e)}"
                    
                # Continue the conversation with tool responses
                # Implement looping function calls to allow for recursive tool usage
                max_loop_iterations = self.max_tool_iterations  # Use configurable setting
                current_iteration = 0
                
                while current_iteration < max_loop_iterations:
                    # Add retry logic for follow-up API calls
                    max_retries = 3
                    retry_count = 0
                    follow_up = None
                    
                    while retry_count < max_retries:
                        try:
                            follow_up = self.client.chat.completions.create(
                                model=self.model,
                                messages=self.messages,
                                tools=api_tools,  # Pass tools to enable recursive function calling
                                stream=False
                            )
                            break  # Success, exit retry loop
                        except Exception as e:
                            retry_count += 1
                            if retry_count >= max_retries:
                                raise  # Re-raise if we've exhausted retries
                                
                            # Exponential backoff
                            wait_time = 2 ** retry_count
                            if self.verbose:
                                console.print(f"[yellow]Follow-up API call failed, retrying in {wait_time}s... ({retry_count}/{max_retries})[/yellow]")
                            time.sleep(wait_time)
                    
                    # Check if the follow-up response contains more tool calls
                    assistant_message = follow_up.choices[0].message
                    follow_up_text = assistant_message.content or ""
                    
                    # If there are no more tool calls, we're done with the loop
                    if not hasattr(assistant_message, 'tool_calls') or not assistant_message.tool_calls:
                        if follow_up_text:
                            console.print(Markdown(follow_up_text))
                            response_text += "\n" + follow_up_text
                        
                        # Add the final assistant message to the conversation
                        self.messages.append({"role": "assistant", "content": follow_up_text})
                        break
                    
                    # Process the new tool calls
                    current_tool_calls = []
                    for tool_call in assistant_message.tool_calls:
                        # Handle both object-style and dict-style tool calls
                        if isinstance(tool_call, dict):
                            processed_tool_call = tool_call.copy()
                        else:
                            # Convert object to dict
                            processed_tool_call = {
                                "id": tool_call.id,
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments
                                }
                            }
                        
                        # Ensure type field is present
                        processed_tool_call["type"] = "function"
                        current_tool_calls.append(processed_tool_call)
                    
                    # Add the assistant message with tool calls
                    self.messages.append({
                        "role": "assistant",
                        "content": follow_up_text,
                        "tool_calls": current_tool_calls
                    })
                    
                    # Process the new tool calls
                    with console.status(f"[bold green]Running tools (iteration {current_iteration + 1})...[/bold green]"):
                        tool_responses = self.process_tool_calls(assistant_message.tool_calls, query=user_input)
                    
                    # Increment the iteration counter
                    current_iteration += 1
                
                # If we've reached the maximum number of iterations, add a warning
                if current_iteration >= max_loop_iterations:
                    warning_message = f"[yellow]Warning: Reached maximum number of tool call iterations ({max_loop_iterations}). Some operations may be incomplete.[/yellow]"
                    console.print(warning_message)
                    response_text += f"\n\n{warning_message}"
            
            # Add assistant response to messages if there were no tool calls
            # (we already added it above if there were tool calls)
            if not current_tool_calls:
                self.messages.append({"role": "assistant", "content": response_text})
            
            return response_text
        else:
            # Non-streaming response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=api_tools,
                temperature=self.temperature,
                stream=False
            )
            
            # Track token usage
            if hasattr(response, 'usage'):
                self.token_usage["prompt"] += response.usage.prompt_tokens
                self.token_usage["completion"] += response.usage.completion_tokens
                self.token_usage["total"] += response.usage.total_tokens
            
            assistant_message = response.choices[0].message
            response_text = assistant_message.content or ""
            
            # Process tool calls if any
            if assistant_message.tool_calls:
                # Add assistant message with tool_calls to messages
                # Convert tool_calls to a list of dictionaries with "type" field
                processed_tool_calls = []
                for tool_call in assistant_message.tool_calls:
                    # Handle both object-style and dict-style tool calls
                    if isinstance(tool_call, dict):
                        processed_tool_call = tool_call.copy()
                    else:
                        # Convert object to dict
                        processed_tool_call = {
                            "id": tool_call.id,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        }
                    
                    # Ensure type field is present
                    processed_tool_call["type"] = "function"
                    processed_tool_calls.append(processed_tool_call)
                
                self.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "tool_calls": processed_tool_calls
                })
                
                with console.status("[bold green]Running tools..."):
                    tool_responses = self.process_tool_calls(assistant_message.tool_calls, query=user_input)
                
                # Continue the conversation with tool responses
                # Implement looping function calls to allow for recursive tool usage
                max_loop_iterations = self.max_tool_iterations  # Use configurable setting
                current_iteration = 0
                
                while current_iteration < max_loop_iterations:
                    # Add retry logic for follow-up API calls
                    max_retries = 3
                    retry_count = 0
                    follow_up = None
                    
                    while retry_count < max_retries:
                        try:
                            follow_up = self.client.chat.completions.create(
                                model=self.model,
                                messages=self.messages,
                                tools=api_tools,  # Pass tools to enable recursive function calling
                                stream=False
                            )
                            break  # Success, exit retry loop
                        except Exception as e:
                            retry_count += 1
                            if retry_count >= max_retries:
                                raise  # Re-raise if we've exhausted retries
                                
                            # Exponential backoff
                            wait_time = 2 ** retry_count
                            if self.verbose:
                                console.print(f"[yellow]Follow-up API call failed, retrying in {wait_time}s... ({retry_count}/{max_retries})[/yellow]")
                            time.sleep(wait_time)
                    
                    # Check if the follow-up response contains more tool calls
                    assistant_message = follow_up.choices[0].message
                    follow_up_text = assistant_message.content or ""
                    
                    # If there are no more tool calls, we're done with the loop
                    if not hasattr(assistant_message, 'tool_calls') or not assistant_message.tool_calls:
                        if follow_up_text:
                            console.print(Markdown(follow_up_text))
                            response_text += "\n" + follow_up_text
                        
                        # Add the final assistant message to the conversation
                        self.messages.append({"role": "assistant", "content": follow_up_text})
                        break
                    
                    # Process the new tool calls
                    current_tool_calls = []
                    for tool_call in assistant_message.tool_calls:
                        # Handle both object-style and dict-style tool calls
                        if isinstance(tool_call, dict):
                            processed_tool_call = tool_call.copy()
                        else:
                            # Convert object to dict
                            processed_tool_call = {
                                "id": tool_call.id,
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments
                                }
                            }
                        
                        # Ensure type field is present
                        processed_tool_call["type"] = "function"
                        current_tool_calls.append(processed_tool_call)
                    
                    # Add the assistant message with tool calls
                    self.messages.append({
                        "role": "assistant",
                        "content": follow_up_text,
                        "tool_calls": current_tool_calls
                    })
                    
                    # Process the new tool calls
                    with console.status(f"[bold green]Running tools (iteration {current_iteration + 1})...[/bold green]"):
                        tool_responses = self.process_tool_calls(assistant_message.tool_calls, query=user_input)
                    
                    # Increment the iteration counter
                    current_iteration += 1
                
                # If we've reached the maximum number of iterations, add a warning
                if current_iteration >= max_loop_iterations:
                    warning_message = f"[yellow]Warning: Reached maximum number of tool call iterations ({max_loop_iterations}). Some operations may be incomplete.[/yellow]"
                    console.print(warning_message)
                    response_text += f"\n\n{warning_message}"
            else:
                console.print(Markdown(response_text))
            
            # Add assistant response to messages if not already added
            # (we already added it above if there were tool calls)
            if not assistant_message.tool_calls:
                self.messages.append({"role": "assistant", "content": response_text})
            
            return response_text

# TODO: Create a more flexible system prompt mechanism with customizable templates
def get_system_prompt():
    return """You are OpenAI Code Assistant, a CLI tool that helps users with software engineering tasks and general information.
Use the available tools to assist the user with their requests.

# Tone and style
You should be concise, direct, and to the point. When you run a non-trivial bash command, 
you should explain what the command does and why you are running it.
Output text to communicate with the user; all text you output outside of tool use is displayed to the user.
Remember that your output will be displayed on a command line interface.

# Tool usage policy
- When doing file search, remember to search effectively with the available tools.
- Always use the appropriate tool for the task.
- Use parallel tool calls when appropriate to improve performance.
- NEVER commit changes unless the user explicitly asks you to.
- For weather queries, use the Weather tool to provide real-time information.

# Tasks
The user will primarily request you perform software engineering tasks:
1. Solving bugs
2. Adding new functionality 
3. Refactoring code
4. Explaining code
5. Writing tests

For these tasks:
1. Use search tools to understand the codebase
2. Implement solutions using the available tools
3. Verify solutions with tests if possible
4. Run lint and typecheck commands when appropriate

The user may also ask for general information:
1. Weather conditions
2. Simple calculations
3. General knowledge questions

# Code style
- Follow the existing code style of the project
- Maintain consistent naming conventions
- Use appropriate libraries that are already in the project
- Add comments when code is complex or non-obvious

IMPORTANT: You should minimize output tokens as much as possible while maintaining helpfulness, 
quality, and accuracy. Answer concisely with short lines of text unless the user asks for detail.
"""

# TODO: Add version information and CLI arguments
# TODO: Add logging configuration
# TODO: Create a proper CLI command structure with subcommands

# Hosting and replication capabilities
class HostingManager:
    """Manages hosting and replication of the assistant"""
    
    def __init__(self, host="127.0.0.1", port=8000):
        self.host = host
        self.port = port
        self.app = FastAPI(title="OpenAI Code Assistant API")
        self.conversation_pool = {}
        self.setup_api()
        
    def setup_api(self):
        """Configure the FastAPI application"""
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # In production, restrict this to specific domains
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Define API routes
        @self.app.get("/")
        async def root():
            return {"message": "OpenAI Code Assistant API", "status": "running"}
        
        @self.app.post("/conversation")
        async def create_conversation(
            request: Request,
            background_tasks: BackgroundTasks,
            model: str = DEFAULT_MODEL,
            temperature: float = DEFAULT_TEMPERATURE
        ):
            """Create a new conversation instance"""
            conversation_id = str(uuid4())
            
            # Initialize conversation in background
            background_tasks.add_task(self._init_conversation, conversation_id, model, temperature)
            
            return {
                "conversation_id": conversation_id,
                "status": "initializing",
                "model": model
            }
        
        @self.app.post("/conversation/{conversation_id}/message")
        async def send_message(
            conversation_id: str,
            request: Request
        ):
            """Send a message to a conversation"""
            if conversation_id not in self.conversation_pool:
                raise HTTPException(status_code=404, detail="Conversation not found")
                
            data = await request.json()
            user_input = data.get("message", "")
            
            # Get conversation instance
            conversation = self.conversation_pool[conversation_id]
            
            # Process message
            try:
                response = conversation.get_response(user_input, stream=False)
                return {
                    "conversation_id": conversation_id,
                    "response": response
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")
        
        @self.app.post("/conversation/{conversation_id}/message/stream")
        async def stream_message(
            conversation_id: str,
            request: Request
        ):
            """Stream a message response from a conversation"""
            if conversation_id not in self.conversation_pool:
                raise HTTPException(status_code=404, detail="Conversation not found")
                
            data = await request.json()
            user_input = data.get("message", "")
            
            # Get conversation instance
            conversation = self.conversation_pool[conversation_id]
            
            # Create async generator for streaming
            async def response_generator():
                # Add user message
                conversation.messages.append({"role": "user", "content": user_input})
                
                # Create tools list for API
                api_tools = []
                for tool in conversation.tools:
                    api_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters
                        }
                    })
                
                # Stream response
                try:
                    stream = conversation.client.chat.completions.create(
                        model=conversation.model,
                        messages=conversation.messages,
                        tools=api_tools,
                        temperature=conversation.temperature,
                        stream=True
                    )
                    
                    current_tool_calls = []
                    tool_call_chunks = {}
                    response_text = ""
                    
                    for chunk in stream:
                        # If there's content, yield it
                        if chunk.choices[0].delta.content:
                            content_piece = chunk.choices[0].delta.content
                            response_text += content_piece
                            yield json.dumps({"type": "content", "content": content_piece}) + "\n"
                        
                        # Process tool calls
                        delta = chunk.choices[0].delta
                        if delta.tool_calls:
                            for tool_call_delta in delta.tool_calls:
                                # Initialize tool call in chunks dictionary if new
                                if tool_call_delta.index not in tool_call_chunks:
                                    tool_call_chunks[tool_call_delta.index] = {
                                        "id": "",
                                        "function": {"name": "", "arguments": ""}
                                    }
                                
                                # Update tool call data
                                if tool_call_delta.id:
                                    tool_call_chunks[tool_call_delta.index]["id"] = tool_call_delta.id
                                
                                if tool_call_delta.function:
                                    if tool_call_delta.function.name:
                                        tool_call_chunks[tool_call_delta.index]["function"]["name"] = tool_call_delta.function.name
                                    
                                    if tool_call_delta.function.arguments:
                                        tool_call_chunks[tool_call_delta.index]["function"]["arguments"] += tool_call_delta.function.arguments
                    
                    # Convert tool call chunks to actual tool calls
                    for index, tool_call_data in tool_call_chunks.items():
                        current_tool_calls.append({
                            "id": tool_call_data["id"],
                            "function": {
                                "name": tool_call_data["function"]["name"],
                                "arguments": tool_call_data["function"]["arguments"]
                            }
                        })
                    
                    # Process tool calls if any
                    if current_tool_calls:
                        # Add assistant message with tool_calls to messages
                        processed_tool_calls = []
                        for tool_call in current_tool_calls:
                            processed_tool_call = tool_call.copy()
                            processed_tool_call["type"] = "function"
                            processed_tool_calls.append(processed_tool_call)
                        
                        conversation.messages.append({
                            "role": "assistant", 
                            "content": response_text,
                            "tool_calls": processed_tool_calls
                        })
                        
                        # Notify client that tools are running
                        yield json.dumps({"type": "status", "status": "running_tools"}) + "\n"
                        
                        # Process tool calls
                        tool_responses = conversation.process_tool_calls(current_tool_calls, query=user_input)
                        
                        # Notify client of tool results
                        for response in tool_responses:
                            yield json.dumps({
                                "type": "tool_result", 
                                "tool": response["function_name"],
                                "result": response["result"]
                            }) + "\n"
                        
                        # Continue the conversation with tool responses
                        max_loop_iterations = conversation.max_tool_iterations
                        current_iteration = 0
                        
                        while current_iteration < max_loop_iterations:
                            follow_up = conversation.client.chat.completions.create(
                                model=conversation.model,
                                messages=conversation.messages,
                                tools=api_tools,
                                stream=False
                            )
                            
                            # Check if the follow-up response contains more tool calls
                            assistant_message = follow_up.choices[0].message
                            follow_up_text = assistant_message.content or ""
                            
                            # If there are no more tool calls, we're done with the loop
                            if not hasattr(assistant_message, 'tool_calls') or not assistant_message.tool_calls:
                                if follow_up_text:
                                    yield json.dumps({"type": "content", "content": follow_up_text}) + "\n"
                                
                                # Add the final assistant message to the conversation
                                conversation.messages.append({"role": "assistant", "content": follow_up_text})
                                break
                            
                            # Process the new tool calls
                            current_tool_calls = []
                            for tool_call in assistant_message.tool_calls:
                                if isinstance(tool_call, dict):
                                    processed_tool_call = tool_call.copy()
                                else:
                                    processed_tool_call = {
                                        "id": tool_call.id,
                                        "function": {
                                            "name": tool_call.function.name,
                                            "arguments": tool_call.function.arguments
                                        }
                                    }
                                
                                processed_tool_call["type"] = "function"
                                current_tool_calls.append(processed_tool_call)
                            
                            # Add the assistant message with tool calls
                            conversation.messages.append({
                                "role": "assistant",
                                "content": follow_up_text,
                                "tool_calls": current_tool_calls
                            })
                            
                            # Notify client that tools are running
                            yield json.dumps({
                                "type": "status", 
                                "status": f"running_tools_iteration_{current_iteration + 1}"
                            }) + "\n"
                            
                            # Process the new tool calls
                            tool_responses = conversation.process_tool_calls(assistant_message.tool_calls, query=user_input)
                            
                            # Notify client of tool results
                            for response in tool_responses:
                                yield json.dumps({
                                    "type": "tool_result", 
                                    "tool": response["function_name"],
                                    "result": response["result"]
                                }) + "\n"
                            
                            # Increment the iteration counter
                            current_iteration += 1
                        
                        # If we've reached the maximum number of iterations, add a warning
                        if current_iteration >= max_loop_iterations:
                            warning_message = f"Warning: Reached maximum number of tool call iterations ({max_loop_iterations}). Some operations may be incomplete."
                            yield json.dumps({"type": "warning", "warning": warning_message}) + "\n"
                    else:
                        # Add assistant response to messages
                        conversation.messages.append({"role": "assistant", "content": response_text})
                    
                    # Signal completion
                    yield json.dumps({"type": "status", "status": "complete"}) + "\n"
                    
                except Exception as e:
                    yield json.dumps({"type": "error", "error": str(e)}) + "\n"
            
            return StreamingResponse(response_generator(), media_type="text/event-stream")
        
        @self.app.get("/conversation/{conversation_id}")
        async def get_conversation(conversation_id: str):
            """Get conversation details"""
            if conversation_id not in self.conversation_pool:
                raise HTTPException(status_code=404, detail="Conversation not found")
            
            conversation = self.conversation_pool[conversation_id]
            
            return {
                "conversation_id": conversation_id,
                "model": conversation.model,
                "temperature": conversation.temperature,
                "message_count": len(conversation.messages),
                "token_usage": conversation.token_usage
            }
        
        @self.app.delete("/conversation/{conversation_id}")
        async def delete_conversation(conversation_id: str):
            """Delete a conversation"""
            if conversation_id not in self.conversation_pool:
                raise HTTPException(status_code=404, detail="Conversation not found")
            
            del self.conversation_pool[conversation_id]
            
            return {"status": "deleted", "conversation_id": conversation_id}
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "active_conversations": len(self.conversation_pool),
                "uptime": time.time() - self.start_time
            }
    
    async def _init_conversation(self, conversation_id, model, temperature):
        """Initialize a conversation instance"""
        conversation = Conversation()
        conversation.model = model
        conversation.temperature = temperature
        conversation.messages.append({"role": "system", "content": get_system_prompt()})
        
        self.conversation_pool[conversation_id] = conversation
    
    def start(self):
        """Start the API server"""
        self.start_time = time.time()
        uvicorn.run(self.app, host=self.host, port=self.port)
    
    def start_background(self):
        """Start the API server in a background thread"""
        self.start_time = time.time()
        thread = threading.Thread(target=uvicorn.run, args=(self.app,), 
                                 kwargs={"host": self.host, "port": self.port})
        thread.daemon = True
        thread.start()
        return thread

class ReplicationManager:
    """Manages replication across multiple instances"""
    
    def __init__(self, primary=True, sync_interval=60):
        self.primary = primary
        self.sync_interval = sync_interval
        self.peers = []
        self.conversation_cache = {}
        self.last_sync = time.time()
        self.sync_lock = threading.Lock()
    
    def add_peer(self, host, port):
        """Add a peer instance to replicate with"""
        peer = {"host": host, "port": port}
        if peer not in self.peers:
            self.peers.append(peer)
            return True
        return False
    
    def remove_peer(self, host, port):
        """Remove a peer instance"""
        peer = {"host": host, "port": port}
        if peer in self.peers:
            self.peers.remove(peer)
            return True
        return False
    
    def sync_conversation(self, conversation_id, conversation):
        """Sync a conversation to all peers"""
        if not self.peers:
            return
        
        # Serialize conversation
        try:
            serialized = pickle.dumps(conversation)
            
            # Calculate hash for change detection
            conversation_hash = hashlib.md5(serialized).hexdigest()
            
            # Check if conversation has changed
            if conversation_id in self.conversation_cache:
                if self.conversation_cache[conversation_id] == conversation_hash:
                    return  # No changes, skip sync
            
            # Update cache
            self.conversation_cache[conversation_id] = conversation_hash
            
            # Sync to peers
            for peer in self.peers:
                try:
                    url = f"http://{peer['host']}:{peer['port']}/sync/conversation/{conversation_id}"
                    requests.post(url, data=serialized, 
                                 headers={"Content-Type": "application/octet-stream"})
                except Exception as e:
                    logging.error(f"Failed to sync with peer {peer['host']}:{peer['port']}: {e}")
        except Exception as e:
            logging.error(f"Error serializing conversation: {e}")
    
    def start_sync_thread(self, conversation_pool):
        """Start background thread for periodic syncing"""
        def sync_worker():
            while True:
                time.sleep(self.sync_interval)
                
                with self.sync_lock:
                    for conversation_id, conversation in conversation_pool.items():
                        self.sync_conversation(conversation_id, conversation)
        
        thread = threading.Thread(target=sync_worker)
        thread.daemon = True
        thread.start()
        return thread

@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Host address to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of worker processes"),
    enable_replication: bool = typer.Option(False, "--enable-replication", help="Enable replication across instances"),
    primary: bool = typer.Option(True, "--primary/--secondary", help="Whether this is a primary or secondary instance"),
    peers: List[str] = typer.Option([], "--peer", help="Peer instances to replicate with (host:port)")
):
    """
    Start the OpenAI Code Assistant as a web service
    """
    console.print(Panel.fit(
        f"[bold green]OpenAI Code Assistant API Server[/bold green]\n"
        f"Host: {host}\n"
        f"Port: {port}\n"
        f"Workers: {workers}\n"
        f"Replication: {'Enabled' if enable_replication else 'Disabled'}\n"
        f"Role: {'Primary' if primary else 'Secondary'}\n"
        f"Peers: {', '.join(peers) if peers else 'None'}",
        title="Server Starting",
        border_style="green"
    ))
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[bold red]Error:[/bold red] No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")
        return
    
    # Start server
    if workers > 1:
        # Use multiprocessing for multiple workers
        console.print(f"Starting server with {workers} workers...")
        uvicorn.run(
            "cli:create_app",
            host=host,
            port=port,
            workers=workers,
            factory=True
        )
    else:
        # Single process mode
        hosting_manager = HostingManager(host=host, port=port)
        
        # Setup replication if enabled
        if enable_replication:
            replication_manager = ReplicationManager(primary=primary)
            
            # Add peers
            for peer in peers:
                try:
                    peer_host, peer_port = peer.split(":")
                    replication_manager.add_peer(peer_host, int(peer_port))
                except ValueError:
                    console.print(f"[yellow]Warning: Invalid peer format: {peer}. Use host:port format.[/yellow]")
            
            # Start sync thread
            replication_manager.start_sync_thread(hosting_manager.conversation_pool)
            
            console.print(f"Replication enabled with {len(replication_manager.peers)} peers")
        
        # Start server
        hosting_manager.start()

def create_app():
    """Factory function for creating the FastAPI app (used with multiple workers)"""
    hosting_manager = HostingManager()
    return hosting_manager.app

@app.command()
def mcp_serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Host address to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on"),
    dev_mode: bool = typer.Option(False, "--dev", help="Enable development mode with additional logging"),
    dependencies: List[str] = typer.Option([], "--dependencies", help="Additional Python dependencies to install"),
    env_file: str = typer.Option(None, "--env-file", help="Path to .env file with environment variables"),
    cache_type: str = typer.Option("memory", "--cache", help="Cache type: 'memory' or 'redis'"),
    redis_url: str = typer.Option(None, "--redis-url", help="Redis URL for cache (if cache_type is 'redis')"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload on code changes")
):
    """
    Start the OpenAI Code Assistant as an MCP (Model Context Protocol) server
    
    This allows the assistant to be used as a context provider for MCP clients
    like Claude Desktop or other MCP-compatible applications.
    """
    # Load environment variables from file if specified
    if env_file:
        if os.path.exists(env_file):
            load_dotenv(env_file)
            console.print(f"[green]Loaded environment variables from {env_file}[/green]")
        else:
            console.print(f"[yellow]Warning: Environment file {env_file} not found[/yellow]")
    
    # Install additional dependencies if specified
    required_deps = ["prometheus-client", "tiktoken"]
    if cache_type == "redis":
        required_deps.append("redis")
    
    all_deps = required_deps + list(dependencies)
    
    if all_deps:
        console.print(f"[bold]Installing dependencies: {', '.join(all_deps)}[/bold]")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", *all_deps])
            console.print("[green]Dependencies installed successfully[/green]")
        except Exception as e:
            console.print(f"[red]Error installing dependencies: {str(e)}[/red]")
            return
    
    # Configure logging for development mode
    if dev_mode:
        import logging
        logging.basicConfig(level=logging.DEBUG)
        console.print("[yellow]Development mode enabled with debug logging[/yellow]")
    
    # Print server information
    cache_info = f"Cache: {cache_type}"
    if cache_type == "redis" and redis_url:
        cache_info += f" ({redis_url})"
    
    console.print(Panel.fit(
        f"[bold green]OpenAI Code Assistant MCP Server[/bold green]\n"
        f"Host: {host}\n"
        f"Port: {port}\n"
        f"Development Mode: {'Enabled' if dev_mode else 'Disabled'}\n"
        f"Auto-reload: {'Enabled' if reload else 'Disabled'}\n"
        f"{cache_info}\n"
        f"API Key: {'Configured' if os.getenv('OPENAI_API_KEY') else 'Not Configured'}",
        title="MCP Server Starting",
        border_style="green"
    ))
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[bold red]Error:[/bold red] No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")
        return
    
    # Create required directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "templates"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "static"), exist_ok=True)
    
    try:
        # Import the MCP server module
        from mcp_server import MCPServer
        
        # Start the MCP server
        server = MCPServer(cache_type=cache_type, redis_url=redis_url)
        server.start(host=host, port=port, reload=reload)
    except ImportError:
        console.print("[bold red]Error:[/bold red] MCP server module not found. Make sure mcp_server.py is in the same directory.")
    except Exception as e:
        console.print(f"[bold red]Error starting MCP server:[/bold red] {str(e)}")
        if dev_mode:
            import traceback
            console.print(traceback.format_exc())

@app.command()
def mcp_client(
    server_path: str = typer.Argument(..., help="Path to the MCP server script or module"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="Model to use for reasoning"),
    host: str = typer.Option("127.0.0.1", "--host", help="Host address for the MCP server"),
    port: int = typer.Option(8000, "--port", "-p", help="Port for the MCP server")
):
    """
    Connect to an MCP server using OpenAI Code Assistant as the reasoning engine
    
    This allows using the assistant to interact with any MCP-compatible server.
    """
    console.print(Panel.fit(
        f"[bold green]OpenAI Code Assistant MCP Client[/bold green]\n"
        f"Server: {server_path}\n"
        f"Model: {model}\n"
        f"Host: {host}\n"
        f"Port: {port}",
        title="MCP Client Starting",
        border_style="green"
    ))
    
    # Check if server path exists
    if not os.path.exists(server_path):
        console.print(f"[bold red]Error:[/bold red] Server script not found at {server_path}")
        return
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[bold red]Error:[/bold red] No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")
        return
    
    try:
        # Start the server in a subprocess
        import subprocess
        import signal
        
        # Start server process
        console.print(f"[bold]Starting MCP server from {server_path}...[/bold]")
        server_process = subprocess.Popen(
            [sys.executable, server_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to start
        time.sleep(2)
        
        # Check if server started successfully
        if server_process.poll() is not None:
            console.print("[bold red]Error:[/bold red] Failed to start MCP server")
            stdout, stderr = server_process.communicate()
            console.print(f"[red]Server output:[/red]\n{stdout}\n{stderr}")
            return
        
        console.print("[green]MCP server started successfully[/green]")
        
        # Initialize conversation
        conversation = Conversation()
        conversation.model = model
        
        # Add system prompt
        conversation.messages.append({
            "role": "system", 
            "content": "You are an MCP client connecting to a Model Context Protocol server. "
                      "Use the available tools to interact with the server and help the user."
        })
        
        # Register MCP-specific tools
        mcp_tools = [
            Tool(
                name="MCPGetContext",
                description="Get context from the MCP server using a prompt template",
                parameters={
                    "type": "object",
                    "properties": {
                        "prompt_id": {
                            "type": "string",
                            "description": "ID of the prompt template to use"
                        },
                        "parameters": {
                            "type": "object",
                            "description": "Parameters for the prompt template"
                        }
                    },
                    "required": ["prompt_id"]
                },
                function=lambda prompt_id, parameters=None: _mcp_get_context(host, port, prompt_id, parameters or {})
            ),
            Tool(
                name="MCPListPrompts",
                description="List available prompt templates from the MCP server",
                parameters={
                    "type": "object",
                    "properties": {}
                },
                function=lambda: _mcp_list_prompts(host, port)
            ),
            Tool(
                name="MCPGetPrompt",
                description="Get details of a specific prompt template from the MCP server",
                parameters={
                    "type": "object",
                    "properties": {
                        "prompt_id": {
                            "type": "string",
                            "description": "ID of the prompt template to get"
                        }
                    },
                    "required": ["prompt_id"]
                },
                function=lambda prompt_id: _mcp_get_prompt(host, port, prompt_id)
            )
        ]
        
        # Add MCP tools to conversation
        conversation.tools.extend(mcp_tools)
        for tool in mcp_tools:
            conversation.tool_map[tool.name] = tool.function
        
        # Main interaction loop
        console.print("[bold]MCP Client ready. Type your questions or commands.[/bold]")
        console.print("[bold]Type 'exit' to quit.[/bold]")
        
        while True:
            try:
                user_input = Prompt.ask("\n[bold blue]>>[/bold blue]")
                
                # Handle exit
                if user_input.lower() in ("exit", "quit", "/exit", "/quit"):
                    console.print("[bold yellow]Shutting down MCP client...[/bold yellow]")
                    break
                
                # Get response
                conversation.get_response(user_input)
                
            except KeyboardInterrupt:
                console.print("\n[bold yellow]Operation cancelled by user.[/bold yellow]")
                if Prompt.ask("[bold]Exit?[/bold]", choices=["y", "n"], default="n") == "y":
                    break
                continue
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {str(e)}")
        
        # Clean up
        console.print("[bold]Stopping MCP server...[/bold]")
        server_process.terminate()
        server_process.wait(timeout=5)
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")

# MCP client helper functions
def _mcp_get_context(host, port, prompt_id, parameters):
    """Get context from MCP server"""
    try:
        url = f"http://{host}:{port}/context"
        response = requests.post(
            url,
            json={
                "prompt_id": prompt_id,
                "parameters": parameters
            }
        )
        
        if response.status_code != 200:
            return f"Error: {response.status_code} - {response.text}"
        
        data = response.json()
        return f"Context (ID: {data['context_id']}):\n\n{data['context']}"
    except Exception as e:
        return f"Error connecting to MCP server: {str(e)}"

def _mcp_list_prompts(host, port):
    """List available prompt templates from MCP server"""
    try:
        url = f"http://{host}:{port}/prompts"
        response = requests.get(url)
        
        if response.status_code != 200:
            return f"Error: {response.status_code} - {response.text}"
        
        data = response.json()
        prompts = data.get("prompts", [])
        
        if not prompts:
            return "No prompt templates available"
        
        result = "Available prompt templates:\n\n"
        for prompt in prompts:
            result += f"ID: {prompt['id']}\n"
            result += f"Description: {prompt['description']}\n"
            result += f"Parameters: {', '.join(prompt.get('parameters', {}).keys())}\n\n"
        
        return result
    except Exception as e:
        return f"Error connecting to MCP server: {str(e)}"

def _mcp_get_prompt(host, port, prompt_id):
    """Get details of a specific prompt template"""
    try:
        url = f"http://{host}:{port}/prompts/{prompt_id}"
        response = requests.get(url)
        
        if response.status_code != 200:
            return f"Error: {response.status_code} - {response.text}"
        
        prompt = response.json()
        
        result = f"Prompt Template: {prompt['id']}\n\n"
        result += f"Description: {prompt['description']}\n\n"
        result += "Parameters:\n"
        
        for param_name, param_info in prompt.get("parameters", {}).items():
            result += f"- {param_name}: {param_info.get('description', '')}\n"
        
        result += f"\nTemplate:\n{prompt['template']}\n"
        
        return result
    except Exception as e:
        return f"Error connecting to MCP server: {str(e)}"

@app.command()
def mcp_multi_agent(
    server_path: str = typer.Argument(..., help="Path to the MCP server script or module"),
    config: str = typer.Option(None, "--config", "-c", help="Path to agent configuration JSON file"),
    host: str = typer.Option("127.0.0.1", "--host", help="Host address for the MCP server"),
    port: int = typer.Option(8000, "--port", "-p", help="Port for the MCP server")
):
    """
    Start a multi-agent MCP client with multiple specialized agents
    
    This allows using multiple agents with different roles to collaborate
    on complex tasks by connecting to an MCP server.
    """
    # Load configuration
    if config:
        if not os.path.exists(config):
            console.print(f"[bold red]Error:[/bold red] Configuration file not found at {config}")
            return
        
        try:
            with open(config, 'r') as f:
                config_data = json.load(f)
        except Exception as e:
            console.print(f"[bold red]Error loading configuration:[/bold red] {str(e)}")
            return
    else:
        # Default configuration
        config_data = {
            "agents": [
                {
                    "name": "Primary",
                    "role": "primary",
                    "system_prompt": "You are a helpful assistant that uses an MCP server to provide information.",
                    "model": "gpt-4o",
                    "temperature": 0.0
                }
            ],
            "coordination": {
                "strategy": "single",
                "primary_agent": "Primary"
            },
            "settings": {
                "max_turns_per_agent": 1,
                "enable_agent_reflection": False,
                "enable_cross_agent_communication": False,
                "enable_user_selection": False
            }
        }
    
    # Display configuration
    agent_names = [agent["name"] for agent in config_data["agents"]]
    console.print(Panel.fit(
        f"[bold green]OpenAI Code Assistant Multi-Agent MCP Client[/bold green]\n"
        f"Server: {server_path}\n"
        f"Host: {host}:{port}\n"
        f"Agents: {', '.join(agent_names)}\n"
        f"Coordination: {config_data['coordination']['strategy']}\n"
        f"Primary Agent: {config_data['coordination']['primary_agent']}",
        title="Multi-Agent MCP Client Starting",
        border_style="green"
    ))
    
    # Check if server path exists
    if not os.path.exists(server_path):
        console.print(f"[bold red]Error:[/bold red] Server script not found at {server_path}")
        return
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[bold red]Error:[/bold red] No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")
        return
    
    try:
        # Start the server in a subprocess
        import subprocess
        import signal
        
        # Start server process
        console.print(f"[bold]Starting MCP server from {server_path}...[/bold]")
        server_process = subprocess.Popen(
            [sys.executable, server_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to start
        time.sleep(2)
        
        # Check if server started successfully
        if server_process.poll() is not None:
            console.print("[bold red]Error:[/bold red] Failed to start MCP server")
            stdout, stderr = server_process.communicate()
            console.print(f"[red]Server output:[/red]\n{stdout}\n{stderr}")
            return
        
        console.print("[green]MCP server started successfully[/green]")
        
        # Initialize agents
        agents = {}
        for agent_config in config_data["agents"]:
            # Create conversation for agent
            agent = Conversation()
            agent.model = agent_config.get("model", "gpt-4o")
            agent.temperature = agent_config.get("temperature", 0.0)
            
            # Add system prompt
            agent.messages.append({
                "role": "system", 
                "content": agent_config.get("system_prompt", "You are a helpful assistant.")
            })
            
            # Register MCP-specific tools
            mcp_tools = [
                Tool(
                    name="MCPGetContext",
                    description="Get context from the MCP server using a prompt template",
                    parameters={
                        "type": "object",
                        "properties": {
                            "prompt_id": {
                                "type": "string",
                                "description": "ID of the prompt template to use"
                            },
                            "parameters": {
                                "type": "object",
                                "description": "Parameters for the prompt template"
                            }
                        },
                        "required": ["prompt_id"]
                    },
                    function=lambda prompt_id, parameters=None: _mcp_get_context(host, port, prompt_id, parameters or {})
                ),
                Tool(
                    name="MCPListPrompts",
                    description="List available prompt templates from the MCP server",
                    parameters={
                        "type": "object",
                        "properties": {}
                    },
                    function=lambda: _mcp_list_prompts(host, port)
                ),
                Tool(
                    name="MCPGetPrompt",
                    description="Get details of a specific prompt template from the MCP server",
                    parameters={
                        "type": "object",
                        "properties": {
                            "prompt_id": {
                                "type": "string",
                                "description": "ID of the prompt template to get"
                            }
                        },
                        "required": ["prompt_id"]
                    },
                    function=lambda prompt_id: _mcp_get_prompt(host, port, prompt_id)
                )
            ]
            
            # Add MCP tools to agent
            agent.tools.extend(mcp_tools)
            for tool in mcp_tools:
                agent.tool_map[tool.name] = tool.function
            
            # Add agent to agents dictionary
            agents[agent_config["name"]] = {
                "config": agent_config,
                "conversation": agent,
                "history": []
            }
        
        # Get primary agent
        primary_agent_name = config_data["coordination"]["primary_agent"]
        if primary_agent_name not in agents:
            console.print(f"[bold red]Error:[/bold red] Primary agent '{primary_agent_name}' not found in configuration")
            return
        
        # Main interaction loop
        console.print("[bold]Multi-Agent MCP Client ready. Type your questions or commands.[/bold]")
        console.print("[bold]Special commands:[/bold]")
        console.print("  [blue]/agents[/blue] - List available agents")
        console.print("  [blue]/talk <agent_name> <message>[/blue] - Send message to specific agent")
        console.print("  [blue]/history[/blue] - Show conversation history")
        console.print("  [blue]/exit[/blue] - Exit the client")
        
        conversation_history = []
        
        while True:
            try:
                user_input = Prompt.ask("\n[bold blue]>>[/bold blue]")
                
                # Handle exit
                if user_input.lower() in ("exit", "quit", "/exit", "/quit"):
                    console.print("[bold yellow]Shutting down multi-agent MCP client...[/bold yellow]")
                    break
                
                # Handle special commands
                if user_input.startswith("/agents"):
                    console.print("[bold]Available Agents:[/bold]")
                    for name, agent_data in agents.items():
                        role = agent_data["config"]["role"]
                        model = agent_data["config"]["model"]
                        console.print(f"  [green]{name}[/green] ({role}, {model})")
                    continue
                
                if user_input.startswith("/history"):
                    console.print("[bold]Conversation History:[/bold]")
                    for i, entry in enumerate(conversation_history, 1):
                        if entry["role"] == "user":
                            console.print(f"[blue]{i}. User:[/blue] {entry['content']}")
                        else:
                            console.print(f"[green]{i}. {entry['agent']}:[/green] {entry['content']}")
                    continue
                
                if user_input.startswith("/talk "):
                    parts = user_input.split(" ", 2)
                    if len(parts) < 3:
                        console.print("[yellow]Usage: /talk <agent_name> <message>[/yellow]")
                        continue
                    
                    agent_name = parts[1]
                    message = parts[2]
                    
                    if agent_name not in agents:
                        console.print(f"[yellow]Agent '{agent_name}' not found. Use /agents to see available agents.[/yellow]")
                        continue
                    
                    # Add message to history
                    conversation_history.append({
                        "role": "user",
                        "content": message,
                        "target_agent": agent_name
                    })
                    
                    # Get response from specific agent
                    console.print(f"[bold]Asking {agent_name}...[/bold]")
                    agent = agents[agent_name]["conversation"]
                    response = agent.get_response(message)
                    
                    # Add response to history
                    conversation_history.append({
                        "role": "assistant",
                        "agent": agent_name,
                        "content": response
                    })
                    
                    # Add to agent's history
                    agents[agent_name]["history"].append({
                        "role": "user",
                        "content": message
                    })
                    agents[agent_name]["history"].append({
                        "role": "assistant",
                        "content": response
                    })
                    
                    continue
                
                # Regular message - use coordination strategy
                strategy = config_data["coordination"]["strategy"]
                
                # Add message to history
                conversation_history.append({
                    "role": "user",
                    "content": user_input
                })
                
                if strategy == "single" or strategy == "primary":
                    # Just use the primary agent
                    agent = agents[primary_agent_name]["conversation"]
                    response = agent.get_response(user_input)
                    
                    # Add response to history
                    conversation_history.append({
                        "role": "assistant",
                        "agent": primary_agent_name,
                        "content": response
                    })
                    
                    # Add to agent's history
                    agents[primary_agent_name]["history"].append({
                        "role": "user",
                        "content": user_input
                    })
                    agents[primary_agent_name]["history"].append({
                        "role": "assistant",
                        "content": response
                    })
                    
                elif strategy == "round_robin":
                    # Ask each agent in turn
                    console.print("[bold]Consulting all agents...[/bold]")
                    
                    for agent_name, agent_data in agents.items():
                        console.print(f"[bold]Response from {agent_name}:[/bold]")
                        agent = agent_data["conversation"]
                        response = agent.get_response(user_input)
                        
                        # Add response to history
                        conversation_history.append({
                            "role": "assistant",
                            "agent": agent_name,
                            "content": response
                        })
                        
                        # Add to agent's history
                        agent_data["history"].append({
                            "role": "user",
                            "content": user_input
                        })
                        agent_data["history"].append({
                            "role": "assistant",
                            "content": response
                        })
                
                elif strategy == "voting":
                    # Ask all agents and show all responses
                    console.print("[bold]Collecting responses from all agents...[/bold]")
                    
                    responses = {}
                    for agent_name, agent_data in agents.items():
                        agent = agent_data["conversation"]
                        response = agent.get_response(user_input)
                        responses[agent_name] = response
                        
                        # Add to agent's history
                        agent_data["history"].append({
                            "role": "user",
                            "content": user_input
                        })
                        agent_data["history"].append({
                            "role": "assistant",
                            "content": response
                        })
                    
                    # Display all responses
                    for agent_name, response in responses.items():
                        console.print(f"[bold]Response from {agent_name}:[/bold]")
                        console.print(response)
                        
                        # Add response to history
                        conversation_history.append({
                            "role": "assistant",
                            "agent": agent_name,
                            "content": response
                        })
                
                else:
                    console.print(f"[yellow]Unknown coordination strategy: {strategy}[/yellow]")
                    # Default to primary agent
                    agent = agents[primary_agent_name]["conversation"]
                    response = agent.get_response(user_input)
                    
                    # Add response to history
                    conversation_history.append({
                        "role": "assistant",
                        "agent": primary_agent_name,
                        "content": response
                    })
                    
                    # Add to agent's history
                    agents[primary_agent_name]["history"].append({
                        "role": "user",
                        "content": user_input
                    })
                    agents[primary_agent_name]["history"].append({
                        "role": "assistant",
                        "content": response
                    })
                
            except KeyboardInterrupt:
                console.print("\n[bold yellow]Operation cancelled by user.[/bold yellow]")
                if Prompt.ask("[bold]Exit?[/bold]", choices=["y", "n"], default="n") == "y":
                    break
                continue
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {str(e)}")
        
        # Clean up
        console.print("[bold]Stopping MCP server...[/bold]")
        server_process.terminate()
        server_process.wait(timeout=5)
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")

@app.command()
def main(
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="Specify the model to use"),
    temperature: float = typer.Option(DEFAULT_TEMPERATURE, "--temperature", "-t", help="Set temperature for response generation"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output with additional information"),
    enable_rl: bool = typer.Option(True, "--enable-rl/--disable-rl", help="Enable/disable reinforcement learning for tool optimization"),
    rl_update: bool = typer.Option(False, "--rl-update", help="Manually trigger an update of the RL model"),
):
    """
    OpenAI Code Assistant - A command-line coding assistant 
    that uses OpenAI APIs with function calling and streaming
    """
    # TODO: Check for updates on startup
    # TODO: Add environment setup verification
    
    # Create welcome panel with more details
    rl_status = "enabled" if enable_rl else "disabled"
    console.print(Panel.fit(
        f"[bold green]OpenAI Code Assistant[/bold green]\n"
        f"Model: {model} (Temperature: {temperature})\n"
        f"Reinforcement Learning: {rl_status}\n"
        "Type your questions or commands. Use /help for available commands.",
        title="Welcome",
        border_style="green"
    ))
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[bold red]Error:[/bold red] No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")
        console.print("You can create a .env file with your API key or set it in your environment.")
        return
    
    # Initialize conversation
    conversation = Conversation()
    
    # Override model and temperature if specified
    if model != DEFAULT_MODEL:
        conversation.model = model
    conversation.temperature = temperature
    
    # Configure verbose mode
    conversation.verbose = verbose
    
    # Configure RL mode
    if not enable_rl and hasattr(conversation, 'tool_optimizer') and conversation.tool_optimizer is not None:
        os.environ["ENABLE_TOOL_OPTIMIZATION"] = "0"
        conversation.tool_optimizer = None
        console.print("[yellow]Reinforcement learning disabled[/yellow]")
    
    # Handle manual RL update if requested
    if rl_update and hasattr(conversation, 'tool_optimizer') and conversation.tool_optimizer is not None:
        try:
            with console.status("[bold blue]Updating RL model...[/bold blue]"):
                result = conversation.tool_optimizer.optimizer.update_model()
            console.print(f"[green]RL model update result:[/green] {result['status']}")
            if 'metrics' in result:
                console.print(Panel.fit(
                    "\n".join([f"{k}: {v}" for k, v in result['metrics'].items()]),
                    title="RL Metrics",
                    border_style="blue"
                ))
        except Exception as e:
            console.print(f"[red]Error updating RL model:[/red] {e}")
    
    # Add system prompt
    conversation.messages.append({"role": "system", "content": get_system_prompt()})
    
    # TODO: Add context collection for file system and git information
    # TODO: Add session persistence to allow resuming conversations
    
    # Main interaction loop
    while True:
        try:
            user_input = Prompt.ask("\n[bold blue]>>[/bold blue]")
            
            # Handle exit
            if user_input.lower() in ("exit", "quit", "/exit", "/quit"):
                console.print("[bold yellow]Goodbye![/bold yellow]")
                break
            
            # Get response without wrapping it in a status indicator
            # This allows the streaming to work properly
            try:
                conversation.get_response(user_input)
            except Exception as e:
                console.print(f"[bold red]Error during response generation:[/bold red] {str(e)}")
                
                # Provide more helpful error messages for common issues
                if "api_key" in str(e).lower():
                    console.print("[yellow]Hint: Check your OpenAI API key.[/yellow]")
                elif "rate limit" in str(e).lower():
                    console.print("[yellow]Hint: You've hit a rate limit. Try again in a moment.[/yellow]")
                elif "context_length_exceeded" in str(e).lower() or "maximum context length" in str(e).lower():
                    console.print("[yellow]Hint: The conversation is too long. Try using /compact to reduce its size.[/yellow]")
                elif "Missing required parameter" in str(e):
                    console.print("[yellow]Hint: There's an API format issue. Try restarting the conversation.[/yellow]")
                
                # Offer recovery options
                recovery_choice = Prompt.ask(
                    "[bold]Would you like to:[/bold]",
                    choices=["continue", "debug", "compact", "restart", "exit"],
                    default="continue"
                )
                
                if recovery_choice == "debug":
                    # Show debug information
                    debug_info = {
                        "model": conversation.model,
                        "temperature": conversation.temperature,
                        "message_count": len(conversation.messages),
                        "token_usage": conversation.token_usage,
                        "conversation_id": conversation.conversation_id,
                        "session_duration": time.time() - conversation.session_start_time,
                        "tools_count": len(conversation.tools),
                        "python_version": sys.version,
                        "openai_version": OpenAI.__version__ if hasattr(OpenAI, "__version__") else "Unknown"
                    }
                    console.print(Panel(json.dumps(debug_info, indent=2), title="Debug Information", border_style="yellow"))
                elif recovery_choice == "compact":
                    # Compact the conversation
                    result = conversation.compact()
                    console.print(f"[green]{result}[/green]")
                elif recovery_choice == "restart":
                    # Restart the conversation
                    conversation = Conversation()
                    conversation.model = model
                    conversation.temperature = temperature
                    conversation.verbose = verbose
                    conversation.messages.append({"role": "system", "content": get_system_prompt()})
                    console.print("[green]Conversation restarted.[/green]")
                elif recovery_choice == "exit":
                    console.print("[bold yellow]Goodbye![/bold yellow]")
                    break
            
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Operation cancelled by user.[/bold yellow]")
            # Offer options after cancellation
            cancel_choice = Prompt.ask(
                "[bold]Would you like to:[/bold]",
                choices=["continue", "exit"],
                default="continue"
            )
            if cancel_choice == "exit":
                console.print("[bold yellow]Goodbye![/bold yellow]")
                break
            continue
        except Exception as e:
            console.print(f"[bold red]Unexpected error:[/bold red] {str(e)}")
            import traceback
            console.print(traceback.format_exc())
            # Ask if user wants to continue despite the error
            if Prompt.ask("[bold]Continue?[/bold]", choices=["y", "n"], default="y") == "n":
                break

if __name__ == "__main__":
    app()
