#!/usr/bin/env python3
# claude_code/lib/tools/search_tools.py
"""Web search and information retrieval tools."""

import os
import logging
import json
import urllib.parse
import requests
from typing import Dict, List, Optional, Any

from .base import tool, ToolRegistry

logger = logging.getLogger(__name__)


@tool(
    name="WebSearch",
    description="Search the web for information using various search engines",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query"
            },
            "engine": {
                "type": "string",
                "description": "Search engine to use (google, bing, duckduckgo)",
                "enum": ["google", "bing", "duckduckgo"]
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (max 10)"
            }
        },
        "required": ["query"]
    },
    category="search"
)
def web_search(query: str, engine: str = "google", num_results: int = 5) -> str:
    """Search the web for information.
    
    Args:
        query: Search query
        engine: Search engine to use
        num_results: Number of results to return
        
    Returns:
        Search results as formatted text
    """
    logger.info(f"Searching web for: {query} using {engine}")
    
    # Validate inputs
    if num_results > 10:
        num_results = 10  # Cap at 10 results
    
    # Get API key based on engine
    api_key = None
    if engine == "google":
        api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
        cx = os.getenv("GOOGLE_SEARCH_CX")
        if not api_key or not cx:
            return "Error: Google Search API key or CX not configured. Please set GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_CX environment variables."
    elif engine == "bing":
        api_key = os.getenv("BING_SEARCH_API_KEY")
        if not api_key:
            return "Error: Bing Search API key not configured. Please set BING_SEARCH_API_KEY environment variable."
    
    # Perform search based on engine
    try:
        if engine == "google":
            return _google_search(query, api_key, cx, num_results)
        elif engine == "bing":
            return _bing_search(query, api_key, num_results)
        elif engine == "duckduckgo":
            return _duckduckgo_search(query, num_results)
        else:
            return f"Error: Unsupported search engine: {engine}"
    except Exception as e:
        logger.exception(f"Error during web search: {str(e)}")
        return f"Error performing search: {str(e)}"


def _google_search(query: str, api_key: str, cx: str, num_results: int) -> str:
    """Perform Google search using Custom Search API."""
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cx,
        "q": query,
        "num": min(num_results, 10)
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return f"Error: Google search failed with status code {response.status_code}: {response.text}"
    
    data = response.json()
    if "items" not in data:
        return f"No results found for '{query}'"
    
    results = []
    for i, item in enumerate(data["items"], 1):
        title = item.get("title", "No title")
        link = item.get("link", "No link")
        snippet = item.get("snippet", "No description").replace("\n", " ")
        results.append(f"{i}. {title}\n   URL: {link}\n   {snippet}\n")
    
    return f"Google Search Results for '{query}':\n\n" + "\n".join(results)


def _bing_search(query: str, api_key: str, num_results: int) -> str:
    """Perform Bing search using Bing Web Search API."""
    url = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {
        "q": query,
        "count": min(num_results, 10),
        "responseFilter": "Webpages"
    }
    
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        return f"Error: Bing search failed with status code {response.status_code}: {response.text}"
    
    data = response.json()
    if "webPages" not in data or "value" not in data["webPages"]:
        return f"No results found for '{query}'"
    
    results = []
    for i, item in enumerate(data["webPages"]["value"], 1):
        title = item.get("name", "No title")
        link = item.get("url", "No link")
        snippet = item.get("snippet", "No description").replace("\n", " ")
        results.append(f"{i}. {title}\n   URL: {link}\n   {snippet}\n")
    
    return f"Bing Search Results for '{query}':\n\n" + "\n".join(results)


def _duckduckgo_search(query: str, num_results: int) -> str:
    """Perform DuckDuckGo search using their API."""
    # DuckDuckGo doesn't have an official API, but we can use their instant answer API
    url = "https://api.duckduckgo.com/"
    params = {
        "q": query,
        "format": "json",
        "no_html": 1,
        "skip_disambig": 1
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return f"Error: DuckDuckGo search failed with status code {response.status_code}: {response.text}"
    
    data = response.json()
    
    results = []
    
    # Add the abstract if available
    if data.get("Abstract"):
        results.append(f"Summary: {data['Abstract']}\n")
    
    # Add related topics
    if data.get("RelatedTopics"):
        topics = data["RelatedTopics"][:num_results]
        for i, topic in enumerate(topics, 1):
            if "Text" in topic:
                text = topic.get("Text", "No description")
                url = topic.get("FirstURL", "No URL")
                results.append(f"{i}. {text}\n   URL: {url}\n")
    
    if not results:
        return f"No results found for '{query}'"
    
    return f"DuckDuckGo Search Results for '{query}':\n\n" + "\n".join(results)


@tool(
    name="WikipediaSearch",
    description="Search Wikipedia for information on a topic",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The topic to search for"
            },
            "language": {
                "type": "string",
                "description": "Language code (e.g., 'en', 'es', 'fr')",
                "default": "en"
            }
        },
        "required": ["query"]
    },
    category="search"
)
def wikipedia_search(query: str, language: str = "en") -> str:
    """Search Wikipedia for information on a topic.
    
    Args:
        query: Topic to search for
        language: Language code
        
    Returns:
        Wikipedia article summary
    """
    logger.info(f"Searching Wikipedia for: {query} in {language}")
    
    try:
        # Wikipedia API endpoint
        url = f"https://{language}.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(query)}"
        
        response = requests.get(url)
        if response.status_code != 200:
            # Try search API if direct lookup fails
            search_url = f"https://{language}.wikipedia.org/w/api.php"
            search_params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json"
            }
            
            search_response = requests.get(search_url, params=search_params)
            if search_response.status_code != 200:
                return f"Error: Wikipedia search failed with status code {search_response.status_code}"
            
            search_data = search_response.json()
            if "query" not in search_data or "search" not in search_data["query"] or not search_data["query"]["search"]:
                return f"No Wikipedia articles found for '{query}'"
            
            # Get the first search result
            first_result = search_data["query"]["search"][0]
            title = first_result["title"]
            
            # Get the summary for the first result
            url = f"https://{language}.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
            response = requests.get(url)
            if response.status_code != 200:
                return f"Error: Wikipedia article lookup failed with status code {response.status_code}"
        
        data = response.json()
        
        # Format the response
        title = data.get("title", "Unknown")
        extract = data.get("extract", "No information available")
        url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
        
        result = f"Wikipedia: {title}\n\n{extract}\n"
        if url:
            result += f"\nSource: {url}"
        
        return result
    
    except Exception as e:
        logger.exception(f"Error during Wikipedia search: {str(e)}")
        return f"Error searching Wikipedia: {str(e)}"


def register_search_tools(registry: ToolRegistry) -> None:
    """Register all search tools with the registry.
    
    Args:
        registry: Tool registry to register with
    """
    from .base import create_tools_from_functions
    
    search_tools = [
        web_search,
        wikipedia_search
    ]
    
    create_tools_from_functions(registry, search_tools)
