"""Tools module for Claude Code Python Edition."""

from .base import Tool, ToolParameter, ToolResult, ToolRegistry, tool
from .manager import ToolExecutionManager
from .file_tools import register_file_tools
from .search_tools import register_search_tools
from .code_tools import register_code_tools
from .ai_tools import register_ai_tools

__all__ = [
    "Tool", 
    "ToolParameter", 
    "ToolResult", 
    "ToolRegistry", 
    "ToolExecutionManager", 
    "tool",
    "register_file_tools",
    "register_search_tools",
    "register_code_tools",
    "register_ai_tools"
]

def register_all_tools(registry: ToolRegistry = None) -> ToolRegistry:
    """Register all available tools with the registry.
    
    Args:
        registry: Existing registry or None to create a new one
        
    Returns:
        Tool registry with all tools registered
    """
    if registry is None:
        registry = ToolRegistry()
    
    # Register tool categories
    register_file_tools(registry)
    register_search_tools(registry)
    register_code_tools(registry)
    register_ai_tools(registry)
    
    # Load saved routines
    registry.load_routines()
    
    return registry
