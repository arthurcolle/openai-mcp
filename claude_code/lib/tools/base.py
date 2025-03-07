#!/usr/bin/env python3
# claude_code/lib/tools/base.py
"""Base classes for tools."""

import abc
import inspect
import time
import logging
import os
import json
from typing import Dict, List, Any, Callable, Optional, Type, Union, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class ToolParameter(BaseModel):
    """Definition of a tool parameter."""
    
    name: str
    description: str
    type: str
    required: bool = False
    
    class Config:
        """Pydantic config."""
        extra = "forbid"


class ToolResult(BaseModel):
    """Result of a tool execution."""
    
    tool_call_id: str
    name: str
    result: str
    execution_time: float
    token_usage: int = 0
    status: str = "success"
    error: Optional[str] = None
    
    class Config:
        """Pydantic config."""
        extra = "forbid"


class Routine(BaseModel):
    """Definition of a tool routine."""
    
    name: str
    description: str
    steps: List[Dict[str, Any]]
    usage_count: int = 0
    created_at: float = Field(default_factory=time.time)
    last_used_at: Optional[float] = None
    
    class Config:
        """Pydantic config."""
        extra = "allow"


class Tool(BaseModel):
    """Base class for all tools."""
    
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    needs_permission: bool = False
    category: str = "general"
    
    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True
        extra = "forbid"
    
    def execute(self, tool_call: Dict[str, Any]) -> ToolResult:
        """Execute the tool with the given parameters.
        
        Args:
            tool_call: Dictionary containing tool call information
            
        Returns:
            ToolResult with execution result
        """
        # Extract parameters
        function_name = tool_call.get("function", {}).get("name", "")
        arguments_str = tool_call.get("function", {}).get("arguments", "{}")
        tool_call_id = tool_call.get("id", "unknown")
        
        # Parse arguments
        try:
            arguments = json.loads(arguments_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse arguments: {e}")
            return ToolResult(
                tool_call_id=tool_call_id,
                name=self.name,
                result=f"Error: Failed to parse arguments: {e}",
                execution_time=0,
                status="error",
                error=str(e)
            )
        
        # Execute function
        start_time = time.time()
        try:
            result = self.function(**arguments)
            execution_time = time.time() - start_time
            
            # Convert result to string if it's not already
            if not isinstance(result, str):
                result = str(result)
            
            return ToolResult(
                tool_call_id=tool_call_id,
                name=self.name,
                result=result,
                execution_time=execution_time,
                status="success"
            )
        except Exception as e:
            execution_time = time.time() - start_time
            logger.exception(f"Error executing tool {self.name}: {e}")
            return ToolResult(
                tool_call_id=tool_call_id,
                name=self.name,
                result=f"Error: {str(e)}",
                execution_time=execution_time,
                status="error",
                error=str(e)
            )


class ToolRegistry:
    """Registry for tools."""
    
    def __init__(self):
        """Initialize the tool registry."""
        self.tools: Dict[str, Tool] = {}
        self.routines: Dict[str, Routine] = {}
        self._routine_file = os.path.join(os.path.expanduser("~"), ".claude_code", "routines.json")
    
    def register_tool(self, tool: Tool) -> None:
        """Register a tool.
        
        Args:
            tool: Tool instance to register
            
        Raises:
            ValueError: If a tool with the same name is already registered
        """
        if tool.name in self.tools:
            raise ValueError(f"Tool {tool.name} is already registered")
        
        self.tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")
    
    def register_routine(self, routine: Routine) -> None:
        """Register a routine.
        
        Args:
            routine: Routine to register
            
        Raises:
            ValueError: If a routine with the same name is already registered
        """
        if routine.name in self.routines:
            raise ValueError(f"Routine {routine.name} is already registered")
        
        self.routines[routine.name] = routine
        logger.debug(f"Registered routine: {routine.name}")
        self._save_routines()
    
    def register_routine_from_dict(self, routine_dict: Dict[str, Any]) -> None:
        """Register a routine from a dictionary.
        
        Args:
            routine_dict: Dictionary with routine data
            
        Raises:
            ValueError: If a routine with the same name is already registered
        """
        routine = Routine(**routine_dict)
        self.register_routine(routine)
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name.
        
        Args:
            name: Name of the tool
            
        Returns:
            Tool instance or None if not found
        """
        return self.tools.get(name)
    
    def get_routine(self, name: str) -> Optional[Routine]:
        """Get a routine by name.
        
        Args:
            name: Name of the routine
            
        Returns:
            Routine or None if not found
        """
        return self.routines.get(name)
    
    def get_all_tools(self) -> List[Tool]:
        """Get all registered tools.
        
        Returns:
            List of all registered tools
        """
        return list(self.tools.values())
    
    def get_all_routines(self) -> List[Routine]:
        """Get all registered routines.
        
        Returns:
            List of all registered routines
        """
        return list(self.routines.values())
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible schemas for all tools.
        
        Returns:
            List of tool schemas for OpenAI function calling
        """
        schemas = []
        for tool in self.tools.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            })
        return schemas
    
    def record_routine_usage(self, name: str) -> None:
        """Record usage of a routine.
        
        Args:
            name: Name of the routine
        """
        if name in self.routines:
            routine = self.routines[name]
            routine.usage_count += 1
            routine.last_used_at = time.time()
            self._save_routines()
    
    def _save_routines(self) -> None:
        """Save routines to file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self._routine_file), exist_ok=True)
            
            # Convert routines to dict for serialization
            routines_dict = {name: routine.dict() for name, routine in self.routines.items()}
            
            # Save to file
            with open(self._routine_file, 'w') as f:
                json.dump(routines_dict, f, indent=2)
            
            logger.debug(f"Saved {len(self.routines)} routines to {self._routine_file}")
        except Exception as e:
            logger.error(f"Error saving routines: {e}")
    
    def load_routines(self) -> None:
        """Load routines from file."""
        if not os.path.exists(self._routine_file):
            logger.debug(f"Routines file not found: {self._routine_file}")
            return
        
        try:
            with open(self._routine_file, 'r') as f:
                routines_dict = json.load(f)
            
            # Clear existing routines
            self.routines.clear()
            
            # Register each routine
            for name, routine_data in routines_dict.items():
                self.routines[name] = Routine(**routine_data)
            
            logger.debug(f"Loaded {len(self.routines)} routines from {self._routine_file}")
        except Exception as e:
            logger.error(f"Error loading routines: {e}")


@dataclass
class RoutineStep:
    """A step in a routine."""
    tool_name: str
    args: Dict[str, Any]
    condition: Optional[Dict[str, Any]] = None
    store_result: bool = False
    result_var: Optional[str] = None


@dataclass
class RoutineDefinition:
    """Definition of a routine."""
    name: str
    description: str
    steps: List[RoutineStep]


def tool(name: str, description: str, parameters: Dict[str, Any], 
         needs_permission: bool = False, category: str = "general"):
    """Decorator to register a function as a tool.
    
    Args:
        name: Name of the tool
        description: Description of the tool
        parameters: Parameter schema for the tool
        needs_permission: Whether the tool needs user permission
        category: Category of the tool
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        # Set tool metadata on the function
        func._tool_info = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "needs_permission": needs_permission,
            "category": category
        }
        return func
    return decorator


def create_tools_from_functions(registry: ToolRegistry, functions: List[Callable]) -> None:
    """Create and register tools from functions with _tool_info.
    
    Args:
        registry: Tool registry to register tools with
        functions: List of functions to create tools from
    """
    for func in functions:
        if hasattr(func, "_tool_info"):
            info = func._tool_info
            tool = Tool(
                name=info["name"],
                description=info["description"],
                parameters=info["parameters"],
                function=func,
                needs_permission=info["needs_permission"],
                category=info["category"]
            )
            registry.register_tool(tool)