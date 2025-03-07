#!/usr/bin/env python3
# claude_code/lib/tools/file_tools.py
"""File operation tools."""

import os
import logging
from typing import Dict, List, Optional, Any

from .base import tool, ToolRegistry

logger = logging.getLogger(__name__)


@tool(
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
    category="file"
)
def view_file(file_path: str, limit: Optional[int] = None, offset: Optional[int] = 0) -> str:
    """Read contents of a file.
    
    Args:
        file_path: Absolute path to the file
        limit: Maximum number of lines to read
        offset: Line number to start reading from
        
    Returns:
        File contents as a string
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        PermissionError: If the file can't be read
    """
    logger.info(f"Reading file: {file_path} (offset={offset}, limit={limit})")
    
    if not os.path.isabs(file_path):
        return f"Error: File path must be absolute: {file_path}"
    
    if not os.path.exists(file_path):
        return f"Error: File not found: {file_path}"
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
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
        
        return content
    except Exception as e:
        logger.exception(f"Error reading file: {file_path}")
        return f"Error reading file: {str(e)}"


@tool(
    name="Edit",
    description="This is a tool for editing files. For moving or renaming files, you should generally use the Bash tool with the 'mv' command instead.",
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
    needs_permission=True,
    category="file"
)
def edit_file(file_path: str, old_string: str, new_string: str) -> str:
    """Edit a file by replacing text.
    
    Args:
        file_path: Absolute path to the file
        old_string: Text to replace
        new_string: Replacement text
        
    Returns:
        Success or error message
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        PermissionError: If the file can't be modified
    """
    logger.info(f"Editing file: {file_path}")
    
    if not os.path.isabs(file_path):
        return f"Error: File path must be absolute: {file_path}"
    
    try:
        # Create directory if creating new file
        if not os.path.exists(os.path.dirname(file_path)) and old_string == "":
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
        if old_string == "" and not os.path.exists(file_path):
            # Creating new file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_string)
            return f"Created new file: {file_path}"
        
        # Reading existing file
        if not os.path.exists(file_path):
            return f"Error: File not found: {file_path}"
        
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
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
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        return f"Successfully edited {file_path}"
    
    except Exception as e:
        logger.exception(f"Error editing file: {file_path}")
        return f"Error editing file: {str(e)}"


@tool(
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
    needs_permission=True,
    category="file"
)
def replace_file(file_path: str, content: str) -> str:
    """Replace file contents or create a new file.
    
    Args:
        file_path: Absolute path to the file
        content: New content for the file
        
    Returns:
        Success or error message
        
    Raises:
        PermissionError: If the file can't be written
    """
    logger.info(f"Replacing file: {file_path}")
    
    if not os.path.isabs(file_path):
        return f"Error: File path must be absolute: {file_path}"
    
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        # Write content to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"Successfully wrote to {file_path}"
    
    except Exception as e:
        logger.exception(f"Error writing file: {file_path}")
        return f"Error writing file: {str(e)}"


def register_file_tools(registry: ToolRegistry) -> None:
    """Register all file tools with the registry.
    
    Args:
        registry: Tool registry to register with
    """
    from .base import create_tools_from_functions
    
    file_tools = [
        view_file,
        edit_file,
        replace_file
    ]
    
    create_tools_from_functions(registry, file_tools)