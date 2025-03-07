#!/usr/bin/env python3
# claude_code/mcp_server.py
"""Model Context Protocol server implementation using FastMCP."""

import os
import logging
import platform
import sys
import uuid
import time
from typing import Dict, List, Any, Optional, Callable, Union
import pathlib
import json
from fastmcp import FastMCP, Context, Image

from claude_code.lib.tools.base import Tool, ToolRegistry
from claude_code.lib.tools.manager import ToolExecutionManager
from claude_code.lib.tools.file_tools import register_file_tools
from claude_code.lib.monitoring.server_metrics import get_metrics

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get server metrics
metrics = get_metrics()

# Create the FastMCP server
mcp = FastMCP(
    "Claude Code MCP Server",
    description="A Model Context Protocol server for Claude Code tools",
    dependencies=["fastmcp>=0.4.1", "openai", "pydantic"],
    homepage_html_file=str(pathlib.Path(__file__).parent / "examples" / "claude_mcp_config.html")
)

# Initialize tool registry and manager
tool_registry = ToolRegistry()
tool_manager = ToolExecutionManager(tool_registry)

# Register file tools
register_file_tools(tool_registry)


def setup_tools():
    """Register all tools from the tool registry with FastMCP."""
    
    # Get all registered tools
    registered_tools = tool_registry.get_all_tools()
    
    for tool_obj in registered_tools:
        # Convert the tool execution function to an MCP tool
        @mcp.tool(name=tool_obj.name, description=tool_obj.description)
        async def tool_executor(params: Dict[str, Any], ctx: Context) -> str:
            # Create a tool call in the format expected by ToolExecutionManager
            tool_call = {
                "id": ctx.request_id,
                "function": {
                    "name": tool_obj.name,
                    "arguments": str(params)
                }
            }
            
            try:
                # Log the tool call in metrics
                metrics.log_tool_call(tool_obj.name)
                
                # Execute the tool and get the result
                result = tool_obj.execute(tool_call)
                
                # Report progress when complete
                await ctx.report_progress(1, 1)
                
                return result.result
            except Exception as e:
                # Log error in metrics
                metrics.log_error(f"tool_{tool_obj.name}", str(e))
                raise


# Function to register all View resources
def register_view_resources():
    """Register file viewing as resources."""
    
    @mcp.resource("file://{file_path}")
    def get_file_content(file_path: str) -> str:
        """Get the content of a file"""
        try:
            # Log resource request
            metrics.log_resource_request(f"file://{file_path}")
            
            # Get the View tool
            view_tool = tool_registry.get_tool("View")
            if not view_tool:
                metrics.log_error("resource_error", "View tool not found")
                return "Error: View tool not found"
            
            # Execute the tool to get file content
            tool_call = {
                "id": "resource_call",
                "function": {
                    "name": "View",
                    "arguments": json.dumps({"file_path": file_path})
                }
            }
            
            result = view_tool.execute(tool_call)
            return result.result
        except Exception as e:
            metrics.log_error("resource_error", f"Error viewing file: {str(e)}")
            return f"Error: {str(e)}"


# Register file system resources
@mcp.resource("filesystem://{path}")
def list_directory(path: str) -> str:
    """List files and directories at the given path."""
    try:
        # Log resource request
        metrics.log_resource_request(f"filesystem://{path}")
        
        import os
        
        if not os.path.isabs(path):
            metrics.log_error("resource_error", f"Path must be absolute: {path}")
            return f"Error: Path must be absolute: {path}"
        
        if not os.path.exists(path):
            metrics.log_error("resource_error", f"Path does not exist: {path}")
            return f"Error: Path does not exist: {path}"
        
        if not os.path.isdir(path):
            metrics.log_error("resource_error", f"Path is not a directory: {path}")
            return f"Error: Path is not a directory: {path}"
        
        items = os.listdir(path)
        result = []
        
        for item in items:
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                result.append(f"{item}/")
            else:
                result.append(item)
        
        return "\n".join(result)
    except Exception as e:
        metrics.log_error("resource_error", f"Error listing directory: {str(e)}")
        return f"Error: {str(e)}"


# Add system information resource
@mcp.resource("system://info")
def get_system_info() -> str:
    """Get system information"""
    try:
        # Log resource request
        metrics.log_resource_request("system://info")
        
        info = {
            "os": platform.system(),
            "os_version": platform.version(),
            "python_version": sys.version,
            "hostname": platform.node(),
            "platform": platform.platform(),
            "architecture": platform.architecture(),
            "processor": platform.processor(),
            "uptime": metrics.get_uptime()
        }
        
        return "\n".join([f"{k}: {v}" for k, v in info.items()])
    except Exception as e:
        metrics.log_error("resource_error", f"Error getting system info: {str(e)}")
        return f"Error: {str(e)}"


# Add configuration resource
@mcp.resource("config://json")
def get_config_json() -> str:
    """Get Claude Desktop MCP configuration in JSON format"""
    try:
        # Log resource request
        metrics.log_resource_request("config://json")
        
        config_path = pathlib.Path(__file__).parent / "examples" / "claude_mcp_config.json"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
                # Update working directory to actual path
                current_dir = str(pathlib.Path(__file__).parent.parent.absolute())
                config["workingDirectory"] = current_dir
                
                return json.dumps(config, indent=2)
        except Exception as e:
            logger.error(f"Error reading config file: {e}")
            metrics.log_error("resource_error", f"Error reading config file: {str(e)}")
            
            return json.dumps({
                "name": "Claude Code Tools",
                "type": "local_process",
                "command": "python",
                "args": ["claude.py", "serve"],
                "workingDirectory": str(pathlib.Path(__file__).parent.parent.absolute()),
                "environment": {},
                "description": "A Model Context Protocol server for Claude Code tools"
            }, indent=2)
    except Exception as e:
        metrics.log_error("resource_error", f"Error in config resource: {str(e)}")
        return f"Error: {str(e)}"


# Add metrics resource
@mcp.resource("metrics://json")
def get_metrics_json() -> str:
    """Get server metrics in JSON format"""
    try:
        # Log resource request
        metrics.log_resource_request("metrics://json")
        
        # Get all metrics
        all_metrics = metrics.get_all_metrics()
        
        return json.dumps(all_metrics, indent=2)
    except Exception as e:
        metrics.log_error("resource_error", f"Error getting metrics: {str(e)}")
        return f"Error: {str(e)}"


# Add metrics tool
@mcp.tool(name="GetServerMetrics", description="Get server metrics and statistics")
async def get_server_metrics(metric_type: str = "all") -> str:
    """Get server metrics and statistics.
    
    Args:
        metric_type: Type of metrics to return (all, uptime, tools, resources, errors)
        
    Returns:
        The requested metrics information
    """
    try:
        # Log tool call
        metrics.log_tool_call("GetServerMetrics")
        
        if metric_type.lower() == "all":
            all_metrics = metrics.get_all_metrics()
            return json.dumps(all_metrics, indent=2)
        
        elif metric_type.lower() == "uptime":
            return f"Server uptime: {metrics.get_uptime()}"
        
        elif metric_type.lower() == "tools":
            tool_stats = metrics.get_tool_usage_stats()
            result = "Tool Usage Statistics:\n\n"
            for tool, count in sorted(tool_stats.items(), key=lambda x: x[1], reverse=True):
                result += f"- {tool}: {count} calls\n"
            return result
        
        elif metric_type.lower() == "resources":
            resource_stats = metrics.get_resource_usage_stats()
            result = "Resource Usage Statistics:\n\n"
            for resource, count in sorted(resource_stats.items(), key=lambda x: x[1], reverse=True):
                result += f"- {resource}: {count} requests\n"
            return result
        
        elif metric_type.lower() == "errors":
            error_stats = metrics.get_error_stats()
            if not error_stats:
                return "No errors recorded."
                
            result = "Error Statistics:\n\n"
            for error_type, count in sorted(error_stats.items(), key=lambda x: x[1], reverse=True):
                result += f"- {error_type}: {count} occurrences\n"
            return result
        
        elif metric_type.lower() == "activity":
            recent = metrics.get_recent_activity(15)
            result = "Recent Activity:\n\n"
            for event in recent:
                time_str = event.get("formatted_time", "unknown")
                if event["type"] == "tool":
                    result += f"[{time_str}] Tool call: {event['name']}\n"
                elif event["type"] == "resource":
                    result += f"[{time_str}] Resource request: {event['uri']}\n"
                elif event["type"] == "connection":
                    action = "connected" if event["action"] == "connect" else "disconnected"
                    result += f"[{time_str}] Client {event['client_id']} {action}\n"
                elif event["type"] == "error":
                    result += f"[{time_str}] Error ({event['error_type']}): {event['message']}\n"
            return result
            
        else:
            return f"Unknown metric type: {metric_type}. Available types: all, uptime, tools, resources, errors, activity"
    
    except Exception as e:
        metrics.log_error("tool_error", f"Error in GetServerMetrics: {str(e)}")
        return f"Error retrieving metrics: {str(e)}"


# Add connection tracking
@mcp.on_connect
async def handle_connect(ctx: Context):
    """Track client connections."""
    client_id = str(uuid.uuid4())
    ctx.client_data["id"] = client_id
    metrics.log_connection(client_id, connected=True)
    logger.info(f"Client connected: {client_id}")


@mcp.on_disconnect
async def handle_disconnect(ctx: Context):
    """Track client disconnections."""
    client_id = ctx.client_data.get("id", "unknown")
    metrics.log_connection(client_id, connected=False)
    logger.info(f"Client disconnected: {client_id}")


@mcp.tool(name="GetConfiguration", description="Get Claude Desktop configuration for this MCP server")
async def get_configuration(format: str = "json") -> str:
    """Get configuration for connecting Claude Desktop to this MCP server.
    
    Args:
        format: The format to return (json or text)
        
    Returns:
        The configuration in the requested format
    """
    if format.lower() == "json":
        return get_config_json()
    else:
        # Return text instructions
        config = json.loads(get_config_json())
        
        return f"""
To connect Claude Desktop to this MCP server:

1. Open Claude Desktop and go to Settings
2. Navigate to "Model Context Protocol" section
3. Click "Add New Server"
4. Use the following settings:
   - Name: {config['name']}
   - Type: Local Process
   - Command: {config['command']}
   - Arguments: {" ".join(config['args'])}
   - Working Directory: {config['workingDirectory']}
5. Click Save and connect to the server

You can also visit http://localhost:8000 for more detailed instructions and to download the configuration file.
"""


# Initialize MCP server
def initialize_server():
    """Initialize the MCP server with all tools and resources."""
    # Register all tools
    setup_tools()
    
    # Register resources
    register_view_resources()
    
    # Add metrics tool for server monitoring
    @mcp.tool(name="ResetServerMetrics", description="Reset server metrics tracking")
    async def reset_metrics(confirm: bool = False) -> str:
        """Reset server metrics tracking.
        
        Args:
            confirm: Confirmation flag to prevent accidental resets
            
        Returns:
            Confirmation message
        """
        if not confirm:
            return "Please set confirm=true to reset server metrics."
        
        # Log the call
        metrics.log_tool_call("ResetServerMetrics")
        
        # Reset metrics
        metrics.reset_stats()
        
        return "Server metrics have been reset successfully."
    
    logger.info("MCP server initialized with all tools and resources")
    
    return mcp


# Main function to run the server
def main():
    """Run the MCP server"""
    # Initialize the server
    server = initialize_server()
    
    # Run the server
    server.run()


if __name__ == "__main__":
    main()