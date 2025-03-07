#!/usr/bin/env python3
"""
Example Echo MCP Server for testing the Claude Code MCP client.
This server provides a simple 'echo' tool that returns whatever is sent to it.
"""

from fastmcp import FastMCP
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the MCP server
echo_server = FastMCP(
    "Echo Server",
    description="A simple echo server for testing MCP clients",
    dependencies=[]
)

@echo_server.tool(name="echo", description="Echoes back the input message")
async def echo(message: str) -> str:
    """Echo back the input message.
    
    Args:
        message: The message to echo back
        
    Returns:
        The same message
    """
    logger.info(f"Received message: {message}")
    return f"Echo: {message}"

@echo_server.tool(name="reverse", description="Reverses the input message")
async def reverse(message: str) -> str:
    """Reverse the input message.
    
    Args:
        message: The message to reverse
        
    Returns:
        The reversed message
    """
    logger.info(f"Reversing message: {message}")
    return f"Reversed: {message[::-1]}"

@echo_server.resource("echo://{message}")
def echo_resource(message: str) -> str:
    """Echo resource.
    
    Args:
        message: The message to echo
        
    Returns:
        The echoed message
    """
    return f"Resource Echo: {message}"

if __name__ == "__main__":
    # Run the server
    echo_server.run()