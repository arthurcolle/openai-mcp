#!/usr/bin/env python3
# claude_code/commands/serve.py
"""Command to start the MCP server."""

import os
import sys
import logging
import argparse
from typing import Dict, Any, Optional, List

from claude_code.mcp_server import initialize_server

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add command-specific arguments to the parser.
    
    Args:
        parser: Argument parser
    """
    parser.add_argument(
        "--dev", 
        action="store_true", 
        help="Run in development mode with the MCP Inspector"
    )
    
    parser.add_argument(
        "--host", 
        type=str, 
        default="localhost", 
        help="Host to bind the server to"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port to bind the server to"
    )
    
    parser.add_argument(
        "--dependencies", 
        type=str, 
        nargs="*", 
        help="Additional dependencies to install"
    )
    
    parser.add_argument(
        "--env-file", 
        type=str, 
        help="Path to environment file (.env)"
    )


def execute(args: argparse.Namespace) -> int:
    """Execute the serve command.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    try:
        # Initialize the MCP server
        mcp_server = initialize_server()
        
        # Add any additional dependencies
        if args.dependencies:
            for dep in args.dependencies:
                mcp_server.dependencies.append(dep)
        
        # Load environment variables from file
        if args.env_file:
            if not os.path.exists(args.env_file):
                logger.error(f"Environment file not found: {args.env_file}")
                return 1
                
            import dotenv
            dotenv.load_dotenv(args.env_file)
        
        # Run the server
        if args.dev:
            logger.info(f"Starting MCP server in development mode on {args.host}:{args.port}")
            # Use the fastmcp dev mode
            import subprocess
            cmd = [
                "fastmcp", "dev", 
                "--module", "claude_code.mcp_server:mcp",
                "--host", args.host,
                "--port", str(args.port)
            ]
            return subprocess.call(cmd)
        else:
            # Run directly
            logger.info(f"Starting MCP server on {args.host}:{args.port}")
            logger.info(f"Visit http://{args.host}:{args.port} for Claude Desktop configuration instructions")
            
            # FastMCP.run() method signature changed to accept host/port
            try:
                mcp_server.run(host=args.host, port=args.port)
            except TypeError:
                # Fallback for older versions of FastMCP
                logger.info("Using older FastMCP version without host/port parameters")
                mcp_server.run()
                
            return 0
            
    except Exception as e:
        logger.exception(f"Error running MCP server: {e}")
        return 1


def main() -> int:
    """Run the serve command as a standalone script."""
    parser = argparse.ArgumentParser(description="Run the Claude Code MCP server")
    add_arguments(parser)
    args = parser.parse_args()
    return execute(args)


if __name__ == "__main__":
    sys.exit(main())