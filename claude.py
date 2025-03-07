#!/usr/bin/env python3
"""Main entry point for Claude Code."""

import os
import sys
import argparse
import logging
from typing import Optional, List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Main entry point for Claude Code.
    
    Returns:
        Exit code
    """
    # Create the main parser
    parser = argparse.ArgumentParser(
        description="Claude Code - A powerful LLM-powered CLI for software development"
    )
    
    # Add version information
    from claude_code import __version__
    parser.add_argument(
        "--version", 
        action="version", 
        version=f"Claude Code v{__version__}"
    )
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        help="Command to execute"
    )
    
    # Add the chat command (default)
    chat_parser = subparsers.add_parser(
        "chat", 
        help="Start an interactive chat session with Claude Code"
    )
    # Add chat-specific arguments here
    
    # Add the serve command for MCP server
    serve_parser = subparsers.add_parser(
        "serve", 
        help="Start the Claude Code MCP server"
    )
    
    # Add serve-specific arguments
    from claude_code.commands.serve import add_arguments
    add_arguments(serve_parser)
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no command specified, default to chat
    if not args.command:
        args.command = "chat"
        
    # Execute the appropriate command
    if args.command == "chat":
        # Import and run the chat command
        from claude_code.claude import main as chat_main
        return chat_main()
    elif args.command == "serve":
        # Import and run the serve command
        from claude_code.commands.serve import execute
        return execute(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())