#!/usr/bin/env python3
# claude_code/commands/client.py
"""MCP client implementation for testing MCP servers."""

import asyncio
import sys
import os
import logging
import argparse
from typing import Optional, Dict, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class MCPClient:
    """Model Context Protocol client for testing MCP servers."""
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        """Initialize the MCP client.
        
        Args:
            model: The Claude model to use
        """
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
        self.model = model

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server.

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        logger.info(f"Connected to server with tools: {[tool.name for tool in tools]}")
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools.
        
        Args:
            query: The user query
            
        Returns:
            The response text
        """
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        # Initial Claude API call
        response = self.anthropic.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=messages,
            tools=available_tools
        )

        # Process response and handle tool calls
        tool_results = []
        final_text = []

        assistant_message_content = []
        for content in response.content:
            if content.type == 'text':
                final_text.append(content.text)
                assistant_message_content.append(content)
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input

                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                tool_results.append({"call": tool_name, "result": result})
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                assistant_message_content.append(content)
                messages.append({
                    "role": "assistant",
                    "content": assistant_message_content
                })
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": result.content
                        }
                    ]
                })

                # Get next response from Claude
                response = self.anthropic.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    messages=messages,
                    tools=available_tools
                )

                final_text.append(response.content[0].text)

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop."""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")
                logger.exception("Error processing query")

    async def cleanup(self):
        """Clean up resources."""
        await self.exit_stack.aclose()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add command-specific arguments to the parser.
    
    Args:
        parser: Argument parser
    """
    parser.add_argument(
        "server_script",
        type=str,
        help="Path to the server script (.py or .js)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-5-sonnet-20241022",
        help="Claude model to use"
    )


def execute(args: argparse.Namespace) -> int:
    """Execute the client command.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    try:
        client = MCPClient(model=args.model)
        
        async def run_client():
            try:
                await client.connect_to_server(args.server_script)
                await client.chat_loop()
            finally:
                await client.cleanup()
                
        asyncio.run(run_client())
        return 0
        
    except Exception as e:
        logger.exception(f"Error running MCP client: {e}")
        print(f"\nError: {str(e)}")
        return 1


def main() -> int:
    """Run the client command as a standalone script."""
    parser = argparse.ArgumentParser(description="Run the Claude Code MCP client")
    add_arguments(parser)
    args = parser.parse_args()
    return execute(args)


if __name__ == "__main__":
    sys.exit(main())