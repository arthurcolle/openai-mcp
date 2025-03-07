#!/usr/bin/env python3
# claude_code/commands/multi_agent_client.py
"""Multi-agent MCP client implementation with synchronization capabilities."""

import asyncio
import sys
import os
import json
import logging
import uuid
import argparse
import time
from typing import Optional, Dict, Any, List, Set, Tuple
from contextlib import AsyncExitStack
from dataclasses import dataclass, field, asdict

from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.live import Live
from rich import print as rprint

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Console for rich output
console = Console()

@dataclass
class Agent:
    """Agent representation for multi-agent scenarios."""
    id: str
    name: str
    role: str
    model: str
    system_prompt: str
    conversation: List[Dict[str, Any]] = field(default_factory=list)
    connected_agents: Set[str] = field(default_factory=set)
    message_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    
    def __post_init__(self):
        """Initialize the conversation with system prompt."""
        self.conversation = [{
            "role": "system",
            "content": self.system_prompt
        }]

@dataclass
class Message:
    """Message for agent communication."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    sender_name: str = ""
    recipient_id: Optional[str] = None  # None means broadcast to all
    recipient_name: Optional[str] = None
    content: str = ""
    timestamp: float = field(default_factory=time.time)
    
    @classmethod
    def create(cls, sender_id: str, sender_name: str, content: str, 
               recipient_id: Optional[str] = None, recipient_name: Optional[str] = None) -> 'Message':
        """Create a new message."""
        return cls(
            sender_id=sender_id,
            sender_name=sender_name,
            recipient_id=recipient_id,
            recipient_name=recipient_name,
            content=content
        )

class AgentCoordinator:
    """Coordinates communication between multiple agents."""
    
    def __init__(self):
        """Initialize the agent coordinator."""
        self.agents: Dict[str, Agent] = {}
        self.message_history: List[Message] = []
        self.broadcast_queue: asyncio.Queue = asyncio.Queue()
    
    def add_agent(self, agent: Agent) -> None:
        """Add a new agent to the coordinator.
        
        Args:
            agent: The agent to add
        """
        self.agents[agent.id] = agent
    
    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from the coordinator.
        
        Args:
            agent_id: ID of the agent to remove
        """
        if agent_id in self.agents:
            del self.agents[agent_id]
    
    async def broadcast_message(self, message: Message) -> None:
        """Broadcast a message to all agents.
        
        Args:
            message: The message to broadcast
        """
        self.message_history.append(message)
        
        for agent_id, agent in self.agents.items():
            # Don't send message back to sender
            if agent_id != message.sender_id:
                await agent.message_queue.put(message)
                logger.debug(f"Queued message from {message.sender_name} to {agent.name}")
    
    async def send_direct_message(self, message: Message) -> None:
        """Send a message to a specific agent.
        
        Args:
            message: The message to send
        """
        self.message_history.append(message)
        
        if message.recipient_id in self.agents:
            recipient = self.agents[message.recipient_id]
            await recipient.message_queue.put(message)
            logger.debug(f"Queued direct message from {message.sender_name} to {recipient.name}")
    
    async def process_message(self, message: Message) -> None:
        """Process an incoming message and route appropriately.
        
        Args:
            message: The message to process
        """
        if message.recipient_id is None:
            # Broadcast message
            await self.broadcast_message(message)
        else:
            # Direct message
            await self.send_direct_message(message)
    
    def get_message_history_for_agent(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get conversation messages formatted for a specific agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            List of messages in the format expected by Claude
        """
        agent = self.agents.get(agent_id)
        if not agent:
            return []
        
        messages = []
        
        # Start with the agent's conversation history
        messages.extend(agent.conversation)
        
        # Add relevant messages from the message history
        for msg in self.message_history:
            # Include messages sent by this agent or addressed to this agent
            # or broadcast messages from other agents
            if (msg.sender_id == agent_id or 
                msg.recipient_id == agent_id or 
                (msg.recipient_id is None and msg.sender_id != agent_id)):
                
                if msg.sender_id == agent_id:
                    # This agent's own messages
                    messages.append({
                        "role": "assistant",
                        "content": msg.content
                    })
                else:
                    # Messages from other agents
                    sender = self.agents.get(msg.sender_id)
                    sender_name = sender.name if sender else msg.sender_name
                    
                    if msg.recipient_id is None:
                        # Broadcast message
                        messages.append({
                            "role": "user",
                            "content": f"{sender_name}: {msg.content}"
                        })
                    else:
                        # Direct message
                        messages.append({
                            "role": "user",
                            "content": f"{sender_name} (direct): {msg.content}"
                        })
        
        return messages

class MultiAgentMCPClient:
    """Multi-agent Model Context Protocol client with synchronization capabilities."""
    
    def __init__(self, config_path: str = None):
        """Initialize the multi-agent MCP client.
        
        Args:
            config_path: Path to the agent configuration file
        """
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
        self.coordinator = AgentCoordinator()
        self.available_tools = []
        
        # Configuration
        self.config_path = config_path
        self.agents_config = self._load_agents_config()
    
    def _load_agents_config(self) -> List[Dict[str, Any]]:
        """Load agent configurations from file.
        
        Returns:
            List of agent configurations
        """
        default_config = [{
            "name": "Assistant",
            "role": "general assistant",
            "model": "claude-3-5-sonnet-20241022",
            "system_prompt": "You are a helpful AI assistant participating in a multi-agent conversation. You can communicate with other agents and humans to solve complex problems."
        }]
        
        if not self.config_path:
            return default_config
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load agent configuration: {e}")
            return default_config
    
    def setup_agents(self) -> None:
        """Set up agents based on configuration."""
        for idx, agent_config in enumerate(self.agents_config):
            agent_id = str(uuid.uuid4())
            agent = Agent(
                id=agent_id,
                name=agent_config.get("name", f"Agent-{idx+1}"),
                role=agent_config.get("role", "assistant"),
                model=agent_config.get("model", "claude-3-5-sonnet-20241022"),
                system_prompt=agent_config.get("system_prompt", "You are a helpful AI assistant.")
            )
            self.coordinator.add_agent(agent)
            logger.info(f"Created agent: {agent.name} ({agent.role})")
    
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
        self.available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]
        
        tool_names = [tool.name for tool in tools]
        logger.info(f"Connected to server with tools: {tool_names}")
        console.print(Panel.fit(
            f"[bold green]Connected to MCP server[/bold green]\n"
            f"Available tools: {', '.join(tool_names)}",
            title="Connection Status",
            border_style="green"
        ))
    
    async def process_agent_query(self, agent_id: str, query: str, is_direct_message: bool = False) -> str:
        """Process a query using Claude and available tools for a specific agent.
        
        Args:
            agent_id: The ID of the agent processing the query
            query: The query to process
            is_direct_message: Whether this is a direct message from user
            
        Returns:
            The response text
        """
        agent = self.coordinator.agents.get(agent_id)
        if not agent:
            return "Error: Agent not found"
        
        # Get the conversation history for this agent
        messages = self.coordinator.get_message_history_for_agent(agent_id)
        
        # Add the current query if it's a direct message
        if is_direct_message:
            messages.append({
                "role": "user",
                "content": query
            })
        
        # Initial Claude API call
        response = self.anthropic.messages.create(
            model=agent.model,
            max_tokens=1000,
            messages=messages,
            tools=self.available_tools
        )

        # Process response and handle tool calls
        tool_results = []
        final_text = ""
        assistant_message_content = []
        
        for content in response.content:
            if content.type == 'text':
                final_text = content.text
                assistant_message_content.append(content)
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input

                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                tool_results.append({"call": tool_name, "result": result})
                console.print(f"[bold cyan]Agent {agent.name} calling tool {tool_name}[/bold cyan]")

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
                    model=agent.model,
                    max_tokens=1000,
                    messages=messages,
                    tools=self.available_tools
                )

                final_text = response.content[0].text
        
        # Create a message from the agent's response
        message = Message.create(
            sender_id=agent_id,
            sender_name=agent.name,
            content=final_text,
            recipient_id=None  # Broadcast to all
        )
        
        # Process the message
        await self.coordinator.process_message(message)
        
        return final_text
    
    async def process_user_query(self, query: str, target_agent_id: Optional[str] = None) -> None:
        """Process a query from the user and route it to agents.
        
        Args:
            query: The user query
            target_agent_id: Optional ID of a specific agent to target
        """
        # Handle special commands
        if query.startswith("/"):
            await self._handle_special_command(query)
            return
        
        if target_agent_id:
            # Direct message to a specific agent
            agent = self.coordinator.agents.get(target_agent_id)
            if not agent:
                console.print("[bold red]Error: Agent not found[/bold red]")
                return
            
            console.print(f"[bold blue]User → {agent.name}:[/bold blue] {query}")
            
            response = await self.process_agent_query(target_agent_id, query, is_direct_message=True)
            console.print(f"[bold green]{agent.name}:[/bold green] {response}")
        else:
            # Broadcast to all agents
            console.print(f"[bold blue]User (broadcast):[/bold blue] {query}")
            
            # Create a message from the user
            message = Message.create(
                sender_id="user",
                sender_name="User",
                content=query,
                recipient_id=None  # Broadcast
            )
            
            # Process the message
            await self.coordinator.process_message(message)
            
            # Process in parallel for all agents
            tasks = []
            for agent_id in self.coordinator.agents:
                tasks.append(asyncio.create_task(self.process_agent_query(agent_id, query)))
            
            # Wait for all agents to respond
            await asyncio.gather(*tasks)
    
    async def run_agent_thought_loops(self) -> None:
        """Run continuous thought loops for each agent in the background."""
        while True:
            for agent_id, agent in self.coordinator.agents.items():
                try:
                    # Check if there are new messages for this agent
                    if not agent.message_queue.empty():
                        message = await agent.message_queue.get()
                        
                        # Log the message
                        if message.recipient_id is None:
                            console.print(f"[bold cyan]{message.sender_name} (broadcast):[/bold cyan] {message.content}")
                        else:
                            console.print(f"[bold cyan]{message.sender_name} → {agent.name}:[/bold cyan] {message.content}")
                        
                        # Give the agent a chance to respond
                        await self.process_agent_query(agent_id, message.content)
                        
                        # Mark the message as processed
                        agent.message_queue.task_done()
                
                except Exception as e:
                    logger.exception(f"Error in agent thought loop for {agent.name}: {e}")
            
            # Small delay to prevent CPU hogging
            await asyncio.sleep(0.1)
    
    async def _handle_special_command(self, command: str) -> None:
        """Handle special commands.
        
        Args:
            command: The command string starting with /
        """
        parts = command.strip().split()
        cmd = parts[0].lower()
        args = parts[1:]
        
        if cmd == "/help":
            self._show_help()
        elif cmd == "/agents":
            self._show_agents()
        elif cmd == "/talk":
            if len(args) < 2:
                console.print("[bold red]Error: /talk requires agent name and message[/bold red]")
                return
            
            agent_name = args[0]
            message = " ".join(args[1:])
            
            # Find agent by name
            target_agent = None
            for agent_id, agent in self.coordinator.agents.items():
                if agent.name.lower() == agent_name.lower():
                    target_agent = agent
                    break
            
            if target_agent:
                await self.process_user_query(message, target_agent.id)
            else:
                console.print(f"[bold red]Error: Agent '{agent_name}' not found[/bold red]")
        elif cmd == "/history":
            self._show_message_history()
        elif cmd == "/quit" or cmd == "/exit":
            console.print("[bold yellow]Exiting multi-agent session...[/bold yellow]")
            sys.exit(0)
        else:
            console.print(f"[bold red]Unknown command: {cmd}[/bold red]")
            self._show_help()
    
    def _show_help(self) -> None:
        """Show help information."""
        help_text = """
# Multi-Agent MCP Client Commands

- **/help**: Show this help message
- **/agents**: List all active agents
- **/talk <agent> <message>**: Send a direct message to a specific agent
- **/history**: Show message history
- **/quit**, **/exit**: Exit the application

To broadcast a message to all agents, simply type your message without any command.
        """
        console.print(Markdown(help_text))
    
    def _show_agents(self) -> None:
        """Show information about all active agents."""
        table = Table(title="Active Agents")
        table.add_column("Name", style="cyan")
        table.add_column("Role", style="green")
        table.add_column("Model", style="blue")
        
        for agent_id, agent in self.coordinator.agents.items():
            table.add_row(agent.name, agent.role, agent.model)
        
        console.print(table)
    
    def _show_message_history(self) -> None:
        """Show the message history."""
        if not self.coordinator.message_history:
            console.print("[yellow]No messages in history yet.[/yellow]")
            return
        
        table = Table(title="Message History")
        table.add_column("Time", style="cyan")
        table.add_column("From", style="green")
        table.add_column("To", style="blue")
        table.add_column("Message", style="white")
        
        for msg in self.coordinator.message_history:
            timestamp = time.strftime("%H:%M:%S", time.localtime(msg.timestamp))
            recipient = msg.recipient_name if msg.recipient_name else "All"
            table.add_row(timestamp, msg.sender_name, recipient, msg.content[:50] + ("..." if len(msg.content) > 50 else ""))
        
        console.print(table)
    
    async def chat_loop(self) -> None:
        """Run the interactive chat loop."""
        console.print(Panel.fit(
            "[bold green]Multi-Agent MCP Client Started![/bold green]\n"
            "Type your messages to broadcast to all agents or use /help for commands.",
            title="Welcome",
            border_style="green"
        ))
        
        # Start the agent thought loop in the background
        thought_loop_task = asyncio.create_task(self.run_agent_thought_loops())
        
        try:
            # First, show active agents
            self._show_agents()
            
            # Main chat loop
            while True:
                try:
                    query = Prompt.ask("\n[bold blue]>[/bold blue]").strip()
                    
                    if not query:
                        continue
                    
                    if query.lower() == "quit" or query.lower() == "exit":
                        break
                    
                    await self.process_user_query(query)
                
                except KeyboardInterrupt:
                    console.print("\n[bold yellow]Operation cancelled.[/bold yellow]")
                    continue
                except Exception as e:
                    console.print(f"\n[bold red]Error: {str(e)}[/bold red]")
                    logger.exception("Error processing query")
        
        finally:
            # Cancel the thought loop task
            thought_loop_task.cancel()
            try:
                await thought_loop_task
            except asyncio.CancelledError:
                pass
    
    async def cleanup(self) -> None:
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
        "--config",
        type=str,
        help="Path to agent configuration JSON file"
    )


def execute(args: argparse.Namespace) -> int:
    """Execute the multi-agent client command.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    try:
        client = MultiAgentMCPClient(config_path=args.config)
        client.setup_agents()
        
        async def run_client():
            try:
                await client.connect_to_server(args.server_script)
                await client.chat_loop()
            finally:
                await client.cleanup()
        
        asyncio.run(run_client())
        return 0
    
    except Exception as e:
        logger.exception(f"Error running multi-agent MCP client: {e}")
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        return 1


def main() -> int:
    """Run the multi-agent client command as a standalone script."""
    parser = argparse.ArgumentParser(description="Run the Claude Code Multi-Agent MCP client")
    add_arguments(parser)
    args = parser.parse_args()
    return execute(args)


if __name__ == "__main__":
    sys.exit(main())