#!/usr/bin/env python3
# claude.py
"""Claude Code Python Edition - CLI entry point."""

import os
import sys
import logging
import argparse
from typing import Dict, List, Optional, Any
import json
import signal
from datetime import datetime

import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.logging import RichHandler
from dotenv import load_dotenv

from claude_code.lib.providers import get_provider, list_available_providers
from claude_code.lib.tools.base import ToolRegistry
from claude_code.lib.tools.manager import ToolExecutionManager
from claude_code.lib.tools.file_tools import register_file_tools
from claude_code.lib.ui.tool_visualizer import ToolCallVisualizer, MultiPanelLayout
from claude_code.lib.monitoring.cost_tracker import CostTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("claude_code")

# Load environment variables
load_dotenv()

# Get version from package
VERSION = "0.1.0"

# Create typer app
app = typer.Typer(help="Claude Code Python Edition")
console = Console()

# Global state
conversation: List[Dict[str, Any]] = []
tool_registry = ToolRegistry()
tool_manager: Optional[ToolExecutionManager] = None
cost_tracker: Optional[CostTracker] = None
visualizer: Optional[ToolCallVisualizer] = None
provider_name: str = ""
model_name: str = ""
user_config: Dict[str, Any] = {}


def initialize_tools() -> None:
    """Initialize all available tools."""
    global tool_registry, tool_manager
    
    # Create the registry and manager
    tool_registry = ToolRegistry()
    tool_manager = ToolExecutionManager(tool_registry)
    
    # Register file tools
    register_file_tools(tool_registry)
    
    # TODO: Register more tools
    # register_search_tools(tool_registry)
    # register_bash_tools(tool_registry)
    # register_agent_tools(tool_registry)
    
    logger.info(f"Initialized {len(tool_registry.get_all_tools())} tools")


def setup_visualizer() -> None:
    """Set up the tool visualizer with callbacks."""
    global tool_manager, visualizer
    
    if not tool_manager:
        return
    
    # Create visualizer
    visualizer = ToolCallVisualizer(console)
    
    # Set up callbacks
    def progress_callback(tool_call_id: str, progress: float) -> None:
        if visualizer:
            visualizer.update_progress(tool_call_id, progress)
    
    def result_callback(tool_call_id: str, result: Any) -> None:
        if visualizer:
            visualizer.complete_tool_call(tool_call_id, result)
    
    tool_manager.set_progress_callback(progress_callback)
    tool_manager.set_result_callback(result_callback)


def load_configuration() -> Dict[str, Any]:
    """Load user configuration from file."""
    config_path = os.path.expanduser("~/.config/claude_code/config.json")
    
    # Default configuration
    default_config = {
        "provider": "openai",
        "model": None,  # Use provider default
        "budget_limit": None,
        "history_file": os.path.expanduser("~/.config/claude_code/usage_history.json"),
        "ui": {
            "theme": "dark",
            "show_tool_calls": True,
            "show_cost": True
        }
    }
    
    # If configuration file doesn't exist, create it with defaults
    if not os.path.exists(config_path):
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to create default configuration: {e}")
            return default_config
    
    # Load configuration
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            # Merge with defaults for any missing keys
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
            return config
    except Exception as e:
        logger.warning(f"Failed to load configuration: {e}")
        return default_config


def handle_compact_command() -> str:
    """Handle the /compact command to compress conversation history."""
    global conversation, provider_name, model_name
    
    if not conversation:
        return "No conversation to compact."
    
    # Add a system message requesting summarization
    compact_prompt = (
        "Summarize the conversation so far, focusing on the key points, decisions, and context. "
        "Keep important details about the code and tasks. Retain critical file paths, commands, "
        "and code snippets. The summary should be concise but complete enough to continue the "
        "conversation effectively."
    )
    
    conversation.append({"role": "user", "content": compact_prompt})
    
    # Get the provider
    provider = get_provider(provider_name, model=model_name)
    
    # Make non-streaming API call for compaction
    response = provider.generate_completion(conversation, stream=False)
    
    # Extract summary
    summary = response["content"] or ""
    
    # Reset conversation with summary
    system_message = next((m for m in conversation if m["role"] == "system"), None)
    
    if system_message:
        conversation = [system_message]
    else:
        conversation = []
    
    # Add compacted context
    conversation.append({
        "role": "system", 
        "content": f"This is a compacted conversation. Previous context: {summary}"
    })
    
    return "Conversation compacted successfully."


def handle_help_command() -> str:
    """Handle the /help command."""
    help_text = """
# Claude Code Python Edition Help

## Commands
- **/help**: Show this help message
- **/compact**: Compact the conversation to reduce token usage
- **/version**: Show version information
- **/providers**: List available LLM providers
- **/cost**: Show cost and usage information
- **/budget [amount]**: Set a budget limit (e.g., /budget 5.00)
- **/quit, /exit**: Exit the application

## Routine Commands
- **/routine list**: List all available routines
- **/routine create <name> <description>**: Create a routine from recent tool executions
- **/routine run <name>**: Run a routine
- **/routine delete <name>**: Delete a routine

## Tools
Claude Code has access to these tools:
- **View**: Read files
- **Edit**: Edit files (replace text)
- **Replace**: Overwrite or create files
- **GlobTool**: Find files by pattern
- **GrepTool**: Search file contents
- **LS**: List directory contents
- **Bash**: Execute shell commands

## CLI Commands
- **claude**: Start the Claude Code assistant (main interface)
- **claude mcp-client**: Start the MCP client to connect to MCP servers
  - Usage: `claude mcp-client path/to/server.py [--model MODEL]`
- **claude mcp-multi-agent**: Start the multi-agent MCP client with synchronized agents
  - Usage: `claude mcp-multi-agent path/to/server.py [--config CONFIG_FILE]`

## Multi-Agent Commands
When using the multi-agent client:
- **/agents**: List all active agents
- **/talk <agent> <message>**: Send a direct message to a specific agent
- **/history**: Show message history
- **/help**: Show multi-agent help

## Tips
- Be specific about file paths when requesting file operations
- For complex tasks, break them down into smaller steps
- Use /compact periodically for long sessions to save tokens
- Create routines for repetitive sequences of tool operations
- In multi-agent mode, use agent specialization for complex problems
    """
    return help_text


def handle_version_command() -> str:
    """Handle the /version command."""
    import platform
    python_version = platform.python_version()
    
    version_info = f"""
# Claude Code Python Edition v{VERSION}

- Python: {python_version}
- Provider: {provider_name}
- Model: {model_name}
- Tools: {len(tool_registry.get_all_tools()) if tool_registry else 0} available
    """
    return version_info


def handle_providers_command() -> str:
    """Handle the /providers command."""
    providers = list_available_providers()
    
    providers_text = "# Available LLM Providers\n\n"
    
    for name, info in providers.items():
        providers_text += f"## {info['name']}\n"
        
        if info['available']:
            providers_text += f"- Status: Available\n"
            providers_text += f"- Current model: {info['current_model']}\n"
            providers_text += f"- Available models: {', '.join(info['models'])}\n"
        else:
            providers_text += f"- Status: Not available ({info['error']})\n"
        
        providers_text += "\n"
    
    return providers_text


def handle_cost_command() -> str:
    """Handle the /cost command."""
    global cost_tracker
    
    if not cost_tracker:
        return "Cost tracking is not available."
    
    # Generate a usage report
    return cost_tracker.generate_usage_report(format="markdown")


def handle_budget_command(args: List[str]) -> str:
    """Handle the /budget command."""
    global cost_tracker
    
    if not cost_tracker:
        return "Cost tracking is not available."
    
    if not args:
        # Show current budget
        budget = cost_tracker.check_budget()
        if not budget["has_budget"]:
            return "No budget limit is currently set."
        
        return f"Current budget: ${budget['limit']:.2f} (${budget['used']:.2f} used, ${budget['remaining']:.2f} remaining)"
    
    # Set new budget
    try:
        budget_amount = float(args[0])
        if budget_amount <= 0:
            return "Budget must be a positive number."
        
        cost_tracker.budget_limit = budget_amount
        
        # Update configuration
        user_config["budget_limit"] = budget_amount
        
        # Save configuration
        config_path = os.path.expanduser("~/.config/claude_code/config.json")
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(user_config, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save configuration: {e}")
        
        return f"Budget set to ${budget_amount:.2f}"
    
    except ValueError:
        return f"Invalid budget amount: {args[0]}"


def handle_routine_list_command() -> str:
    """Handle the /routine list command."""
    global tool_manager
    
    if not tool_manager:
        return "Tool manager is not initialized."
    
    routines = tool_manager.registry.get_all_routines()
    if not routines:
        return "No routines available."
    
    routines_text = "# Available Routines\n\n"
    
    for routine in routines:
        usage = f" (Used {routine.usage_count} times)" if routine.usage_count > 0 else ""
        last_used = ""
        if routine.last_used_at:
            last_used_time = datetime.fromtimestamp(routine.last_used_at)
            last_used = f" (Last used: {last_used_time.strftime('%Y-%m-%d %H:%M')})"
        
        routines_text += f"## {routine.name}{usage}{last_used}\n"
        routines_text += f"{routine.description}\n\n"
        routines_text += f"**Steps:** {len(routine.steps)}\n\n"
    
    return routines_text


def handle_routine_create_command(args: List[str]) -> str:
    """Handle the /routine create command."""
    global tool_manager, visualizer
    
    if not tool_manager:
        return "Tool manager is not initialized."
    
    if len(args) < 2:
        return "Usage: /routine create <name> <description>"
    
    name = args[0]
    description = " ".join(args[1:])
    
    # Get recent tool results from visualizer
    if not visualizer or not hasattr(visualizer, "recent_tool_results"):
        return "No recent tool executions to create a routine from."
    
    recent_tool_results = visualizer.recent_tool_results
    if not recent_tool_results:
        return "No recent tool executions to create a routine from."
    
    try:
        routine_id = tool_manager.create_routine_from_tool_history(
            name, description, recent_tool_results
        )
        return f"Created routine '{name}' with {len(recent_tool_results)} steps."
    except Exception as e:
        logger.exception(f"Error creating routine: {e}")
        return f"Error creating routine: {str(e)}"


def handle_routine_run_command(args: List[str]) -> str:
    """Handle the /routine run command."""
    global tool_manager, visualizer
    
    if not tool_manager:
        return "Tool manager is not initialized."
    
    if not args:
        return "Usage: /routine run <name>"
    
    name = args[0]
    
    # Check if routine exists
    routine = tool_manager.registry.get_routine(name)
    if not routine:
        return f"Routine '{name}' not found."
    
    try:
        # Execute the routine
        execution_id = tool_manager.execute_routine(name)
        
        # Wait for completion
        while True:
            routine_status = tool_manager.routine_manager.get_active_routines().get(execution_id, {})
            if routine_status.get("status") != "running":
                break
            time.sleep(0.1)
        
        # Get results
        results = tool_manager.get_routine_results(execution_id)
        if not results:
            return f"Routine '{name}' completed but returned no results."
        
        # Format results
        result_text = f"# Routine '{name}' Results\n\n"
        result_text += f"Executed {len(results)} steps:\n\n"
        
        for i, result in enumerate(results):
            status = "✅" if result.status == "success" else "❌"
            result_text += f"## Step {i+1}: {result.name} {status}\n"
            result_text += f"```\n{result.result}\n```\n\n"
        
        return result_text
    
    except Exception as e:
        logger.exception(f"Error executing routine: {e}")
        return f"Error executing routine: {str(e)}"


def handle_routine_delete_command(args: List[str]) -> str:
    """Handle the /routine delete command."""
    global tool_manager
    
    if not tool_manager:
        return "Tool manager is not initialized."
    
    if not args:
        return "Usage: /routine delete <name>"
    
    name = args[0]
    
    # Check if routine exists
    routine = tool_manager.registry.get_routine(name)
    if not routine:
        return f"Routine '{name}' not found."
    
    try:
        # Remove from registry and save
        tool_manager.registry.routines.pop(name, None)
        tool_manager.registry._save_routines()
        return f"Deleted routine '{name}'."
    except Exception as e:
        logger.exception(f"Error deleting routine: {e}")
        return f"Error deleting routine: {str(e)}"


def process_special_command(user_input: str) -> Optional[str]:
    """Process special commands starting with /."""
    # Split into command and arguments
    parts = user_input.strip().split()
    command = parts[0].lower()
    args = parts[1:]
    
    # Handle commands
    if command == "/help":
        return handle_help_command()
    elif command == "/compact":
        return handle_compact_command()
    elif command == "/version":
        return handle_version_command()
    elif command == "/providers":
        return handle_providers_command()
    elif command == "/cost":
        return handle_cost_command()
    elif command == "/budget":
        return handle_budget_command(args)
    elif command in ["/quit", "/exit"]:
        console.print("[bold yellow]Goodbye![/bold yellow]")
        sys.exit(0)
    
    # Handle routine commands
    elif command == "/routine":
        if not args:
            return "Usage: /routine [list|create|run|delete]"
        
        subcmd = args[0].lower()
        if subcmd == "list":
            return handle_routine_list_command()
        elif subcmd == "create":
            return handle_routine_create_command(args[1:])
        elif subcmd == "run":
            return handle_routine_run_command(args[1:])
        elif subcmd == "delete":
            return handle_routine_delete_command(args[1:])
        else:
            return f"Unknown routine command: {subcmd}\nUsage: /routine [list|create|run|delete]"
    
    # Not a recognized command
    return None


def process_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process tool calls and return results.
    
    Args:
        tool_calls: List of tool call dictionaries
        
    Returns:
        List of tool responses
    """
    global tool_manager, visualizer
    
    if not tool_manager:
        logger.error("Tool manager not initialized")
        return []
    
    # Add tool calls to visualizer
    if visualizer:
        for tool_call in tool_calls:
            function_name = tool_call.get("function", {}).get("name", "")
            tool_call_id = tool_call.get("id", "unknown")
            arguments_str = tool_call.get("function", {}).get("arguments", "{}")
            
            try:
                parameters = json.loads(arguments_str)
                visualizer.add_tool_call(tool_call_id, function_name, parameters)
            except json.JSONDecodeError:
                visualizer.add_tool_call(tool_call_id, function_name, {})
    
    # Execute tools in parallel
    tool_results = tool_manager.execute_tools_parallel(tool_calls)
    
    # Format results for the conversation
    tool_responses = []
    for result in tool_results:
        tool_responses.append({
            "tool_call_id": result.tool_call_id,
            "role": "tool",
            "name": result.name,
            "content": result.result
        })
    
    return tool_responses


@app.command(name="mcp-client")
def mcp_client(
    server_script: str = typer.Argument(..., help="Path to the server script (.py or .js)"),
    model: str = typer.Option("claude-3-5-sonnet-20241022", "--model", "-m", help="Claude model to use")
):
    """Run the MCP client to interact with an MCP server."""
    from claude_code.commands.client import execute as client_execute
    import argparse
    
    # Create a namespace with the arguments
    args = argparse.Namespace()
    args.server_script = server_script
    args.model = model
    
    # Execute the client
    return client_execute(args)


@app.command(name="mcp-multi-agent")
def mcp_multi_agent(
    server_script: str = typer.Argument(..., help="Path to the server script (.py or .js)"),
    config: str = typer.Option(None, "--config", "-c", help="Path to agent configuration JSON file")
):
    """Run the multi-agent MCP client with agent synchronization."""
    from claude_code.commands.multi_agent_client import execute as multi_agent_execute
    import argparse
    
    # Create a namespace with the arguments
    args = argparse.Namespace()
    args.server_script = server_script
    args.config = config
    
    # Execute the multi-agent client
    return multi_agent_execute(args)


@app.command()
def main(
    provider: str = typer.Option(None, "--provider", "-p", help="LLM provider to use"),
    model: str = typer.Option(None, "--model", "-m", help="Model to use"),
    budget: Optional[float] = typer.Option(None, "--budget", "-b", help="Budget limit in dollars"),
    system_prompt: Optional[str] = typer.Option(None, "--system", "-s", help="System prompt file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Claude Code Python Edition - A LLM-powered coding assistant."""
    global conversation, tool_registry, tool_manager, cost_tracker, visualizer
    global provider_name, model_name, user_config
    
    # Set logging level
    if verbose:
        logging.getLogger("claude_code").setLevel(logging.DEBUG)
    
    # Show welcome message
    console.print(Panel.fit(
        f"[bold green]Claude Code Python Edition v{VERSION}[/bold green]\n"
        "Type your questions or commands. Use /help for available commands.",
        title="Welcome",
        border_style="green"
    ))
    
    # Load configuration
    user_config = load_configuration()
    
    # Override with command line arguments
    if provider:
        user_config["provider"] = provider
    if model:
        user_config["model"] = model
    if budget is not None:
        user_config["budget_limit"] = budget
    
    # Set provider and model
    provider_name = user_config["provider"]
    model_name = user_config["model"]
    
    try:
        # Initialize tools
        initialize_tools()
        
        # Set up cost tracking
        cost_tracker = CostTracker(
            budget_limit=user_config["budget_limit"],
            history_file=user_config["history_file"]
        )
        
        # Get provider
        provider = get_provider(provider_name, model=model_name)
        provider_name = provider.name
        model_name = provider.current_model
        
        logger.info(f"Using {provider_name} with model {model_name}")
        
        # Set up tool visualizer
        setup_visualizer()
        if visualizer:
            visualizer.start()
        
        # Load system prompt
        system_message = ""
        if system_prompt:
            try:
                with open(system_prompt, 'r', encoding='utf-8') as f:
                    system_message = f.read()
            except Exception as e:
                logger.error(f"Failed to load system prompt: {e}")
                system_message = get_default_system_prompt()
        else:
            system_message = get_default_system_prompt()
        
        # Initialize conversation
        conversation = [{"role": "system", "content": system_message}]
        
        # Main interaction loop
        while True:
            try:
                # Get user input
                user_input = Prompt.ask("\n[bold blue]>>[/bold blue]")
                
                # Handle special commands
                if user_input.startswith("/"):
                    result = process_special_command(user_input)
                    if result:
                        console.print(Markdown(result))
                        continue
                
                # Add user message to conversation
                conversation.append({"role": "user", "content": user_input})
                
                # Get schemas for all tools
                tool_schemas = tool_registry.get_tool_schemas() if tool_registry else None
                
                # Call the LLM
                with console.status("[bold blue]Thinking...[/bold blue]", spinner="dots"):
                    # Stream the response
                    response_stream = provider.generate_completion(
                        messages=conversation,
                        tools=tool_schemas,
                        stream=True
                    )
                    
                    # Track tool calls from streaming response
                    current_content = ""
                    current_tool_calls = []
                    
                    # Process streaming response
                    for chunk in response_stream:
                        # If there's content, print it
                        if chunk.get("content"):
                            content_piece = chunk["content"]
                            current_content += content_piece
                            console.print(content_piece, end="")
                        
                        # Process tool calls
                        if chunk.get("tool_calls") and not chunk.get("delta", True):
                            # This is a complete tool call
                            current_tool_calls = chunk["tool_calls"]
                            break
                    
                    console.print()  # Add newline after content
                    
                    # Add assistant response to conversation
                    conversation.append({
                        "role": "assistant", 
                        "content": current_content,
                        "tool_calls": current_tool_calls
                    })
                    
                    # Process tool calls if any
                    if current_tool_calls:
                        console.print("[bold green]Executing tools...[/bold green]")
                        
                        # Process tool calls
                        tool_responses = process_tool_calls(current_tool_calls)
                        
                        # Add tool responses to conversation
                        conversation.extend(tool_responses)
                        
                        # Continue the conversation with tool responses
                        console.print("[bold blue]Continuing with tool results...[/bold blue]")
                        
                        follow_up = provider.generate_completion(
                            messages=conversation,
                            tools=tool_schemas,
                            stream=False
                        )
                        
                        follow_up_text = follow_up.get("content", "")
                        if follow_up_text:
                            console.print(Markdown(follow_up_text))
                            
                            # Add to conversation
                            conversation.append({
                                "role": "assistant", 
                                "content": follow_up_text
                            })
                    
                    # Track token usage and cost
                    if cost_tracker:
                        # Get token counts - this is an approximation
                        token_counts = provider.count_message_tokens(conversation[-3:])
                        cost_info = provider.cost_per_1k_tokens
                        
                        # Add request to tracker
                        cost_tracker.add_request(
                            provider=provider_name,
                            model=model_name,
                            tokens_input=token_counts["input"],
                            tokens_output=token_counts.get("output", 0) or 150,  # Estimate if not available
                            input_cost_per_1k=cost_info["input"],
                            output_cost_per_1k=cost_info["output"]
                        )
                        
                        # Check budget
                        budget_status = cost_tracker.check_budget()
                        if budget_status["has_budget"] and budget_status["status"] in ["critical", "exceeded"]:
                            console.print(f"[bold red]{budget_status['message']}[/bold red]")
                
            except KeyboardInterrupt:
                console.print("\n[bold yellow]Operation cancelled by user.[/bold yellow]")
                continue
            except Exception as e:
                logger.exception(f"Error: {str(e)}")
                console.print(f"[bold red]Error:[/bold red] {str(e)}")
    
    finally:
        # Clean up
        if visualizer:
            visualizer.stop()
        
        # Save cost history
        if cost_tracker and hasattr(cost_tracker, '_save_history'):
            cost_tracker._save_history()


def get_default_system_prompt() -> str:
    """Get the default system prompt."""
    return """You are Claude Code Python Edition, a CLI tool that helps users with software engineering tasks.
Use the available tools to assist the user with their requests.

# Tone and style
You should be concise, direct, and to the point. When you run a non-trivial bash command, 
you should explain what the command does and why you are running it.
Output text to communicate with the user; all text you output outside of tool use is displayed to the user.
Remember that your output will be displayed on a command line interface.

# Tool usage policy
- When doing file search, remember to search effectively with the available tools.
- Always use the appropriate tool for the task.
- Use parallel tool calls when appropriate to improve performance.
- NEVER commit changes unless the user explicitly asks you to.

# Routines
You have access to Routines, which are sequences of tool calls that can be created and reused.
To create a routine from recent tool executions, use `/routine create <name> <description>`.
To run a routine, use `/routine run <name>`.
Routines are ideal for repetitive task sequences like:
- Deep research across multiple sources
- Multi-step code updates across files
- Complex search and replace operations
- Data processing pipelines

# Tasks
The user will primarily request you perform software engineering tasks:
1. Solving bugs
2. Adding new functionality 
3. Refactoring code
4. Explaining code
5. Writing tests

For these tasks:
1. Use search tools to understand the codebase
2. Implement solutions using the available tools
3. Verify solutions with tests if possible
4. Run lint and typecheck commands when appropriate
5. Consider creating routines for repetitive operations

# Code style
- Follow the existing code style of the project
- Maintain consistent naming conventions
- Use appropriate libraries that are already in the project
- Add comments when code is complex or non-obvious

IMPORTANT: You should minimize output tokens as much as possible while maintaining helpfulness, 
quality, and accuracy. Answer concisely with short lines of text unless the user asks for detail.
"""


if __name__ == "__main__":
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
    
    # Run app
    app()