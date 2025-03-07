# Claude Code Multi-Agent MCP Client

This is an implementation of a multi-agent Model Context Protocol (MCP) client for Claude Code. It allows you to run multiple Claude-powered agents that can communicate with each other while connected to the same MCP server.

## Key Features

- **Multiple Specialized Agents**: Run agents with different roles and prompts simultaneously
- **Agent Synchronization**: Agents automatically share messages and respond to each other
- **Direct & Broadcast Messaging**: Send messages to specific agents or broadcast to all
- **Rich Interface**: Colorful terminal interface with command-based controls
- **Message History**: Track all conversations between agents
- **Customizable Roles**: Define agent specializations through configuration files

## Prerequisites

- Python 3.8 or later
- Anthropic API key (set in your environment or `.env` file)
- Required packages: `mcp`, `anthropic`, `dotenv`, `rich`

## Usage

### Command Line Interface

The multi-agent client can be run directly from the command line:

```bash
# Using the claude command (recommended)
claude mcp-multi-agent path/to/server.py [--config CONFIG_FILE]

# Or by running the client module directly
python -m claude_code.commands.multi_agent_client path/to/server.py [--config CONFIG_FILE]
```

### Arguments

- `server_script`: Path to the MCP server script (required, must be a `.py` or `.js` file)
- `--config`: Path to agent configuration JSON file (optional, default uses a single assistant agent)

### Environment Variables

Create a `.env` file in your project directory with your Anthropic API key:

```
ANTHROPIC_API_KEY=your_api_key_here
```

## Agent Configuration

Create a JSON file to define your agents:

```json
[
  {
    "name": "Researcher",
    "role": "research specialist",
    "model": "claude-3-5-sonnet-20241022",
    "system_prompt": "You are a research specialist participating in a multi-agent conversation. Your primary role is to find information, analyze data, and provide well-researched answers."
  },
  {
    "name": "Coder",
    "role": "programming expert",
    "model": "claude-3-5-sonnet-20241022",
    "system_prompt": "You are a coding expert participating in a multi-agent conversation. Your primary role is to write, debug, and explain code."
  }
]
```

## Interactive Commands

When running the multi-agent client, you can use these commands:

- `/help`: Show available commands
- `/agents`: List all active agents
- `/talk <agent> <message>`: Send a direct message to a specific agent
- `/history`: Show message history
- `/quit`, `/exit`: Exit the application

To broadcast a message to all agents, simply type your message without any command.

## Example Session

This is a sample session with the multi-agent client:

1. Start a server:
   ```bash
   python examples/echo_server.py
   ```

2. Start the multi-agent client:
   ```bash
   claude mcp-multi-agent examples/echo_server.py --config examples/agents_config.json
   ```

3. Broadcast a message to all agents:
   ```
   > I need to analyze some data and then create a visualization
   ```

4. Send a direct message to the researcher agent:
   ```
   > /talk Researcher What statistical methods would be best for this analysis?
   ```

5. View the message history:
   ```
   > /history
   ```

## Use Cases

The multi-agent client is particularly useful for:

1. **Complex Problem Solving**: Break down problems into parts handled by specialized agents
2. **Collaborative Development**: Use a researcher, coder, and critic to develop better solutions
3. **Debate and Refinement**: Have agents with different perspectives refine ideas
4. **Automated Workflows**: Set up agents that collaborate on tasks without human intervention
5. **Education**: Create teaching scenarios where agents play different roles

## Troubleshooting

- If agents aren't responding to each other, check for errors in your configuration file
- For better performance, use smaller models for simple agents
- Make sure your Anthropic API key has sufficient quota for multiple simultaneous requests
- Use the `/history` command to debug message flow between agents

## License

Same as Claude Code