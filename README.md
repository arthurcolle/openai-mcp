# MCP Coding Assistant with support for OpenAI + other LLM Providers

A powerful Python recreation of Claude Code with enhanced real-time visualization, cost management, and Model Context Protocol (MCP) server capabilities. This tool provides a natural language interface for software development tasks with support for multiple LLM providers.

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)

## Key Features

- **Multi-Provider Support:** Works with OpenAI, Anthropic, and other LLM providers
- **Model Context Protocol Integration:** 
  - Run as an MCP server for use with Claude Desktop and other clients
  - Connect to any MCP server with the built-in MCP client
  - Multi-agent synchronization for complex problem solving
- **Real-Time Tool Visualization:** See tool execution progress and results in real-time
- **Cost Management:** Track token usage and expenses with budget controls
- **Comprehensive Tool Suite:** File operations, search, command execution, and more
- **Enhanced UI:** Rich terminal interface with progress indicators and syntax highlighting
- **Context Optimization:** Smart conversation compaction and memory management
- **Agent Coordination:** Specialized agents with different roles can collaborate on tasks

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:

```
# Choose one or more providers
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional model selection
OPENAI_MODEL=gpt-4o
ANTHROPIC_MODEL=claude-3-opus-20240229
```

## Usage

### CLI Mode

Run the CLI with the default provider (determined from available API keys):

```bash
python claude.py chat
```

Specify a provider and model:

```bash
python claude.py chat --provider openai --model gpt-4o
```

Set a budget limit to manage costs:

```bash
python claude.py chat --budget 5.00
```

### MCP Server Mode

Run as a Model Context Protocol server:

```bash
python claude.py serve
```

Start in development mode with the MCP Inspector:

```bash
python claude.py serve --dev
```

Configure host and port:

```bash
python claude.py serve --host 0.0.0.0 --port 8000
```

Specify additional dependencies:

```bash
python claude.py serve --dependencies pandas numpy
```

Load environment variables from file:

```bash
python claude.py serve --env-file .env
```

### MCP Client Mode

Connect to an MCP server using Claude as the reasoning engine:

```bash
python claude.py mcp-client path/to/server.py
```

Specify a Claude model:

```bash
python claude.py mcp-client path/to/server.py --model claude-3-5-sonnet-20241022
```

Try the included example server:

```bash
# In terminal 1 - start the server
python examples/echo_server.py

# In terminal 2 - connect with the client
python claude.py mcp-client examples/echo_server.py
```

### Multi-Agent MCP Mode

Launch a multi-agent client with synchronized agents:

```bash
python claude.py mcp-multi-agent path/to/server.py
```

Use a custom agent configuration file:

```bash
python claude.py mcp-multi-agent path/to/server.py --config examples/agents_config.json
```

Example with the echo server:

```bash
# In terminal 1 - start the server
python examples/echo_server.py

# In terminal 2 - launch the multi-agent client
python claude.py mcp-multi-agent examples/echo_server.py --config examples/agents_config.json
```

## Available Tools

- **View:** Read files with optional line limits
- **Edit:** Modify files with precise text replacement
- **Replace:** Create or overwrite files
- **GlobTool:** Find files by pattern matching
- **GrepTool:** Search file contents using regex
- **LS:** List directory contents
- **Bash:** Execute shell commands

## Chat Commands

- **/help:** Show available commands
- **/compact:** Compress conversation history to save tokens
- **/version:** Show version information
- **/providers:** List available LLM providers
- **/cost:** Show cost and usage information
- **/budget [amount]:** Set a budget limit
- **/quit, /exit:** Exit the application

## Architecture

Claude Code Python Edition is built with a modular architecture:

```
/claude_code/
  /lib/
    /providers/      # LLM provider implementations
    /tools/          # Tool implementations
    /context/        # Context management
    /ui/             # UI components
    /monitoring/     # Cost tracking & metrics
  /commands/         # CLI commands
  /config/           # Configuration management
  /util/             # Utility functions
  claude.py          # Main CLI entry point
  mcp_server.py      # Model Context Protocol server
```

## Using with Model Context Protocol

### Using Claude Code as an MCP Server

Once the MCP server is running, you can connect to it from Claude Desktop or other MCP-compatible clients:

1. Install and run the MCP server:
   ```bash
   python claude.py serve
   ```

2. Open the configuration page in your browser:
   ```
   http://localhost:8000
   ```

3. Follow the instructions to configure Claude Desktop, including:
   - Copy the JSON configuration
   - Download the auto-configured JSON file
   - Step-by-step setup instructions

### Using Claude Code as an MCP Client

To connect to any MCP server using Claude Code:

1. Ensure you have your Anthropic API key in the environment or .env file
2. Start the MCP server you want to connect to
3. Connect using the MCP client:
   ```bash
   python claude.py mcp-client path/to/server.py
   ```
4. Type queries in the interactive chat interface

### Using Multi-Agent Mode

For complex tasks, the multi-agent mode allows multiple specialized agents to collaborate:

1. Create an agent configuration file or use the provided example
2. Start your MCP server
3. Launch the multi-agent client:
   ```bash
   python claude.py mcp-multi-agent path/to/server.py --config examples/agents_config.json
   ```
4. Use the command interface to interact with multiple agents:
   - Type a message to broadcast to all agents
   - Use `/talk Agent_Name message` for direct communication
   - Use `/agents` to see all available agents
   - Use `/history` to view the conversation history

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes with tests
4. Submit a pull request

## License

MIT

## Acknowledgments

This project is inspired by Anthropic's Claude Code CLI tool, reimplemented in Python with additional features for enhanced visibility, cost management, and MCP server capabilities.# OpenAI Code Assistant

A powerful command-line and API-based coding assistant that uses OpenAI APIs with function calling and streaming.

## Features

- Interactive CLI for coding assistance
- Web API for integration with other applications
- Model Context Protocol (MCP) server implementation
- Replication support for high availability
- Tool-based architecture for extensibility
- Reinforcement learning for tool optimization
- Web client for browser-based interaction

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY=your_api_key
   ```

## Usage

### CLI Mode

Run the assistant in interactive CLI mode:

```bash
python cli.py
```

Options:
- `--model`, `-m`: Specify the model to use (default: gpt-4o)
- `--temperature`, `-t`: Set temperature for response generation (default: 0)
- `--verbose`, `-v`: Enable verbose output with additional information
- `--enable-rl/--disable-rl`: Enable/disable reinforcement learning for tool optimization
- `--rl-update`: Manually trigger an update of the RL model

### API Server Mode

Run the assistant as an API server:

```bash
python cli.py serve
```

Options:
- `--host`: Host address to bind to (default: 127.0.0.1)
- `--port`, `-p`: Port to listen on (default: 8000)
- `--workers`, `-w`: Number of worker processes (default: 1)
- `--enable-replication`: Enable replication across instances
- `--primary/--secondary`: Whether this is a primary or secondary instance
- `--peer`: Peer instances to replicate with (host:port), can be specified multiple times

### MCP Server Mode

Run the assistant as a Model Context Protocol (MCP) server:

```bash
python cli.py mcp-serve
```

Options:
- `--host`: Host address to bind to (default: 127.0.0.1)
- `--port`, `-p`: Port to listen on (default: 8000)
- `--dev`: Enable development mode with additional logging
- `--dependencies`: Additional Python dependencies to install
- `--env-file`: Path to .env file with environment variables

### MCP Client Mode

Connect to an MCP server using the assistant as the reasoning engine:

```bash
python cli.py mcp-client path/to/server.py
```

Options:
- `--model`, `-m`: Model to use for reasoning (default: gpt-4o)
- `--host`: Host address for the MCP server (default: 127.0.0.1)
- `--port`, `-p`: Port for the MCP server (default: 8000)

### Deployment Script

For easier deployment, use the provided script:

```bash
./deploy.sh --host 0.0.0.0 --port 8000 --workers 4
```

To enable replication:

```bash
# Primary instance
./deploy.sh --enable-replication --port 8000

# Secondary instance
./deploy.sh --enable-replication --secondary --port 8001 --peer 127.0.0.1:8000
```

### Web Client

To use the web client, open `web-client.html` in your browser. Make sure the API server is running.

## API Endpoints

### Standard API Endpoints

- `POST /conversation`: Create a new conversation
- `POST /conversation/{conversation_id}/message`: Send a message to a conversation
- `POST /conversation/{conversation_id}/message/stream`: Stream a message response
- `GET /conversation/{conversation_id}`: Get conversation details
- `DELETE /conversation/{conversation_id}`: Delete a conversation
- `GET /health`: Health check endpoint

### MCP Protocol Endpoints

- `GET /`: Health check (MCP protocol)
- `POST /context`: Get context for a prompt template
- `GET /prompts`: List available prompt templates
- `GET /prompts/{prompt_id}`: Get a specific prompt template
- `POST /prompts`: Create a new prompt template
- `PUT /prompts/{prompt_id}`: Update an existing prompt template
- `DELETE /prompts/{prompt_id}`: Delete a prompt template

## Replication

The replication system allows running multiple instances of the assistant with synchronized state. This provides:

- High availability
- Load balancing
- Fault tolerance

To set up replication:
1. Start a primary instance with `--enable-replication`
2. Start secondary instances with `--enable-replication --secondary --peer [primary-host:port]`

## Tools

The assistant includes various tools:
- Weather: Get current weather for a location
- View: Read files from the filesystem
- Edit: Edit files
- Replace: Write files
- Bash: Execute bash commands
- GlobTool: File pattern matching
- GrepTool: Content search
- LS: List directory contents
- JinaSearch: Web search using Jina.ai
- JinaFactCheck: Fact checking using Jina.ai
- JinaReadURL: Read and summarize webpages

## CLI Commands

- `/help`: Show help message
- `/compact`: Compact the conversation to reduce token usage
- `/status`: Show token usage and session information
- `/config`: Show current configuration settings
- `/rl-status`: Show RL tool optimizer status (if enabled)
- `/rl-update`: Update the RL model manually (if enabled)
- `/rl-stats`: Show tool usage statistics (if enabled)
