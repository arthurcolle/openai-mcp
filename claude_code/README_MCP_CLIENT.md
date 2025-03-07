# Claude Code MCP Client

This is an implementation of a Model Context Protocol (MCP) client for Claude Code. It allows you to connect to any MCP-compatible server and interact with it using Claude as the reasoning engine.

## Prerequisites

- Python 3.8 or later
- Anthropic API key (set in your environment or `.env` file)
- Required packages: `mcp`, `anthropic`, `python-dotenv`

## Installation

The MCP client is included as part of Claude Code. If you have Claude Code installed, you already have access to the MCP client.

If you need to install the dependencies separately:

```bash
pip install mcp anthropic python-dotenv
```

## Usage

### Command Line Interface

The MCP client can be run directly from the command line:

```bash
# Using the claude command (recommended)
claude mcp-client path/to/server.py [--model MODEL]

# Or by running the client module directly
python -m claude_code.commands.client path/to/server.py [--model MODEL]
```

### Arguments

- `server_script`: Path to the MCP server script (required, must be a `.py` or `.js` file)
- `--model`: Claude model to use (optional, defaults to `claude-3-5-sonnet-20241022`)

### Environment Variables

Create a `.env` file in your project directory with your Anthropic API key:

```
ANTHROPIC_API_KEY=your_api_key_here
```

## Features

- Connect to any MCP-compatible server (Python or JavaScript)
- Interactive chat interface
- Automatically handles tool calls between Claude and the MCP server
- Maintains conversation context
- Clean resource management with proper error handling

## Example

1. Start your MCP server (e.g., a weather server)
2. Run the MCP client targeting that server:

```bash
claude mcp-client path/to/weather_server.py
```

3. Interact with the server through the client:
```
Query: What's the weather in San Francisco?
[Claude will use the tools provided by the server to answer your query]
```

## Troubleshooting

- If the client can't find the server, double-check the path to your server script
- Ensure your environment variables are correctly set (ANTHROPIC_API_KEY)
- For Node.js servers, make sure Node.js is installed on your system
- The first response might take up to 30 seconds while the server initializes

## Extending the Client

The MCP client is designed to be modular. You can extend its functionality by:

1. Adding custom response processing
2. Implementing specific tool handling
3. Enhancing the user interface
4. Adding support for additional authentication methods

## License

Same as Claude Code