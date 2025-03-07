# Claude Code MCP Examples

This directory contains examples for using the Claude Code MCP client with different MCP servers.

## Echo Server

A simple server that provides two tools:
- `echo`: Echoes back any message sent to it
- `reverse`: Reverses any message sent to it

To run the echo server example:

1. Start the server:
```bash
python examples/echo_server.py
```

2. In a separate terminal, connect to it with the MCP client:
```bash
claude mcp-client examples/echo_server.py
```

3. Try these example queries:
   - "Echo the phrase 'hello world'"
   - "Can you reverse the text 'Claude is awesome'?"

## Multi-Agent Example

The `agents_config.json` file contains a configuration for a multi-agent setup with three specialized roles:
- **Researcher**: Focuses on finding and analyzing information
- **Coder**: Specializes in writing and debugging code
- **Critic**: Evaluates solutions and suggests improvements

To run the multi-agent example:

1. Start the echo server:
```bash
python examples/echo_server.py
```

2. In a separate terminal, launch the multi-agent client:
```bash
claude mcp-multi-agent examples/echo_server.py --config examples/agents_config.json
```

3. Try these example interactions:
   - "I need to write a function that calculates the Fibonacci sequence"
   - "/talk Researcher What are the applications of Fibonacci sequences?"
   - "/talk Critic What are the efficiency concerns with recursive Fibonacci implementations?"
   - "/agents" (to see all available agents)
   - "/history" (to view the conversation history)

## Adding Your Own Examples

Feel free to create your own MCP servers by following these steps:

1. Create a new Python file in this directory
2. Import FastMCP: `from fastmcp import FastMCP`
3. Create a server instance: `my_server = FastMCP("Server Name", description="...")`
4. Define tools using the `@my_server.tool` decorator
5. Define resources using the `@my_server.resource` decorator
6. Run your server with `my_server.run()`

### Creating Custom Agent Configurations

To create your own agent configurations:

1. Create a JSON file with an array of agent definitions:
```json
[
  {
    "name": "AgentName",
    "role": "agent specialization",
    "model": "claude model to use",
    "system_prompt": "Detailed instructions for the agent's behavior and role"
  },
  ...
]
```

2. Launch the multi-agent client with your configuration:
```bash
claude mcp-multi-agent path/to/server.py --config path/to/your_config.json
```