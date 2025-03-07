# Modal MCP Server

This project provides an OpenAI-compatible API server running on Modal.com with a Model Context Protocol (MCP) adapter.

## Components

1. **Modal OpenAI-compatible Server** (`modal_mcp_server.py`): A full-featured OpenAI-compatible API server that runs on Modal.com's infrastructure.

2. **MCP Adapter** (`mcp_modal_adapter.py`): A FastAPI server that adapts the OpenAI API to the Model Context Protocol (MCP).

3. **Deployment Script** (`deploy_modal_mcp.py`): A helper script to deploy both components.

## Features

- **OpenAI-compatible API**: Full compatibility with OpenAI's chat completions API
- **Multiple Models**: Support for various models including Llama 3, Phi-4, DeepSeek-R1, and more
- **Streaming Support**: Real-time streaming of model outputs
- **Advanced Caching**: Efficient caching of responses for improved performance
- **Rate Limiting**: Token bucket algorithm for fair API usage
- **MCP Compatibility**: Adapter for Model Context Protocol support

## Prerequisites

- Python 3.10+
- Modal.com account and CLI set up (`pip install modal`)
- FastAPI and Uvicorn (`pip install fastapi uvicorn`)
- HTTPX for async HTTP requests (`pip install httpx`)

## Installation

1. Install dependencies:

```bash
pip install modal fastapi uvicorn httpx
```

2. Set up Modal CLI:

```bash
modal token new
```

## Deployment

### Option 1: Using the deployment script

The easiest way to deploy is using the provided script:

```bash
python deploy_modal_mcp.py
```

This will:
1. Deploy the OpenAI-compatible server to Modal
2. Start the MCP adapter locally
3. Open a browser to verify the deployment

### Option 2: Manual deployment

1. Deploy the Modal server:

```bash
modal deploy modal_mcp_server.py
```

2. Note the URL of your deployed Modal app.

3. Set environment variables for the MCP adapter:

```bash
export MODAL_API_URL="https://your-modal-app-url.modal.run"
export MODAL_API_KEY="sk-modal-llm-api-key"  # Default key
export DEFAULT_MODEL="phi-4"  # Or any other supported model
```

4. Start the MCP adapter:

```bash
uvicorn mcp_modal_adapter:app --host 0.0.0.0 --port 8000
```

## Usage

### MCP API Endpoints

- `GET /health`: Health check endpoint
- `GET /prompts`: List available prompt templates
- `GET /prompts/{prompt_id}`: Get a specific prompt template
- `POST /context/{prompt_id}`: Generate context from a prompt template
- `POST /prompts`: Add a new prompt template
- `DELETE /prompts/{prompt_id}`: Delete a prompt template

### Example: Generate context

```bash
curl -X POST "http://localhost:8000/context/default" \
  -H "Content-Type: application/json" \
  -d '{
    "parameters": {
      "prompt": "Explain quantum computing in simple terms"
    },
    "model": "phi-4",
    "stream": false
  }'
```

### Example: Streaming response

```bash
curl -X POST "http://localhost:8000/context/default" \
  -H "Content-Type: application/json" \
  -d '{
    "parameters": {
      "prompt": "Write a short story about AI"
    },
    "model": "phi-4",
    "stream": true
  }'
```

## Advanced Configuration

### Adding Custom Prompt Templates

```bash
curl -X POST "http://localhost:8000/prompts" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "code-generator",
    "name": "Code Generator",
    "description": "Generates code based on a description",
    "template": "Write code in {language} that accomplishes the following: {task}",
    "parameters": {
      "language": {
        "type": "string",
        "description": "Programming language"
      },
      "task": {
        "type": "string",
        "description": "Task description"
      }
    }
  }'
```

### Using Custom Prompt Templates

```bash
curl -X POST "http://localhost:8000/context/code-generator" \
  -H "Content-Type: application/json" \
  -d '{
    "parameters": {
      "language": "Python",
      "task": "Create a function that calculates the Fibonacci sequence"
    },
    "model": "phi-4"
  }'
```

## Supported Models

- **vLLM Models**:
  - `llama3-8b`: Meta Llama 3.1 8B Instruct (quantized)
  - `mistral-7b`: Mistral 7B Instruct v0.2
  - `tiny-llama-1.1b`: TinyLlama 1.1B Chat

- **Llama.cpp Models**:
  - `deepseek-r1`: DeepSeek R1 (quantized)
  - `phi-4`: Microsoft Phi-4 (quantized)
  - `phi-2`: Microsoft Phi-2 (quantized)

## License

MIT
