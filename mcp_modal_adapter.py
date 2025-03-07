import os
import json
import logging
import asyncio
import httpx
from typing import Dict, List, Optional, Any, AsyncIterator
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create FastAPI app
app = FastAPI(
    title="MCP Server Modal Adapter",
    description="Model Context Protocol server adapter for Modal OpenAI API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODAL_API_URL = os.environ.get("MODAL_API_URL", "https://your-modal-app-url.modal.run")
MODAL_API_KEY = os.environ.get("MODAL_API_KEY", "sk-modal-llm-api-key")  # Default key from modal_mcp_server.py
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "phi-4")

# MCP Protocol Models
class MCPHealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"

class MCPPromptTemplate(BaseModel):
    id: str
    name: str
    description: str
    template: str
    parameters: Dict[str, Any] = Field(default_factory=dict)

class MCPPromptLibraryResponse(BaseModel):
    prompts: List[MCPPromptTemplate]

class MCPContextResponse(BaseModel):
    context_id: str
    content: str
    model: str
    prompt_id: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

# Default prompt template
DEFAULT_TEMPLATE = MCPPromptTemplate(
    id="default",
    name="Default Template",
    description="Default prompt template for general use",
    template="{prompt}",
    parameters={"prompt": {"type": "string", "description": "The prompt to send to the model"}}
)

# In-memory prompt library
prompt_library = {
    "default": DEFAULT_TEMPLATE.dict()
}

# Health check endpoint
@app.get("/health", response_model=MCPHealthResponse)
async def health_check():
    """Health check endpoint"""
    return MCPHealthResponse()

# List prompts endpoint
@app.get("/prompts", response_model=MCPPromptLibraryResponse)
async def list_prompts():
    """List available prompt templates"""
    return MCPPromptLibraryResponse(prompts=[MCPPromptTemplate(**prompt) for prompt in prompt_library.values()])

# Get prompt endpoint
@app.get("/prompts/{prompt_id}", response_model=MCPPromptTemplate)
async def get_prompt(prompt_id: str):
    """Get a specific prompt template"""
    if prompt_id not in prompt_library:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prompt template with ID {prompt_id} not found"
        )
    return MCPPromptTemplate(**prompt_library[prompt_id])

# Get context endpoint
@app.post("/context/{prompt_id}")
async def get_context(prompt_id: str, request: Request):
    """Get context from a prompt template"""
    try:
        # Get request data
        data = await request.json()
        parameters = data.get("parameters", {})
        model = data.get("model", DEFAULT_MODEL)
        stream = data.get("stream", False)
        
        # Get prompt template
        if prompt_id not in prompt_library:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prompt template with ID {prompt_id} not found"
            )
        
        prompt_template = prompt_library[prompt_id]
        
        # Process template
        template = prompt_template["template"]
        prompt_text = template.format(**parameters)
        
        # Create OpenAI-compatible request
        openai_request = {
            "model": model,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": parameters.get("temperature", 0.7),
            "max_tokens": parameters.get("max_tokens", 1024),
            "stream": stream
        }
        
        # If streaming is requested, return a streaming response
        if stream:
            return StreamingResponse(
                stream_from_modal(openai_request),
                media_type="text/event-stream"
            )
        
        # Otherwise, make a regular request to Modal API
        async with httpx.AsyncClient(timeout=60.0) as client:
            headers = {
                "Authorization": f"Bearer {MODAL_API_KEY}",
                "Content-Type": "application/json"
            }
            
            response = await client.post(
                f"{MODAL_API_URL}/v1/chat/completions",
                json=openai_request,
                headers=headers
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Error from Modal API: {response.text}"
                )
            
            result = response.json()
            
            # Extract content from OpenAI response
            content = ""
            if "choices" in result and len(result["choices"]) > 0:
                if "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
                    content = result["choices"][0]["message"]["content"]
            
            # Create MCP response
            mcp_response = MCPContextResponse(
                context_id=result.get("id", ""),
                content=content,
                model=model,
                prompt_id=prompt_id,
                parameters=parameters
            )
            
            return mcp_response.dict()
            
    except Exception as e:
        logging.error(f"Error in get_context: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating context: {str(e)}"
        )

async def stream_from_modal(openai_request: Dict[str, Any]) -> AsyncIterator[str]:
    """Stream response from Modal API"""
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            headers = {
                "Authorization": f"Bearer {MODAL_API_KEY}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream"
            }
            
            async with client.stream(
                "POST",
                f"{MODAL_API_URL}/v1/chat/completions",
                json=openai_request,
                headers=headers
            ) as response:
                if response.status_code != 200:
                    error_detail = await response.aread()
                    yield f"data: {json.dumps({'error': f'Error from Modal API: {error_detail.decode()}'})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                
                # Process streaming response
                buffer = ""
                content_buffer = ""
                
                async for chunk in response.aiter_text():
                    buffer += chunk
                    
                    # Process complete SSE messages
                    while "\n\n" in buffer:
                        message, buffer = buffer.split("\n\n", 1)
                        
                        if message.startswith("data: "):
                            data = message[6:]  # Remove "data: " prefix
                            
                            if data == "[DONE]":
                                # End of stream, send final MCP response
                                final_response = MCPContextResponse(
                                    context_id="stream-" + str(hash(content_buffer))[:8],
                                    content=content_buffer,
                                    model=openai_request.get("model", DEFAULT_MODEL),
                                    prompt_id="default",
                                    parameters={}
                                )
                                
                                yield f"data: {json.dumps(final_response.dict())}\n\n"
                                yield "data: [DONE]\n\n"
                                return
                            
                            try:
                                # Parse JSON data
                                chunk_data = json.loads(data)
                                
                                # Extract content from chunk
                                if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                    if 'delta' in chunk_data['choices'][0] and 'content' in chunk_data['choices'][0]['delta']:
                                        content = chunk_data['choices'][0]['delta']['content']
                                        content_buffer += content
                                        
                                        # Create partial MCP response
                                        partial_response = {
                                            "context_id": "stream-" + str(hash(content_buffer))[:8],
                                            "content": content,
                                            "model": openai_request.get("model", DEFAULT_MODEL),
                                            "is_partial": True
                                        }
                                        
                                        yield f"data: {json.dumps(partial_response)}\n\n"
                                        
                            except json.JSONDecodeError:
                                logging.error(f"Invalid JSON in stream: {data}")
                
    except Exception as e:
        logging.error(f"Error in stream_from_modal: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

# Add a custom prompt template
@app.post("/prompts")
async def add_prompt(prompt: MCPPromptTemplate):
    """Add a new prompt template"""
    prompt_library[prompt.id] = prompt.dict()
    return {"status": "success", "message": f"Added prompt template with ID {prompt.id}"}

# Delete a prompt template
@app.delete("/prompts/{prompt_id}")
async def delete_prompt(prompt_id: str):
    """Delete a prompt template"""
    if prompt_id == "default":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete the default prompt template"
        )
        
    if prompt_id not in prompt_library:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prompt template with ID {prompt_id} not found"
        )
        
    del prompt_library[prompt_id]
    return {"status": "success", "message": f"Deleted prompt template with ID {prompt_id}"}

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
