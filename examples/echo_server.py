#!/usr/bin/env python3
"""
Simple Echo MCP Server Example

This is a basic implementation of a Model Context Protocol (MCP) server
that simply echoes back the parameters it receives.
"""

import os
import json
import time
import uuid
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# MCP Protocol Models
class MCPHealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    protocol_version: str = "0.1.0"
    provider: str = "Echo MCP Server"
    models: List[str] = ["echo-model"]

class MCPContextRequest(BaseModel):
    prompt_id: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    model: Optional[str] = None
    stream: bool = False
    user: Optional[str] = None
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None

class MCPContextResponse(BaseModel):
    context: str
    context_id: str
    model: str
    usage: Dict[str, int] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MCPPromptTemplate(BaseModel):
    id: str
    template: str
    description: Optional[str] = None
    parameters: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    default_model: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MCPPromptLibraryResponse(BaseModel):
    prompts: List[MCPPromptTemplate]
    count: int

# Create FastAPI app
app = FastAPI(
    title="Echo MCP Server",
    description="A simple MCP server that echoes back parameters",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define prompt templates
prompt_templates = {
    "echo": {
        "template": "You said: {message}",
        "description": "Echoes back the message",
        "parameters": {
            "message": {
                "type": "string",
                "description": "The message to echo"
            }
        },
        "default_model": "echo-model",
        "metadata": {
            "category": "utility"
        }
    },
    "reverse": {
        "template": "Reversed: {message}",
        "description": "Reverses the message",
        "parameters": {
            "message": {
                "type": "string",
                "description": "The message to reverse"
            }
        },
        "default_model": "echo-model",
        "metadata": {
            "category": "utility"
        }
    }
}

# MCP Protocol Routes
@app.get("/", response_model=MCPHealthResponse)
async def health_check():
    """Health check endpoint required by MCP protocol"""
    return MCPHealthResponse()

@app.post("/context", response_model=MCPContextResponse)
async def get_context(request: MCPContextRequest):
    """Get context for a prompt template with parameters"""
    try:
        # Check if prompt template exists
        if request.prompt_id not in prompt_templates:
            raise HTTPException(
                status_code=404,
                detail=f"Prompt template '{request.prompt_id}' not found"
            )
        
        # Get prompt template
        template = prompt_templates[request.prompt_id]
        
        # Use default model if not specified
        model = request.model or template.get("default_model", "echo-model")
        
        # Generate context ID
        context_id = str(uuid.uuid4())
        
        # Process template with parameters
        try:
            if request.prompt_id == "echo":
                context = f"Echo: {request.parameters.get('message', '')}"
            elif request.prompt_id == "reverse":
                message = request.parameters.get('message', '')
                context = f"Reversed: {message[::-1]}"
            else:
                context = template["template"].format(**request.parameters)
        except KeyError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required parameter: {e}"
            )
        
        # Calculate token usage (simplified)
        token_estimate = len(context.split())
        usage = {
            "prompt_tokens": token_estimate,
            "completion_tokens": 0,
            "total_tokens": token_estimate
        }
        
        return MCPContextResponse(
            context=context,
            context_id=context_id,
            model=model,
            usage=usage,
            metadata={
                "prompt_id": request.prompt_id,
                "timestamp": time.time()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing context: {str(e)}"
        )

@app.get("/prompts", response_model=MCPPromptLibraryResponse)
async def get_prompts():
    """Get available prompt templates"""
    prompts = [
        MCPPromptTemplate(
            id=prompt_id,
            template=template["template"],
            description=template.get("description", ""),
            parameters=template.get("parameters", {}),
            default_model=template.get("default_model", "echo-model"),
            metadata=template.get("metadata", {})
        )
        for prompt_id, template in prompt_templates.items()
    ]
    
    return MCPPromptLibraryResponse(
        prompts=prompts,
        count=len(prompts)
    )

@app.get("/prompts/{prompt_id}", response_model=MCPPromptTemplate)
async def get_prompt(prompt_id: str):
    """Get a specific prompt template"""
    if prompt_id not in prompt_templates:
        raise HTTPException(
            status_code=404,
            detail=f"Prompt template '{prompt_id}' not found"
        )
    
    template = prompt_templates[prompt_id]
    return MCPPromptTemplate(
        id=prompt_id,
        template=template["template"],
        description=template.get("description", ""),
        parameters=template.get("parameters", {}),
        default_model=template.get("default_model", "echo-model"),
        metadata=template.get("metadata", {})
    )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions in MCP format"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "error_type": "http_error",
            "status_code": exc.status_code,
            "details": exc.detail if isinstance(exc.detail, dict) else None
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions in MCP format"""
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "error_type": "server_error",
            "status_code": 500,
            "details": None
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
