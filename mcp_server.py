#!/usr/bin/env python3
"""
Model Context Protocol (MCP) Server Implementation

This module implements the Model Context Protocol server capabilities,
allowing the assistant to be used as an MCP-compatible context provider.
"""

import os
import json
import time
import uuid
import sys
import logging
import asyncio
import tiktoken
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from fastapi import FastAPI, HTTPException, Request, Response, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import uvicorn
import openai
from openai import OpenAI
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mcp_server")

# MCP Protocol Models
class MCPHealthResponse(BaseModel):
    """Health check response for MCP protocol"""
    status: str = "healthy"
    version: str = "1.0.0"
    protocol_version: str = "0.1.0"
    provider: str = "OpenAI Code Assistant"
    models: List[str] = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "o1", "o1-mini"]
    uptime: Optional[float] = None
    request_count: Optional[int] = None
    cache_hit_ratio: Optional[float] = None

class MCPContextRequest(BaseModel):
    """Request for context generation from a prompt template"""
    prompt_id: str
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters to fill in the prompt template")
    model: Optional[str] = Field(None, description="Model to use for context generation")
    stream: bool = Field(False, description="Whether to stream the response")
    user: Optional[str] = Field(None, description="User identifier for tracking")
    conversation_id: Optional[str] = Field(None, description="Conversation identifier")
    message_id: Optional[str] = Field(None, description="Message identifier")

class MCPContextResponse(BaseModel):
    """Response containing generated context"""
    context: str = Field(..., description="The generated context")
    context_id: str = Field(..., description="Unique identifier for this context")
    model: str = Field(..., description="Model used for generation")
    usage: Dict[str, int] = Field(default_factory=dict, description="Token usage statistics")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class MCPErrorResponse(BaseModel):
    """Error response format"""
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
    status_code: int = Field(..., description="HTTP status code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

class MCPPromptTemplate(BaseModel):
    """Prompt template definition"""
    id: str = Field(..., description="Unique identifier for the template")
    template: str = Field(..., description="The prompt template with parameter placeholders")
    description: Optional[str] = Field(None, description="Description of the template")
    parameters: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Parameter definitions")
    default_model: Optional[str] = Field(None, description="Default model to use with this template")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class MCPPromptLibraryResponse(BaseModel):
    """Response containing a list of prompt templates"""
    prompts: List[MCPPromptTemplate] = Field(..., description="List of prompt templates")
    count: int = Field(..., description="Number of templates")

# MCP Server Implementation
# Prometheus metrics
REQUEST_COUNT = Counter('mcp_requests_total', 'Total number of requests processed', ['endpoint', 'status'])
REQUEST_LATENCY = Histogram('mcp_request_latency_seconds', 'Request latency in seconds', ['endpoint'])
CACHE_HIT = Counter('mcp_cache_hits_total', 'Total number of cache hits')
CACHE_MISS = Counter('mcp_cache_misses_total', 'Total number of cache misses')
ACTIVE_CONNECTIONS = Gauge('mcp_active_connections', 'Number of active connections')
TOKEN_USAGE = Counter('mcp_token_usage_total', 'Total number of tokens used', ['model', 'type'])

# Cache implementation
class CacheManager:
    """Manages caching for context responses"""
    
    def __init__(self, cache_type="memory", redis_url=None, ttl=3600):
        self.cache_type = cache_type
        self.redis_url = redis_url
        self.ttl = ttl
        self.memory_cache = {}
        self.redis_client = None
        
        if cache_type == "redis" and redis_url:
            try:
                import redis
                self.redis_client = redis.from_url(redis_url)
                logging.info(f"Redis cache initialized with URL: {redis_url}")
            except ImportError:
                logging.warning("Redis package not installed. Falling back to memory cache.")
                self.cache_type = "memory"
            except Exception as e:
                logging.error(f"Failed to connect to Redis: {str(e)}")
                self.cache_type = "memory"
    
    async def get(self, key):
        """Get item from cache"""
        if self.cache_type == "redis" and self.redis_client:
            try:
                value = self.redis_client.get(key)
                if value:
                    CACHE_HIT.inc()
                    return json.loads(value)
                CACHE_MISS.inc()
                return None
            except Exception as e:
                logging.error(f"Redis get error: {str(e)}")
                CACHE_MISS.inc()
                return None
        else:
            # Memory cache
            if key in self.memory_cache:
                if time.time() - self.memory_cache[key]["timestamp"] < self.ttl:
                    CACHE_HIT.inc()
                    return self.memory_cache[key]["data"]
                else:
                    # Expired
                    del self.memory_cache[key]
            CACHE_MISS.inc()
            return None
    
    async def set(self, key, value, ttl=None):
        """Set item in cache"""
        if ttl is None:
            ttl = self.ttl
            
        if self.cache_type == "redis" and self.redis_client:
            try:
                self.redis_client.setex(key, ttl, json.dumps(value))
            except Exception as e:
                logging.error(f"Redis set error: {str(e)}")
        else:
            # Memory cache
            self.memory_cache[key] = {
                "data": value,
                "timestamp": time.time()
            }
    
    async def delete(self, key):
        """Delete item from cache"""
        if self.cache_type == "redis" and self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception as e:
                logging.error(f"Redis delete error: {str(e)}")
        else:
            # Memory cache
            if key in self.memory_cache:
                del self.memory_cache[key]
    
    async def clear(self):
        """Clear all cache"""
        if self.cache_type == "redis" and self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception as e:
                logging.error(f"Redis flush error: {str(e)}")
        else:
            # Memory cache
            self.memory_cache = {}

class MCPServer:
    """Model Context Protocol Server Implementation"""
    
    def __init__(self, cache_type="memory", redis_url=None):
        self.app = FastAPI(
            title="OpenAI Code Assistant MCP Server",
            description="Model Context Protocol server for OpenAI Code Assistant",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json",
        )
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize cache
        self.cache = CacheManager(cache_type=cache_type, redis_url=redis_url)
        
        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Setup routes and middleware
        self.setup_routes()
        self.setup_middleware()
        
        # Load templates and static files
        self.templates_dir = os.path.join(os.path.dirname(__file__), "templates")
        os.makedirs(self.templates_dir, exist_ok=True)
        self.static_dir = os.path.join(os.path.dirname(__file__), "static")
        os.makedirs(self.static_dir, exist_ok=True)
        
        # Create default template if it doesn't exist
        self._create_default_template()
        
        # Initialize templates
        self.templates = Jinja2Templates(directory=self.templates_dir)
        
        # Mount static files
        self.app.mount("/static", StaticFiles(directory=self.static_dir), name="static")
        
        # Load prompt templates
        self.prompt_templates = self._load_prompt_templates()
        
        # Initialize metrics
        self.request_count = 0
        self.start_time = time.time()
        
    def setup_middleware(self):
        """Configure middleware for the FastAPI app"""
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add request tracking middleware
        @self.app.middleware("http")
        async def track_requests(request: Request, call_next):
            # Increment active connections
            ACTIVE_CONNECTIONS.inc()
            
            # Track request start time
            start_time = time.time()
            
            # Process request
            try:
                response = await call_next(request)
                
                # Record metrics
                endpoint = request.url.path
                status = response.status_code
                REQUEST_COUNT.labels(endpoint=endpoint, status=status).inc()
                REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)
                
                # Increment total request count
                self.request_count += 1
                
                return response
            finally:
                # Decrement active connections
                ACTIVE_CONNECTIONS.dec()
    
    def _create_default_template(self):
        """Create default dashboard template if it doesn't exist"""
        index_path = os.path.join(self.templates_dir, "index.html")
        if not os.path.exists(index_path):
            with open(index_path, "w") as f:
                f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>OpenAI Code Assistant MCP Server</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
    <style>
        body { padding: 20px; }
        .card { margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>OpenAI Code Assistant MCP Server</h1>
        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">Server Status</div>
                    <div class="card-body">
                        <p><strong>Status:</strong> {{ status }}</p>
                        <p><strong>Uptime:</strong> {{ uptime }}</p>
                        <p><strong>Requests Served:</strong> {{ request_count }}</p>
                        <p><strong>Cache Hit Ratio:</strong> {{ cache_hit_ratio }}%</p>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">Available Models</div>
                    <div class="card-body">
                        <ul>
                            {% for model in models %}
                            <li>{{ model }}</li>
                            {% endfor %}
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        
        <h2>Available Prompt Templates</h2>
        <div class="row">
            {% for template in templates %}
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">{{ template.id }}</div>
                    <div class="card-body">
                        <p><strong>Description:</strong> {{ template.description }}</p>
                        <p><strong>Parameters:</strong> {{ template.parameters|join(", ") }}</p>
                        <p><strong>Default Model:</strong> {{ template.default_model }}</p>
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
        
        <h2>API Documentation</h2>
        <p>
            <a href="/docs" class="btn btn-primary">Interactive API Docs</a>
            <a href="/redoc" class="btn btn-secondary">ReDoc API Docs</a>
            <a href="/metrics" class="btn btn-info">Prometheus Metrics</a>
        </p>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
                """)
    
    def setup_routes(self):
        """Configure API routes for MCP protocol"""
        
        # MCP Protocol Routes
        # Dashboard route
        @self.app.get("/", tags=["Dashboard"])
        async def dashboard(request: Request):
            """Dashboard showing server status and available templates"""
            # Calculate cache hit ratio
            cache_hits = prometheus_client.REGISTRY.get_sample_value('mcp_cache_hits_total') or 0
            cache_misses = prometheus_client.REGISTRY.get_sample_value('mcp_cache_misses_total') or 0
            total_cache_requests = cache_hits + cache_misses
            cache_hit_ratio = (cache_hits / total_cache_requests * 100) if total_cache_requests > 0 else 0
            
            # Format uptime
            uptime_seconds = time.time() - self.start_time
            days, remainder = divmod(uptime_seconds, 86400)
            hours, remainder = divmod(remainder, 3600)
            minutes, seconds = divmod(remainder, 60)
            uptime_str = f"{int(days)}d {int(hours)}h {int(minutes)}m {int(seconds)}s"
            
            # Get template information
            templates = []
            for template_id, template in self.prompt_templates.items():
                templates.append({
                    "id": template_id,
                    "description": template.get("description", ""),
                    "parameters": list(template.get("parameters", {}).keys()),
                    "default_model": template.get("default_model", "gpt-4o")
                })
            
            return self.templates.TemplateResponse("index.html", {
                "request": request,
                "status": "Healthy",
                "uptime": uptime_str,
                "request_count": self.request_count,
                "cache_hit_ratio": round(cache_hit_ratio, 2),
                "models": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
                "templates": templates
            })
        
        # Prometheus metrics endpoint
        @self.app.get("/metrics", tags=["Monitoring"])
        async def metrics():
            """Expose Prometheus metrics"""
            return Response(prometheus_client.generate_latest(), media_type="text/plain")
        
        # Health check endpoints
        @self.app.get("/health", response_model=MCPHealthResponse, tags=["Health"])
        async def health():
            """Health check endpoint"""
            # Calculate cache hit ratio
            cache_hits = prometheus_client.REGISTRY.get_sample_value('mcp_cache_hits_total') or 0
            cache_misses = prometheus_client.REGISTRY.get_sample_value('mcp_cache_misses_total') or 0
            total_cache_requests = cache_hits + cache_misses
            cache_hit_ratio = (cache_hits / total_cache_requests) if total_cache_requests > 0 else 0
            
            return MCPHealthResponse(
                status="healthy",
                uptime=time.time() - self.start_time,
                request_count=self.request_count,
                cache_hit_ratio=cache_hit_ratio
            )
        
        @self.app.post("/context", response_model=MCPContextResponse, tags=["Context"])
        async def get_context(
            request: MCPContextRequest, 
            background_tasks: BackgroundTasks,
            use_cache: bool = Query(True, description="Whether to use cached results if available")
        ):
            """
            Get context for a prompt template with parameters.
            
            This endpoint processes a prompt template with the provided parameters
            and returns the generated context. It can optionally use OpenAI models
            to enhance the context.
            """
            try:
                # Check if prompt template exists
                if request.prompt_id not in self.prompt_templates:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Prompt template '{request.prompt_id}' not found"
                    )
                
                # Get prompt template
                template = self.prompt_templates[request.prompt_id]
                
                # Use default model if not specified
                model = request.model or template.get("default_model", "gpt-4o")
                
                # Generate context ID
                context_id = str(uuid.uuid4())
                
                # Generate cache key
                cache_key = f"{request.prompt_id}:{json.dumps(request.parameters, sort_keys=True)}:{model}"
                
                # Check cache if enabled
                if use_cache:
                    cached_result = await self.cache.get(cache_key)
                    if cached_result:
                        # Update context ID for this request
                        cached_result["context_id"] = context_id
                        return MCPContextResponse(**cached_result)
                
                # Process template with parameters
                processed_template = self._process_template(template["template"], request.parameters)
                
                # Check if we should use OpenAI to enhance the context
                if template.get("use_openai", False):
                    # Generate context using OpenAI
                    context, usage = await self._generate_with_openai(
                        processed_template, 
                        model, 
                        template.get("system_prompt"),
                        template.get("metadata", {})
                    )
                else:
                    # Use the processed template directly
                    context = processed_template
                    
                    # Calculate token usage
                    token_count = len(self.tokenizer.encode(context))
                    usage = {
                        "prompt_tokens": token_count,
                        "completion_tokens": 0,
                        "total_tokens": token_count
                    }
                
                # Track token usage in Prometheus
                TOKEN_USAGE.labels(model=model, type="prompt").inc(usage["prompt_tokens"])
                TOKEN_USAGE.labels(model=model, type="completion").inc(usage["completion_tokens"])
                
                # Create response
                response = MCPContextResponse(
                    context=context,
                    context_id=context_id,
                    model=model,
                    usage=usage,
                    metadata={
                        "prompt_id": request.prompt_id,
                        "timestamp": time.time(),
                        "parameters": request.parameters
                    }
                )
                
                # Store in cache
                await self.cache.set(cache_key, response.dict())
                
                return response
                
            except Exception as e:
                logger.error(f"Error processing context request: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing context: {str(e)}"
                )
        
        @self.app.post("/context/stream", tags=["Context"])
        async def stream_context(request: MCPContextRequest):
            """
            Stream context generation.
            
            Similar to /context but streams the response as it's generated.
            """
            try:
                # Check if prompt template exists
                if request.prompt_id not in self.prompt_templates:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Prompt template '{request.prompt_id}' not found"
                    )
                
                # Get prompt template
                template = self.prompt_templates[request.prompt_id]
                
                # Use default model if not specified
                model = request.model or template.get("default_model", "gpt-4o")
                
                # Generate context ID
                context_id = str(uuid.uuid4())
                
                # Process template with parameters
                processed_template = self._process_template(template["template"], request.parameters)
                
                # Stream the context generation
                return StreamingResponse(
                    self._stream_context(processed_template, model, context_id, template.get("system_prompt"), template.get("metadata", {})),
                    media_type="text/event-stream"
                )
                
            except Exception as e:
                logger.error(f"Error streaming context: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"Error streaming context: {str(e)}"
                )
        
        @self.app.get("/prompts", response_model=MCPPromptLibraryResponse, tags=["Prompts"])
        async def get_prompts():
            """
            Get available prompt templates.
            
            Returns a list of all prompt templates available in the system.
            """
            prompts = [
                MCPPromptTemplate(
                    id=prompt_id,
                    template=template["template"],
                    description=template.get("description", ""),
                    parameters=template.get("parameters", {}),
                    default_model=template.get("default_model", "gpt-4o"),
                    metadata=template.get("metadata", {})
                )
                for prompt_id, template in self.prompt_templates.items()
            ]
            
            return MCPPromptLibraryResponse(
                prompts=prompts,
                count=len(prompts)
            )
        
        @self.app.get("/prompts/{prompt_id}", response_model=MCPPromptTemplate, tags=["Prompts"])
        async def get_prompt(prompt_id: str):
            """
            Get a specific prompt template.
            
            Returns the details of a specific prompt template by ID.
            """
            if prompt_id not in self.prompt_templates:
                raise HTTPException(
                    status_code=404,
                    detail=f"Prompt template '{prompt_id}' not found"
                )
            
            template = self.prompt_templates[prompt_id]
            return MCPPromptTemplate(
                id=prompt_id,
                template=template["template"],
                description=template.get("description", ""),
                parameters=template.get("parameters", {}),
                default_model=template.get("default_model", "gpt-4o"),
                metadata=template.get("metadata", {})
            )
        
        @self.app.post("/prompts", response_model=MCPPromptTemplate, status_code=201, tags=["Prompts"])
        async def create_prompt(prompt: MCPPromptTemplate):
            """
            Create a new prompt template.
            
            Adds a new prompt template to the system.
            """
            if prompt.id in self.prompt_templates:
                raise HTTPException(
                    status_code=409,
                    detail=f"Prompt template '{prompt.id}' already exists"
                )
            
            self.prompt_templates[prompt.id] = {
                "template": prompt.template,
                "description": prompt.description,
                "parameters": prompt.parameters,
                "default_model": prompt.default_model,
                "metadata": prompt.metadata
            }
            
            # Save updated templates
            self._save_prompt_templates()
            
            return prompt
        
        @self.app.put("/prompts/{prompt_id}", response_model=MCPPromptTemplate, tags=["Prompts"])
        async def update_prompt(prompt_id: str, prompt: MCPPromptTemplate):
            """
            Update an existing prompt template.
            
            Updates the details of an existing prompt template.
            """
            if prompt_id != prompt.id:
                raise HTTPException(
                    status_code=400,
                    detail="Prompt ID in path must match prompt ID in body"
                )
            
            if prompt_id not in self.prompt_templates:
                raise HTTPException(
                    status_code=404,
                    detail=f"Prompt template '{prompt_id}' not found"
                )
            
            self.prompt_templates[prompt_id] = {
                "template": prompt.template,
                "description": prompt.description,
                "parameters": prompt.parameters,
                "default_model": prompt.default_model,
                "metadata": prompt.metadata
            }
            
            # Save updated templates
            self._save_prompt_templates()
            
            return prompt
        
        @self.app.delete("/prompts/{prompt_id}", tags=["Prompts"])
        async def delete_prompt(prompt_id: str):
            """
            Delete a prompt template.
            
            Removes a prompt template from the system.
            """
            if prompt_id not in self.prompt_templates:
                raise HTTPException(
                    status_code=404,
                    detail=f"Prompt template '{prompt_id}' not found"
                )
            
            del self.prompt_templates[prompt_id]
            
            # Save updated templates
            self._save_prompt_templates()
            
            return {"status": "deleted", "prompt_id": prompt_id}
        
        # Additional endpoints for a more complete MCP server
        @self.app.get("/models", tags=["Models"])
        async def get_models():
            """
            Get available models.
            
            Returns a list of models that can be used with this MCP server.
            """
            return {
                "models": [
                    {
                        "id": "gpt-4o",
                        "name": "GPT-4o",
                        "description": "OpenAI's most advanced model",
                        "context_length": 128000,
                        "is_default": True
                    },
                    {
                        "id": "gpt-4-turbo",
                        "name": "GPT-4 Turbo",
                        "description": "Optimized version of GPT-4",
                        "context_length": 128000,
                        "is_default": False
                    },
                    {
                        "id": "gpt-3.5-turbo",
                        "name": "GPT-3.5 Turbo",
                        "description": "Fast and efficient model",
                        "context_length": 16385,
                        "is_default": False
                    },
                    {
                        "id": "o1",
                        "name": "o1",
                        "description": "OpenAI's reasoning-focused model with advanced capabilities",
                        "context_length": 128000,
                        "is_default": False,
                        "features": ["reasoning_effort", "web_search"]
                    },
                    {
                        "id": "o1-mini",
                        "name": "o1-mini",
                        "description": "Smaller and faster version of the o1 model",
                        "context_length": 128000,
                        "is_default": False,
                        "features": ["reasoning_effort", "web_search"]
                    }
                ],
                "count": 5
            }
        
        @self.app.get("/stats", tags=["System"])
        async def get_stats():
            """
            Get server statistics.
            
            Returns usage statistics and system information.
            """
            return {
                "uptime": time.time() - self.start_time,
                "prompt_templates_count": len(self.prompt_templates),
                "cache_size": len(self.context_cache),
                "requests_served": {
                    "context": 0,  # This would be tracked in a real implementation
                    "prompts": 0,
                    "total": 0
                },
                "system_info": {
                    "python_version": sys.version,
                    "platform": sys.platform
                }
            }
            
        # Error handlers
        @self.app.exception_handler(HTTPException)
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
        
        @self.app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            """Handle general exceptions in MCP format"""
            logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "error": str(exc),
                    "error_type": "server_error",
                    "status_code": 500,
                    "details": None
                }
            )
    
    def _load_prompt_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load prompt templates from file or initialize defaults"""
        templates_file = os.path.join(os.path.dirname(__file__), "data", "prompt_templates.json")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(templates_file), exist_ok=True)
        
        # Try to load existing templates
        if os.path.exists(templates_file):
            try:
                with open(templates_file, "r") as f:
                    templates = json.load(f)
                    logger.info(f"Loaded {len(templates)} prompt templates from {templates_file}")
                    return templates
            except Exception as e:
                logger.error(f"Error loading prompt templates: {str(e)}")
        
        # Initialize with enhanced default templates
        default_templates = {
            "greeting": {
                "template": "Hello! The current time is {time}. How can I help you today?",
                "description": "A simple greeting template",
                "parameters": {
                    "time": {
                        "type": "string",
                        "description": "The current time"
                    }
                },
                "default_model": "gpt-4o",
                "metadata": {
                    "category": "general"
                }
            },
            "code_review": {
                "template": "Please review the following code:\n\n```{language}\n{code}\n```\n\nFocus on: {focus_areas}",
                "description": "Template for code review requests",
                "parameters": {
                    "language": {
                        "type": "string",
                        "description": "Programming language of the code"
                    },
                    "code": {
                        "type": "string",
                        "description": "The code to review"
                    },
                    "focus_areas": {
                        "type": "string",
                        "description": "Areas to focus on during review (e.g., 'performance, security')"
                    }
                },
                "default_model": "gpt-4o",
                "use_openai": True,
                "system_prompt": "You are a code review expert. Analyze the provided code and provide constructive feedback focusing on the specified areas.",
                "metadata": {
                    "category": "development"
                }
            },
            "system_prompt": {
                "template": "You are OpenAI Code Assistant, a CLI tool that helps users with software engineering tasks and general information.\nUse the available tools to assist the user with their requests.\n\n# Tone and style\nYou should be concise, direct, and to the point. When you run a non-trivial bash command, \nyou should explain what the command does and why you are running it.\nOutput text to communicate with the user; all text you output outside of tool use is displayed to the user.\nRemember that your output will be displayed on a command line interface.\n\n# Tool usage policy\n- When doing file search, remember to search effectively with the available tools.\n- Always use the appropriate tool for the task.\n- Use parallel tool calls when appropriate to improve performance.\n- NEVER commit changes unless the user explicitly asks you to.\n- For weather queries, use the Weather tool to provide real-time information.\n\n# Tasks\nThe user will primarily request you perform software engineering tasks:\n1. Solving bugs\n2. Adding new functionality \n3. Refactoring code\n4. Explaining code\n5. Writing tests\n\nFor these tasks:\n1. Use search tools to understand the codebase\n2. Implement solutions using the available tools\n3. Verify solutions with tests if possible\n4. Run lint and typecheck commands when appropriate\n\nThe user may also ask for general information:\n1. Weather conditions\n2. Simple calculations\n3. General knowledge questions\n\n# Code style\n- Follow the existing code style of the project\n- Maintain consistent naming conventions\n- Use appropriate libraries that are already in the project\n- Add comments when code is complex or non-obvious\n\nIMPORTANT: You should minimize output tokens as much as possible while maintaining helpfulness, \nquality, and accuracy. Answer concisely with short lines of text unless the user asks for detail.",
                "description": "System prompt for the assistant",
                "parameters": {},
                "default_model": "gpt-4o",
                "metadata": {
                    "category": "system"
                }
            },
            "documentation": {
                "template": "Generate documentation for the following code:\n\n```{language}\n{code}\n```\n\nFormat: {format}",
                "description": "Generate code documentation",
                "parameters": {
                    "language": {
                        "type": "string",
                        "description": "Programming language of the code"
                    },
                    "code": {
                        "type": "string",
                        "description": "The code to document"
                    },
                    "format": {
                        "type": "string",
                        "description": "Documentation format (e.g., 'markdown', 'docstring', 'jsdoc')",
                        "default": "markdown"
                    }
                },
                "default_model": "gpt-4o",
                "use_openai": True,
                "system_prompt": "You are a technical documentation expert. Generate clear, concise, and accurate documentation for the provided code.",
                "metadata": {
                    "category": "development"
                }
            },
            "explain_code": {
                "template": "Explain how the following code works:\n\n```{language}\n{code}\n```\n\nDetail level: {detail_level}",
                "description": "Explain code functionality",
                "parameters": {
                    "language": {
                        "type": "string",
                        "description": "Programming language of the code"
                    },
                    "code": {
                        "type": "string",
                        "description": "The code to explain"
                    },
                    "detail_level": {
                        "type": "string",
                        "description": "Level of detail in the explanation (e.g., 'basic', 'intermediate', 'advanced')",
                        "default": "intermediate"
                    }
                },
                "default_model": "gpt-4o",
                "use_openai": True,
                "system_prompt": "You are a programming instructor. Explain the provided code clearly at the requested level of detail.",
                "metadata": {
                    "category": "education"
                }
            },
            "current_time": {
                "template": "The current time is {{now:%Y-%m-%d %H:%M:%S}}.",
                "description": "Get the current time",
                "parameters": {},
                "default_model": "gpt-4o",
                "metadata": {
                    "category": "utility"
                }
            }
        }
        
        # Save default templates
        try:
            with open(templates_file, "w") as f:
                json.dump(default_templates, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving default prompt templates: {str(e)}")
        
        return default_templates
    
    def _save_prompt_templates(self):
        """Save prompt templates to file"""
        templates_file = os.path.join(os.path.dirname(__file__), "data", "prompt_templates.json")
        
        try:
            with open(templates_file, "w") as f:
                json.dump(self.prompt_templates, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving prompt templates: {str(e)}")
    
    async def _generate_with_openai(self, prompt: str, model: str, system_prompt: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> tuple:
        """Generate context using OpenAI API"""
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add user prompt
        messages.append({"role": "user", "content": prompt})
        
        # Prepare API parameters
        params = {
            "model": model,
            "messages": messages,
            "temperature": 0.0,  # Use deterministic output for context generation
            "max_tokens": 4000
        }
        
        # Add reasoning_effort for o1 models if provided in metadata
        if metadata and "reasoning_effort" in metadata and model.startswith("o1"):
            reasoning_effort = metadata["reasoning_effort"]
            
            # Ensure reasoning_effort is a valid string value
            valid_efforts = ["low", "medium", "high"]
            
            # Convert numeric values to string equivalents if needed
            if isinstance(reasoning_effort, (int, float)):
                if reasoning_effort <= 0.3:
                    reasoning_effort = "low"
                elif reasoning_effort <= 0.7:
                    reasoning_effort = "medium"
                else:
                    reasoning_effort = "high"
                logger.info(f"Converting numeric reasoning_effort to string value: {reasoning_effort}")
            
            # Validate string values
            if isinstance(reasoning_effort, str) and reasoning_effort.lower() in valid_efforts:
                params["reasoning_effort"] = reasoning_effort.lower()
                logger.info(f"Using reasoning_effort={reasoning_effort} for o1 model")
            else:
                logger.warning(f"Invalid reasoning_effort value: {reasoning_effort}. Using 'medium' instead.")
                params["reasoning_effort"] = "medium"
        
        # Call OpenAI API
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                **params
            )
            
            # Extract content and usage
            content = response.choices[0].message.content
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            return content, usage
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise ValueError(f"Error generating context with OpenAI: {str(e)}")
    
    async def _stream_context(self, prompt: str, model: str, context_id: str, system_prompt: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> AsyncGenerator[str, None]:
        """Stream context generation using OpenAI API"""
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add user prompt
        messages.append({"role": "user", "content": prompt})
        
        # Prepare API parameters
        params = {
            "model": model,
            "messages": messages,
            "temperature": 0.0,  # Use deterministic output for context generation
            "stream": True,
            "max_tokens": 4000
        }
        
        # Add reasoning_effort for o1 models if provided in metadata
        if metadata and "reasoning_effort" in metadata and model.startswith("o1"):
            reasoning_effort = metadata["reasoning_effort"]
            
            # Ensure reasoning_effort is a valid string value
            valid_efforts = ["low", "medium", "high"]
            
            # Convert numeric values to string equivalents if needed
            if isinstance(reasoning_effort, (int, float)):
                if reasoning_effort <= 0.3:
                    reasoning_effort = "low"
                elif reasoning_effort <= 0.7:
                    reasoning_effort = "medium"
                else:
                    reasoning_effort = "high"
                logger.info(f"Converting numeric reasoning_effort to string value: {reasoning_effort}")
            
            # Validate string values
            if isinstance(reasoning_effort, str) and reasoning_effort.lower() in valid_efforts:
                params["reasoning_effort"] = reasoning_effort.lower()
                logger.info(f"Using reasoning_effort={reasoning_effort} for o1 model")
            else:
                logger.warning(f"Invalid reasoning_effort value: {reasoning_effort}. Using 'medium' instead.")
                params["reasoning_effort"] = "medium"
        
        # Initial event with context ID
        yield f"data: {json.dumps({'context_id': context_id, 'event': 'start'})}\n\n"
        
        try:
            # Call OpenAI API with streaming
            stream = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                **params
            )
            
            full_content = ""
            
            # Process the stream
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content_piece = chunk.choices[0].delta.content
                    full_content += content_piece
                    
                    # Yield the content piece
                    yield f"data: {json.dumps({'content': content_piece, 'event': 'content'})}\n\n"
            
            # Calculate token usage
            prompt_tokens = len(self.tokenizer.encode(prompt))
            completion_tokens = len(self.tokenizer.encode(full_content))
            total_tokens = prompt_tokens + completion_tokens
            
            # Track token usage
            TOKEN_USAGE.labels(model=model, type="prompt").inc(prompt_tokens)
            TOKEN_USAGE.labels(model=model, type="completion").inc(completion_tokens)
            
            # Final event with complete context and usage
            data = {
                'event': 'end',
                'context': full_content,
                'usage': {
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': total_tokens
                }
            }
            yield f"data: {json.dumps(data)}\n\n"
            
        except Exception as e:
            logger.error(f"Error streaming context: {str(e)}")
            yield f"data: {json.dumps({'event': 'error', 'error': str(e)})}\n\n"
    
    def _process_template(self, template: str, parameters: Dict[str, Any]) -> str:
        """Process a template with parameters"""
        try:
            # Handle date/time formatting if needed
            processed_params = parameters.copy()
            for key, value in processed_params.items():
                if isinstance(value, str) and value.startswith("{{now") and value.endswith("}}"):
                    # Extract format string if present
                    format_match = re.search(r"{{now:(.+)}}", value)
                    if format_match:
                        format_string = format_match.group(1)
                        processed_params[key] = datetime.now().strftime(format_string)
                    else:
                        processed_params[key] = datetime.now().isoformat()
            
            return template.format(**processed_params)
        except KeyError as e:
            raise ValueError(f"Missing required parameter: {e}")
        except Exception as e:
            raise ValueError(f"Error processing template: {str(e)}")
    
    def start(self, host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
        """Start the MCP server"""
        uvicorn.run(self.app, host=host, port=port, reload=reload)

def create_mcp_app():
    """Factory function for creating the FastAPI app"""
    server = MCPServer()
    return server.app

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(__file__), "data"), exist_ok=True)
    
    # Start server
    server = MCPServer()
    server.start()
