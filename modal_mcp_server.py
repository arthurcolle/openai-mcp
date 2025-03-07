import modal
import logging
import time
import uuid
import json
import asyncio
import hashlib
import threading
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, AsyncIterator
from datetime import datetime, timedelta
from collections import deque

from fastapi import FastAPI, Request, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create FastAPI app
api_app = FastAPI(
    title="Advanced LLM Inference API", 
    description="Enterprise-grade OpenAI-compatible LLM serving API with multiple model support, streaming, and advanced caching",
    version="1.1.0"
)

# Add CORS middleware
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify specific origins instead of wildcard
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security setup
security = HTTPBearer()

# Token bucket rate limiter
class TokenBucket:
    """
    Token bucket algorithm for rate limiting.
    Each user gets a bucket that fills at a constant rate.
    """
    def __init__(self):
        self.buckets = {}
        self.lock = threading.Lock()
    
    def _get_bucket(self, user_id, rate_limit):
        """Get or create a bucket for a user"""
        now = time.time()
        
        if user_id not in self.buckets:
            # Initialize with full bucket
            self.buckets[user_id] = {
                "tokens": rate_limit,
                "last_refill": now,
                "rate": rate_limit / 60.0  # tokens per second
            }
            return self.buckets[user_id]
        
        bucket = self.buckets[user_id]
        
        # Update rate if it changed
        bucket["rate"] = rate_limit / 60.0
        
        # Refill tokens based on time elapsed
        elapsed = now - bucket["last_refill"]
        new_tokens = elapsed * bucket["rate"]
        
        bucket["tokens"] = min(rate_limit, bucket["tokens"] + new_tokens)
        bucket["last_refill"] = now
        
        return bucket
    
    def consume(self, user_id, tokens=1, rate_limit=60):
        """
        Consume tokens from a user's bucket.
        Returns True if tokens were consumed, False otherwise.
        """
        with self.lock:
            bucket = self._get_bucket(user_id, rate_limit)
            
            if bucket["tokens"] >= tokens:
                bucket["tokens"] -= tokens
                return True
            return False

# Create rate limiter
rate_limiter = TokenBucket()

# Define the container image with necessary dependencies
vllm_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "vllm==0.7.3",  # Updated version
        "huggingface_hub[hf_transfer]==0.26.2",
        "flashinfer-python==0.2.0.post2",
        "fastapi>=0.95.0",
        "uvicorn>=0.15.0",
        "pydantic>=2.0.0",
        "tiktoken>=0.5.1",
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.5",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster model transfers
    .env({"VLLM_USE_V1": "1"})  # Enable V1 engine for better performance
)

# Define llama.cpp image for alternative models
llama_cpp_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "build-essential", "cmake", "curl", "libcurl4-openssl-dev")
    .pip_install(
        "huggingface_hub==0.26.2",
        "hf_transfer>=0.1.4",
        "fastapi>=0.95.0",
        "uvicorn>=0.15.0",
        "pydantic>=2.0.0"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_commands(
        "git clone https://github.com/ggerganov/llama.cpp",
        "cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=ON",
        "cmake --build llama.cpp/build --config Release -j --target llama-cli",
        "cp llama.cpp/build/bin/llama-* /usr/local/bin/"
    )
)

# Set up model configurations
MODELS_DIR = "/models"
VLLM_MODELS = {
    "llama3-8b": {
        "id": "llama3-8b",
        "name": "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16",
        "config": "config.json",  # Ensure this file is present in the model directory
        "revision": "a7c09948d9a632c2c840722f519672cd94af885d",
        "max_tokens": 4096,
        "loaded": False
    },
    "mistral-7b": {
        "id": "mistral-7b",
        "name": "mistralai/Mistral-7B-Instruct-v0.2",
        "revision": "main",
        "max_tokens": 4096,
        "loaded": False
    },
    # Small model for quick loading
    "tiny-llama-1.1b": {
        "id": "tiny-llama-1.1b",
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "revision": "main",
        "max_tokens": 2048,
        "loaded": False
    }
}

LLAMA_CPP_MODELS = {
    "deepseek-r1": {
        "id": "deepseek-r1",
        "name": "unsloth/DeepSeek-R1-GGUF",
        "quant": "UD-IQ1_S",
        "pattern": "*UD-IQ1_S*",
        "revision": "02656f62d2aa9da4d3f0cdb34c341d30dd87c3b6",
        "gpu": "L40S:4",
        "max_tokens": 4096,
        "loaded": False
    },
    "phi-4": {
        "id": "phi-4",
        "name": "unsloth/phi-4-GGUF",
        "quant": "Q2_K",
        "pattern": "*Q2_K*",
        "revision": None,
        "gpu": "L40S:4",  # Use GPU for better performance
        "max_tokens": 4096,
        "loaded": False
    },
    # Small model for quick loading
    "phi-2": {
        "id": "phi-2",
        "name": "TheBloke/phi-2-GGUF",
        "quant": "Q4_K_M",
        "pattern": "*Q4_K_M.gguf",
        "revision": "main",
        "gpu": None,  # Can run on CPU
        "max_tokens": 2048,
        "loaded": False
    }
}

DEFAULT_MODEL = "phi-4"

# Create volumes for caching
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
llama_cpp_cache_vol = modal.Volume.from_name("llama-cpp-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("model-results", create_if_missing=True)

# Create the Modal app
app = modal.App("openai-compatible-llm-server")

# Create shared data structures
model_stats_dict = modal.Dict.from_name("model-stats", create_if_missing=True)
user_usage_dict = modal.Dict.from_name("user-usage", create_if_missing=True)
request_queue = modal.Queue.from_name("request-queue", create_if_missing=True)
response_dict = modal.Dict.from_name("response-cache", create_if_missing=True)
api_keys_dict = modal.Dict.from_name("api-keys", create_if_missing=True)
stream_queues = modal.Dict.from_name("stream-queues", create_if_missing=True)

# Advanced caching system
class AdvancedCache:
    """
    Advanced caching system with TTL and LRU eviction.
    """
    def __init__(self, max_size=1000, default_ttl=3600):
        self.cache = {}
        self.ttl_map = {}
        self.access_times = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.lock = threading.Lock()
    
    def get(self, key):
        """Get a value from the cache"""
        with self.lock:
            now = time.time()
            
            # Check if key exists and is not expired
            if key in self.cache:
                # Check TTL
                if key in self.ttl_map and self.ttl_map[key] < now:
                    # Expired
                    self._remove(key)
                    return None
                
                # Update access time
                self.access_times[key] = now
                return self.cache[key]
            
            return None
    
    def set(self, key, value, ttl=None):
        """Set a value in the cache with optional TTL"""
        with self.lock:
            now = time.time()
            
            # Evict if needed
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()
            
            # Set value
            self.cache[key] = value
            self.access_times[key] = now
            
            # Set TTL
            if ttl is not None:
                self.ttl_map[key] = now + ttl
            elif self.default_ttl > 0:
                self.ttl_map[key] = now + self.default_ttl
    
    def _remove(self, key):
        """Remove a key from the cache"""
        if key in self.cache:
            del self.cache[key]
        if key in self.ttl_map:
            del self.ttl_map[key]
        if key in self.access_times:
            del self.access_times[key]
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if not self.access_times:
            return
        
        # Find oldest access time
        oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        self._remove(oldest_key)
    
    def clear_expired(self):
        """Clear all expired entries"""
        with self.lock:
            now = time.time()
            expired_keys = [k for k, v in self.ttl_map.items() if v < now]
            for key in expired_keys:
                self._remove(key)

# Constants
MAX_CACHE_AGE = 3600  # 1 hour in seconds

# Create memory cache
memory_cache = AdvancedCache(max_size=10000, default_ttl=MAX_CACHE_AGE)

# Initialize with default key if empty
if "default" not in api_keys_dict:
    api_keys_dict["default"] = {
        "key": "sk-modal-llm-api-key",
        "rate_limit": 60,  # requests per minute
        "quota": 1000000,  # tokens per day
        "created_at": datetime.now().isoformat(),
        "owner": "default"
    }

# Add a default ADMIN API key
if "admin" not in api_keys_dict:
    api_keys_dict["admin"] = {
        "key": "sk-modal-admin-api-key",
        "rate_limit": 1000,  # Higher rate limit for admin
        "quota": 10000000,  # Higher quota for admin
        "created_at": datetime.now().isoformat(),
        "owner": "admin"
    }

# Constants
DEFAULT_API_KEY = api_keys_dict["default"]["key"]
MINUTES = 60  # seconds
SERVER_PORT = 8000
CACHE_DIR = "/root/.cache"
RESULTS_DIR = "/root/results"

# Request/response models
class GenerationRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    user: Optional[str] = None
    stream: bool = False
    timestamp: float = Field(default_factory=time.time)
    api_key: str = DEFAULT_API_KEY
    
class StreamChunk(BaseModel):
    """Model for streaming response chunks"""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    
class StreamManager:
    """Manages streaming responses for clients"""
    def __init__(self):
        self.streams = {}
        self.lock = threading.Lock()
    
    def create_stream(self, request_id):
        """Create a new stream for a request"""
        with self.lock:
            self.streams[request_id] = {
                "queue": asyncio.Queue(),
                "finished": False,
                "created_at": time.time()
            }
    
    def add_chunk(self, request_id, chunk):
        """Add a chunk to a stream"""
        with self.lock:
            if request_id in self.streams:
                stream = self.streams[request_id]
                if not stream["finished"]:
                    stream["queue"].put_nowait(chunk)
    
    def finish_stream(self, request_id):
        """Mark a stream as finished"""
        with self.lock:
            if request_id in self.streams:
                self.streams[request_id]["finished"] = True
                # Add None to signal end of stream
                self.streams[request_id]["queue"].put_nowait(None)
    
    async def get_chunks(self, request_id):
        """Get chunks from a stream as an async generator"""
        if request_id not in self.streams:
            return
        
        stream = self.streams[request_id]
        queue = stream["queue"]
        
        while True:
            chunk = await queue.get()
            if chunk is None:  # End of stream
                break
            yield chunk
            queue.task_done()
        
        # Clean up after streaming is done
        with self.lock:
            if request_id in self.streams:
                del self.streams[request_id]
    
    def clean_old_streams(self, max_age=3600):
        """Clean up old streams"""
        with self.lock:
            now = time.time()
            to_remove = []
            
            for request_id, stream in self.streams.items():
                if now - stream["created_at"] > max_age:
                    to_remove.append(request_id)
            
            for request_id in to_remove:
                if request_id in self.streams:
                    # Mark as finished to stop any ongoing processing
                    self.streams[request_id]["finished"] = True
                    # Add None to unblock any waiting consumers
                    self.streams[request_id]["queue"].put_nowait(None)
                    # Remove from streams
                    del self.streams[request_id]

# Create stream manager
stream_manager = StreamManager()

# API Authentication dependency
def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify that the API key in the authorization header is valid and check rate limits"""
    if credentials.scheme != "Bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme. Use Bearer",
        )
    
    api_key = credentials.credentials
    valid_key = False
    key_info = None
    
    # Check if this is a known API key
    for user_id, user_data in api_keys_dict.items():
        if user_data.get("key") == api_key:
            valid_key = True
            key_info = user_data
            break
    
    if not valid_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    
    # Check rate limits
    user_id = key_info.get("owner", "unknown")
    rate_limit = key_info.get("rate_limit", 60)  # Default: 60 requests per minute
    
    # Get or initialize user usage tracking
    if user_id not in user_usage_dict:
        user_usage_dict[user_id] = {
            "requests": [],
            "tokens": {
                "input": 0,
                "output": 0,
                "last_reset": datetime.now().isoformat()
            }
        }
    
    usage = user_usage_dict[user_id]
    
    # Check if user exceeded rate limit using token bucket algorithm
    if not rate_limiter.consume(user_id, tokens=1, rate_limit=rate_limit):
        # Calculate retry-after based on rate
        retry_after = int(60 / rate_limit)  # seconds until at least one token is available
        
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {rate_limit} requests per minute.",
            headers={"Retry-After": str(retry_after)}
        )
    
    # Add current request timestamp for analytics
    now = datetime.now()
    usage["requests"].append(now.timestamp())
    
    # Clean up old requests (older than 1 day) to prevent unbounded growth
    day_ago = (now - timedelta(days=1)).timestamp()
    usage["requests"] = [req for req in usage["requests"] if req > day_ago]
    
    # Update usage dict
    user_usage_dict[user_id] = usage
    
    # Return the API key and user ID
    return {"key": api_key, "user_id": user_id}

# API Endpoints
@api_app.get("/", response_class=HTMLResponse)
async def index():
    """Root endpoint that returns HTML with API information"""
    return """
    <html>
        <head>
            <title>Modal LLM Inference API</title>
            <style>
                body { font-family: system-ui, sans-serif; max-width: 800px; margin: 0 auto; padding: 2rem; }
                h1 { color: #4a56e2; }
                code { background: #f4f4f8; padding: 0.2rem 0.4rem; border-radius: 3px; }
            </style>
        </head>
        <body>
            <h1>Modal LLM Inference API</h1>
            <p>This is an OpenAI-compatible API for LLM inference powered by Modal.</p>
            <p>Use the following endpoints:</p>
            <ul>
                <li><a href="/docs">/docs</a> - API documentation</li>
                <li><a href="/v1/models">/v1/models</a> - List available models</li>
                <li><code>/v1/chat/completions</code> - Chat completions endpoint</li>
            </ul>
        </body>
    </html>
    """

@api_app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@api_app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models():
    """List all available models in OpenAI-compatible format"""
    # Combine vLLM and llama.cpp models
    all_models = []
    
    for model_id, model_info in VLLM_MODELS.items():
        all_models.append({
            "id": model_info["id"],
            "object": "model",
            "created": 1677610602,
            "owned_by": "modal",
            "engine": "vllm",
            "loaded": model_info.get("loaded", False)
        })
        
    for model_id, model_info in LLAMA_CPP_MODELS.items():
        all_models.append({
            "id": model_info["id"],
            "object": "model",
            "created": 1677610602,
            "owned_by": "modal",
            "engine": "llama.cpp",
            "loaded": model_info.get("loaded", False)
        })
        
    return {"data": all_models, "object": "list"}

# Model management endpoints
class ModelLoadRequest(BaseModel):
    """Request model to load a specific model"""
    model_id: str
    force_reload: bool = False
    
class HFModelLoadRequest(BaseModel):
    """Request to load a model directly from Hugging Face"""
    repo_id: str
    model_type: str = "vllm"  # "vllm" or "llama.cpp"
    revision: Optional[str] = None
    quant: Optional[str] = None  # For llama.cpp models
    max_tokens: int = 4096
    gpu: Optional[str] = None  # For llama.cpp models

@api_app.post("/admin/models/load", dependencies=[Depends(verify_api_key)])
async def load_model(request: ModelLoadRequest, background_tasks: BackgroundTasks):
    """Load a specific model into memory"""
    model_id = request.model_id
    force_reload = request.force_reload
    
    # Check if model exists
    if model_id in VLLM_MODELS:
        model_type = "vllm"
        model_info = VLLM_MODELS[model_id]
    elif model_id in LLAMA_CPP_MODELS:
        model_type = "llama.cpp"
        model_info = LLAMA_CPP_MODELS[model_id]
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )
    
    # Check if model is already loaded
    if model_info.get("loaded", False) and not force_reload:
        return {
            "status": "success",
            "message": f"Model {model_id} is already loaded",
            "model_id": model_id,
            "model_type": model_type
        }
    
    # Start loading the model in the background
    if model_type == "vllm":
        # Start vLLM server for this model
        background_tasks.add_task(serve_vllm_model.remote, model_id=model_id)
        # Update model status
        VLLM_MODELS[model_id]["loaded"] = True
    else:  # llama.cpp
        # For llama.cpp models, we'll preload the model
        background_tasks.add_task(preload_llama_cpp_model, model_id)
        # Update model status
        LLAMA_CPP_MODELS[model_id]["loaded"] = True
    
    return {
        "status": "success",
        "message": f"Started loading model {model_id}",
        "model_id": model_id,
        "model_type": model_type
    }

@api_app.post("/admin/models/load-from-hf", dependencies=[Depends(verify_api_key)])
async def load_model_from_hf(request: HFModelLoadRequest, background_tasks: BackgroundTasks):
    """Load a model directly from Hugging Face"""
    repo_id = request.repo_id
    model_type = request.model_type
    revision = request.revision
    
    # Generate a unique model_id based on the repo name
    repo_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
    model_id = f"hf-{repo_name}-{uuid.uuid4().hex[:6]}"
    
    # Create model info based on type
    if model_type.lower() == "vllm":
        # Add to VLLM_MODELS
        VLLM_MODELS[model_id] = {
            "id": model_id,
            "name": repo_id,
            "revision": revision or "main",
            "max_tokens": request.max_tokens,
            "loaded": False,
            "hf_direct": True  # Mark as directly loaded from HF
        }
        
        # Start vLLM server for this model
        background_tasks.add_task(serve_vllm_model.remote, model_id=model_id)
        # Update model status
        VLLM_MODELS[model_id]["loaded"] = True
        
    elif model_type.lower() == "llama.cpp":
        # For llama.cpp we need quant info
        quant = request.quant or "Q4_K_M"  # Default quantization
        pattern = f"*{quant}*"
        
        # Add to LLAMA_CPP_MODELS
        LLAMA_CPP_MODELS[model_id] = {
            "id": model_id,
            "name": repo_id,
            "quant": quant,
            "pattern": pattern,
            "revision": revision,
            "gpu": request.gpu,  # Can be None for CPU
            "max_tokens": request.max_tokens,
            "loaded": False,
            "hf_direct": True  # Mark as directly loaded from HF
        }
        
        # Preload the model
        background_tasks.add_task(preload_llama_cpp_model, model_id)
        # Update model status
        LLAMA_CPP_MODELS[model_id]["loaded"] = True
        
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model type: {model_type}. Must be 'vllm' or 'llama.cpp'"
        )
    
    return {
        "status": "success",
        "message": f"Started loading model {repo_id} as {model_id}",
        "model_id": model_id,
        "model_type": model_type,
        "repo_id": repo_id
    }

@api_app.post("/admin/models/unload", dependencies=[Depends(verify_api_key)])
async def unload_model(request: ModelLoadRequest):
    """Unload a specific model from memory"""
    model_id = request.model_id
    
    # Check if model exists
    if model_id in VLLM_MODELS:
        model_type = "vllm"
        model_info = VLLM_MODELS[model_id]
    elif model_id in LLAMA_CPP_MODELS:
        model_type = "llama.cpp"
        model_info = LLAMA_CPP_MODELS[model_id]
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )
    
    # Check if model is loaded
    if not model_info.get("loaded", False):
        return {
            "status": "success",
            "message": f"Model {model_id} is not loaded",
            "model_id": model_id,
            "model_type": model_type
        }
    
    # Update model status
    if model_type == "vllm":
        VLLM_MODELS[model_id]["loaded"] = False
    else:  # llama.cpp
        LLAMA_CPP_MODELS[model_id]["loaded"] = False
    
    return {
        "status": "success",
        "message": f"Unloaded model {model_id}",
        "model_id": model_id,
        "model_type": model_type
    }

@api_app.get("/admin/models/status/{model_id}", dependencies=[Depends(verify_api_key)])
async def get_model_status(model_id: str):
    """Get the status of a specific model"""
    # Check if model exists
    if model_id in VLLM_MODELS:
        model_type = "vllm"
        model_info = VLLM_MODELS[model_id]
    elif model_id in LLAMA_CPP_MODELS:
        model_type = "llama.cpp"
        model_info = LLAMA_CPP_MODELS[model_id]
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )
    
    # Get model stats if available
    model_stats = model_stats_dict.get(model_id, {})
    
    # Include HF info if available
    hf_info = {}
    if model_info.get("hf_direct"):
        hf_info = {
            "repo_id": model_info.get("name"),
            "revision": model_info.get("revision"),
        }
        if model_type == "llama.cpp":
            hf_info["quant"] = model_info.get("quant")
    
    return {
        "model_id": model_id,
        "model_type": model_type,
        "loaded": model_info.get("loaded", False),
        "stats": model_stats,
        "hf_info": hf_info if hf_info else None
    }

# Admin API endpoints
class APIKeyRequest(BaseModel):
    user_id: str
    rate_limit: int = 60
    quota: int = 1000000
    
class APIKey(BaseModel):
    key: str
    user_id: str
    rate_limit: int
    quota: int
    created_at: str

@api_app.post("/admin/api-keys", response_model=APIKey)
async def create_api_key(request: APIKeyRequest, auth_info: dict = Depends(verify_api_key)):
    """Create a new API key for a user (admin only)"""
    # Check if this is an admin request
    if auth_info["user_id"] != "default":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admin users can create API keys"
        )
    
    # Generate a new API key
    new_key = f"sk-modal-{uuid.uuid4()}"
    user_id = request.user_id
    
    # Store the key
    api_keys_dict[user_id] = {
        "key": new_key,
        "rate_limit": request.rate_limit,
        "quota": request.quota,
        "created_at": datetime.now().isoformat(),
        "owner": user_id
    }
    
    # Initialize user usage
    if not user_usage_dict.contains(user_id):
        user_usage_dict[user_id] = {
            "requests": [],
            "tokens": {
                "input": 0,
                "output": 0,
                "last_reset": datetime.now().isoformat()
            }
        }
    
    return APIKey(
        key=new_key,
        user_id=user_id,
        rate_limit=request.rate_limit,
        quota=request.quota,
        created_at=datetime.now().isoformat()
    )

@api_app.get("/admin/api-keys")
async def list_api_keys(auth_info: dict = Depends(verify_api_key)):
    """List all API keys (admin only)"""
    # Check if this is an admin request
    if auth_info["user_id"] != "default":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admin users can list API keys"
        )
    
    # Return all keys (except the actual key values for security)
    keys = []
    for user_id, key_info in api_keys_dict.items():
        keys.append({
            "user_id": user_id,
            "rate_limit": key_info.get("rate_limit", 60),
            "quota": key_info.get("quota", 1000000),
            "created_at": key_info.get("created_at", datetime.now().isoformat()),
            # Mask the actual key
            "key": key_info.get("key", "")[:8] + "..." if key_info.get("key") else "None"
        })
    
    return {"keys": keys}

@api_app.get("/admin/stats")
async def get_stats(auth_info: dict = Depends(verify_api_key)):
    """Get usage statistics (admin only)"""
    # Check if this is an admin request
    if auth_info["user_id"] != "default":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admin users can view stats"
        )
    
    # Get model stats
    model_stats = {}
    for model_id in list(VLLM_MODELS.keys()) + list(LLAMA_CPP_MODELS.keys()):
        if model_id in model_stats_dict:
            model_stats[model_id] = model_stats_dict[model_id]
    
    # Get user stats
    user_stats = {}
    for user_id in user_usage_dict.keys():
        usage = user_usage_dict[user_id]
        # Don't include request timestamps for brevity
        if "requests" in usage:
            usage = usage.copy()
            usage["request_count"] = len(usage["requests"])
            del usage["requests"]
        user_stats[user_id] = usage
    
    # Get queue info
    queue_info = {
        "pending_requests": request_queue.len(),
        "active_workers": model_stats_dict.get("workers_running", 0)
    }
    
    return {
        "models": model_stats,
        "users": user_stats,
        "queue": queue_info,
        "timestamp": datetime.now().isoformat()
    }

@api_app.delete("/admin/api-keys/{user_id}")
async def delete_api_key(user_id: str, auth_info: dict = Depends(verify_api_key)):
    """Delete an API key (admin only)"""
    # Check if this is an admin request
    if auth_info["user_id"] != "default":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admin users can delete API keys"
        )
    
    # Check if the key exists
    if not api_keys_dict.contains(user_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No API key found for user {user_id}"
        )
    
    # Can't delete the default key
    if user_id == "default":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete the default API key"
        )
    
    # Delete the key
    api_keys_dict.pop(user_id)
    
    return {"status": "success", "message": f"API key deleted for user {user_id}"}

@api_app.post("/v1/chat/completions")
async def chat_completions(request: Request, background_tasks: BackgroundTasks, auth_info: dict = Depends(verify_api_key)):
    """OpenAI-compatible chat completions endpoint with request queueing, streaming and response caching"""
    try:
        json_data = await request.json()
        
        # Extract model or use default
        model_id = json_data.get("model", DEFAULT_MODEL)
        messages = json_data.get("messages", [])
        temperature = json_data.get("temperature", 0.7)
        max_tokens = json_data.get("max_tokens", 1024)
        stream = json_data.get("stream", False)
        user = json_data.get("user", auth_info["user_id"])
        
        # Calculate a cache key based on the request parameters
        cache_key = calculate_cache_key(model_id, messages, temperature, max_tokens)
        
        # Check if we have a cached response in memory cache first (faster)
        cached_response = memory_cache.get(cache_key)
        if cached_response and not stream:  # Don't use cache for streaming requests
            # Update stats
            update_stats(model_id, "cache_hit")
            return cached_response
        
        # Check if we have a cached response in Modal's persistent cache
        if not cached_response and cache_key in response_dict and not stream:
            cached_response = response_dict[cache_key]
            cache_age = time.time() - cached_response.get("timestamp", 0)
            
            # Use cached response if it's fresh enough
            if cache_age < MAX_CACHE_AGE:
                # Update stats
                update_stats(model_id, "cache_hit")
                response_data = cached_response["response"]
                
                # Also cache in memory for faster access next time
                memory_cache.set(cache_key, response_data)
                
                return response_data
        
        # Select best model if "auto" is specified
        if model_id == "auto" and len(messages) > 0:
            # Get the last user message
            last_message = None
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    last_message = msg.get("content", "")
                    break
            
            if last_message:
                prompt = last_message
                # Select best model based on prompt and parameters
                model_id = select_best_model(prompt, max_tokens, temperature)
                logging.info(f"Auto-selected model: {model_id} for prompt")
        
        # Check if model exists
        if model_id not in VLLM_MODELS and model_id not in LLAMA_CPP_MODELS:
            # Default to the default model if specified model not found
            logging.warning(f"Model {model_id} not found, using default: {DEFAULT_MODEL}")
            model_id = DEFAULT_MODEL
        
        # Create a unique request ID
        request_id = str(uuid.uuid4())
        
        # Create request object
        gen_request = GenerationRequest(
            request_id=request_id,
            model_id=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=json_data.get("top_p", 1.0),
            frequency_penalty=json_data.get("frequency_penalty", 0.0),
            presence_penalty=json_data.get("presence_penalty", 0.0),
            user=user,
            stream=stream,
            api_key=auth_info["key"]
        )
        
        # For streaming requests, set up streaming response
        if stream:
            # Create a new stream
            stream_manager.create_stream(request_id)
            
            # Put the request in the queue
            await request_queue.put.aio(gen_request.model_dump())
            
            # Update stats
            update_stats(model_id, "request_count")
            update_stats(model_id, "stream_count")
            
            # Start a background worker to process the request if needed
            background_tasks.add_task(ensure_worker_running)
            
            # Return a streaming response using FastAPI's StreamingResponse
            from fastapi.responses import StreamingResponse as FastAPIStreamingResponse
            return FastAPIStreamingResponse(
                content=stream_response(request_id, model_id, auth_info["user_id"]),
                media_type="text/event-stream"
            )
            
        # For non-streaming, enqueue the request and wait for result
        # Put the request in the queue
        await request_queue.put.aio(gen_request.model_dump())
        
        # Update stats
        update_stats(model_id, "request_count")
        
        # Start a background worker to process the request if needed
        background_tasks.add_task(ensure_worker_running)
        
        # Wait for the response with timeout
        start_time = time.time()
        timeout = 120  # 2-minute timeout for non-streaming requests
        
        while time.time() - start_time < timeout:
            # Check memory cache first (faster)
            response_data = memory_cache.get(request_id)
            if response_data:
                # Update stats
                update_stats(model_id, "success_count")
                estimate_tokens(messages, response_data, auth_info["user_id"], model_id)
                
                # Save to persistent cache
                response_dict[cache_key] = {
                    "response": response_data,
                    "timestamp": time.time()
                }
                
                # Clean up request-specific cache
                memory_cache.set(request_id, None)
                
                return response_data
                
            # Check persistent cache
            if response_dict.contains(request_id):
                response_data = response_dict[request_id]
                
                # Remove from response dict to save memory
                try:
                    response_dict.pop(request_id)
                except Exception:
                    pass
                
                # Save to cache
                response_dict[cache_key] = {
                    "response": response_data,
                    "timestamp": time.time()
                }
                
                # Also cache in memory
                memory_cache.set(cache_key, response_data)
                
                # Update stats
                update_stats(model_id, "success_count")
                estimate_tokens(messages, response_data, auth_info["user_id"], model_id)
                
                return response_data
            
            # Wait a bit before checking again
            await asyncio.sleep(0.1)
        
        # If we get here, we timed out
        update_stats(model_id, "timeout_count")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request timed out. The model may be busy. Please try again later."
        )
            
    except Exception as e:
        logging.error(f"Error in chat completions: {str(e)}")
        # Update error stats
        if "model_id" in locals():
            update_stats(model_id, "error_count")
            
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating response: {str(e)}"
        )

async def stream_response(request_id: str, model_id: str, user_id: str) -> AsyncIterator[str]:
    """Stream response chunks to the client"""
    try:
        # Stream header
        yield "data: " + json.dumps({"object": "chat.completion.chunk"}) + "\n\n"
        
        # Stream chunks
        async for chunk in stream_manager.get_chunks(request_id):
            if chunk:
                yield f"data: {json.dumps(chunk)}\n\n"
        
        # Stream done
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logging.error(f"Error streaming response: {str(e)}")
        # Update error stats
        update_stats(model_id, "stream_error_count")
        
        # Send error as SSE
        error_json = json.dumps({"error": str(e)})
        yield f"data: {error_json}\n\n"
        yield "data: [DONE]\n\n"
        
async def ensure_worker_running():
    """Ensure that a worker is running to process the queue"""
    # Check if workers are already running via a sentinel in shared dict
    workers_running_key = "workers_running"
    
    if not model_stats_dict.contains(workers_running_key):
        model_stats_dict[workers_running_key] = 0
    
    current_workers = model_stats_dict[workers_running_key]
    
    # If no workers or too few workers, start more
    if current_workers < 3:  # Keep up to 3 workers running
        # Increment worker count
        model_stats_dict[workers_running_key] = current_workers + 1
        
        # Start a worker
        await process_queue_worker.spawn.aio()

def calculate_cache_key(model_id: str, messages: List[dict], temperature: float, max_tokens: int) -> str:
    """Calculate a deterministic cache key for a request using SHA-256"""
    # Create a simplified version of the request for cache key
    cache_dict = {
        "model": model_id,
        "messages": messages,
        "temperature": round(temperature, 2),  # Round to reduce variations
        "max_tokens": max_tokens
    }
    # Convert to a stable string representation and hash it with SHA-256
    cache_str = json.dumps(cache_dict, sort_keys=True)
    hash_obj = hashlib.sha256(cache_str.encode())
    return f"cache:{hash_obj.hexdigest()[:16]}"

def update_stats(model_id: str, stat_type: str):
    """Update usage statistics for a model"""
    if not model_stats_dict.contains(model_id):
        model_stats_dict[model_id] = {
            "request_count": 0,
            "success_count": 0,
            "error_count": 0,
            "timeout_count": 0,
            "cache_hit": 0,
            "token_count": 0,
            "avg_latency": 0
        }
    
    stats = model_stats_dict[model_id]
    stats[stat_type] = stats.get(stat_type, 0) + 1
    model_stats_dict[model_id] = stats
    
def estimate_tokens(messages: List[dict], response: dict, user_id: str, model_id: str):
    """Estimate token usage and update user quotas"""
    # Very simple token estimation based on whitespace-split words * 1.3
    input_tokens = 0
    for msg in messages:
        input_tokens += len(msg.get("content", "").split()) * 1.3
    
    output_tokens = 0
    if response and "choices" in response:
        for choice in response["choices"]:
            if "message" in choice and "content" in choice["message"]:
                output_tokens += len(choice["message"]["content"].split()) * 1.3
    
    # Update model stats
    if model_stats_dict.contains(model_id):
        stats = model_stats_dict[model_id]
        stats["token_count"] = stats.get("token_count", 0) + input_tokens + output_tokens
        model_stats_dict[model_id] = stats
    
    # Update user usage
    if user_id in user_usage_dict:
        usage = user_usage_dict[user_id]
        
        # Check if we need to reset daily counters
        last_reset = datetime.fromisoformat(usage["tokens"]["last_reset"])
        now = datetime.now()
        
        if now.date() > last_reset.date():
            # Reset daily counters
            usage["tokens"]["input"] = 0
            usage["tokens"]["output"] = 0
            usage["tokens"]["last_reset"] = now.isoformat()
        
        # Update token counts
        usage["tokens"]["input"] += int(input_tokens)
        usage["tokens"]["output"] += int(output_tokens)
        user_usage_dict[user_id] = usage

def select_best_model(prompt: str, n_predict: int, temperature: float) -> str:
    """
    Intelligently selects the best model based on input parameters.

    Args:
        prompt (str): The input prompt for the model.
        n_predict (int): The number of tokens to predict.
        temperature (float): The sampling temperature.

    Returns:
        str: The identifier of the best model to use.
    """
    # Check for code generation patterns
    code_indicators = ["```", "def ", "class ", "function", "import ", "from ", "<script", "<style", 
                      "SELECT ", "CREATE TABLE", "const ", "let ", "var ", "function(", "=>"]
    
    is_likely_code = any(indicator in prompt for indicator in code_indicators)
    
    # Check for creative writing patterns
    creative_indicators = ["story", "poem", "creative", "imagine", "fiction", "narrative", 
                          "write a", "compose", "create a"]
    
    is_creative_task = any(indicator in prompt.lower() for indicator in creative_indicators)
    
    # Check for analytical/reasoning tasks
    analytical_indicators = ["explain", "analyze", "compare", "contrast", "reason", 
                            "evaluate", "assess", "why", "how does"]
    
    is_analytical_task = any(indicator in prompt.lower() for indicator in analytical_indicators)
    
    # Decision logic
    if is_likely_code:
        # For code generation, prefer phi-4 for all code tasks
        return "phi-4"  # Excellent for code generation
            
    elif is_creative_task:
        # For creative tasks, use models with higher creativity
        if temperature > 0.8:
            return "deepseek-r1"  # More creative at high temperatures
        else:
            return "phi-4"  # Good balance of creativity and coherence
            
    elif is_analytical_task:
        # For analytical tasks, use models with strong reasoning
        return "phi-4"  # Strong reasoning capabilities
        
    # Length-based decisions
    if len(prompt) > 2000:
        # For very long prompts, use models with good context handling
        return "llama3-8b"
    elif len(prompt) < 1000:
        # For shorter prompts, prefer phi-4
        return "phi-4"
        
    # Temperature-based decisions
    if temperature < 0.5:
        # For deterministic outputs
        return "phi-4"
    elif temperature > 0.9:
        # For very creative outputs
        return "deepseek-r1"
        
    # Default to phi-4 instead of the standard model
    return "phi-4"

# vLLM serving function
@app.function(
    image=vllm_image,
    gpu="H100:1",
    allow_concurrent_inputs=100,
    volumes={
        f"{CACHE_DIR}/huggingface": hf_cache_vol,
        f"{CACHE_DIR}/vllm": vllm_cache_vol,
    },
    timeout=30 * MINUTES,
)
@modal.web_server(port=SERVER_PORT)
def serve_vllm_model(model_id: str = DEFAULT_MODEL):
    """
    Serves a model using vLLM with an OpenAI-compatible API.

    Args:
        model_id (str): The identifier of the model to serve. Defaults to DEFAULT_MODEL.

    Raises:
        ValueError: If the specified model_id is not found in VLLM_MODELS.
    """
    import subprocess
    
    if model_id not in VLLM_MODELS:
        available_models = list(VLLM_MODELS.keys())
        logging.error(f"Error: Unknown model: {model_id}. Available models: {available_models}")
        raise ValueError(f"Unknown model: {model_id}. Available models: {available_models}")
    
    model_info = VLLM_MODELS[model_id]
    model_name = model_info["name"]
    revision = model_info["revision"]
    
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Starting vLLM server with model: {model_name}")
    
    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        model_name,
        "--revision",
        revision,
        "--host",
        "0.0.0.0",
        "--port",
        str(SERVER_PORT),
        "--api-key",
        DEFAULT_API_KEY,
    ]

    # Use subprocess.run instead of Popen to ensure the server is fully started
    # before returning, and don't use shell=True for better process management
    process = subprocess.Popen(cmd)
    
    # Log that we've started the server
    logging.info(f"Started vLLM server with PID {process.pid}")

# Define the worker that will process the queue
@app.function(
    image=vllm_image,
    gpu=None,  # Worker will spawn GPU functions as needed
    allow_concurrent_inputs=10,
    volumes={
        f"{CACHE_DIR}/huggingface": hf_cache_vol,
    },
    timeout=30 * MINUTES,
)
async def process_queue_worker():
    """Worker function that processes requests from the queue"""
    import asyncio
    import time
    
    try:
        # Signal that we're starting a worker
        worker_id = str(uuid.uuid4())[:8]
        logging.info(f"Starting queue processing worker {worker_id}")
        
        # Process requests until timeout or empty queue
        empty_count = 0
        max_empty_count = 10  # Stop after 10 consecutive empty polls
        
        while empty_count < max_empty_count:
            # Try to get a request from the queue
            try:
                request_dict = await request_queue.get.aio(timeout_ms=5000)
                empty_count = 0  # Reset empty counter
                
                # Process the request
                try:
                    # Create request object
                    request_id = request_dict.get("request_id")
                    model_id = request_dict.get("model_id")
                    messages = request_dict.get("messages", [])
                    temperature = request_dict.get("temperature", 0.7)
                    max_tokens = request_dict.get("max_tokens", 1024)
                    api_key = request_dict.get("api_key", DEFAULT_API_KEY)
                    stream_mode = request_dict.get("stream", False)
                    
                    logging.info(f"Worker {worker_id} processing request {request_id} for model {model_id}")
                    
                    # Start time for latency calculation
                    start_time = time.time()
                    
                    if stream_mode:
                        # Generate streaming response
                        await generate_streaming_response(
                            request_id=request_id,
                            model_id=model_id,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            api_key=api_key
                        )
                    else:
                        # Generate non-streaming response
                        response = await generate_response(
                            model_id=model_id,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            api_key=api_key
                        )
                        
                        # Calculate latency
                        latency = time.time() - start_time
                        
                        # Update latency stats
                        if model_stats_dict.contains(model_id):
                            stats = model_stats_dict[model_id]
                            old_avg = stats.get("avg_latency", 0)
                            old_count = stats.get("success_count", 0) 
                            
                            # Calculate new average (moving average)
                            if old_count > 0:
                                new_avg = (old_avg * old_count + latency) / (old_count + 1)
                            else:
                                new_avg = latency
                                
                            stats["avg_latency"] = new_avg
                            model_stats_dict[model_id] = stats
                        
                        # Store the response in both caches
                        memory_cache.set(request_id, response)
                        response_dict[request_id] = response
                        
                        logging.info(f"Worker {worker_id} completed request {request_id} in {latency:.2f}s")
                    
                except Exception as e:
                    # Log error and move on
                    logging.error(f"Worker {worker_id} error processing request {request_id}: {str(e)}")
                    
                    # Create error response
                    error_response = {
                        "error": {
                            "message": str(e),
                            "type": "internal_error",
                            "code": 500
                        }
                    }
                    
                    # Store the error as a response
                    memory_cache.set(request_id, error_response)
                    response_dict[request_id] = error_response
                    
                    # If streaming, send error and finish stream
                    if "stream_mode" in locals() and stream_mode:
                        stream_manager.add_chunk(request_id, {
                            "id": f"chatcmpl-{int(time.time())}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_id,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": f"Error: {str(e)}"},
                                "finish_reason": "error"
                            }]
                        })
                        stream_manager.finish_stream(request_id)
            
            except asyncio.TimeoutError:
                # No requests in queue
                empty_count += 1
                logging.info(f"Worker {worker_id}: No requests in queue. Empty count: {empty_count}")
                
                # Clean up expired cache entries and old streams
                if empty_count % 5 == 0:  # Every 5 empty polls
                    memory_cache.clear_expired()
                    stream_manager.clean_old_streams()
                
                await asyncio.sleep(1)  # Wait a bit before checking again
        
        # If we get here, we've had too many consecutive empty polls
        logging.info(f"Worker {worker_id} shutting down due to empty queue")
        
    finally:
        # Signal that this worker is done
        workers_running_key = "workers_running"
        if model_stats_dict.contains(workers_running_key):
            current_workers = model_stats_dict[workers_running_key]
            model_stats_dict[workers_running_key] = max(0, current_workers - 1)
            logging.info(f"Worker {worker_id} shutdown. Workers remaining: {max(0, current_workers - 1)}")

async def generate_streaming_response(
    request_id: str,
    model_id: str,
    messages: List[dict],
    temperature: float,
    max_tokens: int,
    api_key: str
):
    """
    Generate a streaming response and send chunks to the stream manager.
    
    Args:
        request_id: The unique ID for this request
        model_id: The ID of the model to use
        messages: The chat messages
        temperature: The sampling temperature
        max_tokens: The maximum tokens to generate
        api_key: The API key for authentication
    """
    import httpx
    import time
    import json
    import asyncio
    
    try:
        # Create response ID
        response_id = f"chatcmpl-{int(time.time())}"
        
        if model_id in VLLM_MODELS:
            # Start vLLM server for this model
            server_url = await serve_vllm_model.remote(model_id=model_id)
            
            # Need to wait for server startup
            await wait_for_server(serve_vllm_model.web_url, timeout=120)
            
            # Forward request to vLLM with streaming enabled
            async with httpx.AsyncClient(timeout=120.0) as client:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream"
                }
                
                # Format request for vLLM OpenAI-compatible endpoint
                vllm_request = {
                    "model": VLLM_MODELS[model_id]["name"],
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": True
                }
                
                # Make streaming request
                async with client.stream(
                    "POST",
                    f"{serve_vllm_model.web_url}/v1/chat/completions",
                    json=vllm_request,
                    headers=headers
                ) as response:
                    # Process streaming response
                    buffer = ""
                    async for chunk in response.aiter_text():
                        buffer += chunk
                        
                        # Process complete SSE messages
                        while "\n\n" in buffer:
                            message, buffer = buffer.split("\n\n", 1)
                            
                            if message.startswith("data: "):
                                data = message[6:]  # Remove "data: " prefix
                                
                                if data == "[DONE]":
                                    # End of stream
                                    stream_manager.finish_stream(request_id)
                                    return
                                
                                try:
                                    # Parse JSON data
                                    chunk_data = json.loads(data)
                                    # Forward to client
                                    stream_manager.add_chunk(request_id, chunk_data)
                                except json.JSONDecodeError:
                                    logging.error(f"Invalid JSON in stream: {data}")
                    
                    # Ensure stream is finished
                    stream_manager.finish_stream(request_id)
                    
        elif model_id in LLAMA_CPP_MODELS:
            # For llama.cpp models, we need to simulate streaming
            # First convert the chat format to a prompt
            prompt = format_messages_to_prompt(messages)
            
            # Run llama.cpp with the prompt
            output = await run_llama_cpp_stream.remote(
                model_id=model_id,
                prompt=prompt,
                n_predict=max_tokens,
                temperature=temperature,
                request_id=request_id
            )
            
            # Streaming is handled by the run_llama_cpp_stream function
            # which directly adds chunks to the stream manager
            
            # Wait for completion signal
            while True:
                if request_id in stream_queues and stream_queues[request_id] == "DONE":
                    # Clean up
                    stream_queues.pop(request_id)
                    break
                await asyncio.sleep(0.1)
                
        else:
            raise ValueError(f"Unknown model: {model_id}")
            
    except Exception as e:
        logging.error(f"Error in streaming generation: {str(e)}")
        # Send error chunk
        stream_manager.add_chunk(request_id, {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{
                "index": 0,
                "delta": {"content": f"Error: {str(e)}"},
                "finish_reason": "error"
            }]
        })
        # Finish stream
        stream_manager.finish_stream(request_id)

async def generate_response(model_id: str, messages: List[dict], temperature: float, max_tokens: int, api_key: str):
    """
    Generate a response using the appropriate model based on model_id.
    
    Args:
        model_id: The ID of the model to use
        messages: The chat messages
        temperature: The sampling temperature
        max_tokens: The maximum tokens to generate
        api_key: The API key for authentication
        
    Returns:
        A response in OpenAI-compatible format
    """
    import httpx
    import time
    import json
    import asyncio
    
    if model_id in VLLM_MODELS:
        # Start vLLM server for this model
        server_url = await serve_vllm_model.remote(model_id=model_id)
        
        # Need to wait for server startup
        await wait_for_server(serve_vllm_model.web_url, timeout=120)
        
        # Forward request to vLLM
        async with httpx.AsyncClient(timeout=60.0) as client:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Format request for vLLM OpenAI-compatible endpoint
            vllm_request = {
                "model": VLLM_MODELS[model_id]["name"],
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = await client.post(
                f"{serve_vllm_model.web_url}/v1/chat/completions",
                json=vllm_request,
                headers=headers
            )
            
            return response.json()
    elif model_id in LLAMA_CPP_MODELS:
        # For llama.cpp models, use the run_llama_cpp function
        # First convert the chat format to a prompt
        prompt = format_messages_to_prompt(messages)
        
        # Run llama.cpp with the prompt
        output = await run_llama_cpp.remote(
            model_id=model_id,
            prompt=prompt,
            n_predict=max_tokens,
            temperature=temperature
        )
        
        # Format the response in the OpenAI format
        completion_text = output.strip()
        finish_reason = "stop" if len(completion_text) < max_tokens else "length"
        
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": completion_text
                    },
                    "finish_reason": finish_reason
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt) // 4,  # Rough estimation
                "completion_tokens": len(completion_text) // 4,  # Rough estimation
                "total_tokens": (len(prompt) + len(completion_text)) // 4  # Rough estimation
            }
        }
    else:
        raise ValueError(f"Unknown model: {model_id}")

def format_messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Convert chat messages to a text prompt format for llama.cpp.
    
    Args:
        messages: List of message dictionaries with role and content
    
    Returns:
        Formatted prompt string
    """
    formatted_prompt = ""
    
    for message in messages:
        role = message.get("role", "").lower()
        content = message.get("content", "")
        
        if role == "system":
            formatted_prompt += f"<|system|>\n{content}\n"
        elif role == "user":
            formatted_prompt += f"<|user|>\n{content}\n"
        elif role == "assistant":
            formatted_prompt += f"<|assistant|>\n{content}\n"
        else:
            # For unknown roles, treat as user
            formatted_prompt += f"<|user|>\n{content}\n"
    
    # Add final assistant marker to prompt the model to respond
    formatted_prompt += "<|assistant|>\n"
    
    return formatted_prompt

async def wait_for_server(url: str, timeout: int = 120, check_interval: int = 2):
    """
    Wait for a server to be ready by checking its health endpoint.
    
    Args:
        url: The base URL of the server
        timeout: Maximum time to wait in seconds
        check_interval: Interval between checks in seconds
    
    Returns:
        True if server is ready, False otherwise
    """
    import httpx
    import asyncio
    import time
    
    start_time = time.time()
    health_url = f"{url}/health"
    
    logging.info(f"Waiting for server at {url} to be ready...")
    
    while time.time() - start_time < timeout:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(health_url)
                if response.status_code == 200:
                    logging.info(f"Server at {url} is ready!")
                    return True
        except Exception as e:
            elapsed = time.time() - start_time
            logging.info(f"Server not ready yet after {elapsed:.1f}s: {str(e)}")
            
        await asyncio.sleep(check_interval)
    
    logging.error(f"Timed out waiting for server at {url} after {timeout} seconds")
    return False

@app.function(
    image=llama_cpp_image,
    gpu=None,  # Will be set dynamically based on model
    volumes={
        f"{CACHE_DIR}/huggingface": hf_cache_vol,
        f"{CACHE_DIR}/llama_cpp": llama_cpp_cache_vol,
        RESULTS_DIR: results_vol,
    },
    timeout=30 * MINUTES,
)
async def run_llama_cpp_stream(
    model_id: str,
    prompt: str,
    n_predict: int = 1024,
    temperature: float = 0.7,
    request_id: str = None,
):
    """
    Run streaming inference with llama.cpp for models like DeepSeek-R1 and Phi-4
    """
    import subprocess
    import os
    import json
    import time
    import threading
    from uuid import uuid4
    from pathlib import Path
    from huggingface_hub import snapshot_download
    
    if model_id not in LLAMA_CPP_MODELS:
        available_models = list(LLAMA_CPP_MODELS.keys())
        error_msg = f"Unknown model: {model_id}. Available models: {available_models}"
        logging.error(error_msg)
        
        if request_id:
            # Send error to stream
            stream_manager.add_chunk(request_id, {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "delta": {"content": f"Error: {error_msg}"},
                    "finish_reason": "error"
                }]
            })
            stream_manager.finish_stream(request_id)
            # Signal completion
            stream_queues[request_id] = "DONE"
            
        raise ValueError(error_msg)
    
    model_info = LLAMA_CPP_MODELS[model_id]
    repo_id = model_info["name"]
    pattern = model_info["pattern"]
    revision = model_info["revision"]
    quant = model_info["quant"]
    
    # Download model if not already cached
    logging.info(f"Downloading model {repo_id} if not present")
    try:
        model_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=f"{CACHE_DIR}/llama_cpp",
            allow_patterns=[pattern],
        )
    except ValueError as e:
        if "hf_transfer" in str(e):
            # Fallback to standard download if hf_transfer fails
            logging.warning("hf_transfer failed, falling back to standard download")
            # Temporarily disable hf_transfer
            import os
            old_env = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "1")
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
            try:
                model_path = snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    local_dir=f"{CACHE_DIR}/llama_cpp",
                    allow_patterns=[pattern],
                )
            finally:
                # Restore original setting
                os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = old_env
        else:
            raise
    
    # Find the model file
    model_files = list(Path(model_path).glob(pattern))
    if not model_files:
        error_msg = f"No model files found matching pattern {pattern}"
        logging.error(error_msg)
        
        if request_id:
            # Send error to stream
            stream_manager.add_chunk(request_id, {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "delta": {"content": f"Error: {error_msg}"},
                    "finish_reason": "error"
                }]
            })
            stream_manager.finish_stream(request_id)
            # Signal completion
            stream_queues[request_id] = "DONE"
            
        raise FileNotFoundError(error_msg)
    
    model_file = str(model_files[0])
    logging.info(f"Using model file: {model_file}")
    
    # Set up command
    cmd = [
        "llama-cli",
        "--model", model_file,
        "--prompt", prompt,
        "--n-predict", str(n_predict),
        "--temp", str(temperature),
        "--ctx-size", "8192",
    ]
    
    # Add GPU layers if needed
    if model_info["gpu"] is not None:
        cmd.extend(["--n-gpu-layers", "9999"])  # Use all layers on GPU
    
    # Run inference
    result_id = str(uuid4())
    logging.info(f"Running streaming inference with ID: {result_id}")
    
    # Create response ID for streaming
    response_id = f"chatcmpl-{int(time.time())}"
    
    # Function to process output in real-time and send to stream
    def process_output(process, request_id):
        content_buffer = ""
        last_send_time = time.time()
        
        # Send initial chunk with role
        if request_id:
            stream_manager.add_chunk(request_id, {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                }]
            })
        
        for line in iter(process.stdout.readline, b''):
            try:
                line_str = line.decode('utf-8', errors='replace')
                
                # Skip llama.cpp info lines
                if line_str.startswith("llama_"):
                    continue
                
                # Add to buffer
                content_buffer += line_str
                
                # Send chunks at reasonable intervals or when buffer gets large
                now = time.time()
                if (now - last_send_time > 0.1 or len(content_buffer) > 20) and request_id:
                    # Send chunk
                    stream_manager.add_chunk(request_id, {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_id,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": content_buffer},
                        }]
                    })
                    
                    # Reset buffer and time
                    content_buffer = ""
                    last_send_time = now
                    
            except Exception as e:
                logging.error(f"Error processing output: {str(e)}")
        
        # Send any remaining content
        if content_buffer and request_id:
            stream_manager.add_chunk(request_id, {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "delta": {"content": content_buffer},
                }]
            })
        
        # Send final chunk with finish reason
        if request_id:
            stream_manager.add_chunk(request_id, {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            })
            
            # Finish stream
            stream_manager.finish_stream(request_id)
            
            # Signal completion
            stream_queues[request_id] = "DONE"
    
    # Start process
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        bufsize=1  # Line buffered
    )
    
    # Start output processing thread if streaming
    if request_id:
        thread = threading.Thread(target=process_output, args=(process, request_id))
        thread.daemon = True
        thread.start()
        
        # Return immediately for streaming
        return "Streaming in progress"
    else:
        # For non-streaming, collect all output
        stdout, stderr = collect_output(process)
        
        # Save results
        result_dir = Path(RESULTS_DIR) / result_id
        result_dir.mkdir(parents=True, exist_ok=True)
        
        (result_dir / "output.txt").write_text(stdout)
        (result_dir / "stderr.txt").write_text(stderr)
        (result_dir / "prompt.txt").write_text(prompt)
        
        logging.info(f"Results saved to {result_dir}")
        return stdout

@app.function(
    image=llama_cpp_image,
    gpu=None,  # Will be set dynamically based on model
    volumes={
        f"{CACHE_DIR}/huggingface": hf_cache_vol,
        f"{CACHE_DIR}/llama_cpp": llama_cpp_cache_vol,
        RESULTS_DIR: results_vol,
    },
    timeout=30 * MINUTES,
)
async def run_llama_cpp(
    model_id: str,
    prompt: str = "Tell me about Modal and how it helps with ML deployments.",
    n_predict: int = 1024,
    temperature: float = 0.7,
):
    """
    Run inference with llama.cpp for models like DeepSeek-R1 and Phi-4
    """
    import subprocess
    import os
    from uuid import uuid4
    from pathlib import Path
    from huggingface_hub import snapshot_download
    
    if model_id not in LLAMA_CPP_MODELS:
        available_models = list(LLAMA_CPP_MODELS.keys())
        print(f"Error: Unknown model: {model_id}. Available models: {available_models}")
        raise ValueError(f"Unknown model: {model_id}. Available models: {available_models}")
    
    model_info = LLAMA_CPP_MODELS[model_id]
    repo_id = model_info["name"]
    pattern = model_info["pattern"]
    revision = model_info["revision"]
    quant = model_info["quant"]
    
    # Download model if not already cached
    logging.info(f"Downloading model {repo_id} if not present")
    try:
        model_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=f"{CACHE_DIR}/llama_cpp",
            allow_patterns=[pattern],
        )
    except ValueError as e:
        if "hf_transfer" in str(e):
            # Fallback to standard download if hf_transfer fails
            logging.warning("hf_transfer failed, falling back to standard download")
            # Temporarily disable hf_transfer
            import os
            old_env = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "1")
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
            try:
                model_path = snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    local_dir=f"{CACHE_DIR}/llama_cpp",
                    allow_patterns=[pattern],
                )
            finally:
                # Restore original setting
                os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = old_env
        else:
            raise
    
    # Find the model file
    model_files = list(Path(model_path).glob(pattern))
    if not model_files:
        logging.error(f"No model files found matching pattern {pattern}")
        raise FileNotFoundError(f"No model files found matching pattern {pattern}")
    
    model_file = str(model_files[0])
    print(f"Using model file: {model_file}")
    
    # Set up command
    cmd = [
        "llama-cli",
        "--model", model_file,
        "--prompt", prompt,
        "--n-predict", str(n_predict),
        "--temp", str(temperature),
        "--ctx-size", "8192",
    ]
    
    # Add GPU layers if needed
    if model_info["gpu"] is not None:
        cmd.extend(["--n-gpu-layers", "9999"])  # Use all layers on GPU
    
    # Run inference
    result_id = str(uuid4())
    print(f"Running inference with ID: {result_id}")
    
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False
    )
    
    stdout, stderr = collect_output(process)
    
    # Save results
    result_dir = Path(RESULTS_DIR) / result_id
    result_dir.mkdir(parents=True, exist_ok=True)
    
    (result_dir / "output.txt").write_text(stdout)
    (result_dir / "stderr.txt").write_text(stderr)
    (result_dir / "prompt.txt").write_text(prompt)
    
    print(f"Results saved to {result_dir}")
    return stdout

@app.function(
    image=vllm_image,
    volumes={
        f"{CACHE_DIR}/huggingface": hf_cache_vol,
    },
)
def list_available_models():
    """
    Lists available models that can be used with this server.

    Returns:
        dict: A dictionary containing lists of available vLLM and llama.cpp models.
    """
    print("Available vLLM models:")
    for model_id, model_info in VLLM_MODELS.items():
        print(f"- {model_id}: {model_info['name']}")
    
    print("\nAvailable llama.cpp models:")
    for model_id, model_info in LLAMA_CPP_MODELS.items():
        gpu_info = f"(GPU: {model_info['gpu']})" if model_info['gpu'] else "(CPU)"
        print(f"- {model_id}: {model_info['name']} {gpu_info}")
    
    return {
        "vllm": list(VLLM_MODELS.keys()),
        "llama_cpp": list(LLAMA_CPP_MODELS.keys())
    }

def collect_output(process):
    """
    Collect output from a process while streaming it.

    Args:
        process: The process from which to collect output.

    Returns:
        tuple: A tuple containing the collected stdout and stderr as strings.
    """
    import sys
    from queue import Queue
    from threading import Thread
    
    def stream_output(stream, queue, write_stream):
        for line in iter(stream.readline, b""):
            line_str = line.decode("utf-8", errors="replace")
            write_stream.write(line_str)
            write_stream.flush()
            queue.put(line_str)
        stream.close()
    
    stdout_queue = Queue()
    stderr_queue = Queue()
    
    stdout_thread = Thread(target=stream_output, args=(process.stdout, stdout_queue, sys.stdout))
    stderr_thread = Thread(target=stream_output, args=(process.stderr, stderr_queue, sys.stderr))
    
    stdout_thread.start()
    stderr_thread.start()
    
    stdout_thread.join()
    stderr_thread.join()
    process.wait()
    
    stdout_collected = "".join(list(stdout_queue.queue))
    stderr_collected = "".join(list(stderr_queue.queue))
    
    return stdout_collected, stderr_collected

# Main ASGI app for Modal
@app.function(
    image=vllm_image,
    gpu=None,  # No GPU for the API frontend
    allow_concurrent_inputs=100,
    volumes={
        f"{CACHE_DIR}/huggingface": hf_cache_vol,
    },
)
@modal.asgi_app()
def inference_api():
    """The main ASGI app that serves the FastAPI application"""
    return api_app

@app.local_entrypoint()
def main(
    prompt: str = "What can you tell me about Modal?",
    n_predict: int = 1024,
    temperature: float = 0.7,
    create_admin_key: bool = False,
    stream: bool = False,
    model: str = "auto",
    load_model: str = None,
    load_hf_model: str = None,
    hf_model_type: str = "vllm",
):
    """
    Main entrypoint for testing the API
    """
    import json
    import time
    import urllib.request
    
    # Initialize the API
    print(f"Starting API at {inference_api.web_url}")
    
    # Wait for API to be ready
    print("Checking if API is ready...")
    up, start, delay = False, time.time(), 10
    while not up:
        try:
            with urllib.request.urlopen(inference_api.web_url + "/health") as response:
                if response.getcode() == 200:
                    up = True
        except Exception:
            if time.time() - start > 5 * MINUTES:
                break
            time.sleep(delay)

    assert up, f"Failed health check for API at {inference_api.web_url}"
    print(f"API is up and running at {inference_api.web_url}")
    
    # Create a test API key if requested
    if create_admin_key:
        print("Creating a test API key...")
        key_request = {
            "user_id": "test_user",
            "rate_limit": 120,
            "quota": 2000000
        }
        headers = {
            "Authorization": f"Bearer {DEFAULT_API_KEY}",  # Admin key
            "Content-Type": "application/json",
        }
        req = urllib.request.Request(
            inference_api.web_url + "/admin/api-keys",
            data=json.dumps(key_request).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode())
                print("Created API key:")
                print(json.dumps(result, indent=2))
                # Use this key for the test message
                test_key = result["key"]
        except Exception as e:
            print(f"Error creating API key: {str(e)}")
            test_key = DEFAULT_API_KEY
    else:
        test_key = DEFAULT_API_KEY
            
    # List available models
    print("\nAvailable models:")
    try:
        headers = {
            "Authorization": f"Bearer {test_key}",
            "Content-Type": "application/json",
        }
        req = urllib.request.Request(
            inference_api.web_url + "/v1/models",
            headers=headers,
            method="GET",
        )
        with urllib.request.urlopen(req) as response:
            models = json.loads(response.read().decode())
            print(json.dumps(models, indent=2))
    except Exception as e:
        print(f"Error listing models: {str(e)}")
        
    # Select best model for the prompt
    model = select_best_model(prompt, n_predict, temperature)
    
    # Send a test message
    print(f"\nSending a sample message to {inference_api.web_url}")
    messages = [{"role": "user", "content": prompt}]

    headers = {
        "Authorization": f"Bearer {test_key}",
        "Content-Type": "application/json",
    }
    payload = json.dumps({
        "messages": messages, 
        "model": model,
        "temperature": temperature,
        "max_tokens": n_predict,
        "stream": stream
    })
    req = urllib.request.Request(
        inference_api.web_url + "/v1/chat/completions",
        data=payload.encode("utf-8"),
        headers=headers,
        method="POST",
    )
    
    try:
        if stream:
            print("Streaming response:")
            with urllib.request.urlopen(req) as response:
                for line in response:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:].strip()
                        if data == '[DONE]':
                            print("\n[DONE]")
                        else:
                            try:
                                chunk = json.loads(data)
                                if 'choices' in chunk and len(chunk['choices']) > 0:
                                    if 'delta' in chunk['choices'][0] and 'content' in chunk['choices'][0]['delta']:
                                        content = chunk['choices'][0]['delta']['content']
                                        print(content, end='', flush=True)
                            except json.JSONDecodeError:
                                print(f"Error parsing: {data}")
        else:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode())
                print("Response:")
                print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Check API stats
    print("\nChecking API stats...")
    headers = {
        "Authorization": f"Bearer {DEFAULT_API_KEY}",  # Admin key
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(
        inference_api.web_url + "/admin/stats",
        headers=headers,
        method="GET",
    )
    try:
        with urllib.request.urlopen(req) as response:
            stats = json.loads(response.read().decode())
            print("API Stats:")
            print(json.dumps(stats, indent=2))
    except Exception as e:
        print(f"Error getting stats: {str(e)}")
        
    # Start a worker if none running
    try:
        current_workers = stats.get("queue", {}).get("active_workers", 0)
        if current_workers < 1:
            print("\nStarting a queue worker...")
            process_queue_worker.spawn()
    except Exception as e:
        print(f"Error starting worker: {str(e)}")
        
    print(f"\nAPI is available at {inference_api.web_url}")
    print(f"Documentation is at {inference_api.web_url}/docs")
    print(f"Default Bearer token: {DEFAULT_API_KEY}")
    
    if create_admin_key:
        print(f"Test Bearer token: {test_key}")
        
    # If a model was specified to load, load it
    if load_model:
        print(f"\nLoading model: {load_model}")
        load_url = f"{inference_api.web_url}/admin/models/load"
        headers = {
            "Authorization": f"Bearer {test_key}",
            "Content-Type": "application/json",
        }
        payload = json.dumps({
            "model_id": load_model,
            "force_reload": True
        })
        req = urllib.request.Request(
            load_url,
            data=payload.encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode())
                print("Load response:")
                print(json.dumps(result, indent=2))
                
                # If it's a small model, wait a bit for it to load
                if load_model in ["tiny-llama-1.1b", "phi-2"]:
                    print(f"Waiting for {load_model} to load...")
                    time.sleep(10)
                    
                    # Check status
                    status_url = f"{inference_api.web_url}/admin/models/status/{load_model}"
                    status_req = urllib.request.Request(
                        status_url,
                        headers={"Authorization": f"Bearer {test_key}"},
                        method="GET",
                    )
                    with urllib.request.urlopen(status_req) as status_response:
                        status_result = json.loads(status_response.read().decode())
                        print("Model status:")
                        print(json.dumps(status_result, indent=2))
                
                # Use this model for the test
                model = load_model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            
    # If a HF model was specified to load directly
    if load_hf_model:
        print(f"\nLoading HF model: {load_hf_model} with type {hf_model_type}")
        load_url = f"{inference_api.web_url}/admin/models/load-from-hf"
        headers = {
            "Authorization": f"Bearer {test_key}",
            "Content-Type": "application/json",
        }
        payload = json.dumps({
            "repo_id": load_hf_model,
            "model_type": hf_model_type,
            "max_tokens": n_predict
        })
        req = urllib.request.Request(
            load_url,
            data=payload.encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode())
                print("HF Load response:")
                print(json.dumps(result, indent=2))
                
                # Get the model_id from the response
                hf_model_id = result.get("model_id")
                
                # Wait a bit for it to start loading
                print(f"Waiting for {load_hf_model} to start loading...")
                time.sleep(5)
                
                # Check status
                if hf_model_id:
                    status_url = f"{inference_api.web_url}/admin/models/status/{hf_model_id}"
                    status_req = urllib.request.Request(
                        status_url,
                        headers={"Authorization": f"Bearer {test_key}"},
                        method="GET",
                    )
                    with urllib.request.urlopen(status_req) as status_response:
                        status_result = json.loads(status_response.read().decode())
                        print("Model status:")
                        print(json.dumps(status_result, indent=2))
                
                # Use this model for the test
                if hf_model_id:
                    model = hf_model_id
        except Exception as e:
            print(f"Error loading HF model: {str(e)}")

    # Show curl examples
    print("\nExample curl commands:")
    
    # Regular completion
    print(f"""# Regular completion:
curl -X POST {inference_api.web_url}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer {test_key}" \\
  -d '{{
    "model": "{model}",
    "messages": [
      {{
        "role": "user",
        "content": "Hello, how can you help me today?"
      }}
    ],
    "temperature": 0.7,
    "max_tokens": 500
  }}'""")
    
    # Streaming completion
    print(f"""\n# Streaming completion:
curl -X POST {inference_api.web_url}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer {test_key}" \\
  -d '{{
    "model": "{model}",
    "messages": [
      {{
        "role": "user",
        "content": "Write a short story about AI"
      }}
    ],
    "temperature": 0.8,
    "max_tokens": 1000,
    "stream": true
  }}' --no-buffer""")
    
    # List models
    print(f"""\n# List available models:
curl -X GET {inference_api.web_url}/v1/models \\
  -H "Authorization: Bearer {test_key}" """)
    
    # Model management commands
    print(f"""\n# Load a model:
curl -X POST {inference_api.web_url}/admin/models/load \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer {test_key}" \\
  -d '{{
    "model_id": "phi-2",
    "force_reload": false
  }}'""")
    
    print(f"""\n# Load a model directly from Hugging Face:
curl -X POST {inference_api.web_url}/admin/models/load-from-hf \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer {test_key}" \\
  -d '{{
    "repo_id": "microsoft/phi-2",
    "model_type": "vllm",
    "max_tokens": 4096
  }}'""")
    
    print(f"""\n# Get model status:
curl -X GET {inference_api.web_url}/admin/models/status/phi-2 \\
  -H "Authorization: Bearer {test_key}" """)
    
    print(f"""\n# Unload a model:
curl -X POST {inference_api.web_url}/admin/models/unload \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer {test_key}" \\
  -d '{{
    "model_id": "phi-2"
  }}'""")
async def preload_llama_cpp_model(model_id: str):
    """Preload a llama.cpp model to make inference faster on first request"""
    if model_id not in LLAMA_CPP_MODELS:
        logging.error(f"Unknown model: {model_id}")
        return
    
    try:
        # Run a simple inference to load the model
        logging.info(f"Preloading llama.cpp model: {model_id}")
        await run_llama_cpp.remote(
            model_id=model_id,
            prompt="Hello, this is a test to preload the model.",
            n_predict=10,
            temperature=0.7
        )
        logging.info(f"Successfully preloaded llama.cpp model: {model_id}")
    except Exception as e:
        logging.error(f"Error preloading llama.cpp model {model_id}: {str(e)}")
        # Mark as not loaded
        LLAMA_CPP_MODELS[model_id]["loaded"] = False
