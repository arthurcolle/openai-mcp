#!/usr/bin/env python3
"""
Deployment script for Modal MCP Server
"""
import os
import sys
import argparse
import subprocess
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import modal
        import httpx
        import fastapi
        import uvicorn
        print("✅ All required dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install required dependencies:")
        print("pip install modal httpx fastapi uvicorn")
        return False

def deploy_modal_server(args):
    """Deploy the Modal OpenAI-compatible server"""
    print("Deploying Modal OpenAI-compatible server...")
    
    # Run the Modal deployment command
    cmd = ["modal", "deploy", "modal_mcp_server.py"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Error deploying Modal server: {result.stderr}")
            return None
        
        # Extract the deployment URL from the output
        for line in result.stdout.splitlines():
            if "https://" in line and "modal.run" in line:
                url = line.strip()
                print(f"✅ Modal server deployed at: {url}")
                return url
        
        print("❌ Could not find deployment URL in output")
        print(result.stdout)
        return None
        
    except Exception as e:
        print(f"❌ Error deploying Modal server: {e}")
        return None

def deploy_mcp_adapter(modal_url, args):
    """Deploy the MCP adapter server"""
    print("Deploying MCP adapter server...")
    
    # Set environment variables for the adapter
    os.environ["MODAL_API_URL"] = modal_url
    os.environ["MODAL_API_KEY"] = args.api_key
    os.environ["DEFAULT_MODEL"] = args.model
    
    # Start the adapter server
    try:
        import uvicorn
        from mcp_modal_adapter import app
        
        # Start in a separate process if not in foreground mode
        if not args.foreground:
            print(f"Starting MCP adapter server on port {args.port}...")
            cmd = [
                sys.executable, "-m", "uvicorn", "mcp_modal_adapter:app", 
                "--host", "0.0.0.0", "--port", str(args.port)
            ]
            
            # Use subprocess.Popen to run in background
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE if not args.verbose else None,
                stderr=subprocess.PIPE if not args.verbose else None
            )
            
            # Wait a bit to make sure it starts
            time.sleep(2)
            
            # Check if process is still running
            if process.poll() is None:
                print(f"✅ MCP adapter server running on http://localhost:{args.port}")
                return f"http://localhost:{args.port}"
            else:
                stdout, stderr = process.communicate()
                print(f"❌ Error starting MCP adapter server: {stderr.decode() if stderr else 'Unknown error'}")
                return None
        else:
            # Run in foreground
            print(f"Starting MCP adapter server on port {args.port} in foreground mode...")
            uvicorn.run(app, host="0.0.0.0", port=args.port)
            return None  # Will never reach here in foreground mode
            
    except Exception as e:
        print(f"❌ Error starting MCP adapter server: {e}")
        return None

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Deploy Modal MCP Server")
    parser.add_argument("--port", type=int, default=8000, help="Port for MCP adapter server")
    parser.add_argument("--api-key", type=str, default="sk-modal-llm-api-key", help="API key for Modal server")
    parser.add_argument("--model", type=str, default="phi-4", help="Default model to use")
    parser.add_argument("--foreground", action="store_true", help="Run MCP adapter in foreground")
    parser.add_argument("--verbose", action="store_true", help="Show verbose output")
    parser.add_argument("--skip-modal-deploy", action="store_true", help="Skip Modal server deployment")
    parser.add_argument("--modal-url", type=str, help="Use existing Modal server URL")
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Deploy Modal server if not skipped
    modal_url = args.modal_url
    if not args.skip_modal_deploy and not modal_url:
        modal_url = deploy_modal_server(args)
        if not modal_url:
            return 1
    
    # Deploy MCP adapter
    mcp_url = deploy_mcp_adapter(modal_url, args)
    if not mcp_url and not args.foreground:
        return 1
    
    # Open browser if not in foreground mode
    if mcp_url and not args.foreground:
        print(f"Opening browser to MCP server health check...")
        webbrowser.open(f"{mcp_url}/health")
        
        print("\nMCP Server is now running!")
        print(f"- Health check: {mcp_url}/health")
        print(f"- List prompts: {mcp_url}/prompts")
        print(f"- Modal API: {modal_url}")
        
        print("\nPress Ctrl+C to stop the server")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping server...")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
