#!/bin/bash

# OpenAI Code Assistant Deployment Script

# Default values
HOST="127.0.0.1"
PORT=8000
WORKERS=1
ENABLE_REPLICATION=false
PRIMARY=true
PEERS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --enable-replication)
      ENABLE_REPLICATION=true
      shift
      ;;
    --secondary)
      PRIMARY=false
      shift
      ;;
    --peer)
      if [ -z "$PEERS" ]; then
        PEERS="--peer $2"
      else
        PEERS="$PEERS --peer $2"
      fi
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
  echo "Error: OPENAI_API_KEY environment variable is not set"
  echo "Please set it with: export OPENAI_API_KEY=your_api_key"
  exit 1
fi

# Create log directory if it doesn't exist
mkdir -p logs

# Start the server
echo "Starting OpenAI Code Assistant API Server..."
echo "Host: $HOST"
echo "Port: $PORT"
echo "Workers: $WORKERS"
echo "Replication: $ENABLE_REPLICATION"
echo "Role: $([ "$PRIMARY" = true ] && echo "Primary" || echo "Secondary")"
echo "Peers: $PEERS"

# Build the command
CMD="python cli.py serve --host $HOST --port $PORT --workers $WORKERS"

if [ "$ENABLE_REPLICATION" = true ]; then
  CMD="$CMD --enable-replication"
fi

if [ "$PRIMARY" = false ]; then
  CMD="$CMD --secondary"
fi

if [ -n "$PEERS" ]; then
  CMD="$CMD $PEERS"
fi

# Run the command
echo "Running: $CMD"
$CMD > logs/server_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Save the PID
echo $! > server.pid
echo "Server started with PID $(cat server.pid)"
echo "Logs are being written to logs/server_*.log"
