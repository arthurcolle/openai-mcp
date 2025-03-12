#!/usr/bin/env python3
"""
direct_o1_client.py - Client script to directly call OpenAI API with o1 model and medium reasoning effort.
"""

import os
import json
import sys
import time
from openai import OpenAI

# Default configuration
MODEL = "o1"
REASONING_EFFORT = "medium"

def call_openai_api(message, model=MODEL, reasoning_effort=REASONING_EFFORT):
    """
    Call the OpenAI API directly with a message using the specified model and reasoning effort.
    
    Args:
        message: The message to send to the model
        model: The model to use (default: o1)
        reasoning_effort: Reasoning effort level (low, medium, high)
        
    Returns:
        The response from the OpenAI API
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    if not client.api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    # Create the messages list
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant with expertise in software architecture and system design. Analyze the provided system design critique and provide further simplification recommendations."},
        {"role": "user", "content": message}
    ]
    
    # Prepare API parameters
    params = {
        "model": model,
        "messages": messages,
        "reasoning_effort": reasoning_effort
    }
    
    print(f"Calling OpenAI API with model={model}, reasoning_effort={reasoning_effort}")
    
    try:
        start_time = time.time()
        response = client.chat.completions.create(**params)
        elapsed_time = time.time() - start_time
        
        result = {
            "content": response.choices[0].message.content,
            "model": model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "elapsed_time": elapsed_time,
            "finish_reason": response.choices[0].finish_reason
        }
        
        return result
    except Exception as e:
        print(f"Error calling OpenAI API: {str(e)}")
        return None

def main():
    if len(sys.argv) > 1:
        # Check if the argument is a file path
        if sys.argv[1].startswith("@"):
            # Read message from file
            file_path = sys.argv[1][1:]  # Remove the @ prefix
            try:
                with open(file_path, 'r') as file:
                    message = file.read().strip()
                print(f"Reading message from file: {file_path}")
            except Exception as e:
                print(f"Error reading file: {e}")
                sys.exit(1)
        else:
            # Read message from command line argument
            message = sys.argv[1]
    else:
        # Read message from stdin
        print("Enter your message (Ctrl+D to finish):")
        message = sys.stdin.read().strip()
    
    if not message:
        print("Error: No message provided")
        sys.exit(1)
    
    response = call_openai_api(message)
    
    if response:
        print("\n=== RESPONSE ===")
        print(f"Model: {response.get('model')}")
        print(f"Elapsed Time: {response.get('elapsed_time'):.2f} seconds")
        print(f"Token Usage: {response.get('usage')}")
        print(f"Finish Reason: {response.get('finish_reason')}")
        print("\n=== CONTENT ===")
        print(response.get('content', 'No content received'))
    else:
        print("No response received from OpenAI API")

if __name__ == "__main__":
    main() 