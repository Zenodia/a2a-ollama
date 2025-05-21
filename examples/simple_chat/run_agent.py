"""
Simple Chat Example - Agent Server

This example demonstrates how to run a single A2A agent.
"""

import os
import sys
import argparse

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from a2a.server import A2AServer


def main():
    """Run a simple A2A agent server."""
    parser = argparse.ArgumentParser(description="Run an A2A Agent")
    parser.add_argument("--model", type=str, default="gemma3:27b", help="The Ollama model to use")
    parser.add_argument("--port", type=int, default=8000, help="The port to run the server on")
    parser.add_argument("--ollama-host", type=str, default="http://localhost:11434", help="The Ollama host URL")
    parser.add_argument("--nim", type=bool, default="True", help="Using NVIDIA NIM LLM")
    args = parser.parse_args()
    if args.nim:
        model="qwen/qwen3-235b-a22b"
        description="A simple A2A-compatible agent powered by NVIDIA NIM LLM"
    else:
        model=args.model
        description="A simple A2A-compatible agent powered by Ollama"

    # Define the agent's skills
    skills = [
        {
            "id": "answer_questions",
            "name": "Answer Questions",
            "description": "Can answer general knowledge questions"
        },
        {
            "id": "summarize_text",
            "name": "Summarize Text",
            "description": "Can summarize text content"
        }
    ]
    myserver=A2AServer(model=model,name="Simple A2A Agent", description=description, skills=skills, port=args.port)
    
    # Start the A2A server
    myserver._run_server()      
    


if __name__ == "__main__":
    main() 