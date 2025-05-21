"""
Multi-Agent Example - Knowledge Agent

This agent provides factual information and research.
"""

import os
import sys
import argparse

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

#from a2a.server import run_server
from a2a.server import A2AServer


def main():
    """Run the knowledge agent server."""
    parser = argparse.ArgumentParser(description="Run Knowledge Agent")
    parser.add_argument("--model", type=str, default="gemma3:27b", help="The Ollama model to use")
    parser.add_argument("--port", type=int, default=8003, help="The port to run the server on")
    parser.add_argument("--ollama-host", type=str, default="http://localhost:11434", help="The Ollama host URL")
    parser.add_argument("--nim", type=bool, default="True", help="Using NVIDIA NIM LLM")
    args = parser.parse_args()
    if args.nim:
        model="qwen/qwen3-235b-a22b"
        
    else:
        model=args.model
    name="Research Assistant"
    description="A research assistant that is able to do do research on its own and provide factual information on various topics"
    # Define the agent's skills
    skills = [
        {
            "id": "research",
            "name": "Research",
            "description": "Provides factual information on various topics"
        },
        {
            "id": "fact_check",
            "name": "Fact Checking",
            "description": "Verifies claims against known facts"
        }
    ]
    
    # Create a system prompt to guide the model behavior
    system_prompt = """
    You are a specialized Knowledge Agent that focuses on providing factual information.
    Your responses should be:
    - Based on factual information
    - Well-structured with clear sections
    - Comprehensive yet concise
    - Focused on verifiable data and statistics
    - Neutral in tone

    As a Knowledge Agent, your goal is to provide accurate information without speculation or opinion.
    """
    myserver=A2AServer(model=model,name=name, description=description, skills=skills, port=args.port)
    # Start the A2A server with the Knowledge Agent
    myserver._run_server()      
    


if __name__ == "__main__":
    main() 