"""
A2A NVIDIA Integration - Main Module

This module provides the main functionality for integrating NVIDIA's API with Google's A2A protocol.
"""

import json
import uuid
import time
from typing import Dict, List, Optional, Union, Any, Generator, Iterator

from openai import OpenAI

from a2a.core.agent_card import AgentCard
from a2a.core.task_manager import TaskManager
from a2a.core.message_handler import MessageHandler
from a2a.core.mcp.mcp_client import MCPClient
from dotenv import load_dotenv
import os
load_dotenv()
global api_key
api_key=os.environ["NVIDIA_API_KEY"]
print("a2a_nvidia.py set NVIDIA_API_KEY=", api_key[-4:])

class A2ANvidia:
    """
    Main class for A2A NVIDIA integration.
    
    This class integrates NVIDIA's API with the A2A protocol, allowing NVIDIA
    to communicate with other A2A-compatible agents.
    """
    
    def __init__(
        self,
        model: str ,
        name: str ,
        description: str ,
        skills: List[Dict[str, Any]],
        base_url: str = "https://integrate.api.nvidia.com/v1",
        api_key: str = os.environ["NVIDIA_API_KEY"],
        endpoint: str = "http://localhost:8000",
    ):
        """
        Initialize A2ANvidia.
        
        Args:
            model: The NVIDIA model to use
            name: The name of the agent
            description: A description of the agent
            skills: A list of skills the agent has
            base_url: The NVIDIA API base URL
            api_key: The NVIDIA API key
            endpoint: The endpoint where this agent is accessible
        """
        self.model = model
        self.api_key=os.environ["NVIDIA_API_KEY"]
        self.base_url=base_url
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        self.agent_card = AgentCard(
            name=name,
            description=description,
            endpoint=endpoint,
            skills=skills,
        )
        self.task_manager = TaskManager()
        self.message_handler = MessageHandler()
        self.mcp_client = None
    
    def configure_mcp_client(self, mcp_client: MCPClient) -> None:
        """
        Configure MCP client for tool access.
        
        Args:
            mcp_client: The MCP client
        """
        self.mcp_client = mcp_client
    
    def _get_nvidia_messages(self, task_id: str) -> List[Dict[str, Any]]:
        """
        Convert A2A messages to NVIDIA message format.
        
        Args:
            task_id: The task ID
            
        Returns:
            List of messages in NVIDIA format
        """
        messages = self.message_handler.get_messages(task_id)
        nvidia_messages = []
        
        for message in messages:
            content = ""
            for part in message.get("parts", []):
                if part.get("type") == "text":
                    content += part.get("content", "")
            
            nvidia_messages.append({
                "role": message.get("role", "user"),
                "content": content
            })
            
        return nvidia_messages
    
    def _process_task(self, task_id: str) -> Dict[str, Any]:
        """
        Process a task using NVIDIA API.
        
        Args:
            task_id: The ID of the task to process
            
        Returns:
            The result of processing the task
        """
        task = self.task_manager.get_task(task_id)
        
        if not task:
            return {"error": f"Task not found: {task_id}"}
        
        # Check if this is an MCP task
        if self.task_manager.mcp_bridge and self.task_manager._can_use_mcp_for_task(task):
            try:
                import asyncio
                return asyncio.run(self.task_manager.process_task(task_id))
            except Exception as e:
                print(f"Error processing MCP task: {e}")
                # Fall back to normal processing
        
        nvidia_messages = self._get_nvidia_messages(task_id)
        
        # Set up retry parameters
        max_retries = 3
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                # Add available MCP tools to the system message if MCP is configured
                if self.mcp_client and self.mcp_client.available_tools:
                    # Check if we have a system message, if not add one
                    has_system_message = False
                    for msg in nvidia_messages:
                        if msg.get("role") == "system":
                            has_system_message = True
                            # Add MCP tools to existing system message
                            msg["content"] += self._get_mcp_tools_description()
                            break
                            
                    if not has_system_message:
                        # Create a new system message with MCP tools
                        nvidia_messages.insert(0, {
                            "role": "system",
                            "content": f"You are {self.agent_card.name}, {self.agent_card.description}. {self._get_mcp_tools_description()}"
                        })
                
                # Generate a response using NVIDIA API
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=nvidia_messages,
                    temperature=0.2,
                    top_p=0.7,
                    max_tokens=8192,
                    extra_body={"chat_template_kwargs": {"thinking": True}},
                    stream=False
                )
                
                response_content = completion.choices[0].message.content
                
                # Check for MCP tool calls in the response
                tool_calls = self._extract_tool_calls(response_content)
                
                if tool_calls and self.mcp_client:
                    # Execute the tool calls and append results
                    tool_results = []
                    for tool_call in tool_calls:
                        tool_name = tool_call.get("name")
                        parameters = tool_call.get("parameters", {})
                        
                        try:
                            import asyncio
                            result = asyncio.run(self.mcp_client.execute_tool(tool_name, parameters))
                            tool_results.append({
                                "name": tool_name,
                                "result": result.result,
                                "error": result.error
                            })
                        except Exception as e:
                            tool_results.append({
                                "name": tool_name,
                                "result": None,
                                "error": str(e)
                            })
                    
                    # Add the tool results to the messages
                    nvidia_messages.append({
                        "role": "assistant",
                        "content": response_content
                    })
                    
                    # Add tool results message
                    nvidia_messages.append({
                        "role": "system",
                        "content": f"Tool results: {json.dumps(tool_results)}"
                    })
                    
                    # Generate a final response that incorporates the tool results
                    final_completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=nvidia_messages,
                        temperature=0.2,
                        top_p=0.7,
                        max_tokens=8192,
                        extra_body={"chat_template_kwargs": {"thinking": True}},
                        stream=False
                    )
                    
                    response_content = final_completion.choices[0].message.content
                
                # Update task status
                self.task_manager.update_task_status(task_id, "completed")
                
                # Create A2A message from the response
                message_id = str(uuid.uuid4())
                a2a_message = {
                    "id": message_id,
                    "role": "agent",
                    "parts": [
                        {
                            "type": "text",
                            "content": response_content
                        }
                    ]
                }
                
                # Add the message to the task
                self.message_handler.add_message(task_id, a2a_message)
                
                return {
                    "task_id": task_id,
                    "message_id": message_id,
                    "status": "completed",
                    "message": a2a_message
                }
                
            except Exception as e:
                last_error = str(e)
                retry_count += 1
                print(f"Error processing task (attempt {retry_count}): {e}")
                time.sleep(1)  # Wait before retrying
        
        # If we get here, all retries failed
        self.task_manager.update_task_status(task_id, "failed")
        
        return {
            "task_id": task_id,
            "status": "failed",
            "error": last_error
        }
    
    def _extract_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract MCP tool calls from an Ollama response.
        
        Args:
            content: The response content
            
        Returns:
            List of extracted tool calls
        """
        tool_calls = []
        
        # Simple parsing for tool calls - in reality this would need to be more robust
        # Example format to detect: {"name": "tool_name", "parameters": {"param1": "value1"}}
        import re
        
        # Look for JSON objects that might be tool calls
        json_pattern = r'\{\s*"name"\s*:\s*"([^"]*)"\s*,\s*"parameters"\s*:\s*(\{[^}]*\})\s*\}'
        matches = re.finditer(json_pattern, content)
        
        for match in matches:
            try:
                tool_name = match.group(1)
                parameters_str = match.group(2)
                parameters = json.loads(parameters_str)
                
                tool_calls.append({
                    "name": tool_name,
                    "parameters": parameters
                })
            except Exception as e:
                print(f"Error parsing tool call: {e}")
        
        return tool_calls
        
    def _get_mcp_tools_description(self) -> str:
        """
        Get a description of available MCP tools.
        
        Returns:
            Description of MCP tools
        """
        if not self.mcp_client or not self.mcp_client.available_tools:
            return ""
            
        tools_description = "You have access to the following tools:\n\n"
        
        for name, tool in self.mcp_client.available_tools.items():
            tools_description += f"- {name}: {tool.description}\n"
            
            if tool.parameters:
                tools_description += "  Parameters:\n"
                for param in tool.parameters:
                    required = " (required)" if param.required else ""
                    tools_description += f"  - {param.name}{required}: {param.description}\n"
                    
            tools_description += "\n"
            
        tools_description += "\nTo use a tool, respond with JSON in this format: {\"name\": \"tool_name\", \"parameters\": {\"param1\": \"value1\"}}\n"
        
        return tools_description
        
    def _process_task_stream(self, task_id: str) -> Iterator[Dict[str, Any]]:
        """
        Process a task using NVIDIA API and stream the response.
        
        Args:
            task_id: The ID of the task to process
            
        Yields:
            Chunks of the response as they become available
        """
        task = self.task_manager.get_task(task_id)
        
        if not task:
            yield {
                "task_id": task_id,
                "error": f"Task not found: {task_id}",
                "status": "failed",
                "done": True
            }
            return
        
        # Check if this is an MCP task
        if self.task_manager.mcp_bridge and self.task_manager._can_use_mcp_for_task(task):
            yield {
                "task_id": task_id,
                "error": "Streaming not supported for MCP tasks",
                "done": True
            }
            return
        
        nvidia_messages = self._get_nvidia_messages(task_id)
        message_id = str(uuid.uuid4())
        full_content = ""
        
        # Add available MCP tools to the system message if MCP is configured
        if self.mcp_client and self.mcp_client.available_tools:
            # Check if we have a system message, if not add one
            has_system_message = False
            for msg in nvidia_messages:
                if msg.get("role") == "system":
                    has_system_message = True
                    # Add MCP tools to existing system message
                    msg["content"] += self._get_mcp_tools_description()
                    break
                    
            if not has_system_message:
                # Create a new system message with MCP tools
                nvidia_messages.insert(0, {
                    "role": "system",
                    "content": f"You are {self.agent_card.name}, {self.agent_card.description}. {self._get_mcp_tools_description()}"
                })
        
        try:
            # Stream response from NVIDIA API
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=nvidia_messages,
                temperature=0.2,
                top_p=0.7,
                max_tokens=8192,
                extra_body={"chat_template_kwargs": {"thinking": True}},
                stream=True
            )
            
            for chunk in completion:
                reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
                if reasoning:
                    full_content += reasoning
                    yield {
                        "task_id": task_id,
                        "message_id": message_id,
                        "chunk": {
                            "type": "text",
                            "content": reasoning
                        },
                        "done": False
                    }
                
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_content += content
                    yield {
                        "task_id": task_id,
                        "message_id": message_id,
                        "chunk": {
                            "type": "text",
                            "content": content
                        },
                        "done": False
                    }
                    
        except Exception as e:
            # Handle error
            error_message = str(e)
            self.task_manager.update_task_status(task_id, "failed")
            
            yield {
                "task_id": task_id,
                "message_id": message_id,
                "error": error_message,
                "status": "failed",
                "done": True
            }
            return
        
        # Check for MCP tool calls in the response
        tool_calls = self._extract_tool_calls(full_content)
        
        if tool_calls and self.mcp_client:
            # Execute the tool calls and append results
            yield {
                "task_id": task_id,
                "message_id": message_id,
                "chunk": {
                    "type": "text",
                    "content": "\n\nExecuting tool calls..."
                },
                "done": False
            }
            
            tool_results = []
            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                parameters = tool_call.get("parameters", {})
                
                try:
                    import asyncio
                    result = asyncio.run(self.mcp_client.execute_tool(tool_name, parameters))
                    tool_results.append({
                        "name": tool_name,
                        "result": result.result,
                        "error": result.error
                    })
                    
                    # Send a chunk with the tool result
                    yield {
                        "task_id": task_id,
                        "message_id": message_id,
                        "chunk": {
                            "type": "text",
                            "content": f"\nTool '{tool_name}' result: {json.dumps(result.result)}"
                        },
                        "done": False
                    }
                except Exception as e:
                    tool_results.append({
                        "name": tool_name,
                        "result": None,
                        "error": str(e)
                    })
                    
                    # Send a chunk with the error
                    yield {
                        "task_id": task_id,
                        "message_id": message_id,
                        "chunk": {
                            "type": "text",
                            "content": f"\nError executing tool '{tool_name}': {str(e)}"
                        },
                        "done": False
                    }
            
            # Add the tool results to the messages
            nvidia_messages.append({
                "role": "assistant",
                "content": full_content
            })
            
            # Add tool results message
            nvidia_messages.append({
                "role": "system",
                "content": f"Tool results: {json.dumps(tool_results)}"
            })
            
            final_content = ""
            
            try:
                # Stream final response
                final_completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=nvidia_messages,
                    temperature=0.2,
                    top_p=0.7,
                    max_tokens=8192,
                    extra_body={"chat_template_kwargs": {"thinking": True}},
                    stream=True
                )
                
                for chunk in final_completion:
                    reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
                    if reasoning:
                        final_content += reasoning
                        yield {
                            "task_id": task_id,
                            "message_id": message_id,
                            "chunk": {
                                "type": "text",
                                "content": reasoning
                            },
                            "done": False
                        }
                    
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        final_content += content
                        yield {
                            "task_id": task_id,
                            "message_id": message_id,
                            "chunk": {
                                "type": "text",
                                "content": content
                            },
                            "done": False
                        }
            except Exception as e:
                # Handle error in final response
                error_message = str(e)
                yield {
                    "task_id": task_id,
                    "message_id": message_id,
                    "chunk": {
                        "type": "text",
                        "content": f"\n\nError generating final response: {error_message}"
                    },
                    "done": False
                }
                
            # Update the full content to include the final response
            full_content += "\n\n" + final_content
        
        # Create the full A2A message
        a2a_message = {
            "id": message_id,
            "role": "agent",
            "parts": [
                {
                    "type": "text",
                    "content": full_content
                }
            ]
        }
        
        # Store the complete message
        self.message_handler.add_message(task_id, a2a_message)
        
        # Update task status
        self.task_manager.update_task_status(task_id, "completed")
        
        # Send final done message
        yield {
            "task_id": task_id,
            "message_id": message_id,
            "status": "completed",
            "done": True
        }
    
    def _get_mcp_tools_description(self) -> str:
        """
        Get a description of available MCP tools.
        
        Returns:
            Description of MCP tools
        """
        if not self.mcp_client or not self.mcp_client.available_tools:
            return ""
            
        tools_description = "You have access to the following tools:\n\n"
        
        for name, tool in self.mcp_client.available_tools.items():
            tools_description += f"- {name}: {tool.description}\n"
            
            if tool.parameters:
                tools_description += "  Parameters:\n"
                for param in tool.parameters:
                    required = " (required)" if param.required else ""
                    tools_description += f"  - {param.name}{required}: {param.description}\n"
                    
            tools_description += "\n"
            
        tools_description += "\nTo use a tool, respond with JSON in this format: {\"name\": \"tool_name\", \"parameters\": {\"param1\": \"value1\"}}\n"
        
        return tools_description
    
    def _extract_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from the response content.
        
        Args:
            content: The response content
            
        Returns:
            List of tool calls
        """
        try:
            # Look for JSON in the content
            start_idx = content.find("{")
            end_idx = content.rfind("}")
            
            if start_idx == -1 or end_idx == -1:
                return []
                
            json_str = content[start_idx:end_idx + 1]
            tool_call = json.loads(json_str)
            
            if isinstance(tool_call, dict) and "name" in tool_call and "parameters" in tool_call:
                return [tool_call]
                
            return []
        except:
            return [] 