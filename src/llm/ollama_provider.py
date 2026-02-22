"""Ollama LLM provider implementation.

Supports running local models via Ollama server.
"""

from __future__ import annotations

import json
from typing import Any

from src.llm.base import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    ToolDefinition,
)


class OllamaProvider(LLMProvider):
    """LLM provider for Ollama local models.
    
    Ollama allows running models like Llama, Mistral, Mixtral,
    and others locally with a simple API.
    
    Requires Ollama to be installed and running:
    https://ollama.ai
    """
    
    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        """Initialize the Ollama provider.
        
        Args:
            model: Model name (e.g., 'llama3', 'mistral', 'mixtral').
            base_url: Ollama server URL.
            temperature: Default temperature.
            max_tokens: Default max tokens.
        """
        self._model_name = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens
        
        # Check if ollama package is available
        self._use_package = False
        try:
            import ollama
            self._use_package = True
            self._client = ollama.Client(host=base_url)
        except ImportError:
            # Fall back to requests
            try:
                import requests
                self._requests = requests
            except ImportError:
                raise ImportError(
                    "Either 'ollama' or 'requests' package required. "
                    "Install with: pip install ollama"
                )
    
    @property
    def provider_name(self) -> str:
        return "ollama"
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    def _convert_messages(self, messages: list[LLMMessage]) -> list[dict[str, Any]]:
        """Convert messages to Ollama format.
        
        Args:
            messages: List of LLMMessage objects.
            
        Returns:
            List of message dictionaries.
        """
        result = []
        
        for msg in messages:
            if msg.role == "tool":
                # Ollama doesn't have native tool responses, embed in user message
                result.append({
                    "role": "user",
                    "content": f"Tool result: {msg.content}",
                })
            else:
                result.append({
                    "role": msg.role,
                    "content": msg.content,
                })
        
        return result
    
    def _format_tools_prompt(self, tools: list[ToolDefinition]) -> str:
        """Format tools for inclusion in system prompt.
        
        Ollama doesn't have native tool support, so we inject
        tool descriptions into the system prompt.
        
        Args:
            tools: List of ToolDefinition objects.
            
        Returns:
            Formatted tools description.
        """
        if not tools:
            return ""
        
        tool_descriptions = []
        for tool in tools:
            params_str = json.dumps(tool.parameters, indent=2)
            tool_descriptions.append(
                f"- {tool.name}: {tool.description}\n  Parameters: {params_str}"
            )
        
        return (
            "\n\nYou have access to the following tools:\n"
            + "\n".join(tool_descriptions)
            + "\n\nTo use a tool, respond with JSON in this format:\n"
            '{"tool": "tool_name", "arguments": {...}}\n'
            "Only use JSON format when calling a tool."
        )
    
    def _parse_tool_calls(self, response_text: str) -> list[dict[str, Any]]:
        """Try to parse tool calls from response text.
        
        Args:
            response_text: Raw model response.
            
        Returns:
            List of parsed tool calls (may be empty).
        """
        tool_calls = []
        
        try:
            import re
            # Look for JSON objects that look like tool calls
            json_matches = re.findall(r'\{[^{}]+\}', response_text)
            
            for match in json_matches:
                try:
                    parsed = json.loads(match)
                    if "tool" in parsed and "arguments" in parsed:
                        tool_calls.append({
                            "id": f"call_{len(tool_calls)}",
                            "type": "function",
                            "function": {
                                "name": parsed["tool"],
                                "arguments": parsed["arguments"],
                            }
                        })
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass
        
        return tool_calls
    
    def _call_api_requests(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Make API call using requests library.
        
        Args:
            messages: Formatted messages.
            temperature: Temperature setting.
            max_tokens: Max tokens setting.
            
        Returns:
            Raw API response.
        """
        response = self._requests.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self._model_name,
                "messages": messages,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()
    
    async def generate(
        self,
        messages: list[LLMMessage],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a response asynchronously.
        
        Args:
            messages: Conversation history.
            tools: Optional tool definitions.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.
            
        Returns:
            LLMResponse with content and optional tool calls.
        """
        import asyncio
        # Ollama doesn't have native async, run in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate_sync(messages, tools, temperature, max_tokens)
        )
    
    def generate_sync(
        self,
        messages: list[LLMMessage],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a response synchronously.
        
        Args:
            messages: Conversation history.
            tools: Optional tool definitions.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.
            
        Returns:
            LLMResponse with content and optional tool calls.
        """
        temp = temperature if temperature is not None else self._temperature
        max_tok = max_tokens if max_tokens is not None else self._max_tokens
        
        # Convert messages
        ollama_messages = self._convert_messages(messages)
        
        # Inject tools into system message if provided
        if tools:
            tools_prompt = self._format_tools_prompt(tools)
            # Find or create system message
            has_system = any(m["role"] == "system" for m in ollama_messages)
            if has_system:
                for m in ollama_messages:
                    if m["role"] == "system":
                        m["content"] += tools_prompt
                        break
            else:
                ollama_messages.insert(0, {
                    "role": "system",
                    "content": f"You are a helpful assistant.{tools_prompt}",
                })
        
        # Make API call
        if self._use_package:
            response = self._client.chat(
                model=self._model_name,
                messages=ollama_messages,
                options={
                    "temperature": temp,
                    "num_predict": max_tok,
                },
            )
            content = response["message"]["content"]
            raw_response = response
        else:
            response = self._call_api_requests(ollama_messages, temp, max_tok)
            content = response["message"]["content"]
            raw_response = response
        
        # Parse tool calls if tools were provided
        tool_calls = []
        if tools:
            tool_calls = self._parse_tool_calls(content)
        
        # Ollama doesn't provide token counts in all versions
        prompt_tokens = raw_response.get("prompt_eval_count")
        completion_tokens = raw_response.get("eval_count")
        
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason="stop",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=self._model_name,
            raw_response=raw_response,
        )
    
    def supports_tools(self) -> bool:
        # Tool support is prompt-based, not native
        return True
    
    def supports_streaming(self) -> bool:
        return False
    
    def is_available(self) -> bool:
        """Check if Ollama server is available.
        
        Returns:
            True if server responds.
        """
        try:
            if self._use_package:
                self._client.list()
                return True
            else:
                response = self._requests.get(
                    f"{self._base_url}/api/tags",
                    timeout=5,
                )
                return response.status_code == 200
        except Exception:
            return False
    
    def list_models(self) -> list[str]:
        """List available models on the Ollama server.
        
        Returns:
            List of model names.
        """
        try:
            if self._use_package:
                response = self._client.list()
                return [m["name"] for m in response.get("models", [])]
            else:
                response = self._requests.get(
                    f"{self._base_url}/api/tags",
                    timeout=5,
                )
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []
