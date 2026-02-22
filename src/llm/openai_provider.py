"""OpenAI LLM provider implementation."""

from __future__ import annotations

import json
from typing import Any

from src.llm.base import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    ToolDefinition,
    messages_to_dict,
    tools_to_openai_format,
)


class OpenAIProvider(LLMProvider):
    """LLM provider for OpenAI's GPT models.
    
    Supports GPT-4 family with full tool calling capabilities.
    Also works with OpenAI-compatible APIs (e.g., Azure, local).
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        """Initialize the OpenAI provider.
        
        Args:
            api_key: OpenAI API key.
            model: Model name (e.g., 'gpt-4o', 'gpt-4-turbo').
            base_url: Optional base URL for API (for Azure/local).
            temperature: Default temperature.
            max_tokens: Default max tokens.
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )
        
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        
        self._client = openai.OpenAI(**client_kwargs)
        self._async_client = openai.AsyncOpenAI(**client_kwargs)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
    
    @property
    def provider_name(self) -> str:
        return "openai"
    
    @property
    def model_name(self) -> str:
        return self._model
    
    def _convert_messages(self, messages: list[LLMMessage]) -> list[dict[str, Any]]:
        """Convert messages to OpenAI format.
        
        Args:
            messages: List of LLMMessage objects.
            
        Returns:
            List of message dictionaries.
        """
        result = []
        
        for msg in messages:
            if msg.role == "tool":
                result.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                })
            elif msg.role == "assistant" and msg.tool_calls:
                # Convert tool calls to OpenAI format
                tool_calls = []
                for tc in msg.tool_calls:
                    arguments = tc["function"]["arguments"]
                    if isinstance(arguments, dict):
                        arguments = json.dumps(arguments)
                    tool_calls.append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": arguments,
                        }
                    })
                result.append({
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": tool_calls,
                })
            else:
                result.append({
                    "role": msg.role,
                    "content": msg.content,
                })
        
        return result
    
    def _parse_response(self, response) -> LLMResponse:
        """Parse OpenAI response to LLMResponse.
        
        Args:
            response: Raw OpenAI API response.
            
        Returns:
            Parsed LLMResponse.
        """
        choice = response.choices[0]
        message = choice.message
        
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                # Parse arguments back to dict
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = tc.function.arguments
                
                tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": arguments,
                    }
                })
        
        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            prompt_tokens=response.usage.prompt_tokens if response.usage else None,
            completion_tokens=response.usage.completion_tokens if response.usage else None,
            model=response.model,
            raw_response=response,
        )
    
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
        kwargs = {
            "model": self._model,
            "messages": self._convert_messages(messages),
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
        }
        
        if tools:
            kwargs["tools"] = tools_to_openai_format(tools)
        
        response = await self._async_client.chat.completions.create(**kwargs)
        return self._parse_response(response)
    
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
        kwargs = {
            "model": self._model,
            "messages": self._convert_messages(messages),
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
        }
        
        if tools:
            kwargs["tools"] = tools_to_openai_format(tools)
        
        response = self._client.chat.completions.create(**kwargs)
        return self._parse_response(response)
    
    def supports_streaming(self) -> bool:
        return True
