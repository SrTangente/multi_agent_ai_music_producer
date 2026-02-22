"""Anthropic (Claude) LLM provider implementation."""

from __future__ import annotations

import asyncio
from typing import Any

from src.llm.base import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    ToolDefinition,
    tools_to_anthropic_format,
)


class AnthropicProvider(LLMProvider):
    """LLM provider for Anthropic's Claude models.
    
    Supports Claude 3 family (Opus, Sonnet, Haiku) with full
    tool calling capabilities.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        """Initialize the Anthropic provider.
        
        Args:
            api_key: Anthropic API key.
            model: Model name (e.g., 'claude-sonnet-4-6').
            temperature: Default temperature.
            max_tokens: Default max tokens.
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            )
        
        self._client = anthropic.Anthropic(api_key=api_key)
        self._async_client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
    
    @property
    def provider_name(self) -> str:
        return "anthropic"
    
    @property
    def model_name(self) -> str:
        return self._model
    
    def _convert_messages(
        self,
        messages: list[LLMMessage],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert messages to Anthropic format.
        
        Anthropic requires system message to be separate.
        
        Args:
            messages: List of LLMMessage objects.
            
        Returns:
            Tuple of (system_message, conversation_messages).
        """
        system_message = None
        conversation = []
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            elif msg.role == "tool":
                # Convert tool result to Anthropic format
                conversation.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.content,
                        }
                    ]
                })
            elif msg.role == "assistant" and msg.tool_calls:
                # Assistant message with tool calls
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": tc["function"]["arguments"],
                    })
                conversation.append({"role": "assistant", "content": content})
            else:
                conversation.append({
                    "role": msg.role,
                    "content": msg.content,
                })
        
        return system_message, conversation
    
    def _parse_response(self, response) -> LLMResponse:
        """Parse Anthropic response to LLMResponse.
        
        Args:
            response: Raw Anthropic API response.
            
        Returns:
            Parsed LLMResponse.
        """
        content_parts = []
        tool_calls = []
        
        for block in response.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": block.input,
                    }
                })
        
        return LLMResponse(
            content="\n".join(content_parts),
            tool_calls=tool_calls,
            finish_reason=response.stop_reason,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
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
        system_message, conversation = self._convert_messages(messages)
        
        kwargs = {
            "model": self._model,
            "messages": conversation,
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
        }
        
        if system_message:
            kwargs["system"] = system_message
        
        if tools:
            kwargs["tools"] = tools_to_anthropic_format(tools)
        
        response = await self._async_client.messages.create(**kwargs)
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
        system_message, conversation = self._convert_messages(messages)
        
        kwargs = {
            "model": self._model,
            "messages": conversation,
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
        }
        
        if system_message:
            kwargs["system"] = system_message
        
        if tools:
            kwargs["tools"] = tools_to_anthropic_format(tools)
        
        response = self._client.messages.create(**kwargs)
        return self._parse_response(response)
    
    def supports_streaming(self) -> bool:
        return True
