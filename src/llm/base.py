"""Base LLM provider protocol and factory.

Defines the interface that all LLM providers must implement,
enabling seamless switching between Claude, GPT, and open source models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass
class LLMMessage:
    """A message in a conversation."""
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    model: str | None = None
    raw_response: Any = None


@dataclass
class ToolDefinition:
    """Definition of a tool for the LLM."""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for parameters


class LLMProvider(ABC):
    """Abstract base class for LLM providers.
    
    All LLM providers (Anthropic, OpenAI, HuggingFace, Ollama) must
    implement this interface to ensure consistent behavior across
    the application.
    """
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the provider name (e.g., 'anthropic', 'openai')."""
        ...
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the current model name."""
        ...
    
    @abstractmethod
    async def generate(
        self,
        messages: list[LLMMessage],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a response from the LLM.
        
        Args:
            messages: Conversation history.
            tools: Optional tool definitions for function calling.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.
            
        Returns:
            LLMResponse with content and optional tool calls.
        """
        ...
    
    @abstractmethod
    def generate_sync(
        self,
        messages: list[LLMMessage],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Synchronous version of generate.
        
        Args:
            messages: Conversation history.
            tools: Optional tool definitions for function calling.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.
            
        Returns:
            LLMResponse with content and optional tool calls.
        """
        ...
    
    def supports_tools(self) -> bool:
        """Check if this provider supports tool/function calling.
        
        Returns:
            True if tools are supported.
        """
        return True  # Most modern LLMs support tools
    
    def supports_streaming(self) -> bool:
        """Check if this provider supports streaming responses.
        
        Returns:
            True if streaming is supported.
        """
        return False  # Override in providers that support it


def create_llm_provider(
    provider: Literal["anthropic", "openai", "huggingface", "ollama"],
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    **kwargs,
) -> LLMProvider:
    """Factory function to create an LLM provider.
    
    Args:
        provider: Provider type.
        model: Model name.
        api_key: API key (required for cloud providers).
        base_url: Optional base URL for API.
        temperature: Default temperature.
        max_tokens: Default max tokens.
        **kwargs: Additional provider-specific arguments.
        
    Returns:
        Configured LLMProvider instance.
        
    Raises:
        ValueError: If provider is unknown or missing required config.
        ImportError: If provider dependencies are not installed.
    """
    if provider == "anthropic":
        from src.llm.anthropic_provider import AnthropicProvider
        if not api_key:
            raise ValueError("api_key required for Anthropic provider")
        return AnthropicProvider(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
    
    elif provider == "openai":
        from src.llm.openai_provider import OpenAIProvider
        if not api_key:
            raise ValueError("api_key required for OpenAI provider")
        return OpenAIProvider(
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
    
    elif provider == "huggingface":
        from src.llm.huggingface_provider import HuggingFaceProvider
        return HuggingFaceProvider(
            model=model,
            token=api_key,  # HF uses 'token' not 'api_key'
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
    
    elif provider == "ollama":
        from src.llm.ollama_provider import OllamaProvider
        return OllamaProvider(
            model=model,
            base_url=base_url or "http://localhost:11434",
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
    
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def messages_to_dict(messages: list[LLMMessage]) -> list[dict[str, Any]]:
    """Convert LLMMessage list to dict format for APIs.
    
    Args:
        messages: List of LLMMessage objects.
        
    Returns:
        List of message dictionaries.
    """
    result = []
    for msg in messages:
        d = {"role": msg.role, "content": msg.content}
        if msg.tool_calls:
            d["tool_calls"] = msg.tool_calls
        if msg.tool_call_id:
            d["tool_call_id"] = msg.tool_call_id
        if msg.name:
            d["name"] = msg.name
        result.append(d)
    return result


def tools_to_openai_format(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Convert ToolDefinition list to OpenAI tools format.
    
    Args:
        tools: List of ToolDefinition objects.
        
    Returns:
        List in OpenAI tools format.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
        }
        for tool in tools
    ]


def tools_to_anthropic_format(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Convert ToolDefinition list to Anthropic tools format.
    
    Args:
        tools: List of ToolDefinition objects.
        
    Returns:
        List in Anthropic tools format.
    """
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }
        for tool in tools
    ]
