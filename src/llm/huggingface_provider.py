"""HuggingFace Transformers LLM provider implementation.

Supports running open source models locally or via HuggingFace Hub.
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


class HuggingFaceProvider(LLMProvider):
    """LLM provider for HuggingFace Transformers models.
    
    Supports local models and HuggingFace Hub models including:
    - Llama 3
    - Mistral
    - Mixtral
    - And other instruction-tuned models
    
    Note: Tool calling support varies by model. Models fine-tuned
    for function calling (e.g., Llama 3 with tools) work best.
    """
    
    def __init__(
        self,
        model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        token: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        device: str | None = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """Initialize the HuggingFace provider.
        
        Args:
            model: Model name or path (HuggingFace Hub or local).
            token: HuggingFace token for gated models.
            temperature: Default temperature.
            max_tokens: Default max tokens.
            device: Device to use ('cuda', 'cpu', 'auto').
            load_in_8bit: Enable 8-bit quantization.
            load_in_4bit: Enable 4-bit quantization.
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError:
            raise ImportError(
                "transformers and torch required. Install with: "
                "pip install transformers torch accelerate"
            )
        
        self._model_name = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._token = token
        
        # Determine device
        if device is None:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self._device = device
        
        # Model loading configuration
        model_kwargs = {
            "token": token,
            "device_map": "auto" if device != "cpu" else None,
        }
        
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        
        # Lazy loading - model loads on first use
        self._pipeline = None
        self._model_kwargs = model_kwargs
        self._tokenizer = None
    
    def _ensure_loaded(self) -> None:
        """Ensure the model is loaded."""
        if self._pipeline is not None:
            return
        
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            token=self._token,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            **self._model_kwargs,
        )
        
        self._pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=self._tokenizer,
        )
    
    @property
    def provider_name(self) -> str:
        return "huggingface"
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    def _format_messages(self, messages: list[LLMMessage]) -> str:
        """Format messages for the model.
        
        Uses the tokenizer's chat template if available,
        otherwise falls back to a generic format.
        
        Args:
            messages: List of LLMMessage objects.
            
        Returns:
            Formatted prompt string.
        """
        self._ensure_loaded()
        
        # Convert to list of dicts
        chat_messages = []
        for msg in messages:
            chat_messages.append({
                "role": msg.role if msg.role != "tool" else "user",
                "content": msg.content,
            })
        
        # Try to use chat template
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        
        # Fallback format
        formatted = ""
        for msg in chat_messages:
            role = msg["role"].upper()
            formatted += f"{role}: {msg['content']}\n\n"
        formatted += "ASSISTANT: "
        
        return formatted
    
    def _format_tools_prompt(self, tools: list[ToolDefinition]) -> str:
        """Format tools for inclusion in prompt.
        
        Since not all models support native tool calling,
        we format tools as instructions in the prompt.
        
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
        )
    
    def _parse_tool_calls(self, response_text: str) -> list[dict[str, Any]]:
        """Try to parse tool calls from response text.
        
        Args:
            response_text: Raw model response.
            
        Returns:
            List of parsed tool calls (may be empty).
        """
        tool_calls = []
        
        # Try to find JSON in response
        try:
            # Look for JSON blocks
            import re
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
    
    async def generate(
        self,
        messages: list[LLMMessage],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a response asynchronously.
        
        Note: This runs synchronously since transformers doesn't have
        native async support. For true async, consider using a thread pool.
        
        Args:
            messages: Conversation history.
            tools: Optional tool definitions.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.
            
        Returns:
            LLMResponse with content and optional tool calls.
        """
        import asyncio
        # Run sync version in executor for async compatibility
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
        self._ensure_loaded()
        
        # Format prompt
        prompt = self._format_messages(messages)
        
        # Add tools to prompt if provided
        if tools:
            # Insert tools description before the last message
            tools_prompt = self._format_tools_prompt(tools)
            prompt = prompt.rstrip() + tools_prompt + "\n"
        
        # Generate
        temp = temperature if temperature is not None else self._temperature
        max_new = max_tokens if max_tokens is not None else self._max_tokens
        
        # Build generation kwargs for stability
        gen_kwargs = {
            "max_new_tokens": min(max_new, 1024),  # Cap to avoid exceeding model limits
            "return_full_text": False,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        
        if temp > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temp
            gen_kwargs["top_p"] = 0.9  # Nucleus sampling for stability
            gen_kwargs["top_k"] = 50   # Limit vocabulary for stability
        else:
            gen_kwargs["do_sample"] = False
        
        outputs = self._pipeline(prompt, **gen_kwargs)
        
        response_text = outputs[0]["generated_text"]
        
        # Try to parse tool calls if tools were provided
        tool_calls = []
        if tools:
            tool_calls = self._parse_tool_calls(response_text)
        
        # Estimate tokens (rough approximation)
        prompt_tokens = len(self._tokenizer.encode(prompt))
        completion_tokens = len(self._tokenizer.encode(response_text))
        
        return LLMResponse(
            content=response_text,
            tool_calls=tool_calls,
            finish_reason="stop",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=self._model_name,
            raw_response=outputs,
        )
    
    def supports_tools(self) -> bool:
        # Tool support is prompt-based, not native
        return True
    
    def supports_streaming(self) -> bool:
        return False
