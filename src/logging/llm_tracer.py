"""LLM call tracing for debugging and cost tracking.

Logs all LLM interactions to a separate file to avoid
polluting the main logs while enabling detailed debugging.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class LLMCallRecord:
    """Record of a single LLM call."""
    timestamp: str
    agent: str
    provider: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    duration_ms: int | None = None
    prompt: str | None = None  # Only stored at DEBUG level
    response: str | None = None  # Only stored at DEBUG level
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    cost_usd: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMTracer:
    """Traces LLM calls to a separate JSONL file.
    
    Provides detailed LLM interaction logging without cluttering
    the main application logs. Useful for debugging prompt issues
    and tracking API costs.
    """
    
    # Approximate cost per 1K tokens (as of 2024, update as needed)
    TOKEN_COSTS = {
        # Anthropic
        "claude-sonnet-4-6": {"input": 0.003, "output": 0.015},
        "claude-opus-4": {"input": 0.015, "output": 0.075},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        # OpenAI
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        # Open source (free when self-hosted)
        "llama": {"input": 0.0, "output": 0.0},
        "mistral": {"input": 0.0, "output": 0.0},
    }
    
    def __init__(
        self,
        output_dir: str | Path,
        include_prompts: bool = False,
        include_responses: bool = False,
    ):
        """Initialize the LLM tracer.
        
        Args:
            output_dir: Directory for trace file.
            include_prompts: Whether to log full prompts (verbose).
            include_responses: Whether to log full responses (verbose).
        """
        self.output_dir = Path(output_dir)
        self.include_prompts = include_prompts
        self.include_responses = include_responses
        
        # Ensure directory exists
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self.trace_path = log_dir / "llm_traces.jsonl"
        
        # Running totals for cost tracking
        self._total_tokens = 0
        self._total_cost = 0.0
        self._call_count = 0
    
    def _estimate_cost(
        self,
        model: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> float | None:
        """Estimate cost for an LLM call.
        
        Args:
            model: Model name.
            prompt_tokens: Number of input tokens.
            completion_tokens: Number of output tokens.
            
        Returns:
            Estimated cost in USD, or None if unknown.
        """
        if prompt_tokens is None or completion_tokens is None:
            return None
        
        # Find matching cost entry
        costs = None
        for model_key, model_costs in self.TOKEN_COSTS.items():
            if model_key in model.lower():
                costs = model_costs
                break
        
        if costs is None:
            return None
        
        input_cost = (prompt_tokens / 1000) * costs["input"]
        output_cost = (completion_tokens / 1000) * costs["output"]
        
        return input_cost + output_cost
    
    def trace(
        self,
        agent: str,
        provider: str,
        model: str,
        prompt: str | None = None,
        response: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        duration_ms: int | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LLMCallRecord:
        """Record an LLM call.
        
        Args:
            agent: Agent that made the call.
            provider: LLM provider (anthropic, openai, etc.).
            model: Model name.
            prompt: Full prompt (stored only if include_prompts=True).
            response: Full response (stored only if include_responses=True).
            prompt_tokens: Number of input tokens.
            completion_tokens: Number of output tokens.
            duration_ms: Call duration in milliseconds.
            tool_calls: List of tool calls made.
            error: Error message if call failed.
            metadata: Additional metadata.
            
        Returns:
            The recorded LLMCallRecord.
        """
        total_tokens = None
        if prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens
            self._total_tokens += total_tokens
        
        cost = self._estimate_cost(model, prompt_tokens, completion_tokens)
        if cost is not None:
            self._total_cost += cost
        
        self._call_count += 1
        
        record = LLMCallRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent=agent,
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            duration_ms=duration_ms,
            prompt=prompt if self.include_prompts else None,
            response=response if self.include_responses else None,
            tool_calls=tool_calls or [],
            error=error,
            cost_usd=cost,
            metadata=metadata or {},
        )
        
        # Write to trace file
        self._write_record(record)
        
        return record
    
    def _write_record(self, record: LLMCallRecord) -> None:
        """Write a record to the trace file."""
        record_dict = asdict(record)
        
        # Remove None values for cleaner output
        record_dict = {k: v for k, v in record_dict.items() if v is not None}
        
        with open(self.trace_path, "a") as f:
            f.write(json.dumps(record_dict, default=str) + "\n")
    
    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all traced calls.
        
        Returns:
            Dictionary with call count, total tokens, and estimated cost.
        """
        return {
            "call_count": self._call_count,
            "total_tokens": self._total_tokens,
            "estimated_cost_usd": round(self._total_cost, 4),
        }
    
    def log_summary(self) -> None:
        """Write a summary to the trace file."""
        summary = {
            "type": "summary",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **self.get_summary(),
        }
        
        with open(self.trace_path, "a") as f:
            f.write(json.dumps(summary) + "\n")


# Global tracer instance
_tracer: LLMTracer | None = None


def get_tracer(
    output_dir: str | Path | None = None,
    include_prompts: bool = False,
    include_responses: bool = False,
) -> LLMTracer:
    """Get or create the global LLM tracer.
    
    Args:
        output_dir: Output directory (required for first call).
        include_prompts: Whether to log full prompts.
        include_responses: Whether to log full responses.
        
    Returns:
        LLMTracer instance.
    """
    global _tracer
    
    if _tracer is None:
        if output_dir is None:
            raise ValueError("output_dir required for first tracer initialization")
        _tracer = LLMTracer(
            output_dir=output_dir,
            include_prompts=include_prompts,
            include_responses=include_responses,
        )
    
    return _tracer


def reset_tracer() -> None:
    """Reset the global tracer (for testing)."""
    global _tracer
    _tracer = None
