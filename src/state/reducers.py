"""State reducers and state creation utilities.

Reducers define how state updates are merged in LangGraph.
The main reducers (list appends) are defined via Annotated types in schemas.py.
This module provides helper functions for state creation and updates.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.state.schemas import (
    ErrorEntry,
    LLMConfigState,
    LogEntry,
    MusicProducerState,
    SegmentState,
    ToolError,
)


def create_run_id() -> str:
    """Generate a unique run ID with timestamp prefix."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"{timestamp}_{short_uuid}"


def create_initial_state(
    user_prompt: str,
    reference_paths: list[str],
    output_dir: str | Path,
    llm_provider: str = "anthropic",
    llm_model: str = "claude-sonnet-4-6",
    llm_temperature: float = 0.7,
    llm_max_tokens: int = 4096,
    llm_base_url: str | None = None,
    max_retries: int = 3,
    run_id: str | None = None,
) -> MusicProducerState:
    """Create the initial state for a new music production run.
    
    Args:
        user_prompt: The user's text prompt describing desired music.
        reference_paths: List of paths to reference audio files.
        output_dir: Directory for output files.
        llm_provider: LLM provider to use.
        llm_model: LLM model name.
        llm_temperature: LLM temperature setting.
        llm_max_tokens: Maximum tokens for LLM responses.
        llm_base_url: Optional base URL for LLM API.
        max_retries: Maximum retries per segment.
        run_id: Optional run ID (generated if not provided).
        
    Returns:
        Initialized MusicProducerState.
    """
    if run_id is None:
        run_id = create_run_id()
    
    output_dir = str(output_dir)
    
    # Ensure output directory structure exists
    run_output_dir = Path(output_dir) / run_id
    (run_output_dir / "segments").mkdir(parents=True, exist_ok=True)
    (run_output_dir / "logs").mkdir(parents=True, exist_ok=True)
    
    llm_config: LLMConfigState = {
        "provider": llm_provider,  # type: ignore
        "model_name": llm_model,
        "temperature": llm_temperature,
        "max_tokens": llm_max_tokens,
        "base_url": llm_base_url,
    }
    
    initial_log = create_log_entry(
        action="run_initialized",
        message=f"Starting music production run {run_id}",
        metadata={
            "user_prompt": user_prompt,
            "reference_count": len(reference_paths),
            "output_dir": output_dir,
        }
    )
    
    return {
        # Run ID
        "run_id": run_id,
        
        # Input
        "user_prompt": user_prompt,
        "reference_paths": reference_paths,
        "output_dir": str(run_output_dir),
        
        # Configuration
        "llm_config": llm_config,
        "max_retries": max_retries,
        
        # Analysis (to be filled)
        "musical_profile": None,
        
        # Director (to be filled)
        "track_plan": None,
        
        # Segment loop
        "current_segment_index": 0,
        "segments": [],
        "retry_count": 0,
        
        # Accumulated track
        "accumulated_audio_path": None,
        "approved_segment_paths": [],
        
        # Control flow
        "phase": "initialized",
        
        # Observability
        "logs": [initial_log],
        "errors": [],
        
        # Final output (to be filled)
        "final_track_path": None,
        "final_track_duration_sec": None,
        
        # Checkpointing
        "last_checkpoint_path": None,
    }


def create_log_entry(
    action: str,
    message: str,
    level: str = "INFO",
    agent: str | None = None,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    duration_ms: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> LogEntry:
    """Create a structured log entry.
    
    Args:
        action: Action being performed.
        message: Human-readable message.
        level: Log level (INFO, WARNING, ERROR, DEBUG).
        agent: Agent performing the action.
        inputs: Input data for the action.
        outputs: Output data from the action.
        duration_ms: Duration in milliseconds.
        metadata: Additional metadata.
        
    Returns:
        LogEntry dict.
    """
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,  # type: ignore
        "agent": agent,
        "action": action,
        "message": message,
        "inputs": inputs,
        "outputs": outputs,
        "duration_ms": duration_ms,
        "metadata": metadata,
    }


def create_error_entry(
    action: str,
    error: ToolError,
    agent: str | None = None,
    context: dict[str, Any] | None = None,
) -> ErrorEntry:
    """Create a structured error entry.
    
    Args:
        action: Action that failed.
        error: The ToolError that occurred.
        agent: Agent where error occurred.
        context: Additional context about the error.
        
    Returns:
        ErrorEntry dict.
    """
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": agent,
        "action": action,
        "error": error,
        "context": context or {},
        "recoverable": error["recoverable"],
    }


def create_tool_error(
    code: str,
    message: str,
    recoverable: bool = True,
    suggested_action: str | None = None,
) -> ToolError:
    """Create a structured tool error.
    
    Args:
        code: Error code (e.g., "MUSICGEN_OOM").
        message: Human-readable error message.
        recoverable: Whether the operation can be retried.
        suggested_action: Suggested action to resolve.
        
    Returns:
        ToolError dict.
    """
    return {
        "code": code,
        "message": message,
        "recoverable": recoverable,
        "suggested_action": suggested_action,
    }


def create_segment_state(
    segment_id: str,
    parameters: dict[str, Any],
    conditioning_audio_path: str | None = None,
) -> SegmentState:
    """Create initial state for a segment.
    
    Args:
        segment_id: Unique segment identifier.
        parameters: SegmentParameters dict.
        conditioning_audio_path: Path to audio for conditioning.
        
    Returns:
        SegmentState dict.
    """
    return {
        "segment_id": segment_id,
        "parameters": parameters,  # type: ignore
        "status": "pending",
        "attempts": [],
        "current_attempt": 0,
        "best_attempt_index": None,
        "final_audio_path": None,
        "conditioning_audio_path": conditioning_audio_path,
    }


def update_segment_status(
    segment: SegmentState,
    status: str,
) -> SegmentState:
    """Update segment status immutably.
    
    Args:
        segment: Current segment state.
        status: New status.
        
    Returns:
        New SegmentState with updated status.
    """
    return {**segment, "status": status}  # type: ignore


def select_best_attempt(segment: SegmentState) -> int | None:
    """Select the best attempt from a segment's attempt history.
    
    Uses overall_score from critic feedback, or falls back to
    the last attempt if no scores available.
    
    Args:
        segment: Segment state with attempts.
        
    Returns:
        Index of best attempt, or None if no attempts.
    """
    if not segment["attempts"]:
        return None
    
    best_index = 0
    best_score = -1.0
    
    for i, attempt in enumerate(segment["attempts"]):
        feedback = attempt.get("critic_feedback")
        if feedback:
            score = feedback.get("overall_score", 0.0)
            if score > best_score:
                best_score = score
                best_index = i
    
    # If no feedback scores, use last attempt
    if best_score < 0:
        best_index = len(segment["attempts"]) - 1
    
    return best_index
