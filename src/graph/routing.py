"""Routing functions for conditional edges in the workflow.

These functions determine the next node based on current state.
"""

from __future__ import annotations

from typing import Literal

from src.state.schemas import MusicProducerState


def route_after_critic(
    state: MusicProducerState,
) -> Literal["segment_complete", "retry_segment"]:
    """Route after critic evaluation.
    
    If segment approved, proceed to completion.
    If rejected, route to retry.
    
    Args:
        state: Current workflow state.
        
    Returns:
        Next node name.
    """
    current_segment = state.get("current_segment", {})
    
    if not current_segment:
        # No segment, shouldn't happen but handle gracefully
        return "segment_complete"
    
    status = current_segment.get("status", "")
    feedback = current_segment.get("critic_feedback", {})
    
    # Check approval
    is_approved = feedback.get("approved", False) if feedback else False
    
    if is_approved or status == "approved":
        return "segment_complete"
    else:
        return "retry_segment"


def route_after_production(
    state: MusicProducerState,
) -> Literal["critic", "retry_segment"]:
    """Route after production attempt.
    
    If generation succeeded, go to critic.
    If generation failed, retry.
    
    Args:
        state: Current workflow state.
        
    Returns:
        Next node name.
    """
    current_segment = state.get("current_segment", {})
    
    if not current_segment:
        return "retry_segment"
    
    status = current_segment.get("status", "")
    audio_path = current_segment.get("audio_path")
    
    if status == "generated" and audio_path:
        return "critic"
    else:
        return "retry_segment"


def should_continue_segments(
    state: MusicProducerState,
) -> Literal["production", "mastering"]:
    """Determine if more segments need processing.
    
    Args:
        state: Current workflow state.
        
    Returns:
        "production" if more segments, "mastering" if done.
    """
    segment_queue = state.get("segment_queue", [])
    current_index = state.get("current_segment_index", 0)
    status = state.get("status", "")
    
    if status == "mastering":
        return "mastering"
    
    if current_index >= len(segment_queue):
        return "mastering"
    
    return "production"


def route_after_segment_complete(
    state: MusicProducerState,
) -> Literal["production", "mastering"]:
    """Route after segment completion.
    
    Check if there are more segments to process.
    
    Args:
        state: Current workflow state.
        
    Returns:
        Next node name.
    """
    return should_continue_segments(state)


def route_after_retry(
    state: MusicProducerState,
) -> Literal["production", "segment_complete"]:
    """Route after retry decision.
    
    If max retries reached, segment_complete will have used best attempt.
    Otherwise, go back to production.
    
    Args:
        state: Current workflow state.
        
    Returns:
        Next node name.
    """
    current_segment = state.get("current_segment", {})
    
    # If current_segment is None, max retries was reached and we moved on
    if current_segment is None:
        segment_queue = state.get("segment_queue", [])
        current_index = state.get("current_segment_index", 0)
        
        if current_index >= len(segment_queue):
            return "segment_complete"  # Will route to mastering
    
    return "production"


def route_initial(
    state: MusicProducerState,
) -> Literal["analysis", "director"]:
    """Route at workflow start.
    
    If reference tracks provided, analyze them first.
    Otherwise, go straight to director.
    
    Args:
        state: Current workflow state.
        
    Returns:
        First node to execute.
    """
    reference_paths = state.get("reference_paths", [])
    
    if reference_paths:
        return "analysis"
    else:
        return "director"


def is_workflow_complete(state: MusicProducerState) -> bool:
    """Check if the workflow is complete.
    
    Args:
        state: Current workflow state.
        
    Returns:
        True if workflow is done.
    """
    status = state.get("status", "")
    final_path = state.get("final_output_path")
    
    return status == "completed" and final_path is not None


def get_workflow_status(state: MusicProducerState) -> dict:
    """Get a status summary of the workflow.
    
    Args:
        state: Current workflow state.
        
    Returns:
        Status dictionary.
    """
    segment_queue = state.get("segment_queue", [])
    completed = state.get("completed_segments", [])
    current_index = state.get("current_segment_index", 0)
    
    return {
        "status": state.get("status", "unknown"),
        "total_segments": len(segment_queue),
        "completed_segments": len(completed),
        "current_segment": current_index,
        "progress_percent": (
            (len(completed) / len(segment_queue) * 100)
            if segment_queue else 0
        ),
        "final_output": state.get("final_output_path"),
    }
