"""State management for the music producer workflow."""

from src.state.schemas import (
    AttemptRecord,
    CriticFeedback,
    CriticIssue,
    ErrorEntry,
    GeneratedSegment,
    InstrumentEstimation,
    LogEntry,
    MusicProducerState,
    MusicalProfile,
    SegmentParameters,
    SegmentState,
    ToolError,
    ToolResult,
    TrackPlan,
)
from src.state.reducers import create_initial_state
from src.state.validation import validate_state

__all__ = [
    # State
    "MusicProducerState",
    "MusicalProfile",
    "TrackPlan",
    "SegmentParameters",
    "SegmentState",
    "AttemptRecord",
    "CriticFeedback",
    "CriticIssue",
    "LogEntry",
    "ErrorEntry",
    # Tool results
    "ToolResult",
    "ToolError",
    "GeneratedSegment",
    "InstrumentEstimation",
    # Functions
    "create_initial_state",
    "validate_state",
]
