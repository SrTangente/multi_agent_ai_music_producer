"""Type definitions for the music producer workflow state.

All state schemas are defined as TypedDict for LangGraph compatibility.
These types are immutable by convention - updates return new state dicts.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict
import operator


# =============================================================================
# Tool Result Types
# =============================================================================

class ToolError(TypedDict):
    """Structured error from a tool call."""
    code: str  # e.g., "MUSICGEN_OOM", "LIBROSA_CORRUPT_FILE", "FILE_NOT_FOUND"
    message: str
    recoverable: bool
    suggested_action: str | None


class ToolResult(TypedDict):
    """Standard result wrapper for all tool calls."""
    success: bool
    data: Any | None
    error: ToolError | None


class GeneratedSegment(TypedDict):
    """Result from audio generation."""
    audio_path: str
    duration_sec: float
    generation_params: dict[str, Any]


class InstrumentEstimation(TypedDict):
    """Estimated instrumentation from audio analysis."""
    instruments: list[str]
    confidence: float
    dominant_instrument: str | None


# =============================================================================
# Musical Profile Types
# =============================================================================

class BPMResult(TypedDict):
    """BPM analysis result."""
    bpm: float
    confidence: float


class KeyResult(TypedDict):
    """Musical key analysis result."""
    key: str  # e.g., "Am", "C", "F#m"
    mode: Literal["major", "minor"]
    confidence: float


class EnergyProfile(TypedDict):
    """Energy characteristics over time."""
    mean_energy: float
    energy_variance: float
    energy_curve: list[float]  # Normalized energy over time segments


class SpectralFeatures(TypedDict):
    """Spectral characteristics."""
    spectral_centroid_mean: float
    spectral_bandwidth_mean: float
    spectral_rolloff_mean: float
    mfcc_means: list[float]


class MusicalProfile(TypedDict):
    """Complete musical profile from reference analysis."""
    reference_paths: list[str]
    bpm: BPMResult
    key: KeyResult
    energy: EnergyProfile
    spectral: SpectralFeatures
    instruments: InstrumentEstimation
    overall_mood: str  # Derived from analysis
    analysis_timestamp: str


# =============================================================================
# Track Planning Types
# =============================================================================

class SegmentParameters(TypedDict):
    """Parameters for generating a single segment."""
    segment_id: str
    segment_index: int
    segment_type: Literal["intro", "verse", "chorus", "bridge", "breakdown", "buildup", "outro"]
    duration_sec: float
    mood: str  # e.g., "melancholic", "energetic", "peaceful"
    energy_level: Literal["low", "medium", "high", "building", "dropping"]
    tempo_bpm: float
    key: str
    instrumentation_hints: list[str]
    transition_in: str | None  # How this segment should begin
    transition_out: str | None  # How this segment should end
    generation_prompt: str  # Full prompt for MusicGen


class TrackPlan(TypedDict):
    """Complete track structure planned by Director."""
    total_duration_sec: float
    segment_count: int
    segments: list[SegmentParameters]
    overall_mood: str
    overall_tempo_bpm: float
    overall_key: str
    style_description: str
    planning_timestamp: str


# =============================================================================
# Critic Feedback Types
# =============================================================================

class CriticIssue(TypedDict):
    """A specific issue identified by the Critic."""
    category: Literal["tempo", "mood", "instrumentation", "continuity", "artifacts", "duration", "energy"]
    severity: Literal["minor", "major", "critical"]
    description: str


class CriticFeedback(TypedDict):
    """Structured feedback from the Critic agent."""
    approved: bool
    overall_score: float  # 0.0-1.0
    
    # Dimensional scores (0.0-1.0)
    prompt_alignment: float
    director_compliance: float
    continuity_score: float
    technical_quality: float
    
    # Actionable feedback
    issues: list[CriticIssue]
    suggestions: list[str]
    
    # For best attempt selection
    better_than_previous: bool
    evaluation_timestamp: str


# =============================================================================
# Segment State Types
# =============================================================================

class AttemptRecord(TypedDict):
    """Record of a single generation attempt."""
    attempt_number: int
    audio_path: str
    timestamp: str
    generation_params: dict[str, Any]
    critic_feedback: CriticFeedback | None
    approved: bool


class SegmentState(TypedDict):
    """State of a single segment being produced."""
    segment_id: str
    parameters: SegmentParameters
    status: Literal["pending", "generating", "evaluating", "approved", "rejected", "failed"]
    attempts: list[AttemptRecord]
    current_attempt: int
    best_attempt_index: int | None  # Index of highest-scored attempt
    final_audio_path: str | None
    conditioning_audio_path: str | None


# =============================================================================
# Logging Types
# =============================================================================

class LogEntry(TypedDict):
    """Structured log entry."""
    timestamp: str
    level: Literal["INFO", "WARNING", "ERROR", "DEBUG"]
    agent: str | None
    action: str
    message: str
    inputs: dict[str, Any] | None
    outputs: dict[str, Any] | None
    duration_ms: int | None
    metadata: dict[str, Any] | None


class ErrorEntry(TypedDict):
    """Error log entry with full context."""
    timestamp: str
    agent: str | None
    action: str
    error: ToolError
    context: dict[str, Any]
    recoverable: bool


# =============================================================================
# LLM Configuration
# =============================================================================

class LLMConfigState(TypedDict):
    """LLM configuration stored in state."""
    provider: Literal["anthropic", "openai", "huggingface", "ollama"]
    model_name: str
    temperature: float
    max_tokens: int
    base_url: str | None


# =============================================================================
# Main Workflow State
# =============================================================================

class MusicProducerState(TypedDict):
    """Main state for the music producer workflow.
    
    This is the central state passed through all LangGraph nodes.
    Uses Annotated types for reducer functions where needed.
    """
    # === Run Identification ===
    run_id: str
    
    # === Input ===
    user_prompt: str
    reference_paths: list[str]
    output_dir: str
    
    # === Configuration ===
    llm_config: LLMConfigState
    max_retries: int
    
    # === Analysis Output ===
    musical_profile: MusicalProfile | None
    
    # === Director Output ===
    track_plan: TrackPlan | None
    
    # === Segment Loop State ===
    current_segment_index: int
    segments: Annotated[list[SegmentState], operator.add]  # Append reducer
    retry_count: int
    
    # === Accumulated Track ===
    accumulated_audio_path: str | None  # Path to combined audio so far
    approved_segment_paths: Annotated[list[str], operator.add]  # Append reducer
    
    # === Control Flow ===
    phase: Literal[
        "initialized",
        "analyzing",
        "planning",
        "producing",
        "critiquing", 
        "mastering",
        "complete",
        "failed"
    ]
    
    # === Observability ===
    logs: Annotated[list[LogEntry], operator.add]  # Append reducer
    errors: Annotated[list[ErrorEntry], operator.add]  # Append reducer
    
    # === Final Output ===
    final_track_path: str | None
    final_track_duration_sec: float | None
    
    # === Checkpointing ===
    last_checkpoint_path: str | None
