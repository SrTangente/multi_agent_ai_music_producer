"""State validation utilities.

Validates state at critical points to catch issues early.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.state.schemas import MusicProducerState, MusicalProfile, TrackPlan


class StateValidationError(Exception):
    """Raised when state validation fails."""
    
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(f"State validation failed: {'; '.join(errors)}")


def validate_state(
    state: MusicProducerState,
    phase: str | None = None,
) -> list[str]:
    """Validate the music producer state.
    
    Args:
        state: State to validate.
        phase: If provided, validate requirements for that phase.
        
    Returns:
        List of validation errors (empty if valid).
    """
    errors: list[str] = []
    
    # Always required
    if not state.get("run_id"):
        errors.append("run_id is required")
    
    if not state.get("user_prompt"):
        errors.append("user_prompt is required")
    
    if not state.get("reference_paths"):
        errors.append("reference_paths must not be empty")
    
    if not state.get("output_dir"):
        errors.append("output_dir is required")
    
    # Phase-specific validation
    if phase:
        phase_errors = _validate_for_phase(state, phase)
        errors.extend(phase_errors)
    
    return errors


def _validate_for_phase(state: MusicProducerState, phase: str) -> list[str]:
    """Validate state requirements for a specific phase.
    
    Args:
        state: State to validate.
        phase: Phase to validate for.
        
    Returns:
        List of validation errors.
    """
    errors: list[str] = []
    
    if phase in ("planning", "producing", "critiquing", "mastering", "complete"):
        # Must have musical profile after analysis
        if not state.get("musical_profile"):
            errors.append(f"musical_profile required for phase {phase}")
    
    if phase in ("producing", "critiquing", "mastering", "complete"):
        # Must have track plan after planning
        if not state.get("track_plan"):
            errors.append(f"track_plan required for phase {phase}")
    
    if phase == "mastering":
        # Must have at least one approved segment
        if not state.get("approved_segment_paths"):
            errors.append("approved_segment_paths required for mastering")
    
    if phase == "complete":
        # Must have final track
        if not state.get("final_track_path"):
            errors.append("final_track_path required for complete phase")
    
    return errors


def validate_musical_profile(profile: MusicalProfile) -> list[str]:
    """Validate a musical profile.
    
    Args:
        profile: Profile to validate.
        
    Returns:
        List of validation errors.
    """
    errors: list[str] = []
    
    if not profile.get("reference_paths"):
        errors.append("musical_profile must have reference_paths")
    
    bpm = profile.get("bpm", {})
    if not (20 <= bpm.get("bpm", 0) <= 300):
        errors.append(f"Invalid BPM: {bpm.get('bpm')}")
    
    key = profile.get("key", {})
    if not key.get("key"):
        errors.append("musical_profile must have key")
    
    return errors


def validate_track_plan(plan: TrackPlan) -> list[str]:
    """Validate a track plan.
    
    Args:
        plan: Plan to validate.
        
    Returns:
        List of validation errors.
    """
    errors: list[str] = []
    
    if not plan.get("segments"):
        errors.append("track_plan must have segments")
        return errors
    
    total_duration = 0.0
    for i, segment in enumerate(plan["segments"]):
        if not segment.get("segment_id"):
            errors.append(f"Segment {i} missing segment_id")
        
        duration = segment.get("duration_sec", 0)
        if duration <= 0:
            errors.append(f"Segment {i} has invalid duration: {duration}")
        total_duration += duration
        
        if not segment.get("generation_prompt"):
            errors.append(f"Segment {i} missing generation_prompt")
    
    # Check total duration matches
    expected_duration = plan.get("total_duration_sec", 0)
    if abs(total_duration - expected_duration) > 1.0:
        errors.append(
            f"Segment durations ({total_duration}s) don't match "
            f"total_duration_sec ({expected_duration}s)"
        )
    
    return errors


def validate_reference_paths(paths: list[str]) -> list[str]:
    """Validate reference audio file paths.
    
    Args:
        paths: List of file paths.
        
    Returns:
        List of validation errors.
    """
    errors: list[str] = []
    
    if not paths:
        errors.append("No reference paths provided")
        return errors
    
    valid_extensions = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
    
    for path in paths:
        p = Path(path)
        
        if not p.exists():
            errors.append(f"Reference file not found: {path}")
            continue
        
        if p.suffix.lower() not in valid_extensions:
            errors.append(f"Unsupported audio format: {path}")
    
    return errors


def validate_output_directory(output_dir: str) -> list[str]:
    """Validate output directory.
    
    Args:
        output_dir: Output directory path.
        
    Returns:
        List of validation errors.
    """
    errors: list[str] = []
    
    path = Path(output_dir)
    
    if not path.exists():
        try:
            path.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            errors.append(f"Cannot create output directory: {e}")
    elif not path.is_dir():
        errors.append(f"Output path exists but is not a directory: {output_dir}")
    
    return errors
