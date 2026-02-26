"""Prompt building utilities for agents.

Centralized prompt templates and builders to ensure consistency
and maintainability across all agents.
"""

from __future__ import annotations

from typing import Any

from src.state.schemas import (
    CriticFeedback,
    MusicalProfile,
    SegmentParameters,
    SegmentState,
    TrackPlan,
)


# =============================================================================
# Analysis Agent Prompts
# =============================================================================

ANALYSIS_SYSTEM_PROMPT = """You are an expert music analyst AI. Your task is to analyze reference audio tracks and create a comprehensive musical profile.

You have access to audio analysis tools that provide:
- BPM (tempo) detection
- Musical key detection
- Energy profile analysis
- Spectral characteristics
- Instrument estimation

Analyze the provided reference tracks and synthesize a unified musical profile that captures the overall style, mood, and characteristics."""

def build_analysis_prompt(
    reference_paths: list[str],
    user_prompt: str,
) -> str:
    """Build the prompt for the Analysis Agent.
    
    Args:
        reference_paths: Paths to reference audio files.
        user_prompt: The user's original request.
        
    Returns:
        Formatted prompt string.
    """
    ref_list = "\n".join(f"- {path}" for path in reference_paths)
    
    return f"""Analyze the following reference tracks to create a musical profile.

User's Request: "{user_prompt}"

Reference Tracks:
{ref_list}

Instructions:
1. Use the analysis tools to extract BPM, key, energy, and spectral features from each track
2. Identify common instruments and timbral characteristics
3. Determine the overall mood based on the analysis
4. Synthesize findings into a unified musical profile

Return your analysis as a structured musical profile with:
- Consensus BPM (or range if tracks vary significantly)
- Most likely key and mode
- Energy characteristics (low/medium/high, dynamic range)
- Dominant instruments and timbres
- Overall mood description

Focus on characteristics that will guide music generation aligned with the user's request."""


# =============================================================================
# Director Agent Prompts
# =============================================================================

DIRECTOR_SYSTEM_PROMPT = """You are an expert music director AI. Your task is to plan the structure of a music track based on a user's request and musical analysis.

You create detailed track plans that specify:
- Overall structure (intro, verses, choruses, bridges, outros)
- Duration of each segment
- Mood and energy progression
- Instrumentation guidance
- Transition approaches

Your plans must be musically coherent and executable by an AI music generator."""

def build_director_prompt(
    user_prompt: str,
    musical_profile: MusicalProfile | None,
    target_duration_sec: float | None = None,
) -> str:
    """Build the prompt for the Director Agent.
    
    Args:
        user_prompt: The user's original request.
        musical_profile: Analysis results from Analysis Agent (may be None).
        target_duration_sec: Optional target total duration.
        
    Returns:
        Formatted prompt string.
    """
    duration_guidance = ""
    if target_duration_sec:
        duration_guidance = f"\nTarget Duration: {target_duration_sec} seconds"
    
    # Handle case where no musical profile is available
    if musical_profile is None:
        return f"""Create a detailed track plan for the following request.

User's Request: "{user_prompt}"
{duration_guidance}

Note: No reference tracks were provided. Please infer appropriate musical characteristics
(BPM, key, mood, instrumentation) from the user's request.

Instructions:
1. Design an appropriate song structure (e.g., intro → verse → chorus → verse → chorus → outro)
2. Assign duration to each segment (typically 8-16 seconds per segment)
3. Plan mood and energy progression through the track
4. Specify instrumentation hints for each segment
5. Define transition approaches between segments

Return a structured track plan with:
- Total duration and segment count
- For each segment:
  - Type (intro/verse/chorus/bridge/breakdown/buildup/outro)
  - Duration in seconds
  - Mood description
  - Energy level (low/medium/high/building/dropping)
  - Instrumentation hints
  - Transition descriptions (how to enter and exit)
  - A detailed generation prompt for the music generator

Ensure the plan aligns with the user's request."""
    
    return f"""Create a detailed track plan for the following request.

User's Request: "{user_prompt}"

Musical Profile from Reference Analysis:
- BPM: {musical_profile['bpm']['bpm']:.1f} (confidence: {musical_profile['bpm']['confidence']:.2f})
- Key: {musical_profile['key']['key']} {musical_profile['key']['mode']}
- Energy: Mean {musical_profile['energy']['mean_energy']:.2f}, Variance {musical_profile['energy']['energy_variance']:.2f}
- Mood: {musical_profile['overall_mood']}
- Instruments: {', '.join(musical_profile['instruments']['instruments'])}
{duration_guidance}

Instructions:
1. Design an appropriate song structure (e.g., intro → verse → chorus → verse → chorus → outro)
2. Assign duration to each segment (typically 8-16 seconds per segment)
3. Plan mood and energy progression through the track
4. Specify instrumentation hints for each segment
5. Define transition approaches between segments

Return a structured track plan with:
- Total duration and segment count
- For each segment:
  - Type (intro/verse/chorus/bridge/breakdown/buildup/outro)
  - Duration in seconds
  - Mood description
  - Energy level (low/medium/high/building/dropping)
  - Instrumentation hints
  - Transition descriptions (how to enter and exit)
  - A detailed generation prompt for the music generator

Ensure the plan aligns with the user's request and the reference track characteristics."""


# =============================================================================
# Production Agent Prompts
# =============================================================================

PRODUCTION_SYSTEM_PROMPT = """You are an expert music production AI. Your task is to generate audio segments using MusicGen based on detailed parameters.

You create prompts and configure generation settings to produce high-quality audio that:
- Matches the specified mood and energy
- Uses appropriate instrumentation
- Maintains continuity with previous segments
- Aligns with the overall track vision

You have access to audio generation tools and can adjust parameters for optimal results."""

def build_production_prompt(
    segment: SegmentParameters,
    prev_segment: SegmentState | None = None,
    retry_feedback: CriticFeedback | None = None,
) -> str:
    """Build the prompt for the Production Agent.
    
    Args:
        segment: Parameters for the segment to generate.
        prev_segment: State of the previous segment (for continuity).
        retry_feedback: Feedback from Critic if this is a retry.
        
    Returns:
        Formatted prompt string.
    """
    continuity_context = ""
    if prev_segment:
        prev_params = prev_segment["parameters"]
        continuity_context = f"""
Previous Segment Context:
- Type: {prev_params['segment_type']}
- Mood: {prev_params['mood']}
- Energy: {prev_params['energy_level']}
- This segment should flow naturally from the previous one.
"""

    retry_context = ""
    if retry_feedback:
        issues = "\n".join(f"  - [{i['severity']}] {i['category']}: {i['description']}" 
                          for i in retry_feedback['issues'])
        suggestions = "\n".join(f"  - {s}" for s in retry_feedback['suggestions'])
        retry_context = f"""
RETRY CONTEXT - Previous attempt was rejected:
Issues to address:
{issues}

Suggestions:
{suggestions}

Adjust your generation approach to address these issues.
"""

    return f"""Generate audio for the following segment.

Segment Details:
- ID: {segment['segment_id']}
- Type: {segment['segment_type']}
- Duration: {segment['duration_sec']} seconds
- Mood: {segment['mood']}
- Energy Level: {segment['energy_level']}
- Tempo: {segment['tempo_bpm']} BPM
- Key: {segment['key']}
- Instrumentation: {', '.join(segment['instrumentation_hints'])}
- Transition In: {segment['transition_in'] or 'Standard'}
- Transition Out: {segment['transition_out'] or 'Standard'}
{continuity_context}{retry_context}
Base Generation Prompt (from Director):
"{segment['generation_prompt']}"

Instructions:
1. Use the generate_segment tool with the provided parameters
2. Ensure the audio matches the mood, energy, and instrumentation requirements
3. If conditioning audio is provided, maintain musical continuity
4. Verify the output duration matches the requirement

Generate the audio segment now."""


def build_segment_generation_prompt(
    segment: SegmentParameters,
    prev_segment: SegmentParameters | None = None,
    next_segment: SegmentParameters | None = None,
) -> str:
    """Build the MusicGen prompt for a segment.
    
    This is the text prompt passed directly to MusicGen, not the agent prompt.
    
    Args:
        segment: Current segment parameters.
        prev_segment: Previous segment parameters (for transition context).
        next_segment: Next segment parameters (for anticipation).
        
    Returns:
        MusicGen-optimized prompt string.
    """
    base = segment['generation_prompt']
    
    # Add transition context
    transitions = []
    
    if prev_segment:
        if prev_segment['energy_level'] != segment['energy_level']:
            if segment['energy_level'] in ('high', 'building'):
                transitions.append("building energy from previous section")
            elif segment['energy_level'] in ('low', 'dropping'):
                transitions.append("calming down from previous section")
    
    if segment['segment_type'] == 'intro':
        transitions.append("gentle opening")
    elif segment['segment_type'] == 'outro':
        transitions.append("natural conclusion with fade")
    
    if next_segment:
        if next_segment['segment_type'] == 'chorus' and segment['segment_type'] != 'chorus':
            transitions.append("building anticipation toward chorus")
    
    if transitions:
        base = f"{base}, {', '.join(transitions)}"
    
    return base


# =============================================================================
# Critic Agent Prompts
# =============================================================================

CRITIC_SYSTEM_PROMPT = """You are an expert music critic AI. Your task is to evaluate generated audio segments against quality criteria.

You assess segments based on:
- Alignment with the original user request
- Compliance with the Director's parameters
- Musical coherence and continuity with previous segments
- Technical quality (no artifacts, proper levels)

You provide structured feedback with scores and specific, actionable suggestions for improvement."""

def build_critic_prompt(
    segment: SegmentParameters,
    user_prompt: str,
    prev_segment: SegmentState | None = None,
    attempt_number: int = 1,
    audio_path: str | None = None,
) -> str:
    """Build the prompt for the Critic Agent.
    
    Args:
        segment: Parameters the segment should match.
        user_prompt: Original user request.
        prev_segment: Previous segment for continuity evaluation.
        attempt_number: Current attempt number.
        audio_path: Path to the generated audio.
        
    Returns:
        Formatted prompt string.
    """
    continuity_section = ""
    if prev_segment:
        prev_params = prev_segment["parameters"]
        continuity_section = f"""
Previous Segment (for continuity evaluation):
- Type: {prev_params['segment_type']}
- Mood: {prev_params['mood']}
- Energy: {prev_params['energy_level']}
- Key: {prev_params['key']}
- Tempo: {prev_params['tempo_bpm']} BPM
"""

    return f"""Evaluate the generated audio segment.

Original User Request: "{user_prompt}"

Target Segment Parameters:
- Type: {segment['segment_type']}
- Duration: {segment['duration_sec']} seconds
- Mood: {segment['mood']}
- Energy Level: {segment['energy_level']}
- Tempo: {segment['tempo_bpm']} BPM
- Key: {segment['key']}
- Instrumentation: {', '.join(segment['instrumentation_hints'])}
{continuity_section}
Audio to evaluate: {audio_path}
Attempt Number: {attempt_number}

Instructions:
1. Use analysis tools to examine the generated audio
2. Compare actual characteristics to target parameters
3. Evaluate alignment with user request
4. Check for technical issues (artifacts, clipping, silence)
5. Assess continuity with previous segment if applicable

Provide structured feedback with:
- Overall approval decision (approve/reject)
- Overall score (0.0-1.0)
- Dimensional scores:
  - prompt_alignment: How well it matches user intent
  - director_compliance: How well it matches segment parameters
  - continuity_score: Coherence with previous segment (1.0 if first segment)
  - technical_quality: Absence of artifacts, proper levels
- List of specific issues with severity (minor/major/critical)
- Actionable suggestions for improvement if rejected

Be constructive but maintain quality standards. Approve only if the segment meets acceptable quality thresholds."""


# =============================================================================
# Mastering Agent Prompts
# =============================================================================

MASTERING_SYSTEM_PROMPT = """You are an expert audio mastering AI. Your task is to combine and polish the final track from approved segments.

You handle:
- Segment concatenation with smooth crossfades
- Volume normalization to broadcast standards
- Fade in/out application
- Overall coherence verification

You use audio processing tools to create a polished, professional final track."""

def build_mastering_prompt(
    track_plan: TrackPlan,
    segment_paths: list[str],
    user_prompt: str,
) -> str:
    """Build the prompt for the Mastering Agent.
    
    Args:
        track_plan: The complete track plan from Director.
        segment_paths: Paths to approved audio segments in order.
        user_prompt: Original user request.
        
    Returns:
        Formatted prompt string.
    """
    segment_list = "\n".join(f"  {i+1}. {path}" for i, path in enumerate(segment_paths))
    
    return f"""Master the final track from the approved segments.

Original User Request: "{user_prompt}"

Track Plan:
- Total Duration: {track_plan['total_duration_sec']} seconds
- Segment Count: {track_plan['segment_count']}
- Style: {track_plan['style_description']}

Approved Segments (in order):
{segment_list}

Instructions:
1. Concatenate segments with smooth crossfades (500ms recommended)
2. Apply fade-in at the beginning (1-2 seconds)
3. Apply fade-out at the end (2-3 seconds)
4. Normalize to -14 LUFS for streaming compatibility
5. Verify the final track plays smoothly without artifacts

Use the mastering tools to:
- concatenate_segments: Join all segments
- apply_crossfade: Smooth transitions
- apply_fade_in / apply_fade_out: Professional bookends
- normalize_audio: Consistent loudness

Output the path to the final mastered track."""


# =============================================================================
# Utility Functions
# =============================================================================

def truncate_prompt(prompt: str, max_length: int = 4000) -> str:
    """Truncate a prompt to maximum length while keeping it coherent.
    
    Args:
        prompt: The prompt to potentially truncate.
        max_length: Maximum character length.
        
    Returns:
        Truncated prompt with ellipsis if needed.
    """
    if len(prompt) <= max_length:
        return prompt
    
    # Try to truncate at a paragraph boundary
    truncated = prompt[:max_length]
    last_newline = truncated.rfind("\n\n")
    if last_newline > max_length * 0.7:
        truncated = truncated[:last_newline]
    
    return truncated + "\n\n[Content truncated for length]"


def format_tool_result_for_prompt(
    tool_name: str,
    result: dict[str, Any],
    max_length: int = 500,
) -> str:
    """Format a tool result for inclusion in a prompt.
    
    Args:
        tool_name: Name of the tool that produced the result.
        result: The tool result dictionary.
        max_length: Maximum length for the formatted string.
        
    Returns:
        Formatted string representation.
    """
    import json
    
    formatted = f"[{tool_name} result]\n"
    try:
        result_str = json.dumps(result, indent=2, default=str)
        if len(result_str) > max_length:
            result_str = result_str[:max_length] + "\n..."
        formatted += result_str
    except (TypeError, ValueError):
        formatted += str(result)[:max_length]
    
    return formatted
