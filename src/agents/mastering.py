"""Mastering Agent for final track assembly and processing.

The Mastering Agent combines all approved segments into the
final track with proper transitions and mastering effects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agents.base import AgentConfig, BaseAgent, ToolSpec
from src.llm.base import LLMResponse
from src.state.schemas import MusicProducerState, SegmentState
from src.tools.audio_processing import (
    apply_compression,
    apply_crossfade,
    apply_fade_in,
    apply_fade_out,
    concatenate_segments,
    normalize_audio,
)
from src.utils.prompts import build_mastering_prompt


class MasteringAgent(BaseAgent):
    """Agent for assembling and mastering the final track.
    
    Responsible for:
    - Concatenating segments with crossfades
    - Applying fade-in and fade-out
    - Normalizing to target loudness
    - Optional compression for dynamics
    """
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        provider: str = "anthropic",
        temperature: float = 0.3,
        output_dir: str = "output",
        target_lufs: float = -14.0,
        torch_dtype: str | None = None,
        **kwargs,
    ):
        """Initialize the Mastering Agent.
        
        Args:
            model: LLM model to use.
            provider: LLM provider.
            temperature: LLM temperature.
            output_dir: Directory for final output.
            target_lufs: Target loudness in LUFS.
            torch_dtype: Dtype for HuggingFace models (e.g. 'bfloat16' for TPU).
            **kwargs: Additional BaseAgent arguments.
        """
        config = AgentConfig(
            name="mastering",
            description="Assembles and masters the final track",
            model=model,
            provider=provider,
            temperature=temperature,
            max_tokens=2048,
            max_tool_calls=10,
            torch_dtype=torch_dtype,
        )
        super().__init__(config=config, **kwargs)
        
        self.output_dir = output_dir
        self.target_lufs = target_lufs
    
    def _register_tools(self) -> None:
        """Register mastering tools."""
        self.register_tool(ToolSpec(
            name="concatenate_segments",
            description="Concatenate multiple audio segments with crossfades into one track",
            parameters={
                "type": "object",
                "properties": {
                    "segment_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of audio file paths in order",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Where to save the concatenated track",
                    },
                    "crossfade_duration_ms": {
                        "type": "integer",
                        "description": "Duration of crossfade between segments (default: 500ms)",
                    },
                },
                "required": ["segment_paths", "output_path"],
            },
            function=concatenate_segments,
        ))
        
        self.register_tool(ToolSpec(
            name="apply_fade_in",
            description="Apply fade-in to the beginning of a track",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the audio file",
                    },
                    "duration_ms": {
                        "type": "integer",
                        "description": "Duration of fade-in (default: 1000ms)",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Where to save (overwrites input if not specified)",
                    },
                },
                "required": ["path"],
            },
            function=apply_fade_in,
        ))
        
        self.register_tool(ToolSpec(
            name="apply_fade_out",
            description="Apply fade-out to the end of a track",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the audio file",
                    },
                    "duration_ms": {
                        "type": "integer",
                        "description": "Duration of fade-out (default: 2000ms)",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Where to save (overwrites input if not specified)",
                    },
                },
                "required": ["path"],
            },
            function=apply_fade_out,
        ))
        
        self.register_tool(ToolSpec(
            name="normalize_audio",
            description="Normalize audio to a target loudness level (LUFS)",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the audio file",
                    },
                    "target_lufs": {
                        "type": "number",
                        "description": "Target loudness in LUFS (default: -14.0)",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Where to save (overwrites input if not specified)",
                    },
                },
                "required": ["path"],
            },
            function=normalize_audio,
        ))
        
        self.register_tool(ToolSpec(
            name="apply_compression",
            description="Apply dynamic range compression to a track",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the audio file",
                    },
                    "threshold_db": {
                        "type": "number",
                        "description": "Threshold in dB (default: -20.0)",
                    },
                    "ratio": {
                        "type": "number",
                        "description": "Compression ratio (default: 4.0)",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Where to save (overwrites input if not specified)",
                    },
                },
                "required": ["path"],
            },
            function=apply_compression,
        ))
    
    def _get_system_prompt(self, state: MusicProducerState) -> str:
        """Get the system prompt for mastering."""
        return f"""You are a professional mastering engineer. Your task is to assemble all generated segments into a finished track.

You have access to mastering tools:
- concatenate_segments: Combine all segments with crossfades
- apply_fade_in: Add a fade-in to the beginning
- apply_fade_out: Add a fade-out to the ending
- normalize_audio: Set the final loudness level
- apply_compression: Control dynamic range

Standard mastering workflow:
1. Concatenate all segment audio files with appropriate crossfades
2. Apply fade-in to the intro (usually 1-2 seconds)
3. Apply fade-out to the outro (usually 2-4 seconds)
4. Normalize to target LUFS ({self.target_lufs} LUFS for streaming)
5. Optionally apply light compression if dynamics are too extreme

The output should be a polished, professional-sounding track ready for distribution."""
    
    def _get_user_prompt(self, state: MusicProducerState) -> str:
        """Get the user prompt with segment info."""
        completed_segments = state.get("completed_segments", [])
        track_plan = state.get("track_plan", {})
        
        return build_mastering_prompt(
            completed_segments=completed_segments,
            track_plan=track_plan,
            output_dir=self.output_dir,
            target_lufs=self.target_lufs,
        )
    
    def _process_response(
        self,
        response: LLMResponse,
        state: MusicProducerState,
    ) -> dict[str, Any]:
        """Process the response to get final output path."""
        content = response.content or ""
        
        # Parse the response to find the final track path
        final_path = self._parse_output_path(content)
        
        return {
            "final_output_path": final_path,
            "status": "completed" if final_path else "mastering_failed",
            "log": [{
                "timestamp": "",
                "event": "mastering_complete",
                "agent": self.name,
                "details": {
                    "output_path": final_path,
                    "segments_used": len(state.get("completed_segments", [])),
                },
            }],
        }
    
    def _parse_output_path(self, content: str) -> str | None:
        """Parse the response to find the output path."""
        import re
        
        # Look for output path in response
        path_patterns = [
            r"output[/\\][^\s'\"]+\.wav",
            r"final[_\-]?track[^\s'\"]*\.wav",
            r"master[^\s'\"]*\.wav",
        ]
        
        for pattern in path_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                path = match.group()
                if Path(path).exists():
                    return path
        
        # Default path
        default_path = f"{self.output_dir}/final_track.wav"
        return default_path if Path(default_path).exists() else None
    
    def master_direct(
        self,
        segments: list[SegmentState],
        output_filename: str = "final_track.wav",
        crossfade_ms: int = 500,
        fade_in_ms: int = 1000,
        fade_out_ms: int = 2000,
    ) -> dict[str, Any]:
        """Master the track directly without LLM.
        
        Args:
            segments: List of completed segments.
            output_filename: Name for the final file.
            crossfade_ms: Crossfade duration.
            fade_in_ms: Fade-in duration.
            fade_out_ms: Fade-out duration.
            
        Returns:
            Dictionary with output path and status.
        """
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Collect segment paths
        segment_paths = []
        for seg in segments:
            path = seg.get("audio_path")
            if path and Path(path).exists():
                segment_paths.append(path)
        
        if not segment_paths:
            return {
                "success": False,
                "error": "No valid segment audio files found",
                "output_path": None,
            }
        
        # Concatenate
        concat_path = f"{self.output_dir}/concatenated.wav"
        concat_result = concatenate_segments(
            segment_paths=segment_paths,
            output_path=concat_path,
            crossfade_duration_ms=crossfade_ms,
        )
        
        if not concat_result["success"]:
            return {
                "success": False,
                "error": f"Concatenation failed: {concat_result['error']}",
                "output_path": None,
            }
        
        # Apply fade-in
        fade_in_result = apply_fade_in(
            path=concat_path,
            duration_ms=fade_in_ms,
        )
        
        # Apply fade-out
        fade_out_result = apply_fade_out(
            path=concat_path,
            duration_ms=fade_out_ms,
        )
        
        # Normalize
        final_path = f"{self.output_dir}/{output_filename}"
        normalize_result = normalize_audio(
            path=concat_path,
            target_lufs=self.target_lufs,
            output_path=final_path,
        )
        
        if normalize_result["success"]:
            return {
                "success": True,
                "output_path": final_path,
                "duration_sec": normalize_result["data"].get("duration_sec"),
            }
        else:
            # Try to save without normalization
            import shutil
            shutil.copy(concat_path, final_path)
            return {
                "success": True,
                "output_path": final_path,
                "warning": "Normalization failed, output may not be at target loudness",
            }
