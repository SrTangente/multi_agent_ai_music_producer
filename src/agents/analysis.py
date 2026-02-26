"""Analysis Agent for analyzing reference tracks.

The Analysis Agent examines reference audio files to extract
musical features that guide the production process.
"""

from __future__ import annotations

from typing import Any

from src.agents.base import AgentConfig, BaseAgent, ToolSpec
from src.llm.base import LLMResponse
from src.state.schemas import MusicProducerState, MusicalProfile
from src.tools.audio_analysis import (
    analyze_bpm,
    analyze_energy,
    analyze_key,
    analyze_spectral,
    estimate_instruments,
)
from src.tools.audio_io import get_audio_duration, load_audio
from src.utils.prompts import build_analysis_prompt


class AnalysisAgent(BaseAgent):
    """Agent for analyzing reference tracks and building musical profiles.
    
    Uses audio analysis tools to extract:
    - BPM and tempo characteristics
    - Musical key and mode
    - Energy profile over time
    - Spectral characteristics
    - Estimated instrumentation
    
    Produces a MusicalProfile that guides the Director.
    """
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        provider: str = "anthropic",
        temperature: float = 0.3,
        torch_dtype: str | None = None,
        **kwargs,
    ):
        """Initialize the Analysis Agent.
        
        Args:
            model: LLM model to use.
            provider: LLM provider.
            temperature: LLM temperature (lower for more consistent analysis).
            torch_dtype: Dtype for HuggingFace models (e.g. 'bfloat16' for TPU).
            **kwargs: Additional BaseAgent arguments.
        """
        config = AgentConfig(
            name="analysis",
            description="Analyzes reference tracks to extract musical features",
            model=model,
            provider=provider,
            temperature=temperature,
            max_tokens=2048,
            max_tool_calls=20,  # May need to analyze multiple tracks
            torch_dtype=torch_dtype,
        )
        super().__init__(config=config, **kwargs)
    
    def _register_tools(self) -> None:
        """Register audio analysis tools."""
        self.register_tool(ToolSpec(
            name="analyze_bpm",
            description="Analyze the tempo (BPM) of an audio file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the audio file to analyze",
                    },
                },
                "required": ["path"],
            },
            function=analyze_bpm,
        ))
        
        self.register_tool(ToolSpec(
            name="analyze_key",
            description="Analyze the musical key and mode of an audio file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the audio file to analyze",
                    },
                },
                "required": ["path"],
            },
            function=analyze_key,
        ))
        
        self.register_tool(ToolSpec(
            name="analyze_energy",
            description="Analyze the energy profile of an audio file over time",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the audio file to analyze",
                    },
                    "num_segments": {
                        "type": "integer",
                        "description": "Number of segments to divide the track into (default: 10)",
                    },
                },
                "required": ["path"],
            },
            function=analyze_energy,
        ))
        
        self.register_tool(ToolSpec(
            name="analyze_spectral",
            description="Analyze spectral characteristics (centroid, bandwidth, MFCCs)",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the audio file to analyze",
                    },
                },
                "required": ["path"],
            },
            function=analyze_spectral,
        ))
        
        self.register_tool(ToolSpec(
            name="estimate_instruments",
            description="Estimate the instrumentation in an audio file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the audio file to analyze",
                    },
                },
                "required": ["path"],
            },
            function=estimate_instruments,
        ))
        
        self.register_tool(ToolSpec(
            name="get_audio_duration",
            description="Get the duration of an audio file in seconds",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the audio file",
                    },
                },
                "required": ["path"],
            },
            function=get_audio_duration,
        ))
    
    def _get_system_prompt(self, state: MusicProducerState) -> str:
        """Get the system prompt for analysis."""
        return """You are a music analysis expert. Your task is to analyze reference tracks and extract key musical features.

You have access to audio analysis tools. Use them systematically to build a complete picture of the reference tracks.

For each reference track:
1. Get the duration
2. Analyze the BPM
3. Analyze the key
4. Analyze the energy profile
5. Analyze spectral characteristics
6. Estimate the instrumentation

After analyzing all tracks, synthesize the findings into a coherent musical profile that captures:
- The average/target BPM
- The dominant key
- The overall energy pattern
- Key spectral characteristics
- Recommended instruments

Be thorough but efficient. If a tool fails, note the failure but continue with other analyses."""
    
    def _get_user_prompt(self, state: MusicProducerState) -> str:
        """Get the user prompt with reference track info."""
        return build_analysis_prompt(
            reference_paths=state["reference_paths"],
            user_prompt=state["user_prompt"],
        )
    
    def _process_response(
        self,
        response: LLMResponse,
        state: MusicProducerState,
    ) -> dict[str, Any]:
        """Process the response into a musical profile.
        
        Parses the LLM's analysis summary to create the MusicalProfile.
        """
        content = response.content or ""
        
        # Parse the response to extract profile
        # The LLM should have structured its response
        profile = self._parse_profile_from_response(content, state)
        
        return {
            "musical_profile": profile,
            "log": [{
                "timestamp": "",  # Will be set by reducer
                "event": "analysis_complete",
                "agent": self.name,
                "details": {
                    "tracks_analyzed": len(state["reference_paths"]),
                    "profile_confidence": profile.get("analysis_confidence", 0.7),
                },
            }],
        }
    
    def _parse_profile_from_response(
        self,
        content: str,
        state: MusicProducerState,
    ) -> MusicalProfile:
        """Parse the LLM response to extract musical profile.
        
        Uses heuristics to extract structured data from the response.
        Falls back to defaults if parsing fails.
        """
        import re
        
        # Default profile
        profile: MusicalProfile = {
            "bpm": 120.0,
            "key": "C",
            "mode": "major",
            "energy_target": "medium",
            "style_descriptors": [],
            "instrument_suggestions": [],
            "reference_summary": content[:500] if content else "",
        }
        
        # Try to extract BPM
        bpm_match = re.search(r"(?:BPM|tempo)[:\s]*(\d+(?:\.\d+)?)", content, re.IGNORECASE)
        if bpm_match:
            profile["bpm"] = float(bpm_match.group(1))
        
        # Try to extract key
        key_match = re.search(
            r"(?:key|tonic)[:\s]*([A-G][#b]?)\s*(major|minor)?",
            content,
            re.IGNORECASE,
        )
        if key_match:
            profile["key"] = key_match.group(1)
            if key_match.group(2):
                profile["mode"] = key_match.group(2).lower()
        
        # Extract energy level
        if any(word in content.lower() for word in ["high energy", "energetic", "intense"]):
            profile["energy_target"] = "high"
        elif any(word in content.lower() for word in ["low energy", "calm", "relaxed", "chill"]):
            profile["energy_target"] = "low"
        else:
            profile["energy_target"] = "medium"
        
        # Extract instruments
        common_instruments = [
            "drums", "bass", "guitar", "piano", "synth", "strings",
            "vocals", "percussion", "brass", "organ", "pad",
        ]
        found_instruments = []
        for instrument in common_instruments:
            if instrument in content.lower():
                found_instruments.append(instrument)
        if found_instruments:
            profile["instrument_suggestions"] = found_instruments
        
        # Extract style descriptors
        style_words = [
            "electronic", "acoustic", "ambient", "upbeat", "melodic",
            "rhythmic", "atmospheric", "dynamic", "minimal", "complex",
        ]
        found_styles = []
        for style in style_words:
            if style in content.lower():
                found_styles.append(style)
        if found_styles:
            profile["style_descriptors"] = found_styles
        
        return profile
