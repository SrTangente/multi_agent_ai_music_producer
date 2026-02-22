"""Production Agent for generating audio segments.

The Production Agent generates audio for each segment using
MusicGen with appropriate prompts and conditioning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agents.base import AgentConfig, BaseAgent, ToolSpec
from src.llm.base import LLMResponse
from src.state.schemas import (
    GeneratedSegment,
    MusicProducerState,
    SegmentParameters,
    SegmentState,
)
from src.tools.audio_generation import generate_segment, generate_segment_mock
from src.tools.audio_io import extract_audio_tail, get_audio_duration
from src.utils.prompts import build_production_prompt, build_segment_generation_prompt


class ProductionAgent(BaseAgent):
    """Agent for generating audio segments using MusicGen.
    
    For each segment:
    1. Refines the generation prompt based on context
    2. Prepares conditioning audio if available
    3. Generates the audio
    4. Validates the output
    """
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        provider: str = "anthropic",
        temperature: float = 0.5,
        output_dir: str = "output/segments",
        use_mock: bool = False,
        **kwargs,
    ):
        """Initialize the Production Agent.
        
        Args:
            model: LLM model to use.
            provider: LLM provider.
            temperature: LLM temperature.
            output_dir: Directory for generated segments.
            use_mock: Use mock generation (for testing without GPU).
            **kwargs: Additional BaseAgent arguments.
        """
        config = AgentConfig(
            name="production",
            description="Generates audio segments using MusicGen",
            model=model,
            provider=provider,
            temperature=temperature,
            max_tokens=2048,
            max_tool_calls=5,
        )
        super().__init__(config=config, **kwargs)
        
        self.output_dir = output_dir
        self.use_mock = use_mock
    
    def _register_tools(self) -> None:
        """Register audio generation tools."""
        # Choose real or mock generation
        gen_func = generate_segment_mock if self.use_mock else generate_segment
        
        self.register_tool(ToolSpec(
            name="generate_audio_segment",
            description="Generate an audio segment using MusicGen from a text prompt",
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the audio to generate",
                    },
                    "duration_sec": {
                        "type": "number",
                        "description": "Duration of the segment in seconds",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path where the generated audio will be saved",
                    },
                    "conditioning_audio_path": {
                        "type": "string",
                        "description": "Optional path to audio for melodic conditioning",
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Sampling temperature (default: 1.0)",
                    },
                },
                "required": ["prompt", "duration_sec", "output_path"],
            },
            function=gen_func,
        ))
        
        self.register_tool(ToolSpec(
            name="extract_conditioning_audio",
            description="Extract the last N seconds from an audio file for conditioning",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the source audio file",
                    },
                    "duration_sec": {
                        "type": "number",
                        "description": "Duration to extract from the end",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Where to save the extracted audio",
                    },
                },
                "required": ["path", "duration_sec"],
            },
            function=extract_audio_tail,
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
        """Get the system prompt for production."""
        return """You are a music production specialist using AI audio generation.

Your task is to generate a specific segment of a track. You have access to:
- generate_audio_segment: Creates audio from a text prompt
- extract_conditioning_audio: Gets the end of the previous segment for continuity
- get_audio_duration: Checks the duration of generated audio

Guidelines:
1. If there's a previous segment, extract its tail for conditioning
2. Refine the prompt to be specific and detailed for MusicGen
3. Generate the audio segment
4. Verify the output duration

The prompt should include:
- Musical style and genre
- Tempo (e.g., "120 BPM")
- Key (e.g., "key of C major")
- Instruments
- Energy level and mood
- Specific characteristics for this section

Be specific and descriptive for best results."""
    
    def _get_user_prompt(self, state: MusicProducerState) -> str:
        """Get the user prompt with segment parameters."""
        current_segment = state.get("current_segment_index", 0)
        segment_queue = state.get("segment_queue", [])
        
        if current_segment < len(segment_queue):
            params = segment_queue[current_segment]
        else:
            # Shouldn't happen, but fallback
            params = SegmentParameters(
                segment_index=current_segment,
                duration_sec=20.0,
                prompt=state.get("user_prompt", "instrumental music"),
                conditioning_context=None,
                target_energy=None,
                transition_type="crossfade",
            )
        
        # Get previous segment path for conditioning
        previous_path = None
        completed = state.get("completed_segments", [])
        if completed:
            last_segment = completed[-1]
            previous_path = last_segment.get("audio_path")
        
        return build_production_prompt(
            segment_params=params,
            segment_index=current_segment,
            total_segments=len(segment_queue),
            previous_segment_path=previous_path,
            output_dir=self.output_dir,
            musical_profile=state.get("musical_profile"),
        )
    
    def _process_response(
        self,
        response: LLMResponse,
        state: MusicProducerState,
    ) -> dict[str, Any]:
        """Process the response to extract generated segment info."""
        current_index = state.get("current_segment_index", 0)
        
        # Parse the response to find the generated segment path
        # The LLM should have used tools and reported the result
        segment_state = self._parse_segment_from_response(
            response.content or "",
            current_index,
        )
        
        return {
            "current_segment": segment_state,
            "log": [{
                "timestamp": "",
                "event": "segment_generated",
                "agent": self.name,
                "details": {
                    "segment_index": current_index,
                    "audio_path": segment_state.get("audio_path"),
                    "attempt": segment_state.get("attempt_number", 1),
                },
            }],
        }
    
    def _parse_segment_from_response(
        self,
        content: str,
        segment_index: int,
    ) -> SegmentState:
        """Parse the response to create segment state."""
        import re
        
        # Try to find the output path in the response
        path_match = re.search(r"['\"]?([^'\"]*segment[_-]?\d+[^'\"]*\.wav)['\"]?", content)
        if path_match:
            audio_path = path_match.group(1)
        else:
            # Default path
            audio_path = f"{self.output_dir}/segment_{segment_index}.wav"
        
        # Try to find duration
        duration_match = re.search(r"duration[:\s]*(\d+(?:\.\d+)?)\s*(?:sec|s)?", content, re.IGNORECASE)
        duration = float(duration_match.group(1)) if duration_match else 20.0
        
        return SegmentState(
            segment_index=segment_index,
            audio_path=audio_path if Path(audio_path).exists() else None,
            duration_sec=duration,
            generation_params={
                "response_snippet": content[:200],
            },
            status="generated" if Path(audio_path).exists() else "failed",
            attempt_number=1,
            critic_feedback=None,
        )
    
    def generate_segment_direct(
        self,
        params: SegmentParameters,
        conditioning_path: str | None = None,
    ) -> SegmentState:
        """Generate a segment directly without LLM (for fallback).
        
        Args:
            params: Segment parameters.
            conditioning_path: Path to conditioning audio.
            
        Returns:
            SegmentState with generation result.
        """
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        output_path = f"{self.output_dir}/segment_{params['segment_index']}.wav"
        
        # Generate
        gen_func = generate_segment_mock if self.use_mock else generate_segment
        result = gen_func(
            prompt=params["prompt"],
            duration_sec=params["duration_sec"],
            output_path=output_path,
            conditioning_audio_path=conditioning_path,
        )
        
        if result["success"]:
            return SegmentState(
                segment_index=params["segment_index"],
                audio_path=result["data"]["audio_path"],
                duration_sec=result["data"]["duration_sec"],
                generation_params=result["data"]["generation_params"],
                status="generated",
                attempt_number=1,
                critic_feedback=None,
            )
        else:
            return SegmentState(
                segment_index=params["segment_index"],
                audio_path=None,
                duration_sec=0.0,
                generation_params={"error": result["error"]},
                status="failed",
                attempt_number=1,
                critic_feedback=None,
            )
