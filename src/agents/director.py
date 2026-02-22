"""Director Agent for planning the track structure.

The Director Agent takes the musical profile from Analysis and
creates a structured plan for the full track.
"""

from __future__ import annotations

import json
from typing import Any

from src.agents.base import AgentConfig, BaseAgent, ToolSpec
from src.llm.base import LLMResponse
from src.state.schemas import (
    MusicProducerState,
    SegmentParameters,
    TrackPlan,
)
from src.utils.prompts import build_director_prompt


class DirectorAgent(BaseAgent):
    """Agent for planning the overall track structure.
    
    Creates a TrackPlan with:
    - Total target duration
    - Number and duration of segments
    - Parameters for each segment
    - Transitions and energy flow
    """
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        provider: str = "anthropic",
        temperature: float = 0.7,
        **kwargs,
    ):
        """Initialize the Director Agent.
        
        Args:
            model: LLM model to use.
            provider: LLM provider.
            temperature: LLM temperature.
            **kwargs: Additional BaseAgent arguments.
        """
        config = AgentConfig(
            name="director",
            description="Plans the overall track structure and segment breakdown",
            model=model,
            provider=provider,
            temperature=temperature,
            max_tokens=4096,
            max_tool_calls=0,  # Director doesn't use tools, just reasons
        )
        super().__init__(config=config, **kwargs)
    
    def _register_tools(self) -> None:
        """Register tools (none for Director)."""
        pass
    
    def _get_system_prompt(self, state: MusicProducerState) -> str:
        """Get the system prompt for directing."""
        return """You are a music director and arranger. Your task is to plan the structure of a music track.

Given a musical profile from analysis and the user's request, create a detailed plan that:
1. Determines the total duration
2. Breaks the track into logical segments (intro, verse, chorus, bridge, outro, etc.)
3. Specifies the duration and character of each segment
4. Plans energy flow and transitions

Output your plan as a JSON object with this structure:
{
    "total_duration_sec": <number>,
    "segment_count": <number>,
    "segments": [
        {
            "segment_index": <number starting at 0>,
            "duration_sec": <number, typically 10-30>,
            "prompt": "<detailed text prompt for this segment>",
            "target_energy": "<low|medium|high>",
            "transition_type": "<fade|cut|crossfade|build>",
            "notes": "<additional production notes>"
        }
    ],
    "overall_notes": "<general production guidance>"
}

Consider:
- Most segments should be 10-30 seconds for quality generation
- Plan smooth energy transitions
- Include intro and outro segments
- Match the style and mood from the musical profile
- Be specific in segment prompts about instruments, energy, and feel"""
    
    def _get_user_prompt(self, state: MusicProducerState) -> str:
        """Get the user prompt with profile and request."""
        return build_director_prompt(
            musical_profile=state.get("musical_profile"),
            user_prompt=state["user_prompt"],
            target_duration=state.get("target_duration_sec", 120.0),
        )
    
    def _process_response(
        self,
        response: LLMResponse,
        state: MusicProducerState,
    ) -> dict[str, Any]:
        """Process the response into a track plan."""
        content = response.content or ""
        
        try:
            plan = self._parse_plan_from_response(content, state)
        except Exception as e:
            if self.logger:
                self.logger.log_error(
                    error_type="plan_parse_error",
                    message=str(e),
                    agent_name=self.name,
                )
            # Create default plan
            plan = self._create_default_plan(state)
        
        # Create segment parameters from plan
        segment_queue = self._create_segment_queue(plan)
        
        return {
            "track_plan": plan,
            "segment_queue": segment_queue,
            "log": [{
                "timestamp": "",
                "event": "planning_complete",
                "agent": self.name,
                "details": {
                    "total_duration": plan["total_duration_sec"],
                    "segment_count": plan["segment_count"],
                },
            }],
        }
    
    def _parse_plan_from_response(
        self,
        content: str,
        state: MusicProducerState,
    ) -> TrackPlan:
        """Parse the LLM response to extract track plan."""
        import re
        
        # Try to find JSON in the response
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            try:
                plan_data = json.loads(json_match.group())
                
                # Validate and convert to TrackPlan
                return TrackPlan(
                    total_duration_sec=float(plan_data.get(
                        "total_duration_sec",
                        state.get("target_duration_sec", 120.0),
                    )),
                    segment_count=int(plan_data.get("segment_count", 4)),
                    segment_durations=[
                        float(s.get("duration_sec", 20.0))
                        for s in plan_data.get("segments", [])
                    ],
                    segment_prompts=[
                        str(s.get("prompt", ""))
                        for s in plan_data.get("segments", [])
                    ],
                    transitions=[
                        str(s.get("transition_type", "crossfade"))
                        for s in plan_data.get("segments", [])
                    ],
                    overall_notes=str(plan_data.get("overall_notes", "")),
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
        
        # Fallback: create plan from defaults
        return self._create_default_plan(state)
    
    def _create_default_plan(self, state: MusicProducerState) -> TrackPlan:
        """Create a default track plan when parsing fails."""
        target_duration = state.get("target_duration_sec", 120.0)
        segment_duration = 20.0  # Default segment length
        segment_count = max(4, int(target_duration / segment_duration))
        
        # Adjust last segment to match total duration
        durations = [segment_duration] * segment_count
        total = sum(durations)
        if total != target_duration:
            durations[-1] += target_duration - total
        
        # Create prompts based on profile
        profile = state.get("musical_profile", {})
        base_prompt = state.get("user_prompt", "instrumental music")
        
        bpm = profile.get("bpm", 120)
        key = profile.get("key", "C")
        instruments = profile.get("instrument_suggestions", ["synth", "drums", "bass"])
        
        prompts = []
        energy_flow = ["low", "medium", "high", "high", "medium", "low"]
        
        for i in range(segment_count):
            energy = energy_flow[min(i, len(energy_flow) - 1)]
            if i == 0:
                section = "intro"
            elif i == segment_count - 1:
                section = "outro"
            elif i == segment_count // 2:
                section = "climax"
            else:
                section = "development"
            
            prompt = f"{base_prompt}, {section} section, {energy} energy, {bpm} BPM, key of {key}, {', '.join(instruments[:3])}"
            prompts.append(prompt)
        
        return TrackPlan(
            total_duration_sec=target_duration,
            segment_count=segment_count,
            segment_durations=durations,
            segment_prompts=prompts,
            transitions=["crossfade"] * segment_count,
            overall_notes=f"Generated plan for {target_duration}s track at {bpm} BPM in {key}",
        )
    
    def _create_segment_queue(self, plan: TrackPlan) -> list[SegmentParameters]:
        """Create the segment queue from the track plan."""
        queue = []
        
        for i in range(plan["segment_count"]):
            params = SegmentParameters(
                segment_index=i,
                duration_sec=plan["segment_durations"][i] if i < len(plan["segment_durations"]) else 20.0,
                prompt=plan["segment_prompts"][i] if i < len(plan["segment_prompts"]) else "",
                conditioning_context=None,  # Set during production
                target_energy=None,  # Could extract from prompt
                transition_type=plan["transitions"][i] if i < len(plan["transitions"]) else "crossfade",
            )
            queue.append(params)
        
        return queue
