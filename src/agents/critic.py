"""Critic Agent for evaluating generated segments.

The Critic Agent analyzes generated audio segments and provides
structured feedback for quality improvement.
"""

from __future__ import annotations

import json
from typing import Any

from src.agents.base import AgentConfig, BaseAgent, ToolSpec
from src.llm.base import LLMResponse
from src.state.schemas import (
    CriticFeedback,
    MusicProducerState,
    SegmentState,
)
from src.tools.audio_analysis import (
    analyze_bpm,
    analyze_energy,
    analyze_key,
    analyze_spectral,
)
from src.utils.prompts import build_critic_prompt


class CriticAgent(BaseAgent):
    """Agent for evaluating and critiquing generated segments.
    
    Analyzes each generated segment for:
    - Technical quality (clarity, artifacts)
    - Musical consistency (BPM, key)
    - Energy appropriateness
    - Continuity with previous segments
    
    Produces CriticFeedback with scores and revision suggestions.
    """
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        provider: str = "anthropic",
        temperature: float = 0.3,
        approval_threshold: float = 0.7,
        **kwargs,
    ):
        """Initialize the Critic Agent.
        
        Args:
            model: LLM model to use.
            provider: LLM provider.
            temperature: LLM temperature (lower for consistent evaluation).
            approval_threshold: Minimum score to approve a segment.
            **kwargs: Additional BaseAgent arguments.
        """
        config = AgentConfig(
            name="critic",
            description="Evaluates generated audio segments for quality",
            model=model,
            provider=provider,
            temperature=temperature,
            max_tokens=2048,
            max_tool_calls=10,
        )
        super().__init__(config=config, **kwargs)
        
        self.approval_threshold = approval_threshold
    
    def _register_tools(self) -> None:
        """Register analysis tools for evaluation."""
        self.register_tool(ToolSpec(
            name="analyze_bpm",
            description="Analyze the tempo (BPM) of the generated segment",
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
            description="Analyze the musical key of the generated segment",
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
            description="Analyze the energy profile of the generated segment",
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
            function=analyze_energy,
        ))
        
        self.register_tool(ToolSpec(
            name="analyze_spectral",
            description="Analyze spectral characteristics for quality assessment",
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
    
    def _get_system_prompt(self, state: MusicProducerState) -> str:
        """Get the system prompt for critique."""
        return f"""You are an expert music producer and quality critic. Your task is to evaluate a generated audio segment.

You have access to audio analysis tools. Use them to objectively measure the segment's characteristics, then compare against the target profile.

Evaluate the segment on these criteria (0.0 to 1.0 scale):
1. **consistency_score**: How well does it match the target BPM and key?
2. **quality_score**: Audio clarity, absence of artifacts
3. **energy_score**: Does it match the target energy level?
4. **continuity_score**: Does it flow well from the previous segment?

A segment is APPROVED if its average score is >= {self.approval_threshold}

After analysis, output your evaluation as JSON:
{{
    "approved": <true/false>,
    "consistency_score": <0.0-1.0>,
    "quality_score": <0.0-1.0>,
    "energy_score": <0.0-1.0>,
    "continuity_score": <0.0-1.0>,
    "issues": ["list of specific issues found"],
    "revision_suggestions": ["specific suggestions for improvement"],
    "notes": "additional observations"
}}

Be objective and constructive. If issues are minor, still approve. Only reject if there are significant problems."""
    
    def _get_user_prompt(self, state: MusicProducerState) -> str:
        """Get the user prompt with segment to evaluate."""
        current_segment = state.get("current_segment")
        musical_profile = state.get("musical_profile", {})
        segment_queue = state.get("segment_queue", [])
        current_index = state.get("current_segment_index", 0)
        
        # Get target parameters
        if current_index < len(segment_queue):
            target_params = segment_queue[current_index]
        else:
            target_params = None
        
        # Get previous segment for continuity comparison
        completed = state.get("completed_segments", [])
        previous_path = completed[-1].get("audio_path") if completed else None
        
        return build_critic_prompt(
            segment_state=current_segment,
            target_profile=musical_profile,
            target_params=target_params,
            previous_segment_path=previous_path,
        )
    
    def _process_response(
        self,
        response: LLMResponse,
        state: MusicProducerState,
    ) -> dict[str, Any]:
        """Process the response into critic feedback."""
        content = response.content or ""
        
        try:
            feedback = self._parse_feedback_from_response(content)
        except Exception as e:
            if self.logger:
                self.logger.error(
                    action="feedback_parse_error",
                    message=str(e),
                    agent=self.name,
                )
            # Default to approval with moderate scores
            feedback = CriticFeedback(
                approved=True,
                consistency_score=0.7,
                quality_score=0.7,
                energy_score=0.7,
                continuity_score=0.7,
                issues=[],
                revision_suggestions=[],
                notes="Parse error - defaulting to approval",
            )
        
        # Update current segment with feedback
        current_segment = state.get("current_segment", {})
        if current_segment:
            current_segment = dict(current_segment)
            current_segment["critic_feedback"] = feedback
            current_segment["status"] = "approved" if feedback["approved"] else "needs_revision"
        
        return {
            "current_segment": current_segment,
            "log": [{
                "timestamp": "",
                "event": "segment_evaluated",
                "agent": self.name,
                "details": {
                    "approved": feedback["approved"],
                    "avg_score": (
                        feedback["consistency_score"] +
                        feedback["quality_score"] +
                        feedback["energy_score"] +
                        feedback["continuity_score"]
                    ) / 4,
                    "issues_count": len(feedback["issues"]),
                },
            }],
        }
    
    def _parse_feedback_from_response(self, content: str) -> CriticFeedback:
        """Parse the LLM response to extract feedback."""
        import re
        
        # Try to find JSON in the response
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            try:
                data = json.loads(json_match.group())
                
                return CriticFeedback(
                    approved=bool(data.get("approved", True)),
                    consistency_score=float(data.get("consistency_score", 0.7)),
                    quality_score=float(data.get("quality_score", 0.7)),
                    energy_score=float(data.get("energy_score", 0.7)),
                    continuity_score=float(data.get("continuity_score", 0.7)),
                    issues=list(data.get("issues", [])),
                    revision_suggestions=list(data.get("revision_suggestions", [])),
                    notes=str(data.get("notes", "")),
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
        
        # Fallback: try to extract from text
        approved = "approved" in content.lower() and "not approved" not in content.lower()
        
        # Look for scores
        scores = {}
        for score_name in ["consistency", "quality", "energy", "continuity"]:
            match = re.search(rf"{score_name}[_\s]?score[:\s]*(\d+\.?\d*)", content, re.IGNORECASE)
            if match:
                scores[score_name] = min(1.0, float(match.group(1)))
        
        return CriticFeedback(
            approved=approved,
            consistency_score=scores.get("consistency", 0.7),
            quality_score=scores.get("quality", 0.7),
            energy_score=scores.get("energy", 0.7),
            continuity_score=scores.get("continuity", 0.7),
            issues=[],
            revision_suggestions=[],
            notes=f"Parsed from text: {content[:100]}...",
        )
    
    def evaluate_segment_direct(
        self,
        segment: SegmentState,
        target_bpm: float,
        target_key: str,
    ) -> CriticFeedback:
        """Evaluate a segment directly without LLM.
        
        Useful for quick quality checks or when LLM is unavailable.
        
        Args:
            segment: The segment to evaluate.
            target_bpm: Expected BPM.
            target_key: Expected key.
            
        Returns:
            CriticFeedback with evaluation.
        """
        from src.tools.audio_analysis import analyze_bpm, analyze_key
        
        audio_path = segment.get("audio_path")
        if not audio_path:
            return CriticFeedback(
                approved=False,
                consistency_score=0.0,
                quality_score=0.0,
                energy_score=0.0,
                continuity_score=0.0,
                issues=["No audio file generated"],
                revision_suggestions=["Regenerate the segment"],
                notes="Segment has no audio path",
            )
        
        # Analyze actual values
        bpm_result = analyze_bpm(audio_path)
        key_result = analyze_key(audio_path)
        
        issues = []
        
        # Check BPM
        if bpm_result["success"]:
            actual_bpm = bpm_result["data"]["bpm"]
            bpm_diff = abs(actual_bpm - target_bpm)
            consistency_score = max(0.0, 1.0 - (bpm_diff / 20.0))  # 20 BPM diff = 0 score
            if bpm_diff > 10:
                issues.append(f"BPM mismatch: expected {target_bpm}, got {actual_bpm:.1f}")
        else:
            consistency_score = 0.5
            issues.append("Could not analyze BPM")
        
        # Check key
        if key_result["success"]:
            actual_key = key_result["data"]["key"]
            if actual_key.replace("m", "") != target_key.replace("m", ""):
                consistency_score *= 0.8
                issues.append(f"Key mismatch: expected {target_key}, got {actual_key}")
        
        # Default other scores (would need more analysis)
        quality_score = 0.7
        energy_score = 0.7
        continuity_score = 0.8
        
        avg_score = (consistency_score + quality_score + energy_score + continuity_score) / 4
        approved = avg_score >= self.approval_threshold and len(issues) <= 2
        
        return CriticFeedback(
            approved=approved,
            consistency_score=consistency_score,
            quality_score=quality_score,
            energy_score=energy_score,
            continuity_score=continuity_score,
            issues=issues,
            revision_suggestions=[
                f"Adjust BPM to {target_bpm}" if any("BPM" in i for i in issues) else "",
                f"Use {target_key} for harmonic consistency" if any("Key" in i for i in issues) else "",
            ],
            notes="Direct evaluation without LLM",
        )
