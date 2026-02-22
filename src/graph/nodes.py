"""Node functions for the LangGraph workflow.

Each node wraps an agent and handles state transitions.
"""

from __future__ import annotations

from typing import Any, Callable

from src.agents import (
    AnalysisAgent,
    CriticAgent,
    DirectorAgent,
    MasteringAgent,
    ProductionAgent,
)
from src.logging.logger import MusicProducerLogger
from src.logging.llm_tracer import LLMTracer
from src.state.schemas import MusicProducerState, SegmentState
from src.state.reducers import create_segment_state, select_best_attempt


class NodeBase:
    """Base class for workflow nodes."""
    
    def __init__(
        self,
        logger: MusicProducerLogger | None = None,
        tracer: LLMTracer | None = None,
        model: str | None = None,
        provider: str | None = None,
    ):
        """Initialize the node.
        
        Args:
            logger: Logger instance.
            tracer: LLM tracer instance.
            model: LLM model override.
            provider: LLM provider override.
        """
        self.logger = logger
        self.tracer = tracer
        self.model = model
        self.provider = provider
    
    def _get_agent_kwargs(self) -> dict[str, Any]:
        """Get common kwargs for agent initialization."""
        kwargs: dict[str, Any] = {
            "logger": self.logger,
            "tracer": self.tracer,
        }
        if self.model:
            kwargs["model"] = self.model
        if self.provider:
            kwargs["provider"] = self.provider
        return kwargs


class AnalysisNode(NodeBase):
    """Node for the Analysis Agent.
    
    Analyzes reference tracks and produces a musical profile.
    """
    
    def __call__(self, state: MusicProducerState) -> dict[str, Any]:
        """Execute the analysis node.
        
        Args:
            state: Current workflow state.
            
        Returns:
            State updates with musical_profile.
        """
        if self.logger:
            self.logger.log_event(
                event_type="node_start",
                node="analysis",
            )
        
        agent = AnalysisAgent(**self._get_agent_kwargs())
        updates = agent.run(state)
        
        return updates


class DirectorNode(NodeBase):
    """Node for the Director Agent.
    
    Creates the track plan with segment breakdown.
    """
    
    def __call__(self, state: MusicProducerState) -> dict[str, Any]:
        """Execute the director node.
        
        Args:
            state: Current workflow state.
            
        Returns:
            State updates with track_plan and segment_queue.
        """
        if self.logger:
            self.logger.log_event(
                event_type="node_start",
                node="director",
            )
        
        agent = DirectorAgent(**self._get_agent_kwargs())
        updates = agent.run(state)
        
        # Initialize segment tracking
        updates["current_segment_index"] = 0
        updates["completed_segments"] = []
        updates["status"] = "producing"
        
        return updates


class ProductionNode(NodeBase):
    """Node for the Production Agent.
    
    Generates audio for the current segment.
    """
    
    def __init__(
        self,
        output_dir: str = "output/segments",
        use_mock: bool = False,
        **kwargs,
    ):
        """Initialize the production node.
        
        Args:
            output_dir: Directory for generated segments.
            use_mock: Use mock generation for testing.
            **kwargs: Parent class arguments.
        """
        super().__init__(**kwargs)
        self.output_dir = output_dir
        self.use_mock = use_mock
    
    def __call__(self, state: MusicProducerState) -> dict[str, Any]:
        """Execute the production node.
        
        Args:
            state: Current workflow state.
            
        Returns:
            State updates with current_segment.
        """
        if self.logger:
            self.logger.log_event(
                event_type="node_start",
                node="production",
                segment_index=state.get("current_segment_index", 0),
            )
        
        agent = ProductionAgent(
            output_dir=self.output_dir,
            use_mock=self.use_mock,
            **self._get_agent_kwargs(),
        )
        
        updates = agent.run(state)
        
        return updates


class CriticNode(NodeBase):
    """Node for the Critic Agent.
    
    Evaluates the current segment and provides feedback.
    """
    
    def __init__(
        self,
        approval_threshold: float = 0.7,
        **kwargs,
    ):
        """Initialize the critic node.
        
        Args:
            approval_threshold: Score threshold for approval.
            **kwargs: Parent class arguments.
        """
        super().__init__(**kwargs)
        self.approval_threshold = approval_threshold
    
    def __call__(self, state: MusicProducerState) -> dict[str, Any]:
        """Execute the critic node.
        
        Args:
            state: Current workflow state.
            
        Returns:
            State updates with critic_feedback on current_segment.
        """
        if self.logger:
            self.logger.log_event(
                event_type="node_start",
                node="critic",
                segment_index=state.get("current_segment_index", 0),
            )
        
        agent = CriticAgent(
            approval_threshold=self.approval_threshold,
            **self._get_agent_kwargs(),
        )
        
        updates = agent.run(state)
        
        return updates


class MasteringNode(NodeBase):
    """Node for the Mastering Agent.
    
    Assembles all segments into the final track.
    """
    
    def __init__(
        self,
        output_dir: str = "output",
        target_lufs: float = -14.0,
        **kwargs,
    ):
        """Initialize the mastering node.
        
        Args:
            output_dir: Directory for final output.
            target_lufs: Target loudness.
            **kwargs: Parent class arguments.
        """
        super().__init__(**kwargs)
        self.output_dir = output_dir
        self.target_lufs = target_lufs
    
    def __call__(self, state: MusicProducerState) -> dict[str, Any]:
        """Execute the mastering node.
        
        Args:
            state: Current workflow state.
            
        Returns:
            State updates with final_output_path.
        """
        if self.logger:
            self.logger.log_event(
                event_type="node_start",
                node="mastering",
            )
        
        agent = MasteringAgent(
            output_dir=self.output_dir,
            target_lufs=self.target_lufs,
            **self._get_agent_kwargs(),
        )
        
        updates = agent.run(state)
        
        return updates


class SegmentCompleteNode(NodeBase):
    """Node to finalize a segment and advance to next.
    
    Handles segment approval/retry logic and queue management.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        **kwargs,
    ):
        """Initialize segment complete node.
        
        Args:
            max_retries: Maximum retry attempts per segment.
            **kwargs: Parent class arguments.
        """
        super().__init__(**kwargs)
        self.max_retries = max_retries
    
    def __call__(self, state: MusicProducerState) -> dict[str, Any]:
        """Process segment completion.
        
        Args:
            state: Current workflow state.
            
        Returns:
            State updates advancing to next segment.
        """
        current_segment = state.get("current_segment", {})
        current_index = state.get("current_segment_index", 0)
        segment_queue = state.get("segment_queue", [])
        completed = list(state.get("completed_segments", []))
        
        # Add current segment to completed
        if current_segment:
            completed.append(current_segment)
        
        # Move to next segment
        next_index = current_index + 1
        
        # Check if all segments done
        if next_index >= len(segment_queue):
            return {
                "completed_segments": completed,
                "current_segment_index": next_index,
                "current_segment": None,
                "status": "mastering",
            }
        else:
            return {
                "completed_segments": completed,
                "current_segment_index": next_index,
                "current_segment": None,
                "status": "producing",
            }


class RetrySegmentNode(NodeBase):
    """Node to handle segment retry after critic rejection.
    
    Updates attempt counter and prepares for regeneration.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        **kwargs,
    ):
        """Initialize retry node.
        
        Args:
            max_retries: Maximum retry attempts.
            **kwargs: Parent class arguments.
        """
        super().__init__(**kwargs)
        self.max_retries = max_retries
    
    def __call__(self, state: MusicProducerState) -> dict[str, Any]:
        """Prepare segment for retry.
        
        Args:
            state: Current workflow state.
            
        Returns:
            State updates for retry.
        """
        current_segment = state.get("current_segment", {})
        current_index = state.get("current_segment_index", 0)
        attempt_history = list(state.get("attempt_history", []))
        
        # Record this attempt
        if current_segment:
            attempt_record = {
                "segment_index": current_index,
                "attempt_number": current_segment.get("attempt_number", 1),
                "audio_path": current_segment.get("audio_path"),
                "feedback": current_segment.get("critic_feedback"),
                "timestamp": "",  # Will be set
            }
            attempt_history.append(attempt_record)
        
        # Check if max retries reached
        current_attempt = current_segment.get("attempt_number", 1)
        
        if current_attempt >= self.max_retries:
            if self.logger:
                self.logger.log_event(
                    event_type="max_retries_reached",
                    segment_index=current_index,
                    attempts=current_attempt,
                )
            
            # Use best attempt and move on
            best = select_best_attempt(attempt_history, current_index)
            
            completed = list(state.get("completed_segments", []))
            if best:
                completed.append(best)
            
            return {
                "attempt_history": attempt_history,
                "completed_segments": completed,
                "current_segment_index": current_index + 1,
                "current_segment": None,
                "status": "producing",
            }
        else:
            # Prepare for retry
            return {
                "attempt_history": attempt_history,
                "current_segment": create_segment_state(
                    segment_index=current_index,
                    attempt_number=current_attempt + 1,
                ),
                "status": "producing",
            }


def create_analysis_node(**kwargs) -> Callable[[MusicProducerState], dict[str, Any]]:
    """Factory for analysis node."""
    return AnalysisNode(**kwargs)


def create_director_node(**kwargs) -> Callable[[MusicProducerState], dict[str, Any]]:
    """Factory for director node."""
    return DirectorNode(**kwargs)


def create_production_node(**kwargs) -> Callable[[MusicProducerState], dict[str, Any]]:
    """Factory for production node."""
    return ProductionNode(**kwargs)


def create_critic_node(**kwargs) -> Callable[[MusicProducerState], dict[str, Any]]:
    """Factory for critic node."""
    return CriticNode(**kwargs)


def create_mastering_node(**kwargs) -> Callable[[MusicProducerState], dict[str, Any]]:
    """Factory for mastering node."""
    return MasteringNode(**kwargs)
