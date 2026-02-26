"""Main LangGraph workflow definition.

Defines the StateGraph with nodes and edges for the music producer.
"""

from __future__ import annotations

from typing import Any, TypedDict

from src.config import Settings
from src.logging.logger import MusicProducerLogger
from src.logging.llm_tracer import LLMTracer
from src.logging.progress import ProgressCallback, SilentProgressCallback
from src.state.schemas import MusicProducerState
from src.state.reducers import create_initial_state
from src.graph.nodes import (
    AnalysisNode,
    CriticNode,
    DirectorNode,
    MasteringNode,
    ProductionNode,
    RetrySegmentNode,
    SegmentCompleteNode,
)
from src.graph.routing import (
    route_after_critic,
    route_after_retry,
    route_after_segment_complete,
    route_initial,
)


class MusicProducerGraph:
    """LangGraph workflow for the AI Music Producer.
    
    Orchestrates the full pipeline:
    1. Analysis → Analyze reference tracks
    2. Director → Plan track structure
    3. Production Loop → Generate + Critique segments
    4. Mastering → Assemble final track
    """
    
    def __init__(
        self,
        settings: Settings | None = None,
        logger: MusicProducerLogger | None = None,
        tracer: LLMTracer | None = None,
        progress: ProgressCallback | None = None,
    ):
        """Initialize the workflow graph.
        
        Args:
            settings: Application settings.
            logger: Logger instance.
            tracer: LLM call tracer.
            progress: Progress callback.
        """
        self.settings = settings or Settings()
        self.logger = logger
        self.tracer = tracer
        self.progress = progress or SilentProgressCallback()
        
        self._graph = None
        self._compiled = None
    
    def _get_node_kwargs(self) -> dict[str, Any]:
        """Get common kwargs for node initialization."""
        return {
            "logger": self.logger,
            "tracer": self.tracer,
            "model": self.settings.llm.model,
            "provider": self.settings.llm.provider,
        }
    
    def build(self) -> "MusicProducerGraph":
        """Build the workflow graph.
        
        Returns:
            Self for chaining.
        """
        try:
            from langgraph.graph import StateGraph, END
        except ImportError:
            raise ImportError(
                "langgraph package required. Install with: pip install langgraph"
            )
        
        # Create the state graph
        builder = StateGraph(MusicProducerState)
        
        # Get node configuration
        node_kwargs = self._get_node_kwargs()
        
        # Add nodes
        builder.add_node("analysis", AnalysisNode(**node_kwargs))
        builder.add_node("director", DirectorNode(**node_kwargs))
        builder.add_node("production", ProductionNode(
            output_dir=self.settings.audio.output_dir,
            use_mock=False,  # Set True for testing without MusicGen
            **node_kwargs,
        ))
        builder.add_node("critic", CriticNode(
            approval_threshold=self.settings.generation.approval_threshold,
            **node_kwargs,
        ))
        builder.add_node("segment_complete", SegmentCompleteNode(
            max_retries=self.settings.generation.max_retries,
            **node_kwargs,
        ))
        builder.add_node("retry_segment", RetrySegmentNode(
            max_retries=self.settings.generation.max_retries,
            **node_kwargs,
        ))
        builder.add_node("mastering", MasteringNode(
            output_dir=self.settings.audio.output_dir,
            target_lufs=self.settings.audio.target_lufs,
            **node_kwargs,
        ))
        
        # Set entry point with conditional routing
        builder.set_conditional_entry_point(
            route_initial,
            {
                "analysis": "analysis",
                "director": "director",
            }
        )
        
        # Add edges
        builder.add_edge("analysis", "director")
        builder.add_edge("director", "production")
        
        # Production to Critic (always evaluate generated segment)
        builder.add_edge("production", "critic")
        
        # Critic routing: approve → complete, reject → retry
        builder.add_conditional_edges(
            "critic",
            route_after_critic,
            {
                "segment_complete": "segment_complete",
                "retry_segment": "retry_segment",
            }
        )
        
        # Retry routing
        builder.add_conditional_edges(
            "retry_segment",
            route_after_retry,
            {
                "production": "production",
                "segment_complete": "segment_complete",
            }
        )
        
        # Segment complete routing: more segments → production, done → mastering
        builder.add_conditional_edges(
            "segment_complete",
            route_after_segment_complete,
            {
                "production": "production",
                "mastering": "mastering",
            }
        )
        
        # Mastering ends the workflow
        builder.add_edge("mastering", END)
        
        self._graph = builder
        return self
    
    def compile(self, checkpointer=None) -> Any:
        """Compile the graph for execution.
        
        Args:
            checkpointer: Optional checkpoint saver for persistence.
            
        Returns:
            Compiled graph.
        """
        if self._graph is None:
            self.build()
        
        compile_kwargs = {}
        if checkpointer:
            compile_kwargs["checkpointer"] = checkpointer
        
        self._compiled = self._graph.compile(**compile_kwargs)
        return self._compiled
    
    def get_compiled(self) -> Any:
        """Get the compiled graph, building if needed."""
        if self._compiled is None:
            self.compile()
        return self._compiled
    
    def stream(
        self,
        user_prompt: str,
        reference_paths: list[str] | None = None,
        target_duration_sec: float = 120.0,
        config: dict[str, Any] | None = None,
    ):
        """Stream workflow execution with intermediate state updates.
        
        Args:
            user_prompt: User's music description.
            reference_paths: Optional paths to reference tracks.
            target_duration_sec: Target track duration.
            config: Additional configuration.
            
        Yields:
            State updates at each step.
        """
        compiled = self.get_compiled()
        
        # Create initial state
        initial_state = create_initial_state(
            user_prompt=user_prompt,
            reference_paths=reference_paths or [],
            output_dir=self.settings.audio.output_dir if self.settings else "output",
        )
        # Store target duration in state
        initial_state["target_duration_sec"] = target_duration_sec
        
        # Stream execution
        for event in compiled.stream(initial_state, config=config):
            # Emit progress
            if self.progress:
                node_name = list(event.keys())[0] if event else "unknown"
                self.progress.on_node_complete(node_name)
            
            yield event
    
    def invoke(
        self,
        user_prompt: str,
        reference_paths: list[str] | None = None,
        target_duration_sec: float = 120.0,
        config: dict[str, Any] | None = None,
    ) -> MusicProducerState:
        """Run the workflow to completion.
        
        Args:
            user_prompt: User's music description.
            reference_paths: Optional paths to reference tracks.
            target_duration_sec: Target track duration.
            config: Additional configuration.
            
        Returns:
            Final workflow state.
        """
        compiled = self.get_compiled()
        
        # Create initial state
        initial_state = create_initial_state(
            user_prompt=user_prompt,
            reference_paths=reference_paths or [],
            output_dir=self.settings.audio.output_dir if self.settings else "output",
        )
        # Store target duration in state
        initial_state["target_duration_sec"] = target_duration_sec
        
        # Run to completion
        final_state = compiled.invoke(initial_state, config=config)
        
        return final_state


def create_workflow(
    settings: Settings | None = None,
    logger: MusicProducerLogger | None = None,
    tracer: LLMTracer | None = None,
    progress: ProgressCallback | None = None,
) -> MusicProducerGraph:
    """Create and build the workflow graph.
    
    Args:
        settings: Application settings.
        logger: Logger instance.
        tracer: LLM tracer.
        progress: Progress callback.
        
    Returns:
        Built workflow graph.
    """
    graph = MusicProducerGraph(
        settings=settings,
        logger=logger,
        tracer=tracer,
        progress=progress,
    )
    return graph.build()


def run_workflow(
    user_prompt: str,
    reference_paths: list[str] | None = None,
    target_duration_sec: float = 120.0,
    settings: Settings | None = None,
    logger: MusicProducerLogger | None = None,
) -> MusicProducerState:
    """Run the full workflow synchronously.
    
    Convenience function for simple usage.
    
    Args:
        user_prompt: User's music description.
        reference_paths: Optional reference track paths.
        target_duration_sec: Target duration.
        settings: Settings (uses defaults if None).
        logger: Logger instance.
        
    Returns:
        Final workflow state with output path.
    """
    workflow = create_workflow(settings=settings, logger=logger)
    return workflow.invoke(
        user_prompt=user_prompt,
        reference_paths=reference_paths,
        target_duration_sec=target_duration_sec,
    )


async def run_workflow_async(
    user_prompt: str,
    reference_paths: list[str] | None = None,
    target_duration_sec: float = 120.0,
    settings: Settings | None = None,
    logger: MusicProducerLogger | None = None,
) -> MusicProducerState:
    """Run the full workflow asynchronously.
    
    Args:
        user_prompt: User's music description.
        reference_paths: Optional reference track paths.
        target_duration_sec: Target duration.
        settings: Settings (uses defaults if None).
        logger: Logger instance.
        
    Returns:
        Final workflow state with output path.
    """
    workflow = create_workflow(settings=settings, logger=logger)
    compiled = workflow.get_compiled()
    
    initial_state = create_initial_state(
        user_prompt=user_prompt,
        reference_paths=reference_paths or [],
        target_duration_sec=target_duration_sec,
    )
    
    # Use async invoke
    final_state = await compiled.ainvoke(initial_state)
    return final_state


def get_workflow_visualization(workflow: MusicProducerGraph) -> str:
    """Get a Mermaid diagram of the workflow.
    
    Args:
        workflow: The workflow graph.
        
    Returns:
        Mermaid diagram string.
    """
    compiled = workflow.get_compiled()
    
    try:
        return compiled.get_graph().draw_mermaid()
    except Exception:
        # Fallback to manual diagram
        return """
graph TD
    Start([Start]) --> RouteInitial{Has References?}
    RouteInitial -->|Yes| Analysis[Analysis Agent]
    RouteInitial -->|No| Director[Director Agent]
    Analysis --> Director
    Director --> Production[Production Agent]
    Production --> Critic[Critic Agent]
    Critic --> RouteApproval{Approved?}
    RouteApproval -->|Yes| SegmentComplete[Complete Segment]
    RouteApproval -->|No| Retry{Max Retries?}
    Retry -->|No| Production
    Retry -->|Yes| SegmentComplete
    SegmentComplete --> MoreSegments{More Segments?}
    MoreSegments -->|Yes| Production
    MoreSegments -->|No| Mastering[Mastering Agent]
    Mastering --> End([End])
"""
