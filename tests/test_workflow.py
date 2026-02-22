"""Tests for the LangGraph workflow."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestWorkflowRouting:
    """Test workflow routing functions."""
    
    def test_route_initial_with_references(self, initial_state):
        """Test routing to analysis when references exist."""
        from src.graph.routing import route_initial
        
        state = dict(initial_state)
        state["reference_paths"] = ["/path/to/ref.wav"]
        
        result = route_initial(state)
        
        assert result == "analysis"
    
    def test_route_initial_without_references(self, initial_state):
        """Test routing to director without references."""
        from src.graph.routing import route_initial
        
        state = dict(initial_state)
        state["reference_paths"] = []
        
        result = route_initial(state)
        
        assert result == "director"
    
    def test_route_after_critic_approved(self, state_with_plan):
        """Test routing after critic approves."""
        from src.graph.routing import route_after_critic
        
        state = dict(state_with_plan)
        state["current_segment"] = {
            "segment_index": 0,
            "status": "approved",
            "critic_feedback": {"approved": True},
        }
        
        result = route_after_critic(state)
        
        assert result == "segment_complete"
    
    def test_route_after_critic_rejected(self, state_with_plan):
        """Test routing after critic rejects."""
        from src.graph.routing import route_after_critic
        
        state = dict(state_with_plan)
        state["current_segment"] = {
            "segment_index": 0,
            "status": "needs_revision",
            "critic_feedback": {"approved": False},
        }
        
        result = route_after_critic(state)
        
        assert result == "retry_segment"
    
    def test_should_continue_segments(self, state_with_plan):
        """Test segment continuation check."""
        from src.graph.routing import should_continue_segments
        
        # More segments to process
        state = dict(state_with_plan)
        state["current_segment_index"] = 0
        
        assert should_continue_segments(state) == "production"
        
        # All segments done
        state["current_segment_index"] = 3
        
        assert should_continue_segments(state) == "mastering"
    
    def test_is_workflow_complete(self):
        """Test workflow completion check."""
        from src.graph.routing import is_workflow_complete
        
        # Not complete
        state = {"status": "producing", "final_output_path": None}
        assert not is_workflow_complete(state)
        
        # Complete
        state = {"status": "completed", "final_output_path": "/output/final.wav"}
        assert is_workflow_complete(state)


class TestWorkflowNodes:
    """Test workflow node classes."""
    
    def test_analysis_node_creation(self):
        """Test creating analysis node."""
        from src.graph.nodes import AnalysisNode
        
        node = AnalysisNode()
        
        assert node is not None
    
    def test_director_node_creation(self):
        """Test creating director node."""
        from src.graph.nodes import DirectorNode
        
        node = DirectorNode()
        
        assert node is not None
    
    def test_production_node_creation(self):
        """Test creating production node."""
        from src.graph.nodes import ProductionNode
        
        node = ProductionNode(use_mock=True)
        
        assert node.use_mock is True
    
    def test_segment_complete_node(self, state_with_plan):
        """Test segment completion node."""
        from src.graph.nodes import SegmentCompleteNode
        
        node = SegmentCompleteNode()
        
        state = dict(state_with_plan)
        state["current_segment"] = {
            "segment_index": 0,
            "audio_path": "/path/segment.wav",
            "status": "approved",
        }
        
        updates = node(state)
        
        assert len(updates["completed_segments"]) == 1
        assert updates["current_segment_index"] == 1
    
    def test_retry_segment_node_under_max(self, state_with_plan):
        """Test retry node under max retries."""
        from src.graph.nodes import RetrySegmentNode
        
        node = RetrySegmentNode(max_retries=3)
        
        state = dict(state_with_plan)
        state["current_segment"] = {
            "segment_index": 0,
            "attempt_number": 1,
            "status": "needs_revision",
        }
        state["attempt_history"] = []
        
        updates = node(state)
        
        # Should increment attempt number
        assert updates["current_segment"]["attempt_number"] == 2


class TestMusicProducerGraph:
    """Test the main workflow graph."""
    
    def test_graph_creation(self, mock_settings):
        """Test creating the workflow graph."""
        from src.graph.workflow import MusicProducerGraph
        
        graph = MusicProducerGraph(settings=mock_settings)
        
        assert graph is not None
    
    def test_graph_build(self, mock_settings):
        """Test building the workflow graph."""
        from src.graph.workflow import MusicProducerGraph
        
        graph = MusicProducerGraph(settings=mock_settings)
        graph.build()
        
        assert graph._graph is not None
    
    @pytest.mark.slow
    def test_graph_compile(self, mock_settings):
        """Test compiling the workflow graph."""
        pytest.importorskip("langgraph")
        
        from src.graph.workflow import MusicProducerGraph
        
        graph = MusicProducerGraph(settings=mock_settings)
        compiled = graph.build().compile()
        
        assert compiled is not None
    
    def test_create_workflow(self, mock_settings):
        """Test workflow factory function."""
        pytest.importorskip("langgraph")
        
        from src.graph.workflow import create_workflow
        
        workflow = create_workflow(settings=mock_settings)
        
        assert workflow is not None


class TestWorkflowVisualization:
    """Test workflow visualization."""
    
    def test_get_mermaid_diagram(self, mock_settings):
        """Test getting Mermaid diagram."""
        pytest.importorskip("langgraph")
        
        from src.graph.workflow import MusicProducerGraph, get_workflow_visualization
        
        graph = MusicProducerGraph(settings=mock_settings)
        graph.build()
        
        diagram = get_workflow_visualization(graph)
        
        assert "graph" in diagram or "Start" in diagram
        assert "Analysis" in diagram or "analysis" in diagram


class TestEndToEndWorkflow:
    """Test end-to-end workflow execution."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_workflow_mock_execution(self, mock_settings, temp_dir):
        """Test workflow execution with mocks."""
        pytest.importorskip("langgraph")
        
        from src.graph.workflow import MusicProducerGraph
        from src.graph.nodes import ProductionNode
        
        # Patch production to use mock
        with patch.object(ProductionNode, '__init__', lambda self, **kw: None):
            with patch.object(ProductionNode, '__call__', return_value={
                "current_segment": {
                    "segment_index": 0,
                    "audio_path": str(temp_dir / "mock.wav"),
                    "status": "generated",
                },
            }):
                mock_settings.audio.output_dir = str(temp_dir)
                
                graph = MusicProducerGraph(settings=mock_settings)
                graph.build()
        
        # Note: Full execution test would require more extensive mocking
        # This verifies the graph can be constructed
        assert graph._graph is not None
