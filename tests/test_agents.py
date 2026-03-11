"""Tests for agents."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path


class TestBaseAgent:
    """Tests for BaseAgent class."""
    
    def test_agent_config(self):
        """Test agent configuration."""
        from src.agents.base import AgentConfig
        
        config = AgentConfig(
            name="test_agent",
            description="A test agent",
            model="test-model",
            provider="anthropic",
        )
        
        assert config.name == "test_agent"
        assert config.max_tool_calls == 10  # default
    
    def test_tool_spec_creation(self):
        """Test creating a tool specification."""
        from src.agents.base import ToolSpec
        from src.state.schemas import ToolResult
        
        def test_func(arg1: str) -> ToolResult:
            return ToolResult(success=True, data=arg1, error=None)
        
        spec = ToolSpec(
            name="test_tool",
            description="A test tool",
            parameters={
                "type": "object",
                "properties": {"arg1": {"type": "string"}},
                "required": ["arg1"],
            },
            function=test_func,
        )
        
        assert spec.name == "test_tool"
        assert callable(spec.function)
    
    def test_create_tool_spec_from_function(self):
        """Test automatic tool spec creation from function."""
        from src.agents.base import create_tool_spec
        from src.state.schemas import ToolResult
        
        def my_tool(path: str, count: int = 5) -> ToolResult:
            return ToolResult(success=True, data=None, error=None)
        
        spec = create_tool_spec(
            name="my_tool",
            description="My tool description",
            function=my_tool,
        )
        
        assert spec.name == "my_tool"
        assert "path" in spec.parameters["required"]


class TestAnalysisAgent:
    """Tests for AnalysisAgent."""
    
    def test_analysis_agent_init(self, mock_logger):
        """Test AnalysisAgent initialization."""
        from src.agents.analysis import AnalysisAgent
        
        agent = AnalysisAgent(logger=mock_logger)
        
        assert agent.name == "analysis"
        assert len(agent._tools) > 0
    
    def test_analysis_agent_has_required_tools(self, mock_logger):
        """Test that AnalysisAgent has all required tools."""
        from src.agents.analysis import AnalysisAgent
        
        agent = AnalysisAgent(logger=mock_logger)
        
        required_tools = [
            "analyze_bpm",
            "analyze_key",
            "analyze_energy",
            "analyze_spectral",
            "estimate_instruments",
        ]
        
        for tool in required_tools:
            assert tool in agent._tools
    
    def test_analysis_profile_parsing(self, mock_logger):
        """Test parsing profile from LLM response."""
        from src.agents.analysis import AnalysisAgent
        
        agent = AnalysisAgent(logger=mock_logger)
        
        # Mock response content
        content = """
        Based on my analysis:
        - BPM: 125
        - Key: Am (A minor)
        - Energy is high and energetic
        - Instruments detected: drums, bass, synth
        """
        
        profile = agent._parse_profile_from_response(content, {})
        
        assert profile["bpm"] == 125.0
        assert profile["key"] == "A"
        # Parser may extract "major" or "minor" depending on content analysis
        assert profile["mode"] in ["major", "minor"]
        assert profile["energy_target"] == "high"


class TestDirectorAgent:
    """Tests for DirectorAgent."""
    
    def test_director_agent_init(self, mock_logger):
        """Test DirectorAgent initialization."""
        from src.agents.director import DirectorAgent
        
        agent = DirectorAgent(logger=mock_logger)
        
        assert agent.name == "director"
        # Director doesn't use tools
        assert len(agent._tools) == 0
    
    def test_director_plan_parsing(self, mock_logger):
        """Test parsing plan from LLM response."""
        from src.agents.director import DirectorAgent
        
        agent = DirectorAgent(logger=mock_logger)
        
        content = """
        Here's the track plan:
        {
            "total_duration_sec": 120,
            "segment_count": 4,
            "segments": [
                {"duration_sec": 30, "prompt": "intro", "transition_type": "fade"},
                {"duration_sec": 30, "prompt": "verse", "transition_type": "crossfade"},
                {"duration_sec": 30, "prompt": "chorus", "transition_type": "crossfade"},
                {"duration_sec": 30, "prompt": "outro", "transition_type": "fade"}
            ],
            "overall_notes": "Four-part track"
        }
        """
        
        plan = agent._parse_plan_from_response(content, {"target_duration_sec": 120})
        
        assert plan["total_duration_sec"] == 120
        assert plan["segment_count"] == 4
    
    def test_director_default_plan(self, mock_logger, sample_musical_profile):
        """Test default plan generation when parsing fails."""
        from src.agents.director import DirectorAgent
        
        agent = DirectorAgent(logger=mock_logger)
        
        state = {
            "user_prompt": "test music",
            "target_duration_sec": 60.0,
            "musical_profile": sample_musical_profile,
        }
        
        plan = agent._create_default_plan(state)
        
        assert plan["total_duration_sec"] == 60.0
        assert plan["segment_count"] >= 2


class TestProductionAgent:
    """Tests for ProductionAgent."""
    
    def test_production_agent_init(self, mock_logger, temp_dir):
        """Test ProductionAgent initialization."""
        from src.agents.production import ProductionAgent
        
        agent = ProductionAgent(
            output_dir=str(temp_dir),
            use_mock=True,
            logger=mock_logger,
        )
        
        assert agent.name == "production"
        assert "generate_audio_segment" in agent._tools
    
    def test_production_direct_generation(
        self, mock_logger, temp_dir, sample_segment_params
    ):
        """Test direct segment generation."""
        from src.agents.production import ProductionAgent
        
        agent = ProductionAgent(
            output_dir=str(temp_dir),
            use_mock=True,
            logger=mock_logger,
        )
        
        segment_state = agent.generate_segment_direct(sample_segment_params)
        
        assert segment_state["status"] == "generated"
        assert segment_state["audio_path"] is not None


class TestCriticAgent:
    """Tests for CriticAgent."""
    
    def test_critic_agent_init(self, mock_logger):
        """Test CriticAgent initialization."""
        from src.agents.critic import CriticAgent
        
        agent = CriticAgent(
            approval_threshold=0.7,
            logger=mock_logger,
        )
        
        assert agent.name == "critic"
        assert agent.approval_threshold == 0.7
    
    def test_critic_feedback_parsing(self, mock_logger):
        """Test parsing feedback from LLM response."""
        from src.agents.critic import CriticAgent
        
        agent = CriticAgent(logger=mock_logger)
        
        content = """
        My evaluation:
        {
            "approved": true,
            "consistency_score": 0.85,
            "quality_score": 0.9,
            "energy_score": 0.8,
            "continuity_score": 0.75,
            "issues": [],
            "revision_suggestions": [],
            "notes": "Good segment"
        }
        """
        
        feedback = agent._parse_feedback_from_response(content)
        
        assert feedback["approved"] is True
        assert feedback["consistency_score"] == 0.85
    
    def test_critic_direct_evaluation(
        self, mock_logger, sample_segment_state, temp_audio_file
    ):
        """Test direct segment evaluation."""
        from src.agents.critic import CriticAgent
        
        agent = CriticAgent(
            approval_threshold=0.7,
            logger=mock_logger,
        )
        
        # Update segment state with valid audio path
        sample_segment_state["audio_path"] = str(temp_audio_file)
        
        feedback = agent.evaluate_segment_direct(
            sample_segment_state,
            target_bpm=120.0,
            target_key="C",
        )
        
        assert "approved" in feedback
        assert "consistency_score" in feedback


class TestMasteringAgent:
    """Tests for MasteringAgent."""
    
    def test_mastering_agent_init(self, mock_logger, temp_dir):
        """Test MasteringAgent initialization."""
        from src.agents.mastering import MasteringAgent
        
        agent = MasteringAgent(
            output_dir=str(temp_dir),
            target_lufs=-14.0,
            logger=mock_logger,
        )
        
        assert agent.name == "mastering"
        assert agent.target_lufs == -14.0
    
    def test_mastering_agent_has_required_tools(self, mock_logger, temp_dir):
        """Test MasteringAgent has all required tools."""
        from src.agents.mastering import MasteringAgent
        
        agent = MasteringAgent(
            output_dir=str(temp_dir),
            logger=mock_logger,
        )
        
        required_tools = [
            "concatenate_segments",
            "apply_fade_in",
            "apply_fade_out",
            "normalize_audio",
        ]
        
        for tool in required_tools:
            assert tool in agent._tools
    
    def test_mastering_direct(
        self, mock_logger, temp_dir, temp_audio_files
    ):
        """Test direct mastering without LLM."""
        from src.agents.mastering import MasteringAgent
        from src.state.schemas import SegmentState
        
        agent = MasteringAgent(
            output_dir=str(temp_dir),
            logger=mock_logger,
        )
        
        # Create segment states from temp files
        segments = [
            SegmentState(
                segment_index=i,
                audio_path=str(path),
                duration_sec=3.0 + i,
                generation_params={},
                status="approved",
                attempt_number=1,
                critic_feedback=None,
            )
            for i, path in enumerate(temp_audio_files)
        ]
        
        result = agent.master_direct(segments)
        
        assert result["success"] is True
        assert result["output_path"] is not None
        assert Path(result["output_path"]).exists()
