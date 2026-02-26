"""Tests for state schemas and reducers."""

import pytest
from typing import Any


class TestMusicalProfile:
    """Tests for MusicalProfile schema."""
    
    def test_minimal_profile(self):
        """Test creating a minimal profile."""
        from src.state.schemas import MusicalProfile
        
        profile = MusicalProfile(
            bpm=120.0,
            key="C",
            mode="major",
            energy_target="medium",
            style_descriptors=[],
            instrument_suggestions=[],
            reference_summary="",
        )
        
        assert profile["bpm"] == 120.0
        assert profile["key"] == "C"
    
    def test_full_profile(self, sample_musical_profile):
        """Test full profile with all fields."""
        profile = sample_musical_profile
        
        assert profile["bpm"] == 120.0
        assert profile["mode"] == "major"
        assert "synth" in profile["instrument_suggestions"]


class TestTrackPlan:
    """Tests for TrackPlan schema."""
    
    def test_plan_consistency(self, sample_track_plan):
        """Test track plan internal consistency."""
        plan = sample_track_plan
        
        # Segment count should match list lengths
        assert plan["segment_count"] == len(plan["segment_durations"])
        assert plan["segment_count"] == len(plan["segment_prompts"])
        
        # Durations should sum to total (approximately)
        total = sum(plan["segment_durations"])
        assert abs(total - plan["total_duration_sec"]) < 0.01


class TestSegmentState:
    """Tests for SegmentState schema."""
    
    def test_segment_state_created(self, sample_segment_state):
        """Test segment state creation."""
        state = sample_segment_state
        
        assert state["segment_index"] == 0
        assert state["status"] == "generated"
        assert state["attempt_number"] == 1
    
    def test_segment_state_status_values(self):
        """Test valid status values."""
        from src.state.schemas import SegmentState
        
        valid_statuses = ["pending", "generating", "generated", "approved", "needs_revision", "failed"]
        
        for status in valid_statuses:
            state = SegmentState(
                segment_index=0,
                audio_path=None,
                duration_sec=0.0,
                generation_params={},
                status=status,
                attempt_number=1,
                critic_feedback=None,
            )
            assert state["status"] == status


class TestCriticFeedback:
    """Tests for CriticFeedback schema."""
    
    def test_feedback_scores_range(self):
        """Test that feedback scores are in valid range."""
        from src.state.schemas import CriticFeedback
        
        feedback = CriticFeedback(
            approved=True,
            consistency_score=0.8,
            quality_score=0.9,
            energy_score=0.7,
            continuity_score=0.85,
            issues=[],
            revision_suggestions=[],
            notes="Good segment",
        )
        
        scores = [
            feedback["consistency_score"],
            feedback["quality_score"],
            feedback["energy_score"],
            feedback["continuity_score"],
        ]
        
        for score in scores:
            assert 0.0 <= score <= 1.0
    
    def test_feedback_with_issues(self):
        """Test feedback with issues and suggestions."""
        from src.state.schemas import CriticFeedback
        
        feedback = CriticFeedback(
            approved=False,
            consistency_score=0.5,
            quality_score=0.4,
            energy_score=0.6,
            continuity_score=0.3,
            issues=["BPM mismatch", "Audio artifacts"],
            revision_suggestions=["Adjust tempo", "Regenerate"],
            notes="Needs improvement",
        )
        
        assert not feedback["approved"]
        assert len(feedback["issues"]) == 2


class TestToolResult:
    """Tests for ToolResult schema."""
    
    def test_successful_result(self, mock_tool_result):
        """Test successful tool result."""
        result = mock_tool_result
        
        assert result["success"] is True
        assert result["data"] is not None
        assert result["error"] is None
    
    def test_failed_result(self, mock_tool_error):
        """Test failed tool result."""
        result = mock_tool_error
        
        assert result["success"] is False
        assert result["data"] is None
        assert result["error"] is not None
        assert result["error"]["code"] == "TEST_ERROR"


class TestReducers:
    """Tests for state reducer functions."""
    
    def test_create_initial_state(self, temp_dir):
        """Test initial state creation."""
        from src.state.reducers import create_initial_state
        
        state = create_initial_state(
            user_prompt="test prompt",
            reference_paths=["ref1.wav", "ref2.wav"],
            output_dir=str(temp_dir),
        )
        
        assert state["user_prompt"] == "test prompt"
        assert len(state["reference_paths"]) == 2
        assert state["phase"] == "initialized"
        assert state["run_id"] is not None
    
    def test_create_segment_state(self):
        """Test segment state creation."""
        from src.state.reducers import create_segment_state
        
        segment = create_segment_state(
            segment_id="seg_001",
            parameters={"segment_index": 1, "duration_sec": 15.0},
        )
        
        assert segment["segment_id"] == "seg_001"
        assert segment["parameters"]["segment_index"] == 1
        assert segment["status"] == "pending"
        assert segment["attempts"] == []
    
    def test_select_best_attempt(self):
        """Test selecting best attempt from history."""
        from src.state.reducers import select_best_attempt, create_segment_state
        from datetime import datetime
        
        # Create segment state with attempts
        segment = create_segment_state(
            segment_id="seg_001",
            parameters={"segment_index": 0, "duration_sec": 15.0},
        )
        
        # Add attempts with critic_feedback containing overall_score
        segment["attempts"] = [
            {
                "attempt_number": 1,
                "audio_path": "/path/attempt_1.wav",
                "timestamp": datetime.now().isoformat(),
                "generation_params": {},
                "critic_feedback": {
                    "approved": False,
                    "overall_score": 0.5,
                    "prompt_alignment": 0.5,
                    "director_compliance": 0.5,
                    "continuity_score": 0.5,
                    "technical_quality": 0.5,
                    "issues": [],
                    "suggestions": [],
                    "better_than_previous": False,
                    "evaluation_timestamp": datetime.now().isoformat(),
                },
                "approved": False,
            },
            {
                "attempt_number": 2,
                "audio_path": "/path/attempt_2.wav",
                "timestamp": datetime.now().isoformat(),
                "generation_params": {},
                "critic_feedback": {
                    "approved": False,
                    "overall_score": 0.8,
                    "prompt_alignment": 0.8,
                    "director_compliance": 0.8,
                    "continuity_score": 0.8,
                    "technical_quality": 0.8,
                    "issues": [],
                    "suggestions": [],
                    "better_than_previous": True,
                    "evaluation_timestamp": datetime.now().isoformat(),
                },
                "approved": False,
            },
        ]
        
        best_idx = select_best_attempt(segment)
        
        # Should select attempt 2 (index 1) with higher overall_score
        assert best_idx is not None
        assert best_idx == 1


class TestMusicProducerState:
    """Tests for the full MusicProducerState."""
    
    def test_initial_state_structure(self, sample_initial_state):
        """Test initial state has all required fields."""
        state = sample_initial_state
        
        required_fields = [
            "user_prompt",
            "reference_paths",
            "output_dir",
            "phase",
            "run_id",
        ]
        
        for field in required_fields:
            assert field in state
    
    def test_state_progression(self, sample_initial_state, sample_musical_profile, sample_track_plan):
        """Test state can be updated through workflow stages."""
        state = dict(sample_initial_state)
        
        # After analysis
        state["musical_profile"] = sample_musical_profile
        state["status"] = "analyzed"
        
        assert state["musical_profile"]["bpm"] == 120.0
        
        # After planning
        state["track_plan"] = sample_track_plan
        state["status"] = "planned"
        
        assert state["track_plan"]["segment_count"] == 3
