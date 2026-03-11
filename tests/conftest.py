"""Pytest fixtures and configuration for the test suite.

Provides reusable fixtures for testing components in isolation.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pytest


# ============================================================================
# Environment Setup
# ============================================================================

@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")


# ============================================================================
# Temporary Directories
# ============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_audio_file(temp_dir: Path) -> Path:
    """Create a temporary audio file for testing."""
    try:
        import soundfile as sf
    except ImportError:
        pytest.skip("soundfile not installed")
    
    # Generate 5 seconds of noise
    sample_rate = 32000
    duration = 5.0
    samples = int(sample_rate * duration)
    audio = np.random.randn(samples) * 0.1
    
    path = temp_dir / "test_audio.wav"
    sf.write(str(path), audio, sample_rate)
    
    return path


@pytest.fixture
def temp_audio_files(temp_dir: Path) -> list[Path]:
    """Create multiple temporary audio files."""
    try:
        import soundfile as sf
    except ImportError:
        pytest.skip("soundfile not installed")
    
    paths = []
    sample_rate = 32000
    
    for i in range(3):
        duration = 3.0 + i
        samples = int(sample_rate * duration)
        audio = np.random.randn(samples) * 0.1
        
        path = temp_dir / f"test_audio_{i}.wav"
        sf.write(str(path), audio, sample_rate)
        paths.append(path)
    
    return paths


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def sample_config_yaml(temp_dir: Path) -> Path:
    """Create a sample configuration YAML file."""
    config_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  temperature: 0.7
  max_tokens: 4096

generation:
  default_segment_duration: 15.0
  max_retries: 2
  approval_threshold: 0.6

audio:
  sample_rate: 32000
  output_dir: output
  target_lufs: -14.0

logging:
  level: STANDARD
  output_dir: logs
"""
    path = temp_dir / "config.yaml"
    path.write_text(config_content)
    return path


@pytest.fixture
def sample_settings():
    """Create sample Settings instance."""
    from src.config import Settings
    return Settings()


# ============================================================================
# State Fixtures
# ============================================================================

@pytest.fixture
def sample_musical_profile() -> dict[str, Any]:
    """Create a sample musical profile."""
    from src.state.schemas import MusicalProfile
    
    return MusicalProfile(
        bpm=120.0,
        key="C",
        mode="major",
        energy_target="medium",
        style_descriptors=["electronic", "ambient"],
        instrument_suggestions=["synth", "drums", "bass"],
        reference_summary="Ambient electronic track",
    )


@pytest.fixture
def sample_track_plan() -> dict[str, Any]:
    """Create a sample track plan."""
    from src.state.schemas import TrackPlan
    
    return TrackPlan(
        total_duration_sec=60.0,
        segment_count=3,
        segment_durations=[20.0, 20.0, 20.0],
        segment_prompts=[
            "Intro: ambient synth pads, 120 BPM, C major",
            "Main: add drums and bass, building energy",
            "Outro: fade out, return to ambient",
        ],
        transitions=["fade", "crossfade", "fade"],
        overall_notes="Three-part ambient electronic track",
    )


@pytest.fixture
def sample_segment_params() -> dict[str, Any]:
    """Create sample segment parameters."""
    from src.state.schemas import SegmentParameters
    
    return SegmentParameters(
        segment_index=0,
        duration_sec=20.0,
        prompt="Ambient synth intro, 120 BPM, C major, low energy",
        conditioning_context=None,
        target_energy="low",
        transition_type="fade",
    )


@pytest.fixture
def sample_segment_state(temp_audio_file: Path) -> dict[str, Any]:
    """Create a sample segment state."""
    from src.state.schemas import SegmentState
    
    return SegmentState(
        segment_index=0,
        audio_path=str(temp_audio_file),
        duration_sec=5.0,
        generation_params={"prompt": "test prompt"},
        status="generated",
        attempt_number=1,
        critic_feedback=None,
    )


@pytest.fixture
def sample_initial_state(temp_dir: Path) -> dict[str, Any]:
    """Create a sample initial workflow state."""
    from src.state.reducers import create_initial_state
    
    return create_initial_state(
        user_prompt="ambient electronic music with synths",
        reference_paths=[],
        output_dir=str(temp_dir),
    )


# ============================================================================
# Mock Components
# ============================================================================

@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response."""
    from src.llm.base import LLMResponse
    
    return LLMResponse(
        content="This is a test response",
        tool_calls=[],
        usage={"input_tokens": 100, "output_tokens": 50},
        model="test-model",
    )


@pytest.fixture
def mock_tool_result():
    """Create a mock successful tool result."""
    from src.state.schemas import ToolResult
    
    return ToolResult(
        success=True,
        data={"test_key": "test_value"},
        error=None,
    )


@pytest.fixture
def mock_tool_error():
    """Create a mock failed tool result."""
    from src.state.schemas import ToolError, ToolResult
    
    return ToolResult(
        success=False,
        data=None,
        error=ToolError(
            code="TEST_ERROR",
            message="Test error message",
            recoverable=True,
            suggested_action="Retry the operation",
        ),
    )


# ============================================================================
# Logger Fixtures
# ============================================================================

@pytest.fixture
def mock_logger(temp_dir: Path):
    """Create a logger that writes to temp directory."""
    from src.logging.logger import LogLevel, MusicProducerLogger
    
    return MusicProducerLogger(
        run_id="test_run",
        output_dir=str(temp_dir),
        level=LogLevel.DEBUG,
        console_output=False,
    )


@pytest.fixture
def silent_progress():
    """Create a silent progress callback."""
    from src.logging.progress import SilentProgressCallback
    return SilentProgressCallback()


# ============================================================================
# Workflow Test Fixtures
# ============================================================================

@pytest.fixture
def initial_state(sample_initial_state) -> dict[str, Any]:
    """Alias for sample_initial_state for backward compatibility."""
    return sample_initial_state


@pytest.fixture
def state_with_plan(sample_initial_state, sample_track_plan) -> dict[str, Any]:
    """Create a state with a populated track plan."""
    state = dict(sample_initial_state)
    state["track_plan"] = sample_track_plan
    state["current_segment_index"] = 0
    state["segments"] = []
    # Populate segment_queue based on track plan (TrackPlan is a TypedDict)
    state["segment_queue"] = [
        {"index": i, "prompt": sample_track_plan["segment_prompts"][i]}
        for i in range(sample_track_plan["segment_count"])
    ]
    return state


@pytest.fixture
def mock_settings():
    """Create mock settings for workflow tests."""
    from src.config import Settings
    return Settings()


# ============================================================================
# Skip Markers
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: marks tests requiring GPU"
    )
    config.addinivalue_line(
        "markers", "requires_api: marks tests requiring API keys"
    )


def _has_gpu() -> bool:
    """Check if GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# Skip markers
slow = pytest.mark.slow
integration = pytest.mark.integration
requires_gpu = pytest.mark.skipif(
    not _has_gpu(),
    reason="Test requires GPU"
)
requires_api = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY") == "test-api-key",
    reason="Test requires real API key"
)
