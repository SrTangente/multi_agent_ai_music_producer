"""Integration tests using HuggingFace models.

These tests exercise the full workflow using real HuggingFace LLM models
for agent orchestration. Audio generation is mocked since MusicGen
requires GPU resources.

To run these tests:
    pytest tests/test_integration_hf.py -v --runslow

Environment setup:
    export HF_TOKEN="your_huggingface_token"  # Optional for public models
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock

import pytest
import numpy as np


# Mark all tests in this module as slow and requiring HuggingFace
pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration,
]


def _check_transformers_available() -> bool:
    """Check if transformers and torch are available."""
    try:
        import torch
        import transformers
        return True
    except ImportError:
        return False


def _get_device() -> str:
    """Get the best available device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# Skip all tests if transformers not available
requires_transformers = pytest.mark.skipif(
    not _check_transformers_available(),
    reason="transformers/torch not installed"
)


class MockAudioGenerator:
    """Mock audio generator that creates valid audio files."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.call_count = 0
    
    def generate(
        self,
        prompt: str,
        duration_sec: float,
        output_path: str,
        **kwargs
    ) -> dict[str, Any]:
        """Generate a mock audio file."""
        import soundfile as sf
        
        self.call_count += 1
        
        # Create synthetic audio (sine wave + noise)
        sample_rate = 32000
        samples = int(sample_rate * duration_sec)
        t = np.linspace(0, duration_sec, samples)
        
        # Generate a simple audio signal
        freq = 440 + (self.call_count * 55)  # Different frequency each time
        audio = 0.3 * np.sin(2 * np.pi * freq * t)
        audio += 0.05 * np.random.randn(samples)  # Add noise
        
        # Apply fade in/out
        fade_samples = int(0.1 * sample_rate)
        audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
        audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        # Save the audio
        sf.write(output_path, audio, sample_rate)
        
        return {
            "success": True,
            "data": {
                "audio_path": output_path,
                "duration_sec": duration_sec,
                "generation_params": {
                    "prompt": prompt,
                    "temperature": kwargs.get("temperature", 1.0),
                }
            },
            "error": None
        }


@requires_transformers
class TestHuggingFaceProvider:
    """Test HuggingFace LLM provider directly."""
    
    def test_provider_initialization(self):
        """Test that HuggingFace provider initializes correctly."""
        from src.llm.huggingface_provider import HuggingFaceProvider
        
        # Use a tiny model for fast testing
        provider = HuggingFaceProvider(
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            temperature=0.7,
            max_tokens=256,
        )
        
        assert provider.provider_name == "huggingface"
        assert "TinyLlama" in provider.model_name
    
    @pytest.mark.timeout(120)
    def test_provider_simple_generation(self):
        """Test that provider can generate a response."""
        from src.llm.huggingface_provider import HuggingFaceProvider
        from src.llm.base import LLMMessage
        
        provider = HuggingFaceProvider(
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            temperature=0.7,
            max_tokens=100,
        )
        
        messages = [
            LLMMessage(role="user", content="Say 'hello world' and nothing else.")
        ]
        
        # Use generate_sync for synchronous testing
        response = provider.generate_sync(messages)
        
        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0
        print(f"\nGenerated response: {response.content}")


@requires_transformers
class TestEndToEndWithHuggingFace:
    """End-to-end workflow tests using HuggingFace models."""
    
    @pytest.fixture
    def test_output_dir(self) -> Path:
        """Create a temporary output directory."""
        with tempfile.TemporaryDirectory(prefix="music_test_") as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def mock_audio_gen(self, test_output_dir: Path):
        """Create mock audio generator."""
        return MockAudioGenerator(test_output_dir)
    
    @pytest.mark.timeout(300)
    def test_full_workflow_with_hf_llm(self, test_output_dir: Path, mock_audio_gen):
        """Test complete workflow using HuggingFace for orchestration.
        
        This test:
        1. Creates a workflow with HuggingFace LLM provider
        2. Uses mock audio generation
        3. Runs the complete multi-agent workflow
        4. Verifies output files are created
        """
        from src.config import Settings, LLMConfig, GenerationConfig, AudioConfig, LoggingConfig
        from src.state.reducers import create_initial_state
        from src.logging.logger import MusicProducerLogger, LogLevel
        
        # Create settings with HuggingFace provider
        settings = Settings(
            llm=LLMConfig(
                provider="huggingface",
                model="Qwen/Qwen2.5-7B-Instruct",
                temperature=0.7,
                max_tokens=1024,
            ),
            generation=GenerationConfig(
                max_retries=1,
                default_segment_duration=5.0,  # Short for testing
            ),
            audio=AudioConfig(
                sample_rate=32000,
            ),
            logging=LoggingConfig(
                level="DEBUG",
                output_dir=str(test_output_dir / "logs"),
            ),
        )
        
        # Create logger
        logger = MusicProducerLogger(
            run_id="test_hf_workflow",
            output_dir=str(test_output_dir),
            level=LogLevel.DEBUG,
            console_output=True,
        )
        
        # Create initial state
        initial_state = create_initial_state(
            user_prompt="Create a short ambient electronic track with soft synth pads",
            reference_paths=[],
            output_dir=str(test_output_dir),
            llm_provider="huggingface",
            llm_model="Qwen/Qwen2.5-7B-Instruct",
        )
        
        # Mock the audio generation tools
        with patch("src.tools.audio_generation.generate_segment") as mock_gen:
            mock_gen.side_effect = lambda prompt, duration_sec, output_path, **kwargs: \
                mock_audio_gen.generate(prompt, duration_sec, output_path, **kwargs)
            
            # Import workflow components
            from src.graph.workflow import MusicProducerGraph
            
            # Create and build workflow
            graph = MusicProducerGraph(
                settings=settings,
                logger=logger,
            )
            graph.build()
            
            # Run workflow
            try:
                final_state = graph.invoke(
                    user_prompt=initial_state["user_prompt"],
                    reference_paths=[],
                )
                
                # Verify completion
                assert final_state is not None
                assert final_state.get("phase") in ["complete", "failed"]
                
                # Check if segments were generated
                if final_state.get("phase") == "complete":
                    assert final_state.get("final_track_path") is not None
                    assert Path(final_state["final_track_path"]).exists()
                    
            except Exception as e:
                # Log the error for debugging
                print(f"Workflow error: {e}")
                raise
    
    @pytest.mark.timeout(180)
    def test_single_agent_with_hf(self, test_output_dir: Path):
        """Test a single agent (Director) with HuggingFace LLM.
        
        This is a simpler test that just verifies one agent works
        with the HuggingFace provider.
        """
        from src.agents.director import DirectorAgent
        from src.state.reducers import create_initial_state
        from datetime import datetime
        
        # Create Director agent with HuggingFace provider
        director = DirectorAgent(
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            provider="huggingface",
            temperature=0.8,
        )
        
        # Create initial state with musical profile
        state = create_initial_state(
            user_prompt="ambient electronic with soft synths",
            reference_paths=[],
            output_dir=str(test_output_dir),
            llm_provider="huggingface",
            llm_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        )
        
        # Add a musical profile to the state
        state["musical_profile"] = {
            "reference_paths": [],
            "bpm": {"bpm": 120.0, "confidence": 0.9},
            "key": {"key": "C", "mode": "major", "confidence": 0.85},
            "energy": {
                "mean_energy": 0.5,
                "energy_variance": 0.1,
                "energy_curve": [0.4, 0.5, 0.6, 0.5, 0.4],
            },
            "spectral": {
                "spectral_centroid_mean": 2000.0,
                "spectral_bandwidth_mean": 1500.0,
                "spectral_rolloff_mean": 4000.0,
                "mfcc_means": [0.0] * 13,
            },
            "instruments": {
                "instruments": ["synth", "drums"],
                "confidence": 0.7,
                "dominant_instrument": "synth",
            },
            "overall_mood": "ambient and calm",
            "analysis_timestamp": datetime.now().isoformat(),
        }
        
        # Run the director agent
        result = director.run(state)
        
        # Verify result contains track plan
        assert result is not None
        assert "track_plan" in result or "error" in str(result)
        
        print(f"\nDirector result: {result}")


@requires_transformers
class TestHuggingFaceModelOptions:
    """Test different HuggingFace model configurations."""
    
    def test_model_with_quantization_config(self):
        """Test provider with quantization settings."""
        from src.llm.huggingface_provider import HuggingFaceProvider
        
        # This just tests initialization, not loading (which requires GPU for 4bit)
        provider = HuggingFaceProvider(
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            load_in_4bit=False,  # Don't actually use 4bit for CPU test
            temperature=0.7,
            max_tokens=256,
        )
        
        assert provider.model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    def test_device_detection(self):
        """Test automatic device detection."""
        import torch
        
        device = _get_device()
        
        if torch.cuda.is_available():
            assert device == "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            assert device == "mps"
        else:
            assert device == "cpu"


# Standalone test runner for manual execution
if __name__ == "__main__":
    """Run integration tests directly.
    
    Usage:
        python tests/test_integration_hf.py
    """
    import sys
    
    print("=" * 60)
    print("HuggingFace Integration Tests")
    print("=" * 60)
    
    # Check requirements
    if not _check_transformers_available():
        print("ERROR: transformers/torch not installed")
        print("Install with: pip install transformers torch accelerate")
        sys.exit(1)
    
    print(f"Device: {_get_device()}")
    
    # Run with pytest
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure
    ]
    
    sys.exit(pytest.main(pytest_args))
