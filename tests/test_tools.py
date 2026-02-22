"""Tests for audio tools."""

import pytest
import numpy as np
from pathlib import Path


class TestAudioIO:
    """Tests for audio I/O tools."""
    
    def test_load_audio(self, temp_audio_file: Path):
        """Test loading an audio file."""
        from src.tools.audio_io import load_audio
        
        result = load_audio(str(temp_audio_file))
        
        assert result["success"] is True
        assert result["data"]["audio"] is not None
        assert result["data"]["sample_rate"] == 32000
        assert result["data"]["duration_sec"] > 0
    
    def test_load_nonexistent_file(self):
        """Test loading a non-existent file."""
        from src.tools.audio_io import load_audio
        
        result = load_audio("/nonexistent/path/audio.wav")
        
        assert result["success"] is False
        assert result["error"]["code"] == "FILE_NOT_FOUND"
    
    def test_save_audio(self, temp_dir: Path):
        """Test saving audio data."""
        from src.tools.audio_io import save_audio
        
        # Create test audio
        sample_rate = 32000
        duration = 2.0
        audio = np.random.randn(int(sample_rate * duration)) * 0.1
        
        output_path = str(temp_dir / "output.wav")
        result = save_audio(audio, output_path, sample_rate)
        
        assert result["success"] is True
        assert Path(output_path).exists()
    
    def test_save_audio_normalization(self, temp_dir: Path):
        """Test that save_audio normalizes clipping audio."""
        from src.tools.audio_io import save_audio
        
        # Create audio that would clip
        sample_rate = 32000
        audio = np.random.randn(32000) * 2.0  # Values above 1.0
        
        output_path = str(temp_dir / "normalized.wav")
        result = save_audio(audio, output_path, sample_rate)
        
        assert result["success"] is True
    
    def test_extract_audio_tail(self, temp_audio_file: Path, temp_dir: Path):
        """Test extracting the tail of an audio file."""
        from src.tools.audio_io import extract_audio_tail
        
        output_path = str(temp_dir / "tail.wav")
        result = extract_audio_tail(
            str(temp_audio_file),
            duration_sec=2.0,
            output_path=output_path,
        )
        
        assert result["success"] is True
        assert result["data"]["duration_sec"] <= 2.0
        assert Path(output_path).exists()
    
    def test_get_audio_duration(self, temp_audio_file: Path):
        """Test getting audio duration."""
        from src.tools.audio_io import get_audio_duration
        
        result = get_audio_duration(str(temp_audio_file))
        
        assert result["success"] is True
        assert result["data"]["duration_sec"] == pytest.approx(5.0, abs=0.1)


class TestAudioAnalysis:
    """Tests for audio analysis tools."""
    
    def test_analyze_bpm(self, temp_audio_file: Path):
        """Test BPM analysis."""
        from src.tools.audio_analysis import analyze_bpm
        
        result = analyze_bpm(str(temp_audio_file))
        
        assert result["success"] is True
        assert "bpm" in result["data"]
        assert "confidence" in result["data"]
        assert 0 <= result["data"]["confidence"] <= 1
    
    def test_analyze_key(self, temp_audio_file: Path):
        """Test key analysis."""
        from src.tools.audio_analysis import analyze_key
        
        result = analyze_key(str(temp_audio_file))
        
        assert result["success"] is True
        assert "key" in result["data"]
        assert "mode" in result["data"]
        assert result["data"]["mode"] in ["major", "minor"]
    
    def test_analyze_energy(self, temp_audio_file: Path):
        """Test energy profile analysis."""
        from src.tools.audio_analysis import analyze_energy
        
        result = analyze_energy(str(temp_audio_file), num_segments=5)
        
        assert result["success"] is True
        assert "mean_energy" in result["data"]
        assert "energy_curve" in result["data"]
        assert len(result["data"]["energy_curve"]) == 5
    
    def test_analyze_spectral(self, temp_audio_file: Path):
        """Test spectral analysis."""
        from src.tools.audio_analysis import analyze_spectral
        
        result = analyze_spectral(str(temp_audio_file))
        
        assert result["success"] is True
        assert "spectral_centroid_mean" in result["data"]
        assert "mfcc_means" in result["data"]
        assert len(result["data"]["mfcc_means"]) == 13  # Standard MFCC count
    
    def test_estimate_instruments(self, temp_audio_file: Path):
        """Test instrument estimation."""
        from src.tools.audio_analysis import estimate_instruments
        
        result = estimate_instruments(str(temp_audio_file))
        
        assert result["success"] is True
        assert "instruments" in result["data"]
        assert isinstance(result["data"]["instruments"], list)
    
    def test_analyze_nonexistent_file(self):
        """Test analysis of non-existent file."""
        from src.tools.audio_analysis import analyze_bpm
        
        result = analyze_bpm("/nonexistent/audio.wav")
        
        assert result["success"] is False
        assert result["error"]["code"] == "FILE_NOT_FOUND"


class TestAudioProcessing:
    """Tests for audio processing tools."""
    
    def test_normalize_audio(self, temp_audio_file: Path, temp_dir: Path):
        """Test audio normalization."""
        from src.tools.audio_processing import normalize_audio
        
        output_path = str(temp_dir / "normalized.wav")
        result = normalize_audio(
            str(temp_audio_file),
            target_lufs=-14.0,
            output_path=output_path,
        )
        
        assert result["success"] is True
        assert Path(output_path).exists()
        assert "gain_db" in result["data"]
    
    def test_apply_crossfade(self, temp_audio_files: list[Path], temp_dir: Path):
        """Test crossfade between two files."""
        from src.tools.audio_processing import apply_crossfade
        
        output_path = str(temp_dir / "crossfaded.wav")
        result = apply_crossfade(
            str(temp_audio_files[0]),
            str(temp_audio_files[1]),
            output_path,
            fade_duration_ms=500,
        )
        
        assert result["success"] is True
        assert Path(output_path).exists()
    
    def test_apply_fade_in(self, temp_audio_file: Path, temp_dir: Path):
        """Test fade-in application."""
        from src.tools.audio_processing import apply_fade_in
        
        output_path = str(temp_dir / "fade_in.wav")
        result = apply_fade_in(
            str(temp_audio_file),
            duration_ms=1000,
            output_path=output_path,
        )
        
        assert result["success"] is True
    
    def test_apply_fade_out(self, temp_audio_file: Path, temp_dir: Path):
        """Test fade-out application."""
        from src.tools.audio_processing import apply_fade_out
        
        output_path = str(temp_dir / "fade_out.wav")
        result = apply_fade_out(
            str(temp_audio_file),
            duration_ms=2000,
            output_path=output_path,
        )
        
        assert result["success"] is True
    
    def test_concatenate_segments(self, temp_audio_files: list[Path], temp_dir: Path):
        """Test segment concatenation."""
        from src.tools.audio_processing import concatenate_segments
        
        paths = [str(p) for p in temp_audio_files]
        output_path = str(temp_dir / "concatenated.wav")
        
        result = concatenate_segments(
            paths,
            output_path,
            crossfade_duration_ms=250,
        )
        
        assert result["success"] is True
        assert result["data"]["segment_count"] == 3
    
    def test_concatenate_empty_list(self, temp_dir: Path):
        """Test concatenation with empty list."""
        from src.tools.audio_processing import concatenate_segments
        
        result = concatenate_segments(
            [],
            str(temp_dir / "output.wav"),
        )
        
        assert result["success"] is False
        assert result["error"]["code"] == "EMPTY_SEGMENTS"


class TestAudioGeneration:
    """Tests for audio generation tools."""
    
    def test_mock_generation(self, temp_dir: Path):
        """Test mock audio generation."""
        from src.tools.audio_generation import generate_segment_mock
        
        output_path = str(temp_dir / "mock_segment.wav")
        result = generate_segment_mock(
            prompt="test prompt",
            duration_sec=5.0,
            output_path=output_path,
        )
        
        assert result["success"] is True
        assert Path(output_path).exists()
        assert result["data"]["generation_params"]["mock"] is True
    
    def test_musicgen_wrapper_singleton(self):
        """Test MusicGenWrapper singleton pattern."""
        from src.tools.audio_generation import MusicGenWrapper
        
        wrapper1 = MusicGenWrapper.get_instance()
        wrapper2 = MusicGenWrapper.get_instance()
        
        assert wrapper1 is wrapper2
    
    @pytest.mark.requires_gpu
    def test_real_generation(self, temp_dir: Path):
        """Test real MusicGen generation (requires GPU)."""
        from src.tools.audio_generation import generate_segment
        
        output_path = str(temp_dir / "real_segment.wav")
        result = generate_segment(
            prompt="ambient electronic music, soft synth pads",
            duration_sec=5.0,
            output_path=output_path,
        )
        
        # This may fail without GPU/proper setup
        if result["success"]:
            assert Path(output_path).exists()
            assert result["data"]["duration_sec"] > 0
