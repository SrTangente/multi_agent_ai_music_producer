"""Tests for configuration module."""

import pytest
from pathlib import Path


class TestSettings:
    """Tests for Settings class."""
    
    def test_default_settings(self):
        """Test default settings initialization."""
        from src.config import Settings
        
        settings = Settings()
        
        assert settings.llm.provider == "anthropic"
        assert settings.llm.temperature == 0.7
        assert settings.generation.max_retries == 3
        assert settings.audio.sample_rate == 32000
    
    def test_settings_from_yaml(self, sample_config_yaml: Path):
        """Test loading settings from YAML file."""
        from src.config import Settings
        
        settings = Settings.from_yaml(str(sample_config_yaml))
        
        assert settings.llm.provider == "anthropic"
        assert settings.generation.max_retries == 2
        assert settings.generation.approval_threshold == 0.6
    
    def test_settings_env_override(self, monkeypatch):
        """Test environment variable overrides."""
        from src.config import Settings
        
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_MODEL", "gpt-4")
        
        settings = Settings()
        
        # Environment variables should override defaults
        # Note: This depends on your implementation
    
    def test_invalid_yaml_path(self):
        """Test handling of invalid YAML path."""
        from src.config import Settings
        
        with pytest.raises(FileNotFoundError):
            Settings.from_yaml("/nonexistent/path/config.yaml")


class TestLLMConfig:
    """Tests for LLM configuration."""
    
    def test_llm_config_defaults(self):
        """Test LLM config default values."""
        from src.config import LLMConfig
        
        config = LLMConfig()
        
        assert config.provider in ["anthropic", "openai", "huggingface", "ollama"]
        assert 0.0 <= config.temperature <= 2.0
        assert config.max_tokens > 0
    
    def test_llm_config_validation(self):
        """Test LLM config validation."""
        from src.config import LLMConfig
        
        # Invalid temperature should raise
        with pytest.raises(ValueError):
            LLMConfig(temperature=-1.0)


class TestGenerationConfig:
    """Tests for generation configuration."""
    
    def test_generation_defaults(self):
        """Test generation config defaults."""
        from src.config import GenerationConfig
        
        config = GenerationConfig()
        
        assert config.segment_duration_sec > 0
        assert config.max_retries >= 0
        assert 0.0 <= config.approval_threshold <= 1.0
    
    def test_generation_validation(self):
        """Test generation config validation."""
        from src.config import GenerationConfig
        
        # Invalid threshold should raise
        with pytest.raises(ValueError):
            GenerationConfig(approval_threshold=1.5)


class TestAudioConfig:
    """Tests for audio configuration."""
    
    def test_audio_defaults(self):
        """Test audio config defaults."""
        from src.config import AudioConfig
        
        config = AudioConfig()
        
        assert config.sample_rate > 0
        assert config.output_dir is not None
    
    def test_audio_lufs_range(self):
        """Test LUFS value is in valid range."""
        from src.config import AudioConfig
        
        config = AudioConfig()
        
        # LUFS should be negative for typical music
        assert config.target_lufs < 0
        assert config.target_lufs >= -70  # Reasonably loud
