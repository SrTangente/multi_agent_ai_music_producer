"""Configuration management with Pydantic validation.

Loads settings from YAML config file and environment variables.
Secrets (API keys) come from environment, everything else from YAML.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(str, Enum):
    """Logging verbosity levels."""
    MINIMAL = "MINIMAL"
    STANDARD = "STANDARD"
    VERBOSE = "VERBOSE"
    DEBUG = "DEBUG"


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    provider: Literal["anthropic", "openai", "huggingface", "ollama"] = "anthropic"
    model: str = "claude-sonnet-4-6"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    base_url: str | None = None
    torch_dtype: Literal["float16", "bfloat16", "float32"] | None = None  # For HuggingFace TPU


class GenerationConfig(BaseModel):
    """Audio generation configuration."""
    max_retries: int = Field(default=3, ge=1, le=10)
    conditioning_tail_seconds: float = Field(default=5.0, ge=1.0, le=30.0)
    default_segment_duration: float = Field(default=15.0, ge=5.0, le=60.0)
    musicgen_model: str = "facebook/musicgen-melody"
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_k: int = Field(default=250, ge=1)
    cfg_coef: float = Field(default=3.0, ge=1.0, le=10.0)
    retry_temperature_decay: float = Field(default=0.2, ge=0.0, le=0.5)
    retry_cfg_increase: float = Field(default=1.0, ge=0.0, le=3.0)
    approval_threshold: float = Field(default=0.6, ge=0.0, le=1.0)


class AudioConfig(BaseModel):
    """Audio processing configuration."""
    sample_rate: int = Field(default=32000, ge=16000, le=48000)
    output_format: Literal["wav", "mp3", "flac"] = "wav"
    output_dir: str = Field(default="output")
    crossfade_duration_ms: int = Field(default=500, ge=0, le=5000)
    fade_in_duration_ms: int = Field(default=1000, ge=0, le=10000)
    fade_out_duration_ms: int = Field(default=2000, ge=0, le=10000)
    target_lufs: float = Field(default=-14.0, ge=-30.0, le=0.0)


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: LogLevel = LogLevel.STANDARD
    output_dir: str = "output/runs"


class ContinuityConfig(BaseModel):
    """Continuity and conditioning configuration."""
    tail_seconds: float = Field(default=5.0, ge=1.0, le=30.0)
    bpm_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    key_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    energy_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    mood_weight: float = Field(default=0.2, ge=0.0, le=1.0)

    @field_validator("mood_weight")
    @classmethod
    def weights_must_sum_to_one(cls, v: float, info) -> float:
        """Validate that all weights sum to approximately 1.0."""
        data = info.data
        total = data.get("bpm_weight", 0.3) + data.get("key_weight", 0.3) + \
                data.get("energy_weight", 0.2) + v
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Continuity weights must sum to 1.0, got {total}")
        return v


class Settings(BaseSettings):
    """Main settings loaded from environment and YAML config."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # API keys from environment
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    hf_token: str | None = None
    ollama_base_url: str | None = None
    
    # Loaded from YAML
    llm: LLMConfig = Field(default_factory=LLMConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    continuity: ContinuityConfig = Field(default_factory=ContinuityConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str | Path | None = None) -> "Settings":
        """Load settings from YAML file, with environment variable overrides.
        
        Args:
            config_path: Path to YAML config file. If None, uses default location.
            
        Returns:
            Validated Settings instance.
        """
        if config_path is None:
            # Find config relative to this file or workspace root
            possible_paths = [
                Path(__file__).parent.parent / "config" / "settings.yaml",
                Path("config/settings.yaml"),
                Path("settings.yaml"),
            ]
            for path in possible_paths:
                if path.exists():
                    config_path = path
                    break
        
        yaml_config = {}
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                yaml_config = yaml.safe_load(f) or {}
        
        # Merge YAML config with environment variables
        return cls(**yaml_config)
    
    def get_api_key_for_provider(self) -> str | None:
        """Get the appropriate API key for the configured LLM provider."""
        provider_to_key = {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "huggingface": self.hf_token,
            "ollama": None,  # Ollama doesn't need API key
        }
        return provider_to_key.get(self.llm.provider)
    
    def validate_provider_config(self) -> list[str]:
        """Validate that the configured provider has required credentials.
        
        Returns:
            List of validation errors (empty if valid).
        """
        errors = []
        provider = self.llm.provider
        
        if provider == "anthropic" and not self.anthropic_api_key:
            errors.append("ANTHROPIC_API_KEY required for Anthropic provider")
        elif provider == "openai" and not self.openai_api_key:
            errors.append("OPENAI_API_KEY required for OpenAI provider")
        elif provider == "huggingface" and not self.hf_token:
            errors.append("HF_TOKEN required for HuggingFace provider")
        elif provider == "ollama":
            # Ollama just needs base_url, has default
            if not self.ollama_base_url and not self.llm.base_url:
                self.llm.base_url = "http://localhost:11434"
        
        return errors


# Global settings instance (lazy loaded)
_settings: Settings | None = None


def get_settings(config_path: str | Path | None = None, reload: bool = False) -> Settings:
    """Get the global settings instance.
    
    Args:
        config_path: Optional path to YAML config file.
        reload: If True, reload settings even if already loaded.
        
    Returns:
        Settings instance.
    """
    global _settings
    if _settings is None or reload:
        _settings = Settings.from_yaml(config_path)
    return _settings
