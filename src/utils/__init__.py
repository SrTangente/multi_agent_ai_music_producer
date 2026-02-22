"""Utility functions for the music producer."""

from src.utils.audio import (
    get_audio_duration,
    get_audio_format,
    is_valid_audio_file,
)
from src.utils.device import (
    get_available_device,
    get_device_info,
    DeviceType,
)
from src.utils.prompts import (
    build_analysis_prompt,
    build_director_prompt,
    build_production_prompt,
    build_critic_prompt,
    build_segment_generation_prompt,
)

__all__ = [
    # Audio
    "get_audio_duration",
    "get_audio_format",
    "is_valid_audio_file",
    # Device
    "get_available_device",
    "get_device_info",
    "DeviceType",
    # Prompts
    "build_analysis_prompt",
    "build_director_prompt",
    "build_production_prompt",
    "build_critic_prompt",
    "build_segment_generation_prompt",
]
