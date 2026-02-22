"""Audio tools for the music producer agents."""

from src.tools.audio_io import (
    load_audio,
    save_audio,
    extract_audio_tail,
    get_audio_duration,
)
from src.tools.audio_analysis import (
    analyze_bpm,
    analyze_key,
    analyze_energy,
    analyze_spectral,
    estimate_instruments,
)
from src.tools.audio_generation import (
    generate_segment,
    MusicGenWrapper,
)
from src.tools.audio_processing import (
    normalize_audio,
    apply_crossfade,
    apply_fade_in,
    apply_fade_out,
    concatenate_segments,
)

__all__ = [
    # IO
    "load_audio",
    "save_audio",
    "extract_audio_tail",
    "get_audio_duration",
    # Analysis
    "analyze_bpm",
    "analyze_key",
    "analyze_energy",
    "analyze_spectral",
    "estimate_instruments",
    # Generation
    "generate_segment",
    "MusicGenWrapper",
    # Processing
    "normalize_audio",
    "apply_crossfade",
    "apply_fade_in",
    "apply_fade_out",
    "concatenate_segments",
]
