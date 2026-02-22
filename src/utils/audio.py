"""Audio utility functions.

Provides helpers for audio file validation, metadata extraction,
and format detection.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Supported audio formats
SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def get_audio_format(file_path: str | Path) -> str | None:
    """Get the audio format from file extension.
    
    Args:
        file_path: Path to audio file.
        
    Returns:
        Format string (e.g., "wav", "mp3") or None if unsupported.
    """
    ext = Path(file_path).suffix.lower()
    if ext in SUPPORTED_FORMATS:
        return ext[1:]  # Remove leading dot
    return None


def is_valid_audio_file(file_path: str | Path) -> tuple[bool, str | None]:
    """Check if a file is a valid audio file.
    
    Validates:
    - File exists
    - Has supported extension
    - Has non-zero size
    - Can be opened by soundfile
    
    Args:
        file_path: Path to check.
        
    Returns:
        Tuple of (is_valid, error_message).
    """
    path = Path(file_path)
    
    if not path.exists():
        return False, f"File not found: {file_path}"
    
    if not path.is_file():
        return False, f"Not a file: {file_path}"
    
    if path.stat().st_size == 0:
        return False, f"Empty file: {file_path}"
    
    fmt = get_audio_format(path)
    if fmt is None:
        return False, f"Unsupported format: {path.suffix}"
    
    # Try to open with soundfile for deeper validation
    try:
        import soundfile as sf
        with sf.SoundFile(str(path)) as f:
            # Just opening it validates the file
            _ = f.samplerate
        return True, None
    except Exception as e:
        # soundfile might not support all formats (e.g., mp3)
        # Fall back to librosa for those
        try:
            import librosa
            # Just load a tiny bit to validate
            _, _ = librosa.load(str(path), sr=None, duration=0.1)
            return True, None
        except Exception as e2:
            return False, f"Cannot read audio file: {e2}"


def get_audio_duration(file_path: str | Path) -> float | None:
    """Get the duration of an audio file in seconds.
    
    Args:
        file_path: Path to audio file.
        
    Returns:
        Duration in seconds, or None if file cannot be read.
    """
    path = Path(file_path)
    
    if not path.exists():
        return None
    
    try:
        import librosa
        duration = librosa.get_duration(path=str(path))
        return float(duration)
    except Exception:
        # Fallback to soundfile
        try:
            import soundfile as sf
            with sf.SoundFile(str(path)) as f:
                return float(f.frames) / float(f.samplerate)
        except Exception:
            return None


def get_audio_info(file_path: str | Path) -> dict[str, Any] | None:
    """Get detailed information about an audio file.
    
    Args:
        file_path: Path to audio file.
        
    Returns:
        Dictionary with audio info, or None if file cannot be read.
    """
    path = Path(file_path)
    
    if not path.exists():
        return None
    
    try:
        import soundfile as sf
        with sf.SoundFile(str(path)) as f:
            return {
                "path": str(path),
                "format": get_audio_format(path),
                "sample_rate": f.samplerate,
                "channels": f.channels,
                "frames": f.frames,
                "duration_sec": float(f.frames) / float(f.samplerate),
                "subtype": f.subtype,
            }
    except Exception:
        # Fallback for formats soundfile doesn't support
        try:
            import librosa
            y, sr = librosa.load(str(path), sr=None, mono=False)
            if y.ndim == 1:
                channels = 1
                frames = len(y)
            else:
                channels = y.shape[0]
                frames = y.shape[1]
            
            return {
                "path": str(path),
                "format": get_audio_format(path),
                "sample_rate": sr,
                "channels": channels,
                "frames": frames,
                "duration_sec": float(frames) / float(sr),
                "subtype": None,
            }
        except Exception:
            return None


def list_audio_files(
    directory: str | Path,
    recursive: bool = False,
) -> list[Path]:
    """List all audio files in a directory.
    
    Args:
        directory: Directory to search.
        recursive: If True, search subdirectories.
        
    Returns:
        List of paths to audio files.
    """
    path = Path(directory)
    
    if not path.exists() or not path.is_dir():
        return []
    
    audio_files: list[Path] = []
    
    if recursive:
        for ext in SUPPORTED_FORMATS:
            audio_files.extend(path.rglob(f"*{ext}"))
    else:
        for ext in SUPPORTED_FORMATS:
            audio_files.extend(path.glob(f"*{ext}"))
    
    return sorted(audio_files)


def validate_reference_folder(
    folder_path: str | Path,
) -> tuple[list[str], list[str]]:
    """Validate a folder of reference tracks.
    
    Args:
        folder_path: Path to folder containing reference tracks.
        
    Returns:
        Tuple of (valid_paths, error_messages).
    """
    audio_files = list_audio_files(folder_path)
    
    if not audio_files:
        return [], [f"No audio files found in {folder_path}"]
    
    valid_paths: list[str] = []
    errors: list[str] = []
    
    for audio_file in audio_files:
        is_valid, error = is_valid_audio_file(audio_file)
        if is_valid:
            valid_paths.append(str(audio_file))
        else:
            errors.append(error or f"Invalid audio file: {audio_file}")
    
    return valid_paths, errors


def samples_to_seconds(samples: int, sample_rate: int) -> float:
    """Convert samples to seconds."""
    return float(samples) / float(sample_rate)


def seconds_to_samples(seconds: float, sample_rate: int) -> int:
    """Convert seconds to samples."""
    return int(seconds * sample_rate)


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds.
        
    Returns:
        Formatted string like "2:34" or "1:02:34".
    """
    if seconds < 0:
        return "0:00"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"
