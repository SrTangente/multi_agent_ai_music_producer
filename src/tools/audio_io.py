"""Audio I/O tools for loading, saving, and manipulating audio files.

These tools handle all file-based audio operations with proper
error handling and format support.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.state.schemas import ToolError, ToolResult


def load_audio(
    path: str,
    sample_rate: int | None = None,
    mono: bool = True,
) -> ToolResult:
    """Load an audio file.
    
    Args:
        path: Path to audio file.
        sample_rate: Target sample rate (None to keep original).
        mono: Convert to mono if True.
        
    Returns:
        ToolResult with audio data and metadata.
    """
    try:
        import librosa
        
        if not Path(path).exists():
            return ToolResult(
                success=False,
                data=None,
                error=ToolError(
                    code="FILE_NOT_FOUND",
                    message=f"Audio file not found: {path}",
                    recoverable=False,
                    suggested_action="Check the file path",
                ),
            )
        
        # Load audio
        y, sr = librosa.load(path, sr=sample_rate, mono=mono)
        
        duration = len(y) / sr
        
        return ToolResult(
            success=True,
            data={
                "audio": y,
                "sample_rate": sr,
                "duration_sec": duration,
                "path": path,
                "channels": 1 if mono else (2 if y.ndim > 1 else 1),
            },
            error=None,
        )
        
    except Exception as e:
        return ToolResult(
            success=False,
            data=None,
            error=ToolError(
                code="AUDIO_LOAD_ERROR",
                message=f"Failed to load audio: {e}",
                recoverable=False,
                suggested_action="Check if file is a valid audio format",
            ),
        )


def save_audio(
    audio: np.ndarray,
    path: str,
    sample_rate: int,
    format: str = "wav",
) -> ToolResult:
    """Save audio data to a file.
    
    Args:
        audio: Audio data as numpy array.
        path: Output file path.
        sample_rate: Sample rate of the audio.
        format: Output format (wav, mp3, flac).
        
    Returns:
        ToolResult with saved file path.
    """
    try:
        import soundfile as sf
        
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure path has correct extension
        if not path.endswith(f".{format}"):
            path = f"{path}.{format}"
        
        # Normalize audio to prevent clipping
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))
        
        # Save based on format
        if format in ("wav", "flac"):
            sf.write(path, audio, sample_rate)
        elif format == "mp3":
            # soundfile doesn't support mp3, save as wav first
            # For mp3, would need pydub or ffmpeg
            wav_path = path.replace(".mp3", ".wav")
            sf.write(wav_path, audio, sample_rate)
            path = wav_path  # Return wav path for now
        else:
            sf.write(path, audio, sample_rate)
        
        return ToolResult(
            success=True,
            data={
                "path": path,
                "duration_sec": len(audio) / sample_rate,
                "sample_rate": sample_rate,
            },
            error=None,
        )
        
    except Exception as e:
        return ToolResult(
            success=False,
            data=None,
            error=ToolError(
                code="AUDIO_SAVE_ERROR",
                message=f"Failed to save audio: {e}",
                recoverable=True,
                suggested_action="Check write permissions and disk space",
            ),
        )


def extract_audio_tail(
    path: str,
    duration_sec: float,
    output_path: str | None = None,
    sample_rate: int | None = None,
) -> ToolResult:
    """Extract the last N seconds from an audio file.
    
    Used for continuity conditioning - takes the end of a segment
    to use as reference for generating the next segment.
    
    Args:
        path: Path to source audio file.
        duration_sec: Duration to extract from end.
        output_path: Where to save extracted audio (temp file if None).
        sample_rate: Target sample rate.
        
    Returns:
        ToolResult with path to extracted audio.
    """
    try:
        import librosa
        import tempfile
        
        # Load audio
        load_result = load_audio(path, sample_rate=sample_rate)
        if not load_result["success"]:
            return load_result
        
        audio = load_result["data"]["audio"]
        sr = load_result["data"]["sample_rate"]
        total_duration = load_result["data"]["duration_sec"]
        
        # Calculate samples to extract
        if duration_sec >= total_duration:
            # If requested more than available, use entire audio
            tail_audio = audio
        else:
            samples_to_extract = int(duration_sec * sr)
            tail_audio = audio[-samples_to_extract:]
        
        # Determine output path
        if output_path is None:
            # Create temp file
            suffix = Path(path).suffix or ".wav"
            temp_fd, output_path = tempfile.mkstemp(suffix=suffix)
            import os
            os.close(temp_fd)
        
        # Save extracted audio
        save_result = save_audio(
            audio=tail_audio,
            path=output_path,
            sample_rate=sr,
        )
        
        if not save_result["success"]:
            return save_result
        
        return ToolResult(
            success=True,
            data={
                "path": output_path,
                "duration_sec": len(tail_audio) / sr,
                "sample_rate": sr,
                "source_path": path,
            },
            error=None,
        )
        
    except Exception as e:
        return ToolResult(
            success=False,
            data=None,
            error=ToolError(
                code="AUDIO_EXTRACT_ERROR",
                message=f"Failed to extract audio tail: {e}",
                recoverable=True,
                suggested_action="Check source file exists and is valid",
            ),
        )


def get_audio_duration(path: str) -> ToolResult:
    """Get the duration of an audio file in seconds.
    
    Args:
        path: Path to audio file.
        
    Returns:
        ToolResult with duration.
    """
    try:
        import librosa
        
        if not Path(path).exists():
            return ToolResult(
                success=False,
                data=None,
                error=ToolError(
                    code="FILE_NOT_FOUND",
                    message=f"Audio file not found: {path}",
                    recoverable=False,
                    suggested_action="Check the file path",
                ),
            )
        
        duration = librosa.get_duration(path=path)
        
        return ToolResult(
            success=True,
            data={
                "duration_sec": float(duration),
                "path": path,
            },
            error=None,
        )
        
    except Exception as e:
        return ToolResult(
            success=False,
            data=None,
            error=ToolError(
                code="DURATION_ERROR",
                message=f"Failed to get duration: {e}",
                recoverable=False,
                suggested_action="Check if file is a valid audio format",
            ),
        )


def concatenate_audio(
    audio_list: list[np.ndarray],
    sample_rate: int,
    output_path: str,
) -> ToolResult:
    """Concatenate multiple audio arrays.
    
    Args:
        audio_list: List of audio arrays to concatenate.
        sample_rate: Sample rate of all audio.
        output_path: Where to save concatenated audio.
        
    Returns:
        ToolResult with concatenated audio path.
    """
    try:
        if not audio_list:
            return ToolResult(
                success=False,
                data=None,
                error=ToolError(
                    code="EMPTY_LIST",
                    message="Cannot concatenate empty list",
                    recoverable=False,
                    suggested_action="Provide at least one audio segment",
                ),
            )
        
        # Concatenate
        combined = np.concatenate(audio_list)
        
        # Save
        return save_audio(
            audio=combined,
            path=output_path,
            sample_rate=sample_rate,
        )
        
    except Exception as e:
        return ToolResult(
            success=False,
            data=None,
            error=ToolError(
                code="CONCATENATE_ERROR",
                message=f"Failed to concatenate audio: {e}",
                recoverable=True,
                suggested_action="Check all audio arrays have same sample rate",
            ),
        )


def resample_audio(
    path: str,
    target_sample_rate: int,
    output_path: str | None = None,
) -> ToolResult:
    """Resample audio to a target sample rate.
    
    Args:
        path: Path to source audio.
        target_sample_rate: Target sample rate.
        output_path: Where to save (overwrites if None).
        
    Returns:
        ToolResult with resampled audio path.
    """
    try:
        import librosa
        
        # Load at original sample rate
        y, sr = librosa.load(path, sr=None)
        
        if sr == target_sample_rate:
            # Already at target rate
            return ToolResult(
                success=True,
                data={
                    "path": path,
                    "sample_rate": sr,
                    "duration_sec": len(y) / sr,
                },
                error=None,
            )
        
        # Resample
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sample_rate)
        
        # Determine output path
        if output_path is None:
            output_path = path
        
        # Save
        return save_audio(
            audio=y_resampled,
            path=output_path,
            sample_rate=target_sample_rate,
        )
        
    except Exception as e:
        return ToolResult(
            success=False,
            data=None,
            error=ToolError(
                code="RESAMPLE_ERROR",
                message=f"Failed to resample audio: {e}",
                recoverable=True,
                suggested_action="Check source file is valid",
            ),
        )
