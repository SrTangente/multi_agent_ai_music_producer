"""Audio processing tools for mastering.

These tools handle post-processing operations like normalization,
crossfades, and concatenation for the Mastering Agent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.state.schemas import ToolError, ToolResult


def normalize_audio(
    path: str,
    target_lufs: float = -14.0,
    output_path: str | None = None,
) -> ToolResult:
    """Normalize audio to a target loudness level.
    
    Uses LUFS (Loudness Units Full Scale) for broadcast-standard
    normalization. -14 LUFS is typical for streaming platforms.
    
    Args:
        path: Path to input audio.
        target_lufs: Target loudness in LUFS.
        output_path: Where to save (overwrites input if None).
        
    Returns:
        ToolResult with normalized audio path.
    """
    try:
        import librosa
        import soundfile as sf
        
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
        y, sr = librosa.load(path, sr=None)
        
        # Calculate current loudness (simplified LUFS approximation)
        # True LUFS requires ITU-R BS.1770-4, this is a reasonable approximation
        rms = np.sqrt(np.mean(y**2))
        current_db = 20 * np.log10(rms + 1e-10)
        
        # Approximate LUFS (RMS dB is close for most content)
        current_lufs_approx = current_db
        
        # Calculate gain needed
        gain_db = target_lufs - current_lufs_approx
        gain_linear = 10 ** (gain_db / 20)
        
        # Apply gain
        y_normalized = y * gain_linear
        
        # Prevent clipping
        max_val = np.max(np.abs(y_normalized))
        if max_val > 0.99:
            y_normalized = y_normalized * (0.99 / max_val)
        
        # Determine output path
        if output_path is None:
            output_path = path
        
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save
        sf.write(output_path, y_normalized, sr)
        
        return ToolResult(
            success=True,
            data={
                "path": output_path,
                "original_lufs_approx": float(current_lufs_approx),
                "target_lufs": target_lufs,
                "gain_db": float(gain_db),
            },
            error=None,
        )
        
    except Exception as e:
        return ToolResult(
            success=False,
            data=None,
            error=ToolError(
                code="NORMALIZATION_ERROR",
                message=f"Failed to normalize audio: {e}",
                recoverable=True,
                suggested_action="Check audio file format",
            ),
        )


def apply_crossfade(
    path1: str,
    path2: str,
    output_path: str,
    fade_duration_ms: int = 500,
) -> ToolResult:
    """Apply crossfade between two audio files.
    
    Creates a smooth transition by overlapping the end of the
    first file with the beginning of the second.
    
    Args:
        path1: Path to first audio file.
        path2: Path to second audio file.
        output_path: Where to save the crossfaded result.
        fade_duration_ms: Duration of crossfade in milliseconds.
        
    Returns:
        ToolResult with crossfaded audio path.
    """
    try:
        import librosa
        import soundfile as sf
        
        # Load both files
        y1, sr1 = librosa.load(path1, sr=None)
        y2, sr2 = librosa.load(path2, sr=None)
        
        # Ensure same sample rate
        if sr1 != sr2:
            y2 = librosa.resample(y2, orig_sr=sr2, target_sr=sr1)
        sr = sr1
        
        # Calculate fade samples
        fade_samples = int(fade_duration_ms * sr / 1000)
        
        # Ensure we have enough samples
        fade_samples = min(fade_samples, len(y1) // 2, len(y2) // 2)
        
        if fade_samples <= 0:
            # No crossfade possible, just concatenate
            combined = np.concatenate([y1, y2])
        else:
            # Create fade curves
            fade_out = np.linspace(1.0, 0.0, fade_samples)
            fade_in = np.linspace(0.0, 1.0, fade_samples)
            
            # Apply fades to overlap region
            y1_end = y1[-fade_samples:] * fade_out
            y2_start = y2[:fade_samples] * fade_in
            
            # Combine
            crossfade_region = y1_end + y2_start
            
            # Concatenate: y1 (without end) + crossfade + y2 (without start)
            combined = np.concatenate([
                y1[:-fade_samples],
                crossfade_region,
                y2[fade_samples:],
            ])
        
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save
        sf.write(output_path, combined, sr)
        
        return ToolResult(
            success=True,
            data={
                "path": output_path,
                "duration_sec": len(combined) / sr,
                "fade_duration_ms": fade_duration_ms,
            },
            error=None,
        )
        
    except Exception as e:
        return ToolResult(
            success=False,
            data=None,
            error=ToolError(
                code="CROSSFADE_ERROR",
                message=f"Failed to apply crossfade: {e}",
                recoverable=True,
                suggested_action="Check input file formats match",
            ),
        )


def apply_fade_in(
    path: str,
    duration_ms: int = 1000,
    output_path: str | None = None,
) -> ToolResult:
    """Apply fade-in to the beginning of an audio file.
    
    Args:
        path: Path to input audio.
        duration_ms: Fade duration in milliseconds.
        output_path: Where to save (overwrites input if None).
        
    Returns:
        ToolResult with faded audio path.
    """
    try:
        import librosa
        import soundfile as sf
        
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
        y, sr = librosa.load(path, sr=None)
        
        # Calculate fade samples
        fade_samples = int(duration_ms * sr / 1000)
        fade_samples = min(fade_samples, len(y))
        
        # Create fade curve
        fade_curve = np.linspace(0.0, 1.0, fade_samples)
        
        # Apply fade
        y[:fade_samples] = y[:fade_samples] * fade_curve
        
        # Determine output path
        if output_path is None:
            output_path = path
        
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save
        sf.write(output_path, y, sr)
        
        return ToolResult(
            success=True,
            data={
                "path": output_path,
                "fade_duration_ms": duration_ms,
            },
            error=None,
        )
        
    except Exception as e:
        return ToolResult(
            success=False,
            data=None,
            error=ToolError(
                code="FADE_IN_ERROR",
                message=f"Failed to apply fade-in: {e}",
                recoverable=True,
                suggested_action="Check audio file format",
            ),
        )


def apply_fade_out(
    path: str,
    duration_ms: int = 2000,
    output_path: str | None = None,
) -> ToolResult:
    """Apply fade-out to the end of an audio file.
    
    Args:
        path: Path to input audio.
        duration_ms: Fade duration in milliseconds.
        output_path: Where to save (overwrites input if None).
        
    Returns:
        ToolResult with faded audio path.
    """
    try:
        import librosa
        import soundfile as sf
        
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
        y, sr = librosa.load(path, sr=None)
        
        # Calculate fade samples
        fade_samples = int(duration_ms * sr / 1000)
        fade_samples = min(fade_samples, len(y))
        
        # Create fade curve
        fade_curve = np.linspace(1.0, 0.0, fade_samples)
        
        # Apply fade
        y[-fade_samples:] = y[-fade_samples:] * fade_curve
        
        # Determine output path
        if output_path is None:
            output_path = path
        
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save
        sf.write(output_path, y, sr)
        
        return ToolResult(
            success=True,
            data={
                "path": output_path,
                "fade_duration_ms": duration_ms,
            },
            error=None,
        )
        
    except Exception as e:
        return ToolResult(
            success=False,
            data=None,
            error=ToolError(
                code="FADE_OUT_ERROR",
                message=f"Failed to apply fade-out: {e}",
                recoverable=True,
                suggested_action="Check audio file format",
            ),
        )


def concatenate_segments(
    segment_paths: list[str],
    output_path: str,
    crossfade_duration_ms: int = 500,
) -> ToolResult:
    """Concatenate multiple audio segments with crossfades.
    
    Args:
        segment_paths: List of paths to audio segments in order.
        output_path: Where to save the concatenated result.
        crossfade_duration_ms: Duration of crossfade between segments.
        
    Returns:
        ToolResult with concatenated audio path.
    """
    try:
        import librosa
        import soundfile as sf
        
        if not segment_paths:
            return ToolResult(
                success=False,
                data=None,
                error=ToolError(
                    code="EMPTY_SEGMENTS",
                    message="No segments provided for concatenation",
                    recoverable=False,
                    suggested_action="Provide at least one segment",
                ),
            )
        
        # Validate all files exist
        for path in segment_paths:
            if not Path(path).exists():
                return ToolResult(
                    success=False,
                    data=None,
                    error=ToolError(
                        code="FILE_NOT_FOUND",
                        message=f"Segment file not found: {path}",
                        recoverable=False,
                        suggested_action="Check segment paths",
                    ),
                )
        
        # Load first segment to get sample rate
        first_y, sr = librosa.load(segment_paths[0], sr=None)
        
        if len(segment_paths) == 1:
            # Single segment, just copy
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, first_y, sr)
            
            return ToolResult(
                success=True,
                data={
                    "path": output_path,
                    "duration_sec": len(first_y) / sr,
                    "segment_count": 1,
                },
                error=None,
            )
        
        # Load all segments
        segments = [first_y]
        for path in segment_paths[1:]:
            y, segment_sr = librosa.load(path, sr=None)
            # Resample if needed
            if segment_sr != sr:
                y = librosa.resample(y, orig_sr=segment_sr, target_sr=sr)
            segments.append(y)
        
        # Calculate fade samples
        fade_samples = int(crossfade_duration_ms * sr / 1000)
        
        # Concatenate with crossfades
        combined = segments[0]
        
        for next_segment in segments[1:]:
            # Ensure we have enough samples for crossfade
            actual_fade = min(fade_samples, len(combined) // 4, len(next_segment) // 4)
            
            if actual_fade <= 0:
                # No crossfade possible, just concatenate
                combined = np.concatenate([combined, next_segment])
            else:
                # Create fade curves
                fade_out = np.linspace(1.0, 0.0, actual_fade)
                fade_in = np.linspace(0.0, 1.0, actual_fade)
                
                # Apply fades to overlap region
                combined_end = combined[-actual_fade:] * fade_out
                next_start = next_segment[:actual_fade] * fade_in
                
                # Combine
                crossfade_region = combined_end + next_start
                
                # Concatenate
                combined = np.concatenate([
                    combined[:-actual_fade],
                    crossfade_region,
                    next_segment[actual_fade:],
                ])
        
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save
        sf.write(output_path, combined, sr)
        
        return ToolResult(
            success=True,
            data={
                "path": output_path,
                "duration_sec": len(combined) / sr,
                "segment_count": len(segment_paths),
                "crossfade_duration_ms": crossfade_duration_ms,
            },
            error=None,
        )
        
    except Exception as e:
        return ToolResult(
            success=False,
            data=None,
            error=ToolError(
                code="CONCATENATION_ERROR",
                message=f"Failed to concatenate segments: {e}",
                recoverable=True,
                suggested_action="Check all segment files are valid audio",
            ),
        )


def apply_compression(
    path: str,
    threshold_db: float = -20.0,
    ratio: float = 4.0,
    output_path: str | None = None,
) -> ToolResult:
    """Apply basic dynamic range compression.
    
    Simple compressor for evening out loud peaks.
    
    Args:
        path: Path to input audio.
        threshold_db: Threshold in dB above which compression applies.
        ratio: Compression ratio (e.g., 4:1).
        output_path: Where to save (overwrites input if None).
        
    Returns:
        ToolResult with compressed audio path.
    """
    try:
        import librosa
        import soundfile as sf
        
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
        y, sr = librosa.load(path, sr=None)
        
        # Convert to dB
        threshold_linear = 10 ** (threshold_db / 20)
        
        # Simple compression
        y_compressed = np.copy(y)
        
        # Find samples above threshold
        above_threshold = np.abs(y) > threshold_linear
        
        for i in np.where(above_threshold)[0]:
            # Calculate how much above threshold
            excess_db = 20 * np.log10(np.abs(y[i]) / threshold_linear + 1e-10)
            # Apply ratio
            reduced_db = excess_db / ratio
            # Convert back to linear
            new_level = threshold_linear * (10 ** (reduced_db / 20))
            # Apply with original sign
            y_compressed[i] = np.sign(y[i]) * new_level
        
        # Makeup gain (simple)
        makeup_gain = np.sqrt(np.mean(y**2)) / (np.sqrt(np.mean(y_compressed**2)) + 1e-10)
        y_compressed = y_compressed * makeup_gain
        
        # Prevent clipping
        max_val = np.max(np.abs(y_compressed))
        if max_val > 0.99:
            y_compressed = y_compressed * (0.99 / max_val)
        
        # Determine output path
        if output_path is None:
            output_path = path
        
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save
        sf.write(output_path, y_compressed, sr)
        
        return ToolResult(
            success=True,
            data={
                "path": output_path,
                "threshold_db": threshold_db,
                "ratio": ratio,
            },
            error=None,
        )
        
    except Exception as e:
        return ToolResult(
            success=False,
            data=None,
            error=ToolError(
                code="COMPRESSION_ERROR",
                message=f"Failed to apply compression: {e}",
                recoverable=True,
                suggested_action="Check audio file format",
            ),
        )
