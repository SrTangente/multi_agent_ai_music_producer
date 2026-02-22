"""Audio analysis tools using librosa.

These tools extract musical features from audio files for the
Analysis Agent to build musical profiles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.state.schemas import (
    BPMResult,
    EnergyProfile,
    InstrumentEstimation,
    KeyResult,
    SpectralFeatures,
    ToolError,
    ToolResult,
)


def analyze_bpm(path: str) -> ToolResult:
    """Analyze the tempo (BPM) of an audio file.
    
    Args:
        path: Path to audio file.
        
    Returns:
        ToolResult with BPMResult data.
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
        y, sr = librosa.load(path, sr=None)
        
        # Estimate tempo
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        
        # Handle both scalar and array returns (librosa version differences)
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            tempo = float(tempo)
        
        # Estimate confidence based on beat strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        if len(beat_frames) > 0:
            beat_strengths = onset_env[beat_frames]
            confidence = min(1.0, np.mean(beat_strengths) / (np.std(onset_env) + 1e-6))
        else:
            confidence = 0.5
        
        result: BPMResult = {
            "bpm": tempo,
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
        }
        
        return ToolResult(
            success=True,
            data=result,
            error=None,
        )
        
    except Exception as e:
        return ToolResult(
            success=False,
            data=None,
            error=ToolError(
                code="BPM_ANALYSIS_ERROR",
                message=f"Failed to analyze BPM: {e}",
                recoverable=True,
                suggested_action="Try a different audio file or check format",
            ),
        )


def analyze_key(path: str) -> ToolResult:
    """Analyze the musical key of an audio file.
    
    Uses chroma feature analysis to estimate key and mode.
    
    Args:
        path: Path to audio file.
        
    Returns:
        ToolResult with KeyResult data.
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
        y, sr = librosa.load(path, sr=None)
        
        # Compute chroma features
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        
        # Average chroma over time
        chroma_mean = np.mean(chroma, axis=1)
        
        # Key names (major keys)
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Major and minor key profiles (Krumhansl-Schmuckler)
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 
                                   2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                                   2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        # Correlate with all possible keys
        major_correlations = []
        minor_correlations = []
        
        for shift in range(12):
            shifted_chroma = np.roll(chroma_mean, -shift)
            major_correlations.append(np.corrcoef(shifted_chroma, major_profile)[0, 1])
            minor_correlations.append(np.corrcoef(shifted_chroma, minor_profile)[0, 1])
        
        # Find best match
        major_best = np.argmax(major_correlations)
        minor_best = np.argmax(minor_correlations)
        
        if major_correlations[major_best] > minor_correlations[minor_best]:
            key = key_names[major_best]
            mode = "major"
            confidence = float(major_correlations[major_best])
        else:
            key = key_names[minor_best] + "m"
            mode = "minor"
            confidence = float(minor_correlations[minor_best])
        
        # Normalize confidence to 0-1
        confidence = (confidence + 1) / 2  # Correlation is -1 to 1
        
        result: KeyResult = {
            "key": key,
            "mode": mode,
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
        }
        
        return ToolResult(
            success=True,
            data=result,
            error=None,
        )
        
    except Exception as e:
        return ToolResult(
            success=False,
            data=None,
            error=ToolError(
                code="KEY_ANALYSIS_ERROR",
                message=f"Failed to analyze key: {e}",
                recoverable=True,
                suggested_action="Try a different audio file or check format",
            ),
        )


def analyze_energy(path: str, num_segments: int = 10) -> ToolResult:
    """Analyze the energy profile of an audio file.
    
    Computes RMS energy over time to characterize dynamics.
    
    Args:
        path: Path to audio file.
        num_segments: Number of segments to divide audio into.
        
    Returns:
        ToolResult with EnergyProfile data.
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
        y, sr = librosa.load(path, sr=None)
        
        # Compute RMS energy
        rms = librosa.feature.rms(y=y)[0]
        
        # Normalize
        rms_normalized = rms / (np.max(rms) + 1e-6)
        
        # Divide into segments
        segment_size = len(rms_normalized) // num_segments
        energy_curve = []
        
        for i in range(num_segments):
            start = i * segment_size
            end = start + segment_size if i < num_segments - 1 else len(rms_normalized)
            segment_energy = np.mean(rms_normalized[start:end])
            energy_curve.append(float(segment_energy))
        
        result: EnergyProfile = {
            "mean_energy": float(np.mean(rms_normalized)),
            "energy_variance": float(np.var(rms_normalized)),
            "energy_curve": energy_curve,
        }
        
        return ToolResult(
            success=True,
            data=result,
            error=None,
        )
        
    except Exception as e:
        return ToolResult(
            success=False,
            data=None,
            error=ToolError(
                code="ENERGY_ANALYSIS_ERROR",
                message=f"Failed to analyze energy: {e}",
                recoverable=True,
                suggested_action="Try a different audio file or check format",
            ),
        )


def analyze_spectral(path: str) -> ToolResult:
    """Analyze spectral characteristics of an audio file.
    
    Computes spectral centroid, bandwidth, rolloff, and MFCCs.
    
    Args:
        path: Path to audio file.
        
    Returns:
        ToolResult with SpectralFeatures data.
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
        y, sr = librosa.load(path, sr=None)
        
        # Compute spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        result: SpectralFeatures = {
            "spectral_centroid_mean": float(np.mean(spectral_centroid)),
            "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
            "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
            "mfcc_means": [float(m) for m in np.mean(mfccs, axis=1)],
        }
        
        return ToolResult(
            success=True,
            data=result,
            error=None,
        )
        
    except Exception as e:
        return ToolResult(
            success=False,
            data=None,
            error=ToolError(
                code="SPECTRAL_ANALYSIS_ERROR",
                message=f"Failed to analyze spectral features: {e}",
                recoverable=True,
                suggested_action="Try a different audio file or check format",
            ),
        )


def estimate_instruments(path: str) -> ToolResult:
    """Estimate the instrumentation in an audio file.
    
    Uses spectral features to make educated guesses about
    instruments present in the audio.
    
    Note: This is a heuristic-based estimation, not a true
    instrument classifier. For production use, consider using
    a trained model.
    
    Args:
        path: Path to audio file.
        
    Returns:
        ToolResult with InstrumentEstimation data.
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
        y, sr = librosa.load(path, sr=None)
        
        # Compute features for instrument estimation
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # Onset detection for percussive content
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_rate = np.sum(onset_env > np.mean(onset_env) * 2) / (len(y) / sr)
        
        # Heuristic instrument detection
        instruments = []
        
        # High spectral centroid + high ZCR → bright sounds (synths, cymbals)
        if spectral_centroid > 3000 and zero_crossing_rate > 0.1:
            instruments.append("synth")
        
        # Low spectral centroid → bass-heavy
        if spectral_centroid < 1500:
            instruments.append("bass")
        
        # High onset rate + high spectral flatness → drums/percussion
        if onset_rate > 2 and spectral_flatness > 0.1:
            instruments.append("drums")
        
        # Medium centroid + low flatness → melodic (guitar, piano, vocals)
        if 1500 < spectral_centroid < 3000 and spectral_flatness < 0.1:
            instruments.append("piano")
            instruments.append("guitar")
        
        # Very high centroid → potentially has strings/pads
        if spectral_centroid > 2500 and spectral_bandwidth > 2000:
            instruments.append("strings")
        
        # Default instruments if nothing detected
        if not instruments:
            instruments = ["synth", "drums", "bass"]
        
        # Determine dominant instrument (rough heuristic)
        if "drums" in instruments and onset_rate > 3:
            dominant = "drums"
        elif "bass" in instruments and spectral_centroid < 1200:
            dominant = "bass"
        elif "synth" in instruments:
            dominant = "synth"
        else:
            dominant = instruments[0] if instruments else None
        
        result: InstrumentEstimation = {
            "instruments": instruments,
            "confidence": 0.6,  # Heuristic-based, so moderate confidence
            "dominant_instrument": dominant,
        }
        
        return ToolResult(
            success=True,
            data=result,
            error=None,
        )
        
    except Exception as e:
        return ToolResult(
            success=False,
            data=None,
            error=ToolError(
                code="INSTRUMENT_ESTIMATION_ERROR",
                message=f"Failed to estimate instruments: {e}",
                recoverable=True,
                suggested_action="Try a different audio file or check format",
            ),
        )


def analyze_full_track(path: str) -> ToolResult:
    """Perform full analysis on an audio track.
    
    Convenience function that runs all analysis tools and
    combines results.
    
    Args:
        path: Path to audio file.
        
    Returns:
        ToolResult with combined analysis results.
    """
    results = {}
    errors = []
    
    # Run all analyses
    bpm_result = analyze_bpm(path)
    if bpm_result["success"]:
        results["bpm"] = bpm_result["data"]
    else:
        errors.append(f"BPM: {bpm_result['error']['message']}")
    
    key_result = analyze_key(path)
    if key_result["success"]:
        results["key"] = key_result["data"]
    else:
        errors.append(f"Key: {key_result['error']['message']}")
    
    energy_result = analyze_energy(path)
    if energy_result["success"]:
        results["energy"] = energy_result["data"]
    else:
        errors.append(f"Energy: {energy_result['error']['message']}")
    
    spectral_result = analyze_spectral(path)
    if spectral_result["success"]:
        results["spectral"] = spectral_result["data"]
    else:
        errors.append(f"Spectral: {spectral_result['error']['message']}")
    
    instrument_result = estimate_instruments(path)
    if instrument_result["success"]:
        results["instruments"] = instrument_result["data"]
    else:
        errors.append(f"Instruments: {instrument_result['error']['message']}")
    
    # Determine overall success
    if len(results) >= 3:  # At least 3 successful analyses
        return ToolResult(
            success=True,
            data={
                "path": path,
                "analysis": results,
                "warnings": errors if errors else None,
            },
            error=None,
        )
    else:
        return ToolResult(
            success=False,
            data=results if results else None,
            error=ToolError(
                code="ANALYSIS_PARTIAL_FAILURE",
                message=f"Too many analyses failed: {'; '.join(errors)}",
                recoverable=True,
                suggested_action="Check audio file quality and format",
            ),
        )
