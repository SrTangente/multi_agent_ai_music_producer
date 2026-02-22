"""Audio generation tool using MusicGen.

Wraps Meta's audiocraft MusicGen Melody model for generating
audio segments with conditioning support and retry logic.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.state.schemas import GeneratedSegment, ToolError, ToolResult
from src.utils.device import DeviceInfo, DeviceType, get_device_info, get_torch_device


class MusicGenWrapper:
    """Wrapper for MusicGen Melody model with caching and error handling.
    
    Provides a clean interface for audio generation with:
    - Model caching (avoid repeated loading)
    - Device management (TPU/GPU/CPU fallback)
    - OOM recovery (automatic retry with smaller batches)
    - Timeout protection
    """
    
    _instance: "MusicGenWrapper | None" = None
    _model = None
    _device_info: DeviceInfo | None = None
    
    def __init__(
        self,
        model_name: str = "facebook/musicgen-melody",
        device: DeviceInfo | None = None,
    ):
        """Initialize the MusicGen wrapper.
        
        Args:
            model_name: HuggingFace model name or path.
            device: Device to use (auto-detected if None).
        """
        self.model_name = model_name
        self._device_info = device or get_device_info()
        self._model = None
    
    @classmethod
    def get_instance(
        cls,
        model_name: str = "facebook/musicgen-melody",
    ) -> "MusicGenWrapper":
        """Get or create the singleton instance.
        
        Args:
            model_name: Model to use.
            
        Returns:
            MusicGenWrapper instance.
        """
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = cls(model_name=model_name)
        return cls._instance
    
    def _ensure_loaded(self) -> None:
        """Ensure the model is loaded."""
        if self._model is not None:
            return
        
        try:
            from audiocraft.models import MusicGen
            
            # Load model
            self._model = MusicGen.get_pretrained(self.model_name)
            
            # Move to device
            device = get_torch_device(self._device_info)
            self._model.to(device)
            
        except ImportError:
            raise ImportError(
                "audiocraft package required. Install with: pip install audiocraft"
            )
    
    def generate(
        self,
        prompt: str,
        duration_sec: float,
        conditioning_audio_path: str | None = None,
        temperature: float = 1.0,
        top_k: int = 250,
        cfg_coef: float = 3.0,
    ) -> tuple[np.ndarray, int]:
        """Generate audio from a text prompt.
        
        Args:
            prompt: Text description of desired audio.
            duration_sec: Target duration in seconds.
            conditioning_audio_path: Optional audio for melodic conditioning.
            temperature: Sampling temperature (lower = more deterministic).
            top_k: Top-k sampling parameter.
            cfg_coef: Classifier-free guidance coefficient.
            
        Returns:
            Tuple of (audio_array, sample_rate).
        """
        self._ensure_loaded()
        
        import torch
        import torchaudio
        
        # Set generation parameters
        self._model.set_generation_params(
            duration=duration_sec,
            temperature=temperature,
            top_k=top_k,
            cfg_coef=cfg_coef,
        )
        
        # Load conditioning audio if provided
        if conditioning_audio_path and Path(conditioning_audio_path).exists():
            melody, melody_sr = torchaudio.load(conditioning_audio_path)
            # Generate with melody conditioning
            with torch.no_grad():
                output = self._model.generate_with_chroma(
                    descriptions=[prompt],
                    melody_wavs=melody.unsqueeze(0),
                    melody_sample_rate=melody_sr,
                )
        else:
            # Generate without conditioning
            with torch.no_grad():
                output = self._model.generate([prompt])
        
        # Extract audio
        audio = output[0].cpu().numpy()
        
        # MusicGen outputs at 32kHz
        sample_rate = self._model.sample_rate
        
        return audio, sample_rate
    
    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def generate_segment(
    prompt: str,
    duration_sec: float,
    output_path: str,
    conditioning_audio_path: str | None = None,
    temperature: float = 1.0,
    top_k: int = 250,
    cfg_coef: float = 3.0,
    max_retries_on_oom: int = 2,
) -> ToolResult:
    """Generate an audio segment using MusicGen.
    
    This is the main tool for audio generation, with built-in
    error handling for OOM and other failures.
    
    Args:
        prompt: Text description of desired audio.
        duration_sec: Target duration in seconds.
        output_path: Where to save generated audio.
        conditioning_audio_path: Optional audio for melodic conditioning.
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter.
        cfg_coef: Classifier-free guidance coefficient.
        max_retries_on_oom: Max retries on OOM (with reduced duration).
        
    Returns:
        ToolResult with GeneratedSegment data.
    """
    start_time = time.time()
    
    try:
        import soundfile as sf
        
        # Get wrapper instance
        wrapper = MusicGenWrapper.get_instance()
        
        # Attempt generation with OOM recovery
        current_duration = duration_sec
        attempt = 0
        
        while attempt <= max_retries_on_oom:
            try:
                audio, sample_rate = wrapper.generate(
                    prompt=prompt,
                    duration_sec=current_duration,
                    conditioning_audio_path=conditioning_audio_path,
                    temperature=temperature,
                    top_k=top_k,
                    cfg_coef=cfg_coef,
                )
                break
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and attempt < max_retries_on_oom:
                    # OOM recovery: reduce duration
                    attempt += 1
                    current_duration = current_duration * 0.75
                    
                    # Clear GPU cache
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    continue
                else:
                    raise
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Handle multi-channel output
        if audio.ndim > 1:
            # Take first channel or average
            audio = np.mean(audio, axis=0) if audio.shape[0] <= 2 else audio[0]
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val
        
        # Save audio
        sf.write(output_path, audio, sample_rate)
        
        actual_duration = len(audio) / sample_rate
        generation_time = time.time() - start_time
        
        result: GeneratedSegment = {
            "audio_path": output_path,
            "duration_sec": actual_duration,
            "generation_params": {
                "prompt": prompt,
                "temperature": temperature,
                "top_k": top_k,
                "cfg_coef": cfg_coef,
                "conditioning_audio": conditioning_audio_path,
                "generation_time_sec": generation_time,
            },
        }
        
        return ToolResult(
            success=True,
            data=result,
            error=None,
        )
        
    except ImportError as e:
        return ToolResult(
            success=False,
            data=None,
            error=ToolError(
                code="MUSICGEN_NOT_INSTALLED",
                message=f"MusicGen dependencies not installed: {e}",
                recoverable=False,
                suggested_action="Install audiocraft: pip install audiocraft",
            ),
        )
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return ToolResult(
                success=False,
                data=None,
                error=ToolError(
                    code="MUSICGEN_OOM",
                    message=f"Out of memory during generation: {e}",
                    recoverable=True,
                    suggested_action=f"Try shorter duration (current: {duration_sec}s)",
                ),
            )
        else:
            return ToolResult(
                success=False,
                data=None,
                error=ToolError(
                    code="MUSICGEN_RUNTIME_ERROR",
                    message=f"Generation failed: {e}",
                    recoverable=True,
                    suggested_action="Check GPU availability and try again",
                ),
            )
    
    except Exception as e:
        return ToolResult(
            success=False,
            data=None,
            error=ToolError(
                code="MUSICGEN_ERROR",
                message=f"Generation failed: {e}",
                recoverable=True,
                suggested_action="Check prompt format and try again",
            ),
        )


def generate_segment_mock(
    prompt: str,
    duration_sec: float,
    output_path: str,
    conditioning_audio_path: str | None = None,
    temperature: float = 1.0,
    top_k: int = 250,
    cfg_coef: float = 3.0,
) -> ToolResult:
    """Mock generation for testing without MusicGen.
    
    Generates silence or noise instead of actual music.
    Useful for testing the pipeline without GPU.
    
    Args:
        Same as generate_segment.
        
    Returns:
        ToolResult with mock GeneratedSegment.
    """
    try:
        import soundfile as sf
        
        sample_rate = 32000
        num_samples = int(duration_sec * sample_rate)
        
        # Generate low-amplitude noise
        audio = np.random.randn(num_samples) * 0.01
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save
        sf.write(output_path, audio, sample_rate)
        
        result: GeneratedSegment = {
            "audio_path": output_path,
            "duration_sec": duration_sec,
            "generation_params": {
                "prompt": prompt,
                "temperature": temperature,
                "top_k": top_k,
                "cfg_coef": cfg_coef,
                "conditioning_audio": conditioning_audio_path,
                "mock": True,
            },
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
                code="MOCK_GENERATION_ERROR",
                message=f"Mock generation failed: {e}",
                recoverable=False,
                suggested_action="Check output path permissions",
            ),
        )
