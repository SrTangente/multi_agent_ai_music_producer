"""Device detection and fallback utilities.

Handles TPU/GPU/CPU detection with automatic fallback for Google Colab
and local environments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any


class DeviceType(str, Enum):
    """Available device types for computation."""
    TPU = "tpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon
    CPU = "cpu"


@dataclass
class DeviceInfo:
    """Information about the detected device."""
    device_type: DeviceType
    device_name: str
    device_index: int | None
    memory_gb: float | None
    is_available: bool
    fallback_from: DeviceType | None = None


def _check_tpu_available() -> bool:
    """Check if TPU is available (Google Colab/Cloud TPU)."""
    try:
        # Check for TPU via torch_xla
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        return device is not None
    except (ImportError, RuntimeError):
        pass
    
    # Check environment variable (Colab sets this)
    return os.environ.get("COLAB_TPU_ADDR") is not None


def _check_cuda_available() -> tuple[bool, str | None, float | None]:
    """Check if CUDA GPU is available.
    
    Returns:
        Tuple of (is_available, device_name, memory_gb).
    """
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return True, device_name, memory_gb
    except (ImportError, RuntimeError):
        pass
    return False, None, None


def _check_mps_available() -> bool:
    """Check if Apple Silicon MPS is available."""
    try:
        import torch
        return torch.backends.mps.is_available()
    except (ImportError, AttributeError):
        return False


def get_device_info() -> DeviceInfo:
    """Detect the best available device.
    
    Checks in order: TPU → CUDA → MPS → CPU.
    
    Returns:
        DeviceInfo with details about the available device.
    """
    # Check TPU first (highest priority for Colab)
    if _check_tpu_available():
        return DeviceInfo(
            device_type=DeviceType.TPU,
            device_name="TPU",
            device_index=0,
            memory_gb=None,  # TPU memory varies
            is_available=True,
        )
    
    # Check CUDA GPU
    cuda_available, cuda_name, cuda_memory = _check_cuda_available()
    if cuda_available:
        return DeviceInfo(
            device_type=DeviceType.CUDA,
            device_name=cuda_name or "CUDA GPU",
            device_index=0,
            memory_gb=cuda_memory,
            is_available=True,
        )
    
    # Check Apple Silicon MPS
    if _check_mps_available():
        return DeviceInfo(
            device_type=DeviceType.MPS,
            device_name="Apple Silicon",
            device_index=None,
            memory_gb=None,
            is_available=True,
        )
    
    # Fallback to CPU
    return DeviceInfo(
        device_type=DeviceType.CPU,
        device_name="CPU",
        device_index=None,
        memory_gb=None,
        is_available=True,
    )


def get_available_device(
    preferred: DeviceType | None = None,
    fallback_chain: list[DeviceType] | None = None,
) -> DeviceInfo:
    """Get the best available device with optional preferences.
    
    Args:
        preferred: Preferred device type (will be used if available).
        fallback_chain: Custom fallback order. Default: [TPU, CUDA, MPS, CPU].
        
    Returns:
        DeviceInfo for the best available device.
    """
    if fallback_chain is None:
        fallback_chain = [DeviceType.TPU, DeviceType.CUDA, DeviceType.MPS, DeviceType.CPU]
    
    # If preferred device specified, try it first
    if preferred:
        fallback_chain = [preferred] + [d for d in fallback_chain if d != preferred]
    
    availability_checks = {
        DeviceType.TPU: lambda: _check_tpu_available(),
        DeviceType.CUDA: lambda: _check_cuda_available()[0],
        DeviceType.MPS: lambda: _check_mps_available(),
        DeviceType.CPU: lambda: True,
    }
    
    tried_devices: list[DeviceType] = []
    
    for device_type in fallback_chain:
        check_fn = availability_checks.get(device_type)
        if check_fn and check_fn():
            info = get_device_info()
            if tried_devices:
                info.fallback_from = tried_devices[0]
            return info
        tried_devices.append(device_type)
    
    # Should never reach here (CPU always available)
    return DeviceInfo(
        device_type=DeviceType.CPU,
        device_name="CPU",
        device_index=None,
        memory_gb=None,
        is_available=True,
    )


def get_torch_device(device_info: DeviceInfo | None = None) -> Any:
    """Get a torch device from DeviceInfo.
    
    Args:
        device_info: Optional DeviceInfo. If None, auto-detects.
        
    Returns:
        torch.device object.
    """
    import torch
    
    if device_info is None:
        device_info = get_device_info()
    
    if device_info.device_type == DeviceType.TPU:
        try:
            import torch_xla.core.xla_model as xm
            return xm.xla_device()
        except ImportError:
            # Fallback to CUDA or CPU
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
    
    elif device_info.device_type == DeviceType.CUDA:
        idx = device_info.device_index or 0
        return torch.device(f"cuda:{idx}")
    
    elif device_info.device_type == DeviceType.MPS:
        return torch.device("mps")
    
    else:
        return torch.device("cpu")


def estimate_max_audio_duration(
    device_info: DeviceInfo | None = None,
    model_size: str = "medium",
) -> float:
    """Estimate maximum audio duration that can be generated safely.
    
    Based on device memory and model size, estimates how long of an
    audio segment can be generated without OOM.
    
    Args:
        device_info: Device to estimate for. Auto-detects if None.
        model_size: MusicGen model size ("small", "medium", "large").
        
    Returns:
        Estimated maximum duration in seconds.
    """
    if device_info is None:
        device_info = get_device_info()
    
    # Base estimates for MusicGen (conservative)
    # MusicGen-melody medium uses ~4GB for 30s generation
    base_duration = {
        "small": 45.0,   # ~2GB model
        "medium": 30.0,  # ~4GB model
        "large": 20.0,   # ~8GB model
    }.get(model_size, 30.0)
    
    if device_info.device_type == DeviceType.CPU:
        # CPU is slower but less memory constrained
        return base_duration * 0.5  # Be conservative
    
    elif device_info.device_type == DeviceType.MPS:
        # Apple Silicon unified memory - usually good
        return base_duration
    
    elif device_info.device_type == DeviceType.CUDA:
        if device_info.memory_gb:
            # Scale based on available memory
            # Assume 4GB baseline for medium model
            memory_factor = device_info.memory_gb / 4.0
            return min(base_duration * memory_factor, 60.0)
        return base_duration
    
    elif device_info.device_type == DeviceType.TPU:
        # TPUs have lots of memory
        return min(base_duration * 2.0, 60.0)
    
    return base_duration


def is_running_in_colab() -> bool:
    """Check if running in Google Colab environment."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def get_environment_info() -> dict[str, Any]:
    """Get full environment information for logging/debugging.
    
    Returns:
        Dictionary with environment details.
    """
    import platform
    
    device_info = get_device_info()
    
    info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "is_colab": is_running_in_colab(),
        "device_type": device_info.device_type.value,
        "device_name": device_info.device_name,
        "device_memory_gb": device_info.memory_gb,
    }
    
    # Add torch version if available
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else None
    except ImportError:
        pass
    
    return info
