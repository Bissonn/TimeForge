import torch
from contextlib import nullcontext
from torch.amp import autocast
import logging

logger = logging.getLogger(__name__)


def get_autocast_context(device: torch.device, use_amp: bool, dtype=None):
    """
    Context manager for automatic mixed precision (AMP).

    Always creates a new autocast context when use_amp=True, even if one is already active.
    PyTorch handles nested autocast contexts correctly when using the same dtype.

    Args:
        device: The device to use for autocast (cuda, cpu, mps)
        use_amp: Whether to enable AMP (if False, returns nullcontext)
        dtype: Optional dtype override (torch.bfloat16 or torch.float16).
               If None, auto-detects using get_optimal_autocast_dtype()

    Returns:
        Context manager (autocast or nullcontext)

    Example:
        with get_autocast_context(self.device, use_amp=True):
            outputs = model(batch_x)
    """
    # If AMP is disabled, return nullcontext
    if not use_amp:
        return nullcontext()

    # Determine dtype if not provided
    if dtype is None:
        dtype = get_optimal_autocast_dtype(device)

    # Always create autocast context when use_amp=True
    # PyTorch handles nesting correctly with same dtype
    return autocast(device_type=device.type, dtype=dtype)


def get_optimal_autocast_dtype(device: torch.device) -> torch.dtype:
    """
    Returns the optimal autocast dtype based on device capabilities.

    This ensures consistency between training and inference:
    - If the hardware supports BF16 (e.g., Ampere GPUs), both will use it.
    - If not, both fallback to FP16.

    Args:
        device (torch.device): The device to check.

    Returns:
        torch.dtype: torch.bfloat16 or torch.float16
    """
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    elif device.type == "cpu":
        # Newer PyTorch versions support bfloat16 on CPU
        return torch.bfloat16

    elif device.type == "mps":
        # Apple Silicon (Metal Performance Shaders) typically uses float16
        return torch.float16

    # Default fallback
    return torch.float16