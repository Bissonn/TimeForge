"""
Attention capture components for Transformer models.

This module provides utilities for capturing attention weights during forward passes.
The captured weights can be saved to disk for analysis and visualization.
"""

from typing import Dict, List, Set, Optional
import torch
import torch.nn as nn


class AttentionCaptureBuffer:
    """
    Attention capture buffer with async CPU transfer support.
    Stores attention maps on GPU during forward passes; moved to CPU asynchronously.

    Saved tensors are expected as:
      - (B, H, Q, K) from nn.MultiheadAttention with average_attn_weights=False
      - we batch-average to (H, Q, K) immediately to keep files small

    Supports:
      - direct mode: keys like "enc_self_layer_0"
      - iterative sampling: keys like "enc_self_layer_3_step_64"
      - frequency-based capture: only capture every N calls (reduces overhead)
      - async CPU transfer: non-blocking GPU→CPU transfer for minimal training impact
    """
    def __init__(self, capture_frequency: int = 1):
        """
        Args:
            capture_frequency: Capture attention every N calls to store() (default=1, every call)
        """
        self.enabled: bool = False
        self.current_step: Optional[int] = None  # For iterative step suffixing (step_0, step_1, ...)
        self.steps_to_capture: Optional[Set[int]] = None
        self.capture_frequency: int = max(1, capture_frequency)  # Must be >= 1
        self._global_step_counter: int = 0  # Internal counter for frequency control
        self.data: Dict[str, torch.Tensor] = {}

    def configure(
        self,
        enabled: bool,
        steps_to_capture: Optional[List[int]] = None,
        capture_frequency: Optional[int] = None
    ) -> None:
        """
        Configure attention capture.

        Args:
            enabled: Enable/disable capture
            steps_to_capture: List of specific steps to capture (for iterative mode)
            capture_frequency: Capture every N calls (if None, keeps current value)
        """
        self.enabled = bool(enabled)
        self.current_step = None
        # Only clear data when ENABLING capture (to start fresh)
        # Don't clear when disabling, so captured data remains accessible
        if enabled:
            self.data.clear()
            self._global_step_counter = 0  # Reset counter on enable
        if steps_to_capture is None:
            self.steps_to_capture = None
        else:
            self.steps_to_capture = set(int(s) for s in steps_to_capture)
        if capture_frequency is not None:
            self.capture_frequency = max(1, capture_frequency)

    def set_step(self, step: Optional[int]) -> None:
        self.current_step = step

    def _should_store_for_step(self) -> bool:
        """
        Check if we should store attention for current step.

        Combines:
        - enabled flag
        - frequency-based capture (every N calls)
        - iterative step filtering (steps_to_capture)
        """
        if not self.enabled:
            return False

        # Frequency check: capture every N calls
        # Note: counter increments in store(), so check happens before increment
        if self._global_step_counter % self.capture_frequency != 0:
            return False

        # Iterative step filtering (if specified)
        if self.current_step is None:
            return True
        if self.steps_to_capture is None:
            return True
        return self.current_step in self.steps_to_capture

    def store(self, key: str, attn: Optional[torch.Tensor]) -> None:
        """
        Store attention weights with async CPU transfer.

        IMPORTANT: Tensors are transferred to CPU asynchronously (non_blocking=True).
        Call torch.cuda.synchronize() before converting to numpy to ensure transfer completion.

        Args:
            key: Storage key (e.g., "enc_self_layer_0")
            attn: Attention tensor (B, H, Q, K) or (H, Q, K)
        """
        # Always increment counter (for frequency tracking)
        should_store = self._should_store_for_step()
        self._global_step_counter += 1

        if not should_store:
            return
        if attn is None:
            return

        # Expected: (B, H, Q, K) when average_attn_weights=False
        # Check device BEFORE any operations (mean/detach might change tensor type)
        is_cuda = attn.is_cuda

        if attn.dim() == 4:
            # Batch average: (B, H, Q, K) -> (H, Q, K)
            attn = attn.detach().mean(dim=0)
        elif attn.dim() == 3:
            attn = attn.detach()
        else:
            return

        # Async CPU transfer (non-blocking) - minimal GPU overhead
        # CRITICAL: Must call torch.cuda.synchronize() before .numpy() conversion!
        if is_cuda:
            try:
                attn = attn.cpu(non_blocking=True)
            except TypeError:
                # Fallback: Some tensor types don't support non_blocking parameter
                # (e.g., after certain operations that change tensor properties)
                attn = attn.cpu()
        else:
            attn = attn.cpu()  # Already on CPU, no non_blocking needed

        if self.current_step is not None:
            key = f"{key}_step_{self.current_step}"
        self.data[key] = attn


class CapturingMHA(nn.Module):
    """
    Wraps nn.MultiheadAttention (or compatible) and captures attn weights into AttentionCaptureBuffer.
    IMPORTANT: This wrapper overrides need_weights based on capture.enabled, regardless of caller kwargs.
    """
    def __init__(self, mha: nn.MultiheadAttention, capture: AttentionCaptureBuffer, key_prefix: str):
        super().__init__()
        self.mha = mha
        self.capture = capture
        self.key_prefix = key_prefix

    def __repr__(self) -> str:
        enabled = bool(getattr(self.capture, "enabled", False))
        return f"CapturingMHA(key_prefix='{self.key_prefix}', enabled={enabled})"

    @property
    def batch_first(self):
        return getattr(self.mha, "batch_first", True)

    def forward(self, *args, **kwargs):
        need = bool(self.capture.enabled)
        kwargs["need_weights"] = need
        if need:
            kwargs["average_attn_weights"] = False
        out = self.mha(*args, **kwargs)
        # out: (attn_output, attn_weights)
        if isinstance(out, tuple) and len(out) >= 2 and need:
            self.capture.store(self.key_prefix, out[1])
        return out
