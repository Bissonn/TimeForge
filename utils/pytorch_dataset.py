"""
PyTorch Dataset for Time Series Forecasting.

Wraps pre-prepared tensors (encoder inputs, targets, optional decoder inputs)
for efficient DataLoader usage with async prefetching and pin_memory.
"""
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple


class TimeSeriesForecastDataset(Dataset):
    """
    Dataset for time series forecasting with sliding windows.

    Stores tensors on CPU with contiguous memory layout for efficient
    GPU transfer via DataLoader with pin_memory.

    Args:
        encoder_inputs: Tensor of shape (N, window_size, num_features)
        targets: Tensor of shape (N, forecast_steps, num_targets)
        decoder_inputs: Optional tensor of shape (N, forecast_steps, num_decoder_features)
            Used for encoder-decoder architectures (None for encoder-only).
    """

    def __init__(
        self,
        encoder_inputs: torch.Tensor,
        targets: torch.Tensor,
        decoder_inputs: Optional[torch.Tensor] = None
    ):
        assert encoder_inputs.size(0) == targets.size(0), \
            f"Mismatch: encoder_inputs has {encoder_inputs.size(0)} samples, targets has {targets.size(0)}"

        if decoder_inputs is not None:
            assert encoder_inputs.size(0) == decoder_inputs.size(0), (
                f"Mismatch: encoder_inputs has {encoder_inputs.size(0)} samples, "
                f"decoder_inputs has {decoder_inputs.size(0)}"
            )

        # Store on CPU with contiguous layout for efficient pinned memory transfer
        # .contiguous() ensures sliced tensors don't have fragmented memory layout
        self.encoder_inputs = encoder_inputs.cpu().contiguous()
        self.targets = targets.cpu().contiguous()
        self.decoder_inputs = decoder_inputs.cpu().contiguous() if decoder_inputs is not None else None

    def __len__(self) -> int:
        return self.encoder_inputs.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            (encoder_input, target, decoder_input)
            decoder_input is None for encoder-only architectures
        """
        enc = self.encoder_inputs[idx]
        tgt = self.targets[idx]
        dec = self.decoder_inputs[idx] if self.decoder_inputs is not None else None

        return enc, tgt, dec


def collate_forecast_batch(batch):
    """
    Custom collate function for DataLoader.

    Handles optional decoder inputs (None for encoder-only models).

    Args:
        batch: List of tuples (encoder_input, target, decoder_input)

    Returns:
        Tuple of batched tensors: (encoder_batch, target_batch, decoder_batch)
        decoder_batch is None if all decoder_inputs are None
    """
    encoder_inputs, targets, decoder_inputs = zip(*batch)

    # Stack into batches
    encoder_batch = torch.stack(encoder_inputs, dim=0)
    target_batch = torch.stack(targets, dim=0)

    # Check if decoder inputs exist
    if decoder_inputs[0] is not None:
        decoder_batch = torch.stack(decoder_inputs, dim=0)
    else:
        decoder_batch = None

    return encoder_batch, target_batch, decoder_batch
