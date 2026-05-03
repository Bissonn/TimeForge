"""
Gradient flow monitoring for neural network training.

CSV-based streaming approach for crash resistance during gradient explosion experiments.
Includes logic verified by 'inspect_model_structure.py'.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GradientMonitor:
    """
    Monitor gradient flow during neural network training with CSV streaming.

    Architecture:
    - Append-only CSV writing for crash resistance
    - Component-wise gradient aggregation (Encoder vs Head)
    - Automatic flush to disk after each write
    """

    def __init__(
        self,
        model: nn.Module,
        save_dir: Path,
        model_name: str,
        fold_idx: int,
        window_size: int,
        model_type: str = 'auto',
        enabled: bool = True,
        log_interval: int = 1
    ):
        """
        Initialize gradient monitor with CSV streaming.
        """
        self.model = model
        self.save_dir = Path(save_dir)
        self.model_name = model_name
        self.fold_idx = fold_idx
        self.window_size = window_size
        self.model_type = self._detect_model_type(model, model_type)
        self.enabled = enabled
        self.log_interval = log_interval

        self.batch_count = 0
        self.csv_file = None
        self.csv_writer = None

        if self.enabled:
            self._initialize_csv()
            logger.info(
                f"[GradientMonitor] CSV streaming to: {self.csv_path} "
                f"(type={self.model_type}, interval={self.log_interval})"
            )

    def _detect_model_type(self, model: nn.Module, model_type: str) -> str:
        """Auto-detect model type if 'auto'."""
        if model_type != 'auto':
            return model_type

        model_class = model.__class__.__name__.lower()
        if 'lstm' in model_class or 'rnn' in model_class or 'gru' in model_class:
            return 'lstm'
        elif 'transformer' in model_class:
            return 'transformer'

        logger.warning(f"Could not auto-detect type for {model_class}, defaulting to 'lstm'")
        return 'lstm'

    def _initialize_csv(self):
        """Open CSV file and write header."""
        self.save_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{self.model_name}_fold_{self.fold_idx}_w{self.window_size}_gradients.csv"
        self.csv_path = self.save_dir / filename

        self.csv_file = open(self.csv_path, 'w', newline='')

        # Added global_step for continuous plotting
        self.fieldnames = [
            'epoch', 'step', 'global_step', 'batch_loss',
            'total_grad_norm', 'encoder_grad_norm', 'head_grad_norm'
        ]
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
        self.csv_writer.writeheader()
        self.csv_file.flush()

    @staticmethod
    def classify_parameters(
        model: nn.Module,
        model_type: str
    ) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        """
        Classify model parameters into encoder and head components.
        Logic verified by scripts/inspect_model_structure.py.
        """
        encoder_params = []
        head_params = []

        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            name_lower = name.lower()

            if model_type == 'lstm':
                # LSTM Head keywords (verified)
                if any(kw in name_lower for kw in ['fc', 'head', 'output_projection', 'final']):
                    head_params.append(param)
                else:
                    encoder_params.append(param)

            elif model_type == 'transformer':
                # Transformer Head keywords (verified)
                head_keywords = ['output_projection', 'fc_heads', 'final_linear', 'head']

                is_head = False
                # Crucial check: exclude attention projections (which have 'out_proj' or 'head' in attn)
                for kw in head_keywords:
                    if kw in name_lower and 'attn' not in name_lower:
                        is_head = True
                        break

                if is_head:
                    head_params.append(param)
                else:
                    encoder_params.append(param)

            else:
                # Default fallback
                encoder_params.append(param)

        return encoder_params, head_params

    @staticmethod
    def compute_component_norms(
        encoder_params: List[nn.Parameter],
        head_params: List[nn.Parameter]
    ) -> Tuple[float, float]:
        """Compute L2 norm of gradients for components."""
        def get_norm(params):
            if not params:
                return 0.0
            # Use detach() to avoid graph retention
            return torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2) for p in params]),
                2
            ).item()

        return get_norm(encoder_params), get_norm(head_params)

    def log_gradients(
        self,
        epoch: int,
        step: int,
        global_step: int,
        batch_loss: float,
        total_grad_norm: float,
        encoder_grad_norm: float,
        head_grad_norm: float
    ) -> Dict:
        """
        Log gradient statistics to CSV.
        """
        if not self.enabled:
            return {}

        self.batch_count += 1
        if self.batch_count % self.log_interval != 0:
            return {}

        row = {
            'epoch': epoch,
            'step': step,
            'global_step': global_step,
            'batch_loss': batch_loss,
            'total_grad_norm': total_grad_norm,
            'encoder_grad_norm': encoder_grad_norm,
            'head_grad_norm': head_grad_norm
        }

        self.csv_writer.writerow(row)
        self.csv_file.flush()  # Force write to disk

        return row

    def close(self) -> None:
        """Close CSV file."""
        if self.csv_file and not self.csv_file.closed:
            self.csv_file.close()
            logger.info(f"[GradientMonitor] Closed: {self.csv_path}")

    def __del__(self):
        if hasattr(self, 'csv_file') and self.csv_file and not self.csv_file.closed:
            try:
                self.csv_file.close()
            except Exception:
                pass