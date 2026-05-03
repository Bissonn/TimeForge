"""
Module for training PyTorch-based forecasting models.

This module provides a universal training loop with early stopping, supporting both standard and
encoder-decoder architectures for models like LSTM, Transformer. It includes advanced features
like Automated Mixed Precision (AMP), Adaptive Gradient Clipping, and Horizon degradation analysis.
"""
import logging
import time
from typing import Optional, Union, Any, Tuple, List, Dict
import numpy as np
import csv
import os
import torch
from torch import nn
from torch.amp import GradScaler
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from models.base import NeuralTSForecaster
from monitoring.gradient_monitor import GradientMonitor
from utils.logging_utils import log_training_start, log_training_success
from utils.scheduler_monitoring import SchedulerMonitor
from utils.amp_utils import get_autocast_context, get_optimal_autocast_dtype
from utils.logging_config import get_contextual_logger
from utils.pytorch_dataset import TimeSeriesForecastDataset, collate_forecast_batch
import platform
try:
    import optuna
except ImportError:
    optuna = None

logger = logging.getLogger(__name__)


def _count_open_fds() -> int:
    """Count open file descriptors for the current process via /proc/self/fd."""
    try:
        return len(os.listdir('/proc/self/fd'))
    except Exception:
        return -1


def _describe_open_fds() -> dict:
    """
    Categorize open file descriptors by type.
    Returns dict: {pipe, socket, file, anon, sem, other} counts + total.
    """
    counts = {'pipe': 0, 'socket': 0, 'file': 0, 'anon_inode': 0, 'shm': 0, 'other': 0}
    try:
        fd_dir = '/proc/self/fd'
        for fd_name in os.listdir(fd_dir):
            try:
                target = os.readlink(os.path.join(fd_dir, fd_name))
                if 'pipe:' in target:
                    counts['pipe'] += 1
                elif 'socket:' in target:
                    counts['socket'] += 1
                elif '/dev/shm' in target or target.startswith('/tmp/torch_'):
                    counts['shm'] += 1
                elif target.startswith('/'):
                    counts['file'] += 1
                elif 'anon_inode' in target:
                    counts['anon_inode'] += 1
                else:
                    counts['other'] += 1
            except (OSError, PermissionError):
                pass
    except Exception:
        pass
    counts['total'] = sum(counts.values())
    return counts


def _safe_get_lr(optimizer: torch.optim.Optimizer) -> Optional[float]:
    """
    Safely get current learning rate from an optimizer.
    Compatible with mocked optimizers used in unit tests.
    """
    try:
        param_groups = getattr(optimizer, "param_groups", None)
        if param_groups is None:
            return None
        if len(param_groups) == 0:
            return None
        group0 = param_groups[0]
        return group0.get("lr", None)
    except Exception:
        return None

def _scale_optimizer_lr(optimizer: torch.optim.Optimizer, factor: float, min_lr: float = 1e-8) -> None:
    # Scheduler-safe only for non-batch schedulers. Caller must ensure this.
    for pg in getattr(optimizer, "param_groups", []):
        lr = pg.get("lr", None)
        if lr is None:
            continue
        new_lr = max(float(lr) * float(factor), float(min_lr))
        pg["lr"] = new_lr

# --------------------------- Diagnostics helpers ---------------------------

def _finite_mask(t: torch.Tensor) -> torch.Tensor:
    # Works across torch versions without torch.nanmin/nanmax
    return torch.isfinite(t)


def _safe_stats(t: torch.Tensor) -> dict[str, float | int | tuple]:
    """
    Compute robust stats over finite values only.
    Returns shape, dtype, device, n_total, n_nonfinite, finite_ratio, min/max/mean/std.
    Never throws – on degenerate inputs returns zeros.
    """
    try:
        shape = tuple(t.shape)
        dtype = str(t.dtype).replace("torch.", "")
        device = str(t.device)
        n_total = t.numel()
        mask = _finite_mask(t)
        n_finite = int(mask.sum().detach().cpu())
        n_nonfinite = int(n_total - n_finite)

        if n_finite > 0:
            tf = t[mask]
            # detach once; reuse for stats
            tfc = tf.detach()
            # compute stats safely on CPU to avoid device mismatch in logs
            tfc_cpu = tfc.to("cpu", copy=False).float()
            t_min = float(tfc_cpu.min())
            t_max = float(tfc_cpu.max())
            t_mean = float(tfc_cpu.mean())
            t_std = float(tfc_cpu.std(unbiased=False))
        else:
            t_min = t_max = t_mean = t_std = 0.0

        finite_ratio = float(n_finite / max(n_total, 1))

        return {
            "shape": shape, "dtype": dtype, "device": device,
            "n_total": int(n_total), "n_nonfinite": int(n_nonfinite),
            "finite_ratio": finite_ratio,
            "min": t_min, "max": t_max, "mean": t_mean, "std": t_std,
        }
    except Exception:
        return {
            "shape": tuple(getattr(t, "shape", ())), "dtype": str(getattr(t, "dtype", "?")),
            "device": str(getattr(t, "device", "?")),
            "n_total": int(getattr(t, "numel", lambda: 0)()),
            "n_nonfinite": -1, "finite_ratio": 0.0,
            "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0,
        }


def _count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


# --------------------------------------------------------------------------
# --- Compatibility helper: safe min/max without torch.nanmin/torch.nanmax ---

def _safe_minmax(t: torch.Tensor) -> tuple[float, float]:
    """
    Return (min, max) over finite values of tensor `t`.
    Works on all torch versions (no reliance on torch.nanmin/torch.nanmax).
    If no finite values exist, returns (0.0, 0.0).
    """
    try:
        finite = torch.isfinite(t)
        if finite.any():
            t_fin = t[finite]
            t_min = float(t_fin.min().detach().cpu())
            t_max = float(t_fin.max().detach().cpu())
            return t_min, t_max
        else:
            return 0.0, 0.0
    except Exception:
        # Last-resort guard: never let diagnostics crash training
        return 0.0, 0.0

def _train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch_idx: int,
    scaler: Optional[GradScaler],
    use_amp: bool,
    autocast_dtype: torch.dtype,
    max_grad_norm: float,
    gradient_monitor: Optional[GradientMonitor],
    scheduler: Optional[_LRScheduler],
    scheduler_step_frequency: Optional[str],
    global_step: int,
    consecutive_nonfinite: int
) -> Tuple[float, float, int, int, Dict[str, int]]:
    """
    Executes one training epoch using PyTorch DataLoader.
    Returns: (avg_loss, avg_grad_norm, updated_global_step, updated_consecutive_nonfinite)

    Semantics:
      - global_step counts SUCCESSFUL optimizer updates only
      - avg_loss includes only successful updates (weighted by batch size)
      - consecutive_nonfinite increments on: non-finite loss, non-finite grad_norm, or GradScaler skip
      - scheduler (batch frequency) steps only on successful updates
    """
    model.train()
    total_loss = torch.zeros((), device=device)
    num_successful_samples = 0

    total_grad_norm = 0.0
    num_grad_batches = 0
    use_grad_clipping = max_grad_norm is not None and max_grad_norm > 0

    skip_stats = {
        "total_batches": 0,
        "skipped_loss": 0,
        "skipped_grad_norm": 0,
        "skipped_scaler": 0,
    }

    for batch_idx, (batch_enc, batch_y, batch_dec) in enumerate(data_loader):
        skip_stats["total_batches"] += 1

        # Move to device with non_blocking for async transfer (key for pin_memory efficiency)
        batch_enc = batch_enc.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        if batch_dec is not None:
            batch_dec = batch_dec.to(device, non_blocking=True)

        curr_bs = int(batch_enc.size(0))

        optimizer.zero_grad(set_to_none=True)

        # Use smart autocast context that avoids nesting
        with get_autocast_context(device, use_amp=use_amp, dtype=autocast_dtype if use_amp else None):
            outputs = model(batch_enc, batch_dec) if batch_dec is not None else model(batch_enc)
            loss = loss_fn(outputs.float(), batch_y.float())

        # Cheap scalar check (no full-tensor scans)
        if not torch.isfinite(loss):
            if gradient_monitor:
                # Log NaN for visibility
                gradient_monitor.log_gradients(
                    epoch=epoch_idx + 1, step=batch_idx, global_step=global_step,
                    batch_loss=float('nan'), total_grad_norm=float('nan'),
                    encoder_grad_norm=float('nan'), head_grad_norm=float('nan')
                )

            logger.warning(f"Non-finite loss detected at epoch {epoch_idx + 1}, batch {batch_idx}. Skipping.")

            consecutive_nonfinite += 1
            skip_stats["skipped_loss"] += 1
            continue

        # Backward (AMP-safe): unscale BEFORE clipping
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        grad_norm_val: Optional[float] = None

        # Clip + validate grad norm
        if use_grad_clipping:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            grad_norm_val = float(grad_norm.detach().cpu())
            if not np.isfinite(grad_norm_val):
                consecutive_nonfinite += 1
                logger.warning(
                    f"Non-finite grad_norm detected at epoch {epoch_idx + 1}, batch {batch_idx}. "
                    f"Skipping step. lr={_safe_get_lr(optimizer)}"
                )
                # Keep GradScaler healthy even when skipping
                if scaler is not None:
                    scaler.update()
                if gradient_monitor:
                    gradient_monitor.log_gradients(
                        epoch = epoch_idx + 1, step = batch_idx, global_step = global_step,
                        batch_loss = float(loss.detach().cpu()),
                        total_grad_norm = float("nan"),
                        encoder_grad_norm = float("nan"), head_grad_norm = float("nan"),
                    )
                skip_stats["skipped_grad_norm"] += 1
                continue

            total_grad_norm += float(grad_norm_val)
            num_grad_batches += 1

        # Gradient monitoring (optional): reuse grad_norm if available
        if gradient_monitor:
            if grad_norm_val is None:
                # Compute one-pass norm, single sync at the end
                sq_sum = torch.zeros((), device=device)
                for p in model.parameters():
                    g = p.grad
                    if g is None:
                        continue
                    sq_sum = sq_sum + (g.detach().float().pow(2).sum())
                total_norm = torch.sqrt(sq_sum)
                grad_norm_val = float(total_norm.detach().cpu()) if torch.isfinite(total_norm) else float(
                "nan")

            enc_params, head_params = GradientMonitor.classify_parameters(model,gradient_monitor.model_type)
            enc_norm, head_norm = GradientMonitor.compute_component_norms(enc_params, head_params)
            gradient_monitor.log_gradients(
                epoch = epoch_idx + 1, step = batch_idx, global_step = global_step,
                batch_loss = float(loss.detach().cpu()),
                total_grad_norm = float(grad_norm_val) if grad_norm_val is not None else float("nan"),
                encoder_grad_norm = enc_norm, head_grad_norm = head_norm
            )

        # DEBUG: Check params before optimizer step
        if batch_idx == 0 and epoch_idx == 0:
            first_param = next(model.parameters())
            param_before = first_param.data.clone()

        # Optimizer step + detect GradScaler skip
        step_applied = True
        if scaler is not None:
            prev_scale = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            new_scale = float(scaler.get_scale())
            if new_scale < prev_scale:
                step_applied = False
                consecutive_nonfinite += 1
                logger.warning(
                    f"GradScaler skipped optimizer step at epoch {epoch_idx + 1}, batch {batch_idx} "
                    f"(scale {prev_scale:.1f} -> {new_scale:.1f}). lr={_safe_get_lr(optimizer)}"
                )
                skip_stats["skipped_scaler"] += 1
        else:
            optimizer.step()

        # DEBUG: Check if params changed after optimizer step
        if batch_idx == 0 and epoch_idx == 0:
            param_after = first_param.data
            params_changed = not torch.equal(param_before, param_after)
            max_change = (param_after - param_before).abs().max().item() if params_changed else 0.0
            logger.debug(
                f"[OPTIMIZER DEBUG] Batch 0, Epoch 0: params_changed={params_changed}, "
                f"max_change={max_change:.6e}, loss={loss.item():.6f}"
            )

        if not step_applied:
            # No scheduler step, no global_step update, no loss stats update
            continue

        if scheduler is not None and scheduler_step_frequency == 'batch':
            if batch_idx == 0 and epoch_idx == 0:  # Log only first batch of first epoch
                lr_before = _safe_get_lr(optimizer)
                logger.debug(f"[DEBUG_SCHEDULER] First scheduler.step() call: lr_before={lr_before:.2e}")
            scheduler.step()
            if batch_idx == 0 and epoch_idx == 0:  # Log after step
                lr_after = _safe_get_lr(optimizer)
                logger.debug(f"[DEBUG_SCHEDULER] After first scheduler.step(): lr_after={lr_after:.2e}")

        # Commit successful update
        global_step += 1
        consecutive_nonfinite = 0
        total_loss += loss.detach() * curr_bs
        num_successful_samples += curr_bs

    avg_loss = (total_loss / max(num_successful_samples, 1)).item()
    avg_grad = (total_grad_norm / max(num_grad_batches, 1)) if num_grad_batches > 0 else 0.0
    return avg_loss, avg_grad, global_step, consecutive_nonfinite, skip_stats


def _validate_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool,
    autocast_dtype: torch.dtype,
    scaler_params: Optional[Dict[str, np.ndarray]] = None
) -> Tuple[float, Optional[np.ndarray]]:
    """
    Executes one validation epoch using PyTorch DataLoader.

    Args:
        scaler_params: Optional dict with 'mean' and 'scale' arrays for fast inverse scaling.
                       If provided, validation loss will be computed in ORIGINAL scale (not scaled).
                       This ensures consistency with final evaluation metrics.

    Returns: (val_loss, horizon_mse_per_step)
    """
    model.eval()
    val_loss_sum = 0.0
    val_horizon_sq_err = None
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (v_enc, v_y, v_dec) in enumerate(data_loader):
            # Move to device with non_blocking
            v_enc = v_enc.to(device, non_blocking=True)
            v_y = v_y.to(device, non_blocking=True)
            if v_dec is not None:
                v_dec = v_dec.to(device, non_blocking=True)

            curr_bs = v_enc.size(0)

            with get_autocast_context(device, use_amp=use_amp, dtype=autocast_dtype if use_amp else None):
                v_out = model(v_enc, v_dec) if v_dec is not None else model(v_enc)
                if not torch.isfinite(v_out).all(): continue

                # DEBUG: Log first validation prediction
                if batch_idx == 0:
                    logger.debug(
                        f"[VAL DEBUG] v_enc[0] mean: {v_enc[0].float().cpu().mean().item():.6f}, "
                        f"shape: {v_enc[0].shape}"
                    )
                    logger.debug(f"[VAL DEBUG] v_out[0]: {v_out[0].float().cpu().numpy()}")
                    logger.debug(f"[VAL DEBUG] v_y[0]: {v_y[0].float().cpu().numpy()}")

                # ═══════════════════════════════════════════════════════════
                # COMPUTE LOSS IN ORIGINAL SCALE (if scaler_params provided)
                # ═══════════════════════════════════════════════════════════
                # CRITICAL: Production frameworks (PyTorch Forecasting, Darts, GluonTS)
                # always compute validation metrics in original scale for interpretability
                # and consistency with final evaluation. This ensures:
                #   1. Early stopping uses same scale as final test metrics
                #   2. HPO optimization target matches business metrics
                #   3. Validation loss is interpretable (e.g., "MSE=12°C²")
                if scaler_params is not None:
                    # Fast vectorized inverse scaling (O(1), no pandas overhead)
                    v_out_np = v_out.float().cpu().numpy()  # (BS, H, F)
                    v_y_np = v_y.float().cpu().numpy()

                    # Apply inverse: x_orig = x_scaled * scale + mean
                    # Broadcasting: (BS, H, F) * (F,) + (F,) = (BS, H, F)
                    v_out_orig = v_out_np * scaler_params['scale'] + scaler_params['mean']
                    v_y_orig = v_y_np * scaler_params['scale'] + scaler_params['mean']

                    # MSE in original scale
                    loss = float(np.mean((v_out_orig - v_y_orig) ** 2))

                    # Horizon MSE also in original scale
                    if v_out_orig.shape[1] > 1:  # Multi-horizon
                        # Per-horizon MSE: (BS, H, F) -> mean over F, sum over BS -> (H,)
                        batch_sq_orig = np.mean((v_out_orig - v_y_orig) ** 2, axis=-1).sum(axis=0)  # (H,)
                        batch_sq_orig_tensor = torch.from_numpy(batch_sq_orig).to(device)
                        if val_horizon_sq_err is None:
                            val_horizon_sq_err = batch_sq_orig_tensor
                        else:
                            val_horizon_sq_err += batch_sq_orig_tensor
                else:
                    # Fallback: scaled space (backward compatibility)
                    loss = loss_fn(v_out.float(), v_y.float()).item()

                    # Horizon MSE in scaled space (original behavior)
                    if v_out.dim() == 3 and v_out.size(1) > 1:
                        batch_sq = (v_out.float() - v_y.float()).pow(2).mean(dim=-1).sum(dim=0)
                        if val_horizon_sq_err is None:
                            val_horizon_sq_err = batch_sq
                        else:
                            val_horizon_sq_err += batch_sq

            val_loss_sum += loss * curr_bs
            total_samples += curr_bs

    avg_loss = float("inf") if total_samples == 0 else val_loss_sum / total_samples
    horizon_mse = (val_horizon_sq_err / total_samples).cpu().numpy() if val_horizon_sq_err is not None else None
    return avg_loss, horizon_mse


def run_train_loop(
    model: Union[nn.Module, NeuralTSForecaster],
    encoder_inputs_train: torch.Tensor,
    decoder_inputs_train: Optional[torch.Tensor],
    true_outputs_train: torch.Tensor,
    encoder_inputs_val: Optional[torch.Tensor],
    decoder_inputs_val: Optional[torch.Tensor],
    true_outputs_val: Optional[torch.Tensor],
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    early_stopping_patience: Optional[int],  # Allow None for flexibility
    device: torch.device,
    scheduler: Optional[_LRScheduler] = None,
    batch_size: int = 32,
    model_name: str = "unknown",
    min_epochs: int = 5,
    use_amp: bool = True,  # Enable AMP by default; safe on CUDA (fp16/bf16) and CPU (bf16)
    max_grad_norm: float = 1.0,  # Gradient clipping for stability
    log_every: int = 1,  # Throttle logging if needed
    save_horizon_csv: bool = False,  # Export horizon metrics to CSV
    horizon_csv_path: Optional[str] = None,  # Path for CSV file
    auto_tune_horizon: bool = False,  # Automatic horizon tuning based on degradation
    degradation_threshold: float = 3.0,  # Threshold for severe degradation
    optuna_trial: Optional[Any] = None,
    trial_step_offset: int = 0,
    gradient_monitor: Optional[GradientMonitor] = None,  # Gradient monitoring
    save_scheduler_plot: bool = False,
    save_scheduler_csv: bool = False,
    run_context: Optional['RunContext'] = None,
    fail_on_numerical_instability: bool = False,
    # DataLoader workers (0=main process only, safer for HPO; >0 faster but uses file descriptors)
    num_workers: int = 2,
    # Fast inverse scaling params for original-scale validation metrics
    scaler_params: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[Union[nn.Module, NeuralTSForecaster], Dict[str, List[float]]]:
    """
    Run a training loop with early stopping, supporting standard and encoder-decoder models.

    Args:
        scaler_params: Optional dict with 'mean' and 'scale' arrays from preprocessor.
                       If provided, validation metrics will be computed in ORIGINAL scale
                       (following best practices from PyTorch Forecasting, Darts, GluonTS).
                       This ensures early stopping and HPO use interpretable, business-relevant metrics.
    """
    # ── FD DIAGNOSTICS ──────────────────────────────────────────────────────
    _fd_entry = _count_open_fds()
    _fd_types_entry = _describe_open_fds()
    logger.debug(
        f"[FD] run_train_loop ENTRY: {_fd_entry} fds "
        f"(pipe={_fd_types_entry['pipe']}, shm={_fd_types_entry['shm']}, "
        f"file={_fd_types_entry['file']}, anon={_fd_types_entry['anon_inode']}, "
        f"other={_fd_types_entry['other']}) | num_workers={num_workers}"
    )
    # ────────────────────────────────────────────────────────────────────────

    if not isinstance(model, (nn.Module, NeuralTSForecaster)):
        raise ValueError("model must be a torch.nn.Module or NeuralTSForecaster.")
    if not isinstance(encoder_inputs_train, torch.Tensor) or encoder_inputs_train.numel() == 0:
        raise ValueError("encoder_inputs_train must be a non-empty torch.Tensor.")
    if not isinstance(true_outputs_train, torch.Tensor) or true_outputs_train.numel() == 0:
        raise ValueError("true_outputs_train must be a non-empty torch.Tensor.")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer.")

    if early_stopping_patience is not None:
        if not isinstance(early_stopping_patience, int) or early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be a non-negative integer or None.")

    if not isinstance(min_epochs, int) or min_epochs < 1:
        raise ValueError("min_epochs must be a positive integer.")
    if not model_name:
        raise ValueError("model_name cannot be empty.")

    # Normalize loss behavior for comparability
    if isinstance(loss_fn, nn.MSELoss) and getattr(loss_fn, "reduction", "mean") != "mean":
        logger.warning(f"MSELoss reduction is '{loss_fn.reduction}'; switching to 'mean' for stability/scale.")
        try:
            loss_fn.reduction = "mean"
        except Exception:
            loss_fn = nn.MSELoss(reduction="mean")

    # --- Generic normalization for other losses (Huber, L1/MAE) ---
    # We want to enforce 'mean' reduction for stability, regardless of the loss type.
    if hasattr(loss_fn, "reduction") and getattr(loss_fn, "reduction", "mean") != "mean":
        logger.warning(
            f"Loss function {type(loss_fn).__name__} reduction is '{loss_fn.reduction}'; switching to 'mean'.")
        try:
            loss_fn.reduction = "mean"
        except Exception:
            # Fallback: if we can't mutate, we assume the user knows what they are doing
            # or we could try to re-instantiate if we knew the constructor args.
            # For now, logging the warning is safer than crashing.
            logger.warning(
                f"Could not enforce 'mean' reduction on {type(loss_fn).__name__}. Gradients might be unstable.")

    # Validate tensor shapes
    if encoder_inputs_train.shape[0] != true_outputs_train.shape[0]:
        raise ValueError(
            f"encoder_inputs_train ({encoder_inputs_train.shape}) and true_outputs_train "
            f"({true_outputs_train.shape}) must have the same number of samples."
        )
    if decoder_inputs_train is not None and decoder_inputs_train.shape[0] != encoder_inputs_train.shape[0]:
        raise ValueError(
            f"decoder_inputs_train ({decoder_inputs_train.shape}) must have the same number of "
            f"samples as encoder_inputs_train ({encoder_inputs_train.shape})."
        )

    # Validate validation data
    has_validation_data = (
        encoder_inputs_val is not None
        and true_outputs_val is not None
        and encoder_inputs_val.numel() > 0
        and true_outputs_val.numel() > 0
        and (decoder_inputs_val is None or decoder_inputs_val.numel() > 0)
    )

    if has_validation_data:
        if encoder_inputs_val.shape[0] != true_outputs_val.shape[0]:
            raise ValueError(
                f"encoder_inputs_val ({encoder_inputs_val.shape}) and true_outputs_val "
                f"({true_outputs_val.shape}) must have the same number of samples."
            )
        if decoder_inputs_val is not None and decoder_inputs_val.shape[0] != encoder_inputs_val.shape[0]:
            raise ValueError(
                f"decoder_inputs_val ({decoder_inputs_val.shape}) must have the same number of "
                f"samples as encoder_inputs_val ({encoder_inputs_val.shape})."
            )

    # Move model and tensors to device
    # Detect LSTM + CPU incompatibility with AMP (PyTorch oneDNN bug)
    if device.type == 'cpu' and use_amp:
        is_lstm = hasattr(model, 'lstm') or 'LSTM' in model.__class__.__name__
        if is_lstm:
            logger.warning(
                "[LSTM CPU FIX] Disabling AMP for LSTM on CPU due to PyTorch/oneDNN "
                "compatibility issues (known PyTorch bug with bfloat16 + LSTM + CPU)"
            )
            use_amp = False


    # Init Automated Mixed Precision
    amp_enabled = use_amp
    scaler = None
    autocast_dtype = None  # Will be auto-detected by get_autocast_context

    if amp_enabled:
        if device.type == 'cuda':
            autocast_dtype = get_optimal_autocast_dtype(device)
            if autocast_dtype == torch.float16:
                # Only FP16 needs GradScaler, BF16 doesn't
                scaler = GradScaler()
                logger.info("AMP enabled on CUDA with float16 and GradScaler.")
            else:
                logger.info("AMP enabled on CUDA with bfloat16 (no scaler needed).")
        elif device.type == 'cpu':
            autocast_dtype = torch.bfloat16
            logger.info("AMP enabled on CPU with bfloat16 (no scaler needed).")
        else:
            amp_enabled = False
            logger.warning(f"AMP requested but unsupported on {device.type}. Disabling.")
    else:
        logger.info("AMP disabled.")

    model.to(device)

    # ───────────────────────────────────────────────────────────────────────
    # CREATE DATALOADERS (ONCE, before epoch loop)
    # ───────────────────────────────────────────────────────────────────────
    # Data stays on CPU; DataLoader with pin_memory handles async GPU transfer

    # Training DataLoader (shuffle=True for stateless models)
    train_dataset = TimeSeriesForecastDataset(
        encoder_inputs_train, true_outputs_train, decoder_inputs_train
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Recommended for Transformer/LSTM without state carryover
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),  # Only beneficial for CUDA
        persistent_workers=(num_workers > 0),  # Keep workers alive between epochs
        collate_fn=collate_forecast_batch,
        multiprocessing_context='spawn' if num_workers > 0 else None  # Avoid fork() deprecation warnings
    )

    # Validation DataLoader (shuffle=False - MANDATORY for evaluation)
    val_loader = None
    if has_validation_data:
        val_dataset = TimeSeriesForecastDataset(
            encoder_inputs_val, true_outputs_val, decoder_inputs_val
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # NEVER shuffle validation/test data
            num_workers=num_workers,
            pin_memory=(device.type == 'cuda'),
            persistent_workers=(num_workers > 0),
            collate_fn=collate_forecast_batch,
            multiprocessing_context='spawn' if num_workers > 0 else None  # Avoid fork() deprecation warnings
        )
    # ───────────────────────────────────────────────────────────────────────

    # ── FD DIAGNOSTICS ──────────────────────────────────────────────────────
    _fd_after_dl = _count_open_fds()
    logger.debug(
        f"[FD] After DataLoader creation: {_fd_after_dl} fds "
        f"(delta from entry: +{_fd_after_dl - _fd_entry}) "
        f"| train_loader._iterator={'set' if getattr(train_loader, '_iterator', None) is not None else 'None'}"
    )
    # ────────────────────────────────────────────────────────────────────────

    # Check for potentially dangerous configuration: Early Stopping without Validation
    if early_stopping_patience is not None and not has_validation_data:
        logger.warning(
            "Early stopping enabled without validation data. "
            "Will monitor TRAINING LOSS, which may cause overfitting. "
            "Consider providing validation data or disabling early stopping."
        )

    log_training_start(model_name, model)
    # ───────────────────────────────────────────────────────────────────────
    # SCHEDULER SETUP
    # ───────────────────────────────────────────────────────────────────────
    # Determine scheduler step frequency and whether it requires validation metric

    from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, ReduceLROnPlateau

    scheduler_step_frequency = None  # None = per-epoch, 'batch' = per-batch
    scheduler_requires_metric = False

    if scheduler is not None:
        scheduler_name = scheduler.__class__.__name__

        # ReduceLROnPlateau needs validation metric
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler_requires_metric = True
            scheduler_step_frequency = None  # Per-epoch with metric
            logger.info(f"Scheduler {scheduler_name}: per-epoch with validation metric")

        # OneCycleLR and CosineAnnealingLR are stepped per-batch
        elif isinstance(scheduler, (OneCycleLR, CosineAnnealingLR)):
            scheduler_step_frequency = 'batch'
            logger.debug(
                f"[DEBUG_SCHEDULER] Scheduler {scheduler_name}: per-batch, "
                f"initial_lr={_safe_get_lr(optimizer):.2e}"
            )

        # Other schedulers (StepLR, ExponentialLR, etc) are per-epoch
        else:
            scheduler_step_frequency = None  # Per-epoch
            logger.info(f"Scheduler {scheduler_name}: per-epoch")
    else:
        logger.debug("[DEBUG_SCHEDULER] NO SCHEDULER PROVIDED (scheduler=None) - using constant LR")

    # ───────────────────────────────────────────────────────────────────────

    # ============================ ONE-TIME DIAGNOSTICS ============================
    try:
        logger.debug("=== Diagnostics @train_loop entry ===")
        logger.debug(
            f"Env: torch={torch.__version__}, cuda_available={torch.cuda.is_available()}, "
            f"device={device}, python={platform.python_version()}, "
            f"amp_enabled={amp_enabled}, autocast_dtype={autocast_dtype}, "
            f"max_grad_norm={max_grad_norm}, epochs={epochs}, batch_size={batch_size}, log_every={log_every}"
        )
        # (Omitted redundant logs for brevity, assume _safe_stats is used here as in prompt)
        try:
            enc_tr_stats = _safe_stats(encoder_inputs_train)
            y_tr_stats = _safe_stats(true_outputs_train)
            logger.debug(f"Train encoder stats: {enc_tr_stats}")
            logger.debug(f"Train target stats: {y_tr_stats}")
        except Exception:
            pass
        logger.debug("=== End diagnostics ===")
    except Exception as _:
        logger.warning("Diagnostics failed (non-fatal). Continuing training.")
    # =============================================================================

    use_grad_clipping = max_grad_norm is not None and max_grad_norm > 0
    best_val_loss = float("inf")
    best_epoch = 0
    best_model_state = None
    epochs_no_improve = 0
    consecutive_nonfinite = 0

    # === 1. Init history ===
    history = {
        "train_loss": [],
        "val_loss": []
    }

    # ─────────────────────────────────────────────────────────────────────
    # SCHEDULER MONITORING SETUP
    # ─────────────────────────────────────────────────────────────────────
    scheduler_monitor = None
    scheduler_plot_path = None

    if scheduler is not None and save_scheduler_plot:
        # Construct path using run_context (similar to gradient_monitor pattern)
        if run_context:
            # Use plots_dir from run_context (similar to gradients_dir)
            plots_dir = run_context.plots_dir if hasattr(run_context, 'plots_dir') else None

            if plots_dir is None:
                # Fallback: construct from run_dir
                plots_dir = run_context.run_dir / "plots" if hasattr(run_context, 'run_dir') else None

            if plots_dir:
                os.makedirs(plots_dir, exist_ok=True)
                scheduler_plot_path = plots_dir / f"{model_name}_lr_schedule.png"
                scheduler_monitor = SchedulerMonitor()
                logger.info(f"Scheduler monitoring enabled, will save to: {scheduler_plot_path}")
            else:
                logger.warning("run_context provided but no plots_dir available")
        else:
            logger.warning("save_scheduler_plot=True but no run_context provided, skipping monitoring")
    # ─────────────────────────────────────────────────────────────────────

    # --- HORIZON TRACKING INIT ---
    csv_file_handle = None
    csv_writer = None
    if save_horizon_csv and horizon_csv_path:
        os.makedirs(os.path.dirname(horizon_csv_path) if os.path.dirname(horizon_csv_path) else ".", exist_ok=True)
        csv_file_handle = open(horizon_csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file_handle)
        logger.info(f"Horizon metrics will be saved to: {horizon_csv_path}")

    best_horizon_mse = None
    best_degradation_ratio = None
    suggested_horizon = None
    global_step = 0

    try:
        for epoch in range(epochs):
            epoch_start_time = time.time()

            # --- TRAINING PHASE ---
            # === 1. train one epoch ===
            train_loss, avg_grad_norm, global_step, consecutive_nonfinite, skip_stats = _train_one_epoch(
                model, train_loader,
                optimizer, loss_fn, device, epoch,
                scaler, amp_enabled, autocast_dtype, max_grad_norm,
                gradient_monitor, scheduler, scheduler_step_frequency, global_step,
                consecutive_nonfinite
            )

            # Skip rate diagnostics (actionable, cheap)
            total_batches = max(int(skip_stats.get("total_batches", 0)), 1)
            skipped_total = int(skip_stats.get("skipped_loss", 0)) + int(
                skip_stats.get("skipped_grad_norm", 0)) + int(skip_stats.get("skipped_scaler", 0))
            skip_rate = skipped_total / total_batches
            if skipped_total > 0:
                msg = (
                    f"[Numerical instability] epoch={epoch + 1} skip_rate={skip_rate:.1%} "
                    f"(skipped_total={skipped_total}/{total_batches}, "
                    f"loss={skip_stats.get('skipped_loss', 0)}, "
                    f"grad_norm={skip_stats.get('skipped_grad_norm', 0)}, "
                    f"scaler={skip_stats.get('skipped_scaler', 0)}) "
                    f"amp_enabled={amp_enabled} lr={_safe_get_lr(optimizer)}"
                )
                logger.warning(f"[DIVERGED] {msg}")
                if fail_on_numerical_instability:
                    raise RuntimeError(msg)

            # -------------------- Recovery policy --------------------
            # Notes:
            # - We DO NOT mutate LR under batch-based schedulers (OneCycle/CosineAnnealing),
            # because it breaks scheduler semantics. In that case we use AMP-off first,
            # then abort/prune if instability persists.
            # - Under epoch-based/no scheduler, we can backoff LR before disabling AMP.
            if consecutive_nonfinite > 0:
                logger.warning(
                    f"[Stability] epoch={epoch + 1} consecutive_nonfinite={consecutive_nonfinite} "
                    f"amp_enabled={amp_enabled} lr={_safe_get_lr(optimizer)} "
                    f"scheduler_step_frequency={scheduler_step_frequency}"
                )

            if consecutive_nonfinite >= 3:
                if scheduler_step_frequency == "batch":
                    # Do not touch LR; first try disabling AMP to remove overflow/scale issues
                    if amp_enabled:
                        logger.warning("[Stability] Persistent instability under batch-scheduler. Disabling AMP.")
                        amp_enabled = False
                        scaler = None
                        consecutive_nonfinite = 0
                    else:
                        # Still unstable without AMP -> abort/prune
                        msg = "[Stability] Persistent instability even with AMP disabled under batch-scheduler."
                        if optuna_trial is not None and optuna is not None:
                            logger.warning(msg + " Pruning trial.")
                            raise optuna.TrialPruned()
                        raise RuntimeError(msg)
                else:
                    # Epoch-based/no scheduler: LR backoff first, then AMP-off, then abort
                    if amp_enabled:
                        logger.warning("[Stability] Persistent instability. Applying LR backoff x0.5 (AMP kept).")
                        _scale_optimizer_lr(optimizer, factor=0.5, min_lr=1e-8)
                        consecutive_nonfinite = 0
                    else:
                        msg = "[Stability] Persistent instability with AMP disabled. Aborting."
                        if optuna_trial is not None and optuna is not None:
                            logger.warning(msg + " Pruning trial.")
                            raise optuna.TrialPruned()
                        raise RuntimeError(msg)

            # ---------------------------------------------------------

            # === 2. append train_loss ===
            history["train_loss"].append(train_loss)

            # --- VALIDATION PHASE ---
            val_loss = None
            if val_loader is not None:
                val_loss, horizon_mse = _validate_one_epoch(
                    model, val_loader,
                    loss_fn, device, amp_enabled, autocast_dtype,
                    scaler_params=scaler_params
                )

                if not np.isfinite(val_loss):
                    logger.warning(f"Non-finite validation loss at epoch {epoch + 1}: {val_loss}")

                # =========================================================
                # OPTUNA INTEGRATION (PRUNING)
                # =========================================================
                if optuna_trial is not None and optuna is not None:
                    # 1. Report the current metric to Optuna
                    current_step = trial_step_offset + epoch
                    optuna_trial.report(val_loss, current_step)

                    # 2. Check if the trial should be pruned (stopped early)
                    if optuna_trial.should_prune():
                        logger.info(f"Trial pruned by Optuna at epoch {epoch + 1} (Val Loss: {val_loss:.6f})")
                        # Raise standard Optuna exception to be caught in the optimization loop
                        raise optuna.TrialPruned()
                # =========================================================

                # --- HORIZON ANALYSIS (Post-Loop) ---
                if horizon_mse is not None:
                    degradation_ratio = horizon_mse[-1] / (horizon_mse[0] + 1e-8)

                    # CSV Export
                    if csv_writer is not None:
                        if csv_file_handle.tell() == 0:
                            header = ["epoch", "val_loss", "degradation_ratio"] + \
                                     [f"step_{k+1}_mse" for k in range(len(horizon_mse))]
                            csv_writer.writerow(header)
                        row = [epoch + 1, val_loss, degradation_ratio] + list(horizon_mse)
                        csv_writer.writerow(row)
                        csv_file_handle.flush()

                    # Track best
                    if val_loss < best_val_loss:
                        best_horizon_mse = horizon_mse.copy()
                        best_degradation_ratio = degradation_ratio

                        if auto_tune_horizon and degradation_ratio > degradation_threshold:
                            threshold_mse = degradation_threshold * horizon_mse[0]
                            exceeds = np.where(horizon_mse > threshold_mse)[0]
                            if len(exceeds) > 0:
                                suggested_horizon = int(exceeds[0])

                    # Log details
                    should_log_horizon = ((epoch + 1) % 10 == 0 or val_loss < best_val_loss)
                    if should_log_horizon and len(horizon_mse) > 1:
                        mse_str = ", ".join([f"{x:.4f}" for x in horizon_mse[:5]])
                        if len(horizon_mse) > 5:
                            mse_str += f", ..., {horizon_mse[-1]:.4f}"
                        logger.info(f"Epoch {epoch+1} Horizon MSE: [{mse_str}]")
            else:
                val_loss = None

            # Appending val_loss
            history["val_loss"].append(val_loss)

            # ─────────────────────────────────────────────────────────
            # LOG LR TO SCHEDULER MONITOR
            # ─────────────────────────────────────────────────────────
            if scheduler_monitor is not None:
                current_lr = _safe_get_lr(optimizer)
                scheduler_monitor.log_lr(
                    epoch=epoch,
                    lr=current_lr,
                    train_loss=train_loss,
                    val_loss=val_loss
                )
            # ─────────────────────────────────────────────────────────

            # ─────────────────────────────────────────────────────────────
            # SCHEDULER STEP (per-epoch schedulers)
            # ─────────────────────────────────────────────────────────────
            if scheduler is not None and scheduler_step_frequency != 'batch':
                if scheduler_requires_metric:
                    # ReduceLROnPlateau needs validation metric
                    if val_loss is not None:
                        scheduler.step(val_loss)
                    else:
                        logger.warning(
                            "ReduceLROnPlateau requires validation data but none provided. "
                            "Scheduler will not be stepped."
                        )
                else:
                    # Other per-epoch schedulers (StepLR, ExponentialLR)
                    scheduler.step()
            # ─────────────────────────────────────────────────────────────

            # --- LOGGING ---
            epoch_duration = time.time() - epoch_start_time

            if (epoch + 1) % log_every == 0:
                # Create contextual logger with epoch information
                context_dict = {'epoch': epoch + 1}
                if run_context is not None:
                    if hasattr(run_context, 'experiment_name'):
                        context_dict['experiment'] = run_context.experiment_name
                    if hasattr(run_context, 'fold_idx'):
                        context_dict['fold'] = run_context.fold_idx

                epoch_logger = get_contextual_logger(__name__, **context_dict)

                # Build consolidated log message with pipe separators
                current_lr = _safe_get_lr(optimizer)
                lr_str = f" | LR: {current_lr:.2e}" if current_lr else ""
                val_str = f" | Val Loss: {val_loss:.6f}" if val_loss is not None else ""

                epoch_logger.info(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Grad: {avg_grad_norm:.4f}{val_str}{lr_str} | "
                    f"Time: {epoch_duration:.2f}s"
                )

            # --- EARLY STOPPING ---
            current_loss = val_loss if val_loss is not None else train_loss
            if current_loss < best_val_loss:
                best_val_loss = current_loss
                # Deep copy state_dict to avoid reference issues
                import copy
                best_model_state = copy.deepcopy(model.state_dict())

                # DEBUG: Check dtype of saved weights
                first_param_name = list(best_model_state.keys())[0] if best_model_state else None
                if first_param_name:
                    first_param = best_model_state[first_param_name]

                epochs_no_improve = 0
                best_epoch = epoch + 1
            else:
                epochs_no_improve += 1
                if early_stopping_patience is not None and \
                   epochs_no_improve >= early_stopping_patience and \
                   epoch + 1 >= min_epochs:
                    epoch_logger.info("Early stopping triggered")
                    break

        # --- END OF TRAINING ---
        if best_model_state:
            logger.info(f"[CHECKPOINT] Restoring best model from epoch {best_epoch} (val_loss={best_val_loss:.6f})")

            # DEBUG: Check model state before restore
            first_param_name = list(model.state_dict().keys())[0]
            param_before = model.state_dict()[first_param_name].clone()

            model.load_state_dict(best_model_state)

            # DEBUG: Check if state actually changed
            param_after = model.state_dict()[first_param_name]
            changed = not torch.equal(param_before, param_after)

            if changed:
                logger.debug(f"[CHECKPOINT DEBUG] Model restored. Params changed: {changed}")
                logger.debug(
                    f"[CHECKPOINT DEBUG] First param diff: "
                    f"{(param_after - param_before).abs().max().item():.6e}"
                )
            else:
                logger.warning(f"[CHECKPOINT WARNING] Model weights did NOT change after restore!")

        if csv_file_handle is not None:
            csv_file_handle.close()

        if auto_tune_horizon and best_horizon_mse is not None and best_degradation_ratio > degradation_threshold:
             logger.warning(
                 f"DEGRADATION DETECTED: {best_degradation_ratio:.2f}x. "
                 f"Recommended horizon: {suggested_horizon}"
             )
             setattr(model, 'suggested_forecast_steps', suggested_horizon)

        # ─────────────────────────────────────────────────────────────────
        # GENERATE SCHEDULER PLOT
        # ─────────────────────────────────────────────────────────────────
        if scheduler_monitor is not None and scheduler_plot_path:
            try:
                # Generate plot
                scheduler_monitor.plot_schedule(
                    save_path=str(scheduler_plot_path),
                    title=f"{model_name} Learning Rate Schedule",
                    show_losses=True
                )

                # Export CSV (if requested)
                if save_scheduler_csv:
                    csv_path = scheduler_plot_path.with_suffix('.csv')
                    scheduler_monitor.export_to_csv(
                        save_path=str(csv_path),
                        include_summary=True
                    )

                # Log summary
                summary = scheduler_monitor.get_summary()
                logger.info(f"LR Schedule Summary: {summary}")
            except Exception as e:
                logger.warning(f"Failed to generate scheduler outputs: {e}")
        # ─────────────────────────────────────────────────────────────────

        log_training_success(model_name, best_val_loss, best_epoch)
        setattr(model, 'best_val_loss', best_val_loss)

        # GPU Memory Cleanup after training completes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        return model, history

    except optuna.TrialPruned:
        # Re-raise this specific exception so it can be handled by the HPO loop in base.py
        raise

    except Exception as e:
        if 'csv_file_handle' in locals() and csv_file_handle is not None:
            csv_file_handle.close()
        if "Numerical instability" in str(e):
            raise  # ← Re-raise original, don't wrap!
        logger.error(f"Training loop failed for model '{model_name}': {str(e)}", exc_info=True)
        raise RuntimeError(f"Training loop failed: {str(e)}")

    finally:
        # CRITICAL: Release all DataLoader worker processes and their pipe file descriptors.
        #
        # Root cause of [Errno 24] Too many open files in HPO:
        #   Each DataLoader with num_workers>0 creates N worker processes, each backed by
        #   mp.Queue objects (index_queue per worker + shared result_queue). Queues are
        #   implemented with OS pipes. _shutdown_workers() sends a poison pill but does NOT
        #   join the processes or close the queue objects in the main process.
        #   Over 15 trials × 3 folds = 45 runs, unreleased pipe fds accumulate until EMFILE.
        #
        # Fix: after graceful shutdown signal, explicitly terminate+join every worker
        #      process and close every queue object. This guarantees the OS reclaims
        #      all pipe fds before the next training run starts.
        import gc

        def _force_shutdown_loader(loader, loader_name="loader"):
            iterator = getattr(loader, '_iterator', None)
            _fd_before = _count_open_fds()

            if iterator is None:
                logger.debug(f"[FD] {loader_name}: _iterator=None, no workers to clean up ({_fd_before} fds)")
                return

            workers = getattr(iterator, '_workers', []) or []
            index_queues = getattr(iterator, '_index_queues', []) or []
            result_q = getattr(iterator, '_worker_result_queue', None)
            logger.debug(
                f"[FD] {loader_name}: cleaning up "
                f"workers={len(workers)}, index_queues={len(index_queues)}, "
                f"has_result_q={result_q is not None} | fds before={_fd_before}"
            )

            # Step 1: graceful shutdown (sends poison pill to workers).
            # Skip if already in KeyboardInterrupt handling — _shutdown_workers()
            # blocks on w.join(timeout=5s) per worker, which hangs on Ctrl+C.
            import sys
            if not isinstance(sys.exc_info()[1], KeyboardInterrupt):
                try:
                    iterator._shutdown_workers()
                except Exception as e:
                    logger.debug(f"[FD] {loader_name}: _shutdown_workers() raised: {e}")

            # Step 2: explicitly terminate + join each worker process, then close
            # the Popen object to release spawn pipe fds immediately.
            #
            # Root cause of sentinel fd leak:
            #   PyTorch with persistent_workers=True + pin_memory=True (line 1237-1241
            #   in torch/utils/data/dataloader.py) calls atexit.register(cleanup, w)
            #   for every worker. atexit holds Process → Popen object alive → Popen's
            #   finalizer (which closes parent_r + parent_w spawn pipe fds) is NEVER
            #   called during the HPO run. Across 45 training runs × 2 DataLoaders ×
            #   N workers, these fds accumulate until EMFILE.
            #
            # Fix: explicitly call popen.close() which invokes the finalizer NOW,
            #   releasing parent_r and parent_w before the next training run starts.
            for i, w in enumerate(workers):
                if w is None:
                    continue
                try:
                    alive_before = w.is_alive()
                    if alive_before:
                        w.terminate()
                    w.join(timeout=2.0)
                    alive_after = w.is_alive() if hasattr(w, 'is_alive') else None
                    logger.debug(
                        f"[FD] {loader_name} worker[{i}]: alive_before={alive_before}, "
                        f"alive_after={alive_after}"
                    )

                    # Explicitly close Popen to trigger the finalizer immediately.
                    # popen.close() calls self.finalizer() which closes parent_r and
                    # parent_w (spawn bootstrap pipe fds) without waiting for GC.
                    popen = getattr(w, '_popen', None)
                    if popen is not None:
                        try:
                            popen.close()  # Calls finalizer → closes sentinel/spawn fds
                            logger.debug(f"[FD] {loader_name} worker[{i}]: popen.close() OK")
                        except Exception as pe:
                            # Fallback: call finalizer directly
                            finalizer = getattr(popen, 'finalizer', None)
                            if callable(finalizer):
                                try:
                                    finalizer()
                                    logger.debug(
                                        f"[FD] {loader_name} worker[{i}]: "
                                        f"popen.finalizer() OK (fallback)"
                                    )
                                except Exception:
                                    pass
                            logger.debug(
                                f"[FD] {loader_name} worker[{i}]: popen.close() failed: {pe}"
                            )

                except Exception as e:
                    logger.debug(f"[FD] {loader_name} worker[{i}] cleanup failed: {e}")

            # Step 3: close queue objects in the MAIN process.
            # mp.Queue holds a pipe fd in the caller process until .close() is called.
            # del queue / GC does not reliably close it in time for HPO back-to-back runs.
            for i, q in enumerate(index_queues):
                try:
                    q.cancel_join_thread()
                    q.close()
                except Exception as e:
                    logger.debug(f"[FD] {loader_name} index_queue[{i}].close() failed: {e}")

            if result_q is not None:
                try:
                    result_q.cancel_join_thread()
                    result_q.close()
                except Exception as e:
                    logger.debug(f"[FD] {loader_name} result_queue.close() failed: {e}")

            loader._iterator = None

            _fd_after = _count_open_fds()
            logger.debug(
                f"[FD] {loader_name}: cleanup done. "
                f"fds before={_fd_before}, after={_fd_after}, released={_fd_before - _fd_after}"
            )

        try:
            _fd_before_cleanup = _count_open_fds()
            _fd_types_before = _describe_open_fds()
            logger.debug(
                f"[FD] finally BEFORE cleanup: {_fd_before_cleanup} fds "
                f"(pipe={_fd_types_before['pipe']}, shm={_fd_types_before['shm']}, "
                f"file={_fd_types_before['file']}, anon={_fd_types_before['anon_inode']})"
            )

            if 'train_loader' in locals() and train_loader is not None:
                _force_shutdown_loader(train_loader, "train_loader")
                del train_loader

            if 'val_loader' in locals() and val_loader is not None:
                _force_shutdown_loader(val_loader, "val_loader")
                del val_loader

            # Final GC pass to collect any remaining cyclic references.
            gc.collect()

            if 'csv_file_handle' in locals() and csv_file_handle is not None:
                csv_file_handle.close()

            _fd_after_cleanup = _count_open_fds()
            _fd_types_after = _describe_open_fds()
            logger.debug(
                f"[FD] finally AFTER cleanup: {_fd_after_cleanup} fds "
                f"(pipe={_fd_types_after['pipe']}, shm={_fd_types_after['shm']}, "
                f"file={_fd_types_after['file']}, anon={_fd_types_after['anon_inode']}) "
                f"| net_delta_vs_entry={_fd_after_cleanup - _fd_entry}"
            )

        except Exception as cleanup_error:
            logger.warning(f"DataLoader cleanup failed (non-critical): {cleanup_error}")
