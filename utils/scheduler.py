"""
Scheduler utilities for learning rate scheduling in neural network training.

This module provides a unified interface for creating and managing various
PyTorch learning rate schedulers, with support for:
- OneCycleLR (recommended for most cases)
- CosineAnnealingLR (smooth annealing)
- StepLR (simple step decay)
- ExponentialLR (exponential decay)
- ReduceLROnPlateau (adaptive based on metrics)
"""

import logging
import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    OneCycleLR,
    CosineAnnealingLR,
    StepLR,
    ExponentialLR,
    ReduceLROnPlateau,
    _LRScheduler
)
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def create_scheduler(
        optimizer: Optimizer,
        scheduler_config: Dict[str, Any],
        train_size: int,
        batch_size: int,
        max_epochs: int,
        default_lr: float
) -> Optional[_LRScheduler]:
    """
    Create learning rate scheduler from configuration.

    This function provides a unified interface for creating various PyTorch
    learning rate schedulers with sensible defaults and comprehensive logging.

    Args:
        optimizer: PyTorch optimizer instance
        scheduler_config: Scheduler configuration dictionary
        train_size: Number of training samples
        batch_size: Training batch size
        max_epochs: Maximum number of training epochs
        default_lr: Default/base learning rate (used as max_lr for OneCycleLR)

    Returns:
        Scheduler instance or None if no scheduler configured

    Supported scheduler types:
        - "onecycle": OneCycleLR with warmup and cosine annealing
        - "cosine": CosineAnnealingLR for smooth annealing
        - "step": StepLR for periodic LR reduction
        - "exponential": ExponentialLR for exponential decay
        - "plateau": ReduceLROnPlateau for adaptive reduction based on metrics

    Example configs:
        # OneCycleLR (recommended for most cases)
        # NOTE: max_lr defaults to learning_rate if not specified (HPO-friendly)
        #       Specify max_lr explicitly for fixed configurations
        {
            "type": "onecycle",
            "max_lr": 0.01,      # Optional: omit to use learning_rate from model
            "pct_start": 0.3,
            "div_factor": 25.0,
            "final_div_factor": 1e4
        }

        # ReduceLROnPlateau (adaptive)
        {
            "type": "plateau",
            "mode": "min",
            "factor": 0.5,
            "patience": 10
        }

        # StepLR (simple)
        {
            "type": "step",
            "step_size": 20,
            "gamma": 0.1
        }

    Notes:
        - OneCycleLR and CosineAnnealingLR are stepped per-batch
        - StepLR, ExponentialLR are stepped per-epoch
        - ReduceLROnPlateau is stepped per-epoch with validation metric
    """
    sched_type = scheduler_config.get("type", None)

    if not sched_type:
        logger.debug("No scheduler configured (scheduler_config.type not set)")
        return None

    logger.info(f"Creating LR scheduler: {sched_type}")

    # ───────────────────────────────────────────────────────────────────────
    # OneCycleLR: Batch-level scheduler with warmup and annealing
    # ───────────────────────────────────────────────────────────────────────
    # Recommended for most time series forecasting tasks.
    # Provides smooth warmup, peak learning, and annealing phases.

    if sched_type == "onecycle":
        # Security: Avoid division by zero if batch_size is invalid
        steps_per_epoch = int(np.ceil(train_size / max(1, batch_size)))
        total_steps = steps_per_epoch * max_epochs

        # ─────────────────────────────────────────────────────────────
        # VALIDATION: Check for zero or negative total_steps
        # ─────────────────────────────────────────────────────────────
        if total_steps <= 0:
            logger.warning(
                f"OneCycleLR: total_steps={total_steps} (invalid). "
                f"max_epochs={max_epochs}, steps_per_epoch={steps_per_epoch}. "
                f"Skipping scheduler creation."
            )
            return None
        # ─────────────────────────────────────────────────────────────

        # Get config with sensible defaults
        # If max_lr not specified in config, use default_lr (learning_rate from model)
        # This enables HPO compatibility: omit max_lr in config to use optimized learning_rate
        # Or specify max_lr explicitly for tests/fixed configurations
        max_lr = scheduler_config.get("max_lr", default_lr)
        pct_start = scheduler_config.get("pct_start", 0.3)  # 30% warmup
        div_factor = scheduler_config.get("div_factor", 25.0)  # PyTorch default
        final_div_factor = scheduler_config.get("final_div_factor", 1e4)  # PyTorch default
        anneal_strategy = scheduler_config.get("anneal_strategy", "cos")  # 'cos' or 'linear'

        # Calculate derived values
        initial_lr = max_lr / div_factor
        final_lr = max_lr / final_div_factor

        logger.info(
            f"OneCycleLR config:\n"
            f"  max_lr={max_lr:.6f}, initial_lr={initial_lr:.6f}, final_lr={final_lr:.6f}\n"
            f"  total_steps={total_steps} ({steps_per_epoch} steps/epoch × {max_epochs} epochs)\n"
            f"  pct_start={pct_start} (warmup for {int(total_steps * pct_start)} steps)\n"
            f"  anneal_strategy={anneal_strategy}"
        )

        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            anneal_strategy=anneal_strategy
        )

    # ───────────────────────────────────────────────────────────────────────
    # CosineAnnealingLR: Smooth cosine annealing (batch-level)
    # ───────────────────────────────────────────────────────────────────────
    # Good for fine-tuning and smooth LR curves.
    # Anneals from current LR to eta_min following a cosine curve.

    elif sched_type == "cosine":
        # Default to batch-level for smooth annealing
        # Security: Avoid division by zero if batch_size is invalid
        steps_per_epoch = int(np.ceil(train_size / max(1, batch_size)))

        # T_max: Number of iterations until first restart (if using restarts)
        # Default to full training duration
        T_max = scheduler_config.get("T_max", steps_per_epoch * max_epochs)
        eta_min = scheduler_config.get("eta_min", 0.0)  # Minimum LR

        logger.info(
            f"CosineAnnealingLR config:\n"
            f"  T_max={T_max} steps, eta_min={eta_min:.6f}\n"
            f"  Note: This is a batch-level scheduler"
        )

        return CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min
        )

    # ───────────────────────────────────────────────────────────────────────
    # StepLR: Simple step decay (epoch-level)
    # ───────────────────────────────────────────────────────────────────────
    # Reduces LR by gamma every step_size epochs.
    # Simple and predictable, good for long training runs.

    elif sched_type == "step":
        step_size = scheduler_config.get("step_size", 10)  # Epochs between reductions
        gamma = scheduler_config.get("gamma", 0.1)  # Multiplicative factor

        logger.info(
            f"StepLR config:\n"
            f"  step_size={step_size} epochs, gamma={gamma}\n"
            f"  LR will be multiplied by {gamma} every {step_size} epochs"
        )

        return StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )

    # ───────────────────────────────────────────────────────────────────────
    # ExponentialLR: Exponential decay (epoch-level)
    # ───────────────────────────────────────────────────────────────────────
    # Multiplies LR by gamma each epoch.
    # Simple exponential decay, good for gradual reduction.

    elif sched_type == "exponential":
        gamma = scheduler_config.get("gamma", 0.95)  # Decay factor per epoch

        logger.info(
            f"ExponentialLR config:\n"
            f"  gamma={gamma}\n"
            f"  LR will be multiplied by {gamma} each epoch"
        )

        return ExponentialLR(
            optimizer,
            gamma=gamma
        )

    # ───────────────────────────────────────────────────────────────────────
    # ReduceLROnPlateau: Adaptive reduction based on metrics (epoch-level)
    # ───────────────────────────────────────────────────────────────────────
    # Reduces LR when validation metric plateaus.
    # Good for adaptive training when you don't know the optimal schedule.

    elif sched_type == "plateau":
        mode = scheduler_config.get("mode", "min")  # 'min' for loss, 'max' for accuracy
        factor = scheduler_config.get("factor", 0.5)  # Reduction factor
        patience = scheduler_config.get("patience", 10)  # Epochs to wait
        threshold = scheduler_config.get("threshold", 1e-4)  # Minimum improvement
        threshold_mode = scheduler_config.get("threshold_mode", "rel")  # 'rel' or 'abs'
        cooldown = scheduler_config.get("cooldown", 0)  # Epochs to wait after reduction
        min_lr = scheduler_config.get("min_lr", 0.0)  # Minimum LR
        eps = scheduler_config.get("eps", 1e-8)  # Minimum decay

        logger.info(
            f"ReduceLROnPlateau config:\n"
            f"  mode={mode}, factor={factor}, patience={patience}\n"
            f"  threshold={threshold} ({threshold_mode}), min_lr={min_lr:.6e}\n"
            f"  cooldown={cooldown} epochs\n"
            f"  Note: Requires validation metric, stepped with scheduler.step(val_loss)"
        )

        return ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps
        )

    else:
        logger.warning(
            f"Unknown scheduler type: '{sched_type}'. "
            f"Supported types: onecycle, cosine, step, exponential, plateau. "
            f"No scheduler will be used."
        )
        return None


def get_scheduler_step_info(scheduler: Optional[_LRScheduler]) -> Dict[str, Any]:
    """
    Get information about when and how to step the scheduler.

    Args:
        scheduler: LR scheduler instance or None

    Returns:
        Dictionary with keys:
            - step_frequency: 'batch', 'epoch', or None
            - requires_metric: bool, True if scheduler needs validation metric
            - scheduler_name: Name of scheduler class

    Example:
        >>> info = get_scheduler_step_info(scheduler)
        >>> if info['step_frequency'] == 'batch':
        >>>     scheduler.step()  # Call after each batch
        >>> elif info['requires_metric']:
        >>>     scheduler.step(val_loss)  # Call with validation metric
    """
    if scheduler is None:
        return {
            'step_frequency': None,
            'requires_metric': False,
            'scheduler_name': None
        }

    scheduler_name = scheduler.__class__.__name__

    # Determine step frequency
    if isinstance(scheduler, (OneCycleLR, CosineAnnealingLR)):
        step_frequency = 'batch'
        requires_metric = False
    elif isinstance(scheduler, ReduceLROnPlateau):
        step_frequency = 'epoch'
        requires_metric = True
    else:
        # StepLR, ExponentialLR, and other standard schedulers
        step_frequency = 'epoch'
        requires_metric = False

    return {
        'step_frequency': step_frequency,
        'requires_metric': requires_metric,
        'scheduler_name': scheduler_name
    }