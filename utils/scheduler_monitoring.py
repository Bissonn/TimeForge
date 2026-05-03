"""
Scheduler Monitoring and Visualization Utilities

This module provides tools for monitoring and visualizing learning rate schedules,
including warmup plots, schedule previews, and training diagnostics.

Usage:
    # Plot schedule before training
    preview_scheduler_schedule(scheduler, num_epochs=100, steps_per_epoch=32)

    # Monitor during training
    monitor = SchedulerMonitor()
    monitor.log_lr(epoch, current_lr, train_loss, val_loss)
    monitor.plot_schedule()
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict, Any
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import os

logger = logging.getLogger(__name__)


class SchedulerMonitor:
    """
    Monitor and visualize learning rate schedule during training.

    Tracks LR changes, loss curves, and provides visualization tools
    for understanding scheduler behavior.

    Example:
        monitor = SchedulerMonitor()

        for epoch in range(epochs):
            # Training...
            current_lr = optimizer.param_groups[0]['lr']
            monitor.log_lr(epoch, current_lr, train_loss, val_loss)

        monitor.plot_schedule(save_path="results/lr_schedule.png")
    """

    def __init__(self):
        """Initialize scheduler monitor."""
        self.epochs = []
        self.learning_rates = []
        self.train_losses = []
        self.val_losses = []

    def log_lr(
            self,
            epoch: int,
            lr: float,
            train_loss: Optional[float] = None,
            val_loss: Optional[float] = None
    ):
        """
        Log learning rate and losses for an epoch.

        Args:
            epoch: Current epoch number
            lr: Current learning rate
            train_loss: Training loss (optional)
            val_loss: Validation loss (optional)
        """
        self.epochs.append(epoch)
        self.learning_rates.append(lr)

        if train_loss is not None:
            self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)

    def plot_schedule(
            self,
            save_path: Optional[str] = None,
            title: str = "Learning Rate Schedule",
            show_losses: bool = True,
            figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Plot learning rate schedule with optional loss curves.

        Args:
            save_path: Path to save figure (if None, displays instead)
            title: Plot title
            show_losses: Whether to include loss curves
            figsize: Figure size (width, height)
        """
        if not self.epochs:
            logger.warning("No data logged. Cannot create plot.")
            return

        if show_losses and (self.train_losses or self.val_losses):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)
            ax2 = None

        # Plot learning rate
        ax1.plot(self.epochs, self.learning_rates, 'b-', linewidth=2, label='Learning Rate')
        ax1.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)

        # Add log scale option if LR varies by orders of magnitude
        lr_range = max(self.learning_rates) / min(self.learning_rates) if min(self.learning_rates) > 0 else 1
        if lr_range > 100:
            ax1.set_yscale('log')
            ax1.set_ylabel('Learning Rate (log scale)', fontsize=12, fontweight='bold')

        # Annotate key points
        max_lr_epoch = self.epochs[np.argmax(self.learning_rates)]
        max_lr = max(self.learning_rates)
        ax1.annotate(
            f'Peak LR: {max_lr:.6f}',
            xy=(max_lr_epoch, max_lr),
            xytext=(10, 10),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
        )

        # Plot losses
        if ax2 is not None and (self.train_losses or self.val_losses):
            if self.train_losses:
                ax2.plot(self.epochs[:len(self.train_losses)], self.train_losses,
                         'g-', linewidth=2, label='Train Loss', alpha=0.7)
            if self.val_losses:
                ax2.plot(self.epochs[:len(self.val_losses)], self.val_losses,
                         'r-', linewidth=2, label='Val Loss', alpha=0.7)

            ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=10)

            # Mark minimum validation loss
            if self.val_losses:
                min_val_epoch = self.epochs[np.argmin(self.val_losses)]
                min_val_loss = min(self.val_losses)
                ax2.annotate(
                    f'Best: {min_val_loss:.6f}',
                    xy=(min_val_epoch, min_val_loss),
                    xytext=(10, 10),
                    textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                )

        if not ax2:
            ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved LR schedule plot to {save_path}")
        else:
            plt.show()

        plt.close()

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the learning rate schedule.

        Returns:
            Dictionary with summary statistics
        """
        if not self.learning_rates:
            return {}

        return {
            'num_epochs': len(self.epochs),
            'initial_lr': self.learning_rates[0],
            'final_lr': self.learning_rates[-1],
            'max_lr': max(self.learning_rates),
            'min_lr': min(self.learning_rates),
            'avg_lr': np.mean(self.learning_rates),
            'lr_reduction_factor': self.learning_rates[0] / self.learning_rates[-1] if self.learning_rates[
                                                                                           -1] > 0 else float('inf'),
            'best_train_loss': min(self.train_losses) if self.train_losses else None,
            'best_val_loss': min(self.val_losses) if self.val_losses else None,
        }

    def export_to_csv(
            self,
            save_path: str,
            include_summary: bool = True
    ):
        '''Export scheduler data to CSV file.'''
        import csv

        if not self.epochs:
            logger.warning("No data to export. CSV not created.")
            return

        try:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

            with open(save_path, 'w', newline='') as f:
                # Write summary as comments
                if include_summary:
                    summary = self.get_summary()
                    f.write("# Scheduler Data Summary\n")
                    for key, value in summary.items():
                        if value is not None:
                            f.write(f"# {key}: {value}\n")
                    f.write("#\n")

                # Write CSV
                writer = csv.writer(f)
                writer.writerow(['epoch', 'learning_rate', 'train_loss', 'val_loss'])

                for i, epoch in enumerate(self.epochs):
                    lr = self.learning_rates[i]
                    train_loss = self.train_losses[i] if i < len(self.train_losses) else None
                    val_loss = self.val_losses[i] if i < len(self.val_losses) else None

                    writer.writerow([
                        epoch,
                        f"{lr:.10e}",
                        f"{train_loss:.6f}" if train_loss is not None else "",
                        f"{val_loss:.6f}" if val_loss is not None else ""
                    ])

            logger.info(f"Exported scheduler data to {save_path}")

        except Exception as e:
            logger.error(f"Failed to export scheduler data to CSV: {e}")

def preview_scheduler_schedule(
        scheduler: _LRScheduler,
        optimizer: Optimizer,
        num_epochs: int,
        steps_per_epoch: int = 1,
        save_path: Optional[str] = None,
        title: Optional[str] = None
):
    """
    Preview learning rate schedule before training.

    Simulates the scheduler for specified number of epochs and plots
    the resulting LR curve. Useful for validating scheduler configuration.

    Args:
        scheduler: Scheduler instance to preview
        optimizer: Optimizer instance
        num_epochs: Number of epochs to simulate
        steps_per_epoch: Steps per epoch (for batch-level schedulers)
        save_path: Path to save figure (if None, displays instead)
        title: Plot title (auto-generated if None)

    Example:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = OneCycleLR(optimizer, max_lr=0.01, total_steps=1000)
        preview_scheduler_schedule(scheduler, optimizer, num_epochs=100, steps_per_epoch=10)
    """
    from torch.optim.lr_scheduler import (
        OneCycleLR, CosineAnnealingLR, ReduceLROnPlateau
    )

    # Determine if batch-level scheduler
    is_batch_level = isinstance(scheduler, (OneCycleLR, CosineAnnealingLR))
    is_plateau = isinstance(scheduler, ReduceLROnPlateau)

    scheduler_name = scheduler.__class__.__name__

    if title is None:
        title = f"{scheduler_name} Schedule Preview"

    # Simulate schedule
    lrs = []
    steps = []

    if is_plateau:
        logger.warning(
            "ReduceLROnPlateau cannot be previewed accurately "
            "(requires validation metrics). Showing constant LR."
        )
        current_lr = optimizer.param_groups[0]['lr']
        lrs = [current_lr] * num_epochs
        steps = list(range(num_epochs))

    elif is_batch_level:
        # Batch-level scheduler
        total_steps = num_epochs * steps_per_epoch
        for step in range(total_steps):
            current_lr = optimizer.param_groups[0]['lr']
            lrs.append(current_lr)
            steps.append(step / steps_per_epoch)  # Convert to epoch units
            scheduler.step()

    else:
        # Epoch-level scheduler
        for epoch in range(num_epochs):
            current_lr = optimizer.param_groups[0]['lr']
            lrs.append(current_lr)
            steps.append(epoch)
            scheduler.step()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(steps, lrs, 'b-', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add log scale if needed
    lr_range = max(lrs) / min(lrs) if min(lrs) > 0 else 1
    if lr_range > 100:
        ax.set_yscale('log')
        ax.set_ylabel('Learning Rate (log scale)', fontsize=12, fontweight='bold')

    # Annotate key points
    if not is_plateau:
        # Peak LR
        max_idx = np.argmax(lrs)
        ax.annotate(
            f'Peak: {lrs[max_idx]:.6f}\nat epoch {steps[max_idx]:.1f}',
            xy=(steps[max_idx], lrs[max_idx]),
            xytext=(10, 10),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
        )

        # Initial and final LR
        ax.axhline(y=lrs[0], color='g', linestyle='--', alpha=0.5, label=f'Initial: {lrs[0]:.6f}')
        ax.axhline(y=lrs[-1], color='r', linestyle='--', alpha=0.5, label=f'Final: {lrs[-1]:.6f}')
        ax.legend(fontsize=10)

    # Add scheduler info
    info_text = f"Scheduler: {scheduler_name}\n"
    info_text += f"Initial LR: {lrs[0]:.6f}\n"
    info_text += f"Final LR: {lrs[-1]:.6f}\n"
    if not is_plateau:
        info_text += f"Max LR: {max(lrs):.6f}\n"
        info_text += f"Reduction: {lrs[0] / lrs[-1]:.1f}x" if lrs[-1] > 0 else "Reduction: ∞"

    ax.text(
        0.02, 0.98, info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved schedule preview to {save_path}")
    else:
        plt.show()

    plt.close()


def compare_schedulers(
        scheduler_configs: Dict[str, Dict[str, Any]],
        base_lr: float = 0.001,
        num_epochs: int = 100,
        steps_per_epoch: int = 32,
        save_path: Optional[str] = None
):
    """
    Compare multiple scheduler configurations side-by-side.

    Args:
        scheduler_configs: Dict mapping names to scheduler configs
        base_lr: Base learning rate
        num_epochs: Number of epochs to simulate
        steps_per_epoch: Steps per epoch
        save_path: Path to save comparison plot

    Example:
        configs = {
            "OneCycle": {"type": "onecycle", "max_lr": 0.01, "pct_start": 0.3},
            "Step": {"type": "step", "step_size": 20, "gamma": 0.1},
            "Cosine": {"type": "cosine", "eta_min": 1e-6}
        }
        compare_schedulers(configs, save_path="results/scheduler_comparison.png")
    """
    from utils.scheduler import create_scheduler

    fig, ax = plt.subplots(figsize=(14, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(scheduler_configs)))

    for (name, config), color in zip(scheduler_configs.items(), colors):
        # Create dummy model and optimizer
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

        # Create scheduler
        scheduler = create_scheduler(
            optimizer=optimizer,
            scheduler_config=config,
            train_size=steps_per_epoch * 32,  # Dummy
            batch_size=32,
            max_epochs=num_epochs,
            default_lr=base_lr
        )

        if scheduler is None:
            logger.warning(f"Could not create scheduler for {name}")
            continue

        # Simulate schedule
        from torch.optim.lr_scheduler import (
            OneCycleLR, CosineAnnealingLR, ReduceLROnPlateau
        )

        is_batch_level = isinstance(scheduler, (OneCycleLR, CosineAnnealingLR))
        is_plateau = isinstance(scheduler, ReduceLROnPlateau)

        if is_plateau:
            # Constant for plateau
            lrs = [base_lr] * num_epochs
            epochs = list(range(num_epochs))
        elif is_batch_level:
            lrs = []
            epochs = []
            for step in range(num_epochs * steps_per_epoch):
                lrs.append(optimizer.param_groups[0]['lr'])
                epochs.append(step / steps_per_epoch)
                scheduler.step()
        else:
            lrs = []
            epochs = list(range(num_epochs))
            for _ in range(num_epochs):
                lrs.append(optimizer.param_groups[0]['lr'])
                scheduler.step()

        ax.plot(epochs, lrs, linewidth=2, label=name, color=color)

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_title('Scheduler Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')

    # Check if log scale needed
    if ax.get_ylim()[1] / ax.get_ylim()[0] > 100:
        ax.set_yscale('log')
        ax.set_ylabel('Learning Rate (log scale)', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved scheduler comparison to {save_path}")
    else:
        plt.show()

    plt.close()


def analyze_warmup_phase(
        scheduler: _LRScheduler,
        optimizer: Optimizer,
        warmup_epochs: int = 30,
        steps_per_epoch: int = 32,
        save_path: Optional[str] = None
):
    """
    Analyze and visualize the warmup phase of a scheduler.

    Useful for OneCycleLR to verify warmup behavior.

    Args:
        scheduler: Scheduler instance
        optimizer: Optimizer instance
        warmup_epochs: Number of epochs to analyze
        steps_per_epoch: Steps per epoch
        save_path: Path to save plot
    """
    from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR

    is_batch_level = isinstance(scheduler, (OneCycleLR, CosineAnnealingLR))

    if not is_batch_level:
        logger.warning("Warmup analysis is most useful for batch-level schedulers like OneCycleLR")

    # Simulate warmup
    lrs = []
    steps = []

    total_steps = warmup_epochs * steps_per_epoch if is_batch_level else warmup_epochs

    for step in range(total_steps):
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        steps.append(step / steps_per_epoch if is_batch_level else step)
        scheduler.step()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(steps, lrs, 'b-', linewidth=2)
    ax.set_xlabel('Epoch' if is_batch_level else 'Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_title('Warmup Phase Analysis', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Highlight warmup region
    if isinstance(scheduler, OneCycleLR):
        # Calculate warmup end
        pct_start = scheduler.pct_start if hasattr(scheduler, 'pct_start') else 0.3
        warmup_end = total_steps * pct_start / steps_per_epoch if is_batch_level else total_steps * pct_start

        ax.axvspan(0, warmup_end, alpha=0.2, color='green', label=f'Warmup ({pct_start * 100:.0f}%)')
        ax.legend()

    # Add stats
    info_text = f"Initial LR: {lrs[0]:.6f}\n"
    info_text += f"Peak LR (in window): {max(lrs):.6f}\n"
    info_text += f"Growth rate: {(max(lrs) / lrs[0]):.2f}x"

    ax.text(
        0.98, 0.02, info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved warmup analysis to {save_path}")
    else:
        plt.show()

    plt.close()