"""
Module for analyzing gradient flow and training dynamics from CSV logs.
"""
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

class GradientAnalyzer:
    """
    Analyzer for gradient logs produced by GradientMonitor (CSV format).
    """

    def __init__(self, gradients_dir: Union[str, Path]):
        self.gradients_dir = Path(gradients_dir)
        if not self.gradients_dir.exists():
            logger.warning(f"Gradients directory does not exist: {self.gradients_dir}")

    def load_gradient_logs(
        self,
        model_name: str,
        window_size: int,
        fold_idx: int
    ) -> pd.DataFrame:
        """
        Load gradient logs from CSV file.

        Args:
            model_name: Model name (e.g., "transformer")
            window_size: Window size (e.g., 96)
            fold_idx: Fold index (e.g., 0)

        Returns:
            DataFrame with columns: epoch, step, global_step, batch_loss,
                                   total_grad_norm, encoder_grad_norm, head_grad_norm

        Raises:
            FileNotFoundError: If gradient CSV not found
        """
        # CSV filename convention from GradientMonitor
        filename = f"{model_name}_fold_{fold_idx}_w{window_size}_gradients.csv"
        filepath = self.gradients_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Gradient CSV not found: {filepath}")

        try:
            df = pd.read_csv(filepath)
            logger.info(f"[GradientAnalyzer] Loaded CSV: {filepath} ({len(df)} rows)")
            return df
        except Exception as e:
            logger.error(f"Failed to read CSV {filepath}: {e}")
            raise

    def discover_all_artifacts(self) -> List[Path]:
        """
        Discover all gradient CSV files in directory.

        Returns:
            List of paths to gradient CSV files
        """
        if not self.gradients_dir.exists():
            logger.warning(f"Directory does not exist: {self.gradients_dir}")
            return []

        artifacts = list(self.gradients_dir.glob("*_gradients.csv"))
        logger.info(f"[GradientAnalyzer] Discovered {len(artifacts)} CSV file(s)")
        return artifacts

    def load_all_folds(
        self,
        model_name: str,
        window_size: int,
        max_folds: int = 10
    ) -> Dict[int, pd.DataFrame]:
        """
        Load gradient logs for all available folds.

        Args:
            model_name: Model name
            window_size: Window size
            max_folds: Maximum folds to try

        Returns:
            Dictionary mapping fold_idx to DataFrame
        """
        all_folds = {}

        for fold_idx in range(max_folds):
            try:
                df = self.load_gradient_logs(model_name, window_size, fold_idx)
                all_folds[fold_idx] = df
            except FileNotFoundError:
                # Sequential folds assumption; stop if fold N missing
                if fold_idx > 0:
                    break
                else:
                    continue

        logger.info(f"[GradientAnalyzer] Loaded {len(all_folds)} fold(s) for {model_name}")
        return all_folds

    def compute_gradient_decay(self, df: pd.DataFrame) -> float:
        """
        Compute gradient decay rate from encoder norm (simple linear fit on log norms).

        Args:
            df: Gradient DataFrame

        Returns:
            Decay rate (slope). Negative = vanishing, Positive = exploding.
        """
        if len(df) < 2:
            return 0.0

        # Filter finite values only
        valid_df = df[np.isfinite(df['encoder_grad_norm']) & (df['encoder_grad_norm'] > 0)]

        if len(valid_df) < 2:
            return 0.0

        norms = valid_df['encoder_grad_norm'].values
        # Use simple index as time proxy
        t = np.arange(len(norms))
        # Log space for exponential decay/explosion analysis
        log_norms = np.log(norms + 1e-10)

        # Linear regression: log_norm = a * t + b
        coeffs = np.polyfit(t, log_norms, deg=1)
        return float(coeffs[0])  # Slope

    def plot_gradient_flow(
        self,
        model_names: List[str],
        window_sizes: List[int],
        fold_idx: Optional[int] = None,
        output_path: Optional[Path] = None
    ):
        """
        Plot gradient flow by layer using CSV data.
        Generates grid of plots: Rows=Window Sizes, Cols=Models.
        """
        n_rows = len(window_sizes)
        n_cols = len(model_names)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(6 * n_cols, 4 * n_rows),
            squeeze=False # Ensure 2D array
        )

        for i, window_size in enumerate(window_sizes):
            for j, model_name in enumerate(model_names):
                ax = axes[i, j]

                try:
                    df = None
                    if fold_idx is not None:
                        df = self.load_gradient_logs(model_name, window_size, fold_idx)
                    else:
                        # Load all folds and concatenate
                        all_folds = self.load_all_folds(model_name, window_size)
                        if all_folds:
                            df = pd.concat(all_folds.values(), ignore_index=True)
                            # Sort by global step just in case
                            if 'global_step' in df.columns:
                                df = df.sort_values('global_step')

                    if df is None or len(df) == 0:
                        ax.text(0.5, 0.5, "No data", ha='center', va='center')
                        ax.set_title(f"{model_name}\nWindow={window_size}")
                        continue

                    # Determine X-axis
                    x_col = 'global_step' if 'global_step' in df.columns else 'step'

                    # Check for explosion (NaNs in norms usually mean explosion happened just before)
                    # Or extreme values
                    has_explosion = df['encoder_grad_norm'].isna().any() or (df['encoder_grad_norm'] > 1e4).any()

                    # Plot encoder and head norms
                    # Matplotlib handles NaNs by breaking the line - perfect for showing crashes
                    ax.plot(df[x_col], df['encoder_grad_norm'],
                           label='Encoder (Body)', alpha=0.8, linewidth=1.5, color='blue')
                    ax.plot(df[x_col], df['head_grad_norm'],
                           label='Head (Output)', alpha=0.8, linewidth=1.5, color='orange', linestyle='--')

                    # Annotate explosion if detected
                    if has_explosion:
                        ax.text(0.95, 0.95, "💥 Unstable", transform=ax.transAxes,
                                color='red', ha='right', va='top', fontweight='bold')

                    # Formatting
                    ax.set_xlabel("Training Step")
                    ax.set_ylabel("Gradient Norm (L2)")
                    ax.set_title(f"{model_name} (W={window_size})")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.set_yscale('log') # Log scale helps see vanishing gradients

                    # Add decay statistic
                    decay_rate = self.compute_gradient_decay(df)
                    status_text = f"Slope: {decay_rate:.4f}"
                    if decay_rate < -0.001: status_color = 'red'  # Vanishing
                    elif decay_rate > 0.001: status_color = 'orange' # Exploding
                    else: status_color = 'green' # Stable

                    ax.text(
                        0.02, 0.02, status_text,
                        transform=ax.transAxes, fontsize=9,
                        verticalalignment='bottom', color=status_color,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                    )

                except FileNotFoundError:
                    ax.text(0.5, 0.5, "File not found", ha='center', va='center', color='gray')
                    ax.set_title(f"{model_name}\nWindow={window_size}")
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error:\n{str(e)}", ha='center', va='center', color='red', fontsize=8)
                    logger.error(f"Error plotting {model_name} w={window_size}: {e}")

        plt.tight_layout()

        if output_path is None:
            output_path = self.gradients_dir.parent / "plots" / "gradient_flow_matrix.png"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"[GradientAnalyzer] Saved plot: {output_path}")
        plt.close()