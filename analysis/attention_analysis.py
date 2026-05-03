"""
Attention pattern analysis and visualization.

Clean Architecture: Analyzer works with any directory containing attention artifacts.
No dependency on RunContext - just needs a path.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import seaborn as sns
import logging

sns.set_style("whitegrid")
logger = logging.getLogger(__name__)


class AttentionAnalyzer:
    """
    Analyze attention patterns from Transformer models.
    
    Clean Architecture Design:
    - Works with any directory containing attention .npz files
    - No dependency on RunContext (infrastructure)
    - Can discover artifacts automatically
    - Follows Single Responsibility Principle
    
    Example:
        >>> # Trainer provides directory from context
        >>> analyzer = AttentionAnalyzer(attention_dir=fold_ctx.attention_dir)
        >>> 
        >>> # Load specific attention patterns
        >>> patterns = analyzer.load_attention_patterns(
        ...     model_name="transformer",
        ...     window_size=96,
        ...     fold_idx=0
        ... )
    """

    def __init__(self, attention_dir: Path):
        """
        Initialize attention analyzer.
        
        Args:
            attention_dir: Directory containing attention weight files (.npz)
        
        Design Note:
            Accepts only the directory it needs. Makes analyzer reusable
            with any directory structure, not tied to RunContext.
        """
        self.attention_dir = Path(attention_dir)
        logger.debug(f"[AttentionAnalyzer] Initialized with dir: {self.attention_dir}")

    def load_attention_patterns(
        self,
        model_name: str,
        window_size: int,
        fold_idx: int
    ) -> Dict[int, np.ndarray]:
        """
        Load attention patterns for specific model/fold/window combination.
        
        Args:
            model_name: Model name (e.g., "transformer")
            window_size: Window size (e.g., 96)
            fold_idx: Fold index (e.g., 0)
        
        Returns:
            Dictionary mapping layer index to attention weights
        
        Raises:
            FileNotFoundError: If attention file not found
        
        Design Note:
            Explicit parameters - no implicit state or context dependency.
        """
        # Build filename using standard convention
        filename = f"{model_name}_fold_{fold_idx}_w{window_size}_attention.npz"
        filepath = self.attention_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Attention file not found: {filepath}")
        
        data = np.load(filepath)
        
        attention_by_layer = {}
        for key in data.files:
            # Assuming keys like "layer_0", "layer_1", etc.
            layer_idx = int(key.split('_')[1])
            attention_by_layer[layer_idx] = data[key]
        
        logger.info(f"[AttentionAnalyzer] Loaded {len(attention_by_layer)} layer(s)")
        return attention_by_layer

    def discover_all_artifacts(self) -> List[Path]:
        """
        Discover all attention artifact files in directory.
        
        Returns:
            List of paths to attention files
        """
        if not self.attention_dir.exists():
            logger.warning(f"Directory does not exist: {self.attention_dir}")
            return []
        
        artifacts = list(self.attention_dir.glob("*_attention.npz"))
        logger.info(f"[AttentionAnalyzer] Discovered {len(artifacts)} artifact(s)")
        return artifacts

    def compute_effective_receptive_field(
        self,
        attention_weights: np.ndarray,
        threshold: float = 0.5
    ) -> float:
        """
        Compute effective receptive field (ERF).
        
        ERF = temporal distance that captures `threshold` of attention mass.
        
        Args:
            attention_weights: Attention weights (batch, heads, seq_len, seq_len)
            threshold: Attention mass threshold (default 0.5)
        
        Returns:
            Effective receptive field in timesteps
        """
        # Average over batch and heads
        avg_attention = attention_weights.mean(axis=(0, 1))  # (seq_len, seq_len)

        seq_len = avg_attention.shape[0]

        distances = []
        attention_strengths = []

        for query_pos in range(seq_len):
            for key_pos in range(query_pos + 1):
                distance = query_pos - key_pos
                attention_strength = avg_attention[query_pos, key_pos]

                distances.append(distance)
                attention_strengths.append(attention_strength)

        # Sort by distance
        distances = np.array(distances)
        attention_strengths = np.array(attention_strengths)

        sorted_idx = np.argsort(distances)
        sorted_distances = distances[sorted_idx]
        sorted_attention = attention_strengths[sorted_idx]

        # Cumulative attention
        cumsum = np.cumsum(sorted_attention)
        total = cumsum[-1]

        # Find threshold
        threshold_idx = np.argmax(cumsum >= threshold * total)
        erf = sorted_distances[threshold_idx]

        return float(erf)

    def plot_attention_heatmap(
        self,
        model_name: str,
        window_size: int,
        fold_idx: int,
        layer_idx: int = 0,
        output_path: Optional[Path] = None
    ):
        """
        Plot attention heatmap for a specific layer.
        
        Args:
            model_name: Model name
            window_size: Window size
            fold_idx: Fold index
            layer_idx: Layer to visualize (default 0)
            output_path: Where to save plot (optional)
        """
        attention_by_layer = self.load_attention_patterns(model_name, window_size, fold_idx)

        if layer_idx not in attention_by_layer:
            raise ValueError(f"Layer {layer_idx} not found")

        attention = attention_by_layer[layer_idx]
        avg_attention = attention.mean(axis=(0, 1))  # (seq_len, seq_len)

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(avg_attention, cmap='viridis', aspect='auto')

        ax.set_xlabel("Key Position (Timestep)")
        ax.set_ylabel("Query Position (Timestep)")
        ax.set_title(f"Attention: {model_name}, W={window_size}, Fold={fold_idx}")

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Attention Weight")

        # Default output path
        if output_path is None:
            output_path = (
                self.attention_dir.parent / "plots" /
                f"attention_heatmap_{model_name}_fold_{fold_idx}_w{window_size}.png"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"[AttentionAnalyzer] Saved heatmap: {output_path}")
        plt.close()

    def compare_effective_receptive_field(
        self,
        model_name: str,
        window_sizes: List[int],
        fold_idx: int,
        output_path: Optional[Path] = None
    ):
        """
        Compare ERF across window sizes for a specific fold.
        
        Args:
            model_name: Model name
            window_sizes: List of window sizes to compare
            fold_idx: Fold index
            output_path: Where to save plot (optional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax1, ax2 = axes

        erfs = []
        erf_ratios = []

        for window_size in window_sizes:
            try:
                attention = self.load_attention_patterns(model_name, window_size, fold_idx)
                last_layer = max(attention.keys())
                erf = self.compute_effective_receptive_field(attention[last_layer])

                erfs.append(erf)
                erf_ratios.append(erf / window_size)
            except FileNotFoundError:
                logger.warning(f"Attention file not found for window={window_size}")
                erfs.append(0)
                erf_ratios.append(0)

        # Plot absolute ERF
        ax1.plot(window_sizes, erfs, 'o-', linewidth=2, markersize=8, label='Effective RF')
        ax1.plot(window_sizes, window_sizes, '--', linewidth=2, alpha=0.5, label='Full Window')
        ax1.set_xlabel("Window Size")
        ax1.set_ylabel("Effective Receptive Field")
        ax1.set_title(f"ERF: {model_name}, Fold={fold_idx}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot ratio
        ax2.bar(range(len(window_sizes)), erf_ratios, alpha=0.7)
        ax2.set_xlabel("Window Size")
        ax2.set_ylabel("ERF / Window Size")
        ax2.set_title("Context Utilization Ratio")
        ax2.set_xticks(range(len(window_sizes)))
        ax2.set_xticklabels([f"W={w}" for w in window_sizes])
        ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3, axis='y')

        for i, (erf, ratio) in enumerate(zip(erfs, erf_ratios)):
            if ratio > 0:  # Only annotate if we have data
                ax2.text(i, ratio + 0.02, f"{ratio:.0%}\n({erf:.0f})",
                         ha='center', fontsize=9)

        plt.tight_layout()
        
        # Default output path
        if output_path is None:
            output_path = self.attention_dir.parent / "plots" / f"erf_comparison_fold_{fold_idx}.png"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"[AttentionAnalyzer] Saved ERF comparison: {output_path}")
        plt.close()
