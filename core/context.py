# core/context.py
"""
RunContext - Centralized artifact management for SGOLD.

This module provides a single source of truth for directory structure and
artifact naming conventions, eliminating distributed I/O logic across models.
"""

import copy
import json
import logging
from dataclasses import dataclass, field, replace, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class PathJSONEncoder(json.JSONEncoder):
    """
    JSON encoder that handles Path objects and numpy data types.

    Essential for ML workflows where Path and numpy types are ubiquitous.
    """
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        if hasattr(obj, 'tolist'):  # Handle numpy arrays/scalars
            return obj.tolist()
        return super().default(obj)


@dataclass
class RunContext:
    """
    Execution context and directory paths for artifacts.

    Acts as a single source of truth for artifact locations and run metadata.
    Used in trainer to organize experiment structure; components receive
    concrete values extracted from context (loose coupling pattern).

    Design:
        - Mutable DTO (not frozen - we have I/O operations)
        - Deep copy in with_metadata() (ensures metadata isolation)
        - Pure get_artifact_path() (no side effects)
        - Fallback for unknown categories (extensibility)

    Example:
        # In trainer:
        ctx = RunContext.from_base_path(
            Path("results/runs/exp_run123"),
            run_id="run123",
            experiment_name="vanishing_gradient"
        )
        ctx.create_directories()

        # Per fold:
        fold_ctx = ctx.with_metadata(
            model_name="transformer",
            fold_idx=0,
            window_size=96
        )

        # Extract values for components:
        model = TransformerForecaster(
            gradients_dir=fold_ctx.gradients_dir,
            model_name=fold_ctx.model_name,
            fold_idx=fold_ctx.fold_idx,
            window_size=fold_ctx.window_size
        )
    """

    # Core identifiers
    run_id: str
    experiment_name: str
    base_dir: Path

    # Artifact subdirectories
    gradients_dir: Path
    attention_dir: Path
    plots_dir: Path
    data_dir: Path
    checkpoints_dir: Optional[Path] = None

    # Run-specific metadata (optional, set per model/fold)
    model_name: Optional[str] = None
    model_type: Optional[str] = None
    fold_idx: Optional[int] = None
    window_size: Optional[int] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def create_directories(self) -> None:
        """
        Create physical directory structure for known artifact types.

        Call once at the beginning to set up directory structure.
        Optional directories (e.g., checkpoints_dir) are only created if defined.

        Raises:
            OSError: If directory creation fails due to permissions or other OS issues
        """
        dirs_to_create = [
            self.gradients_dir,
            self.attention_dir,
            self.plots_dir,
            self.data_dir
        ]
        if self.checkpoints_dir:
            dirs_to_create.append(self.checkpoints_dir)

        for path in dirs_to_create:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.error(f"Failed to create directory {path}: {e}")
                raise

    def with_metadata(self, **kwargs) -> 'RunContext':
        """
        Performs DEEP COPY of the metadata dictionary to prevent side effects.

        Useful for creating fold-specific contexts from a base context.
        Original context remains unchanged (copy-on-update pattern).

        Args:
            **kwargs: Fields to update (model_name, fold_idx, window_size, etc.)

        Returns:
            New RunContext instance with updated fields

        Example:
            fold_ctx = base_ctx.with_metadata(
                model_name="transformer",
                fold_idx=0,
                window_size=96
            )

        Safety:
            If 'metadata' is NOT provided in kwargs, we perform a deep copy of
            the existing metadata. This ensures that modifying nested structures
            (lists, dicts) in the new context does not affect the parent context.
        """
        if 'metadata' not in kwargs:
            kwargs['metadata'] = copy.deepcopy(self.metadata)

        return replace(self, **kwargs)

    def get_artifact_path(
        self,
        category: str,
        suffix: str,
        extension: str,
        include_window: bool = True,
        include_fold: bool = True
    ) -> Path:
        """
        Generate standardized artifact path without side effects (pure function).

        Logic:
            1. Selects directory based on category.
               Falls back to base_dir/category if category is unknown or None.
            2. Constructs filename: {model}_{fold}_{window}_{suffix}.{ext}

        Args:
            category: Artifact type ("gradients", "attention", "plots", "data", "checkpoints")
                     Unknown categories fallback to base_dir/category
            suffix: Descriptive suffix (e.g., "gradients", "attention", "predictions")
            extension: File extension without dot (e.g., "json", "npz", "png")
            include_window: Include window size in filename
            include_fold: Include fold index in filename

        Returns:
            Full path to artifact (directory may not exist yet - caller's responsibility)

        Example:
            path = ctx.get_artifact_path("gradients", "gradients", "json")
            # → .../gradients/transformer_fold_0_w96_gradients.json

            # Component must ensure parent exists before saving:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(data, f)
        """
        # 1. Determine Directory
        known_dirs = {
            "gradients": self.gradients_dir,
            "attention": self.attention_dir,
            "plots": self.plots_dir,
            "data": self.data_dir,
            "checkpoints": self.checkpoints_dir,
        }

        # Retrieve directory from map; handle case where key exists but value is None
        target_dir = known_dirs.get(category)

        if target_dir is None:
            # Fallback for unknown categories or undefined optional dirs (e.g., checkpoints_dir=None)
            target_dir = self.base_dir / category
            logger.debug(f"Using fallback directory for category '{category}': {target_dir}")

        # 2. Build Filename parts
        parts: List[str] = []

        # Model name (default to "model" if not set)
        parts.append(self.model_name if self.model_name else "model")

        # Fold index (optional)
        if include_fold and self.fold_idx is not None:
            parts.append(f"fold_{self.fold_idx}")

        # Window size (optional)
        if include_window and self.window_size is not None:
            parts.append(f"w{self.window_size}")

        # Suffix
        if suffix:
            parts.append(suffix)

        filename = f"{'_'.join(parts)}.{extension}"
        return target_dir / filename

    def save_metadata(self) -> None:
        """
        Save context metadata to JSON using standard naming convention.

        Snapshots the full state of the context for reproducibility.
        Uses PathJSONEncoder to handle Path and numpy types automatically.

        The metadata file follows the same naming convention as other artifacts:
        {model_name}_fold_{fold_idx}_w{window_size}_metadata.json

        Raises:
            Exception: If metadata saving fails (logged and re-raised)

        Note:
            Creates parent directory if needed (crucial for fallback categories
            not created in create_directories()).
        """
        file_path = self.get_artifact_path(
            category="data",
            suffix="metadata",
            extension="json"
        )

        try:
            # Ensure parent exists (crucial for fallback categories not created in create_directories)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            data = asdict(self)
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, cls=PathJSONEncoder)

            logger.info(f"Saved run metadata to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata to {file_path}: {e}")
            raise

    @property
    def run_name(self) -> str:
        """
        Generate run identifier for logging/display.

        Returns:
            String in format: {experiment_name}_{run_id}
            Example: "vanishing_gradient_run123"
        """
        return f"{self.experiment_name}_{self.run_id}"

    @classmethod
    def from_base_path(cls, base_path: Path, **kwargs) -> 'RunContext':
        """
        Factory method to initialize context from a root directory.

        Automatically sets up standard subdirectory structure:
        - base_path/gradients
        - base_path/attention
        - base_path/plots
        - base_path/data
        - base_path/checkpoints

        Args:
            base_path: Root directory for the run
            **kwargs: Additional context fields (run_id, experiment_name, etc.)

        Returns:
            Initialized RunContext with standard directory structure

        Example:
            ctx = RunContext.from_base_path(
                Path("results/runs/exp_run123"),
                run_id="run123",
                experiment_name="vanishing_gradient"
            )
            ctx.create_directories()
        """
        base_path = Path(base_path)
        return cls(
            base_dir=base_path,
            gradients_dir=base_path / "gradients",
            attention_dir=base_path / "attention",
            plots_dir=base_path / "plots",
            data_dir=base_path / "data",
            checkpoints_dir=base_path / "checkpoints",
            **kwargs
        )
