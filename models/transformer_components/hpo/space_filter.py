"""
Search space filter for Transformer HPO.

Filters out invalid parameter combinations based on architecture and attention type.
"""
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class SearchSpaceFilter:
    """
    Filters HPO search space based on architecture and attention configuration.

    Removes parameters that are not applicable given the fixed configuration,
    preventing wasted HPO trials on invalid parameter combinations.
    """

    @staticmethod
    def filter(
        param_space: Dict[str, Any],
        fixed_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Filter search space based on architecture, attention type, and constraints.

        Args:
            param_space: Parameters to optimize
            fixed_params: Fixed parameters from config

        Returns:
            Filtered param_space with invalid parameters removed
        """
        filtered = param_space.copy()
        removed = []

        architecture = fixed_params.get("architecture", "encoder-only")
        attention_type = fixed_params.get("attention_type", "full")

        # Apply architecture-based filtering
        removed.extend(
            SearchSpaceFilter._filter_by_architecture(filtered, architecture)
        )

        # Apply attention-based filtering
        removed.extend(
            SearchSpaceFilter._filter_by_attention(filtered, attention_type)
        )

        # Remove non-optimizable training params
        removed.extend(
            SearchSpaceFilter._filter_training_params(filtered)
        )

        # Log results
        SearchSpaceFilter._log_filtering_results(filtered, removed)

        return filtered

    @staticmethod
    def _filter_by_architecture(
        filtered: Dict[str, Any],
        architecture: str
    ) -> List[str]:
        """
        Filter parameters based on architecture.

        Args:
            filtered: Mutable dict to filter (modified in-place)
            architecture: Architecture type ("encoder-only" or "encoder-decoder")

        Returns:
            List of removed parameter names
        """
        removed = []

        if architecture == "encoder-only":
            # Remove decoder specific params
            for param in ["tgt_init", "num_decoder_layers", "decoder_input_size"]:
                if param in filtered:
                    del filtered[param]
                    removed.append(param)

        elif architecture == "encoder-decoder":
            # Remove encoder-only specific params
            if "readout" in filtered:
                del filtered["readout"]
                removed.append("readout")

        return removed

    @staticmethod
    def _filter_by_attention(
        filtered: Dict[str, Any],
        attention_type: str
    ) -> List[str]:
        """
        Filter parameters based on attention type.

        Args:
            filtered: Mutable dict to filter (modified in-place)
            attention_type: Attention type ("full" or "local")

        Returns:
            List of removed parameter names
        """
        removed = []

        if attention_type == "full":
            if "attention_window_size" in filtered:
                del filtered["attention_window_size"]
                removed.append("attention_window_size")

        return removed

    @staticmethod
    def _filter_training_params(filtered: Dict[str, Any]) -> List[str]:
        """
        Remove training parameters that should not be optimized.

        Args:
            filtered: Mutable dict to filter (modified in-place)

        Returns:
            List of removed parameter names
        """
        removed = []

        # early_stopping_patience: Remove (not optimized, controlled by validation)
        if "early_stopping_patience" in filtered:
            del filtered["early_stopping_patience"]
            removed.append("early_stopping_patience (not optimized)")

        # batch_size: KEEP (important optimization target!)
        # epochs: KEEP if present (though usually fixed)

        return removed

    @staticmethod
    def _log_filtering_results(
        filtered: Dict[str, Any],
        removed: List[str]
    ) -> None:
        """
        Log filtering results.

        Args:
            filtered: Filtered parameter space
            removed: List of removed parameters
        """
        if removed:
            logger.info(
                f"[Transformer SmartHPO] Filtered {len(removed)} invalid params: {removed}"
            )

        # Log if training params are being optimized
        training_in_search = [p for p in ["batch_size", "epochs"] if p in filtered]
        if training_in_search:
            logger.info(
                f"[Transformer SmartHPO] Training params in search space: {training_in_search}"
            )
