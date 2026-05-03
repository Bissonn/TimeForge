"""
Search space analyzer for Transformer HPO.

Provides utilities for analyzing HPO search space complexity and coverage.
"""
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class SearchSpaceAnalyzer:
    """
    Analyzes HPO search space complexity and provides coverage warnings.

    This helps prevent common HPO mistakes like:
    - Huge search spaces with few trials (under-sampling)
    - Too many discrete parameters (combinatorial explosion)
    - Unrealistic expectations about coverage
    """

    @staticmethod
    def analyze(
        param_space: Dict[str, Any],
        n_trials: int
    ) -> Dict[str, Any]:
        """
        Analyze search space complexity and warn if too large relative to n_trials.

        Args:
            param_space: HPO search space definition
            n_trials: Number of trials to run

        Returns:
            Dict with analysis results:
            - discrete_combos: Number of discrete parameter combinations
            - discrete_params: List of discrete parameter descriptions
            - continuous_params: List of continuous parameter names
            - estimated_combos: Rough estimate of total search space size
            - coverage: Percentage of space that will be sampled
            - n_trials: Number of trials (echoed back)

        Example:
            >>> space = {
            >>>     "hidden_size": [64, 128, 256],  # 3 choices
            >>>     "num_heads": [4, 8],            # 2 choices
            >>>     "dropout": {"min": 0.0, "max": 0.3}  # continuous
            >>> }
            >>> result = SearchSpaceAnalyzer.analyze(space, n_trials=20)
            >>> # discrete_combos = 3 * 2 = 6
            >>> # continuous: ~10 reasonable values per param
            >>> # estimated_combos = 6 * 10 = 60
            >>> # coverage = 20 / 60 = 33%
        """
        # ───────────────────────────────────────────────────────────────────────
        # Count Discrete Combinations
        # ───────────────────────────────────────────────────────────────────────

        discrete_combos = 1
        discrete_params = []
        continuous_params = []

        for key, value in param_space.items():
            if isinstance(value, list):
                # Categorical or discrete parameter
                discrete_combos *= len(value)
                discrete_params.append(f"{key}({len(value)})")
            elif isinstance(value, dict) and 'min' in value:
                # Continuous parameter
                continuous_params.append(key)

        # ───────────────────────────────────────────────────────────────────────
        # Estimate Total Combinations
        # ───────────────────────────────────────────────────────────────────────
        # Assume ~10 reasonable values per continuous parameter

        continuous_multiplier = 10 ** len(continuous_params)
        estimated_combos = discrete_combos * continuous_multiplier

        # ───────────────────────────────────────────────────────────────────────
        # Calculate Coverage
        # ───────────────────────────────────────────────────────────────────────

        coverage = n_trials / estimated_combos if estimated_combos > 0 else 1.0

        # ───────────────────────────────────────────────────────────────────────
        # Log Info
        # ───────────────────────────────────────────────────────────────────────

        logger.info(
            f"[HPO] Search space: {len(param_space)} parameters "
            f"({len(discrete_params)} discrete, {len(continuous_params)} continuous)"
        )

        if discrete_params:
            logger.info(f"[HPO] Discrete combos: {discrete_combos} = " + " × ".join(discrete_params))

        if continuous_params:
            logger.info(f"[HPO] Continuous params: {', '.join(continuous_params)}")

        logger.info(
            f"[HPO] Estimated space size: ~{estimated_combos:.0f} combinations"
        )
        logger.info(
            f"[HPO] n_trials={n_trials} → coverage: {coverage:.1%}"
        )

        # ───────────────────────────────────────────────────────────────────────
        # Generate Warnings
        # ───────────────────────────────────────────────────────────────────────

        warnings = SearchSpaceAnalyzer._generate_warnings(
            estimated_combos, n_trials, coverage
        )

        return {
            "discrete_combos": discrete_combos,
            "discrete_params": discrete_params,
            "continuous_params": continuous_params,
            "estimated_combos": estimated_combos,
            "coverage": coverage,
            "n_trials": n_trials,
            "warnings": warnings
        }

    @staticmethod
    def _generate_warnings(
        estimated_combos: float,
        n_trials: int,
        coverage: float
    ) -> List[str]:
        """
        Generate warning messages based on coverage analysis.

        Args:
            estimated_combos: Estimated total combinations in search space
            n_trials: Number of trials to run
            coverage: Fraction of space that will be sampled

        Returns:
            List of warning/info messages
        """
        warnings = []

        if coverage < 0.01:  # Less than 1% coverage
            warnings.append(
                f"[HPO] ⚠️  Search space is VERY LARGE (~{estimated_combos:.0f} "
                f"combinations) relative to n_trials={n_trials} (coverage: {coverage:.3%}). "
                f"\n    Recommendations:"
                f"\n    1. Increase n_trials to at least {int(estimated_combos * 0.05)}"
                f"\n    2. Narrow search space (fix some parameters)"
                f"\n    3. Use smart priors to seed good starting points"
            )
        elif coverage < 0.05:  # 1-5% coverage
            warnings.append(
                f"[HPO] Search space is large (~{estimated_combos:.0f} combinations) "
                f"with n_trials={n_trials} (coverage: {coverage:.1%}). "
                f"Consider narrowing space or increasing trials."
            )
        elif coverage > 0.8:  # >80% coverage
            warnings.append(
                f"[HPO] ✓ Good coverage ({coverage:.1%}). Search space is well-matched to n_trials."
            )

        return warnings
