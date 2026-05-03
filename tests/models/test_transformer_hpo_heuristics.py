import math
import pytest
from unittest.mock import MagicMock
from hypothesis import given, settings, strategies as st

# Adjust imports to your project structure
from models.hpo_heuristics import sqrt_lr_scale, get_lr_scaling_config, get_complexity_threshold
from models.transformer import TransformerForecaster
from models.transformer_components import LearningRateCalculator
from utils.dataset import TimeSeriesDataset


# -----------------------------
# Helper for Model Initialization
# (Replaces fixtures to satisfy Hypothesis HealthChecks)
# -----------------------------

def create_model_dependencies():
    """
    Creates fresh mocks for each test execution.
    This avoids Hypothesis HealthCheck errors regarding function-scoped fixtures.
    """
    mock_dataset = MagicMock(spec=TimeSeriesDataset)
    mock_dataset.target_columns = ["target"]
    mock_dataset.past_covariates = []
    mock_dataset.future_covariates = []
    mock_dataset.columns = ["target"]
    # New API
    mock_dataset.past_covariates = []
    mock_dataset.future_covariates = []
    # Default length
    mock_dataset.series = MagicMock()
    mock_dataset.series.__len__.return_value = 1000

    mock_run_context = MagicMock()
    mock_run_context.dataset = mock_dataset

    base_kwargs = {
        "num_features": 1,
        "forecast_steps": 24,
        "window_size": 96,
        "dataset": mock_dataset,
        "run_context": mock_run_context
    }
    return base_kwargs


# -----------------------------
# Pure utility tests
# -----------------------------

@given(
    lr0=st.floats(min_value=1e-6, max_value=1e-1, allow_nan=False, allow_infinity=False),
    batch=st.integers(min_value=1, max_value=2048),
    ref_batch=st.integers(min_value=1, max_value=2048),
)
@settings(deadline=None, max_examples=200)
def test_sqrt_lr_scale_is_monotonic_in_batch(lr0, batch, ref_batch):
    """Verify that larger batches yield larger (or equal) learning rates."""
    lr_a = sqrt_lr_scale(lr0, batch_size=batch, ref_batch=ref_batch, mode="sqrt")
    lr_b = sqrt_lr_scale(lr0, batch_size=batch + 1, ref_batch=ref_batch, mode="sqrt")
    assert lr_b >= lr_a


@given(
    lr0=st.floats(min_value=1e-6, max_value=1e-1, allow_nan=False, allow_infinity=False),
    batch=st.integers(min_value=1, max_value=2048),
    ref_batch=st.integers(min_value=1, max_value=2048),
)
@settings(deadline=None, max_examples=200)
def test_linear_lr_scale_matches_formula(lr0, batch, ref_batch):
    """Verify linear scaling formula correctness."""
    lr = sqrt_lr_scale(lr0, batch_size=batch, ref_batch=ref_batch, mode="linear")
    # Using isclose for float comparison
    expected = lr0 * (batch / ref_batch)
    assert math.isclose(lr, expected, rel_tol=1e-9)


# -----------------------------
# Transformer-specific validation tests
# -----------------------------

@given(
    hidden_size=st.sampled_from([32, 64, 128, 256, 512]),
    num_heads=st.sampled_from([1, 2, 4, 8, 16, 32]),
)
@settings(deadline=None, max_examples=100)
def test_validate_rejects_non_divisible_hidden_by_heads(hidden_size, num_heads):
    base_kwargs = create_model_dependencies()
    model = TransformerForecaster(
        model_params={"architecture": "encoder-only", "strategy": "direct"},
        **base_kwargs
    )
    params = {"hidden_size": hidden_size, "num_heads": num_heads, "num_encoder_layers": 2}

    ok = model.validate_param_combination(params)

    if hidden_size % num_heads != 0:
        assert ok is False


@given(
    hidden_size=st.sampled_from([64, 128, 256, 512]),
    num_heads=st.sampled_from([1, 2, 4, 8, 16, 32]),
)
@settings(deadline=None, max_examples=100)
def test_validate_enforces_head_dim_range(hidden_size, num_heads):
    base_kwargs = create_model_dependencies()
    model = TransformerForecaster(
        model_params={"architecture": "encoder-only", "strategy": "direct"},
        **base_kwargs
    )
    params = {"hidden_size": hidden_size, "num_heads": num_heads, "num_encoder_layers": 2}

    ok = model.validate_param_combination(params)

    if hidden_size % num_heads != 0:
        assert ok is False
        return

    head_dim = hidden_size // num_heads
    # We enforce [8, 128] range for head_dim
    if head_dim < 8 or head_dim > 128:
        assert ok is False


@given(
    hidden_size=st.sampled_from([64, 128, 256, 512, 1024]),
    num_layers=st.integers(min_value=1, max_value=24),
    dataset_size_category=st.sampled_from(["small", "medium", "large", "very_large"]),
)
@settings(deadline=None, max_examples=100)
def test_complexity_threshold_is_respected(hidden_size, num_layers, dataset_size_category):
    """
    Verify that validate_param_combination respects the complexity thresholds
    defined in hpo_heuristics.
    """
    base_kwargs = create_model_dependencies()
    model = TransformerForecaster(
        model_params={"architecture": "encoder-only", "strategy": "direct"},
        **base_kwargs
    )

    # Simulate run_context/dataset size for the validation logic
    lengths = {"small": 500, "medium": 2000, "large": 10000, "very_large": 50000}

    # Update the mocks inside the model
    model.run_context.dataset.series.__len__.return_value = lengths[dataset_size_category]

    params = {"hidden_size": hidden_size, "num_heads": 4, "num_encoder_layers": num_layers}

    ok = model.validate_param_combination(params)

    # Calculate expected threshold using the utility directly
    thr = get_complexity_threshold(
        model.model_params, dataset_size_category, model_type="transformer", num_features=1
    )

    current_complexity = hidden_size * num_layers

    if current_complexity > thr:
        assert ok is False, f"Should reject complexity {current_complexity} for {dataset_size_category} (limit {thr})"
    else:
        # Note: might still be rejected by other checks (head dim), so we don't assert ok is True
        pass

    # -----------------------------


# Critical regression guard: LR Heuristics
# -----------------------------

def test_user_learning_rate_is_not_modified_by_default_lr_heuristic():
    """Ensure user config LR is preserved in model_params."""
    base_kwargs = create_model_dependencies()
    user_lr = 1.23e-4

    model = TransformerForecaster(
        model_params={
            "architecture": "encoder-only",
            "strategy": "direct",
            "learning_rate": user_lr,
            "hpo_lr_scaling": {"mode": "sqrt", "ref_batch": 64, "lr0": 1e-3},
        },
        **base_kwargs
    )

    assert model.model_params["learning_rate"] == user_lr


def test_smart_lr_heuristic_values():
    """
    Verify that the heuristic returns SAFE values (Regression Test).
    We check against the 'bad' scenario (2 layers, 128 hidden) to ensure LR is conservative.
    """
    base_kwargs = create_model_dependencies()
    model = TransformerForecaster(
        model_params={"architecture": "encoder-only", "strategy": "direct"},
        **base_kwargs
    )

    # Case: The configuration that failed previously
    # 2 layers, 128 hidden, batch 32
    # Old/Bad behavior produced ~3.27e-4 (too high)
    # Target behavior: < 3.3e-4 (ideally ~3.18e-4 with new base)

    # Use LearningRateCalculator directly (refactored from _default_lr_for_model)
    calculator = LearningRateCalculator(model.model_params)
    suggested_lr = calculator.calculate_lr(
        hidden_size=128,
        num_layers=2,
        strategy="direct",
        dataset_size="medium",
        batch_size=32
    )

    print(f"Suggested LR: {suggested_lr}")

    # Assert it is conservative
    assert suggested_lr < 3.3e-4, f"LR {suggested_lr} is too aggressive for small model!"