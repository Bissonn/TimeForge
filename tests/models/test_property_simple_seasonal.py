import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path
from hypothesis import given, settings, strategies as st

from models.simple import SimpleSeasonalForecaster
from utils.dataset import TimeSeriesDataset
from core.context import RunContext

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@st.composite
def seasonal_scenario(draw):
    """
    Generates a consistent scenario for SimpleSeasonalForecaster:
    - seasonal_period (S)
    - forecast_steps (H)
    - num_features (F)
    - train_data (DataFrame of length N >= S)
    """
    # 1. Dimensions
    seasonal_period = draw(st.integers(min_value=2, max_value=50))
    forecast_steps = draw(st.integers(min_value=1, max_value=100))
    num_features = draw(st.integers(min_value=1, max_value=5))

    # 2. Train length must be at least seasonal_period to fit
    min_train = seasonal_period
    max_train = seasonal_period * 5
    n_train = draw(st.integers(min_value=min_train, max_value=max_train))

    # 3. Data generation
    # Use simple range index for robustness in this specific property test
    idx = pd.RangeIndex(start=0, stop=n_train)
    cols = [f"feat_{i}" for i in range(num_features)]

    # Random floats
    values = draw(
        st.lists(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=n_train * num_features,
            max_size=n_train * num_features
        )
    )
    data_np = np.array(values).reshape(n_train, num_features)
    df = pd.DataFrame(data_np, index=idx, columns=cols)

    return {
        "seasonal_period": seasonal_period,
        "forecast_steps": forecast_steps,
        "num_features": num_features,
        "train_df": df
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None)
@given(scenario=seasonal_scenario())
def test_simple_seasonal_repetition_invariant(scenario):
    """
    Property:
        For any seasonal_period S and forecast_steps H:
        The prediction at step h (0 <= h < H) MUST be equal to the value
        stored in the last observed season at index (h % S).

        Pred[h] == LastSeasonBuffer[h % S]
    """
    # Create ephemeral context manually for Hypothesis
    with tempfile.TemporaryDirectory() as tmp_dir:
        run_context = RunContext.from_base_path(
            base_path=Path(tmp_dir),
            run_id="hyp_run",
            experiment_name="hyp_exp"
        )

        S = scenario["seasonal_period"]
        H = scenario["forecast_steps"]
        F = scenario["num_features"]
        train_df = scenario["train_df"]

        # 1. Setup Dataset (Mock wrapper)
        # The model init requires a dataset object, mainly for configuration context
        ds = TimeSeriesDataset(
            "dummy",
            {"datasets": {"dummy": {}}},
            num_features=F,
            data=train_df,
            columns=list(train_df.columns)
        )

        # 2. Initialize Model
        model = SimpleSeasonalForecaster(
            model_params={"seasonal_period": S},
            num_features=F,
            forecast_steps=H,
            window_size=S,  # Window size is just a constraint for validation
            dataset=ds,
            run_context=run_context # <--- Inject manual context
        )

        # 3. Fit
        model.fit(train_df)

        # Verify internal buffer correctness
        # The buffer should be the tail of training data of length S
        expected_buffer = train_df.iloc[-S:]
        pd.testing.assert_frame_equal(model.last_season_buffer, expected_buffer)

        # 4. Predict
        preds = model.predict(forecast_steps=H)

        # --- VERIFICATION ---

        # A. Shape check
        assert preds.shape == (H, F)
        assert list(preds.columns) == list(train_df.columns)

        # B. Index continuity check (for RangeIndex)
        last_train_idx = train_df.index[-1]
        expected_start = last_train_idx + 1
        assert preds.index[0] == expected_start
        assert preds.index[-1] == expected_start + H - 1

        # C. The Core Invariant: Tiling
        # Convert to numpy for easy indexing
        buffer_values = expected_buffer.values  # (S, F)
        pred_values = preds.values  # (H, F)

        for h in range(H):
            # The mathematical definition of seasonal naive forecast
            buffer_idx = h % S

            row_pred = pred_values[h]
            row_expected = buffer_values[buffer_idx]

            np.testing.assert_allclose(
                row_pred,
                row_expected,
                err_msg=f"Mismatch at step h={h} (buffer index {buffer_idx})"
            )
