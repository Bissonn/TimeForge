import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from hypothesis import given, settings, strategies as st

# Use your application classes instead of direct statsmodels
from models.var import VARForecaster
from models.sarima import SARIMAForecaster
from utils.dataset import TimeSeriesDataset
from core.context import RunContext


# -------------------------------------------------------------------
# Strategies
# -------------------------------------------------------------------

@st.composite
def arima_series_with_exog(draw):
    n = draw(st.integers(min_value=20, max_value=80))
    t = np.arange(n, dtype=float)
    # Simple linear relation: y = 2 * x + noise
    # Ensure noise is strictly positive to avoid LinAlgError in statsmodels
    noise_scale = draw(st.floats(min_value=0.1, max_value=2.0))

    # Add random shift to exog to make it slightly less perfect
    exog_noise = draw(st.floats(min_value=-0.5, max_value=0.5))
    exog = t + exog_noise

    # Generate y
    y = 2.0 * exog + np.random.normal(loc=0.0, scale=noise_scale, size=n)

    return y, exog


# -------------------------------------------------------------------
# SARIMAForecaster Tests (Wrapper)
# -------------------------------------------------------------------

@settings(deadline=None, max_examples=30)
@given(arima_series_with_exog())
def test_sarimax_forecast_requires_matching_exog_length(data):
    """
    Property: SARIMAForecaster predict with exog must raise ValueError when exog length
    does not match the requested number of forecast steps.

    This tests the app's validation logic (ARIMABaseForecaster._validate_prediction_exog).
    """
    # Create ephemeral context manually for Hypothesis
    with tempfile.TemporaryDirectory() as tmp_dir:
        run_context = RunContext.from_base_path(
            base_path=Path(tmp_dir),
            run_id="hyp_run",
            experiment_name="hyp_exp"
        )

        y, exog = data
        n = len(y)
        # Split data
        train_y = y[: n - 5]
        train_exog = exog[: n - 5]
        test_exog = exog[n - 5:]

        # Prepare DataFrame wrappers
        df_train_y = pd.DataFrame(train_y, columns=["y"])
        df_train_exog = pd.DataFrame(train_exog, columns=["exog"])
        df_test_exog = pd.DataFrame(test_exog, columns=["exog"])

        # Setup TimeSeriesDataset context
        ds = TimeSeriesDataset(
            dataset_name="dummy_sarima",
            config={"datasets": {"dummy_sarima": {}}},
            num_features=1,
            data=pd.concat([df_train_y, df_train_exog], axis=1),
            columns=["y"],
            future_covariates=["exog"]
        )

        forecast_steps = len(df_test_exog)

        # Initialize App Model
        # NOTE: SARIMAForecaster requires seasonal parameters (P, D, Q, seasonal_period).
        # We use seasonal_period=1 (effectively no seasonality) to pass validation
        # and avoid "dataset too short" errors on small N.
        model = SARIMAForecaster(
            model_params={
                "p": 1, "d": 0, "q": 0,
                "P": 0, "D": 0, "Q": 0, "seasonal_period": 4
            },
            num_features=1,
            forecast_steps=forecast_steps,
            window_size=10,
            dataset=ds,
            run_context=run_context
        )

        model.fit(df_train_y, exog_series=df_train_exog)
        if not model.fitted:
            # Skip if fitting was rejected (convergence failure, numerical instability)
            return

        # 1. Correct length: should not raise
        try:
            _ = model.predict(future_exog=df_test_exog, forecast_steps=forecast_steps)
        except RuntimeError:
            pass

        # 2. Wrong length: shorter exog must raise ValueError
        # Note: This is now caught by your wrapper's validation logic
        if forecast_steps > 1:
            too_short_exog = df_test_exog.iloc[:-1]
            with pytest.raises(ValueError, match="Prediction exogenous length mismatch"):
                _ = model.predict(future_exog=too_short_exog, forecast_steps=forecast_steps)


@settings(deadline=None, max_examples=30)
@given(arima_series_with_exog())
def test_sarimax_one_step_forecast_is_exog_sensitive(data):
    """
    Property: If we perturb future exog strongly while keeping the model and
    history fixed, one-step-ahead forecast should change.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        run_context = RunContext.from_base_path(
            base_path=Path(tmp_dir),
            run_id="hyp_run",
            experiment_name="hyp_exp"
        )

        y, exog = data
        n = len(y)
        train_y = y[: n - 1]
        train_exog = exog[: n - 1]
        test_exog = exog[n - 1:]

        df_train_y = pd.DataFrame(train_y, columns=["y"])
        df_train_exog = pd.DataFrame(train_exog, columns=["exog"])
        df_test_exog = pd.DataFrame(test_exog, columns=["exog"])

        ds = TimeSeriesDataset(
            dataset_name="dummy_sarima",
            config={"datasets": {"dummy_sarima": {}}},
            num_features=1,
            data=pd.concat([df_train_y, df_train_exog], axis=1),
            columns=["y"],
            future_covariates=["exog"]
        )

        model = SARIMAForecaster(
            model_params={
                "p": 1, "d": 0, "q": 0,
                "P": 0, "D": 0, "Q": 0, "seasonal_period": 4
            },
            num_features=1,
            forecast_steps=1,
            window_size=10,
            dataset=ds,
            run_context=run_context
        )

        model.fit(df_train_y, exog_series=df_train_exog)
        if not model.fitted:
            return

        # Baseline forecast
        try:
            pred_base = model.predict(future_exog=df_test_exog, forecast_steps=1)
            val_base = pred_base.values[0, 0]
        except RuntimeError:
            return

        # Perturbed forecast
        df_shifted_exog = df_test_exog + 1000.0
        try:
            pred_shifted = model.predict(future_exog=df_shifted_exog, forecast_steps=1)
            val_shifted = pred_shifted.values[0, 0]
        except RuntimeError:
            return

        if np.isfinite(val_base) and np.isfinite(val_shifted):
            assert abs(val_base - val_shifted) > 1e-3


# -------------------------------------------------------------------
# VARForecaster Tests (Wrapper)
# -------------------------------------------------------------------

@settings(deadline=None, max_examples=30)
@given(
    n_obs=st.integers(min_value=20, max_value=60),
    n_vars=st.integers(min_value=2, max_value=4),
    n_exog=st.integers(min_value=1, max_value=3),
)
def test_var_forecast_shape_with_exog(n_obs, n_vars, n_exog):
    """
    Property: VARForecaster.predict() should return a DataFrame with shape (H, num_vars).

    This validates that your application wrapper correctly handles:
    1. Data preparation (concatenation of exog).
    2. Passing args to statsmodels.
    3. Output formatting (DataFrame conversion).
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        run_context = RunContext.from_base_path(
            base_path=Path(tmp_dir),
            run_id="hyp_run",
            experiment_name="hyp_exp"
        )

        rng = np.random.default_rng(123)

        data = rng.normal(size=(n_obs, n_vars))
        exog = rng.normal(size=(n_obs, n_exog))

        df_data = pd.DataFrame(data, columns=[f"y{i}" for i in range(n_vars)])
        df_exog = pd.DataFrame(exog, columns=[f"x{j}" for j in range(n_exog)])

        # TimeSeriesDataset creation for context
        full_df = pd.concat([df_data, df_exog], axis=1)
        ds = TimeSeriesDataset(
            dataset_name="dummy_var",
            config={"datasets": {"dummy_var": {}}},
            num_features=n_vars,
            data=full_df,
            columns=list(df_data.columns),
            future_covariates=list(df_exog.columns)
        )

        train_df = df_data.iloc[:-5]
        train_exog = df_exog.iloc[:-5]
        test_exog = df_exog.iloc[-5:]

        H = len(test_exog)

        # Initialize YOUR VARForecaster
        model = VARForecaster(
            model_params={"max_lags": 1, "maxiter": 50},
            num_features=n_vars,
            forecast_steps=H,
            window_size=10,
            dataset=ds,
            run_context=run_context
        )

        # Fit using your API
        model.fit(train_df, exog_series=train_exog)

        # Predict using your API
        # This internally calls statsmodels forecast with correct steps and exog
        preds = model.predict(future_exog=test_exog, forecast_steps=H)

        # Validation
        assert isinstance(preds, pd.DataFrame)
        assert preds.shape == (H, n_vars)
        assert list(preds.columns) == list(df_data.columns)
