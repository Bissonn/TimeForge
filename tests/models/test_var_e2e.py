"""
End-to-End tests for VARForecaster.

This module verifies the Vector Autoregression (VAR) model integration,
specifically focusing on the handling of exogenous variables, preprocessing pipeline,
and forecast quality on synthetic linear data.
"""
import pytest
import pandas as pd
import numpy as np
from models.factory import ModelFactory
from utils.dataset import TimeSeriesDataset
import models.var

pytestmark = pytest.mark.integration

@pytest.fixture
def base_var_config():
    """Minimal configuration for VAR model."""
    return {
        "max_lags": 1,
        "maxiter": 50,
        "optimize": False,
        "preprocessing": {
            "preprocessing_groups": [{
                "name": "default",
                "apply_to": "__targets__",
                "pipeline": {"scaling": {"enabled": True, "method": "standard"}}
            }]
        }
    }

def create_multivariate_dataset(n_samples=100, n_targets=2, n_enc_exog=0, n_dec_exog=0):
    """
    Creates a synthetic multivariate dataset compatible with VAR requirements.
    """
    data = {}
    date_range = pd.date_range(start="2023-01-01", periods=n_samples, freq="D")

    # Base signals
    t = np.linspace(0, 10, n_samples)

    # Targets (initialized with noise)
    for i in range(n_targets):
        data[f"y{i}"] = np.random.randn(n_samples) * 0.1

    targets = [f"y{i}" for i in range(n_targets)]

    # Exogenous Variables
    enc_cols = []
    for i in range(n_enc_exog):
        col_name = f"enc_exog_{i}"
        data[col_name] = np.sin(t + i) # Deterministic signal
        enc_cols.append(col_name)

    dec_cols = []
    for i in range(n_dec_exog):
        col_name = f"dec_exog_{i}"
        data[col_name] = np.cos(t + i) # Deterministic signal
        dec_cols.append(col_name)

    df = pd.DataFrame(data, index=date_range)

    config = {"datasets": {"var_mock": {}}}

    ds = TimeSeriesDataset(
        "var_mock",
        config,
        data=df,
        num_features=n_targets,
        columns=targets,
        past_covariates=enc_cols,
        future_covariates=dec_cols
    )
    return ds

@pytest.mark.parametrize("n_enc_exog", [0, 1])
@pytest.mark.parametrize("n_dec_exog", [0, 1])
def test_var_fit_predict_exog_combinations(
        base_var_config,
        n_enc_exog,
        n_dec_exog,
        base_context
):
    """
    Smoke test: Verifies that the fit-predict pipeline works without errors
    for various combinations of exogenous variables and active preprocessing.
    """
    n_targets = 2
    forecast_steps = 5
    window_size = 10

    dataset = create_multivariate_dataset(
        n_samples=60,
        n_targets=n_targets,
        n_enc_exog=n_enc_exog,
        n_dec_exog=n_dec_exog
    )
    dataset.split_data(forecast_steps)

    model = ModelFactory.create(
        "var",
        "test_var_model",
        run_context=base_context,
        model_params=base_var_config,
        num_features=n_targets,
        forecast_steps=forecast_steps,
        window_size=window_size,
        dataset=dataset
    )

    train_df = dataset.development_data
    target_cols = dataset.target_columns
    exog_cols = dataset.past_covariates + dataset.future_covariates

    train_endog = train_df[target_cols]
    train_exog = train_df[exog_cols] if exog_cols else None

    # Fit
    model.fit(train_endog, exog_series=train_exog)
    assert model.fitted is True
    assert model.preprocessor is not None # Preprocessor should be initialized

    # Predict
    future_exog = dataset.test_data[exog_cols] if exog_cols else None
    predictions = model.predict(forecast_steps=forecast_steps, future_exog=future_exog)

    # Verify
    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions) == forecast_steps
    assert predictions.shape[1] == n_targets
    assert not predictions.isna().any().any()


def test_var_forecast_quality(
        base_var_config,
        base_context
):
    """
    Quality test: Verifies that VAR can learn a simple linear multivariate relationship
    with exogenous variables.

    System:
    y0[t] = 0.5 * y0[t-1] + 0.2 * y1[t-1] + 0.5 * exog[t] + noise
    y1[t] = 0.2 * y0[t-1] + 0.5 * y1[t-1] - 0.5 * exog[t] + noise
    """
    np.random.seed(42)
    n_samples = 300

    # Generate exogenous variable
    exog = np.random.randn(n_samples)

    # Generate autoregressive targets
    y0 = np.zeros(n_samples)
    y1 = np.zeros(n_samples)

    # Initial values
    y0[0], y1[0] = 0.1, 0.1

    for t in range(1, n_samples):
        y0[t] = 0.8 * y0[t-1] + 0.1 * y1[t-1] + 1.0 * exog[t] + np.random.normal(0, 0.05)
        y1[t] = 0.1 * y0[t-1] + 0.8 * y1[t-1] - 1.0 * exog[t] + np.random.normal(0, 0.05)

    df = pd.DataFrame({
        "y0": y0,
        "y1": y1,
        "exog": exog
    }, index=pd.date_range("2023-01-01", periods=n_samples, freq="D"))

    config = {"datasets": {"quality_test": {}}}
    ds = TimeSeriesDataset(
        "quality_test",
        config,
        num_features=2,
        data=df,
        columns=["y0", "y1"],
        past_covariates=[],
        future_covariates=["exog"]  # Treated as general exog in VAR
    )

    forecast_steps = 5
    ds.split_data(forecast_steps)

    # Configure VAR to use the correct lags
    quality_config = base_var_config.copy()
    quality_config["max_lags"] = 1 # Data is generated with lag 1
    quality_config["preprocessing"] = {
        "preprocessing_groups": [{
            "name": "all",
            "apply_to": ["y0", "y1", "exog"],
            "pipeline": {"scaling": {"enabled": True, "method": "standard"}}
        }]
    }

    model = ModelFactory.create(
        "var",
        "test_var_model",
        run_context=base_context,
        model_params=quality_config,
        num_features=2,
        forecast_steps=forecast_steps,
        window_size=20,
        dataset=ds
    )

    # Fit
    train_exog = ds.development_data[["exog"]]
    train_endog = ds.development_data[["y0", "y1"]]
    model.fit(train_endog, exog_series=train_exog)

    # Predict
    future_exog = ds.test_data[["exog"]]
    preds = model.predict(forecast_steps=forecast_steps, future_exog=future_exog)

    # Evaluate
    actuals = ds.test_data[["y0", "y1"]]
    mse = ((preds - actuals) ** 2).mean().mean()

    print(f"\nVAR Quality Test MSE: {mse:.4f}")

    # Threshold: A good model should have low MSE on this deterministic process
    assert mse < 0.2, f"VAR failed to learn linear system. MSE={mse:.4f}"


def test_var_raises_error_on_univariate_data(
    base_var_config,
    base_context
):
    """VAR should raise an error if initialized with < 2 features."""
    dataset = create_multivariate_dataset(n_targets=1) # Univariate

    # ModelFactory re-raises the ValueError from model init.
    with pytest.raises(ValueError, match="VAR/MAX requires at least 2"):
        ModelFactory.create(
            "var",
            "test_var_model",
            run_context=base_context,
            model_params=base_var_config,
            num_features=1,
            forecast_steps=5,
            window_size=10,
            dataset=dataset
        )
