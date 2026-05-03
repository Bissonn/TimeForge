"""
End-to-End tests for ARIMA and SARIMA models.

This module verifies the integration of ARIMA/SARIMA models, focusing on:
1. Correct handling of exogenous variables (merging encoder/decoder exog).
2. Forecast accuracy on synthetic trends and seasonality.
3. Input validation constraints (univariate only).
"""
import pytest
import pandas as pd
import numpy as np
from models.factory import ModelFactory
from utils.dataset import TimeSeriesDataset
import models.arima
import models.sarima

pytestmark = pytest.mark.integration


@pytest.fixture
def base_config():
    """Base configuration for ARIMA/SARIMA models."""
    return {
        "preprocessing": {
            "preprocessing_groups": [{
                "name": "default",
                "apply_to": "__targets__",
                "pipeline": {"scaling": {"enabled": True, "method": "standard"}}
            }]
        }
    }


def create_univariate_dataset(n_samples=100, n_enc_exog=0, n_dec_exog=0, trend=False, seasonal=False):
    """
    Creates a synthetic univariate dataset.
    Optionally adds linear trend or seasonality to the target.
    """
    np.random.seed(42)
    data = {}
    date_range = pd.date_range(start="2023-01-01", periods=n_samples, freq="D")

    t = np.linspace(0, 10, n_samples)

    # Generate Target
    y = np.random.normal(0, 0.1, n_samples)  # Noise base
    if trend:
        y += t  # Linear trend
    if seasonal:
        y += np.sin(t * 2 * np.pi / 2.5)  # Seasonality

    data["target"] = y

    # Generate Exogenous
    enc_cols = []
    for i in range(n_enc_exog):
        col = f"enc_{i}"
        data[col] = np.random.randn(n_samples)
        enc_cols.append(col)

    dec_cols = []
    for i in range(n_dec_exog):
        col = f"dec_{i}"
        data[col] = np.random.randn(n_samples)
        dec_cols.append(col)

    df = pd.DataFrame(data, index=date_range)

    config = {"datasets": {"arima_mock": {}}}

    ds = TimeSeriesDataset(
        "arima_mock", config,
        num_features=1,
        data=df,
        columns=["target"],
        past_covariates=enc_cols,
        future_covariates=dec_cols
    )
    return ds


@pytest.mark.parametrize("model_name", ["arima", "sarima"])
@pytest.mark.parametrize("n_enc_exog", [0, 1])
@pytest.mark.parametrize("n_dec_exog", [0, 1])
def test_arima_sarima_lifecycle_exog_combinations(base_config, model_name, n_enc_exog, n_dec_exog, base_context):
    """
    Smoke Test: Verifies fit/predict lifecycle with various exog configurations.
    Ensures that both encoder and decoder exogenous variables are correctly passed
    to the underlying statsmodels implementation.
    """
    forecast_steps = 5
    window_size = 20

    dataset = create_univariate_dataset(n_enc_exog=n_enc_exog, n_dec_exog=n_dec_exog)
    dataset.split_data(forecast_steps)

    # Setup model params
    params = base_config.copy()
    if model_name == "arima":
        params.update({"p": 1, "d": 0, "q": 0})
    else:
        params.update({"p": 1, "d": 0, "q": 0, "P": 0, "D": 0, "Q": 0, "seasonal_period": 4})

    model = ModelFactory.create(
        model_name,
        "test_arima_sarima_model",
        run_context=base_context,
        model_params=params,
        num_features=1,
        forecast_steps=forecast_steps,
        window_size=window_size,
        dataset=dataset
    )

    # Prepare data manually (simulating Trainer logic)
    train_df = dataset.development_data
    target_cols = dataset.target_columns
    exog_cols = dataset.past_covariates + dataset.future_covariates

    train_endog = train_df[target_cols]
    train_exog = train_df[exog_cols] if exog_cols else None

    # Fit
    model.fit(train_endog, exog_series=train_exog)
    assert model.fitted

    # Predict
    future_exog = dataset.test_data[exog_cols] if exog_cols else None
    preds = model.predict(forecast_steps=forecast_steps, future_exog=future_exog)

    assert isinstance(preds, pd.DataFrame)
    assert len(preds) == forecast_steps
    assert not preds.isna().any().any()
    assert list(preds.columns) == target_cols


def test_arima_quality_linear_trend(base_config, base_context):
    """
    Quality Test: ARIMA(1,1,0) should perfectly forecast a linear trend.
    d=1 handles the trend (differencing makes it constant), AR(1) handles dynamics.
    """
    dataset = create_univariate_dataset(n_samples=100, trend=True)  # Linear trend y=t
    dataset.split_data(forecast_steps=5)

    params = base_config.copy()
    # ARIMA(1,1,0) is good for linear trend
    params.update({"p": 1, "d": 1, "q": 0})

    model = ModelFactory.create(
        "arima",
        "test_arima_model",
        run_context=base_context,
        model_params=params,
        num_features=1,
        forecast_steps=5,
        window_size=20,
        dataset=dataset
    )

    model.fit(dataset.development_data[["target"]])
    preds = model.predict(forecast_steps=5)

    actuals = dataset.test_data[["target"]]
    mse = ((preds - actuals) ** 2).mean().item()

    print(f"\nARIMA Trend MSE: {mse:.4f}")
    # Since data is y=t + small noise, MSE should be very low
    assert mse < 0.5


def test_sarima_quality_seasonality(base_config, base_context):
    """
    Quality Test: SARIMA should capture seasonality.
    Data is a sine wave with period ~12.
    """
    n_samples = 200
    # Create seasonal data with small noise (pure sine destabilizes MLE variance estimator)
    np.random.seed(42)
    t = np.arange(n_samples)
    y = np.sin(t * 2 * np.pi / 12) + np.random.normal(0, 0.05, n_samples)

    df = pd.DataFrame({"target": y}, index=pd.date_range("2020-01-01", periods=n_samples, freq="D"))
    config = {"datasets": {"seasonal": {}}}
    dataset = TimeSeriesDataset("seasonal", config, num_features=1, data=df, columns=["target"])
    dataset.split_data(forecast_steps=12)

    params = base_config.copy()
    # Classic "airline model" SARIMA(0,0,0)(1,1,1,12) — Q=1 gives seasonal MA
    # needed for MLE convergence after seasonal differencing
    params.update({
        "p": 0, "d": 0, "q": 0,
        "P": 1, "D": 1, "Q": 1, "seasonal_period": 12
    })

    model = ModelFactory.create(
        "sarima",
        "test_sarima_model",
        run_context=base_context,
        model_params=params,
        num_features=1,
        forecast_steps=12,
        window_size=30,
        dataset=dataset
    )

    model.fit(dataset.development_data[["target"]])
    preds = model.predict(forecast_steps=12)

    actuals = dataset.test_data[["target"]]
    mse = ((preds - actuals) ** 2).mean().item()

    print(f"\nSARIMA Seasonality MSE: {mse:.4f}")
    # Tolerance for seasonal forecast
    assert mse < 0.5


def test_arima_exog_dependency(base_config, base_context):
    """
    Quality Test: Verify ARIMA uses exogenous variables.
    Target is defined as: y = 2.0 * exog + noise.
    ARIMA with exog should learn this relationship perfectly.
    """
    n_samples = 100
    np.random.seed(42)
    exog = np.random.randn(n_samples)
    target = 2.0 * exog + np.random.normal(0, 0.1, n_samples)  # Strong dependency + small noise for MLE stability

    df = pd.DataFrame({"target": target, "exog": exog},
                      index=pd.date_range("2023-01-01", periods=n_samples))

    config = {"datasets": {"exog_test": {}}}
    dataset = TimeSeriesDataset(
        "exog_test",
        config,
        num_features=1,
        data=df,
        columns=["target"], future_covariates=["exog"]
    )
    dataset.split_data(forecast_steps=5)

    params = base_config.copy()
    # Simple AR model is enough, the regression on exog happens inside statsmodels
    params.update({"p": 1, "d": 0, "q": 0})

    model = ModelFactory.create(
        "arima",
        "test_arima_model",
        run_context=base_context,
        model_params=params,
        num_features=1,
        forecast_steps=5,
        window_size=20,
        dataset=dataset
    )

    train_df = dataset.development_data
    model.fit(train_df[["target"]], exog_series=train_df[["exog"]])

    future_exog = dataset.test_data[["exog"]]
    preds = model.predict(forecast_steps=5, future_exog=future_exog)

    actuals = dataset.test_data[["target"]]
    mse = ((preds - actuals) ** 2).mean().item()

    print(f"\nARIMA Exog Dependency MSE: {mse:.4f}")
    # Without exog, prediction would be ~0 (mean), so MSE ~ 4.0.
    # With exog, MSE should be ~0.
    assert mse < 0.1


def test_arima_raises_error_on_multivariate(base_config, base_context):
    """ARIMA/SARIMA should reject datasets with >1 target variables."""
    df = pd.DataFrame({"y1": [1] * 50, "y2": [2] * 50}, index=pd.date_range("2020-01-01", periods=50))
    config = {"datasets": {"multi": {}}}
    dataset = TimeSeriesDataset("multi", config, num_features=2,data=df, columns=["y1", "y2"])

    # Check ARIMA
    with pytest.raises(ValueError, match="Univariate ARIMA/SARIMA models require num_features=1"):
        ModelFactory.create(
            "arima",
            "test_arima_model",
            run_context=base_context,
            model_params={"p": 1, "d": 0, "q": 0},
            num_features=2,
            forecast_steps=5,
            window_size=10,
            dataset=dataset
        )

    # Check SARIMA
    with pytest.raises(ValueError, match="Univariate ARIMA/SARIMA models require num_features=1"):
        ModelFactory.create(
            "sarima",
            "test_sarima_model",
            run_context=base_context,
            model_params={"p": 1, "d": 0, "q": 0, "seasonal_period": 12},
            num_features=2,
            forecast_steps=5,
            window_size=10,
            dataset=dataset
        )
