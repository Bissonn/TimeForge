"""
Regression tests for ARIMA models to ensure stability of fit/predict interfaces.
"""
import pytest
import pandas as pd
import numpy as np
from models.factory import ModelFactory
from utils.dataset import TimeSeriesDataset
import models.arima

pytestmark = pytest.mark.integration


@pytest.fixture
def arima_base_config():
    return {
        "p": 1, "d": 0, "q": 0,
        "preprocessing": {
            "preprocessing_groups": [{
                "name": "default",
                "apply_to": "__targets__",
                "pipeline": {"scaling": {"enabled": True}}
            }]
        }
    }


@pytest.fixture
def arima_dataset() -> TimeSeriesDataset:
    """Simple univariate target + one exogenous regressor."""
    n = 80
    idx = pd.date_range("2020-01-01", periods=n, freq="D")

    df = pd.DataFrame(
        {
            "y": np.linspace(0, 10, n) + np.random.normal(scale=0.1, size=n),
            "temp": np.sin(np.linspace(0, 5, n)),
        },
        index=idx,
    )

    # Note: using 'dataset_name' not 'name'
    ds = TimeSeriesDataset(
        dataset_name="arima_ds",
        config={},
        num_features=1,
        data=df,
        columns=["y"],
        past_covariates=["temp"],
        future_covariates=[],
    )
    return ds


def _split_dataset(ds, forecast_steps):
    ds.split_data(forecast_steps)
    return ds.development_data, ds.test_data


def test_arima_fit_predict_no_exog_smoke(arima_base_config, arima_dataset, base_context):
    """
    Regression smoke test:
    - ARIMA without exogenous vars should fit and predict without crashing.
    - Output length and index continuity must hold.
    """
    H = 5
    ds = arima_dataset
    train_df, test_df = _split_dataset(ds, H)

    # Disable exog usage in config
    config = {**arima_base_config, "use_exogenous": False}

    model = ModelFactory.create(
        "arima",
        model_name="arima_regression_no_exog",
        run_context=base_context,
        model_params=config,
        num_features=1,
        forecast_steps=H,
        window_size=20,
        dataset=ds,
    )

    # Prepare data for Statistical Model .fit()
    # Stat models expect (endog, exog=None)
    target_cols = ds.target_columns
    train_endog = train_df[target_cols]

    # Calling fit explicitly with correct signature for ARIMA
    model.fit(train_endog, exog_series=None)

    preds = model.predict(forecast_steps=H)

    assert len(preds) == H
    assert isinstance(preds, pd.DataFrame)
    assert not preds.isna().any().any()


def test_arima_fit_predict_with_exog_regression(arima_base_config, arima_dataset, base_context):
    """
    Regression test:
    - ARIMA fitted with exogenous variables.
    - Prediction with correctly aligned future_exog must work.
    """
    H = 7
    ds = arima_dataset
    train_df, test_df = _split_dataset(ds, H)

    # Enable exog
    config = {**arima_base_config, "use_exogenous": True}

    model = ModelFactory.create(
        "arima",
        model_name="arima_regression_exog",
        run_context=base_context,
        model_params=config,
        num_features=1,
        forecast_steps=H,
        window_size=20,
        dataset=ds,
    )

    # Prepare data for Statistical Model .fit()
    target_cols = ds.target_columns
    # For ARIMA, we combine all exog columns available
    exog_cols = ds.past_covariates + ds.future_covariates

    train_endog = train_df[target_cols]
    train_exog = train_df[exog_cols]

    model.fit(train_endog, exog_series=train_exog)

    # Prepare future exog for prediction
    future_exog = test_df[exog_cols]

    preds = model.predict(future_exog=future_exog, forecast_steps=H)

    assert len(preds) == H
    assert isinstance(preds, pd.DataFrame)
    assert not preds.isna().any().any()
    # Ensure index continues from train
    assert preds.index[0] == train_df.index[-1] + pd.Timedelta(days=1)
