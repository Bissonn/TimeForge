"""Regression tests for VAR model."""

import pytest
import pandas as pd
import numpy as np

from models.factory import ModelFactory
from utils.dataset import TimeSeriesDataset
import models.var  # ensure VAR model is registered

pytestmark = pytest.mark.integration


@pytest.fixture
def var_base_config():
    """
    Minimal sensible config for VARForecaster based on VARMAX.
    No exog params.
    """
    return {
        "max_lags": 2,
        "preprocessing": {
            "preprocessing_groups": [
                {
                    "name": "default",
                    "apply_to": "__targets__",
                    "pipeline": {
                        "scaling": {"enabled": True},
                    },
                }
            ]
        },
    }


@pytest.fixture
def var_dataset() -> TimeSeriesDataset:
    """Bivariate endogenous + single exog regresor."""
    n = 120
    idx = pd.date_range("2021-01-01", periods=n, freq="D")

    # two correlated exog series
    base = np.sin(2 * np.pi * np.arange(n) / 30.0)
    y1 = 10 + base + np.random.normal(scale=0.2, size=n)
    y2 = 5 + 0.5 * base + np.random.normal(scale=0.2, size=n)

    # simple exog regresor
    temp = 15 + 10 * np.sin(2 * np.pi * np.arange(n) / 365.0) + np.random.normal(
        scale=1.0, size=n
    )

    df = pd.DataFrame(
        {
            "y1": y1,
            "y2": y2,
            "temp": temp,
        },
        index=idx,
    )

    ds = TimeSeriesDataset(
        dataset_name="var_ds",
        config={},
        num_features=2,
        data=df,
        columns=["y1", "y2"],          # 2 exog vars
        past_covariates=["temp"], # exog regresor
        future_covariates=[],
    )
    return ds


def _split_dataset(ds: TimeSeriesDataset, forecast_steps: int):
    ds.split_data(forecast_steps)
    return ds.development_data, ds.test_data


def test_var_fit_predict_no_exog_smoke(
        var_base_config,
        var_dataset,
        base_context
):
    """
    Regression smoke test:
    - VAR without exog should train and infere without any errors.
    - Length of output, columns and continuity must be preserved.
    """
    H = 5
    ds = var_dataset
    train_df, test_df = _split_dataset(ds, H)

    config = var_base_config.copy()  # here we dont use exog

    model = ModelFactory.create(
        "var",
        model_name="var_regression_no_exog",
        run_context=base_context,
        model_params=config,
        num_features=2,
        forecast_steps=H,
        window_size=20,
        dataset=ds
    )

    target_cols = ds.target_columns
    train_endog = train_df[target_cols]

    # Fit only on endog
    model.fit(train_endog)

    # H-step prediction without exog
    preds = model.predict(forecast_steps=H)

    assert isinstance(preds, pd.DataFrame)
    assert list(preds.columns) == target_cols
    assert len(preds) == H
    assert not preds.isna().any().any()

    # Continuity of time index
    step = preds.index[1] - preds.index[0]
    assert preds.index[0] == train_df.index[-1] + step


def test_var_fit_predict_with_exog_regression(
        var_base_config,
        var_dataset,
        base_context
):
    """
    Regression test:
    - VAR trained with exogenous.
    - Prediction with properly aligned future_exog must pass without any errors and with perfect shape.
    """
    H = 7
    ds = var_dataset
    train_df, test_df = _split_dataset(ds, H)

    config = var_base_config.copy()

    model = ModelFactory.create(
        "var",
        model_name="var_regression_exog",
        run_context=base_context,
        model_params=config,
        num_features=2,
        forecast_steps=H,
        window_size=20,
        dataset=ds
    )

    target_cols = ds.target_columns
    exog_cols = ds.past_covariates + ds.future_covariates

    train_endog = train_df[target_cols]
    train_exog = train_df[exog_cols]

    # Fit with exog regressor
    model.fit(train_endog, exog_series=train_exog)

    # Future exog for horizont H
    future_exog = test_df[exog_cols]

    preds = model.predict(future_exog=future_exog, forecast_steps=H)

    assert isinstance(preds, pd.DataFrame)
    assert list(preds.columns) == target_cols
    assert len(preds) == H
    assert not preds.isna().any().any()

    # Continuity of time index
    step = preds.index[1] - preds.index[0]
    assert preds.index[0] == train_df.index[-1] + step
