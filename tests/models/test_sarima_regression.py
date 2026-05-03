"""
Regression tests for SARIMA models.
"""
import pytest
import pandas as pd
import numpy as np

from models.factory import ModelFactory
from utils.dataset import TimeSeriesDataset
import models.sarima  # noqa: F401

pytestmark = pytest.mark.integration


@pytest.fixture
def sarima_base_config():
    return {
        "p": 1, "d": 0, "q": 0,
        "P": 1, "D": 0, "Q": 0, "seasonal_period": 7,
        "preprocessing": {
            "preprocessing_groups": [{
                "name": "default",
                "apply_to": "__targets__",
                "pipeline": {"scaling": {"enabled": True}},
            }]
        },
    }


@pytest.fixture
def sarima_dataset() -> TimeSeriesDataset:
    """Seasonal target + one exogenous regressor."""
    n = 120
    idx = pd.date_range("2021-01-01", periods=n, freq="D")
    seasonal = np.sin(2 * np.pi * np.arange(n) / 7.0)

    df = pd.DataFrame(
        {
            "y": 10 + seasonal + np.random.normal(scale=0.2, size=n),
            "promo": np.random.randint(0, 2, size=n),
        },
        index=idx,
    )

    ds = TimeSeriesDataset(
        dataset_name="sarima_ds",
        config={},
        num_features=1,
        data=df,
        columns=["y"],
        past_covariates=["promo"],
        future_covariates=[],
    )
    return ds


def _split_dataset(ds: TimeSeriesDataset, forecast_steps: int):
    ds.split_data(forecast_steps)
    return ds.development_data, ds.test_data


def test_sarima_fit_predict_no_exog_smoke(sarima_base_config, sarima_dataset, base_context):
    """
    Regression smoke test:
    - SARIMA without exogenous vars should fit and predict without crashing.
    - Output length and index continuity must hold.
    """
    H = 5
    ds = sarima_dataset
    train_df, test_df = _split_dataset(ds, H)

    # Disable exog
    config = {**sarima_base_config, "use_exogenous": False}

    model = ModelFactory.create(
        "sarima",
        model_name="sarima_regression_no_exog",
        run_context=base_context,
        model_params=config,
        num_features=1,
        forecast_steps=H,
        window_size=20,
        dataset=ds,
    )

    target_cols = ds.target_columns
    train_endog = train_df[target_cols]

    # Statsmodels-style fit: we pass only endog; exog=None
    model.fit(train_endog)

    # ARIMA/SARIMA-style predict: history jest w model_fit, nie przekazujemy okna
    preds = model.predict(forecast_steps=H)

    assert isinstance(preds, pd.DataFrame)
    assert len(preds) == H
    assert not preds.isna().any().any()
    # Index powinien zaczynać się dzień po ostatnim punkcie trenowania
    assert preds.index[0] == train_df.index[-1] + pd.Timedelta(days=1)


def test_sarima_fit_predict_with_exog_regression(sarima_base_config, sarima_dataset, base_context):
    """
    Regression test:
    - SARIMA fitted with exogenous variables.
    - Prediction with correctly aligned future_exog must work and produce stable output shape.
    """
    H = 7
    ds = sarima_dataset
    train_df, test_df = _split_dataset(ds, H)

    # Enable exog
    config = {**sarima_base_config, "use_exogenous": True}

    model = ModelFactory.create(
        "sarima",
        model_name="sarima_regression_exog",
        run_context=base_context,
        model_params=config,
        num_features=1,
        forecast_steps=H,
        window_size=20,
        dataset=ds,
    )

    target_cols = ds.target_columns
    exog_cols = ds.past_covariates + ds.future_covariates

    train_endog = train_df[target_cols]
    train_exog = train_df[exog_cols]

    # Fit z exog
    model.fit(train_endog, exog_series=train_exog)

    # future exog dla H-krokowej prognozy
    future_exog = test_df[exog_cols]

    # Uwaga: NIE przekazujemy history window – tylko future_exog + forecast_steps
    preds = model.predict(future_exog=future_exog, forecast_steps=H)

    assert isinstance(preds, pd.DataFrame)
    assert len(preds) == H
    assert not preds.isna().any().any()
    assert preds.index[0] == train_df.index[-1] + pd.Timedelta(days=1)
