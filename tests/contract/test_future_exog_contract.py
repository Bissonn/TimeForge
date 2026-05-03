import pandas as pd
import pytest


@pytest.mark.parametrize("model_type", ["lstm", "transformer"])
def test_predict_accepts_future_exog_none(model_factory, mock_dataset, model_type):
    # Create dataset first so we can reference it
    dataset = mock_dataset(n_targets=1, n_enc=0, n_dec=0)
    model = model_factory(model_type=model_type, n_targets=1, n_enc=0, n_dec=0)

    history = dataset.development_data.iloc[-model.window_size:]

    preds = model.predict(history, future_exog=None)

    assert isinstance(preds, pd.DataFrame)
    assert preds.shape[0] == model.forecast_steps


@pytest.mark.parametrize("model_type", ["lstm", "transformer"])
def test_predict_accepts_future_exog_dataframe(model_factory, mock_dataset, model_type):
    # Create dataset with exogenous features
    dataset = mock_dataset(n_targets=1, n_enc=2, n_dec=2)
    model = model_factory(model_type=model_type, n_targets=1, n_enc=2, n_dec=2)

    history = dataset.development_data.iloc[-model.window_size:]

    exog_cols = (
        dataset.past_covariates
        + dataset.future_covariates
    )

    if exog_cols:
        future_exog = dataset.test_data[exog_cols].iloc[:model.forecast_steps]
        assert isinstance(future_exog, pd.DataFrame)
    else:
        future_exog = None

    preds = model.predict(history, future_exog=future_exog)

    assert isinstance(preds, pd.DataFrame)
    assert preds.shape[0] == model.forecast_steps
