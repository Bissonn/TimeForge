import pandas as pd
import pytest


@pytest.mark.parametrize("model_type", ["lstm", "transformer"])
def test_predict_returns_dataframe(model_factory, mock_dataset, model_type):
    # Create dataset with targets only (simplest case)
    dataset = mock_dataset(n_targets=2, n_enc=0, n_dec=0)
    model = model_factory(model_type=model_type, n_targets=2, n_enc=0, n_dec=0)

    # Use dataset fixture directly, not model.dataset
    history = dataset.development_data.iloc[-model.window_size:]

    preds = model.predict(history)

    assert isinstance(preds, pd.DataFrame)
    assert preds.shape[0] == model.forecast_steps
    assert list(preds.columns) == dataset.target_columns
    assert not preds.isna().any().any()
