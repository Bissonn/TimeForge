# tests/models/test_varmax.py

import pandas as pd
import pytest
from pathlib import Path
import numpy as np
from unittest.mock import MagicMock
from utils.dataset import TimeSeriesDataset

# Assuming your project root is correctly identified
ROOT = Path(__file__).resolve().parents[2]


# --- Helper Functions (with FutureWarning fix) ---

def _make_mv_series(n=100, k=2):
    """Creates a multivariate time series DataFrame."""
    idx = pd.date_range("2020-01-01", periods=n, freq="h")  # 'H' -> 'h'
    data = np.random.randn(n, k)
    return pd.DataFrame(data, index=idx, columns=[f"var{i}" for i in range(k)])


def _make_exog(n=100, m=1):
    """Creates an exogenous variables DataFrame."""
    idx = pd.date_range("2020-01-01", periods=n, freq="h")  # 'H' -> 'h'
    data = np.random.randn(n, m)
    return pd.DataFrame(data, index=idx, columns=[f"exog{i}" for i in range(m)])


# --- Test Case (with TypeError fix) ---

def test_varmax_fit_predict_with_exog_smoke(base_context):
    import importlib
    # This dynamic import logic is complex but assumed correct for your setup.
    mod_path = ROOT / "var_patched.py"
    module_name = "var_patched" if mod_path.exists() else "models.var"

    try:
        var = importlib.import_module(module_name)
    except ImportError:
        pytest.skip(f"Could not import module {module_name}")

    assert hasattr(var, "VARForecaster"), "VARForecaster class not found"

    endog = _make_mv_series(n=160, k=2)
    exog = _make_exog(n=160, m=1)

    mock_ds = MagicMock(spec=TimeSeriesDataset)
    mock_ds.target_columns = ['var0', 'var1']  # Consistent with k=2
    mock_ds.past_covariates = ['exog0']  # Consistent with m=1
    mock_ds.future_covariates = []
    # New API
    mock_ds.past_covariates = ['exog0']
    mock_ds.future_covariates = []
    mock_ds.columns = mock_ds.target_columns + mock_ds.past_covariates + mock_ds.future_covariates

    model = var.VARForecaster(
        model_params={
            "max_lags": 2,
            "maxiter": 50,
            "preprocessing": {"scaler": "standard"}
        },
        num_features=2,
        forecast_steps=10,
        window_size=30,
        dataset=mock_ds,
        run_context=base_context
    )

    # The rest of the test can now proceed
    model.fit(endog, exog_series=exog)
    assert model.fitted

    future_exog = _make_exog(n=10, m=1)
    predictions = model.predict(future_exog=future_exog, forecast_steps=10)

    assert isinstance(predictions, pd.DataFrame)
    assert predictions.shape == (10, 2)
    assert not predictions.isnull().values.any()
