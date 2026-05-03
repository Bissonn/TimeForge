# test_neural_predict_refactor.py
# NOTE: Consider a more neutral long-term name, e.g.:
# - test_neural_prediction_pipeline.py
# - test_neural_predict_contract.py
# - test_prediction_safety.py

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
import pytest
import torch

from models.base import NeuralTSForecaster


@dataclass
class DummyPreprocessor:
    """
    Minimal preprocessor stub that supports:
    - transform(input_data)
    - inverse_transforms(predictions_np_2d, start_after=...)
    and provides:
    - target_columns
    - _full_raw_data_context (for index inference)
    """
    target_columns: List[str]
    _full_raw_data_context: pd.DataFrame

    def transform(self, df: pd.DataFrame, allow_subset: bool = False) -> pd.DataFrame:
        if df is None or df.empty:
            raise ValueError("DummyPreprocessor.transform got empty input.")
        # Keep only target columns when present, otherwise pass through.
        cols = [c for c in self.target_columns if c in df.columns]
        if cols:
            return df.loc[:, cols].copy()
        if allow_subset:
            return df.copy()
        raise ValueError("Missing target columns in input_data for transform().")

    def inverse_transforms(
        self,
        predictions: Union[np.ndarray, pd.DataFrame],
        start_after: Optional[Union[pd.Timestamp, int]] = None,
    ) -> pd.DataFrame:
        """
        Simplified version compatible with the real contract:
        - Accepts 2D numpy array (H, F) and builds a time index based on context.
        - Returns DataFrame with target_columns.
        """
        if isinstance(predictions, pd.DataFrame):
            return predictions.copy()

        if not isinstance(predictions, np.ndarray):
            raise TypeError(f"Unsupported predictions type: {type(predictions)}")

        if start_after is None:
            raise ValueError("start_after must be provided for numpy predictions")

        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        if predictions.ndim != 2:
            raise ValueError(f"Expected 2D predictions array, got ndim={predictions.ndim}")

        n_rows, n_cols = predictions.shape
        if n_cols != len(self.target_columns):
            raise ValueError(
                f"Predictions have {n_cols} columns but target_columns has {len(self.target_columns)}"
            )

        context_index = self._full_raw_data_context.index

        if isinstance(context_index, pd.DatetimeIndex):
            freq = context_index.freq or pd.infer_freq(context_index)
            if freq is None:
                raise ValueError("Cannot infer frequency from context data.")
            start_date = start_after + pd.tseries.frequencies.to_offset(freq)
            pred_index = pd.date_range(start=start_date, periods=n_rows, freq=freq)
        elif isinstance(context_index, pd.RangeIndex):
            if not isinstance(start_after, int):
                raise TypeError("start_after must be int for RangeIndex context.")
            pred_index = pd.RangeIndex(start=start_after + 1, stop=start_after + 1 + n_rows)
        else:
            raise TypeError(f"Unsupported index type: {type(context_index)}")

        return pd.DataFrame(predictions, index=pred_index, columns=self.target_columns)


class MinimalNeuralForecaster(NeuralTSForecaster):
    """
    Minimal NeuralTSForecaster subclass for testing shared prediction logic.
    It avoids heavy model initialization and provides a controllable _internal_predict.
    """

    def __init__(
        self,
        preprocessor: DummyPreprocessor,
        forecast_steps: int,
        num_features: int,
        *,
        device: Optional[torch.device] = None,
    ) -> None:
        # We intentionally do not call super().__init__ because project base ctor
        # might require many params. We set the minimal fields needed by predict().
        self.preprocessor = preprocessor
        self.forecast_steps = int(forecast_steps)
        self.num_features = int(num_features)
        self.device = device or torch.device("cpu")
        self.fitted = True
        self.model = object()  # Non-None sentinel to satisfy base checks (if any)

        # Controlled output for _internal_predict
        self._next_internal_output: Any = None

        # Capture last kwargs seen by _internal_predict
        self.last_seen_kwargs: dict[str, Any] = {}

    def _inference_context(self):
        return contextlib.nullcontext()

    # ------------------------------------------------------------------
    # Abstract API required by your TSForecaster/NeuralTSForecaster
    # ------------------------------------------------------------------

    def _train_model(self, *args: Any, **kwargs: Any) -> None:
        """
        Required by base ABC in this project.
        Not used in this test module (we test prediction only).
        """
        raise NotImplementedError("Training is not exercised in MinimalNeuralForecaster tests.")

    def set_internal_output(self, out: Any) -> None:
        self._next_internal_output = out

    def _internal_predict(self, input_tensor: torch.Tensor, **kwargs) -> np.ndarray:
        # Store kwargs to assert behavior in tests.
        self.last_seen_kwargs = dict(kwargs)

        out = self._next_internal_output
        if out is None:
            # Default: finite, correct shape in 3D (B=1,H,F) to test squeeze path.
            out = np.zeros((1, self.forecast_steps, self.num_features), dtype=np.float32)
        return np.asarray(out)


# ----------------------------
# Fixtures
# ----------------------------

@pytest.fixture
def dt_context_df() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=10, freq="D")
    return pd.DataFrame({"y": np.arange(10, dtype=np.float32)}, index=idx)


@pytest.fixture
def range_context_df() -> pd.DataFrame:
    idx = pd.RangeIndex(start=0, stop=10)
    return pd.DataFrame({"y": np.arange(10, dtype=np.float32)}, index=idx)


@pytest.fixture
def dt_preprocessor(dt_context_df: pd.DataFrame) -> DummyPreprocessor:
    return DummyPreprocessor(target_columns=["y", "z"], _full_raw_data_context=dt_context_df.assign(z=0.0))


@pytest.fixture
def range_preprocessor(range_context_df: pd.DataFrame) -> DummyPreprocessor:
    return DummyPreprocessor(target_columns=["y", "z"], _full_raw_data_context=range_context_df.assign(z=0.0))


@pytest.fixture
def forecaster_dt(dt_preprocessor: DummyPreprocessor) -> MinimalNeuralForecaster:
    return MinimalNeuralForecaster(dt_preprocessor, forecast_steps=5, num_features=2)


@pytest.fixture
def forecaster_range(range_preprocessor: DummyPreprocessor) -> MinimalNeuralForecaster:
    return MinimalNeuralForecaster(range_preprocessor, forecast_steps=5, num_features=2)


# ----------------------------
# _sanitize_predictions_np tests
# ----------------------------

def test_sanitize_squeezes_3d_batch_to_2d(forecaster_dt: MinimalNeuralForecaster):
    H, F = forecaster_dt.forecast_steps, forecaster_dt.num_features
    pred = np.zeros((1, H, F), dtype=np.float32)

    out = forecaster_dt._sanitize_predictions_np(pred)
    assert out.ndim == 2
    assert out.shape == (H, F)


def test_sanitize_reshapes_1d_to_2d_column_vector(forecaster_dt: MinimalNeuralForecaster):
    H = forecaster_dt.forecast_steps
    pred = np.zeros((H,), dtype=np.float32)  # (H,) -> reshape (H,1)

    out = forecaster_dt._sanitize_predictions_np(pred)
    assert out.ndim == 2
    # Contract is strict: output must be (H, num_features).
    # Since (H,1) mismatches expected (H,2), sanitizer returns NaN array (H,2).
    assert out.shape == (H, forecaster_dt.num_features)
    assert np.isnan(out).all()


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_sanitize_non_finite_returns_nan_array(forecaster_dt: MinimalNeuralForecaster, bad_value: float):
    H, F = forecaster_dt.forecast_steps, forecaster_dt.num_features
    pred = np.zeros((H, F), dtype=np.float32)
    pred[0, 0] = bad_value

    out = forecaster_dt._sanitize_predictions_np(pred)
    assert out.shape == (H, F)
    assert np.isnan(out).all()


def test_sanitize_bad_ndim_returns_nan_array(forecaster_dt: MinimalNeuralForecaster):
    H, F = forecaster_dt.forecast_steps, forecaster_dt.num_features
    pred = np.zeros((1, 1, H, F), dtype=np.float32)

    out = forecaster_dt._sanitize_predictions_np(pred)
    assert out.shape == (H, F)
    assert np.isnan(out).all()


def test_sanitize_horizon_mismatch_returns_nan_array(forecaster_dt: MinimalNeuralForecaster):
    H, F = forecaster_dt.forecast_steps, forecaster_dt.num_features
    pred = np.zeros((H - 1, F), dtype=np.float32)

    out = forecaster_dt._sanitize_predictions_np(pred)
    assert out.shape == (H, F)
    assert np.isnan(out).all()


def test_sanitize_feature_mismatch_returns_nan_array(forecaster_dt: MinimalNeuralForecaster):
    H, F = forecaster_dt.forecast_steps, forecaster_dt.num_features
    pred = np.zeros((H, F + 1), dtype=np.float32)

    out = forecaster_dt._sanitize_predictions_np(pred)
    assert out.shape == (H, F)
    assert np.isnan(out).all()


# ----------------------------
# _fallback_nan_dataframe index tests
# ----------------------------

def test_fallback_nan_dataframe_builds_datetime_index(forecaster_dt: MinimalNeuralForecaster, dt_context_df: pd.DataFrame):
    start_after = dt_context_df.index[-1]
    df = forecaster_dt._fallback_nan_dataframe(start_after=start_after)

    assert isinstance(df.index, pd.DatetimeIndex)
    assert len(df) == forecaster_dt.forecast_steps
    assert list(df.columns) == forecaster_dt.preprocessor.target_columns
    assert df.isna().all().all()

    # Expect start at next tick given daily freq
    expected_start = start_after + pd.tseries.frequencies.to_offset("D")
    assert df.index[0] == expected_start


def test_fallback_nan_dataframe_builds_range_index(forecaster_range: MinimalNeuralForecaster, range_context_df: pd.DataFrame):
    start_after = int(range_context_df.index[-1])
    df = forecaster_range._fallback_nan_dataframe(start_after=start_after)

    assert isinstance(df.index, pd.RangeIndex)
    assert df.index.start == start_after + 1
    assert df.index.stop == start_after + 1 + forecaster_range.forecast_steps
    assert list(df.columns) == forecaster_range.preprocessor.target_columns
    assert df.isna().all().all()


def test_fallback_nan_dataframe_gracefully_falls_back_when_freq_missing(forecaster_dt: MinimalNeuralForecaster):
    # Create irregular datetime index (infer_freq likely None)
    idx = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-04", "2025-01-07"])
    ctx = pd.DataFrame({"y": [1, 2, 3, 4], "z": [0, 0, 0, 0]}, index=idx)
    forecaster_dt.preprocessor._full_raw_data_context = ctx

    start_after = idx[-1]
    df = forecaster_dt._fallback_nan_dataframe(start_after=start_after)

    assert len(df) == forecaster_dt.forecast_steps
    assert list(df.columns) == forecaster_dt.preprocessor.target_columns
    assert df.isna().all().all()

def test_fallback_nan_dataframe_forced_freq_missing_via_monkeypatch(
    forecaster_dt: MinimalNeuralForecaster,
    monkeypatch,
):
    # Force pd.infer_freq -> None deterministically
    idx = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-04", "2025-01-07"])
    ctx = pd.DataFrame({"y": [1, 2, 3, 4], "z": [0, 0, 0, 0]}, index=idx)
    forecaster_dt.preprocessor._full_raw_data_context = ctx

    monkeypatch.setattr(pd, "infer_freq", lambda *_args, **_kwargs: None)

    df = forecaster_dt._fallback_nan_dataframe(start_after=idx[-1])
    assert len(df) == forecaster_dt.forecast_steps
    assert list(df.columns) == forecaster_dt.preprocessor.target_columns
    assert df.isna().all().all()


def test_predict_raises_when_input_missing_target_columns(forecaster_dt: MinimalNeuralForecaster, dt_context_df: pd.DataFrame):
    input_df = pd.DataFrame({"not_y": np.ones(len(dt_context_df.index), dtype=np.float32)},
                            index=dt_context_df.index)
    # Missing target column "y"
    bad_input = dt_context_df[["y"]].copy()
    bad_input = bad_input.rename(columns={"y": "not_y"})
    # Contract: predict should not crash; it returns NaN fallback for divergence classification.
    out = forecaster_dt.predict(bad_input)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (forecaster_dt.forecast_steps, forecaster_dt.num_features)
    assert out.isna().all().all()

# ----------------------------
# predict() integration tests (lightweight)
# ----------------------------

def _make_input_df(index: pd.Index) -> pd.DataFrame:
    # Provide both target columns (y, z)
    return pd.DataFrame({"y": np.ones(len(index), dtype=np.float32),
                         "z": np.ones(len(index), dtype=np.float32)}, index=index)


def test_predict_accepts_future_exog_tensor_kwarg_no_typeerror(forecaster_dt: MinimalNeuralForecaster, dt_context_df: pd.DataFrame):
    input_df = _make_input_df(dt_context_df.index)

    # Provide a dummy future_exog_tensor to ensure kwargs path is exercised.
    fut = torch.zeros((1, forecaster_dt.forecast_steps, 1), dtype=torch.float32)

    # Make internal output valid, finite, and 3D (B=1,H,F) to test sanitize squeeze.
    out = np.zeros((1, forecaster_dt.forecast_steps, forecaster_dt.num_features), dtype=np.float32)
    forecaster_dt.set_internal_output(out)

    pred_df = forecaster_dt.predict(input_df, future_exog=None, future_exog_tensor=fut)

    assert isinstance(pred_df, pd.DataFrame)
    assert pred_df.shape == (forecaster_dt.forecast_steps, forecaster_dt.num_features)
    assert "future_exog_tensor" in forecaster_dt.last_seen_kwargs
    assert isinstance(forecaster_dt.last_seen_kwargs["future_exog_tensor"], torch.Tensor)


def test_predict_non_finite_internal_output_returns_nan_df(forecaster_dt: MinimalNeuralForecaster, dt_context_df: pd.DataFrame):
    input_df = _make_input_df(dt_context_df.index)

    bad = np.zeros((forecaster_dt.forecast_steps, forecaster_dt.num_features), dtype=np.float32)
    bad[0, 0] = np.nan
    forecaster_dt.set_internal_output(bad)

    pred_df = forecaster_dt.predict(input_df)

    assert pred_df.shape == (forecaster_dt.forecast_steps, forecaster_dt.num_features)
    assert pred_df.isna().any().any()


def test_predict_inverse_transforms_exception_returns_fallback_with_index(forecaster_dt: MinimalNeuralForecaster, dt_context_df: pd.DataFrame, monkeypatch):
    input_df = _make_input_df(dt_context_df.index)

    good = np.zeros((forecaster_dt.forecast_steps, forecaster_dt.num_features), dtype=np.float32)
    forecaster_dt.set_internal_output(good)

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(forecaster_dt.preprocessor, "inverse_transforms", boom)

    pred_df = forecaster_dt.predict(input_df)

    assert pred_df.shape == (forecaster_dt.forecast_steps, forecaster_dt.num_features)
    assert pred_df.isna().all().all()
    # Should preserve temporal index when possible
    assert isinstance(pred_df.index, pd.DatetimeIndex)
