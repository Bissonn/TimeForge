"""
Unit tests for the abstract base forecaster classes.

This module provides a comprehensive suite of tests for the `TSForecaster`,
`StatTSForecaster`, and `NeuralTSForecaster` abstract base classes from
`models/base.py`. The tests verify the core, non-abstract logic of these
classes, including initialization, evaluation, data preparation (with full
support for exogenous features), and the hyperparameter optimization loop.

To test the abstract classes, simple concrete subclasses are defined locally.
All external dependencies, such as the dataset and preprocessor, are mocked
to ensure the tests are isolated and deterministic.
"""

import os
import pytest
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import MagicMock, call, patch
from models.base import TSForecaster, StatTSForecaster, NeuralTSForecaster
from models.model_registry import model_registry
from utils.preprocessor import Preprocessor
from utils.dataset import TimeSeriesDataset

pytestmark = pytest.mark.unit

# --- Mocks for Dependencies ---

@pytest.fixture
def mock_preprocessor_class(mocker):
    """Mocks the Preprocessor class to intercept its instantiation and methods."""
    mock_instance = MagicMock()
    # The fit_transform method should return a usable DataFrame, dropping NaNs as the original would.
    mock_instance.fit_transform.side_effect = lambda data: data.dropna()
    preprocessor_class_mock = mocker.patch('models.base.Preprocessor', return_value=mock_instance)
    return preprocessor_class_mock

@pytest.fixture(autouse=True)
def register_mock_models():
    """
    Temporarily registers the concrete test forecasters in the model registry
    for the duration of the tests in this module. This allows ModelFactory to
    instantiate them during the save/load cycle.
    """
    if 'ConcreteStatForecaster' not in model_registry: model_registry['ConcreteStatForecaster'] = ConcreteStatForecaster
    if 'ConcreteNeuralForecaster' not in model_registry: model_registry['ConcreteNeuralForecaster'] = ConcreteNeuralForecaster

    # This part is crucial for teardown
    yield

    # Clean up the registry after tests are done
    if 'ConcreteStatForecaster' in model_registry: del model_registry['ConcreteStatForecaster']
    if 'ConcreteNeuralForecaster' in model_registry: del model_registry['ConcreteNeuralForecaster']
# --- Concrete Subclasses for Testing ---

class ConcreteStatForecaster(StatTSForecaster):
    """A concrete implementation of StatTSForecaster for testing."""
    model_name = "ConcreteStatForecaster"
    def fit(self, train_series: pd.DataFrame, exog_series: pd.DataFrame = None, **kwargs):
        self.fitted = True
        self.model = MagicMock()
        return 0.0, {}  # Unified signature: return (val_loss, training_history)

    def predict(self, *args, **kwargs):
        return pd.DataFrame(np.random.rand(self.forecast_steps, self.num_features))

    def _fit_and_evaluate_fold(self, *args, **kwargs): return 0.5
    def generate_predictions(self, dataset): return self.predict()
    def get_valid_params(self): return {"p"}

class ConcreteNeuralForecaster(NeuralTSForecaster):
    """A concrete implementation of NeuralTSForecaster for testing."""
    model_name = "ConcreteNeuralForecaster"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize a real, simple nn.Module. This is crucial for save/load tests
        # and safe for other tests, which do not depend on self.model being a mock.

        if self.feature_layout.encoder_input_size > 0:
            self.model = torch.nn.Linear(self.feature_layout.encoder_input_size,
                                         self.num_features * self._get_y_window_steps())
        else:
            self.model = None
        self.preprocessor = None  # Preprocessor is set by the fit method in a real scenario

    def _has_decoder(self) -> bool:
        return True

    def _get_y_window_steps(self) -> int:
        if self.model_params.get("strategy") == "iterative":
            return 1
        return self.forecast_steps

    def _train_model(self, *args, **kwargs):
        # In a real scenario, this returns the trained model. For tests, we can just
        # return the existing model instance with a dummy loss attribute.
        setattr(self.model, "best_val_loss", 0.123)
        return self.model

    def fit(self, *args, **kwargs):
        if kwargs.get('dataset') is None:
            kwargs['dataset'] = self.dataset_context

        self._train_model = MagicMock(return_value=self.model)

        if 'train_series' in kwargs and 'dataset' in kwargs:
            try:
                super().fit(*args, **kwargs)
            except Exception as e:
                if not hasattr(self, 'preprocessor') or self.preprocessor is None:
                    self.preprocessor = Preprocessor(
                        self.model_params.get("preprocessing", {}),
                        target_columns = kwargs['dataset'].target_columns,
                        exog_columns = kwargs['dataset'].past_covariates + kwargs['dataset'].future_covariates
                    )
            self.fitted = True
        else:
            self.fitted = True

        return 0.0, {}  # Unified signature: return (val_loss, training_history)

    def predict(self, *args, **kwargs):
        dummy_output = torch.randn(1, self.forecast_steps, self.num_features)
        return pd.DataFrame(dummy_output.squeeze(0).numpy())

    def _fit_and_evaluate_fold(self, *args, **kwargs): return 0.5
    def generate_predictions(self, dataset): return self.predict()
    def get_valid_params(self): return {"hidden_size"}

# --- TSForecaster Tests ---

def test_tsforecaster_initialization(mock_dataset_simple, base_context):
    """
    Scenario: A forecaster is initialized with valid parameters.
    Assumptions: The __init__ method should correctly assign attributes. The `preprocessor`
                 attribute should be None upon initialization, as it can only be created
                 when the data context (columns) is known during the `fit` stage.
    """
    params = {'p': 1, 'preprocessing': {'scaling': {'enabled': True}}}
    forecaster = ConcreteStatForecaster(
        params,
        num_features=2,
        forecast_steps=10,
        window_size=10,
        dataset=mock_dataset_simple,
        run_context=base_context
    )

    assert forecaster.num_features == 2
    assert forecaster.forecast_steps == 10
    assert forecaster.window_size == 10
    assert forecaster.model_params == params
    assert forecaster.preprocessor is None

def test_tsforecaster_initialization_with_exog(base_context):
    """
    Scenario: A forecaster is initialized for a model that will use exogenous data.
    Assumptions: The `feature_layout` attribute should correctly calculate
                 sizes based on the provided dataset.
    """
    mock_ds = MagicMock(spec=TimeSeriesDataset)
    mock_ds.target_columns = ['target1', 'target2']  # 2 targets

    # New API: exog1 is encoder-only (past_covariate)
    #          exog2 is decoder-only (future_covariate)
    mock_ds.past_covariates = ['exog1']  # 1 encoder-only feature
    mock_ds.future_covariates = ['exog2']  # 1 decoder-only feature
    mock_ds.columns = mock_ds.target_columns + mock_ds.past_covariates + mock_ds.future_covariates

    params = {}
    forecaster = ConcreteNeuralForecaster(
        params,
        num_features=2,
        forecast_steps=10,
        window_size=10,
        dataset=mock_ds,
        run_context=base_context
    )

    # encoder_input_size = targets (2) + past_cov (1) + future_cov (1) = 4
    assert forecaster.feature_layout.encoder_input_size == 4
    # decoder_input_size = targets (2) + future_cov (1) = 3 (if model supports decoder)
    # total_features = 2 + 1 + 1 = 4
    assert forecaster.feature_layout.total_features == 4

# --- StatTSForecaster Tests ---

def test_stat_prepare_data_univariate_with_exog(mock_dataset_with_exog, base_context):
    """
    Verifies that `prepare_data` for a univariate model correctly creates
    sub-datasets, each containing one target and ALL original exogenous variables.
    """
    mock_dataset_with_exog.target_columns = ['target_1', 'target_2']
    mock_dataset_with_exog.columns = mock_dataset_with_exog.target_columns + mock_dataset_with_exog.past_covariates + mock_dataset_with_exog.future_covariates
    all_cols = sorted(
        mock_dataset_with_exog.target_columns + mock_dataset_with_exog.past_covariates + mock_dataset_with_exog.future_covariates)
    mock_dataset_with_exog.series = pd.DataFrame(np.random.rand(120, len(all_cols)), columns=all_cols)
    mock_dataset_with_exog.development_data = mock_dataset_with_exog.series
    mock_dataset_with_exog.test_data = mock_dataset_with_exog.series

    forecaster = ConcreteStatForecaster(
        {},
        num_features=2,
        forecast_steps=10,
        window_size=10,
        dataset=mock_dataset_with_exog,
        run_context=base_context
    )
    forecaster.is_univariate = True

    with pytest.MonkeyPatch.context() as m:
        mock_ts_dataset_class = MagicMock()
        m.setattr("models.base.TimeSeriesDataset", mock_ts_dataset_class)
        datasets = forecaster.prepare_data(mock_dataset_with_exog)

    assert len(datasets) == 2

    kwargs_1 = mock_ts_dataset_class.call_args_list[0].kwargs
    assert kwargs_1['columns'] == ['target_1']
    assert kwargs_1['past_covariates'] == ['enc_exog_1']
    assert kwargs_1['future_covariates'] == ['dec_exog_1']

    kwargs_2 = mock_ts_dataset_class.call_args_list[1].kwargs
    assert kwargs_2['columns'] == ['target_2']
    assert kwargs_2['past_covariates'] == ['enc_exog_1']
    assert kwargs_2['future_covariates'] == ['dec_exog_1']

def test_stat_prepare_data_multivariate(mock_dataset_with_exog, base_context):
    """
    Scenario: `prepare_data` is called on a multivariate statistical model.
    Assumptions: The method should return the original dataset unmodified, wrapped in a list.
    """
    forecaster = ConcreteStatForecaster(
        {},
        num_features=2,
        forecast_steps=10,
        window_size=10,
        dataset=mock_dataset_with_exog,
        run_context=base_context
    )
    forecaster.is_univariate = False
    datasets = forecaster.prepare_data(mock_dataset_with_exog)

    assert len(datasets) == 1
    assert datasets[0] == mock_dataset_with_exog

# --- NeuralTSForecaster Tests ---

def test_neural_fit_instantiates_preprocessor_correctly(
        mocker,
        mock_dataset_with_exog,
        mock_preprocessor_class,
        base_context
):
    """
    Verifies that the `fit` method correctly instantiates the Preprocessor
    with a combined list of all exogenous columns.
    """
    params = { 'preprocessing': {'scaling': True}}
    window_size = 10
    forecaster = ConcreteNeuralForecaster(
        params,
        num_features=1,
        forecast_steps=5,
        window_size=window_size,
        dataset=mock_dataset_with_exog,
        run_contex=base_context,
        run_context=base_context
    )

    mocker.patch('models.base.create_sliding_window',
                 return_value=(np.ones((50, 10, 3)), np.ones((50, 5, 1)), np.ones((50, 5, 1))))
    forecaster.fit(
        train_series=mock_dataset_with_exog.development_data,
        is_final_fit=True,
        dataset=mock_dataset_with_exog
    )

    mock_preprocessor_class.assert_called_once_with(
        {'scaling': True},
        target_columns=['target_1'],
        exog_columns=['enc_exog_1', 'dec_exog_1']
    )

def test_neural_fit_calculates_target_indices_correctly(
        mocker,
        mock_dataset_with_exog,
        mock_preprocessor_class,
        base_context
):
    """
    Verifies that the `fit` method correctly identifies column indices and
    passes them to `create_sliding_window`.
    """
    mock_create_window = mocker.patch(
        'models.base.create_sliding_window',
        return_value=(np.ones((50, 10, 3)), np.ones((50, 5, 1)), np.ones((50, 5, 1)))
    )
    processed_cols = ['dec_exog_1', 'enc_exog_1', 'target_1']
    processed_df = pd.DataFrame(np.random.randn(100, 3), columns=processed_cols)
    mock_preprocessor = mock_preprocessor_class.return_value
    mock_preprocessor.fit_transform.return_value = processed_df

    params = {'preprocessing': {}}
    window_size=10
    forecaster = ConcreteNeuralForecaster(
        params,
        num_features=1,
        forecast_steps=5,
        window_size=window_size,
        dataset=mock_dataset_with_exog,
        run_context=base_context
    )

    forecaster.fit(
        train_series=mock_dataset_with_exog.development_data,
        is_final_fit=True,
        dataset=mock_dataset_with_exog
    )

    _call_args, call_kwargs = mock_create_window.call_args
    assert call_kwargs['target_indices'] == [2]
    assert call_kwargs['decoder_exog_indices'] == [0]

def test_neural_fit_calculates_target_indices_correctly_A(
        mocker,
        mock_dataset_with_exog,
        mock_preprocessor_class,
        base_context
):
    """
    Verifies that `fit` passes correct target/decoder-exog indices to create_sliding_window
    based on the ACTUAL column names returned by the preprocessor.
    """

    # 1) Set up create_sliding_window mock
    mock_create_window = mocker.patch(
        'models.base.create_sliding_window',
        return_value=(np.ones((50, 10, 3)), np.ones((50, 5, 1)), np.ones((50, 5, 1)))
    )

    # 2) Ensure preprocessor returns DataFrame with NAME CONTROL
    processed_cols = ['dec_exog_1', 'enc_exog_1', 'target_1']
    processed_df = pd.DataFrame(np.random.randn(100, 3), columns=processed_cols)
    mock_preprocessor = mock_preprocessor_class.return_value
    mock_preprocessor.fit_transform.return_value = processed_df

    # 3) Build forecaster
    params = {'preprocessing': {}}
    window_size=10
    forecaster = ConcreteNeuralForecaster(
        params,
        num_features=1,
        forecast_steps=5,
        window_size=window_size,
        dataset=mock_dataset_with_exog,
        run_context=base_context
    )

    # 4) call fit
    forecaster.fit(
        train_series=mock_dataset_with_exog.development_data,
        is_final_fit=True,
        dataset=mock_dataset_with_exog
    )

    # 5) Read create_sliding_window arguments
    _call_args, call_kwargs = mock_create_window.call_args
    got_target_indices = call_kwargs['target_indices']
    got_decoder_indices = call_kwargs['decoder_exog_indices']

    # 6) Calculate EXPECTED dynamically from NAMES (this is the key)
    all_cols = processed_cols  # after preprocessing
    expected_target_indices = [all_cols.index(c) for c in mock_dataset_with_exog.target_columns if c in all_cols]
    expected_decoder_indices = [all_cols.index(c) for c in mock_dataset_with_exog.future_covariates if c in all_cols]

    # 7) If dataset names do not match post-preprocessing names, expected_* will be empty.
    # Then provide a very clear message WHY the test fails:
    assert got_target_indices == expected_target_indices, (
        f"target_indices mismatch.\n"
        f"Dataset target_columns={mock_dataset_with_exog.target_columns}\n"
        f"Processed columns={all_cols}\n"
        f"Expected (by NAME)={expected_target_indices}\n"
        f"Got={got_target_indices}\n"
        "Most likely cause: dataset uses logical names (e.g. 'target') while preprocessor renames to 'target_1'."
    )
    assert got_decoder_indices == expected_decoder_indices, (
        f"decoder_exog_indices mismatch.\n"
        f"Dataset future_covariates={mock_dataset_with_exog.future_covariates}\n"
        f"Processed columns={all_cols}\n"
        f"Expected (by NAME)={expected_decoder_indices}\n"
        f"Got={got_decoder_indices}\n"
    )

def test_preprocessor_preserves_column_names_and_order(mocker, base_context):
    """
    Diagnostic test (no codebase changes):
    - Does Preprocessor.fit_transform change BASE COLUMN NAMES?
    - Does it change BASE COLUMN ORDER?
    - Are new columns (if any) APPENDED EXCLUSIVELY at the end?
    """

    # 1) Dataset mock with "configuration" in cononical order
    base_targets = ['target_1']
    base_enc     = ['enc_exog_1']
    base_dec     = ['dec_exog_1']
    base_order   = base_targets + base_enc + base_dec  # [targets, enc_exog, dec_exog]

    df_in = pd.DataFrame(np.random.randn(120, len(base_order)), columns=base_order)

    # NeuralTSForecaster.__init__ uses .columns
    dataset = mocker.MagicMock(spec=[
        'development_data', 'target_columns', 'past_covariates', 'future_covariates', 'columns'
    ])
    dataset.development_data      = df_in
    dataset.target_columns        = list(base_targets)
    dataset.past_covariates       = list(base_enc)
    dataset.future_covariates     = list(base_dec)
    dataset.columns               = list(base_order)

    # 2) SPY on real Preprocessor.fit_transform method (without mocking the class)
    import models.base as base_mod
    recorded = {'in_cols': None, 'out_cols': None}
    _orig_fit_transform = base_mod.Preprocessor.fit_transform  # keep original

    def _spy_fit_transform(self, df):
        recorded['in_cols'] = list(df.columns)
        out = _orig_fit_transform(self, df)  # call REAL preprocessor
        # if numpy is returned (no names), wrap in DF with input columns
        if not hasattr(out, 'columns'):
            out = pd.DataFrame(out, columns=recorded['in_cols'])
        recorded['out_cols'] = list(out.columns)
        return out

    mocker.patch.object(base_mod.Preprocessor, 'fit_transform', autospec=True, side_effect=_spy_fit_transform)

    # 3) Minimal forecaster to trigger real fit() from base.py
    from models.base import NeuralTSForecaster

    class _NoTrainModel(nn.Module):
        def __init__(self, forecast_steps, num_features):
            super().__init__()
            self._fs = forecast_steps
            self._nf = num_features
        def forward(self, x):
            B = x.size(0)
            return torch.zeros(B, self._fs, self._nf, device=x.device)

    class DummyForecaster(NeuralTSForecaster):
        def __init__(self, model_params, num_features, forecast_steps, window_size, **kwargs):
            super().__init__(model_params, num_features, forecast_steps, window_size, **kwargs)
            self.model = _NoTrainModel(forecast_steps=self.forecast_steps, num_features=self.num_features)
        def _has_decoder(self) -> bool:
            return True
        def _train_model(self, **tensors):
            return self.model
        def _internal_predict(self, input_tensor: torch.Tensor, **kwargs):
            self.model.eval()
            with torch.no_grad():
                return self.model(input_tensor).cpu().numpy()

    # 4) Run fit() - passes through REAL preprocessor and create_sliding_window
    params = {'preprocessing': {}}
    window_size=10
    f = DummyForecaster(
        params,
        num_features=len(base_targets),
        forecast_steps=5,
        window_size=window_size,
        dataset=dataset,
        run_context=base_context
    )
    f.fit(train_series=dataset.development_data, is_final_fit=True, dataset=dataset)

    # 5) Diagnostic assertions
    in_cols  = recorded['in_cols']
    out_cols = recorded['out_cols']

    # (a) base column names must be preserved (no drop/rename)
    missing = [c for c in in_cols if c not in out_cols]
    assert not missing, (
        "Preprocessor removed or renamed base columns.\n"
        f"in_cols={in_cols}\n"
        f"out_cols={out_cols}\n"
        f"missing={missing}"
    )

    # (b) ORDER of base columns must be 1:1 (new columns - if any - only at the end)
    out_head = out_cols[:len(in_cols)]
    assert out_head == in_cols, (
        "Preprocessor changed the ORDER of base columns.\n"
        f"expected(in_cols)={in_cols}\n"
        f"actual(out_head)={out_head}"
    )

    # (c) for info: new columns (if any) - should be at the end
    new_cols = out_cols[len(in_cols):]
    if new_cols:
        print(f"[INFO] Preprocessor appended new columns at tail: {new_cols}")
