import pytest
import pandas as pd
from unittest.mock import MagicMock
from models.base import TSForecaster
from typing import Tuple, Dict, List
from utils.dataset import TimeSeriesDataset


def test_optimize_hyperparameters_uses_factory(monkeypatch, base_context):
    """
    Scenario: The hyperparameter optimization loop is run.
    Verifies: The loop correctly calls the ModelFactory to create candidates
              and evaluates the loss using the candidate instance.
    """

    # 1. Minimal dataset stub
    class DummyDataset(TimeSeriesDataset):
        def __init__(self):
            self.development_data = pd.DataFrame({"y": list(range(1, 9))})
            self.config = {"experiments": [{}]}
            self.target_columns = ["y"]
            self.columns = ["y"]
            self.past_covariates = []
            self.future_covariates = []

        def generate_walk_forward_folds(self, max_window_size, n_folds):
            return [self.development_data.iloc[:5]]

    # 2. Concrete implementation of the abstract base class for testing
    class DummyModel(TSForecaster):
        model_name = "DummyModel"

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def _fit_and_evaluate_fold(self, train_fold: pd.DataFrame, eval_fold: pd.DataFrame, validation_params: dict,
                                   dataset=None, is_final_fit=False, **kwargs) -> Tuple[float, pd.DataFrame, Dict[str, List[float]]]:
            return 0.5, pd.DataFrame(), {}

        def fit(self, *args, **kwargs) -> None:
            self.fitted = True

        def predict(self, *args, **kwargs) -> pd.DataFrame:
            return pd.DataFrame()

        def get_valid_params(self) -> set:
            return {"param1"}

    # 3. Mock ModelFactory.create instead of registry create_model
    #    This matches the new logic in models/base.py
    def mock_factory_create(model_type, model_name, *args, **kwargs):
        # We verify that base.py passed the correct type and generated a name
        if model_type == "DummyModelType" and "trial" in model_name:
            return DummyModel(*args, **kwargs)
        return MagicMock()

    monkeypatch.setattr("models.factory.ModelFactory.create", mock_factory_create)

    # 4. Mock hyperparameter generation to return a predictable candidate
    monkeypatch.setattr("models.base.generate_random_params", lambda *args, **kwargs: [{"param1": 10}])

    # 5. Mock dataset
    mock_ds = MagicMock(spec=TimeSeriesDataset)
    mock_ds.target_columns = ["y"]
    # New API
    mock_ds.past_covariates = []
    mock_ds.future_covariates = []
    mock_ds.columns = ["y"]

    # --- Test Execution ---
    model_config = {
        "type": "DummyModelType",  # Required by new validation logic
        "optimize": True,
        "optimization": {
            "method": "random",
            "params": {"param1": [10, 20]}
        }
    }

    # Initial instance to call optimize on
    forecaster = DummyModel(
        model_params=model_config,
        num_features=1,
        forecast_steps=2,
        window_size=10,
        dataset=mock_ds,
        run_context=base_context)

    mock_folds = [(pd.DataFrame({'y': [1, 2]}), pd.DataFrame({'y': [3, 4]}))]

    best_params, best_loss = forecaster.optimize_hyperparameters(
        dataset=mock_ds,
        model_config=model_config,
        validation_params={"n_folds": 1, "max_window_size": 1},
        folds=mock_folds
    )

    # Assertions
    assert best_loss == 0.5
    assert best_params["param1"] == 10
    assert best_params["type"] == "DummyModelType"
