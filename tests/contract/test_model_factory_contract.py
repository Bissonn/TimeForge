import pytest
from unittest.mock import MagicMock

from models.factory import ModelFactory


def test_model_factory_lists_models():
    models = ModelFactory.list_models()
    assert isinstance(models, list)
    assert len(models) > 0


@pytest.fixture
def dummy_dataset():
    """
    Minimal dataset satisfying model constructor contract.
    """
    ds = MagicMock()
    ds.num_features = 1
    ds.target_columns = ["y"]
    ds.past_covariates = []
    ds.future_covariates = []
    ds.columns = ["y"]
    return ds


@pytest.fixture
def contract_run_context(base_context):
    """
    Ensure RunContext satisfies ModelFactory contract.
    """
    base_context.forecast_steps = 3
    base_context.window_size = 5
    return base_context


@pytest.mark.parametrize("model_type", ModelFactory.list_models())
def test_model_factory_creates_model_instance(
    model_type,
    contract_run_context,
    dummy_dataset,
):
    # Provide minimal valid parameters for each model type
    minimal_params = {
        "lstm": {"hidden_size": 32, "num_layers": 1},
        "transformer": {"hidden_size": 32, "num_heads": 2},
        "arima": {"p": 1, "d": 1, "q": 1},
        "sarima": {"p": 1, "d": 1, "q": 1, "P": 0, "D": 0, "Q": 0, "m": 12},
        "var": {"max_lags": 2},
    }

    model_params = minimal_params.get(model_type, {})

    model = ModelFactory.create(
        model_type=model_type,
        model_name="contract_test_model",
        model_params=model_params,
        num_features=1,
        forecast_steps=3,
        window_size=5,
        dataset=dummy_dataset,
        run_context=contract_run_context,
    )

    assert model is not None
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
    assert model.model_type == model_type
    assert model.model_name == "contract_test_model"
