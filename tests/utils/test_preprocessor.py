import pytest
import pandas as pd
import numpy as np
from typing import List
from utils.preprocessor import Preprocessor
import logging

pytestmark = pytest.mark.unit


# --- Fixtures ---

@pytest.fixture
def sample_data():
    """Provides a sample DataFrame with targets and different types of exogenous features."""
    index = pd.to_datetime(pd.date_range(start='2020-01-01', periods=100, freq='D'))
    data = pd.DataFrame({
        'sales': 100 + np.arange(100) + np.sin(np.arange(100) / 7) * 10,
        'marketing_spend': 10 + np.arange(100) * 0.5,
        'is_promotion': (np.arange(100) % 10 == 0).astype(int),
    }, index=index)
    return data


@pytest.fixture
def target_cols() -> List[str]:
    return ["sales"]


@pytest.fixture
def exog_cols() -> List[str]:
    # This list also includes time features that will be generated
    return ["marketing_spend", "is_promotion", "day_of_week"]


@pytest.fixture
def full_config():
    """
    Provides a full, valid group-based preprocessing configuration.
    Differencing has been REMOVED from this config.
    """
    return {
        "preprocessing_groups": [
            {"name": "target_pipeline", "apply_to": "__targets__",
             "pipeline": {"log_transform": {"enabled": True, "method": "log1p"},
                          "winsorize": {"enabled": True, "limits": [0.05, 0.05]},
                          "scaling": {"enabled": True, "method": "minmax"}}},
            {"name": "timeseries_exog_pipeline", "apply_to": ["marketing_spend"],
             "pipeline": {"scaling": {"enabled": True, "method": "standard"}}},
            {"name": "static_features_pipeline", "apply_to": ["is_promotion", "day_of_week"],
             "pipeline": {"scaling": {"enabled": True, "method": "standard"}}}
        ]
    }


# --- Tests ---

def test_initialization_with_groups(full_config, target_cols, exog_cols):
    """
    Tests that the Preprocessor correctly initializes pipelines for each column based on groups.
    """
    preprocessor = Preprocessor(full_config, target_columns=target_cols, exog_columns=exog_cols)

    # Assertions to check if the correct pipelines were assigned to the correct columns
    assert "log_transform" in preprocessor.column_pipelines["sales"]
    assert "winsorize" in preprocessor.column_pipelines["sales"]
    assert "scaling" in preprocessor.column_pipelines["marketing_spend"]

    # Assert that differencing is GONE
    assert "differencing" not in preprocessor.column_pipelines["sales"]
    assert "differencing" not in preprocessor.column_pipelines.get("marketing_spend", {})
    assert "differencing" not in preprocessor.column_pipelines.get("is_promotion", {})


def test_transform_requires_fit(full_config, sample_data, target_cols, exog_cols):
    """transform() must require a prior fit_transform()."""
    preprocessor = Preprocessor(full_config, target_columns=target_cols, exog_columns=exog_cols)
    with pytest.raises(RuntimeError):
        _ = preprocessor.transform(sample_data)


def test_winsorization_clips_outliers():
    """
     Verifies that the winsorize step correctly clips outliers.
    """
    np.random.seed(42)
    config = {
        "preprocessing_groups": [{
            "name": "test", "apply_to": ["outlier_data"],
            "pipeline": {"winsorize": {"enabled": True, "limits": [0.1, 0.1]}}
        }]
    }
    data = pd.DataFrame({'outlier_data': [-100, 10, 11, 12, 13, 14, 15, 16, 17, 500]})
    preprocessor = Preprocessor(config, target_columns=["outlier_data"], exog_columns=[])

    transformed = preprocessor.fit_transform(data)

    assert transformed['outlier_data'].iloc[0] == 10
    assert transformed['outlier_data'].iloc[-1] == 17
    pd.testing.assert_series_equal(
        transformed['outlier_data'].iloc[1:-1],
        data['outlier_data'].iloc[1:-1],
        check_names=False
    )


def test_winsorization_uses_train_bounds_on_transform():
    """Winsorization must use bounds fitted on train when transforming new data."""
    config = {
        "preprocessing_groups": [{
            "name": "test", "apply_to": ["x"],
            "pipeline": {"winsorize": {"enabled": True, "limits": [0.2, 0.2]}}
        }]
    }
    train = pd.DataFrame({"x": [0, 1, 2, 3, 4, 100]})
    test = pd.DataFrame({"x": [-1000, 2, 1000]})
    pre = Preprocessor(config, target_columns=["x"], exog_columns=[])
    _ = pre.fit_transform(train)
    got = pre.transform(test)["x"]
    assert got.iloc[0] == 1
    assert got.iloc[1] == 2
    assert got.iloc[2] == 4

# @pytest.mark.xfail(reason="support for 1D input missing or needs review", strict=True) <-- REMOVED
def test_inverse_transforms_accepts_1d_ndarray_with_start_after(mocker):
    """1D ndarray with start_after should return a single-column DataFrame."""
    # Mock pd.infer_freq to return a valid frequency
    mocker.patch('pandas.infer_freq', return_value='D')

    config = {"preprocessing_groups": [{
        "name": "g", "apply_to": ["y"],
        "pipeline": {"scaling": {"enabled": True, "method": "minmax"}}
    }]}
    df = pd.DataFrame({"y": np.arange(20, dtype=float)}, index=pd.date_range("2021-01-01", periods=20, freq="D"))
    pre = Preprocessor(config, target_columns=["y"], exog_columns=[])

    # Fit on [0, 1, ..., 14]
    pre.fit_transform(df.iloc[:15])

    # Transform [15, 16, 17, 18, 19]
    tr = pre.transform(df.iloc[15:])

    # Get 1D numpy array of transformed data
    yhat = tr["y"].to_numpy()  # This is a 1D array

    # Invert, passing the last timestamp of the *training* data
    inv = pre.inverse_transforms(yhat, start_after=df.index[14])

    # The result 'inv' should be equal to the original test data
    pd.testing.assert_frame_equal(inv, df.iloc[15:], rtol=1e-6)


def test_winsorize_limits_invalid_values_raise():
    """Invalid winsorization limits must raise ValueError."""
    bad_limits_cases = [
        [-0.1, 0.1], [0.1, -0.1], [0.5, 0.1], [0.1, 0.5],
        [0.7, 0.4], [0.6, 0.6], [0.2], [0.2, 0.2, 0.2],
    ]
    for limits in bad_limits_cases:
        config = {
            "preprocessing_groups": [{
                "name": "g", "apply_to": ["x"],
                "pipeline": {"winsorize": {"enabled": True, "limits": limits}}
            }]
        }
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100]})
        pre = Preprocessor(config, target_columns=["x"], exog_columns=[])
        with pytest.raises(ValueError):
            _ = pre.fit_transform(df)


def test_winsorize_limits_valid_values_pass():
    """Edge-but-valid winsorization limits should pass."""
    ok_limits_cases = [
        [0, 0], [0.01, 0.01], [0.0, 0.49], [0.49, 0.0],
    ]
    for limits in ok_limits_cases:
        config = {
            "preprocessing_groups": [{
                "name": "g", "apply_to": ["x"],
                "pipeline": {"winsorize": {"enabled": True, "limits": limits}}
            }]
        }
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100]})
        pre = Preprocessor(config, target_columns=["x"], exog_columns=[])
        out = pre.fit_transform(df)
        assert "x" in out.columns


def test_preprocessor_preserves_column_names_and_order():
    """
    Verifies that the Preprocessor does not alter column names or their order.
    """
    rng = np.random.default_rng(0)
    input_data = pd.DataFrame({
        'c_metric': rng.random(50) + 1.0,
        'a_metric': rng.random(50) + 5.0,
        'b_metric': rng.random(50) * 10 + 1.0,
    })
    input_column_order = list(input_data.columns)
    input_column_names = set(input_data.columns)

    config = {
        "preprocessing_groups": [
            {
                "name": "full_pipeline",
                "apply_to": ["c_metric", "a_metric", "b_metric"],
                "pipeline": {
                    "log_transform": {"enabled": True},
                    "scaling": {"enabled": True, "method": "standard"}
                    # "differencing" removed
                }
            }
        ]
    }

    from utils.preprocessor import Preprocessor

    preprocessor = Preprocessor(
        config=config,
        target_columns=['c_metric', 'a_metric', 'b_metric'],
        exog_columns=[]
    )

    transformed_data = preprocessor.fit_transform(input_data)
    output_column_order = list(transformed_data.columns)
    output_column_names = set(transformed_data.columns)

    assert output_column_names == input_column_names, \
        f"Preprocessor changed column names. Expected {input_column_names}, got {output_column_names}."

    assert output_column_order == input_column_order, \
        f"Preprocessor changed column order. Expected {input_column_order}, got {output_column_order}."


def test_scaling_minmax_smoke_test(target_cols):
    """
    Verifies that the MinMaxScaler (minmax) correctly scales data to [0, 1] range.
    """
    config = {
        "preprocessing_groups": [{
            "name": "minmax_test", "apply_to": ["sales"],
            "pipeline": {"scaling": {"enabled": True, "method": "minmax", "range": [0, 1]}}
        }]
    }
    data = pd.DataFrame({'sales': np.linspace(0, 100, 50)})
    preprocessor = Preprocessor(config, target_columns=["sales"], exog_columns=[])

    transformed = preprocessor.fit_transform(data)

    assert transformed['sales'].min() == pytest.approx(0.0)
    assert transformed['sales'].max() == pytest.approx(1.0)
    assert transformed['sales'].mean() == pytest.approx(0.5, abs=0.01)


def test_initialization_correctly_maps_targets_to_scaling(target_cols, exog_cols):
    """
    Verifies that Preprocessor correctly maps 'apply_to: "__targets__"'
    to the target columns list and enables scaling for them.
    """
    config = {
        "preprocessing_groups": [
            {"name": "minmax_test", "apply_to": "__targets__",
             "pipeline": {"scaling": {"enabled": True, "method": "minmax"}}},
        ]
    }
    preprocessor = Preprocessor(config, target_columns=target_cols, exog_columns=exog_cols)
    assert "scaling" in preprocessor.column_pipelines[
        target_cols[0]], "Scaling pipeline was not assigned to target column."
    assert preprocessor.column_pipelines[target_cols[0]]["scaling"][
               "enabled"] is True, "Scaling was assigned but not enabled."


def test_standard_scaling_pipeline_execution_success(target_cols):
    """
    Unit test to verify if the Preprocessor correctly executes Standard Scaling
    for target columns.
    """
    config = {
        "preprocessing_groups": [
            {"name": "standard_scaling_check", "apply_to": "__targets__",
             "pipeline": {"scaling": {"enabled": True, "method": "standard"}}}
        ]
    }
    data = pd.DataFrame({'sales': np.linspace(0, 100, 100)})
    preprocessor = Preprocessor(config, target_columns=["sales"], exog_columns=[])

    transformed = preprocessor.fit_transform(data)

    assert (transformed['sales'].mean() ==
            pytest.approx(0.0, abs=1e-9)), "Standard Scaling failed: Mean is not close to zero."
    assert (transformed['sales'].std(ddof=0) ==  # Use ddof=0 for population std dev
            pytest.approx(1.0, abs=1e-9)), "Standard Scaling failed: Std Dev is not close to one."