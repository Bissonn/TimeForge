import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st
from core.trainer import ModelTrainer


# --- 1. MOCK / ISOLATION ---
class AggregationTester(ModelTrainer):
    def __init__(self):
        pass


tester = AggregationTester()

# --- 2. HYPOTHESIS STRATEGIES ---

metric_names = st.sampled_from(["mae", "rmse", "smape"])
channel_names = st.sampled_from(["OT", "HUFL", "LULL", "WTH"])

# Combine bounded floats with explicit NaN strategy
metric_values = st.one_of(
    st.floats(min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False),
    st.just(float('nan'))
)


@st.composite
def fold_metrics_strategy(draw):
    """
    Generates a dictionary of metrics for a single fold.
    """
    # 1. Global metrics
    metrics = draw(st.dictionaries(keys=metric_names, values=metric_values, min_size=1))

    # 2. Per-channel metrics decision
    has_per_channel = draw(st.booleans())

    if has_per_channel:
        pc_data = {}
        selected_metrics = draw(st.lists(metric_names, unique=True))

        for m in selected_metrics:
            channels = draw(st.dictionaries(keys=channel_names, values=metric_values))
            if channels:
                pc_data[m] = channels

        if pc_data:
            metrics['per_channel'] = pc_data

    return metrics


# --- 3. PROPERTY TESTS ---

@given(st.lists(fold_metrics_strategy(), min_size=0, max_size=50))
def test_aggregation_robustness_no_crash(all_metrics):
    """
    PROPERTY: The method never raises an exception for valid input types.
    """
    try:
        result = tester._aggregate_backtest_metrics(all_metrics)
        assert isinstance(result, dict)
    except Exception as e:
        pytest.fail(f"Aggregation crashed on input: {all_metrics}. Error: {e}")


@given(st.lists(fold_metrics_strategy(), min_size=1, max_size=50))
def test_aggregation_global_math_correctness(all_metrics):
    """
    PROPERTY: For global metrics, result equals numpy.nanmean of inputs.
    """
    result = tester._aggregate_backtest_metrics(all_metrics)

    all_keys = set()
    for m in all_metrics:
        all_keys.update([k for k in m.keys() if k != 'per_channel'])

    for key in all_keys:
        values = [m[key] for m in all_metrics if key in m and isinstance(m[key], (int, float))]

        if not values or all(np.isnan(v) for v in values):
            continue

        expected_mean = np.nanmean(values)

        assert key in result, f"Missing key {key}"
        assert result[key] == pytest.approx(expected_mean, nan_ok=True)

        if len([v for v in values if not np.isnan(v)]) > 1:
            expected_std = np.nanstd(values, ddof=1)
            std_key = f"{key}_std"
            assert std_key in result
            assert result[std_key] == pytest.approx(expected_std, nan_ok=True)


@given(st.lists(fold_metrics_strategy(), min_size=1, max_size=20))
def test_aggregation_per_channel_logic(all_metrics):
    """
    PROPERTY: Per-channel aggregation logic matches manual calculation.
    """
    result = tester._aggregate_backtest_metrics(all_metrics)

    if 'per_channel_agg' not in result:
        return

    # Find a valid metric->channel path to verify
    target_metric = None
    target_channel = None

    for m in all_metrics:
        if 'per_channel' in m:
            for met, channels in m['per_channel'].items():
                for chan in channels:
                    target_metric = met
                    target_channel = chan
                    break

    if target_metric is None:
        return

    collected_values = []
    for m in all_metrics:
        try:
            val = m['per_channel'][target_metric][target_channel]
            if not np.isnan(val):
                collected_values.append(val)
        except KeyError:
            continue

    if not collected_values:
        return

    try:
        agg_result = result['per_channel_agg'][target_metric][target_channel]
        expected_mean = np.mean(collected_values)

        assert agg_result['mean'] == pytest.approx(expected_mean)

        if len(collected_values) > 1:
            expected_std = np.std(collected_values, ddof=1)
            assert agg_result['std'] == pytest.approx(expected_std)
        else:
            assert agg_result['std'] == 0.0

    except KeyError:
        pytest.fail(f"Missing result for {target_metric}->{target_channel}")


# --- 4. CORNER CASES ---

def test_empty_input():
    assert tester._aggregate_backtest_metrics([]) == {}


def test_single_fold_std_zero():
    metrics = [{"mae": 10.0, "per_channel": {"mae": {"OT": 5.0}}}]
    res = tester._aggregate_backtest_metrics(metrics)
    assert np.isnan(res.get("mae_std", np.nan))
    assert res["per_channel_agg"]["mae"]["OT"]["std"] == 0.0


def test_mixed_failures_and_success():
    inputs = [
        {"mae": 10.0},
        {},
        {"mae": np.nan},
        {"mae": 20.0}
    ]
    res = tester._aggregate_backtest_metrics(inputs)
    assert res["mae"] == 15.0
    assert res["mae_std"] == pytest.approx(7.0710678)
