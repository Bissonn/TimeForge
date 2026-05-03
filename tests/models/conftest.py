import pytest
import pandas as pd
import numpy as np
import torch
from utils.dataset import TimeSeriesDataset
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

WINDOW_SIZE = 10


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def large_trend_dataset():
    """
    Dataset for predict path test:
    - target: deterministic, periodic (sum of sines, no noise),
    - enc_exog: additional, smooth time function,
    - important: target value range at the end of series is the same,
      as at the beginning -> no extrapolation with min-max scaler.
    """

    n = 400
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    t = np.arange(n)

    # Periodic, bounded signal:
    # - main component with period 50 days,
    # - additional component with shorter period 15 days.
    base1 = np.sin(2.0 * np.pi * t / 50.0)
    base2 = np.sin(2.0 * np.pi * t / 15.0)

    # Target in limited, constant range (e.g. [7, 13]).
    target = 10.0 + 2.0 * base1 + 1.0 * base2  # ~ 10 ± 3

    # Simple, smooth enc_exog feature - also periodic / smooth.
    enc_exog = np.cos(2.0 * np.pi * t / 30.0)

    data = pd.DataFrame(
        {
            "target": target,
            "enc_exog": enc_exog,
        },
        index=idx,
    )

    ds = TimeSeriesDataset(
        dataset_name="large_trend",
        config={"datasets": {"large_trend": {}}},
        num_features=1,
        data=data,
        columns=["target"],
        past_covariates=["enc_exog"],  # Encoder-only → past_covariate
    )

    # This split is used only by TimeSeriesDataset / other tests
    # possibly - in the test itself we operate on ds.series anyway.
    ds.split_data(forecast_steps=5)

    return ds

@pytest.fixture
def base_transformer_config() -> dict:
    """
    A minimal, valid baseline configuration for the Transformer model.
    This is used across multiple test modules as a starting point and is
    then overridden/extended with architecture-specific options.

    DETERMINISM: num_workers=0 ensures deterministic DataLoader behavior
    by disabling multiprocessing. This is crucial for reproducible test results.
    Production code may use num_workers>0 for performance, but tests require
    stability across different hardware configurations.
    """
    return {
        "hidden_size": 128,
        "num_heads": 4,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "dim_ff_multiplier": 4.0,
        "epochs": 1,
        "batch_size": 4,
        "learning_rate": 0.01,
        "num_workers": 0,  # CRITICAL: Deterministic DataLoader (no multiprocessing)
        "preprocessing": {
            "preprocessing_groups": [
                {
                    "name": "default",
                    "apply_to": "__targets__",
                    "pipeline": {"scaling": {"enabled": True}},
                }
            ]
        },
    }


@pytest.fixture(params=[{'dec_exog_cols': []}, {'dec_exog_cols': ['dec_exog']}])
def real_mock_dataset(request):
    """Parametrized real dataset: univariate target + enc_exog; optional dec_exog."""
    np.random.seed(42)  # Deterministic for reproducible tests
    n_samples = 100
    data_dict = {
        'date': pd.to_datetime(pd.date_range(start="2023-01-01", periods=n_samples)),
        'target': np.arange(n_samples, dtype=float),
        'enc_exog': np.random.rand(n_samples)
    }
    if request.param['dec_exog_cols']:
        data_dict['dec_exog'] = np.random.rand(n_samples)

    data = pd.DataFrame(data_dict)

    # Migrate to new API:
    # - enc_exog is encoder-only → past_covariate
    # - dec_exog (if present) is decoder-only → future_covariate
    dec_exog_cols = request.param['dec_exog_cols']
    ds = TimeSeriesDataset(
        "real_mock",
        {"datasets": {"real_mock": {}}},
        num_features=1,
        data=data,
        columns=['target'],
        past_covariates=['enc_exog'],
        future_covariates=dec_exog_cols if dec_exog_cols else []
    )
    ds.split_data(forecast_steps=5)  # Default split; tests can override
    return ds


@pytest.fixture
def enc_only_dataset():
    """Dataset without dec exog for enc-only tests."""
    np.random.seed(42)
    n_samples = 100
    data = pd.DataFrame({
        'date': pd.to_datetime(pd.date_range(start="2023-01-01", periods=n_samples)),
        'target': np.arange(n_samples, dtype=float),
        'enc_exog': np.random.rand(n_samples)
    })
    ds = TimeSeriesDataset(
        "enc_only",
        {"datasets": {"enc_only": {}}},
        num_features=1,
        data=data,
        columns=['target'],
        past_covariates=['enc_exog'],  # Encoder-only → past_covariate
        future_covariates=[]  # No decoder exog
    )
    ds.split_data(forecast_steps=5)
    return ds

@pytest.fixture
def full_dataset():
    """
    Dataset with dec exog and multi-feature (F=2) for enc-dec tests.

    IMPROVED: Generates cyclical/stationary data with small trend.
    This ensures test data (last 5-10 samples) have similar distribution to train data.

    Data characteristics:
    - Length: 200 samples (increased from 100)
    - Target: Sinusoidal pattern + small trend + noise
    - Target2: Correlated with target but different amplitude
    - Both targets stay in similar value range throughout the series
    - Test data (last samples) are within train distribution
    """
    np.random.seed(42)
    n_samples = 200

    # ============================================================
    # IMPROVED TARGET GENERATION
    # ============================================================
    # Instead of linear trend (np.arange), we use:
    # - Sinusoids (cyclicity)
    # - Small trend (not dominant)
    # - Noise (realism)

    # Parameters
    base_level = 50.0  # Series mean level
    amplitude = 30.0  # Seasonal fluctuation amplitude
    trend_total = 10.0  # Total trend (only +10 over 200 samples)
    noise_std = 5.0  # Noise standard deviation

    # Components
    time = np.arange(n_samples)
    trend = np.linspace(0, trend_total, n_samples)  # Small linear trend
    seasonal = amplitude * np.sin(
        np.linspace(0, 4 * np.pi, n_samples))  # 2 full cycles
    noise = np.random.randn(n_samples) * noise_std

    # Target 1: Base + trend + seasonal + noise
    target = base_level + trend + seasonal + noise

    # Target 2: Correlated with target, but different amplitude and phase
    target2_amplitude = 20.0
    target2_base = 40.0
    target2_seasonal = target2_amplitude * np.sin(np.linspace(0, 4 * np.pi, n_samples) + np.pi / 4)  # Phase shift
    target2_noise = np.random.randn(n_samples) * 3.0

    target2 = target2_base + trend * 0.5 + target2_seasonal + target2_noise

    # ============================================================
    # VERIFY DISTRIBUTION
    # ============================================================
    # Let's check if train and test have similar distribution
    train_size = int(n_samples * 0.95)  # 190 train, 10 test
    test_size = n_samples - train_size

    train_target = target[:train_size]
    test_target = target[train_size:]

    print(f"\n{'=' * 70}")
    print("DATASET STATISTICS (after fix)")
    print(f"{'=' * 70}")
    print(f"Total samples: {n_samples}")
    print(f"Train samples: {train_size}")
    print(f"Test samples:  {test_size}")
    print(f"\nTarget distribution:")
    print(
        f"  Train: min={train_target.min():.2f}, max={train_target.max():.2f}, mean={train_target.mean():.2f}, std={train_target.std():.2f}")
    print(
        f"  Test:  min={test_target.min():.2f}, max={test_target.max():.2f}, mean={test_target.mean():.2f}, std={test_target.std():.2f}")
    print(f"\nOverlap check:")
    print(f"  Train range: [{train_target.min():.2f}, {train_target.max():.2f}]")
    print(f"  Test range:  [{test_target.min():.2f}, {test_target.max():.2f}]")

    # Calculate overlap percentage
    test_in_train_range = ((test_target >= train_target.min()) & (test_target <= train_target.max())).sum()
    overlap_pct = 100 * test_in_train_range / len(test_target)
    print(f"  Test values within train range: {test_in_train_range}/{len(test_target)} ({overlap_pct:.1f}%)")

    if overlap_pct >= 80:
        print(f"  ✓ GOOD: {overlap_pct:.1f}% overlap - test data are within train distribution!")
    else:
        print(f"  ⚠ WARNING: Only {overlap_pct:.1f}% overlap - test may be too far from train!")
    print(f"{'=' * 70}\n")

    # ============================================================
    # CREATE DATAFRAME
    # ============================================================
    data = pd.DataFrame({
        'date': pd.date_range(start="2023-01-01", periods=n_samples, freq='D'),
        'target': target,
        'target2': target2,
        'enc_exog': np.random.rand(n_samples),  # Encoder exogenous
        'dec_exog': np.random.rand(n_samples)  # Decoder exogenous
    })
    data.set_index('date', inplace=True)

    # ============================================================
    # CREATE DATASET
    # ============================================================
    ds = TimeSeriesDataset(
        "full",
        {"datasets": {"full": {}}},
        num_features=2,
        data=data,
        columns=['target', 'target2'],  # F=2
        past_covariates=['enc_exog'],    # Encoder-only
        future_covariates=['dec_exog']   # Decoder-only
    )

    # Split will be done in test via ds.split_data(forecast_steps=5)

    return ds

@pytest.fixture
def mock_run_dataset() -> TimeSeriesDataset:
    """
    Mocks a `run_dataset` instance with 1 target, 1 encoder exog, and 1 decoder exog.
    This simulates the tailored dataset a model receives for a specific run.
    """
    config = {"datasets": {"mock_data": {}}}
    data = pd.DataFrame({
        'target': np.arange(100, dtype=float),
        'enc_exog': np.random.rand(100),
        'dec_exog': np.random.rand(100)
    })

    dataset = TimeSeriesDataset(
        "mock_data", config,
        num_features=1,
        data=data,
        columns=['target'],
        past_covariates=['enc_exog'],    # Encoder-only
        future_covariates=['dec_exog']   # Decoder-only
    )
    dataset.split_data(forecast_steps=10)
    return dataset

@pytest.fixture
def mock_dataset_simple():
    """Simple mock dataset with 2 targets, no exog."""
    mock_ds = MagicMock(spec=TimeSeriesDataset)
    mock_ds.columns = ['target1', 'target2']
    mock_ds.target_columns = ['target1', 'target2']
    # New API
    mock_ds.past_covariates = []
    mock_ds.future_covariates = []
    return mock_ds

@pytest.fixture
def mock_dataset_with_exog():
    """
    Mocks a TimeSeriesDataset with one target, one encoder exog, and one decoder exog variable.
    """
    mock = MagicMock(spec=TimeSeriesDataset)

    mock.target_columns = ['target_1']

    # New API: enc_exog_1 is encoder-only (past_covariate)
    #          dec_exog_1 is decoder-only (future_covariate)
    mock.past_covariates = ['enc_exog_1']
    mock.future_covariates = ['dec_exog_1']

    all_cols = mock.target_columns + mock.past_covariates + mock.future_covariates
    mock.columns = sorted(all_cols)

    mock.series = pd.DataFrame(np.random.rand(120, 3), columns=mock.columns)
    mock.development_data = mock.series.iloc[:100]
    mock.test_data = mock.series.iloc[100:]
    mock.name = "mock_dataset_with_exog"
    mock.config = {}
    mock.freq = 'D'
    mock.generate_walk_forward_folds.return_value = [
        pd.DataFrame(np.random.rand(100, 3), columns=mock.columns),
    ]
    return mock
