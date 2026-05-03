import pytest
import pandas as pd

from utils.dataset import TimeSeriesDataset


def _make_series(n: int) -> pd.DataFrame:
    """Create a simple single-column series of length n with strictly increasing values."""
    return pd.DataFrame({"y": range(n)})


@pytest.mark.parametrize("N,K,F", [(1000, 3, 30), (500, 4, 20), (200, 2, 50)])
def test_generate_sequential_folds_backtest_basic(N, K, F):
    """
    Backtesting: folds are generated on the FULL dataset of length N.
    Properties:
      - There are exactly K folds.
      - Each holdout has length F.
      - Train lengths: N - K*F, N - (K-1)*F, ..., N - F  (i.e., base + i*F).
      - Concatenated holdouts equal the trailing K*F block of the original series.
      - The last fold ends at the end of the series (val_end == N).
      - Holdout blocks are contiguous and non-overlapping.
    """
    series = _make_series(N)
    folds = TimeSeriesDataset.generate_sequential_folds(series=series, n_folds=K, forecast_steps=F)

    assert len(folds) == K

    base = N - K * F
    # Check fold-by-fold lengths and boundaries
    for i, (train_df, holdout_df) in enumerate(folds):
        expected_train_len = base + i * F
        assert len(holdout_df) == F
        assert len(train_df) == expected_train_len

        # Holdout starts immediately after the train and extends F steps
        assert holdout_df.index[0] == expected_train_len
        assert holdout_df.index[-1] == expected_train_len + F - 1

        # No overlap between train and holdout
        assert (len(train_df) == 0) or (train_df.index[-1] < holdout_df.index[0])

    # The K holdouts must cover exactly the last K*F rows of the series, contiguously
    holdouts_concat = pd.concat([h for _, h in folds], axis=0)
    tail_block = series.iloc[N - K * F :]
    pd.testing.assert_frame_equal(holdouts_concat.reset_index(drop=True), tail_block.reset_index(drop=True))

    # Last fold must end at N
    last_holdout = folds[-1][1]
    assert last_holdout.index[-1] == N - 1


@pytest.mark.parametrize("N,K,F", [(1000, 3, 30), (500, 4, 20), (210, 2, 50)])
def test_generate_sequential_folds_hpo_basic(N, K, F):
    """
    HPO: folds are generated on the HEAD dataset of length N' = N - K*F (no access to backtest pool).
    Properties mirror the backtesting case but with N' instead of N:
      - Exactly K folds.
      - Each holdout has length F.
      - Train lengths: N' - K*F, N' - (K-1)*F, ..., N' - F  (i.e., base' + i*F), where base' = N' - K*F = N - 2*K*F.
      - Concatenated holdouts equal the trailing K*F block of the HPO dataset.
      - Last fold ends at the end of the HPO dataset (val_end == N').
    Note: This test assumes N is large enough such that N' >= K*F + 1 (enforced by the generator).
    """
    assert N - K * F > 0, "HPO dataset must be positive length."
    hpo_series = _make_series(N).iloc[: N - K * F].copy()
    N_prime = len(hpo_series)

    # Guard: generator requires at least K*F + 1 points to form K folds (>=1 training point in the first fold)
    if N_prime < K * F + 1:
        pytest.skip(f"Skipping case with N'={N_prime} < K*F+1={K*F+1}")

    folds = TimeSeriesDataset.generate_sequential_folds(series=hpo_series, n_folds=K, forecast_steps=F)
    assert len(folds) == K

    base_prime = N_prime - K * F
    for i, (train_df, holdout_df) in enumerate(folds):
        expected_train_len = base_prime + i * F
        assert len(holdout_df) == F
        assert len(train_df) == expected_train_len

        # Boundaries within the HPO dataset
        assert holdout_df.index[0] == expected_train_len
        assert holdout_df.index[-1] == expected_train_len + F - 1

        # No overlap
        assert (len(train_df) == 0) or (train_df.index[-1] < holdout_df.index[0])

    # Holdouts cover the tail of the HPO dataset
    holdouts_concat = pd.concat([h for _, h in folds], axis=0)
    tail_block = hpo_series.iloc[N_prime - K * F :]
    pd.testing.assert_frame_equal(holdouts_concat.reset_index(drop=True), tail_block.reset_index(drop=True))

    # Last fold ends at N'
    last_holdout = folds[-1][1]
    assert last_holdout.index[-1] == N_prime - 1


@pytest.mark.parametrize("N,K,F", [(100, 3, 33), (60, 2, 30), (40, 4, 10)])
def test_generate_sequential_folds_invalid_too_short(N, K, F):
    """
    The generator requires N >= K*F + 1 (at least one training point in the first fold).
    """
    series = _make_series(N)
    if N >= K * F + 1:
        # Should succeed
        _ = TimeSeriesDataset.generate_sequential_folds(series=series, n_folds=K, forecast_steps=F)
    else:
        with pytest.raises(ValueError):
            _ = TimeSeriesDataset.generate_sequential_folds(series=series, n_folds=K, forecast_steps=F)


@pytest.mark.parametrize("N,K,F", [(300, 3, 50), (240, 4, 30)])
def test_holdout_blocks_are_contiguous_and_non_overlapping(N, K, F):
    """
    The K holdout blocks must be exactly adjacent (contiguous) and non-overlapping,
    covering the last K*F rows of the series.
    """
    series = _make_series(N)
    folds = TimeSeriesDataset.generate_sequential_folds(series=series, n_folds=K, forecast_steps=F)
    # Contiguity: start of block i+1 == end of block i + 1
    for i in range(K - 1):
        a_end = folds[i][1].index[-1]
        b_start = folds[i + 1][1].index[0]
        assert b_start == a_end + 1
    # Global coverage check
    holdouts_concat = pd.concat([h for _, h in folds], axis=0)
    tail_block = series.iloc[N - K * F :]
    pd.testing.assert_frame_equal(holdouts_concat.reset_index(drop=True), tail_block.reset_index(drop=True))
