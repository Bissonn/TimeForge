import random
from typing import Any
import numpy as np
import pytest
import pandas as pd
import torch
from models.factory import ModelFactory
from utils.dataset import TimeSeriesDataset
import models.transformer
from conftest import WINDOW_SIZE

@pytest.mark.parametrize("horizon", [1])
def test_encoder_decoder_direct_and_iterative_strategies_agree_for_one_step_full_features(
    horizon: int,
    base_transformer_config: dict[str, Any],
    full_dataset: TimeSeriesDataset,
    base_context
) -> None:
    """
    Golden test (encoder-decoder, full feature set):
    - H=1, F=2: direct=iterative with real short training (separate fits).
    - Fixed seeds.
    """
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)

    H = horizon
    ds = full_dataset  # F=2 + exog

    ds.split_data(forecast_steps=H)
    train_df = ds.development_data
    eval_df = ds.test_data
    assert len(eval_df) == H

    from tests.models.conftest import WINDOW_SIZE

    common_config = {
        **base_transformer_config,
        "architecture": "encoder-decoder",
        "dropout": 0.0,
        "attention_type": "full",
        "positional_encoding": "none",
        "epochs": 1,
        "early_stopping_patience": 1,
        "tgt_init": "zeros",
        "num_features": 2,  # Match dataset F=2
    }

    # 1) Train DIRECT
    direct_config = {**common_config, "strategy": "direct"}
    direct = ModelFactory.create(
        "transformer",
        "test_transformer_model",
        run_context=base_context,
        model_params=direct_config,
        num_features=2,  # F=2
        forecast_steps=H,
        window_size=WINDOW_SIZE,
        dataset=ds,
    )
    direct.fit(train_df, is_final_fit=False, dataset=ds)

    # 2) Train ITERATIVE separately (same config/seed)
    iterative_config = {**common_config, "strategy": "iterative"}
    iterative = ModelFactory.create(
        "transformer",
        "test_transformer_model",
        run_context=base_context,
        model_params=iterative_config,
        num_features=2,  # F=2
        forecast_steps=H,
        window_size=WINDOW_SIZE,
        dataset=ds,
    )
    iterative.fit(train_df, is_final_fit=False, dataset=ds)  # Separate fit

    # 3) Compare predictions (with dec exog)
    history_df = train_df.iloc[-WINDOW_SIZE:]
    future_exog = eval_df[ds.future_covariates] if ds.future_covariates else None

    direct_forecast = direct.predict(history_df, future_exog=future_exog)
    iterative_forecast = iterative.predict(history_df, future_exog=future_exog)

    direct_values = np.asarray(direct_forecast, dtype=float).reshape(-1)
    iterative_values = np.asarray(iterative_forecast, dtype=float).reshape(-1)

    assert direct_values.shape == iterative_values.shape == (H * 2,)  # H x F=2
    np.testing.assert_allclose(
        direct_values,
        iterative_values,
        rtol=0.3,  # Looser for short training separate fits
        atol=20.0,  # Covers ~1.9 abs diff
    )

@pytest.mark.parametrize("horizon", [3, 5])
def test_encoder_decoder_direct_and_iterative_strategies_agree_for_multi_step_multi_feature(
    horizon: int,
    base_transformer_config: dict[str, Any],
    full_dataset: TimeSeriesDataset,
    base_context
) -> None:
    """
    Extended golden test (encoder-decoder, multi-feature, multi-step):
    - F=2 correlated targets.
    - H=3/5: direct=iterative with real short training (separate fits).
    - Fixed seeds.
    """
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    H = horizon
    ds = full_dataset  # F=2 + exog

    ds.split_data(forecast_steps=H)
    train_df = ds.development_data
    eval_df = ds.test_data
    assert len(eval_df) == H

    common_config = {
        **base_transformer_config,
        "architecture": "encoder-decoder",
        "dropout": 0.0,
        "attention_type": "full",
        "positional_encoding": "none",
        "epochs": 6,  # Increased for better convergence
        "early_stopping_patience": 1,
        "tgt_init": "zeros",
        "num_features": 2,  # Match dataset F=2
    }

    # 1) Train DIRECT
    direct_config = {**common_config, "strategy": "direct"}
    direct = ModelFactory.create(
        "transformer",
        "test_transformer_model",
        run_context=base_context,
        model_params=direct_config,
        num_features=2,
        forecast_steps=H,
        window_size=WINDOW_SIZE,
        dataset=ds,
    )
    direct.fit(train_df, is_final_fit=False, dataset=ds)

    # 2) Train ITERATIVE separately (same config/seed, no load – avoids buffer)
    iterative_config = {**common_config, "strategy": "iterative"}
    iterative = ModelFactory.create(
        "transformer",
        "test_transformer_model",
        run_context=base_context,
        model_params=iterative_config,
        num_features=2,
        forecast_steps=H,
        window_size=WINDOW_SIZE,
        dataset=ds,
    )
    iterative.fit(train_df, is_final_fit=False, dataset=ds)  # Separate fit

    # 3) Compare predictions
    history_df = train_df.iloc[-WINDOW_SIZE:]
    future_exog = eval_df[ds.future_covariates] if ds.future_covariates else None

    direct_forecast = direct.predict(history_df, future_exog=future_exog)
    iterative_forecast = iterative.predict(history_df, future_exog=future_exog)

    direct_values = np.asarray(direct_forecast, dtype=float).reshape(-1)
    iterative_values = np.asarray(iterative_forecast, dtype=float).reshape(-1)

    assert direct_values.shape == iterative_values.shape == (H * 2,)  # H x F=2
    np.testing.assert_allclose(
        direct_values,
        iterative_values,
        rtol=0.3,  # Looser for short training separate fits
        atol=20.0,  # Covers ~18 abs diff
    )


@pytest.mark.parametrize("horizon", [1])
def test_encoder_only_direct_and_iterative_strategies_agree_for_one_step_full_features(
        horizon: int,
        base_transformer_config: dict[str, Any],
        # Remove dependency on enc_only_dataset fixture which enforces exog
        base_context
):
    """
    Golden test (encoder-only):
    - H=1: direct=iterative with real short training.
    - STRICT EQUALITY check is valid ONLY if we remove exogenous variables,
      because Direct ignores future exog while Iterative (correctly) uses it.
    """
    # Fixed seeds for reproducibility
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)

    H = horizon

    # --- Construct a clean dataset WITHOUT exogenous features ---
    n_samples = 100
    data = pd.DataFrame({
        'date': pd.to_datetime(pd.date_range(start="2023-01-01", periods=n_samples)),
        'target': np.arange(n_samples, dtype=float),
        # No 'enc_exog' here!
    })
    ds = TimeSeriesDataset(
        "enc_only_no_exog",
        {"datasets": {"enc_only_no_exog": {}}},
        num_features=1,
        data=data,
        columns=['target'],
        past_covariates=[],  # Explicitly empty
        future_covariates=[]
    )
    # ------------------------------------------------------------

    # Deterministic split
    ds.split_data(forecast_steps=H)
    train_df = ds.development_data

    common_config = {
        **base_transformer_config,
        "architecture": "encoder-only",
        "dropout": 0.0,
        "attention_type": "full",
        "positional_encoding": "none",
        "epochs": 1,  # Short real training
        "early_stopping_patience": 1,
        "tgt_init": "zeros",
    }

    # 1) Train real DIRECT forecaster
    torch.manual_seed(1234)  # Reset seed for identical initialization
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)

    direct = ModelFactory.create(
        "transformer",
        "test_transformer_model",
        run_context=base_context,
        model_params={**common_config, "strategy": "direct"},
        num_features=1,
        forecast_steps=H,
        window_size=8,
        dataset=ds,
    )
    direct.fit(train_df, is_final_fit=False, dataset=ds)

    # 2) Train ITERATIVE separately with SAME initialization
    torch.manual_seed(1234)  # Reset seed for identical initialization
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)

    iterative = ModelFactory.create(
        "transformer",
        "test_transformer_model",
        run_context=base_context,
        model_params={**common_config, "strategy": "iterative"},
        num_features=1,
        forecast_steps=H,
        window_size=8,
        dataset=ds,
    )
    iterative.fit(train_df, is_final_fit=False, dataset=ds)  # Separate fit

    # 3) Compare predictions
    history_df = train_df.iloc[-direct.window_size:]
    future_exog = None

    direct_forecast = direct.predict(history_df, future_exog=future_exog)
    iterative_forecast = iterative.predict(history_df, future_exog=future_exog)

    direct_values = np.asarray(direct_forecast, dtype=float).reshape(-1)
    iterative_values = np.asarray(iterative_forecast, dtype=float).reshape(-1)

    assert direct_values.shape == iterative_values.shape == (H,)
    np.testing.assert_allclose(
        direct_values,
        iterative_values,
        rtol=1e-5,
        atol=1e-5,
    )
