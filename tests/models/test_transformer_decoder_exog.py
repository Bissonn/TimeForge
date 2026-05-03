# tests/models/test_transformer_decoder_exog.py
import pytest
import torch
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from models.transformer import TransformerForecaster, TransformerModel
from utils.dataset import TimeSeriesDataset

# WINDOW_SIZE constant from conftest
try:
    from tests.models.conftest import WINDOW_SIZE
except ImportError:
    WINDOW_SIZE = 8


# Helper functions
def make_forecaster(num_features, forecast_steps, num_enc_exog, num_dec_exog, cfg_overrides, run_context, window_size=WINDOW_SIZE):
    """
    Helper to instantiate TransformerForecaster with a mocked dataset.
    Reconstructs the dataset column structure based on desired feature counts.
    """
    cfg = {**cfg_overrides}

    # 1. Create Mock Dataset with correct column structure
    mock_ds = MagicMock(spec=TimeSeriesDataset)
    mock_ds.target_columns = [f'tgt_{i}' for i in range(num_features)]

    # Old API (for backward compatibility)
    enc_cols = [f'enc_{i}' for i in range(num_enc_exog)]
    dec_cols = [f'dec_{i}' for i in range(num_dec_exog)]

    # New API: Compute past_covariates and future_covariates
    # For tests: enc_cols and dec_cols are disjoint (typical case)
    mock_ds.past_covariates = enc_cols  # encoder-only
    mock_ds.future_covariates = dec_cols  # decoder-only (treated as future_covariates)

    # Backward-compatible properties (for code that still uses old API)
    mock_ds.past_covariates = enc_cols
    mock_ds.future_covariates = dec_cols

    mock_ds.columns = mock_ds.target_columns + enc_cols + dec_cols

    return TransformerForecaster(
        model_params=cfg,
        num_features=num_features,
        forecast_steps=forecast_steps,
        window_size=window_size,
        dataset=mock_ds,
        run_context=run_context
    )


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize("num_dec_exog", [0, 2])
def test_encdec_direct_decoder_exog_variants(base_transformer_config, device, num_dec_exog, enc_only_dataset, base_context,
                                             full_dataset):
    """Encoder-decoder, strategy=direct: tgt provided, last-dim F or F + E_dec, error if exog missing."""
    B, W, F, H = 2, WINDOW_SIZE, 3, 5
    E_enc = 4  # encoder exog
    E_dec = num_dec_exog  # decoder exog

    cfg = {
        **base_transformer_config,
        "architecture": "encoder-decoder",
        "strategy": "direct",
        "tgt_init": "zeros",
        "dropout": 0.0,
    }

    forecaster = make_forecaster(F, H, E_enc, E_dec, cfg, run_context=base_context)
    forecaster.fitted = True
    forecaster.model.eval()

    # NEW API: encoder input includes targets + past_covariates + future_covariates
    # In our mock: past_covariates = E_enc, future_covariates = E_dec
    enc_input_size = F + E_enc + E_dec
    x = torch.randn(B, W, enc_input_size, device=device)

    if E_dec > 0:
        future_exog = torch.randn(B, H, E_dec, device=device)
        preds = forecaster._internal_predict(x, future_exog_tensor=future_exog)
        assert preds.shape == (B, H, F)

        initializer = forecaster.tgt_initializer
        tgt_base = initializer.initialize_direct(x, H, F, device, future_exog_tensor=future_exog)
        assert tgt_base.shape == (B, H, F + E_dec)

        with pytest.raises(ValueError, match="requires.*future_exog_tensor"):
            forecaster._internal_predict(x, future_exog_tensor=None)
    else:
        preds = forecaster._internal_predict(x, future_exog_tensor=None)
        assert preds.shape == (B, H, F)


def test_encdec_direct_decoder_exog_missing_raises(base_transformer_config, device, base_context):
    """Encoder-decoder, strategy=direct: missing/short future_exog_tensor raises ValueError."""
    B, W, F, H = 1, 6, 2, 4
    E_enc = 3
    E_dec = 2  # Decoder expects 2 exog features

    cfg = {
        **base_transformer_config,
        "architecture": "encoder-decoder",
        "strategy": "direct",
    }

    forecaster = make_forecaster(F, H, E_enc, E_dec, cfg, run_context=base_context, window_size=W)
    forecaster.fitted = True

    # NEW API: encoder includes future_covariates
    enc_input_size = F + E_enc + E_dec
    x = torch.randn(B, W, enc_input_size, device=device)

    with pytest.raises(ValueError, match="requires 'future_exog_tensor'"):
        _ = forecaster._internal_predict(x, future_exog_tensor=None)

    bad_future_exog = torch.randn(B, H - 1, E_dec, device=device)
    with pytest.raises(ValueError, match="insufficient time steps|too short"):
        _ = forecaster._internal_predict(x, future_exog_tensor=bad_future_exog)


@pytest.mark.parametrize("num_dec_exog", [0, 2])
def test_encdec_iterative_decoder_exog_variants(base_transformer_config, device, num_dec_exog, mocker, base_context):
    """Encoder-decoder, strategy=iterative: growing tgt, exog preserved."""
    B, W, F, H = 2, WINDOW_SIZE, 3, 5
    E_enc = 4
    E_dec = num_dec_exog

    cfg = {
        **base_transformer_config,
        "architecture": "encoder-decoder",
        "strategy": "iterative",
        "tgt_init": "zeros",
        "dropout": 0.0,
    }

    forecaster = make_forecaster(F, H, E_enc, E_dec, cfg, run_context=base_context)
    forecaster.fitted = True
    forecaster.model.eval()

    # NEW API: encoder includes future_covariates
    enc_input_size = F + E_enc + E_dec
    x = torch.randn(B, W, enc_input_size, device=device)

    if E_dec > 0:
        future_exog = torch.randn(B, H, E_dec, device=device)
        decode_spy = mocker.spy(forecaster.model, "decode")

        preds = forecaster._predict_iterative(x, future_exog_tensor=future_exog)
        assert preds.shape == (B, H, F)
        assert decode_spy.call_count == H

        for i in range(H):
            call_args = decode_spy.call_args_list[i]
            # 'decode' takes 'tgt' as the first positional argument
            tgt = call_args.args[0]
            assert tgt is not None
            expected_len = i + 1
            assert tgt.shape == (B, expected_len, F + E_dec)

        with pytest.raises(ValueError, match="Model requires 'future_exog_tensor' for decoder"):
            forecaster._predict_iterative(x, future_exog_tensor=None)
    else:
        preds = forecaster._predict_iterative(x, future_exog_tensor=None)
        assert preds.shape == (B, H, F)


def test_iterative_behavior_when_encoder_input_has_exog(base_transformer_config, device, base_context):
    """
    If encoder has exogenous features (E_enc > 0) but decoder has none (E_dec=0),
    iterative mode should NOT require future_exog_tensor.
    """
    B, W, F, H = 2, 8, 2, 3
    E_enc = 5
    E_dec = 0

    cfg = {
        **base_transformer_config,
        "architecture": "encoder-decoder",
        "strategy": "iterative",
    }
    forecaster = make_forecaster(F, H, E_enc, E_dec, cfg, run_context=base_context, window_size=W)
    forecaster.fitted = True

    forecaster.model = MagicMock(spec=TransformerModel)
    forecaster.model.architecture = "encoder-decoder"

    # Since decode side effect returns tensor of size F, output_head can be identity
    forecaster.model.output_head = MagicMock(side_effect=lambda x: x)

    forecaster.model.encode.return_value = torch.randn(B, W, cfg['hidden_size'], device=device)

    # Simulating decoder output (must be a tensor)
    def decode_side_effect(tgt, memory, **kwargs):
        # The real decode method returns (B, T, num_features), NOT hidden_size
        return torch.randn(tgt.shape[0], tgt.shape[1], F, device=device)

    forecaster.model.decode.side_effect = decode_side_effect

    forecaster.model.denormalize_output.side_effect = lambda x: x

    # NEW API: encoder includes future_covariates (but E_dec=0 here)
    enc_input_size = F + E_enc + E_dec
    x = torch.randn(B, W, enc_input_size, device=device)

    # Should NOT raise
    _ = forecaster._predict_iterative(x, future_exog_tensor=None)
    assert forecaster.model.decode.call_count == H


def test_encdec_iterative_no_decoder_exog_ignores_future_exog(base_transformer_config, device, base_context):
    """
    If decoder does not use exog (E_dec=0) but a future_exog_tensor IS passed,
    ensure it is ignored (tgt has only F features).
    """
    B, W, F, H = 1, 8, 2, 3
    E_enc = 4
    E_dec = 0
    # NEW API: encoder includes future_covariates (but E_dec=0 here)
    enc_input_size = F + E_enc + E_dec

    cfg = {
        **base_transformer_config,
        "architecture": "encoder-decoder",
        "strategy": "iterative",
    }

    forecaster = make_forecaster(F, H, E_enc, E_dec, cfg, run_context=base_context, window_size=W)
    forecaster.fitted = True

    forecaster.model = MagicMock(spec=TransformerModel)
    forecaster.model.architecture = "encoder-decoder"

    forecaster.model.output_head = MagicMock(side_effect=lambda x: x)

    forecaster.model.encode.return_value = torch.randn(B, W, cfg['hidden_size'], device=device)

    def decode_side_effect(tgt, memory, **kwargs):
        # Real decoder returns (B, T, num_features)
        return torch.randn(tgt.shape[0], tgt.shape[1], F, device=device)

    forecaster.model.decode.side_effect = decode_side_effect

    forecaster.model.denormalize_output.side_effect = lambda x: x

    x = torch.randn(B, W, enc_input_size, device=device)
    future_exog_ignored = torch.randn(B, H, 999, device=device)

    _ = forecaster._predict_iterative(x, future_exog_tensor=future_exog_ignored)

    # Check first call to decode
    args = forecaster.model.decode.call_args.args
    tgt = args[0]
    # tgt shape should be (B, step, F) -> no extra exog features
    assert tgt.shape[-1] == F


# =============================================================================
# PREDICT()-LEVEL TESTS: FUTURE_EXOG WITH ENCODER + DECODER EXOG
# =============================================================================

def test_predict_uses_only_decoder_exog_when_both_encoder_and_decoder_provided(base_transformer_config, device, base_context):
    """
    Verify predict() filters future_exog to only pass decoder columns to _internal_predict.
    """
    B = 1
    W = WINDOW_SIZE
    F = 1
    H = 5
    E_enc = 2
    E_dec = 2

    cfg = {
        **base_transformer_config,
        "architecture": "encoder-decoder",
        "strategy": "direct",
        "tgt_init": "zeros",
        "dropout": 0.0,
    }

    forecaster = make_forecaster(F, H, E_enc, E_dec, cfg, run_context=base_context, window_size=W)
    forecaster.fitted = True

    # Dummy preprocessor
    class DummyPreprocessor:
        def __init__(self, target_columns):
            self.target_columns = target_columns

        def transform(self, df, allow_subset: bool = False):
            return df

        def inverse_transforms(self, df, start_after=None):
            return df

    forecaster.preprocessor = DummyPreprocessor(target_columns=[f"tgt_{i}" for i in range(F)])

    # Spy on _predict_direct (since strategy is direct)
    captured = {}

    def fake_internal_predict(input_tensor, future_exog_tensor=None):
        captured["future_exog_tensor"] = future_exog_tensor
        return np.zeros((B, H, F), dtype=np.float32)

    forecaster._predict_direct = fake_internal_predict
    # Also patch iterative just in case strategy changes or logic shares dispatch
    forecaster._predict_iterative = fake_internal_predict

    hist_index = pd.date_range("2023-01-01", periods=W, freq="D")
    # NEW API: input includes targets + past_covariates + future_covariates (historical values)
    # Based on make_forecaster: past_covariates=enc_cols, future_covariates=dec_cols
    past_cov_cols = [f'enc_{i}' for i in range(E_enc)]
    future_cov_cols = [f'dec_{i}' for i in range(E_dec)]
    hist_columns = [f"tgt_{i}" for i in range(F)] + past_cov_cols + future_cov_cols
    input_data = pd.DataFrame(np.random.randn(W, len(hist_columns)), index=hist_index, columns=hist_columns)

    fut_index = pd.date_range(hist_index[-1] + pd.Timedelta(days=1), periods=H, freq="D")
    # Future exog only needs future_covariates (decoder columns in this context)
    fut_columns = future_cov_cols + ["junk_col"]
    future_exog = pd.DataFrame(np.random.randn(H, len(fut_columns)), index=fut_index, columns=fut_columns)

    _ = forecaster.predict(input_data, future_exog=future_exog)

    fe = captured.get("future_exog_tensor")
    assert fe is not None
    assert fe.shape == (B, H, len(forecaster.future_covariates))


def test_predict_uses_decoder_exog_when_only_decoder_exog_configured(base_transformer_config, device, base_context):
    """
    Verify predict() handles pure decoder exog correctly.
    """
    B = 1
    W = WINDOW_SIZE
    F = 1
    H = 4
    E_enc = 0
    E_dec = 3

    cfg = {
        **base_transformer_config,
        "architecture": "encoder-decoder",
        "strategy": "direct",
        "tgt_init": "zeros",
        "dropout": 0.0,
    }

    forecaster = make_forecaster(F, H, E_enc, E_dec, cfg, run_context=base_context, window_size=W)
    forecaster.fitted = True

    class DummyPreprocessor:
        def __init__(self, target_columns):
            self.target_columns = target_columns

        def transform(self, df, allow_subset: bool = False):
            return df

        def inverse_transforms(self, df, start_after=None):
            return df

    forecaster.preprocessor = DummyPreprocessor(target_columns=[f"tgt_{i}" for i in range(F)])

    captured = {}

    def fake_internal_predict(input_tensor, future_exog_tensor=None):
        captured["future_exog_tensor"] = future_exog_tensor
        return np.zeros((B, H, F), dtype=np.float32)

    forecaster._predict_direct = fake_internal_predict
    forecaster._predict_iterative = fake_internal_predict

    hist_index = pd.date_range("2023-02-01", periods=W, freq="D")
    # NEW API: input includes targets + past_covariates + future_covariates (historical values)
    # E_enc=0, so only targets + future_covariates
    # Based on make_forecaster: past_covariates=enc_cols, future_covariates=dec_cols
    past_cov_cols = [f'enc_{i}' for i in range(E_enc)]
    future_cov_cols = [f'dec_{i}' for i in range(E_dec)]
    hist_columns = [f"tgt_{i}" for i in range(F)] + past_cov_cols + future_cov_cols
    input_data = pd.DataFrame(np.random.randn(W, len(hist_columns)), index=hist_index, columns=hist_columns)

    fut_index = pd.date_range(hist_index[-1] + pd.Timedelta(days=1), periods=H, freq="D")
    future_exog = pd.DataFrame(np.random.randn(H, len(future_cov_cols)), index=fut_index,
                               columns=future_cov_cols)

    _ = forecaster.predict(input_data, future_exog=future_exog)

    fe = captured.get("future_exog_tensor")
    assert fe is not None
    assert fe.shape == (B, H, len(forecaster.future_covariates))
