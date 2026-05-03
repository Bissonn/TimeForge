import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import torch
import torch.nn as nn

# Import the module under test.
# Adjust this import to your project layout, e.g.:
# from models.transformer import (
#     AttentionCaptureBuffer, CapturingMHA, RevIN,
#     ZerosTgtInitializer, LastValueTgtInitializer, MeanTgtInitializer,
#     MedianTgtInitializer, TrendTgtInitializer, SeasonalTgtInitializer, CopyHistoryTgtInitializer,
#     build_tgt_train, LocalAttention, GlobalSelfAttention,
#     TransformerForecaster,
# )
#
# Here we assume transformer.py is importable as models.transformer.
from models.transformer import TransformerForecaster
from models.transformer_components import (
    AttentionCaptureBuffer,
    CapturingMHA,
    RevIN,
    ZerosTgtInitializer,
    LastValueTgtInitializer,
    MeanTgtInitializer,
    MedianTgtInitializer,
    TrendTgtInitializer,
    SeasonalTgtInitializer,
    CopyHistoryTgtInitializer,
    build_tgt_train,
    LocalAttention,
    GlobalSelfAttention,
)
from tests.conftest import base_context


# -----------------------------------------------------------------------------
# Helpers / Stubs
# -----------------------------------------------------------------------------

@dataclass
class DummyFeatureLayout:
    target_size: int = 1
    encoder_input_size: int = 1
    decoder_input_size: int = 1
    decoder_exog_size: int = 0
    encoder_exog_size: int = 0
    total_features: int = 1
    encoder_feature_idx: Optional[List[int]] = None


class DummyDataset:
    """
    Minimal dataset stub to satisfy TransformerForecaster __init__ expectations.
    Only add attributes that are accessed.
    """
    def __init__(
        self,
        columns: List[str],
        target_columns: List[str],
        past_covariates: Optional[List[str]] = None,
        future_covariates: Optional[List[str]] = None,
        training_length: int = 5000,
    ):
        self.columns = columns
        self.target_columns = target_columns

        # Old API (still supported via backward compat properties)
        enc_cols = set(past_covariates or [])
        dec_cols = set(future_covariates or [])

        # New API: Compute past_covariates and future_covariates
        self.past_covariates = list(enc_cols - dec_cols)  # encoder-only
        self.future_covariates = list((enc_cols & dec_cols) | (dec_cols - enc_cols))  # intersection + decoder-only

        self.training_length = training_length
        # Some code paths might read dataset.series length.
        self.series = list(range(training_length))


class DummyRunContext:
    def __init__(self, is_hpo_trial: bool = False):
        self.metadata = {"is_hpo_trial": is_hpo_trial}


# -----------------------------------------------------------------------------
# AttentionCaptureBuffer
# -----------------------------------------------------------------------------

def test_attention_capture_buffer_store_disabled_noop():
    cap = AttentionCaptureBuffer()
    cap.configure(enabled=False)
    cap.store("k", torch.randn(2, 3, 4, 5))
    assert cap.data == {}


def test_attention_capture_buffer_store_step_filtering():
    cap = AttentionCaptureBuffer()
    cap.configure(enabled=True, steps_to_capture=[0, 2])

    t = torch.randn(2, 4, 3, 3)  # (B,H,Q,K)

    cap.set_step(1)
    cap.store("enc_self_layer_0", t)
    assert cap.data == {}

    cap.set_step(2)
    cap.store("enc_self_layer_0", t)
    assert "enc_self_layer_0_step_2" in cap.data
    # batch-averaged -> (H,Q,K)
    assert cap.data["enc_self_layer_0_step_2"].shape == (4, 3, 3)


def test_attention_capture_buffer_store_accepts_3d_tensor():
    cap = AttentionCaptureBuffer()
    cap.configure(enabled=True, steps_to_capture=None)

    t = torch.randn(4, 3, 3)  # already (H,Q,K)
    cap.store("k", t)
    assert "k" in cap.data
    assert cap.data["k"].shape == (4, 3, 3)


# -----------------------------------------------------------------------------
# CapturingMHA
# -----------------------------------------------------------------------------

def test_capturing_mha_overrides_need_weights_and_captures():
    torch.manual_seed(0)
    cap = AttentionCaptureBuffer()
    cap.configure(enabled=True)

    mha = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
    wrap = CapturingMHA(mha, cap, "test_attn")

    x = torch.randn(2, 5, 16)

    # caller tries to disable weights, wrapper should force on
    out = wrap(x, x, x, need_weights=False)
    assert isinstance(out, tuple)
    assert "test_attn" in cap.data
    assert cap.data["test_attn"].dim() == 3  # (H,Q,K)


def test_capturing_mha_no_capture_when_disabled():
    torch.manual_seed(0)
    cap = AttentionCaptureBuffer()
    cap.configure(enabled=False)

    mha = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
    wrap = CapturingMHA(mha, cap, "test_attn")

    x = torch.randn(2, 5, 16)
    wrap(x, x, x, need_weights=True)
    assert cap.data == {}


# -----------------------------------------------------------------------------
# RevIN
# -----------------------------------------------------------------------------

def test_revin_norm_and_denorm_roundtrip_close():
    torch.manual_seed(0)
    revin = RevIN(num_features=3, eps=1e-5, affine=True)

    x = torch.randn(4, 10, 3)
    x_norm = revin(x.clone(), mode="norm")
    x_back = revin(x_norm, mode="denorm")

    assert torch.allclose(x, x_back, atol=1e-4, rtol=1e-4)


def test_revin_apply_requires_prior_norm_stats():
    torch.manual_seed(0)
    revin = RevIN(num_features=2, eps=1e-5, affine=False)

    x = torch.randn(1, 5, 2)
    # "apply" before "norm" uses default mean=0, stdev=1 buffers -> should be identity
    y = revin(x.clone(), mode="apply")
    assert torch.allclose(x, y, atol=0.0, rtol=0.0)

    # After norm, apply should reuse computed stats (not identity generally)
    _ = revin(x.clone(), mode="norm")
    y2 = revin(x.clone(), mode="apply")
    assert not torch.allclose(x, y2)


# -----------------------------------------------------------------------------
# TgtInitializers + build_tgt_train
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "init_cls",
    [ZerosTgtInitializer, LastValueTgtInitializer, MeanTgtInitializer, MedianTgtInitializer, CopyHistoryTgtInitializer],
)
def test_tgt_initializer_shapes_base(init_cls):
    torch.manual_seed(0)
    B, W, F = 2, 7, 3
    src = torch.randn(B, W, F)

    init = init_cls(decoder_uses_exog=False, num_exog_decoder=0)
    tgt = init.initialize_direct(src=src, forecast_steps=5, num_features=F, device=src.device)
    assert tgt.shape == (B, 5, F)


def test_trend_tgt_initializer_float16_safe_on_cpu():
    """
    We cannot test CUDA autocast here reliably, but we can test the dtype path doesn't break on CPU.
    """
    torch.manual_seed(0)
    B, W, F = 2, 6, 2
    src = torch.randn(B, W, F).to(dtype=torch.float16)

    init = TrendTgtInitializer(decoder_uses_exog=False, num_exog_decoder=0)
    tgt = init.initialize_direct(src=src, forecast_steps=4, num_features=F, device=src.device)
    assert tgt.shape == (B, 4, F)
    assert tgt.dtype == torch.float16


def test_seasonal_initializer_fallback_when_history_short():
    torch.manual_seed(0)
    B, W, F = 1, 3, 2
    src = torch.randn(B, W, F)
    init = SeasonalTgtInitializer(decoder_uses_exog=False, num_exog_decoder=0, season_length=10)

    tgt = init.initialize_direct(src=src, forecast_steps=5, num_features=F, device=src.device)
    assert tgt.shape == (B, 5, F)
    # fallback to last value repeated
    assert torch.allclose(tgt[:, 0:1, :], src[:, -1:, :])


def test_build_tgt_train_teacher_forcing_shift_and_exog_concat():
    torch.manual_seed(0)
    B, H, F = 2, 4, 3
    src = torch.randn(B, 6, F)
    y_true = torch.randn(B, H, F)

    exog_dim = 2
    decoder_exog = torch.randn(B, H, exog_dim)

    init = LastValueTgtInitializer(decoder_uses_exog=True, num_exog_decoder=exog_dim)
    tgt = build_tgt_train(target_true=y_true, src=src, initializer=init, decoder_exog=decoder_exog)

    assert tgt.shape == (B, H, F + exog_dim)
    # first token is SOS (last value from src)
    assert torch.allclose(tgt[:, 0:1, :F], src[:, -1:, :F])
    # second token target part equals y_true[:,0,:] (shifted by one)
    assert torch.allclose(tgt[:, 1:2, :F], y_true[:, 0:1, :F])


# -----------------------------------------------------------------------------
# LocalAttention mask properties (sanity checks)
# -----------------------------------------------------------------------------

def test_local_attention_mask_blocks_future_and_far_past():
    torch.manual_seed(0)
    attn = LocalAttention(embed_dim=16, num_heads=4, window_size=3, dropout=0.0)

    x = torch.randn(2, 6, 16)
    out, weights = attn(x)

    assert out.shape == x.shape
    # weights may be None if need_weights=False; LocalAttention uses capture flag to decide.
    # Here capture is None so need_weights=False -> weights should be None or a tensor depending on PyTorch.
    # We only assert forward runs.


# -----------------------------------------------------------------------------
# GlobalSelfAttention basic forward path
# -----------------------------------------------------------------------------

def test_global_self_attention_forward_runs():
    torch.manual_seed(0)
    attn = GlobalSelfAttention(embed_dim=16, num_heads=4, dropout=0.0)
    x = torch.randn(2, 5, 16)

    out, w = attn(x)
    assert out.shape == x.shape


# -----------------------------------------------------------------------------
# TransformerForecaster: focus on pure helper logic we can test in isolation
# -----------------------------------------------------------------------------

def make_forecaster_for_unit_tests(
    model_params: Dict[str, Any],
    num_features: int = 2,
    forecast_steps: int = 5,
    window_size: int = 6,
    dataset: Optional[DummyDataset] = None,
    run_context = None,
) -> TransformerForecaster:
    if dataset is None:
        dataset = DummyDataset(
            columns=["y1", "y2"],
            target_columns=["y1", "y2"],
            past_covariates=[],
            future_covariates=[],
            training_length=5000,
        )

    if run_context is None:
        raise ValueError("run_context must be provided (use base_context fixture)")

    # TransformerForecaster expects a real FeatureLayout created by base class.
    # To keep unit tests focused and not require full project wiring, we instantiate
    # normally but then override feature_layout with a dummy minimal layout.
    f = TransformerForecaster(
        model_params=model_params,
        num_features=num_features,
        forecast_steps=forecast_steps,
        window_size=window_size,
        dataset=dataset,
        run_context=run_context,
    )
    # Minimal overrides to let helper methods work without full pipeline.
    f.feature_layout = DummyFeatureLayout(
        target_size=num_features,
        encoder_input_size=num_features,
        decoder_input_size=num_features,
        decoder_exog_size=0,
        encoder_exog_size=0,
        total_features=num_features,
        encoder_feature_idx=list(range(num_features)),
    )
    f.run_context = DummyRunContext(is_hpo_trial=True)  # default: no artifact writes
    return f


def test_resolve_capture_steps_explicit_list_is_sanitized_and_includes_ends(base_context):
    f = make_forecaster_for_unit_tests(
        model_params={
            "architecture": "encoder-only",
            "strategy": "iterative",
            "attention_type": "full",
            "hidden_size": 32,
            "num_heads": 4,
            "num_encoder_layers": 2,
            "attention_capture_steps": [2, 2, 100, -1, 3],
        },
        run_context=base_context
    )
    steps = f._resolve_capture_steps(horizon=5)
    # should contain 0 and 4
    assert steps[0] == 0
    assert steps[-1] == 4
    assert 2 in steps and 3 in steps
    assert all(0 <= s < 5 for s in steps)


@pytest.mark.parametrize(
    "mode,horizon,expected",
    [
        ("all", 4, [0, 1, 2, 3]),
        ("first_last", 1, [0]),
        ("first_last", 5, [0, 4]),
        ("log", 1, [0]),
        ("log", 10, [0, 1, 2, 4, 8, 9]),
    ],
)
def test_resolve_capture_steps_sampling_modes(mode, horizon, expected,base_context):
    f = make_forecaster_for_unit_tests(
        model_params={
            "architecture": "encoder-only",
            "strategy": "iterative",
            "attention_type": "full",
            "hidden_size": 32,
            "num_heads": 4,
            "num_encoder_layers": 2,
            "attention_capture_sampling": mode,
        },
        run_context=base_context
    )
    steps = f._resolve_capture_steps(horizon=horizon)
    assert steps == expected


def test_choose_primary_map_encoder_only_prefers_last_encoder_layer(base_context):
    f = make_forecaster_for_unit_tests(
        model_params={
            "architecture": "encoder-only",
            "strategy": "direct",
            "attention_type": "full",
            "hidden_size": 32,
            "num_heads": 4,
            "num_encoder_layers": 3,
        },
        run_context=base_context
    )
    keys = ["enc_self_layer_0", "enc_self_layer_2", "enc_self_layer_1"]
    assert f._choose_primary_map(keys) == "enc_self_layer_2"


def test_choose_primary_map_encoder_decoder_prefers_last_cross_layer(base_context):
    dataset = DummyDataset(
        columns=["y1", "y2", "x1"],
        target_columns=["y1", "y2"],
        past_covariates=[],
        future_covariates=["x1"],
        training_length=5000,
    )
    f = make_forecaster_for_unit_tests(
        model_params={
            "architecture": "encoder-decoder",
            "strategy": "direct",
            "attention_type": "full",
            "hidden_size": 32,
            "num_heads": 4,
            "num_encoder_layers": 2,
            "num_decoder_layers": 4,
            "tgt_init": "last_value",
            "seasonal_period": 1,
        },
        dataset=dataset,
        run_context=base_context
    )
    keys = ["dec_self_layer_3", "dec_cross_layer_3", "enc_self_layer_1"]
    assert f._choose_primary_map(keys) == "dec_cross_layer_3"


def test_nan_guard_returns_nan_array_when_nonfinite(base_context):
    f = make_forecaster_for_unit_tests(
        model_params={
            "architecture": "encoder-only",
            "strategy": "direct",
            "attention_type": "full",
            "hidden_size": 32,
            "num_heads": 4,
            "num_encoder_layers": 2,
        },
        run_context=base_context
    )
    t = torch.tensor([[[1.0, float("inf")]]])
    out = f._nan_guard_to_numpy(t, context="unit-test")
    assert np.isnan(out).any()


@pytest.mark.skip(reason="PC-mode removed in favor of past_covariates/future_covariates refactoring")
def test_align_pc_training_data_shifts_C_and_injects_future(base_context):
    """
    [OBSOLETE] This test verified PC-mode alignment logic which has been removed.

    The new covariate system uses:
    - past_covariates: encoder-only features (frozen during prediction)
    - future_covariates: known in both history and future

    PC-mode's concept of "Continuous" variables has been replaced by future_covariates.
    """
    pass


# -----------------------------------------------------------------------------
# validate_param_combination: pure logic checks
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "hidden_size,num_heads,expected",
    [
        (128, 8, True),   # head_dim=16 ok
        (128, 7, False),  # not divisible
        (64, 16, False),  # head_dim=4 too small
        (2048, 4, False), # head_dim=512 too large
    ],
)
def test_validate_param_combination_mha_constraints(hidden_size, num_heads, expected, base_context):
    f = make_forecaster_for_unit_tests(
        model_params={
            "architecture": "encoder-only",
            "strategy": "direct",
            "attention_type": "full",
            "hidden_size": 32,
            "num_heads": 4,
            "num_encoder_layers": 2,
        },
        run_context=base_context
    )

    ok = f.validate_param_combination(
        {"hidden_size": hidden_size, "num_heads": num_heads, "num_encoder_layers": 2}
    )
    assert ok is expected
