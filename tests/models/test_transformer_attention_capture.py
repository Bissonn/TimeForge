
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch

# NOTE:
# These tests assume your project exposes the following symbols from models.transformer:
# - AttentionCaptureBuffer, CapturingMHA
# - LocalAttention, GlobalSelfAttention
# - TransformerModel, TransformerForecaster
#
# If your import path differs, adjust the import below accordingly.
from models.transformer import (
    AttentionCaptureBuffer,
    CapturingMHA,
    LocalAttention,
    GlobalSelfAttention,
    TransformerModel,
    TransformerForecaster,
)


# -----------------------------
# Helpers
# -----------------------------

def get_artifact_path(self, category: str, filename: str, extension: str) -> str:
    out_dir = self.base_dir / category
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"{filename}.{extension}")


def _rand_src(B=2, W=8, D=6, device="cpu"):
    torch.manual_seed(0)
    return torch.randn(B, W, D, device=device)


def _rand_tgt(B=2, H=5, D=3, device="cpu"):
    torch.manual_seed(1)
    return torch.randn(B, H, D, device=device)


# -----------------------------
# Unit tests: buffer + wrappers
# -----------------------------

def test_attention_capture_buffer_store_batch_average_and_step_suffix():
    cap = AttentionCaptureBuffer()
    cap.configure(enabled=True, steps_to_capture=None)

    attn = torch.randn(4, 2, 7, 7)  # (B,H,Q,K)
    cap.set_step(3)
    cap.store("enc_self_layer_0", attn)

    assert "enc_self_layer_0_step_3" in cap.data
    stored = cap.data["enc_self_layer_0_step_3"]
    assert stored.shape == (2, 7, 7)  # batch-averaged


def test_attention_capture_buffer_step_filtering():
    cap = AttentionCaptureBuffer()
    cap.configure(enabled=True, steps_to_capture=[0, 2, 4])

    attn = torch.randn(1, 2, 3, 3)
    cap.set_step(1)
    cap.store("k", attn)
    assert cap.data == {}

    cap.set_step(2)
    cap.store("k", attn)
    assert "k_step_2" in cap.data


def test_capturing_mha_overrides_need_weights_and_stores_only_when_enabled():
    cap = AttentionCaptureBuffer()
    mha = torch.nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
    wrapped = CapturingMHA(mha, cap, "enc_self_layer_0")

    x = torch.randn(2, 5, 8)

    # Disabled: should not store anything, and need_weights forced False.
    cap.configure(enabled=False, steps_to_capture=None)
    out, weights = wrapped(x, x, x, need_weights=True, average_attn_weights=False)
    assert weights is None
    assert cap.data == {}

    # Enabled: should store weights (batch-avg to (H,Q,K))
    cap.configure(enabled=True, steps_to_capture=None)
    out, weights = wrapped(x, x, x)  # caller doesn't need to pass need_weights
    assert weights is not None
    assert "enc_self_layer_0" in cap.data
    assert cap.data["enc_self_layer_0"].dim() == 3  # (H,Q,K)


def test_global_self_attention_capture_writes_expected_key_and_shape():
    cap = AttentionCaptureBuffer()
    cap.configure(enabled=True, steps_to_capture=None)

    attn = GlobalSelfAttention(embed_dim=8, num_heads=2, dropout=0.0)
    attn.capture = cap
    attn.key_prefix = "enc_self_layer_0"

    x = torch.randn(2, 6, 8)
    out, weights = attn(x)

    assert out.shape == (2, 6, 8)
    assert weights is not None
    assert "enc_self_layer_0" in cap.data
    assert cap.data["enc_self_layer_0"].shape == (2, 6, 6)


def test_local_attention_capture_and_mask_cache_reuse():
    cap = AttentionCaptureBuffer()
    cap.configure(enabled=True, steps_to_capture=None)

    attn = LocalAttention(embed_dim=8, num_heads=2, window_size=3, dropout=0.0)
    attn.capture = cap
    attn.key_prefix = "enc_self_layer_0"

    x = torch.randn(2, 6, 8)

    # first call builds mask
    out1, w1 = attn(x)
    mask_id_1 = id(attn._cached_mask)

    # second call same seq_len should reuse cached mask object
    out2, w2 = attn(x)
    mask_id_2 = id(attn._cached_mask)

    assert mask_id_1 == mask_id_2
    assert "enc_self_layer_0" in cap.data
    assert cap.data["enc_self_layer_0"].shape == (2, 6, 6)


# -----------------------------
# Integration tests: TransformerModel capture
# -----------------------------

@pytest.mark.parametrize("attention_type", ["full", "local"])
def test_transformer_model_encoder_only_direct_captures_all_encoder_layers(attention_type):
    model = TransformerModel(
        encoder_input_size=6,
        decoder_input_size=3,
        num_features=3,
        forecast_steps=4,
        window_size=8,
        hidden_size=8,
        num_heads=2,
        num_encoder_layers=3,
        num_decoder_layers=2,
        dim_ff_multiplier=2.0,
        dropout=0.0,
        architecture="encoder-only",
        readout="last",
        attention_type=attention_type,
        attention_window_size=3,
        use_revin=False,
        norm_first=True,
        activation="gelu",
    )

    src = _rand_src(B=2, W=8, D=6)

    model.attn_capture.configure(enabled=True, steps_to_capture=None)
    _ = model(src)

    keys = set(model.attn_capture.data.keys())
    # Should have per-layer deterministic keys
    assert {"enc_self_layer_0", "enc_self_layer_1", "enc_self_layer_2"}.issubset(keys)


def test_transformer_model_encoder_only_iterative_style_step_suffixes():
    model = TransformerModel(
        encoder_input_size=6,
        decoder_input_size=3,
        num_features=3,
        forecast_steps=1,  # encoder-only iterative model typically has internal horizon=1
        window_size=8,
        hidden_size=8,
        num_heads=2,
        num_encoder_layers=2,
        num_decoder_layers=1,
        dim_ff_multiplier=2.0,
        dropout=0.0,
        architecture="encoder-only",
        readout="last",
        attention_type="full",
        attention_window_size=3,
        use_revin=False,
        norm_first=True,
        activation="gelu",
    )

    src = _rand_src(B=2, W=8, D=6)

    model.attn_capture.configure(enabled=True, steps_to_capture=[0, 2])

    for step in [0, 1, 2]:
        model.attn_capture.set_step(step)
        _ = model(src)

    keys = set(model.attn_capture.data.keys())
    # step=1 should be filtered out (not in steps_to_capture)
    assert any(k.startswith("enc_self_layer_0_step_0") for k in keys)
    assert not any(k.startswith("enc_self_layer_0_step_1") for k in keys)
    assert any(k.startswith("enc_self_layer_0_step_2") for k in keys)


@pytest.mark.parametrize("attention_type", ["full", "local"])
def test_transformer_model_encoder_decoder_direct_captures_encoder_and_decoder(attention_type):
    model = TransformerModel(
        encoder_input_size=6,
        decoder_input_size=3,
        num_features=3,
        forecast_steps=5,
        window_size=8,
        hidden_size=8,
        num_heads=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_ff_multiplier=2.0,
        dropout=0.0,
        architecture="encoder-decoder",
        attention_type=attention_type,
        attention_window_size=3,
        use_revin=False,
        norm_first=True,
        activation="gelu",
    )

    src = _rand_src(B=2, W=8, D=6)
    tgt = _rand_tgt(B=2, H=5, D=3)

    model.attn_capture.configure(enabled=True, steps_to_capture=None)
    _ = model(src, tgt=tgt)

    keys = set(model.attn_capture.data.keys())
    # Encoder capture
    assert {"enc_self_layer_0", "enc_self_layer_1"}.issubset(keys)
    # Decoder capture (wrapped MHAs)
    assert {"dec_self_layer_0", "dec_self_layer_1"}.issubset(keys)
    assert {"dec_cross_layer_0", "dec_cross_layer_1"}.issubset(keys)


# -----------------------------
# Unit tests: Forecaster helpers (sampling, primary map, context manager)
# -----------------------------

def test_resolve_capture_steps_explicit_list_sanitization():
    # Create a forecaster object without running full init
    f = TransformerForecaster.__new__(TransformerForecaster)
    f.model_params = {"attention_capture_steps": ["1", 3, 999]}  # includes coercion + out of range
    # horizon=5 => valid steps: 1,3 plus ensure 0 and 4 included
    steps = TransformerForecaster._resolve_capture_steps(f, horizon=5)
    assert steps == [0, 1, 3, 4]


@pytest.mark.parametrize(
    "mode,horizon,expected",
    [
        ("all", 4, [0, 1, 2, 3]),
        ("first_last", 1, [0]),
        ("first_last", 5, [0, 4]),
        ("log", 1, [0]),
        ("log", 2, [0, 1]),
        ("log", 10, [0, 1, 2, 4, 8, 9]),
    ],
)
def test_resolve_capture_steps_sampling_modes(mode, horizon, expected):
    f = TransformerForecaster.__new__(TransformerForecaster)
    f.model_params = {"attention_capture_sampling": mode}
    steps = TransformerForecaster._resolve_capture_steps(f, horizon=horizon)
    assert steps == expected


def test_choose_primary_map_encoder_only_prefers_last_encoder_layer():
    f = TransformerForecaster.__new__(TransformerForecaster)
    f.model_params = {"architecture": "encoder-only", "num_encoder_layers": 3}
    keys = ["enc_self_layer_0", "enc_self_layer_1", "enc_self_layer_2"]
    assert TransformerForecaster._choose_primary_map(f, keys) == "enc_self_layer_2"


def test_choose_primary_map_encoder_decoder_prefers_last_decoder_cross_layer():
    f = TransformerForecaster.__new__(TransformerForecaster)
    f.model_params = {"architecture": "encoder-decoder", "num_decoder_layers": 4}
    keys = ["dec_cross_layer_0", "dec_cross_layer_3_step_2"]
    assert TransformerForecaster._choose_primary_map(f, keys) == "dec_cross_layer_3_step_2"


def test_capture_attention_weights_context_manager_is_noop_without_capture():
    f = TransformerForecaster.__new__(TransformerForecaster)
    f.model = None
    with TransformerForecaster.capture_attention_weights(f):
        pass  # should not raise


def test_capture_attention_weights_context_manager_toggles_enabled_state():
    f = TransformerForecaster.__new__(TransformerForecaster)
    f.model = SimpleNamespace(attn_capture=AttentionCaptureBuffer())
    assert f.model.attn_capture.enabled is False

    with TransformerForecaster.capture_attention_weights(f):
        assert f.model.attn_capture.enabled is True

    assert f.model.attn_capture.enabled is False


def test_save_attention_to_disk_writes_npz_and_metadata(tmp_path: Path, base_context):
    """save_attention_to_disk() should emit an NPZ file plus JSON metadata sidecar."""
    f = TransformerForecaster.__new__(TransformerForecaster)
    f.model_params = {
        "model": "transformer",
        "architecture": "encoder-decoder",
        "window_size": 8,
        "forecast_steps": 5,
        "num_heads": 2,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "attention_capture_sampling": "all",
    }
    f.num_features = 3
    f.forecast_steps = 5
    f.window_size = 8
    f.device = torch.device("cpu")

    # Use real RunContext from existing conftest.py
    base_context.metadata = {"is_hpo_trial": False}
    f.run_context = base_context
    f._get_artifact_path = lambda category, suffix, extension: str(
        f.run_context.get_artifact_path(category, f"unit_{suffix}", extension)
    )

    cap = AttentionCaptureBuffer()
    cap.configure(enabled=True, steps_to_capture=None)
    cap.data = {
        "enc_self_layer_0": torch.randn(2, 4, 4),
        "dec_cross_layer_1_step_3": torch.randn(2, 4, 4),
    }
    f.model = SimpleNamespace(attn_capture=cap)

    TransformerForecaster.save_attention_to_disk(f)

    npz_files = list(f.run_context.attention_dir.glob("*.npz"))
    meta_files = list(f.run_context.attention_dir.glob("*_metadata.json"))
    assert len(npz_files) == 1
    assert len(meta_files) == 1

    meta = json.loads(Path(meta_files[0]).read_text())
    assert meta["model"] == "transformer"
    assert meta["architecture"] == "encoder-decoder"
    assert meta["primary_map"].startswith("dec_cross_layer_")
    assert "timestamp" in meta
