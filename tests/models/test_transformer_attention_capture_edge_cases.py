
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

# Import from your codebase
from models.transformer import (
    AttentionCaptureBuffer,
    LocalAttention,
    TransformerModel,
    TransformerForecaster,
)


# -----------------------------------------------------------------------------
# Helpers / Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_forecaster():
    """
    Minimal forecaster instance for testing helper methods without running __init__.

    We intentionally bypass TransformerForecaster.__init__ to keep tests fast and focused.
    """
    f = TransformerForecaster.__new__(TransformerForecaster)
    f.model_params = {}
    f.num_features = 3
    f.forecast_steps = 5
    f.window_size = 8
    f.device = torch.device("cpu")
    f.run_context = None
    f.model = None
    return f


class DummyRunContext:
    """
    Small stub with a writable metadata dict.

    TransformerForecaster.save_attention_to_disk only reads:
      - run_context.metadata['is_hpo_trial']
    """
    def __init__(self, metadata=None):
        self.metadata = dict(metadata or {})


def _artifact_path_builder(tmp_path: Path):
    """
    Builds a drop-in replacement for NeuralTSForecaster._get_artifact_path used in save_attention_to_disk.
    """
    def _get_artifact_path(category: str, suffix: str, extension: str) -> str:
        out_dir = tmp_path / category
        out_dir.mkdir(parents=True, exist_ok=True)
        # Keep deterministic filename
        return str(out_dir / f"unit_{suffix}.{extension}")
    return _get_artifact_path


def build_transformer_model(**overrides) -> TransformerModel:
    """
    Build TransformerModel with sensible, small defaults for fast tests.
    """
    defaults = dict(
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
        positional_encoding_config=None,
        readout="last",
        attention_type="full",
        attention_window_size=3,
        use_revin=False,
        revin_affine=True,
        revin_eps=1e-5,
        activation="gelu",
        norm_first=True,
    )
    defaults.update(overrides)
    return TransformerModel(**defaults)


def _rand_src(B=2, W=8, D=6, device="cpu"):
    torch.manual_seed(0)
    return torch.randn(B, W, D, device=device)


def _rand_tgt(B=2, H=5, D=3, device="cpu"):
    torch.manual_seed(1)
    return torch.randn(B, H, D, device=device)


# -----------------------------------------------------------------------------
# Priority 1: Critical missing tests
# -----------------------------------------------------------------------------

def test_resolve_capture_steps_all_out_of_range_returns_none(mock_forecaster):
    """CRITICAL: explicit step list entirely out of range should fall back to defaults (None)."""
    mock_forecaster.model_params = {"attention_capture_steps": [100, 200, 300]}
    assert TransformerForecaster._resolve_capture_steps(mock_forecaster, horizon=10) is None


def test_attention_capture_buffer_store_when_disabled_is_noop():
    """CRITICAL: store() must be a no-op when capture is disabled."""
    cap = AttentionCaptureBuffer()
    cap.configure(enabled=False, steps_to_capture=None)
    attn = torch.randn(2, 2, 4, 4)
    cap.store("key", attn)
    assert cap.data == {}


def test_save_attention_to_disk_skips_during_hpo_trial(tmp_path: Path, mock_forecaster):
    """CRITICAL: HPO trial should skip saving to avoid disk bloat."""
    mock_forecaster.run_context = DummyRunContext(metadata={"is_hpo_trial": True})
    mock_forecaster._get_artifact_path = _artifact_path_builder(tmp_path)

    cap = AttentionCaptureBuffer()
    cap.configure(enabled=True, steps_to_capture=None)
    cap.data = {"enc_self_layer_0": torch.randn(2, 4, 4)}

    mock_forecaster.model = SimpleNamespace(attn_capture=cap)
    TransformerForecaster.save_attention_to_disk(mock_forecaster)

    assert list((tmp_path / "attention").glob("*.npz")) == []
    assert list((tmp_path / "attention").glob("*_metadata.json")) == []


# -----------------------------------------------------------------------------
# Priority 2: Important missing tests
# -----------------------------------------------------------------------------

def test_attention_capture_buffer_store_none_tensor_is_noop():
    """store() should gracefully ignore None tensors."""
    cap = AttentionCaptureBuffer()
    cap.configure(enabled=True, steps_to_capture=None)
    cap.store("k", None)
    assert cap.data == {}


def test_resolve_capture_steps_invalid_mode_raises(mock_forecaster):
    """Invalid sampling mode should raise a clear ValueError."""
    mock_forecaster.model_params = {"attention_capture_sampling": "invalid_mode"}
    with pytest.raises(ValueError, match="Invalid attention_capture_sampling"):
        TransformerForecaster._resolve_capture_steps(mock_forecaster, horizon=10)


def test_resolve_capture_steps_invalid_steps_type_raises(mock_forecaster):
    """attention_capture_steps must be list/tuple of ints, not arbitrary objects (e.g. dict)."""
    mock_forecaster.model_params = {"attention_capture_steps": {"invalid": "dict"}}
    with pytest.raises(ValueError, match="attention_capture_steps must be a list/tuple"):
        TransformerForecaster._resolve_capture_steps(mock_forecaster, horizon=10)


def test_save_attention_to_disk_empty_data_writes_nothing(tmp_path: Path, mock_forecaster):
    """If capture buffer is empty, save_attention_to_disk should not create files."""
    mock_forecaster.run_context = DummyRunContext(metadata={"is_hpo_trial": False})
    mock_forecaster._get_artifact_path = _artifact_path_builder(tmp_path)

    cap = AttentionCaptureBuffer()
    cap.configure(enabled=True, steps_to_capture=None)
    cap.data = {}  # empty

    mock_forecaster.model = SimpleNamespace(attn_capture=cap)
    TransformerForecaster.save_attention_to_disk(mock_forecaster)

    assert list((tmp_path / "attention").glob("*.npz")) == []
    assert list((tmp_path / "attention").glob("*_metadata.json")) == []


def test_save_attention_to_disk_metadata_contains_required_fields(tmp_path: Path, mock_forecaster):
    """Metadata sidecar should include required fields with correct types and values."""
    mock_forecaster.run_context = DummyRunContext(metadata={"is_hpo_trial": False})
    mock_forecaster._get_artifact_path = _artifact_path_builder(tmp_path)

    mock_forecaster.model_params = {
        "architecture": "encoder-decoder",
        "strategy": "direct",
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "num_heads": 2,
        "attention_type": "full",
        "attention_capture_sampling": "first_last",
        "attention_capture_steps": [0, 2],
    }
    mock_forecaster.window_size = 8
    mock_forecaster.forecast_steps = 5

    cap = AttentionCaptureBuffer()
    cap.configure(enabled=True, steps_to_capture=None)
    cap.data = {"dec_cross_layer_1": torch.randn(2, 4, 4)}  # any non-empty

    mock_forecaster.model = SimpleNamespace(attn_capture=cap)

    TransformerForecaster.save_attention_to_disk(mock_forecaster)

    npz_files = list((tmp_path / "attention").glob("*.npz"))
    meta_files = list((tmp_path / "attention").glob("*_metadata.json"))
    assert len(npz_files) == 1
    assert len(meta_files) == 1

    meta = json.loads(Path(meta_files[0]).read_text())

    # Required string fields
    assert isinstance(meta["model"], str) and meta["model"] == "transformer"
    assert isinstance(meta["architecture"], str) and meta["architecture"] == "encoder-decoder"
    assert isinstance(meta["strategy"], str) and meta["strategy"] == "direct"
    assert isinstance(meta["primary_map"], str) and meta["primary_map"]
    assert isinstance(meta["timestamp"], str) and meta["timestamp"]

    # Required numeric fields
    assert meta["window_size"] == 8
    assert meta["forecast_steps"] == 5
    assert meta["num_heads"] == 2
    assert meta["num_encoder_layers"] == 2
    assert meta["num_decoder_layers"] == 2

    # Required boolean fields
    assert meta["batch_averaged"] is True
    assert meta["heads_preserved"] is True

    # Required list fields
    assert isinstance(meta["keys"], list) and len(meta["keys"]) >= 1


# -----------------------------------------------------------------------------
# Priority 3: Nice-to-have tests
# -----------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_local_attention_mask_cache_regenerates_on_device_change():
    """Mask cache should follow the input device (CPU -> CUDA)."""
    # Important: module parameters start on CPU. To run a CUDA forward pass,
    # the module must be moved to CUDA as well (otherwise MHA mixes devices).
    # The cached mask is a plain attribute (not a buffer), so it is expected
    # to be re-cast/re-moved by the forward() cache logic.
    attn = LocalAttention(embed_dim=8, num_heads=2, window_size=3, dropout=0.0)

    x_cpu = torch.randn(2, 6, 8, device="cpu")
    _ = attn(x_cpu)
    assert attn._cached_mask is not None
    assert attn._cached_mask.device.type == "cpu"

    # Move module weights to CUDA before feeding CUDA inputs.
    attn = attn.to("cuda")

    x_cuda = torch.randn(2, 6, 8, device="cuda")
    _ = attn(x_cuda)
    assert attn._cached_mask.device.type == "cuda"


def test_choose_primary_map_fallback_to_first_key_when_no_match(mock_forecaster):
    """If expected keys don't exist, _choose_primary_map should fall back to first key."""
    mock_forecaster.model_params = {"architecture": "encoder-decoder", "num_decoder_layers": 2}
    keys = ["unexpected_key_1", "unexpected_key_2"]
    assert TransformerForecaster._choose_primary_map(mock_forecaster, keys) == "unexpected_key_1"


def test_resolve_capture_steps_horizon_one_no_duplicates(mock_forecaster):
    """Boundary: horizon=1 should always produce [0] when explicit list contains 0."""
    mock_forecaster.model_params = {"attention_capture_steps": [0]}
    assert TransformerForecaster._resolve_capture_steps(mock_forecaster, horizon=1) == [0]


def test_forecaster_encoder_only_iterative_sets_step_each_iteration(monkeypatch):
    """
    End-to-end-ish: encoder-only iterative loop should call set_step(step) for every step.

    We use a minimal forecaster instance and a real TransformerModel (forecast_steps=1)
    to exercise the iterative path without requiring dataset/preprocessor plumbing.
    """
    f = TransformerForecaster.__new__(TransformerForecaster)
    f.model_params = {
        "architecture": "encoder-only",
        "strategy": "iterative",
        "attention_type": "full",
        # keep sampling None so we capture every step
    }
    f.num_features = 3
    f.forecast_steps = 6
    f.window_size = 8
    f.device = torch.device("cpu")
    f.fitted = True
    f._past_only_size = 0
    f._continuous_size = 0
    f.feature_layout = SimpleNamespace(
        decoder_exog_size=0,
        encoder_input_size=3  # num_features
    )

    # Internal horizon for encoder-only iterative is 1
    f.model = build_transformer_model(
        architecture="encoder-only",
        encoder_input_size=3,
        num_features=3,
        forecast_steps=1,
        window_size=8,
        attention_type="full",
        readout="last",
    ).to(f.device)

    # Enable capture
    f.model.attn_capture.configure(enabled=True, steps_to_capture=None)

    # Spy on set_step
    calls = []

    orig_set_step = f.model.attn_capture.set_step

    def spy_set_step(step):
        calls.append(step)
        return orig_set_step(step)

    monkeypatch.setattr(f.model.attn_capture, "set_step", spy_set_step)

    x = torch.randn(1, f.window_size, f.num_features)
    _ = TransformerForecaster._predict_iterative(f, x)

    assert calls == list(range(f.forecast_steps)), f"Expected step calls 0..{f.forecast_steps-1}, got {calls}"
