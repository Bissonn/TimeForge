import pytest
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from models.transformer import (
    TransformerForecaster,
    TransformerModel,
)
from models.transformer_components import (
    LocalAttention,
    GlobalSelfAttention,
    CustomTransformerEncoderLayer,
    build_tgt_train,
    ZerosTgtInitializer,
    LastValueTgtInitializer,
    TrendTgtInitializer,
    MedianTgtInitializer,
    TgtInitializer,
)
from models.transformer_components.positional_encoding import (
    SinusoidalPositionalEncoding,
    LearnablePositionalEncoding,
    NoPositionalEncoding,
)
from models.factory import ModelFactory

# Import constant from conftest (if defined there) or local definition,
# if in conftest it is only inside a fixture.
# I assume WINDOW_SIZE is importable from conftest.py or models.conftest
try:
    from tests.models.conftest import WINDOW_SIZE
except ImportError:
    WINDOW_SIZE = 10

pytestmark = pytest.mark.unit

# --- Local constants specific to these tests ---
BATCH_SIZE = 2
NUM_FEATURES = 2
NUM_EXOG_ENCODER = 1
INPUT_SIZE = NUM_FEATURES + NUM_EXOG_ENCODER
FORECAST_STEPS = 3
BASE_WINDOW_SIZE = 1  # Used in boundary tests
HIDDEN_SIZE = 16
NUM_HEADS = 2


# --- Local Tensor Fixtures (Specific for TgtInitializer) ---
# These are kept here because they are used only in this file for mathematical tests

@pytest.fixture
def real_small_src(device):
    """Small src tensor: (1,10,2) – sin target + random exog."""
    np.random.seed(42)
    target = np.sin(np.linspace(0, 2 * np.pi, 10))
    exog = np.random.rand(10)
    data = np.column_stack([target, exog])
    src = torch.tensor(data).unsqueeze(0).float().to(device)  # (1,10,2)
    return src


@pytest.fixture
def real_small_src_short(device):
    """Short src for pinv: (1,2,1) [1.0,2.0] linear."""
    data = torch.tensor([[1.0], [2.0]]).float().to(device)
    return data.unsqueeze(0)  # (1,2,1)


@pytest.fixture
def real_multi_src(device):
    """Multivariate src: (1,3,2) feat0=[1,2,3], feat1=[10,20,5]."""
    data = torch.tensor([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 5.0]
    ]).float().to(device)
    return data.unsqueeze(0)  # (1,3,2)


# =============================================================================
# --- Section 1: Architecture and Configuration Tests ---
# (Merged from all.py, all_edges.py, model_unit.py)
# =============================================================================

def test_architecture_selection_encoder_only_vs_encoder_decoder(base_transformer_config, enc_only_dataset,
                                                                full_dataset, base_context):
    """Verify architecture flag correctly configures encoder-only vs encoder-decoder."""

    # 1. Encoder-only (using enc_only_dataset)
    cfg_enc = {**base_transformer_config, "architecture": "encoder-only"}
    f_enc = ModelFactory.create(
        "transformer",
        "test_transformer_model",
        run_context=base_context,
        model_params=cfg_enc,
        num_features=1,  # enc_only_dataset has 1 target
        forecast_steps=FORECAST_STEPS,
        window_size=WINDOW_SIZE,
        dataset=enc_only_dataset,
    )
    assert f_enc.model.architecture == "encoder-only"
    assert f_enc.model.transformer_decoder is None

    # 2. Encoder-decoder (using full_dataset - has 2 targets)
    cfg_encdec = {**base_transformer_config, "architecture": "encoder-decoder"}
    f_encdec = ModelFactory.create(
        "transformer",
        "test_transformer_model",
        run_context=base_context,
        model_params=cfg_encdec,
        num_features=2,  # full_dataset has 2 targets
        forecast_steps=FORECAST_STEPS,
        window_size=WINDOW_SIZE,
        dataset=full_dataset,
    )
    assert f_encdec.model.architecture == "encoder-decoder"
    assert f_encdec.model.transformer_decoder is not None


@pytest.mark.parametrize("invalid_param, expected_exception, match_str", [
    ({"num_heads": 0}, ZeroDivisionError, "integer modulo by zero"),
    ({"hidden_size": 15, "num_heads": 2}, ValueError, "hidden_size must be divisible by num_heads"),
    ({"readout": "invalid"}, ValueError, "readout must be one of"),
    ({"attention_type": "invalid"}, ValueError, "Invalid attention_type"),
])
def test_validate_model_params_raises_for_invalid(base_transformer_config, invalid_param, expected_exception, match_str,
                                                  enc_only_dataset, base_context):
    """Test _validate_model_params: Raises for missing/invalid params."""
    # (Merged 'readout' validation from all_edges.py)
    cfg = {**base_transformer_config, "architecture": "encoder-only", **invalid_param}

    # Add 'cls' to allowed options so 'invalid' tests work correctly
    # If testing readout='cls' and the model already supports it, the 'invalid' test might not raise an error,
    # so skip this specific case in error parametrization if it occurs.
    if invalid_param.get("readout") == "cls":
        pytest.skip("Skipping redundant 'cls' check if it is now considered valid.")

    with pytest.raises(expected_exception, match=match_str):
        ModelFactory.create(
            "transformer",
            "test_transformer_model",
            run_context=base_context,
            model_params=cfg,
            num_features=1,
            forecast_steps=1,
            window_size=WINDOW_SIZE,
            dataset=enc_only_dataset,
        )


def test_encoder_only_minimal_config_runs(base_transformer_config, enc_only_dataset, base_context):
    """Minimal boundary config (window=1, heads=1, no PE) should execute."""
    cfg = {
        **base_transformer_config,
        "hidden_size": 8,
        "num_heads": 1,
        "dropout": 0.0,
        "positional_encoding_config": {"type": "none"},  # Corrected configuration
        "architecture": "encoder-only",
    }
    f = ModelFactory.create(
        "transformer",
        "test_transformer_model",
        run_context=base_context,
        model_params=cfg,
        num_features=1,
        forecast_steps=FORECAST_STEPS,
        window_size=WINDOW_SIZE,  # Using full Window_size
        dataset=enc_only_dataset,
    )

    # Testing on a small window
    x = torch.randn(BATCH_SIZE, BASE_WINDOW_SIZE, f.feature_layout.encoder_input_size, device=f.device)
    y = f.model(x)
    assert y.shape == (BATCH_SIZE, FORECAST_STEPS, 1)


def test_num_heads_divides_hidden_size_boundary(base_transformer_config, enc_only_dataset, base_context):
    """hidden_size must remain divisible by num_heads in real configs."""
    # Using hidden_size=48, num_heads=6 → head_dim=8 (meets SDPA optimized kernels requirement)
    cfg = {**base_transformer_config, "hidden_size": 48, "num_heads": 6}
    f = ModelFactory.create(
        "transformer",
        "test_transformer_model",
        run_context=base_context,
        model_params=cfg,
        num_features=1,
        forecast_steps=FORECAST_STEPS,
        window_size=BASE_WINDOW_SIZE,
        dataset=enc_only_dataset,
    )

    assert cfg["hidden_size"] % cfg["num_heads"] == 0
    x = torch.randn(BATCH_SIZE, BASE_WINDOW_SIZE, f.feature_layout.encoder_input_size, device=f.device)
    _ = f.model(x)  # Smoke test


# =============================================================================
# --- Section 2: Positional Encoding and Readout ---
# =============================================================================

@pytest.mark.parametrize("mode", ["sinusoidal", "learnable", "none"])
@pytest.mark.parametrize("readout", ["last", "mean", "max", "cls"])
def test_positional_encoding_x_readout_forward(base_transformer_config, mode, readout, enc_only_dataset, base_context):
    """All PE modes should work with all readouts in encoder-only setup."""

    cfg = {**base_transformer_config,
           "architecture": "encoder-only",
           "positional_encoding_config": {"type": mode},
           "readout": readout}

    f = ModelFactory.create(
        "transformer",
        "test_transformer_model",
        run_context=base_context,
        model_params=cfg,
        num_features=1,
        forecast_steps=FORECAST_STEPS,
        window_size=BASE_WINDOW_SIZE,
        dataset=enc_only_dataset
    )

    x = torch.randn(BATCH_SIZE, BASE_WINDOW_SIZE, f.feature_layout.encoder_input_size).to(f.device)
    y = f.model(x)
    assert y.shape == (BATCH_SIZE, FORECAST_STEPS, 1)

    if mode == "learnable":
        assert isinstance(f.model.pos_encoder, LearnablePositionalEncoding)
    elif mode == "sinusoidal":
        assert isinstance(f.model.pos_encoder, SinusoidalPositionalEncoding)
    else:  # 'none'
        assert isinstance(f.model.pos_encoder, NoPositionalEncoding)


# =============================================================================
# --- Section 3: API Tests (forward, device) ---
# (From test_transformer_all_edges.py and test_transformer_all.py)
# =============================================================================

def test_encoder_decoder_forward_missing_tgt_raises(base_transformer_config, full_dataset, base_context):
    """Encoder-decoder forward() must raise if tgt is not provided."""
    cfg = {**base_transformer_config, "architecture": "encoder-decoder"}
    f = ModelFactory.create(
        "transformer",
        "test_transformer_model",
        run_context=base_context,
        model_params=cfg,
        num_features=2,
        forecast_steps=FORECAST_STEPS,
        window_size=WINDOW_SIZE,
        dataset=full_dataset,
    )

    src = torch.randn(BATCH_SIZE, WINDOW_SIZE, f.feature_layout.encoder_input_size, device=f.device)
    with pytest.raises(ValueError, match="tgt .* must be provided"):
        _ = f.model.forward(src=src)


def test_encoder_decoder_forward_rejects_wrong_feature_dims(base_transformer_config, full_dataset, base_context):
    """Forward() should raise clear errors when src/tgt last dims are inconsistent."""
    cfg = {**base_transformer_config, "architecture": "encoder-decoder"}
    f = ModelFactory.create(
        "transformer",
        "test_transformer_model",
        run_context=base_context,
        model_params=cfg,
        num_features=2,
        forecast_steps=FORECAST_STEPS,
        window_size=WINDOW_SIZE,
        dataset=full_dataset,
    )

    B, S, T = BATCH_SIZE, WINDOW_SIZE, FORECAST_STEPS
    correct_encoder_input = f.model.input_projection.in_features
    correct_decoder_input = f.model.tgt_projection.in_features

    # Wrong src features (using INPUT_SIZE + 1 instead of correct_encoder_input)
    src_wrong = torch.randn(B, S, correct_encoder_input + 1, device=f.device)
    tgt_ok = torch.randn(B, T, correct_decoder_input, device=f.device)

    # Error is raised by nn.Linear, so catching RuntimeError
    with pytest.raises(RuntimeError, match=r"mat1 and mat2 shapes cannot be multiplied"):
        _ = f.model.forward(src=src_wrong, tgt=tgt_ok)

    # Wrong tgt features
    src_ok = torch.randn(B, S, correct_encoder_input, device=f.device)
    tgt_wrong = torch.randn(B, T, correct_decoder_input + 1, device=f.device)
    with pytest.raises(RuntimeError, match=r"mat1 and mat2 shapes cannot be multiplied"):
        _ = f.model.forward(src=src_ok, tgt=tgt_wrong)


def test_device_handling_in_tensor_creation(base_transformer_config, full_dataset, base_context):
    """Ensures tensors created during internal predict (like tgt) are on the correct device."""
    # Test (fixed) from test_transformer_all.py
    cfg = {**base_transformer_config, "architecture": "encoder-decoder", "strategy": "direct"}
    f = ModelFactory.create(
        "transformer",
        "test_transformer_model",
        run_context=base_context,
        model_params=cfg,
        num_features=2,
        forecast_steps=3,
        window_size=WINDOW_SIZE,
        dataset=full_dataset
    )
    f.fitted = True

    real_input_size = f.model.input_projection.in_features
    real_hidden_size = f.model_params['hidden_size']

    # Mock model to simulate TransformerModel with encode/decode methods
    f.model = MagicMock()
    f.model.architecture = "encoder-decoder"
    f.model.encode.return_value = torch.randn(1, WINDOW_SIZE, real_hidden_size, device=f.device)
    f.model.decode.return_value = torch.randn(1, 3, 2, device=f.device)

    x = torch.randn(1, WINDOW_SIZE, real_input_size, device=f.device)
    # Mock dataset has no future_covariates, so len(...) = 0
    exog = torch.randn(1, FORECAST_STEPS, len(full_dataset.future_covariates), device=f.device)

    f._internal_predict(x, future_exog_tensor=exog)

    f.model.decode.assert_called_once()
    # checking positional arguments (.args), not keyword arguments (.kwargs)
    call_args = f.model.decode.call_args.args
    assert len(call_args) >= 1
    tgt_tensor = call_args[0]

    assert tgt_tensor.device.type == f.device.type


# =============================================================================
# --- Section 4: Local Attention Tests ---
# (From test_transformer_all.py and test_transformer_all_edges.py)
# =============================================================================

def test_local_attention_forward_pass_unit():
    """LocalAttention should produce outputs of the same shape as its inputs."""
    batch_size, seq_len, embed_dim = 2, 16, 32
    num_heads = 4
    window_size = 8
    la = LocalAttention(embed_dim=embed_dim, num_heads=num_heads, window_size=window_size)
    x = torch.randn(batch_size, seq_len, embed_dim)
    out, weights = la.forward(x)

    assert out.shape == x.shape
    assert weights is None

def test_local_attention_validates_window_size():
    """LocalAttention should accept odd sizes but reject < 1."""
    # Should pass for odd size (causal padding handles it)
    try:
        LocalAttention(embed_dim=32, num_heads=4, window_size=7)
    except ValueError:
        pytest.fail("LocalAttention should accept odd window_size.")

    # Should fail for invalid size
    with pytest.raises(ValueError, match="window_size must be >= 1"):
        LocalAttention(embed_dim=32, num_heads=4, window_size=0)


def test_attention_mechanism_selection_full_vs_local():
    """
    Verify that the model selects the correct attention mechanism
    (Global vs Local) inside the CustomTransformerEncoderLayer.
    """
    # 1. Test FULL Attention
    model_full = TransformerModel(
        encoder_input_size=10,
        decoder_input_size=10,
        num_features=1,
        forecast_steps=5,
        window_size=10,
        hidden_size=16,
        num_heads=2,
        attention_type="full"
    )

    # Get the first encoder layer
    enc_layer_full = model_full.transformer_encoder.layers[0]

    # Verify it is CustomTransformerEncoderLayer (not the standard PyTorch one)
    assert isinstance(enc_layer_full, CustomTransformerEncoderLayer)
    # Verify the internal attention module is GlobalSelfAttention
    assert isinstance(enc_layer_full.self_attn, GlobalSelfAttention)

    # 2. Test LOCAL Attention
    model_local = TransformerModel(
        encoder_input_size=10,
        decoder_input_size=10,
        num_features=1,
        forecast_steps=5,
        window_size=10,
        hidden_size=16,
        num_heads=2,
        attention_type="local",
        attention_window_size=4
    )

    enc_layer_local = model_local.transformer_encoder.layers[0]

    # Here we also expect CustomTransformerEncoderLayer
    assert isinstance(enc_layer_local, CustomTransformerEncoderLayer)
    # But the internal module must be LocalAttention
    assert isinstance(enc_layer_local.self_attn, LocalAttention)

# =============================================================================
# --- Section 5: TgtInitializer Tests (Hooks) ---
# (From test_tgt_initializers.py)
# =============================================================================

@pytest.mark.parametrize("init_cls, expected_matcher", [
    (ZerosTgtInitializer, lambda src: torch.zeros(1, 5, 1)),
    (LastValueTgtInitializer, lambda src: src[:, -1:, :1].repeat(1, 5, 1)),
    (MedianTgtInitializer, lambda src: torch.median(src[:, :, :1], dim=1, keepdim=True).values.repeat(1, 5, 1)),
    (TrendTgtInitializer, None)
])
def test_tgt_initializers_create_base_tgt(real_small_src, init_cls, expected_matcher, device):
    """Test _create_base_tgt for all strategies: shapes and base values."""
    src = real_small_src.to(device)
    steps, features = 5, 1  # H=5, F=1 (univariate target slice)

    initializer = init_cls(decoder_uses_exog=False, num_exog_decoder=0)

    # Checking device type (cuda/cpu), not object identity
    base_tgt = initializer._create_base_tgt(src, steps, features, device, src.dtype)

    assert base_tgt.shape == (1, steps, features)
    assert base_tgt.device.type == device.type
    assert base_tgt.dtype == src.dtype

    if expected_matcher is not None:
        expected = expected_matcher(src).to(device)
        assert torch.allclose(base_tgt, expected, atol=1e-6)


def test_trend_initializer_pinv_stability_and_fallback(real_small_src_short, device):
    """Trend init: Uses pinv for fit; fallback to last_value for L<2."""
    # Using passed device (can be cuda)
    src_short = real_small_src_short.to(device)
    initializer = TrendTgtInitializer(decoder_uses_exog=False, num_exog_decoder=0)

    # Requesting CPU explicitly to check if transfer is handled
    target_dev = torch.device('cpu')
    base_tgt = initializer._create_base_tgt(src_short, steps=3, num_features=1, device=target_dev, dtype=torch.float)

    assert base_tgt.device.type == 'cpu'
    expected = torch.tensor([[[3.0], [4.0], [5.0]]])
    assert torch.allclose(base_tgt, expected, atol=1e-6)

    src_one = torch.tensor([[[42.0]]]).float().to(device)  # (1,1,1)
    base_short = initializer._create_base_tgt(src_one, steps=3, num_features=1, device=target_dev, dtype=torch.float)
    expected_repeat = torch.full((1, 3, 1), 42.0)
    assert torch.allclose(base_short, expected_repeat)


def test_median_initializer_multivariate(real_multi_src, device):
    """Median: Computes per-feature median across seq."""
    src_multi = real_multi_src.to(device)
    initializer = MedianTgtInitializer(decoder_uses_exog=False, num_exog_decoder=0)
    target_dev = torch.device('cpu')

    base_tgt = initializer._create_base_tgt(src_multi, steps=2, num_features=2, device=target_dev, dtype=torch.float)

    assert base_tgt.device.type == 'cpu'
    expected = torch.tensor([[[2.0, 10.0], [2.0, 10.0]]])
    assert torch.allclose(base_tgt, expected, atol=1e-6)


# =============================================================================
# --- Section 6: build_tgt_train Tests (Teacher Forcing) ---
# (From test_tgt_build_train.py and test_transformer_train_tgt.py)
# =============================================================================

class DummyModel(nn.Module):
    """Minimal model from test_transformer_train_tgt.py"""

    def __init__(self, decoder_input_size: int, num_features: int):
        super().__init__()
        self.decoder_input_size = decoder_input_size
        self.num_features = num_features
        self.dec_proj = nn.Linear(decoder_input_size, 8)
        self.out = nn.Linear(8, num_features)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        assert tgt.size(-1) == self.decoder_input_size, \
            f"Decoder input dim mismatch: expected {self.decoder_input_size}, got {tgt.size(-1)}"
        h = torch.tanh(self.dec_proj(tgt))
        return self.out(h)


def test_build_tgt_train_no_exog_shift_right(device):
    """build_tgt_train: shifts targets right with a BOS of zeros."""
    B, H, F = 2, 5, 3
    target_true = torch.arange(B * H * F, dtype=torch.float32, device=device).reshape(B, H, F)
    src_dummy = torch.randn(B, 10, F, device=device)
    initializer = ZerosTgtInitializer(decoder_uses_exog=False, num_exog_decoder=0)

    tgt = build_tgt_train(
        target_true=target_true,
        src=src_dummy,
        initializer=initializer,
        decoder_exog=None,
    )

    assert tgt.shape == (B, H, F)
    assert torch.allclose(tgt[:, :1, :], torch.zeros(B, 1, F, device=device))
    assert torch.allclose(tgt[:, 1:, :], target_true[:, :-1, :])


def test_build_tgt_train_with_exog_concat(device):
    """build_tgt_train: shifts targets, but concatenates un-shifted exog."""
    B, H, F, E_dec = 1, 4, 2, 3
    target_true = torch.randn(B, H, F, device=device)
    decoder_exog = torch.randn(B, H, E_dec, device=device)
    src_dummy = torch.randn(B, 10, F, device=device)
    initializer = ZerosTgtInitializer(decoder_uses_exog=True, num_exog_decoder=E_dec)

    tgt = build_tgt_train(
        target_true=target_true,
        src=src_dummy,
        initializer=initializer,
        decoder_exog=decoder_exog,
    )

    assert tgt.shape == (B, H, F + E_dec)
    assert torch.allclose(tgt[:, 0:1, :F], torch.zeros(B, 1, F, device=device))
    assert torch.allclose(tgt[:, 1:, :F], target_true[:, :-1, :])
    assert torch.allclose(tgt[:, :, F:], decoder_exog)


def test_build_tgt_train_missing_exog_raises_when_required(device):
    """build_tgt_train: raises clear error if decoder_exog is missing but required."""
    B, H, F, E_dec = 1, 4, 2, 2
    target_true = torch.randn(B, H, F, device=device)
    src_dummy = torch.randn(B, 10, F, device=device)
    initializer = ZerosTgtInitializer(decoder_uses_exog=True, num_exog_decoder=E_dec)

    with pytest.raises(ValueError, match="decoder exogenous features.*missing|is None"):
        build_tgt_train(
            target_true=target_true,
            src=src_dummy,
            initializer=initializer,
            decoder_exog=None,
        )


@pytest.mark.parametrize("num_exog_decoder", [0, 2])
def test_train_direct_build_tgt_and_forward(device, num_exog_decoder):
    """Verifies build_tgt_train output works with a DummyModel (from train_tgt.py)"""
    torch.manual_seed(42)
    B, H, F = 2, 5, 3
    E_dec = num_exog_decoder
    decoder_uses_exog = E_dec > 0

    y_future = torch.randn(B, H, F, device=device)
    future_exog_dec = torch.randn(B, H, E_dec, device=device) if decoder_uses_exog else None
    src_dummy = torch.randn(B, 10, F + 4, device=device)
    initializer = ZerosTgtInitializer(decoder_uses_exog=decoder_uses_exog, num_exog_decoder=E_dec)

    tgt_train = build_tgt_train(
        target_true=y_future,
        src=src_dummy,
        initializer=initializer,
        decoder_exog=future_exog_dec
    )

    expected_dim = F + E_dec
    assert tgt_train.shape == (B, H, expected_dim)

    # Verification of integration with the model
    model = DummyModel(decoder_input_size=expected_dim, num_features=F).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    crit = nn.MSELoss()

    model.train()
    opt.zero_grad()
    pred = model(src_dummy, tgt_train)  # DummyModel ignores src

    assert pred.shape == (B, H, F)
    loss = crit(pred, y_future)
    loss.backward()
    opt.step()
