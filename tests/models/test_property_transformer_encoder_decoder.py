# tests/models/test_property_transformer_encoder_decoder.py

import numpy as np
import torch

from hypothesis import given, settings, strategies as st

from models.transformer_components.tgt_initializers import (
    ZerosTgtInitializer,
    LastValueTgtInitializer,
    MeanTgtInitializer,
    MedianTgtInitializer,
    TrendTgtInitializer,
    build_tgt_train,
)


# --------------------------------------------------------------------
# Helper strategies
# --------------------------------------------------------------------

@st.composite
def shape_triplets(draw):
    """
    Generate (B, H, F) with small but non-trivial sizes.
    """
    B = draw(st.integers(min_value=1, max_value=5))
    H = draw(st.integers(min_value=1, max_value=8))
    F = draw(st.integers(min_value=1, max_value=4))
    return B, H, F


@st.composite
def shape_quadruples(draw):
    """
    Generate (B, H, F, E) for encoder-decoder targets + decoder exog.
    """
    B = draw(st.integers(min_value=1, max_value=5))
    H = draw(st.integers(min_value=1, max_value=8))
    F = draw(st.integers(min_value=1, max_value=4))
    E = draw(st.integers(min_value=0, max_value=4))  # allow no-exog case
    return B, H, F, E


# --------------------------------------------------------------------
# Property tests: build_tgt_train (training path)
# --------------------------------------------------------------------

@settings(deadline=None, max_examples=100)
@given(shapes=shape_quadruples())
def test_build_tgt_train_zero_initializer_shift_and_exog_alignment(shapes):
    """
    Property for build_tgt_train with ZerosTgtInitializer:

      * Target channel (F dims):
          - For H > 1: y_shifted[:, 0] = SOS = 0
                        y_shifted[:, 1:] = target_true[:, :-1]
          - For H = 1: y_shifted[:, 0] = SOS = 0
      * Decoder exog (E dims), if present:
          - exog is NOT shifted: decoder_exog[:, :H, :]
          - Concatenation: tgt = cat(shifted_targets, decoder_exog_slice)
    """
    B, H, F, E = shapes
    device = torch.device("cpu")
    dtype = torch.float32

    target_true = torch.randn(B, H, F, device=device, dtype=dtype)
    src = torch.randn(B, H, F, device=device, dtype=dtype)  # only for SOS

    if E > 0:
        decoder_exog = torch.randn(B, H + 3, E, device=device, dtype=dtype)
        initializer = ZerosTgtInitializer(decoder_uses_exog=True, num_exog_decoder=E)
    else:
        decoder_exog = None
        initializer = ZerosTgtInitializer(decoder_uses_exog=False, num_exog_decoder=0)

    tgt = build_tgt_train(
        target_true=target_true,
        src=src,
        initializer=initializer,
        decoder_exog=decoder_exog,
    )

    expected_F_total = F + (E if E > 0 else 0)
    assert tgt.shape == (B, H, expected_F_total)

    # Split into targets + exog if present
    if E > 0:
        tgt_targets = tgt[:, :, :F]
        tgt_exog = tgt[:, :, F:]
    else:
        tgt_targets = tgt
        tgt_exog = None

    # Target shift semantics
    if H > 1:
        # First timestep = SOS = zeros
        np.testing.assert_allclose(
            tgt_targets[:, 0].cpu().numpy(),
            np.zeros((B, F), dtype=np.float32),
            err_msg="First decoder step must be SOS zeros for ZerosTgtInitializer.",
        )
        # Remaining timesteps = ground truth shifted right
        np.testing.assert_allclose(
            tgt_targets[:, 1:].cpu().numpy(),
            target_true[:, :-1].cpu().numpy(),
            err_msg="Decoder targets must be right-shifted ground truth.",
        )
    else:
        # H == 1 → only SOS
        np.testing.assert_allclose(
            tgt_targets[:, 0].cpu().numpy(),
            np.zeros((B, F), dtype=np.float32),
            err_msg="For H=1, build_tgt_train must return only SOS for ZerosTgtInitializer.",
        )

    # Decoder exogenous alignment
    if E > 0:
        assert tgt_exog.shape == (B, H, E)
        expected_exog = decoder_exog[:, :H, :].cpu().numpy()
        np.testing.assert_allclose(
            tgt_exog.cpu().numpy(),
            expected_exog,
            err_msg="Decoder exogenous part must be unshifted decoder_exog[:, :H, :].",
        )


# --------------------------------------------------------------------
# Property tests: LastValueTgtInitializer.initialize_direct (inference)
# --------------------------------------------------------------------

@settings(deadline=None, max_examples=75)
@given(shapes=shape_quadruples())
def test_last_value_initializer_direct_semantics(shapes):
    """
    Property for LastValueTgtInitializer.initialize_direct:

      * Target part (F dims) for H steps:
          - All steps equal to the last target value from src (persistence).
      * Decoder exog (E dims), if present:
          - future_exog_tensor[:, :H, :E] is used directly.
      * Combined tgt shape: (B, H, F + E)
    """
    B, H, F, E = shapes
    device = torch.device("cpu")
    dtype = torch.float32

    # src: (B, W, input_size); only first F dims are target channels
    W = max(1, min(4, H))  # at least 1, small but non-trivial
    src = torch.randn(B, W, F + max(E, 1), device=device, dtype=dtype)

    if E > 0:
        # Direct mode requires exactly `forecast_steps` timesteps in future_exog_tensor
        future_exog = torch.randn(B, H, E, device=device, dtype=dtype)
        initializer = LastValueTgtInitializer(decoder_uses_exog=True, num_exog_decoder=E)
    else:
        future_exog = None
        initializer = LastValueTgtInitializer(decoder_uses_exog=False, num_exog_decoder=0)

    tgt = initializer.initialize_direct(
        src=src,
        forecast_steps=H,
        num_features=F,
        device=device,
        future_exog_tensor=future_exog,
    )

    expected_F_total = F + (E if E > 0 else 0)
    assert tgt.shape == (B, H, expected_F_total)

    if E > 0:
        tgt_targets = tgt[:, :, :F]
        tgt_exog = tgt[:, :, F:]
    else:
        tgt_targets = tgt
        tgt_exog = None

    # Targets: persistence of last encoder target
    last_val = src[:, -1:, :F]  # (B, 1, F)
    expected_targets = last_val.expand(-1, H, -1).cpu().numpy()
    np.testing.assert_allclose(
        tgt_targets.cpu().numpy(),
        expected_targets,
        err_msg="LastValueTgtInitializer must repeat last encoder target for all horizon steps.",
    )

    # Exog: direct slice from future_exog_tensor
    if E > 0:
        expected_exog = future_exog[:, :H, :].cpu().numpy()
        np.testing.assert_allclose(
            tgt_exog.cpu().numpy(),
            expected_exog,
            err_msg="Decoder exogenous part must be future_exog_tensor[:, :H, :E].",
        )


# --------------------------------------------------------------------
# Property tests: ZerosTgtInitializer.initialize_iterative (step-wise exog)
# --------------------------------------------------------------------

@settings(deadline=None, max_examples=75)
@given(shapes=shape_quadruples(), step=st.integers(min_value=0, max_value=7))
def test_zeros_initializer_iterative_decoder_exog_alignment(shapes, step):
    """
    Property for ZerosTgtInitializer.initialize_iterative:

      Given:
        - future_exog_tensor with length >= H
        - step in [0, H-1]
      Then:
        - tgt has shape (B, 1, F + E) or (B, 1, F) if E == 0
        - target part is zeros (SOS-like)
        - exogenous part equals future_exog_tensor[:, step:step+1, :E]
    """
    B, H, F, E = shapes
    device = torch.device("cpu")
    dtype = torch.float32

    # Clamp step to a valid range
    step = step % max(H, 1)

    # src: arbitrary content, only shape/device matter for initializer logic
    W = max(2, H)
    src = torch.randn(B, W, F + max(E, 1), device=device, dtype=dtype)

    # future_exog_tensor: we always create at least 1 column, but logical E may be 0
    exog_dim = max(E, 1)
    future_exog = torch.randn(B, H + 3, exog_dim, device=device, dtype=dtype)

    if E > 0:
        initializer = ZerosTgtInitializer(decoder_uses_exog=True, num_exog_decoder=E)
    else:
        initializer = ZerosTgtInitializer(decoder_uses_exog=False, num_exog_decoder=0)

    tgt_step = initializer.initialize_iterative(
        src=src,
        num_features=F,
        device=device,
        future_exog_tensor=future_exog,
        step=step,
    )

    if E > 0:
        assert tgt_step.shape == (B, 1, F + E)
        tgt_targets = tgt_step[:, :, :F]
        tgt_exog = tgt_step[:, :, F:]

        # Targets are zeros
        np.testing.assert_allclose(
            tgt_targets.cpu().numpy(),
            np.zeros((B, 1, F), dtype=np.float32),
            err_msg="ZerosTgtInitializer.iterative must output zeros in target dims.",
        )

        # Exog: slice for this step
        expected_exog = future_exog[:, step:step + 1, :E].cpu().numpy()
        np.testing.assert_allclose(
            tgt_exog.cpu().numpy(),
            expected_exog,
            err_msg="Decoder exogenous values must align with future_exog_tensor[:, step:step+1, :E].",
        )
    else:
        # No exog: only target dims
        assert tgt_step.shape == (B, 1, F)
        np.testing.assert_allclose(
            tgt_step.cpu().numpy(),
            np.zeros((B, 1, F), dtype=np.float32),
            err_msg="ZerosTgtInitializer without exog must return only zeros in target dims.",
        )


# --------------------------------------------------------------------
# Property tests: shape / device stability for other initializers
# --------------------------------------------------------------------

@settings(deadline=None, max_examples=30)
@given(shapes=shape_quadruples())
def test_other_initializers_are_shape_and_device_stable(shapes):
    """
    Property for Mean/Median/Trend initializers:

      * For any (B, H, F, E):
          - initialize_direct returns (B, H, F+E) if E>0 else (B, H, F)
          - initialize_iterative returns (B, 1, F+E) if E>0 else (B, 1, F)
      * They must not crash on short windows or zero-exog cases.
    """
    B, H, F, E = shapes
    device = torch.device("cpu")
    dtype = torch.float32

    # Make sure window is at least 2 for Trend (needs history)
    W = max(2, H)
    src = torch.randn(B, W, F + max(E, 1), device=device, dtype=dtype)

    exog_dim = max(E, 1)
    future_exog = torch.randn(B, H + 2, exog_dim, device=device, dtype=dtype)

    InitializerClasses = [MeanTgtInitializer, MedianTgtInitializer, TrendTgtInitializer]

    for init_cls in InitializerClasses:
        # Case 1: with exog (if E > 0)
        if E > 0:
            init_with_exog = init_cls(decoder_uses_exog=True, num_exog_decoder=E)

            # Direct mode: must see exactly `forecast_steps` steps
            tgt_direct = init_with_exog.initialize_direct(
                src=src,
                forecast_steps=H,
                num_features=F,
                device=device,
                future_exog_tensor=future_exog[:, :H, :E],
            )
            assert tgt_direct.shape == (B, H, F + E)

            # Iterative mode: allowed to use a longer exog buffer (H+2),
            # implementations usually require length >= forecast_steps and index by `step`.
            for step in range(H):
                tgt_it = init_with_exog.initialize_iterative(
                    src=src,
                    num_features=F,
                    device=device,
                    future_exog_tensor=future_exog[:, :, :E],
                    step=step,
                )
                assert tgt_it.shape == (B, 1, F + E)

        # Case 2: without exog
        init_no_exog = init_cls(decoder_uses_exog=False, num_exog_decoder=0)

        tgt_direct2 = init_no_exog.initialize_direct(
            src=src,
            forecast_steps=H,
            num_features=F,
            device=device,
            future_exog_tensor=None,
        )
        assert tgt_direct2.shape == (B, H, F)

        for step in range(H):
            tgt_it2 = init_no_exog.initialize_iterative(
                src=src,
                num_features=F,
                device=device,
                future_exog_tensor=None,
                step=step,
            )
            assert tgt_it2.shape == (B, 1, F)
