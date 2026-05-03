import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import pytest

from utils.dataset import TimeSeriesDataset
from utils.preprocessor import Preprocessor
from models.transformer import TransformerForecaster

pytestmark = pytest.mark.integration


class CaptureEncoderDecoderModel(nn.Module):
    """
    Dummy encoder-decoder Transformer used to capture the 'tgt' inputs passed
    into decode() during iterative prediction.

    This ensures we can inspect whether decoder exogenous variables are aligned properly.
    """

    def __init__(self, capture_list, num_features: int):
        super().__init__()
        self.capture_list = capture_list
        self.num_features = num_features
        self.architecture = "encoder-decoder"
        self.last_attention_weights = None
        self.output_head = lambda x: x[..., :1]

    # ---- Encoder stub ----
    def encode(self, src: torch.Tensor):
        # src: (B, T_src, D_src)
        B, T, _ = src.shape
        return torch.zeros(B, T, self.num_features, device=src.device)

    # ---- Decoder stub ----
    def decode(self, tgt: torch.Tensor, memory: torch.Tensor):
        """
        tgt: (B, T_tgt, D_tgt)
        memory: ignored
        return: (B, T_tgt, num_features)
        """
        self.capture_list.append(tgt.detach().cpu().clone())
        B, T, _ = tgt.shape
        return torch.zeros(B, T, self.num_features, device=tgt.device)

    def denormalize_output(self, y: torch.Tensor) -> torch.Tensor:
        """Mock implementation: identity."""
        return y

    # ---- Attention Capture Stubs ----
    def enable_attention_capture(self):
        """Stub to satisfy TransformerForecaster checks."""
        pass

    def disable_attention_capture(self):
        """Stub to satisfy TransformerForecaster checks."""
        pass

def test_transformer_encoder_decoder_iterative_exog_full_pipeline(base_context):
    """
    Full pipeline test:
        TimeSeriesDataset -> Preprocessor -> TransformerForecaster.predict()

    Configuration:
        - architecture="encoder-decoder"
        - strategy="iterative"
        - decoder exogenous variables enabled

    Goal:
        Check that decoder receives correctly aligned future exogenous values
        for each iterative step:
            step 1 -> future_exog[1]
            step 2 -> future_exog[2]
            ...
    """

    device = torch.device("cpu")

    W = 5   # window_size
    H = 4   # forecast_steps
    F = 1   # num_features (target size)
    E_dec = 1  # decoder exog size

    # --------------------------------------
    # 1. Construct raw data
    # --------------------------------------
    idx = pd.date_range("2020-01-01", periods=W + H, freq="D")
    df = pd.DataFrame(
        {
            "y": np.zeros(W + H, dtype=np.float32),
            "ex_dec": np.arange(W + H, dtype=np.float32),
        },
        index=idx,
    )

    # --------------------------------------
    # 2. Build dataset with decoder exogenous columns
    # --------------------------------------
    dataset = TimeSeriesDataset(
        dataset_name="test-transformer-encoder-decoder",
        config={},
        num_features=1,
        data=df,
        columns=["y"],
        past_covariates=[],      # encoder has NO exog in this test
        future_covariates=["ex_dec"],  # decoder exog only
        freq="D",
    )

    # --------------------------------------
    # 3. Preprocessor (identity transform)
    # --------------------------------------
    preprocessor = Preprocessor(
        config={"preprocessing_groups": []},  # no transformations
        target_columns=dataset.target_columns,
        exog_columns=dataset.future_covariates,
    )
    _ = preprocessor.fit_transform(dataset.series)

    # --------------------------------------
    # 4. Initialize TransformerForecaster
    # --------------------------------------
    model_params = {
        "architecture": "encoder-decoder",
        "strategy": "iterative",
        "hidden_size": 8,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "num_heads": 1,
        "dim_ff_multiplier": 2.0,
        "dropout": 0.0,
        "preprocessing": {},
    }

    forecaster = TransformerForecaster(
        model_params=model_params,
        num_features=F,
        forecast_steps=H,
        window_size=W,
        dataset=dataset,
        run_context=base_context
    )

    # Override preprocessor & device
    forecaster.preprocessor = preprocessor
    forecaster.fitted = True
    forecaster.device = device

    # --------------------------------------
    # 5. Replace underlying model with capturing dummy
    # --------------------------------------
    captured_tgts = []
    forecaster.model = CaptureEncoderDecoderModel(captured_tgts, num_features=F)

    # --------------------------------------
    # 6. Prepare input_data and future_exog for forecast
    # --------------------------------------
    input_data = df.iloc[:W].copy()
    future_exog = df.iloc[W:W + H].copy()  # rows 5..8

    # --------------------------------------
    # 7. Run prediction
    # --------------------------------------
    preds_df = forecaster.predict(input_data=input_data, future_exog=future_exog)

    # Basic sanity checks
    assert isinstance(preds_df, pd.DataFrame)
    assert len(preds_df) == H
    assert np.all(np.isfinite(preds_df.to_numpy()))

    # --------------------------------------
    # 8. Verify decode() was called H times
    # --------------------------------------
    assert len(captured_tgts) == H, "decode() must be called once per iterative step"

    # --------------------------------------
    # 9. Check that decoder exog is aligned correctly
    # --------------------------------------
    future_exog_values = np.arange(W, W + H, dtype=np.float32)
    # step -> exog should be future_exog[step]

    # Step 0 is determined by initializer and may include past exog
    for step in range(1, H):
        tgt = captured_tgts[step]  # shape: (1, T_tgt, ?)
        assert tgt.shape[0] == 1

        # Last timestep features
        last_features = tgt[0, -1, :].numpy().astype(np.float32)

        # decoder exog is stored in last E_dec entries
        last_exog = last_features[-E_dec:]

        expected = np.array([future_exog_values[step]], dtype=np.float32)

        np.testing.assert_array_equal(
            last_exog,
            expected,
            err_msg=(
                f"Decoder exog misalignment at step {step}: expected {expected}, got {last_exog}"
            ),
        )
