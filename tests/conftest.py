import pytest
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import tempfile
from unittest.mock import MagicMock
from dataclasses import dataclass
from typing import List, Union, Optional
from models.lstm import LSTMForecaster
from models.transformer import TransformerForecaster
from utils.dataset import TimeSeriesDataset
from core.context import RunContext


# ============================================================================
# Dummy Preprocessor for Tests
# ============================================================================

@dataclass
class DummyPreprocessor:
    """Minimal preprocessor stub for contract tests."""
    target_columns: List[str]
    _full_raw_data_context: pd.DataFrame = None

    def transform(self, df: pd.DataFrame, allow_subset: bool = False) -> pd.DataFrame:
        """Pass-through transform."""
        if df is None or df.empty:
            raise ValueError("DummyPreprocessor.transform got empty input.")
        return df.copy()

    def inverse_transforms(
        self,
        predictions: Union[np.ndarray, pd.DataFrame],
        start_after: Optional[Union[pd.Timestamp, int]] = None,
    ) -> pd.DataFrame:
        """Convert numpy array to DataFrame with target columns."""
        if isinstance(predictions, pd.DataFrame):
            return predictions.copy()

        if not isinstance(predictions, np.ndarray):
            raise TypeError(f"Unsupported predictions type: {type(predictions)}")

        H, F = predictions.shape

        # Create simple sequential index
        if start_after is not None and hasattr(start_after, '__add__'):
            # DatetimeIndex
            idx = pd.date_range(start=start_after + pd.Timedelta(days=1), periods=H, freq='D')
        else:
            idx = pd.RangeIndex(start=0, stop=H)

        return pd.DataFrame(predictions, columns=self.target_columns[:F], index=idx)


@pytest.fixture(autouse=True)
def force_gpu_amp_consistency():
    """
    Automatically enforces Mixed Precision (AMP) for all GPU tests.

    CRITICAL: This logic MUST match the production training loop logic.
    - If GPU supports BFloat16 (Ampere+), use BFloat16 (Best for stability/FlashAttn).
    - Otherwise, use Float16.
    """
    if torch.cuda.is_available():
        # Dynamic hardware detection
        target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        # DEBUG: Log exact fixture autocast parameters
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[FIXTURE AUTOCAST] device_type='cuda', dtype={target_dtype}, bf16_supported={torch.cuda.is_bf16_supported()}")

        # Enforce the detected dtype
        with torch.autocast(device_type="cuda", dtype=target_dtype):
            # DEBUG: Verify it's enabled
            logger.info(f"[FIXTURE AUTOCAST] Inside context: is_autocast_enabled('cuda')={torch.is_autocast_enabled('cuda')}")
            if torch.is_autocast_enabled('cuda'):
                logger.info(f"[FIXTURE AUTOCAST] get_autocast_dtype('cuda')={torch.get_autocast_dtype('cuda')}")
            yield
    else:
        # CPU Fallback (Float32)
        yield

# ============================================================================
# RunContext Fixtures
# ============================================================================

@pytest.fixture
def base_context(tmp_path):
    """
    Create a basic RunContext for testing.

    Provides a minimal context with standard directory structure.
    """
    ctx = RunContext.from_base_path(
        base_path=tmp_path / "test_run",
        run_id="test_run",
        experiment_name="test_experiment"
    )
    ctx.create_directories()
    return ctx


@pytest.fixture
def fold_context(base_context):
    """
    Create a fold-specific context with typical metadata.

    Provides a context configured for fold 0 with transformer model.
    """
    return base_context.with_metadata(
        model_name="transformer",
        model_type="Transformer",
        fold_idx=0,
        window_size=96
    )


@pytest.fixture
def multi_fold_contexts(base_context):
    """
    Create multiple fold contexts for testing multi-fold scenarios.

    Returns a list of 3 fold contexts (fold 0, 1, 2).
    """
    return [
        base_context.with_metadata(
            model_name="transformer",
            fold_idx=fold_idx,
            window_size=96
        )
        for fold_idx in range(3)
    ]


# ============================================================================
# Dataset Fixtures
# ============================================================================

@pytest.fixture
def mock_dataset():
    """
    Creates a lightweight mock TimeSeriesDataset compliant with the Base API.
    """

    def _create(n_targets=1, n_enc=0, n_dec=0, length=50, use_shared_col_names=False):
        # Create columns names
        targets = [f"target_{i}" for i in range(n_targets)]

        if use_shared_col_names:
            # Force intersection: use same names for enc and dec
            # This is crucial for testing PC Mode 'Continuous' (C) group
            common_count = min(n_enc, n_dec)
            enc_only_count = n_enc - common_count
            dec_only_count = n_dec - common_count

            enc_exog = [f"shared_{i}" for i in range(common_count)] + \
                       [f"enc_{i}" for i in range(enc_only_count)]

            dec_exog = [f"shared_{i}" for i in range(common_count)] + \
                       [f"dec_{i}" for i in range(dec_only_count)]
        else:
            enc_exog = [f"enc_{i}" for i in range(n_enc)]
            dec_exog = [f"dec_{i}" for i in range(n_dec)]

        all_cols = sorted(list(set(targets + enc_exog + dec_exog)))

        # Create dummy dataframe
        data = pd.DataFrame(
            np.random.randn(length, len(all_cols)),
            columns=all_cols,
            index=pd.date_range("2023-01-01", periods=length, freq="D")
        )

        # Create mock object
        ds = MagicMock(spec=TimeSeriesDataset)
        ds.columns = all_cols
        ds.target_columns = targets

        # Compute past_covariates and future_covariates
        if use_shared_col_names:
            # Shared columns are future_covariates, encoder-only are past_covariates
            common_count = min(n_enc, n_dec)
            enc_only_count = n_enc - common_count
            dec_only_count = n_dec - common_count

            ds.past_covariates = [f"enc_{i}" for i in range(enc_only_count)]
            ds.future_covariates = ([f"shared_{i}" for i in range(common_count)] +
                                   [f"dec_{i}" for i in range(dec_only_count)])
        else:
            # Disjoint: enc_exog are past_covariates, dec_exog are future_covariates
            ds.past_covariates = enc_exog
            ds.future_covariates = dec_exog

        ds.series = data
        ds.freq = "D"

        split_idx = int(length * 0.8)
        ds.development_data = data.iloc[:split_idx]
        ds.test_data = data.iloc[split_idx:]

        return ds

    return _create


# ============================================================================
# Model Factory
# ============================================================================

@pytest.fixture
def model_factory(mock_dataset, base_context):
    """
    Factory fixture to create properly initialized Forecasters.

    Now supports run_context parameter for testing with/without instrumentation.
    """

    def _create(model_type="lstm", strategy="iterative",
                n_targets=1, n_enc=0, n_dec=0, window_size=10, forecast_steps=5,
                use_shared_col_names=False, run_context=None, fold_idx=None, **kwargs):

        # 1. Create matching dataset
        dataset = mock_dataset(n_targets, n_enc, n_dec, use_shared_col_names=use_shared_col_names)

        # 2. Create fold-specific context if requested
        if run_context is not None and fold_idx is not None:
            run_context = run_context.with_metadata(
                model_name=model_type,
                model_type=model_type.capitalize(),
                fold_idx=fold_idx,
                window_size=window_size
            )
        elif run_context is None and fold_idx is not None:
            # Create temporary context
            run_context = base_context.with_metadata(
                model_name=model_type,
                fold_idx=fold_idx,
                window_size=window_size
            )
        elif run_context is None:
            # Hypothesis tests: Always ensure a valid context exists
            tmp_dir = tempfile.mkdtemp()
            run_context = RunContext.from_base_path(
                base_path=Path(tmp_dir),
                run_id="factory_run",
                experiment_name="factory_exp"
            ).with_metadata(fold_idx=0, window_size=window_size)

        # 3. Construct valid configuration
        # Auto-detect future_covariate_mode based on dataset future_covariates
        future_covariate_mode = "none"
        if hasattr(dataset, 'future_covariates') and dataset.future_covariates:
            future_covariate_mode = "stepwise" if strategy == "iterative" else "global"

        params = {
            "strategy": strategy,
            "future_covariate_mode": future_covariate_mode,
            "hidden_size": 8,
            "num_layers": 1,
            "dropout": 0.0,
            "batch_size": 4,
            "epochs": 1,
            "learning_rate": 0.01,
            "future_injection": False,
            "architecture": "encoder-only" if n_dec == 0 else "encoder-decoder",
            "d_model": 8,
            "nhead": 1,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "dim_feedforward": 16,
            "attention_type": "full",
            **kwargs
        }

        if model_type == "lstm":
            cls = LSTMForecaster
        elif model_type == "transformer":
            cls = TransformerForecaster
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        model = cls(
            model_params=params,
            num_features=n_targets,
            forecast_steps=forecast_steps,
            window_size=window_size,
            dataset=dataset,
            run_context=run_context  # Pass run_context!
        )

        model.device = torch.device("cpu")
        if model.model:
            model.model.to(model.device)

        model.fitted = True

        # Add dummy preprocessor for contract tests
        model.preprocessor = DummyPreprocessor(
            target_columns=dataset.target_columns,
            _full_raw_data_context=dataset.series
        )

        return model

    return _create
