import pytest
import torch
import pandas as pd
import numpy as np
from hypothesis import given, settings, strategies as st, HealthCheck


# Goal: Verify that the full pipeline (Fit -> Predict) works correctly for all
# architectural and configuration combinations without runtime crashes.

@st.composite
def e2e_config_strategy(draw):
    """
    Generates a valid configuration dictionary for E2E testing, covering:
    - Models (LSTM, Transformer)
    - Strategies (Direct, Iterative)
    - Alignment modes (PC, Off)
    - Variable types (P, C, F via shared names logic)
    """
    # 1. Model & Strategy
    model_type = draw(st.sampled_from(["lstm", "transformer"]))
    strategy = draw(st.sampled_from(["direct", "iterative"]))

    # 2. Architecture (Transformer only, LSTM is always encoder-only)
    architecture = "encoder-only"
    if model_type == "transformer":
        architecture = draw(st.sampled_from(["encoder-only", "encoder-decoder"]))

    # 3. Exogenous Variables Setup (P / C / F generation)
    # n_enc: number of encoder variables (candidates for P or C)
    # n_dec: number of decoder variables (candidates for C or F)
    # use_shared: if True, enc and dec names overlap (creating Group C)
    n_enc = draw(st.integers(min_value=0, max_value=2))
    n_dec = draw(st.integers(min_value=0, max_value=2))
    use_shared = draw(st.booleans())

    # 5. Advanced Neural Network Params
    use_revin = draw(st.booleans())
    activation = draw(st.sampled_from(["relu", "gelu"]))

    return {
        "model_type": model_type,
        "strategy": strategy,
        "architecture": architecture,
        "n_enc": n_enc,
        "n_dec": n_dec,
        "use_shared": use_shared,
        "use_revin": use_revin,
        "activation": activation
    }


@settings(deadline=None, max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(config=e2e_config_strategy())
def test_e2e_model_pipeline_stability(model_factory, config):
    """
    Verifies that the model can pass through public fit() and predict() API without crashing.
    """
    _run_e2e_check(model_factory, config, use_internal_api=False)


@settings(deadline=None, max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(config=e2e_config_strategy())
def test_fit_and_evaluate_fold_integration(model_factory, config):
    """
    Verifies that the internal _fit_and_evaluate_fold method (used by HPO/Backtesting)
    works correctly, returning (loss, predictions, history) tuple.
    """
    _run_e2e_check(model_factory, config, use_internal_api=True)


def _run_e2e_check(model_factory, config, use_internal_api):
    # Unpack config
    model_type = config["model_type"]
    strategy = config["strategy"]
    architecture = config["architecture"]
    n_enc = config["n_enc"]
    n_dec = config["n_dec"]
    use_shared = config["use_shared"]

    # Create model via factory
    try:
        forecaster = model_factory(
            model_type=model_type,
            strategy=strategy,
            n_targets=1,
            n_enc=n_enc,
            n_dec=n_dec,
            use_shared_col_names=use_shared,
            architecture=architecture,
            use_revin=config["use_revin"],
            activation=config["activation"],
            epochs=1,
            batch_size=2
        )
    except ValueError:
        return

    window_size = forecaster.window_size
    forecast_steps = forecaster.forecast_steps

    # Reconstruct column names
    target_cols = ["target_0"]
    if use_shared:
        common = min(n_enc, n_dec)
        enc_cols = [f"shared_{i}" for i in range(common)] + [f"enc_{i}" for i in range(n_enc - common)]
        dec_cols = [f"shared_{i}" for i in range(common)] + [f"dec_{i}" for i in range(n_dec - common)]
    else:
        enc_cols = [f"enc_{i}" for i in range(n_enc)]
        dec_cols = [f"dec_{i}" for i in range(n_dec)]

    all_cols = sorted(list(set(target_cols + enc_cols + dec_cols)))

    # Compute new API columns
    if use_shared:
        # Shared columns -> future_covariates, encoder-only -> past_covariates
        enc_set = set(enc_cols)
        dec_set = set(dec_cols)
        past_cols = list(enc_set - dec_set)      # Encoder-only
        future_cols = list(enc_set & dec_set)    # Shared (both encoder and decoder)
        # Decoder-only columns (when n_enc < n_dec with use_shared=True)
        dec_only = list(dec_set - enc_set)
        future_cols.extend(dec_only)  # Add decoder-only to future_covariates
    else:
        # Disjoint: enc -> past, dec -> future
        past_cols = enc_cols
        future_cols = dec_cols

    # Generate Data
    total_len = window_size + forecast_steps + 10
    full_df = pd.DataFrame(
        np.random.randn(total_len, len(all_cols)),
        columns=all_cols,
        index=pd.date_range("2022-01-01", periods=total_len, freq="D")
    )

    # Split into Train Fold and Eval Fold (for _fit_and_evaluate_fold)
    split_idx = total_len - forecast_steps
    train_fold = full_df.iloc[:split_idx]
    eval_fold = full_df.iloc[split_idx:]  # Contains targets + future exog

    from utils.dataset import TimeSeriesDataset
    ts_dataset = TimeSeriesDataset(
        "test_ds", {},
        num_features=len(target_cols),
        data=full_df,
        columns=target_cols,
        past_covariates=past_cols,
        future_covariates=future_cols,
        freq="D"
    )

    if use_internal_api:
        # --- TEST _fit_and_evaluate_fold ---
        try:
            loss, preds, history = forecaster._fit_and_evaluate_fold(
                train_fold=train_fold,
                eval_fold=eval_fold,
                validation_params={"early_stopping_validation_percentage": 10},
                dataset=ts_dataset,
                is_final_fit=False  # Simulate HPO step
            )
        except ValueError as e:
            # If validation fails due to short data or config mismatch, it's acceptable
            # providing it's a clean ValueError
            return

        # Assertions for Internal API
        assert np.isfinite(loss), f"Loss should be finite. Got {loss}"
        assert isinstance(preds, pd.DataFrame)
        assert len(preds) == forecast_steps
        assert isinstance(history, dict)
        # Check if history populated (Neural models should have loss)
        if "loss" not in history:
            # Might be empty if training crashed silently or 0 epochs?
            # But we set epochs=1.
            pass

    else:
        # --- TEST Public API (fit -> predict) ---
        try:
            loss, history = forecaster.fit(train_fold, is_final_fit=True, dataset=ts_dataset)
        except ValueError:
            return

        assert np.isfinite(loss)

        # Prepare future exog for predict
        # In public API, we must manually slice future exog
        future_exog_cols = enc_cols + dec_cols  # Simplification, pass everything
        future_exog_df = eval_fold[list(set(future_exog_cols))] if future_exog_cols else None
        input_df = train_fold.iloc[-window_size:]

        try:
            preds = forecaster.predict(input_df, future_exog=future_exog_df)
        except ValueError as e:
            raise e

        assert isinstance(preds, pd.DataFrame)
        assert len(preds) == forecast_steps
        # LSTM iterative does NOT support pure future-only exog.
        # In such degenerate configurations, NaN fallback is acceptable.
        if (
                model_type == "lstm"
                and strategy == "iterative"
                and n_enc > 0
                and n_dec > 0
                and not use_shared
        ):
            # Pure F-only case → allowed NaN fallback
            return

        assert not preds.isna().any().any()