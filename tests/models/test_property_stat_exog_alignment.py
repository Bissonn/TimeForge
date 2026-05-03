# tests/models/test_property_stat_exog_alignment.py
import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from hypothesis import given, strategies as st, settings

from models.base_arima import ARIMABaseForecaster
from models.var import VARForecaster


# ---------- Helpers ----------

@st.composite
def exog_cases_matching(draw):
    """
    Generate cases where exogenous columns and fitted_exog_names are compatible.

    - steps: forecast horizon length
    - fitted_exog_names: the order that the model expects
    - exog_df: DataFrame with the same columns in arbitrary order
               (and possibly extra columns which should be ignored)
    """
    steps = draw(st.integers(min_value=1, max_value=16))
    n_fitted = draw(st.integers(min_value=1, max_value=4))
    n_extra = draw(st.integers(min_value=0, max_value=3))

    fitted_exog_names = [f"ex{i}" for i in range(n_fitted)]
    extra_names = [f"extra{i}" for i in range(n_extra)]
    all_names = fitted_exog_names + extra_names

    # Permute columns to simulate arbitrary column ordering
    permuted_cols = draw(st.permutations(all_names)) if len(all_names) > 1 else all_names

    data = np.random.randn(steps, len(all_names)).astype(np.float64)
    exog_df = pd.DataFrame(data, columns=list(permuted_cols))

    return steps, fitted_exog_names, exog_df


@st.composite
def exog_cases_too_short(draw):
    """
    Generate cases where exogenous DataFrame has fewer rows than required steps.
    Expectation: _validate_prediction_exog must raise ValueError.
    """
    steps = draw(st.integers(min_value=2, max_value=16))
    n_rows = draw(st.integers(min_value=1, max_value=steps - 1))
    n_fitted = draw(st.integers(min_value=1, max_value=4))

    fitted_exog_names = [f"ex{i}" for i in range(n_fitted)]
    data = np.random.randn(n_rows, n_fitted).astype(np.float64)
    exog_df = pd.DataFrame(data, columns=fitted_exog_names)

    return steps, fitted_exog_names, exog_df


# ---------- ARIMABaseForecaster properties ----------

@settings(deadline=None, max_examples=50)
@given(exog_cases_matching())
def test_arima_exog_reorders_and_filters_correctly(case):
    """
    Property:
      - ARIMABaseForecaster._validate_prediction_exog is a strict validator.
      - It requires exog_pred columns to match fitted_exog_names EXACTLY (names and order).
      - It returns None.
    """
    steps, fitted_exog_names, exog_df = case

    dummy_self = SimpleNamespace(fitted_exog_names=fitted_exog_names)

    # The implementation is strict. If columns don't match exactly, it raises ValueError.
    # We manually ensure strict match for the "success" case.

    # 1. Test Success Case: Exact match
    clean_df = exog_df[fitted_exog_names].copy()
    ARIMABaseForecaster._validate_prediction_exog(
        dummy_self,
        exog_pred=clean_df,
        steps=steps,
    )

    # 2. Test Failure Case: Mismatch (if applicable)
    # If the generated exog_df has different columns/order, validation MUST fail.
    # We match either "Unexpected number..." (if length differs) or "Mismatch" (if names differ)
    if list(exog_df.columns) != fitted_exog_names:
        with pytest.raises(ValueError, match="Exogenous feature mismatch|Unexpected number of exogenous columns"):
            ARIMABaseForecaster._validate_prediction_exog(
                dummy_self,
                exog_pred=exog_df,
                steps=steps,
            )


@settings(deadline=None, max_examples=50)
@given(exog_cases_too_short())
def test_arima_exog_raises_if_too_short(case):
    """
    Property:
      If exog_pred has fewer rows than 'steps', _validate_prediction_exog
      must raise a ValueError (cannot silently broadcast or truncate).
    """
    steps, fitted_exog_names, exog_df = case
    dummy_self = SimpleNamespace(fitted_exog_names=fitted_exog_names)

    with pytest.raises(ValueError, match="Prediction exogenous length mismatch"):
        ARIMABaseForecaster._validate_prediction_exog(
            dummy_self,
            exog_pred=exog_df,
            steps=steps,
        )


# ---------- VARForecaster properties ----------

@settings(deadline=None, max_examples=50)
@given(exog_cases_matching())
def test_var_exog_reorders_and_filters_correctly(case):
    """
    Same property as ARIMA, but for VARForecaster._validate_prediction_exog.
    This keeps the exogenous semantics consistent across stat models.
    """
    steps, fitted_exog_names, exog_df = case

    dummy_self = SimpleNamespace(fitted_exog_names=fitted_exog_names)

    # 1. Test Success Case: Exact match
    clean_df = exog_df[fitted_exog_names].copy()
    VARForecaster._validate_prediction_exog(
        dummy_self,
        exog_pred=clean_df,
        steps=steps,
    )

    # 2. Test Failure Case
    if list(exog_df.columns) != fitted_exog_names:
        with pytest.raises(ValueError, match="Exogenous feature mismatch|Unexpected number of exogenous columns"):
            VARForecaster._validate_prediction_exog(
                dummy_self,
                exog_pred=exog_df,
                steps=steps,
            )


@settings(deadline=None, max_examples=50)
@given(exog_cases_too_short())
def test_var_exog_raises_if_too_short(case):
    """
    Same validation as for ARIMA:
      if exog_pred has fewer rows than steps, raise ValueError.
    """
    steps, fitted_exog_names, exog_df = case
    dummy_self = SimpleNamespace(fitted_exog_names=fitted_exog_names)

    with pytest.raises(ValueError, match="Prediction exogenous data length mismatch"):
        VARForecaster._validate_prediction_exog(
            dummy_self,
            exog_pred=exog_df,
            steps=steps,
        )