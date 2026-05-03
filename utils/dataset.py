"""
Manages time series datasets within the forecasting framework.

This module provides the TimeSeriesDataset class to load, preprocess, split,
and manage time series data for training and evaluating forecasting models
like ARIMA, VAR, LSTM, and Transformer.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import numpy as np
import pandas as pd
import os

logger = logging.getLogger(__name__)


def _normalize_freq(freq: str) -> str:
    """
    Normalize pandas frequency string to use lowercase (pandas 2.2+ style).

    Converts deprecated uppercase frequency codes to their lowercase equivalents
    to avoid FutureWarnings.

    Args:
        freq: Frequency string (e.g., 'H', 'D', 'W')

    Returns:
        Normalized frequency string (e.g., 'h', 'D', 'W')

    Examples:
        >>> _normalize_freq('H')
        'h'
        >>> _normalize_freq('15T')
        '15min'
        >>> _normalize_freq('2H')
        '2h'
        >>> _normalize_freq('D')  # Already lowercase-friendly
        'D'
    """
    if not freq:
        return freq

    # Mapping of deprecated uppercase to new lowercase
    replacements = {
        'H': 'h',  # Hourly
        'T': 'min',  # Minutely (T → min)
        'S': 's',  # Seconds
    }

    # Handle composite frequencies like '15T', '2H'
    if len(freq) > 1 and freq[-1] in replacements:
        # e.g., '15T' → '15min', '2H' → '2h'
        return freq[:-1] + replacements[freq[-1]]

    # Handle simple single-character frequencies like 'H', 'T'
    if freq in replacements:
        return replacements[freq]

    # Already normalized or doesn't need normalization (D, W, M, etc.)
    return freq

class TimeSeriesDataset:
    """
    Manages a time series dataset, including loading, preparation, and splitting.

    This class serves as the primary container for a complete dataset. It is
    responsible for loading data from a source (e.g., a CSV file), setting a
    proper time index, and holding all potential columns (targets and various
    exogenous features). The logic for creating specific subsets of data for
    individual model runs is handled by the `DataProvider` class.

    Attributes:
        name (str): The name of the dataset.
        config (Dict): The main configuration dictionary.
        series (pd.DataFrame): The primary DataFrame holding the time series data.
            (Note: This series may be differenced if configured).
        target_columns (List[str]): A list of column names to be treated as targets.
        past_covariates (List[str]): Features known only in history (encoder-only).
        future_covariates (List[str]): Features known in both history and future (encoder + decoder).
        development_data (Optional[pd.DataFrame]): The training/validation part of the data.
        test_data (Optional[pd.DataFrame]): The hold-out test part of the data.
        _original_series (Optional[pd.DataFrame]): A copy of the raw, non-differenced
            data, used as context for inverse differencing.
        _diff_state (Optional[Dict[str, Any]]): Stores the parameters (d, D, s)
            used during the differencing step.
    """

    CYCLIC_PERIODS = {
        "day_of_week": 7, "dayofweek": 7,
        "week": 52, "weekofyear": 52, "week_of_year": 52,
        "day_of_year": 365, "dayofyear": 365,
        "month": 12, "quarter": 4,
        "hour": 24, "minute": 60, "second": 60,
    }

    # Features needing 0-based adjustment
    ONE_BASED_FEATURES = {
        "month", "day_of_year", "dayofyear", "week", "weekofyear", "week_of_year"
    }

    def __init__(
        self,
        dataset_name: str,
        config: Dict,
        num_features: int,  # Mandatory contract: how many targets to expect
        data: Optional[pd.DataFrame] = None,
        columns: Optional[List[str]] = None,
        past_covariates: Optional[List[str]] = None,
        future_covariates: Optional[List[str]] = None,
        freq: Optional[str] = None,
        date_column: str = "date",
    ) -> None:
        """
        Initializes the TimeSeriesDataset object.

        The constructor handles loading the data (either from a file path specified
        in the config or from a provided DataFrame), identifying all relevant columns
        (targets, exogenous), and preparing the main `series` DataFrame with a
        proper index.

        Args:
            dataset_name: Name of the dataset
            config: Configuration dictionary
            num_features: Number of target features (contract)
            data: Optional DataFrame with time series data
            columns: Target column names
            past_covariates: Features known only in history (encoder-only)
            future_covariates: Features known in both history and future
            freq: Frequency string (e.g., 'h', 'D')
            date_column: Name of the date column
        """
        if not dataset_name:
            raise ValueError("dataset_name cannot be empty.")

        is_data_provided = data is not None

        if not is_data_provided and (dataset_name not in config.get("datasets", {})):
            raise ValueError(f"Dataset '{dataset_name}' not found in config['datasets'].")

        self.name = dataset_name
        self.config = config
        self.num_features = num_features  # Storage of the contract
        self.date_column = date_column
        self.path = config.get("datasets", {}).get(dataset_name, {}).get("path") if not is_data_provided else None

        dataset_cfg = config.get("datasets", {}).get(dataset_name, {})

        raw_data = data if is_data_provided else self._load_data()

        # Load covariates using new API
        if is_data_provided:
            self.target_columns = columns or []
            self.past_covariates = past_covariates or []
            self.future_covariates = future_covariates or []

            # If targets are not explicitly provided, infer them by exclusion
            if not self.target_columns:
                excluded = set(self.past_covariates + self.future_covariates + [self.date_column])
                self.target_columns = [
                    col for col in raw_data.columns
                    if col not in excluded
                ]
        else:
            # Logic for loading column names from config file
            self.target_columns = columns or dataset_cfg.get("columns", [])
            self.past_covariates = past_covariates or dataset_cfg.get("past_covariates", [])
            self.future_covariates = future_covariates or dataset_cfg.get("future_covariates", [])

            if not self.target_columns:
                inferred_cols = raw_data.select_dtypes(include=np.number).columns.tolist()
                if self.date_column in inferred_cols:
                    inferred_cols.remove(self.date_column)
                if inferred_cols:
                    self.target_columns = inferred_cols
                    logger.info(f"No target columns specified. Inferred from data: {self.target_columns}")

        all_cols = self.target_columns + self.past_covariates + self.future_covariates

        # Validate column definitions
        target_set = set(self.target_columns)
        past_set = set(self.past_covariates)
        future_set = set(self.future_covariates)

        # Disallow overlap between targets and covariates
        bad_overlap = (target_set & past_set) | (target_set & future_set)
        if bad_overlap:
            raise ValueError(
                "The following column(s) are used both as target(s) and covariate(s), "
                f"which is not supported: {sorted(bad_overlap)}"
            )

        # Disallow overlap between past_covariates and future_covariates
        covariate_overlap = past_set & future_set
        if covariate_overlap:
            raise ValueError(
                "The following column(s) are defined in both past_covariates and future_covariates, "
                f"which is not supported: {sorted(covariate_overlap)}\n"
                f"HINT: Features that are known in both past and future should ONLY be in future_covariates."
            )

        # Build canonical column order with unique names only once.
        self.columns = []
        for c in all_cols:
            if c not in self.columns:
                self.columns.append(c)

            # ═══════════════════════════════════════════════════════════════════
            # CONTRACT VALIDATION (Strict Data Contract)
            # ═══════════════════════════════════════════════════════════════════
        if len(self.target_columns) != self.num_features:
            raise ValueError(
                f"[{self.name}] Data contract violation:\n"
                f"  Expected: {self.num_features} target column(s)\n"
                f"  Found:    {len(self.target_columns)} target column(s): {self.target_columns}\n"
                f"  Registered past_covariates: {self.past_covariates}\n"
                f"  Registered future_covariates: {self.future_covariates}\n"
                f"  HINT: Check if covariate columns are properly registered."
            )

        logger.info(
            f"[{self.name}] Data contract validated: "
            f"{self.num_features} targets = {self.target_columns}"
        )

        if not self.columns and raw_data is not None and not raw_data.empty:
            logger.info(f"No specific columns defined; using all columns from provided data: {list(raw_data.columns)}")
            self.columns = list(raw_data.columns)

        if not self.columns:
            raise ValueError("No columns to process. Specify columns or provide data with numeric columns.")

        self.freq = freq if freq is not None else dataset_cfg.get("freq")

        # --- Initialize differencing attributes ---
        self._original_series: Optional[pd.DataFrame] = None
        self._diff_state: Optional[Dict[str, Any]] = None

        self.series = self._prepare_data(raw_data)

        # ---

        self.development_data = None
        self.test_data = None
        logger.info(
            f"TimeSeriesDataset '{self.name}' initialized. "
            f"Target columns: {self.target_columns}, "
            f"All managed feature columns in canonical order: {list(self.columns)}"
        )

    # ═══════════════════════════════════════════════════════════════════
    # BACKWARD COMPATIBILITY PROPERTIES (deprecated)
    # ═══════════════════════════════════════════════════════════════════

    @property
    def training_length(self) -> int:
        """
        Get the length of data available for training.

        Returns the number of time steps in the development set if train/test
        split has been done, otherwise returns the full series length.

        This is the preferred measure for dataset-aware hyperparameter suggestions
        (e.g., batch_size) as it represents the actual data that will be used
        for training.

        Returns:
            int: Number of time steps available for training

        Examples:
            >>> dataset = TimeSeriesDataset(...)
            >>> dataset.training_length  # Full series length
            1000
            >>> dataset.split_data(forecast_steps=100)
            >>> dataset.training_length  # After split
            900
        """
        if self.development_data is not None:
            return len(self.development_data)
        return len(self.series)

    @property
    def total_length(self) -> int:
        """
        Get the total length of the full time series.

        Returns the length of the complete series, regardless of train/test split.

        Returns:
            int: Total number of time steps in the series
        """
        return len(self.series)

    def get_encoder_window_columns(self) -> List[str]:
        """
        Returns the list of columns in the exact order required for the encoder's input window.

        Encoder window contains: targets + past_covariates + future_covariates
        """
        return self.target_columns + self.past_covariates + self.future_covariates

    def get_decoder_exog_columns(self) -> List[str]:
        """
        Returns the list of future-known exogenous columns for the decoder.

        Decoder receives only future_covariates (known in both history and future).
        """
        return self.future_covariates

    def get_past_covariates(self) -> List[str]:
        """Returns the list of past covariates (known only in history)."""
        return self.past_covariates

    def get_future_covariates(self) -> List[str]:
        """Returns the list of future covariates (known in both history and future)."""
        return self.future_covariates

    def _validate_and_sanitize_path(self, path: str) -> Path:
        """
        Validate and sanitize file path to prevent path traversal attacks.

        Args:
            path: File path to validate

        Returns:
            Path: Validated and resolved path object

        Raises:
            ValueError: If path contains path traversal patterns or file is too large
            FileNotFoundError: If file doesn't exist
        """
        # Security: Maximum file size (100 MB)
        MAX_FILE_SIZE = 100 * 1024 * 1024

        # Security: Check for obvious path traversal patterns in the input
        # This prevents attacks like "../../../../etc/passwd"
        if ".." in path or path.startswith("/etc") or path.startswith("/sys"):
            raise ValueError(
                f"Security: Path '{path}' contains suspicious patterns. "
                f"Relative parent references (..) and system paths are not allowed."
            )

        # Resolve to absolute path
        safe_path = Path(path).resolve()

        # Security: Ensure path is within allowed directory (for production use)
        # Allow data/, current working directory, and temp directories (for testing)
        try:
            # Try data/ directory first
            allowed_base = Path("data").resolve()
            safe_path.relative_to(allowed_base)
        except ValueError:
            # Try current working directory
            try:
                safe_path.relative_to(Path.cwd())
            except ValueError:
                # Allow temp directories (for testing) - check if path is in /tmp
                if not str(safe_path).startswith("/tmp/"):
                    raise ValueError(
                        f"Security: Path '{path}' is outside allowed directories. "
                        f"Please use paths within 'data/' directory, current working directory, "
                        f"or temporary directories."
                    )

        # Check if file exists
        if not safe_path.exists():
            raise FileNotFoundError(f"Dataset file not found at path: {safe_path}")

        # Check if it's a file (not directory)
        if not safe_path.is_file():
            raise ValueError(f"Path '{safe_path}' is not a file")

        # Security: Check file size to prevent DoS via large files
        file_size = safe_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            raise ValueError(
                f"File too large: {file_size / (1024*1024):.2f} MB "
                f"(maximum allowed: {MAX_FILE_SIZE / (1024*1024):.0f} MB). "
                f"Please use a smaller dataset or increase MAX_FILE_SIZE if needed."
            )

        return safe_path

    def _load_data(self) -> pd.DataFrame:
        """
        Loads data from a CSV file specified in the configuration.

        Returns:
            pd.DataFrame: A DataFrame containing the loaded data.

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is invalid, too large, or path is unsafe
        """
        if not self.path:
            raise ValueError("No path specified for dataset")

        # Security: Validate and sanitize path
        safe_path = self._validate_and_sanitize_path(self.path)

        # Load CSV with validated path
        df = pd.read_csv(safe_path)

        if df.empty:
            raise ValueError(f"Dataset file '{safe_path}' is empty.")

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates and adds time-based features to the DataFrame.

        Semantics:

        - For *cyclic* features (e.g. "day_of_week", "week", "month",
          "day_of_year", "hour", "minute", "second", "quarter") this method
          generates ONLY sinusoidal encodings:

              <feature>_sin, <feature>_cos

          and does NOT create the raw integer column <feature>. This avoids
          polluting the feature space with redundant raw encodings and
          makes it easier to use only sin/cos in exogenous configs.

        - For non-cyclic features (e.g. "year" or boolean flags like
          "is_month_start"), it generates a single raw column with the
          same name as in `time_features`.
        """
        dataset_cfg = self.config.get("datasets", {}).get(self.name, {})
        time_features_to_add = dataset_cfg.get("time_features", [])

        if not time_features_to_add:
            return df

        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("Cannot add time features because the index is not a DatetimeIndex.")
            return df

        for feature in time_features_to_add:
            # If anything with this name already exists (e.g. supplied in source),
            # do not overwrite it.
            if feature in df.columns:
                continue

            try:
                # 1) Get the base values from the DatetimeIndex
                if feature == 'week' or feature == 'weekofyear':
                    # special handling due to the compatibility problems with pandas lib version
                    if isinstance(df.index, pd.DatetimeIndex):
                        if pd.__version__ < '1.1.0':
                            feature_values = df.index.weekofyear
                        else:
                            feature_values = df.index.isocalendar().week.astype(int)
                    else:
                        logger.warning(
                            f"Cannot generate feature '{feature}' as index is not DatetimeIndex.")
                        continue
                else:
                    feature_values = getattr(df.index, feature)

                # 2) If this is a cyclic feature → generate ONLY sin/cos
                if feature in self.CYCLIC_PERIODS:
                    period = self.CYCLIC_PERIODS[feature]

                    # Work on a float copy to avoid integer issues
                    base = feature_values.astype(float)

                    # For 1-based cyclic features, shift to 0-based before encoding
                    # For 1-based features (month, day_of_year, week...) shift to 0-based
                    if feature in self.ONE_BASED_FEATURES:
                        base = (base - 1) % period

                    angle = 2 * np.pi * (base / period)

                    sin_col = f"{feature}_sin"
                    cos_col = f"{feature}_cos"

                    # Do not overwrite if they already exist
                    if sin_col not in df.columns:
                        df[sin_col] = np.sin(angle)
                    if cos_col not in df.columns:
                        df[cos_col] = np.cos(angle)
                else:
                    # 3) Non-cyclic feature: keep a single raw column.
                    df[feature] = feature_values
                    if pd.api.types.is_bool_dtype(df[feature]):
                        df[feature] = df[feature].astype(int)

            except AttributeError:
                logger.error(
                    f"Failed to generate time feature '{feature}'. It is not a valid attribute of DatetimeIndex.")
            except Exception as e:
                logger.error(f"Error generating feature '{feature}': {e}", exc_info=True)
        return df

    def _handle_missing_values_before_differencing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle genuine missing values in raw data BEFORE differencing.

        Uses tiered approach:
        - Tier 1: Forward fill (limit=3) for short gaps
        - Tier 2: Linear interpolation for longer gaps
        - Tier 3: Drop rows if imputation fails

        This preserves temporal continuity and maximizes data retention.
        """
        missing_count = df.isna().sum().sum()
        total_values = df.shape[0] * df.shape[1]
        missing_pct = (missing_count / total_values) * 100

        logger.warning(
            f"NaN values detected BEFORE differencing: {missing_count}/{total_values} "
            f"({missing_pct:.2f}%) values missing"
        )

        has_datetime_index = isinstance(df.index, pd.DatetimeIndex)

        if has_datetime_index:
            # Tier 1: Forward fill with limit
            df_filled = df.ffill(limit=3)
            nans_after_ffill = df_filled.isna().sum().sum()

            if nans_after_ffill > 0:
                logger.info(
                    f"After forward fill: {nans_after_ffill} NaNs remaining. "
                    f"Applying linear interpolation..."
                )
                # Tier 2: Interpolation
                df_filled = df_filled.interpolate(
                    method='linear',
                    limit_direction='both',
                    axis=0
                )
                nans_after_interp = df_filled.isna().sum().sum()

                if nans_after_interp > 0:
                    # Tier 3: Drop if necessary
                    rows_before = len(df_filled)
                    df_filled = df_filled.dropna()
                    rows_dropped = rows_before - len(df_filled)

                    if rows_dropped > 0:
                        logger.warning(
                            f"After imputation, {nans_after_interp} NaNs remained. "
                            f"Dropped {rows_dropped} rows."
                        )
                else:
                    logger.info("All NaNs successfully imputed via interpolation.")
            else:
                logger.info("All NaNs successfully imputed via forward fill.")

            return df_filled
        else:
            # No DatetimeIndex - simple fallback
            logger.warning(
                "No DatetimeIndex found. Using simple dropna() for missing values."
            )
            rows_before = len(df)
            df = df.dropna()
            rows_dropped = rows_before - len(df)
            if rows_dropped > 0:
                logger.warning(f"Dropped {rows_dropped} rows with NaNs.")
            return df

    def _handle_missing_values_after_differencing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle NaN values created by differencing operations.

        These NaNs are mathematical artifacts from diff() and should be removed,
        not imputed, as there's no valid prior value to reconstruct from.

        Typically drops 1 row for d=1, 2 rows for d=2, etc.
        """
        nans_before = df.isna().sum().sum()
        rows_before = len(df)

        df = df.dropna()

        rows_dropped = rows_before - len(df)

        if rows_dropped > 0:
            logger.info(
                f"Dropped {rows_dropped} rows with NaNs created by differencing "
                f"(from {nans_before} NaN values)"
            )

        return df

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares the raw DataFrame by setting an index, generating time features,
        handling missing values, applying differencing, and selecting columns.
        """
        if df.empty:
            raise ValueError("Input DataFrame for preparation cannot be empty.")

        # 1. Date/index setup
        if isinstance(df.index, pd.DatetimeIndex):
            logger.debug("DataFrame already has a DatetimeIndex. Using it.")
        elif self.date_column in df.columns:
            df[self.date_column] = pd.to_datetime(df[self.date_column])
            df = df.set_index(self.date_column)
        else:
            logger.warning(f"No DatetimeIndex or '{self.date_column}' column found.")
            if not isinstance(df.index, pd.RangeIndex):
                df.index = pd.RangeIndex(len(df))

        # 2. Frequency inference
        if isinstance(df.index, pd.DatetimeIndex):
            if self.freq:
                df = df.asfreq(_normalize_freq(self.freq))
            else:
                inferred_freq = pd.infer_freq(df.index)
                if inferred_freq:
                    self.freq = inferred_freq
                    logger.info(f"Inferred frequency: {self.freq}")
                    df = df.asfreq(self.freq)
                else:
                    logger.warning("Could not infer frequency.")

        # 3. STRICT CONTRACT: Detect unregistered columns BEFORE adding time features
        registered_cols = set(self.target_columns + self.past_covariates + self.future_covariates)
        original_data_cols = set(df.columns)
        unregistered = original_data_cols - registered_cols

        if unregistered:
            logger.warning(
                f"[{self.name}] ⚠️  IGNORING UNREGISTERED COLUMNS\n"
                f"{'=' * 70}\n"
                f"The following columns exist in input data but are NOT registered in configuration:\n"
                f"  {sorted(unregistered)}\n"
                f"These columns will be DROPPED and NOT used by the model.\n"
                f"Action required only if you intended to use them (add to covariates).\n"
                f"{'=' * 70}\n"
            )
        df = df.drop(columns=list(unregistered))
        # 4. Add time features (these will be added to canonical order later)
        df = self._add_time_features(df)

        # 5. Set canonical column order
        all_cols = self.target_columns + self.past_covariates + self.future_covariates
        for col in df.columns:
            if col not in all_cols:
                all_cols.append(col)  # Now only adds time features (month, day, etc.)
        seen = set()
        self.columns = [c for c in all_cols if not (c in seen or seen.add(c))]

        # === ETAP 1: Missing value handling PRZED differencing ===
        if df.isna().any().any():
            df = self._handle_missing_values_before_differencing(df)

        # 5. Store original series BEFORE differencing
        self._original_series = df.copy()

        # 6. Apply differencing if configured
        dataset_cfg = self.config.get("datasets", {}).get(self.name, {})
        diff_config = dataset_cfg.get("differencing", {})

        if diff_config.get("enabled", False):
            logger.info(f"Applying dataset-level differencing for '{self.name}'...")
            df = self._apply_differencing(df, diff_config)

            # === ETAP 2: Missing value handling PO differencing ===
            if df.isna().any().any():
                df = self._handle_missing_values_after_differencing(df)

        # 7. Check required columns
        missing = [col for col in self.target_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Required target columns not found: {missing}")

        # 8. Select columns
        df = df[self.columns].copy()

        # 9. Validate no infinities
        if np.any(np.isinf(df.select_dtypes(include=np.number).values)):
            raise ValueError("Data contains infinite values.")

        # 10. Final check
        if df.empty:
            raise ValueError("DataFrame is empty after processing.")

        return df

    def split_data(self, forecast_steps: int) -> None:
        """
        Splits the main `series` DataFrame into development and test sets.
        """
        if self.series.empty:
            raise ValueError("Dataset series is empty. Cannot split data.")
        if not isinstance(forecast_steps, int) or forecast_steps < 1:
            raise ValueError("forecast_steps must be a positive integer.")
        if len(self.series) <= forecast_steps:
            raise ValueError(
                f"Dataset is too short ({len(self.series)} rows) "
                f"to create a test set of size {forecast_steps}."
            )

        self.development_data = self.series.iloc[:-forecast_steps]
        self.test_data = self.series.iloc[-forecast_steps:]
        logger.info(
            f"Data split into development set ({len(self.development_data)} rows) "
            f"and test set ({len(self.test_data)} rows)."
        )

    def generate_walk_forward_folds(self, max_window_size: int, n_folds: int) -> List[pd.DataFrame]:
        """
        Generates training folds for walk-forward validation from the development data.
        """
        if self.development_data is None:
            raise ValueError("Data must be split into development and test sets first. Call split_data().")
        if not isinstance(max_window_size, int) or max_window_size < 1:
            raise ValueError("max_window_size must be a positive integer.")
        if not isinstance(n_folds, int) or n_folds < 1:
            raise ValueError("n_folds must be a positive integer.")

        experiment_config = self.config.get("experiments", [{}])[0]
        validation_config = experiment_config.get("validation_setup", {})
        fold_forecast_steps = validation_config.get("forecast_steps", 1)

        data_len = len(self.development_data)

        # Calculate the minimum length required for the first fold
        required_len_for_first_fold = max_window_size + fold_forecast_steps

        if data_len < required_len_for_first_fold:
            logger.warning(
                f"Development data ({data_len} rows) is too short to generate even one fold "
                f"with max_window_size ({max_window_size}) and forecast_steps ({fold_forecast_steps}). "
                f"Required at least: {required_len_for_first_fold} rows."
            )
            return []

        initial_train_len = max(data_len - (n_folds * fold_forecast_steps), max_window_size)

        folds = []
        for i in range(n_folds):
            end_idx = initial_train_len + i * fold_forecast_steps
            if end_idx > data_len:
                break
            fold = self.development_data.iloc[:end_idx]
            folds.append(fold)

        logger.info(f"Generated {len(folds)} walk-forward training folds.")
        if not folds:
            logger.warning("No folds were generated. Check data length and validation parameters.")
        return folds

    def get_data_for_model(self, use_exogenous: bool) -> pd.DataFrame:
        """
        Returns the series DataFrame with or without exogenous features.
        """
        if use_exogenous:
            return self.series
        else:
            return self.series[self.target_columns]

    def get_development_data(self) -> pd.DataFrame:
        """
        Returns the development dataset.
        """
        if self.development_data is None:
            raise ValueError("Data has not been split yet. Call split_data() first.")
        return self.development_data

    def get_test_data(self) -> pd.DataFrame:
        """
        Returns the test dataset.
        """
        if self.test_data is None:
            raise ValueError("Data has not been split yet. Call split_data() first.")
        return self.test_data

    @staticmethod
    def generate_sequential_folds(
            series: pd.DataFrame,
            n_folds: int,
            forecast_steps: int,
            initial_train_size: Optional[int] = None,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Build K sequential folds on the provided `series` with an expanding training window.
        Each holdout has length F. The last fold ends at the end of `series`.

        Let N=len(series), K=n_folds, F=forecast_steps, base=N-K*F.
        Optionally enforce a minimum initial train size via `initial_train_size`.
        For i in [0..K-1]:
          train   = series.iloc[: base + i*F]
          holdout = series.iloc[ base + i*F : base + (i+1)*F ]
        """

        if not isinstance(n_folds, int) or n_folds < 1:
            raise ValueError("n_folds must be a positive integer.")
        if not isinstance(forecast_steps, int) or forecast_steps < 1:
            raise ValueError("forecast_steps must be a positive integer.")

        N = len(series)
        K = n_folds
        F = forecast_steps
        if N < K * F + 1:
            raise ValueError(
                f"Series too short: N={N}, need at least K*F+1={K * F + 1} to form folds."
            )

        base = N - K * F
        if initial_train_size is not None:
            if initial_train_size < 0:
                raise ValueError("initial_train_size must be non-negative.")
            base = max(base, initial_train_size)
            needed = base + K * F
            if needed > N:
                raise ValueError(
                    f"Not enough data: need base+K*F={needed} rows, but N={N}."
                )

        folds: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
        for i in range(K):
            train_end = base + i * F
            val_start = train_end
            val_end = val_start + F
            train_df = series.iloc[:train_end].copy()
            holdout_df = series.iloc[val_start:val_end].copy()
            folds.append((train_df, holdout_df))

        logger.info(f"Generated {len(folds)} sequential folds (F={F}, base_train_len={base}).")
        return folds

    # ---------------------------------------------------------------------------------
    # NEW: Differencing logic (migrated from Preprocessor)
    # ---------------------------------------------------------------------------------

    def _apply_differencing(self, df: pd.DataFrame, diff_config: Dict) -> pd.DataFrame:
        """
        Applies standard and seasonal differencing to the target columns of the DataFrame.
        Determines and stores the differencing orders (d, D) based on the config.
        """
        df_out = df.copy()

        target_cols = self.target_columns
        if not target_cols:
            logger.warning("No target columns specified, skipping differencing.")
            self._diff_state = {'d': 0, 'D': 0, 's': 1}
            return df_out

        # Use the first target column as the representative series for auto-detection
        col_name_for_test = target_cols[0]

        seasonal_period = int(diff_config.get("seasonal_period", 1))
        if seasonal_period <= 0:
            raise ValueError("`seasonal_period` must be > 0")

        # Manual differencing only to avoid data leakage and complexity.
        # We rely on the user to provide 'order' (d) and 'seasonal_order' (D).
        d = diff_config.get("order", 0)
        D = diff_config.get("seasonal_order", 0)

        # Safety check
        if d < 0 or D < 0:
            raise ValueError(f"Differencing orders must be non-negative. Got d={d}, D={D}")

        # Store the determined orders in the state
        self._diff_state = {
            'd': d,
            'D': D,
            's': seasonal_period
        }
        logger.info(f"Applying differencing to all target columns: d={d}, D={D}, s={seasonal_period}")

        if d == 0 and D == 0:
            return df_out  # Return original df

        # Apply the transformations to all target columns
        for col in target_cols:
            series = df_out[col].copy()
            if D > 0:
                for _ in range(D):
                    series = series.diff(seasonal_period)
            if d > 0:
                for _ in range(d):
                    series = series.diff(1)
            df_out[col] = series

        # Note: df_out will contain NaNs at the beginning,
        # which will be handled by dropna() in _prepare_data
        return df_out

    # ---------------------------------------------------------------------------------
    # NEW: Inverse Differencing logic (migrated from Preprocessor)
    # ---------------------------------------------------------------------------------

    def inverse_difference_forecast(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        """
        Reconstructs a forecast DataFrame back to its original (pre-differenced) scale.

        Args:
            forecast_df: A DataFrame of predictions. Must have the same target
                         columns as the dataset.

        Returns:
            The reconstructed DataFrame on the original scale.
        """
        if self._diff_state is None or (self._diff_state['d'] == 0 and self._diff_state['D'] == 0):
            logger.debug("No differencing was applied; returning forecast as-is.")
            return forecast_df

        if self._original_series is None:
            raise RuntimeError("Cannot inverse difference: _original_series was not stored.")

        d = self._diff_state['d']
        D = self._diff_state['D']
        s = self._diff_state['s']

        reconstructed_df = forecast_df.copy()

        # Apply inverse differencing to each target column
        for col_name in self.target_columns:
            if col_name in reconstructed_df.columns:
                logger.debug(f"Inverting differencing for column '{col_name}' (d={d}, D={D}, s={s}).")

                # The context is ALWAYS the full original series for this column
                # This is much simpler and more robust than the Preprocessor's logic
                context_series = self._original_series[col_name].dropna()

                predictions_series = reconstructed_df[col_name]

                reconstructed_series = self._perform_inverse_differencing(
                    predictions_series=predictions_series,
                    d_order=d,
                    s_d_order=D,
                    seasonal_period=s,
                    context_series=context_series
                )
                reconstructed_df[col_name] = reconstructed_series
            else:
                logger.warning(f"Cannot inverse difference column '{col_name}': not found in forecast_df.")

        return reconstructed_df

    def prepare_for_evaluation(
            self,
            y_true: pd.DataFrame,
            y_pred: pd.DataFrame,
            model_used_raw: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare predictions and actuals for evaluation by ensuring both are on the same scale.

        This method handles the complexity of inverse differencing and selecting the correct
        actuals based on whether the model was trained on raw or differenced data.

        Args:
            y_true: True values from eval fold (may be differenced or original)
            y_pred: Predicted values (may be differenced or original)
            model_used_raw: Whether model was trained on raw (undifferenced) data source

        Returns:
            Tuple of (y_true, y_pred) both guaranteed to be on the same scale

        Raises:
            RuntimeError: If differencing was applied but _original_series is missing

        Examples:
            # After model prediction
            y_true_eval, y_pred_eval = dataset.prepare_for_evaluation(
                y_true=eval_fold[targets],
                y_pred=predictions,
                model_used_raw=False
            )
            # Both are now in original scale, ready for metric calculation
        """
        # Guard 1: Model used raw data - no transformation needed
        if model_used_raw:
            logger.debug("[prepare_for_evaluation] Model used raw data - no transformation needed")
            # Filter y_true to target columns only to match y_pred shape
            return y_true[self.target_columns], y_pred

        # Guard 2: Check if differencing was applied
        was_differenced = (
                self._diff_state is not None and
                (self._diff_state.get('d', 0) > 0 or self._diff_state.get('D', 0) > 0)
        )

        if not was_differenced:
            logger.debug("[prepare_for_evaluation] No differencing applied - returning as-is")
            # Filter y_true to target columns only to match y_pred shape
            return y_true[self.target_columns], y_pred

        # Differencing was applied - need to convert both to original scale
        logger.debug(
            f"[prepare_for_evaluation] Converting to original scale "
            f"(d={self._diff_state.get('d')}, D={self._diff_state.get('D')})"
        )

        # Step 1: Convert predictions to original scale
        try:
            y_pred = self.inverse_difference_forecast(y_pred)
        except Exception as e:
            raise RuntimeError(f"Failed to inverse difference predictions: {e}") from e

        # Step 2: Get actuals in original scale
        if self._original_series is None:
            raise RuntimeError(
                "Cannot get original actuals: _original_series is None. "
                "This indicates differencing was enabled but original series was not stored."
            )

        try:
            y_true = self._original_series.loc[y_true.index][self.target_columns]
        except Exception as e:
            raise RuntimeError(f"Failed to get original actuals: {e}") from e

        return y_true, y_pred

    # --- Helper methods for inverse differencing (migrated from Preprocessor) ---

    def _slice_context_for(
        self,
        series: pd.Series,
        context: pd.Series,
        d_order: int,
        s_d_order: int,
        seasonal_period: int,
    ) -> pd.Series:
        """
        Trim the context to the minimally required tail and avoid index overlap
        with the segment we are reconstructing.
        """
        if context is None or context.empty:
            return context
        ctx = context.copy()
        if isinstance(series.index, pd.DatetimeIndex) and len(series.index) > 0:
            start_ts = series.index[0]
            # keep strictly earlier history only
            ctx = ctx.loc[ctx.index < start_ts]
        # minimal anchors required for inversion
        min_len = max(1, d_order + s_d_order * max(1, seasonal_period))
        if len(ctx) > min_len:
            ctx = ctx.iloc[-min_len:]
        return ctx

    def _inverse_standard_diff_autoregressive(
        self, series: pd.Series, d_order: int, context_series: pd.Series
    ) -> pd.Series:
        """
        Invert **standard** differencing of order `d` in an autoregressive fashion.
        """
        if d_order == 0 or series.empty:
            return series

        rec = series.copy()
        for k in range(d_order, 0, -1):
            # context for this level = (k-1)-times differenced
            ctx_k = context_series.copy()
            for _ in range(k - 1):
                ctx_k = ctx_k.diff().dropna()
            if ctx_k.empty:
                raise ValueError("Context too short for standard inverse differencing.")
            seed = float(ctx_k.iloc[-1])
            # invert 1st difference
            rec = rec.cumsum() + seed
        return rec

    def _calculate_inverse_seasonal_diff(
            self,
            series: pd.Series,
            s_d_order: int,
            seasonal_period: int,
            context_series: pd.Series,
            is_future: bool = False,
    ) -> pd.Series:
        """
        Invert **seasonal** differencing (order D, period s) in an autoregressive fashion.
        """
        if s_d_order == 0 or series.empty:
            return series
        if seasonal_period <= 0:
            raise ValueError("seasonal_period must be > 0")

        rec = series.copy()
        # Invert from the highest seasonal order down to 1
        for k in range(s_d_order, 0, -1):
            ctx_k = context_series.copy()
            # Context for this level = (k-1)-times seasonally differenced
            for _ in range(k - 1):
                ctx_k = ctx_k.diff(periods=seasonal_period).dropna()

            if is_future:
                if len(ctx_k) < seasonal_period:
                    raise ValueError("Context too short for seasonal inverse differencing.")
                ctx_k = ctx_k.iloc[-seasonal_period:]
            else:
                ctx_k = self._slice_context_for(rec, ctx_k, d_order=0, s_d_order=1, seasonal_period=seasonal_period)
                if len(ctx_k) < seasonal_period:
                    raise ValueError("Context too short for seasonal inverse differencing.")

            combined = pd.concat([ctx_k, rec])
            original_diffs = rec.copy()
            rec_idx_ptr = 0

            start_combined_idx = len(ctx_k)
            for t in range(start_combined_idx, len(combined)):
                seed_value = combined.iloc[t - seasonal_period]
                combined.iloc[t] = original_diffs.iloc[rec_idx_ptr] + seed_value
                rec_idx_ptr += 1

            rec = combined.iloc[-len(series):]

        return rec

    def _perform_inverse_differencing(
            self,
            predictions_series: pd.Series,
            d_order: int,
            s_d_order: int,
            seasonal_period: int,
            context_series: pd.Series,
    ) -> pd.Series:
        """
        Reconstructs the original series from differenced predictions by inverting
        both standard and seasonal differencing.
        """
        if predictions_series.empty or (d_order == 0 and s_d_order == 0):
            return predictions_series.copy()
        if context_series is None or context_series.empty:
            raise ValueError("A valid context_series is required for inverse differencing.")
        min_required = max(1, d_order + s_d_order * max(1, seasonal_period))
        if len(context_series) < min_required:
            raise ValueError(f"Context series too short. Required: {min_required}, Available: {len(context_series)}")

        original_context = context_series

        is_future = False
        try:
            is_future = (
                    isinstance(predictions_series.index, pd.DatetimeIndex)
                    and isinstance(original_context.index, pd.DatetimeIndex)
                    and len(predictions_series) > 0
                    and len(original_context) > 0
                    and predictions_series.index[0] > original_context.index[-1]
            )
        except Exception:
            is_future = False

        # Prepare the context for the standard inversion step.
        context_for_std_inverse = original_context.copy()
        if d_order > 0 and s_d_order > 0:
            for _ in range(s_d_order):
                context_for_std_inverse = context_for_std_inverse.diff(periods=seasonal_period)
            context_for_std_inverse.dropna(inplace=True)

        # --- Step 1: Invert STANDARD differencing (d) ---
        if d_order > 0:
            std_anchor_ctx = self._slice_context_for(
                predictions_series, context_for_std_inverse, d_order=d_order, s_d_order=0, seasonal_period=1
            )
            reconstructed_series = self._inverse_standard_diff_autoregressive(
                predictions_series, d_order, std_anchor_ctx
            )
        else:
            reconstructed_series = predictions_series.copy()

        # --- Step 2: Invert SEASONAL differencing (D) ---
        if s_d_order > 0:
            reconstructed_series = self._calculate_inverse_seasonal_diff(
                reconstructed_series, s_d_order, seasonal_period, original_context, is_future=is_future
            )

        if len(reconstructed_series) == len(predictions_series):
            reconstructed_series.index = predictions_series.index

        return reconstructed_series
