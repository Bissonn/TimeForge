"""
Manages the configuration and data preparation for experiment runs.
"""
from typing import Dict, List, Any, Optional
import pandas as pd
from utils.dataset import TimeSeriesDataset
import logging

logger = logging.getLogger(__name__)

class DataSpec:
    """
    Represents the data specification required for a single model run.

    This class acts as a "contract" or a blueprint that describes exactly which
    data inputs are needed. It decouples the data requirements from the model
    training logic, as the main script can now create this specification and
    pass it to a data provider.

    Attributes:
        target_columns (List[str]): A list of column names to be treated as targets.
        past_covariates (List[str]): Features known only in history (encoder-only).
        future_covariates (List[str]): Features known in both history and future.
        time_features_to_generate (List[str]): A list of time-based features
            that should be automatically generated from the datetime index.
    """
    def __init__(
        self,
        target_columns: List[str],
        past_covariates: List[str] = None,
        future_covariates: List[str] = None,
        time_features_to_generate: List[str] = None,
    ):
        self.target_columns = target_columns
        self.past_covariates = past_covariates or []
        self.future_covariates = future_covariates or []
        self.time_features_to_generate = time_features_to_generate or []

class ExperimentRun:
    """
    Represents a single, fully configured model run within an experiment.

    This object encapsulates all configuration details for one specific model
    training instance, including its name, parameters, and the dataset it uses.
    It contains the logic for resolving configuration overrides (e.g., local
    vs. global exogenous columns).

    Attributes:
        model_name (str): The name of the model for this run (e.g., 'transformer').
        model_params (Dict): The configuration dictionary for the model.
        dataset_name (str): The name of the dataset to be used.
        _dataset_config (Dict): The raw dataset configuration dictionary.
        _validation_config (Dict): The raw validation setup dictionary.
    """
    def __init__(self, model_config: Dict, dataset_config: Dict, validation_config: Dict, dataset_name: str):
        self.model_name: str = model_config['name']
        self.model_params: Dict = model_config
        self.dataset_name: str = dataset_name
        self._dataset_config = dataset_config
        self._validation_config = validation_config

    def get_data_spec(self) -> DataSpec:
        """
        Creates a data specification by resolving configuration overrides.

        This method implements the core logic for data configuration flexibility.
        It checks for model-specific `exog_*_columns` lists within the
        experiment definition. If found, these local lists are used. If not,
        it falls back to the global definitions in the main `datasets` section.

        Returns:
            DataSpec: An object describing the final data requirements for this run.
        """
        use_exogenous_flag = self.model_params.get("use_exogenous", False)

        final_past_covariates = []
        final_future_covariates = []

        if use_exogenous_flag:
            # ---  The override logic is encapsulated here ---
            # 1. Resolve past_covariates: use local override or fall back to global
            if "past_covariates" in self.model_params:
                final_past_covariates = self.model_params["past_covariates"]
                logger.debug(f"Using local override for past_covariates: {final_past_covariates}")
            else:
                final_past_covariates = self._dataset_config.get("past_covariates", [])

            # 2. Resolve future_covariates: use local override or fall back to global
            if "future_covariates" in self.model_params:
                final_future_covariates = self.model_params["future_covariates"]
                logger.debug(f"Using local override for future_covariates: {final_future_covariates}")
            else:
                final_future_covariates = self._dataset_config.get("future_covariates", [])

        # Time features are always sourced from the global dataset definition,
        # as they are generated dynamically. They are not part of the override logic.
        time_features = self._dataset_config.get("time_features", [])

        return DataSpec(
            target_columns=self._dataset_config.get('columns', []),
            past_covariates=final_past_covariates,
            future_covariates=final_future_covariates,
            time_features_to_generate=time_features
        )

class DataProvider:
    """
    Responsible for preparing the final `run_dataset` based on a DataSpec.

    This class decouples the data preparation logic from the `TimeSeriesDataset`
    class. `TimeSeriesDataset` is responsible for loading and holding the complete
    raw data, while `DataProvider` is responsible for slicing and dicing that
    data to create the specific "view" needed for a particular model run.

    Attributes:
        full_dataset (TimeSeriesDataset): The complete, unfiltered dataset object.
    """
    def __init__(self, full_dataset: TimeSeriesDataset):
        self.full_dataset = full_dataset

    def prepare_run_dataset(
        self,
        spec: DataSpec,
        model_run_config: Dict,
        data_override: Optional[pd.DataFrame] = None
    ) -> TimeSeriesDataset:
        """
        Creates a "view" of the data (`run_dataset`) based on a specification
        and the explicit `use_raw_data_source` flag.

        Args:
            spec (DataSpec): The specification object describing the required data.
            model_run_config (Dict): The model's configuration from the experiment,
                used to check for the `use_raw_data_source` flag.
            data_override (Optional[pd.DataFrame]): If provided, this DataFrame (e.g.,
                a time-based slice like development_data) is used as the
                base for creating the new dataset. If None, the full series is used.

        Returns:
            TimeSeriesDataset: A new dataset instance containing only the necessary data.

        Raises:
            ValueError: If the columns requested by the `DataSpec` are not
                available in the `full_dataset`.
        """

        # --- NEW LOGIC (Step 4) ---
        use_raw = model_run_config.get('use_raw_data_source', False)
        model_name = model_run_config.get('name', 'unknown')

        # 1. Determine the base data source (raw or processed)
        source_to_use: pd.DataFrame
        if use_raw and self.full_dataset._original_series is not None:
            source_to_use = self.full_dataset._original_series
            logger.info(f"Model '{model_name}' requested raw data source. Using original series.")
        else:
            source_to_use = self.full_dataset.series
            if use_raw and self.full_dataset._original_series is None:
                 logger.warning(
                     f"Model '{model_name}' requested raw data, but no original_series "
                     f"was stored (differencing likely disabled). Using default processed series."
                 )
            elif use_raw:
                 logger.warning(
                     f"Model '{model_name}' requested raw data, but _original_series is None. "
                     f"Using default processed series."
                 )

        # 2. Apply the time-slice override if provided
        # data_override (e.g., development_data_full) acts as a time-based "mask"
        # on the selected source_to_use.
        if data_override is not None:
            # We must use the *index* of the override to slice the *correct source*
            source_df = source_to_use.loc[data_override.index]
        else:
            source_df = source_to_use
        # --- END NEW LOGIC ---

        # --- START OF NEW VALIDATION BLOCK ---
        dataset_cfg = self.full_dataset.config.get("datasets", {}).get(self.full_dataset.name, {})
        time_features_generated = dataset_cfg.get("time_features", [])

        if time_features_generated:
            explicitly_defined_exog = set(spec.past_covariates) | set(spec.future_covariates)

            unassigned_features = [
                f for f in time_features_generated
                if f in source_df.columns and f not in explicitly_defined_exog
            ]

            if unassigned_features:
                logger.info(
                    f"Model '{model_name}' is ignoring available generated time features: {unassigned_features}. "
                    f"They will not be included in the run dataset."
                )
        # --- END OF NEW VALIDATION BLOCK ---
        run_columns = spec.target_columns + spec.past_covariates + spec.future_covariates

        # Ensure there are no duplicates while preserving order
        seen = set()
        unique_columns = [c for c in run_columns if not (c in seen or seen.add(c))]


        # Validate that all requested columns exist in the source DataFrame
        missing = [col for col in unique_columns if col not in source_df.columns]
        if missing:
            raise ValueError(
                f"Columns {missing} requested by DataSpec are not available in the source data "
                f"(source keys: {list(source_df.columns)})."
            )

        logger.info(f"Preparing run dataset for '{model_name}' with columns: {unique_columns}")

        # Create a new, lightweight TimeSeriesDataset instance for this specific run
        run_dataset = TimeSeriesDataset(
            dataset_name=self.full_dataset.name,
            config=self.full_dataset.config,
            num_features=len(spec.target_columns),
            data=source_df[unique_columns].copy(),
            columns=spec.target_columns,
            # Pass the resolved exogenous column lists so the model knows what is what
            past_covariates=spec.past_covariates,
            future_covariates=spec.future_covariates,
            freq=self.full_dataset.freq
        )

        # --- NEW: Attach differencing state and raw context if it exists ---
        # This is crucial so the ModelTrainer can call inverse_difference_forecast
        run_dataset._diff_state = self.full_dataset._diff_state
        run_dataset._original_series = self.full_dataset._original_series

        return run_dataset
