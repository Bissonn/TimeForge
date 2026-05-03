"""
Module for the ExperimentRunner class, which orchestrates the execution of a
full experiment from configuration to results.
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from argparse import Namespace

from core.experiment_manager import ExperimentRun, DataProvider, DataSpec
from core.trainer import ModelTrainer
from models.base import TSForecaster
from models.factory import ModelFactory
from utils.dataset import TimeSeriesDataset
from utils.config_utils import get_model_config
import core.artifact_manager as am
import pandas as pd

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Manages the end-to-end execution of a single experiment defined in the
    configuration file.
    """
    def __init__(self, experiment_config: Dict, global_config: Dict, args: Namespace):
        """
        Initializes the ExperimentRunner.

        Args:
            experiment_config (Dict): The specific experiment block from the config.
            global_config (Dict): The entire loaded configuration file.
            args (Namespace): The parsed command-line arguments.
        """
        self.experiment_config = experiment_config
        self.global_config = global_config
        self.args = args
        self.exp_name = experiment_config.get('name', 'untitled_experiment')
        self.validation_setup = self.experiment_config['validation_setup']
        self.forecast_steps = self.validation_setup['forecast_steps']
        self.n_folds = self.validation_setup['n_folds']

    def _prepare_and_run_hpo(self, model_name: str, model_type: str, model_config: Dict, data_spec: DataSpec,
                             data_provider: DataProvider, development_data_full: pd.DataFrame):
        """Set the data and run HPO."""
        logger.info(f"Mode: Hyperparameter Optimization (HPO) for {model_name} ({model_type})")

        # The `development_data_full` passed in is already the correct slice
        # (raw or processed)
        run_dataset = data_provider.prepare_run_dataset(
            data_spec,
            model_config,  # Pass the model config
            data_override=development_data_full.copy()
        )

        # Build unified folds directly on the provided HPO series
        hpo_folds = TimeSeriesDataset.generate_sequential_folds(
            series=run_dataset.series,
            n_folds=self.n_folds,
            forecast_steps=self.forecast_steps,
        )
        run_id = am.get_run_id()
        run_path = am.get_run_path(self.exp_name, model_name, run_id)
        trainer = ModelTrainer(
            model_name=model_name,  # Nazwa instancji (konfiguracji)
            model_type=model_type,  # Typ modelu (dla fabryki)
            run_dataset=run_dataset,
            model_config=model_config,
            validation_params=self.validation_setup,
            run_path=run_path,
            experiment_name=self.exp_name,
            run_id_to_load=None
        )
        trainer.optimize(folds=hpo_folds)

    def _prepare_and_run_evaluation(self, model_name: str, model_type: str, model_config: Dict, data_spec: DataSpec,
                                    data_provider: DataProvider, full_series: pd.DataFrame):
        """
        Prepares data and runs walk-forward cross-validation evaluation (backtesting).
        """
        logger.info(f"Mode: Evaluation (backtest) for {model_name} ({model_type})")

        # The `full_series` passed in is already the correct source (raw or processed)
        run_dataset = data_provider.prepare_run_dataset(
            data_spec,
            model_config, # Pass the model config
            data_override=full_series.copy()
        )

        # 1) Determine the source HPO run ID *before* creating the eval run directory.
        if self.args.run_id:
            source_run_id = self.args.run_id
            logger.info(f"Using provided run ID for evaluation: {source_run_id}")
        else:
            source_run_id = am.find_latest_hpo_run_id(self.exp_name, model_name)
            if source_run_id:
                logger.info(f"Auto-detected latest HPO run for evaluation: {source_run_id}")
            else:
                logger.warning(
                    f"No previous HPO run with best_params.json found for {model_name}. "
                    f"Evaluation will use model params from config.")
                source_run_id = None # Explicitly set to None

        # 2) Now create the unique ID and path for this evaluation run
        eval_run_id = am.get_run_id()
        eval_run_path = am.get_run_path(self.exp_name, model_name, eval_run_id)

        # The trainer receives the correct `run_dataset` (raw or processed)
        trainer = ModelTrainer(
            model_name=model_name,
            model_type=model_type,
            run_dataset=run_dataset,
            model_config=model_config,
            validation_params=self.validation_setup,
            run_path=eval_run_path,
            experiment_name=self.exp_name,
            run_id_to_load=source_run_id,
            force_defaults=self.args.force_defaults
        )

        # 3) Run backtesting evaluation
        visualize = not self.args.no_visualization

        logger.info("=" * 80)
        logger.info("BACKTESTING: Walk-Forward Cross-Validation")
        logger.info("=" * 80)

        backtest_folds = TimeSeriesDataset.generate_sequential_folds(
            series=run_dataset.series,
            n_folds=self.n_folds,
            forecast_steps=self.forecast_steps,
        )
        trainer.evaluate(folds=backtest_folds, visualize=visualize)

    def run(self):
        """
        Executes the entire experiment workflow based on the provided configuration
        and command-line arguments.
        """
        logger.info(f"================== Starting Experiment: {self.exp_name} ==================")

        try:
            # --- 1. Get the dataset name  ---
            dataset_name = self.experiment_config['dataset']

            # --- 2. Load and prepare data ---
            # Load the full dataset once. This applies dataset-level differencing.
            dataset_config = self.global_config['datasets'][dataset_name]
            num_features = len(dataset_config.get('columns', []))
            if num_features == 0:
                raise ValueError(
                    f"Dataset '{dataset_name}' does not specify 'columns' in config. "
                    "Cannot determine num_features."
                )
            full_dataset = TimeSeriesDataset(dataset_name, self.global_config, num_features=num_features)
            data_provider = DataProvider(full_dataset)

            # --- 3.Iterate through each model configuration within the experiment.---
            for model_run_config in self.experiment_config.get('models', []):
                config_name = model_run_config["name"]
                logger.info(f"--- Processing configuration: {config_name} ---")

                if config_name not in self.global_config.get('models', {}):
                    logger.warning(
                        f"Model config '{config_name}' not found in the global 'models' config section. Skipping.")
                    continue

                # Get the base model config
                base_model_config = self.global_config['models'][config_name].copy()

                model_type = base_model_config.get('type')
                if not model_type:
                    raise ValueError(f"Model configuration '{config_name}' is missing required field 'type'.")

                # Overlay experiment-specific model params
                base_model_config.update(model_run_config)
                model_config = base_model_config

                run_config_obj = ExperimentRun(
                    model_config=model_config,
                    dataset_config=self.global_config['datasets'][dataset_name],
                    validation_config=self.validation_setup,
                    dataset_name=dataset_name
                )
                data_spec = run_config_obj.get_data_spec()

                # --- Determine which time-slice to pass as data_override ---
                use_raw = model_config.get('use_raw_data_source', False)

                # Select the correct base series (raw or processed)
                base_series_for_model = (
                    full_dataset._original_series if use_raw and full_dataset._original_series is not None
                    else full_dataset.series
                )
                if base_series_for_model is None:
                    raise RuntimeError(f"Could not determine base series for model {config_name} (series is None).")

                # Create the time-slices *from the selected base series*
                backtest_reserved_size = self.n_folds * self.forecast_steps
                if len(base_series_for_model) <= backtest_reserved_size:
                     raise ValueError(
                         f"Model '{config_name}' data source (len={len(base_series_for_model)}) is too short "
                         f"for {self.n_folds} backtest folds of size {self.forecast_steps}."
                     )

                dev_data_slice = base_series_for_model.iloc[:-backtest_reserved_size]
                full_data_slice = base_series_for_model

                # --- Calling the appropriate method for the mode ---
                try:
                    # --- Mode Dispatch Logic ---
                    if self.args.optimize:
                        optimize_value = model_config.get("optimize")

                        if optimize_value is False:
                            # User EXPLICITLY disabled HPO
                            logger.info(f"Skipping HPO for model '{config_name}' (optimize: false in config).")
                        else:
                            # optimize is missing (None), True, or dict -> run HPO
                            if optimize_value is None or optimize_value is True:
                                # Inject default single-trial config for fixed params
                                logger.info(
                                    f"Running single-trial HPO for '{config_name}' with fixed params from config "
                                    f"(optimize section {'missing' if optimize_value is None else 'set to true'}). "
                                    f"This performs cross-validation + early stopping and saves best_params.json."
                                )
                                model_config["optimize"] = {
                                    "n_trials": 1,
                                    "method": "grid",
                                    "params": {}  # Empty = no parameter search, use YAML config
                                }
                            # else: optimize is already a dict with custom config, use as-is

                            self._prepare_and_run_hpo(
                                model_name=config_name,
                                model_type=model_type,
                                model_config=model_config,
                                data_spec=data_spec,
                                data_provider=data_provider,
                                development_data_full=dev_data_slice
                            )
                    else:  # Default: --evaluate
                        self._prepare_and_run_evaluation(
                            model_name=config_name,
                            model_type=model_type,
                            model_config=model_config,
                            data_spec=data_spec,
                            data_provider=data_provider,
                            full_series=full_data_slice
                        )
                except Exception as e:
                    logger.error(
                        f"Failed to process model {config_name} in mode: "
                        f"{self._get_current_mode()}: {e}", exc_info=True
                    )
                    continue # continue with next model

        except KeyError as e:
            logger.error(
                f"Experiment '{self.exp_name}' is missing a required section: {e}. "
                f"Please check your config file."
            )
            return
        except Exception as e:
            logger.error(f"Experiment '{self.exp_name}' setup error: {e}", exc_info=True)
            return

        logger.info(f"================== Experiment Finished: {self.exp_name} ==================")

    def _get_current_mode(self) -> str:
        if self.args.optimize:
            return "optimize"
        else:
            return "evaluate"