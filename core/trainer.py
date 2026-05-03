"""
Module for the ModelTrainer class, responsible for the lifecycle of a single
model, including HPO, training, and evaluation.
"""

import os
import copy
import logging
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
import pandas as pd
import torch

from models.base import TSForecaster, NeuralTSForecaster
from utils.dataset import TimeSeriesDataset
from models.factory import ModelFactory
from core.artifact_manager import save_json, load_json, find_latest_hpo_run_id
from utils.metrics import calculate_metrics, calculate_metrics_per_channel
from utils.visualizer import Visualizer
from utils.logging_config import get_contextual_logger

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Manages the training, optimization, and evaluation lifecycle for a single
    forecasting model instance.
    """
    def __init__(
            self,
            model_name: str,
            model_type: str,
            run_dataset: TimeSeriesDataset,
            model_config: Dict,
            validation_params: Dict,
            run_path: str,
            experiment_name: str,
            run_id_to_load: Optional[str] = None,
            force_defaults: bool = False,
            run_context: Optional['RunContext'] = None  # NEW
    ):
        """
        Initializes the ModelTrainer.

        Args:
            model_name (str): The name of the model (e.g., ',my_transformer').
            model_type (str): The type of the model (e.g., "transformer").
            run_dataset (TimeSeriesDataset): The dataset prepared for this specific model run.
            model_config (Dict): The configuration block for the model from config.yaml.
            validation_params (Dict): The validation setup block from the experiment config.
            run_path (str): The path to the directory for saving artifacts from *this* run.
            experiment_name (str): The name of the parent experiment.
            run_id_to_load (Optional[str]): A specific run ID to load parameters from (for eval/final train).
            run_context (Optional[RunContext]): Execution context. If None, will be created from run_path.
        """
        self.model_name = model_name
        self.model_type = model_type
        self.run_dataset = run_dataset
        self.model_config = model_config
        self.validation_params = validation_params
        self.run_path = run_path
        self.experiment_name = experiment_name
        self.run_id_to_load = run_id_to_load
        self.force_defaults = force_defaults

        # Create or use provided run_context
        if run_context is None:
            from core.context import RunContext
            from pathlib import Path
            import datetime

            # Extract run_id from run_path if possible, otherwise generate new one
            run_id = run_id_to_load or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            self.run_context = RunContext.from_base_path(
                base_path=Path(run_path),
                run_id=run_id,
                experiment_name=experiment_name
            )
            self.run_context.create_directories()
        else:
            self.run_context = run_context

        # Create a template instance to access class methods like optimize_hyperparameters
        # Note: This template instance won't have fold-specific context
        self.model_template = self._create_model_instance(model_config, fold_idx=None)

    def _create_model_instance(self, params: Dict, fold_idx: Optional[int] = None) -> TSForecaster:
        """
        Helper utility to create a fresh model instance with the correct parameters.

        Args:
            params: Model parameters
            fold_idx: Fold index for creating fold-specific context. If None, uses base context.

        Returns:
            Model instance with appropriate run_context
        """
        # Create fold-specific context if fold_idx provided
        if fold_idx is not None:
            fold_context = self.run_context.with_metadata(
                model_name=self.model_name,
                model_type=self.model_type,
                fold_idx=fold_idx,
                window_size=self.validation_params['window_size']
            )
            fold_context.save_metadata()
        else:
            # Use base context (e.g., for template instance)
            fold_context = self.run_context

        return ModelFactory.create(
            model_type=self.model_type,
            model_name=self.model_name,
            run_context=fold_context,  # Pass fold-specific context
            model_params=params,
            num_features=len(self.run_dataset.target_columns),
            forecast_steps=self.validation_params['forecast_steps'],
            window_size=self.validation_params['window_size'],
            dataset=self.run_dataset,
        )

    def _load_best_params(self) -> Dict:
        """Finds and loads the best hyperparameters from an HPO run."""
        if self.force_defaults:
            logger.warning(
                "Use default params (forced by --force-defaults). HPO results are ignored.")
            return self.model_config.copy()
        run_id = self.run_id_to_load
        if run_id is None:  # Auto-detect only if run_id_to_load was not provided
            logger.info(
                f"No run ID specified. Looking for latest HPO run with artifacts for "
                f"{self.model_name} in {self.experiment_name}..."
            )
            run_id = find_latest_hpo_run_id(self.experiment_name,
                                                     self.model_name)  # Use the new robust function
            if run_id is None:
                logger.warning(f"No previous HPO run found for {self.model_name}. Using default params from config.")
                return self.model_config # Fallback to base config

        logger.info(f"Loading best_params.json from run ID: {run_id}")
        # Construct the path to the specified run
        param_path = os.path.join("results", self.experiment_name, f"{self.model_name}_{run_id}")
        try:
            loaded_params = load_json(param_path, "best_params.json")

            # DEEP MERGE: Merge loaded HPO params over the base model config
            # CRITICAL: Use deep merge to preserve nested dicts like scheduler_config
            # shallow copy + update() would replace entire nested dicts, losing fields like 'type', 'div_factor'
            final_params = copy.deepcopy(self.model_config)

            def deep_merge(base, update):
                """Recursively merge update dict into base dict"""
                for key, value in update.items():
                    if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                        deep_merge(base[key], value)  # Recurse for nested dicts
                    else:
                        base[key] = value  # Overwrite for non-dicts

            deep_merge(final_params, loaded_params)
            return final_params
        except FileNotFoundError:
            logger.error(f"Failed to load params. 'best_params.json' not found in {param_path}.")
            raise

    def optimize(self, folds: List[Tuple[pd.DataFrame, pd.DataFrame]]) -> Tuple[Dict, float]:
        """
        Runs the hyperparameter optimization process.

        Args:
            folds (List[Tuple[pd.DataFrame, pd.DataFrame]]): List of (train, eval) folds.

        Returns:
            A tuple containing the best parameters found and the best validation score.
        """
        logger.info(f"Starting hyperparameter optimization for {self.model_template.__class__.__name__}...")
        if not folds:
            raise ValueError("`folds` must be provided by the ExperimentRunner.")
        best_params, best_loss = self.model_template.optimize_hyperparameters(
            dataset=self.run_dataset,
            model_config=self.model_config,
            validation_params = self.validation_params,
            folds = folds
        )

        # Save the results using the ArtifactManager
        save_json(best_params, self.run_path, "best_params.json")

        logger.info(
            f"Finished optimization for {self.model_template.__class__.__name__}. "
            f"Best CV score: {best_loss:.6f}. Best params: {best_params}"
        )
        return best_params, best_loss

    def _handle_fold_failure(self, fold_idx: int, directory: str, filename: str, message: str, metric_names: List[str],
                             exc_info: bool = False) -> Dict[str, float]:
        """
        Helper: Logs error, saves NaN metrics to file, returns NaN dict.
        Now accepts directory and filename separately to match save_json signature.
        """
        if exc_info:
            logger.error(f"Fold {fold_idx}: {message}", exc_info=True)
        else:
            logger.warning(f"Fold {fold_idx}: {message}")

        # Generate NaN values
        nan_metrics = {metric: np.nan for metric in metric_names}

        # Only persist per-fold failure metrics if explicitly requested.
        if self.validation_params.get("save_fold_metrics", False):
            save_json(nan_metrics, directory, filename)

        return nan_metrics

    def _aggregate_backtest_metrics(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        """
        Aggregates a list of metric dictionaries (Mean + Std).
        Handles both global scalars and nested per-channel dictionaries.
        """
        if not all_metrics:
            return {}

        metrics_df = pd.DataFrame(all_metrics)

        # 1. Aggregate Global Scalar Metrics
        numeric_df = metrics_df.select_dtypes(include=[np.number])

        # Pandas .mean() automatically ignores NaNs
        avg_metrics = numeric_df.mean().to_dict()
        std_metrics = numeric_df.std().to_dict()

        final_metrics = avg_metrics.copy()
        for key, value in std_metrics.items():
            final_metrics[f"{key}_std"] = value

        # 2. Aggregate Per-Channel Metrics (if they exist)
        if 'per_channel' in metrics_df.columns:
            try:
                collected_values = {}

                for record in all_metrics:
                    # Skip failed folds
                    if 'per_channel' not in record or not isinstance(record['per_channel'], dict):
                        continue

                    for metric_name, channels_map in record['per_channel'].items():
                        if metric_name not in collected_values:
                            collected_values[metric_name] = {}

                        for channel_name, val in channels_map.items():
                            if channel_name not in collected_values[metric_name]:
                                collected_values[metric_name][channel_name] = []
                            collected_values[metric_name][channel_name].append(val)

                # Compute Mean/Std for collected lists
                pc_aggregated = {}
                for metric_name, channels_data in collected_values.items():
                    pc_aggregated[metric_name] = {}
                    for channel_name, values in channels_data.items():
                        # Use nanmean/nanstd to ignore NaNs from failed folds
                        valid_count = np.sum(~np.isnan(values))

                        if valid_count == 0:
                            mean_val = np.nan
                            std_val = 0.0
                        else:
                            mean_val = float(np.nanmean(values))
                            std_val = float(np.nanstd(values, ddof=1)) if valid_count > 1 else 0.0

                        pc_aggregated[metric_name][channel_name] = {
                            "mean": mean_val,
                            "std": std_val
                        }

                # Fix RMSE in per_channel_agg: recalculate from MSE
                if 'mse' in pc_aggregated and 'rmse' in pc_aggregated:
                    for channel_name in pc_aggregated['mse'].keys():
                        mse_mean = pc_aggregated['mse'][channel_name]['mean']
                        if not np.isnan(mse_mean):
                            # Correct RMSE: sqrt(mean(MSE))
                            pc_aggregated['rmse'][channel_name]['mean'] = float(np.sqrt(mse_mean))

                            # Recalculate RMSE std from individual fold RMSE values
                            if channel_name in collected_values.get('mse', {}):
                                mse_values = np.array(collected_values['mse'][channel_name])
                                rmse_values = np.sqrt(mse_values)
                                rmse_values = rmse_values[~np.isnan(rmse_values)]
                                if len(rmse_values) > 1:
                                    pc_aggregated['rmse'][channel_name]['std'] = float(np.std(rmse_values, ddof=1))

                final_metrics["per_channel_agg"] = pc_aggregated

            except Exception as e:
                logger.warning(f"Failed to aggregate per-channel metrics: {e}")

        # 3. Fix RMSE aggregation: RMSE should be sqrt(mean(MSE)), not mean(sqrt(MSE))
        # Due to Jensen's inequality (sqrt is concave), mean(sqrt(MSE)) < sqrt(mean(MSE))
        # So the naive average underestimates the true RMSE
        if 'mse' in final_metrics:
            # Recalculate RMSE correctly from aggregated MSE
            final_metrics['rmse'] = float(np.sqrt(final_metrics['mse']))

            # Recalculate RMSE_std from individual fold RMSE values
            if 'mse' in numeric_df.columns:
                fold_rmse_values = np.sqrt(numeric_df['mse'].values)
                # Remove NaNs before computing std
                fold_rmse_values = fold_rmse_values[~np.isnan(fold_rmse_values)]
                if len(fold_rmse_values) > 1:
                    final_metrics['rmse_std'] = float(np.std(fold_rmse_values, ddof=1))
                else:
                    final_metrics['rmse_std'] = 0.0

            logger.debug(f"RMSE corrected: sqrt(MSE={final_metrics['mse']:.6f}) = {final_metrics['rmse']:.6f}")

        return final_metrics

    def _calculate_fold_metrics(
        self,
        aligned_actuals: pd.DataFrame,
        aligned_preds: pd.DataFrame,
        metric_names: List[str],
        fold_idx: int
    ) -> Dict[str, Any]:
        """
        Calculate both global and per-channel metrics for a single fold.

        Args:
            aligned_actuals: Aligned actual values
            aligned_preds: Aligned predicted values
            metric_names: List of metric names to calculate
            fold_idx: Index of the current fold (for logging)

        Returns:
            Dictionary containing global metrics and optionally per-channel metrics
        """
        # Calculate global metrics
        full_metrics = calculate_metrics(aligned_actuals.values, aligned_preds.values)
        metrics = {k: v for k, v in full_metrics.items() if k in metric_names}

        # Calculate per-channel metrics if configured
        if self.validation_params.get('per_channel_metrics', True):
            try:
                metrics["per_channel"] = calculate_metrics_per_channel(
                    aligned_actuals.values,
                    aligned_preds.values,
                    metrics=metric_names,
                    channel_names=list(aligned_actuals.columns)
                )
            except Exception as e:
                logger.warning(f"Fold {fold_idx}: Per-channel calc failed: {e}")

        return metrics

    def _visualize_fold(
        self,
        fold_idx: int,
        aligned_actuals: pd.DataFrame,
        aligned_preds: pd.DataFrame,
        history: Optional[Dict],
        metrics: Dict[str, Any]
    ) -> None:
        """
        Generate visualizations and save data for a single fold.

        Args:
            fold_idx: Index of the current fold
            aligned_actuals: Aligned actual values
            aligned_preds: Aligned predicted values
            history: Training history (if available)
            metrics: Calculated metrics for this fold
        """
        try:
            plot_model_name = f"{self.model_name}_fold_{fold_idx}"
            plot_forecast_steps = len(aligned_actuals)

            # 1. Predictions Plot & Data
            Visualizer.plot_predictions(
                self.experiment_name,
                plot_model_name,
                aligned_actuals,
                aligned_preds.values,
                self.run_dataset.target_columns,
                plot_forecast_steps,
                metrics=metrics,
                save_dir=self.run_context.plots_dir
            )

            Visualizer.save_plot_predictions_data(
                dataset_name=self.experiment_name,
                model_name=plot_model_name,
                actuals=aligned_actuals,
                predictions=aligned_preds,
                save_dir=self.run_context.data_dir
            )

            # 2. Error Accumulation Plot & Data
            Visualizer.plot_error_accumulation(
                self.experiment_name,
                plot_model_name,
                aligned_actuals,
                aligned_preds.values,
                self.run_dataset.target_columns,
                plot_forecast_steps,
                save_dir=self.run_context.plots_dir
            )

            Visualizer.save_error_accumulation_data(
                dataset_name=self.experiment_name,
                model_name=plot_model_name,
                actuals=aligned_actuals,
                predictions=aligned_preds,
                save_dir=self.run_context.data_dir
            )

            # 3. Training History (if available)
            if history:
                Visualizer.plot_training_history(
                    self.experiment_name,
                    plot_model_name,
                    history,
                    save_dir=self.run_context.plots_dir
                )

                Visualizer.save_training_history_data(
                    experiment_name=self.experiment_name,
                    model_name=plot_model_name,
                    history=history,
                    save_dir=self.run_context.data_dir
                )

        except Exception as e:
            logger.error(f"Visualization failed for fold {fold_idx}: {e}", exc_info=True)

    def _finalize_evaluation(self, all_metrics: List[Dict]) -> None:
        """
        Aggregate metrics across folds and save final results.

        Args:
            all_metrics: List of metric dictionaries from all folds
        """
        if not all_metrics:
            logger.error("No metrics calculated.")
            return

        final_aggregated_metrics = self._aggregate_backtest_metrics(all_metrics)

        logger.info(f"--- Backtesting Complete ---")
        global_summary = {k: v for k, v in final_aggregated_metrics.items() if not isinstance(v, dict)}
        logger.info(f"Global Summary: {global_summary}")

        save_json(final_aggregated_metrics, self.run_path, "backtest_metrics.json")

    def _evaluate_single_fold(
        self,
        fold_idx: int,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a single fold: Fit -> Predict -> Transform -> Align -> Metrics.

        Args:
            fold_idx: Index of the current fold
            train_df: Training data for this fold
            test_df: Test data for this fold
            params: Model parameters to use

        Returns:
            Dictionary containing:
                - 'failed': bool indicating if evaluation failed
                - 'metrics': Dict of calculated metrics (or NaN if failed)
                - 'aligned_actuals': DataFrame of aligned actuals (if successful)
                - 'aligned_predictions': DataFrame of aligned predictions (if successful)
                - 'history': Training history dict (if successful and available)
        """
        # Create contextual logger with experiment, model, and fold context
        fold_logger = get_contextual_logger(
            __name__,
            experiment=self.experiment_name,
            model=self.model_name,
            fold=fold_idx
        )

        metric_names = ['mae', 'mse', 'rmse', 'smape', 'mase']
        metric_filename = f"fold_{fold_idx}_metrics.json"
        data_dir = self.run_context.data_dir

        # Guard: Empty test set
        if test_df.empty:
            return {
                'failed': True,
                'metrics': self._handle_fold_failure(
                    fold_idx, data_dir, metric_filename, "Empty test set.", metric_names
                )
            }

        try:
            # 1. Create model instance
            model = self._create_model_instance(params, fold_idx=fold_idx)

            # 2. Fit & Predict
            _, predictions, history = model._fit_and_evaluate_fold(
                train_fold=train_df,
                eval_fold=test_df,
                validation_params=self.validation_params,
                dataset=self.run_dataset,
                is_final_fit=True
            )

            # 3. Inverse Transform & Get Actuals
            # Use dataset's prepare_for_evaluation to handle differencing logic
            try:
                actuals, preds_orig_scale = self.run_dataset.prepare_for_evaluation(
                    y_true=test_df,
                    y_pred=predictions,
                    model_used_raw=self.model_config.get('use_raw_data_source', False)
                )
            except Exception as e:
                return {
                    'failed': True,
                    'metrics': self._handle_fold_failure(
                        fold_idx, data_dir, metric_filename,
                        f"Inverse transform/prep failed: {e}",
                        metric_names, exc_info=True
                    )
                }

            # 4. Align predictions with actuals
            try:
                aligned_preds, aligned_actuals = preds_orig_scale.align(actuals, join='inner', axis=0)
            except Exception as e:
                return {
                    'failed': True,
                    'metrics': self._handle_fold_failure(
                        fold_idx, data_dir, metric_filename,
                        f"Alignment failed: {e}",
                        metric_names, exc_info=True
                    )
                }

            # Guard: Empty alignment
            if aligned_preds.empty:
                return {
                    'failed': True,
                    'metrics': self._handle_fold_failure(
                        fold_idx, data_dir, metric_filename,
                        "Empty alignment.",
                        metric_names
                    )
                }

            # 5. Calculate metrics
            try:
                metrics = self._calculate_fold_metrics(
                    aligned_actuals, aligned_preds, metric_names, fold_idx
                )
            except Exception as e:
                return {
                    'failed': True,
                    'metrics': self._handle_fold_failure(
                        fold_idx, data_dir, metric_filename,
                        f"Metrics calculation failed: {e}",
                        metric_names, exc_info=True
                    )
                }

            # Optionally save per-fold metrics
            if self.validation_params.get("save_fold_metrics", False):
                save_json(metrics, data_dir, metric_filename)

            fold_logger.info(f"Metrics: {metrics}")

            # Return successful result with all data needed for visualization
            return {
                'failed': False,
                'metrics': metrics,
                'aligned_actuals': aligned_actuals,
                'aligned_predictions': aligned_preds,
                'history': history
            }

        except Exception as e:
            return {
                'failed': True,
                'metrics': self._handle_fold_failure(
                    fold_idx, data_dir, metric_filename,
                    f"Unexpected fold failure: {e}",
                    metric_names, exc_info=True
                )
            }

    def evaluate(self, folds: List[Tuple[pd.DataFrame, pd.DataFrame]], visualize: bool = True):
        """
        Runs backtesting evaluation across all folds.

        Orchestrates the evaluation process by delegating single-fold processing
        to helper methods, keeping the main loop clean and focused.

        Args:
            folds: List of (train, test) DataFrame tuples for each fold
            visualize: Whether to generate visualizations for each fold

        Refactored using Extract Method pattern for improved maintainability.
        """
        logger.info(f"Initiating backtesting evaluation for {self.model_name}...")

        # 1. Setup
        best_params = self._load_best_params()
        n_folds = int(self.validation_params['n_folds'])

        if len(folds) != n_folds:
            logger.warning(f"Requested {n_folds} folds but received {len(folds)}.")

        all_metrics = []

        # 2. Main Loop - Clean orchestration
        for fold_idx, (train_fold_df, test_fold_df) in enumerate(folds, start=1):
            logger.info(f"--- Backtesting Fold {fold_idx}/{len(folds)} ---")

            # Evaluate single fold (delegated)
            result = self._evaluate_single_fold(
                fold_idx=fold_idx,
                train_df=train_fold_df,
                test_df=test_fold_df,
                params=best_params
            )

            # Collect metrics (always append, even if failed - NaN metrics)
            all_metrics.append(result['metrics'])

            # Visualize only if evaluation succeeded
            if visualize and not result['failed']:
                self._visualize_fold(
                    fold_idx=fold_idx,
                    aligned_actuals=result['aligned_actuals'],
                    aligned_preds=result['aligned_predictions'],
                    history=result['history'],
                    metrics=result['metrics']
                )

            # GPU Memory Cleanup after each backtest fold
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        # 3. Final Aggregation (delegated)
        self._finalize_evaluation(all_metrics)
    def _log_index_diagnostics(
        self,
        fold_num: int,
        preds_df: pd.DataFrame,
        actuals_full: pd.DataFrame,
        context_df: Optional[pd.DataFrame] = None,
        note: str = ""
    ) -> None:
        """Log detailed index diagnostics to explain why alignment may fail."""
        try:
            # Ensure inputs are DataFrames for consistent .index access
            if not isinstance(preds_df, pd.DataFrame):
                preds_df = preds_df.to_frame()
            if not isinstance(actuals_full, pd.DataFrame):
                actuals_full = actuals_full.to_frame()

            p_idx = preds_df.index
            a_idx = actuals_full.index
            c_idx = context_df.index if context_df is not None else None

            # Basic types and lengths
            logger.debug(
                f"[Diag/Fold {fold_num}] {note} index types: pred={type(p_idx).__name__}, "
                f"actual={type(a_idx).__name__}" + (f", context={type(c_idx).__name__}" if c_idx is not None else "")
            )
            logger.debug(
                f"[Diag/Fold {fold_num}] lengths: pred={len(p_idx)}, actual={len(a_idx)}"
                + (f", context={len(c_idx)}" if c_idx is not None else "")
            )

            # Heads / tails
            def _fmt_head_tail(idx: pd.Index) -> str:
                try:
                    head = list(idx[:3])
                    tail = list(idx[-3:])
                    return f"head={head}, tail={tail}"
                except Exception:
                    return "head/tail-unavailable"

            logger.debug(f"[Diag/Fold {fold_num}] pred idx:   {_fmt_head_tail(p_idx)}")
            logger.debug(f"[Diag/Fold {fold_num}] actual idx: {_fmt_head_tail(a_idx)}")
            if c_idx is not None:
                logger.debug(f"[Diag/Fold {fold_num}] context idx: {_fmt_head_tail(c_idx)}")

            # Frequencies (may be None)
            try:
                p_freq = p_idx.freqstr or pd.infer_freq(p_idx)
            except Exception:
                p_freq = None
            try:
                a_freq = a_idx.freqstr or pd.infer_freq(a_idx)
            except Exception:
                a_freq = None

            logger.debug(f"[Diag/Fold {fold_num}] inferred freq: pred={p_freq}, actual={a_freq}")

            # Intersection and equality
            try:
                inter = p_idx.intersection(a_idx)
                logger.debug(f"[Diag/Fold {fold_num}] index intersection size: {len(inter)}")
            except Exception as e:
                logger.debug(f"[Diag/Fold {fold_num}] intersection failed: {e}")

            try:
                eq = p_idx.equals(a_idx)
                logger.debug(f"[Diag/Fold {fold_num}] indices equal: {eq}")
            except Exception as e:
                logger.debug(f"[Diag/Fold {fold_num}] equality check failed: {e}")
        except Exception as e:
            logger.warning(f"[Diag/Fold {fold_num}] diagnostics failed: {e}", exc_info=True)
