"""Module for visualizing time series forecasting results.

This module provides the Visualizer class with methods to create and save plots for actual vs.
predicted values and cumulative prediction errors, supporting evaluation of forecasting models.
"""
import logging
from typing import List, Optional, Dict, Union
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Set non-GUI backend BEFORE importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Visualizer:
    """Class for visualizing time series forecasting results and errors."""

    @staticmethod
    def plot_predictions(
        dataset_name: str,
        model_name: str,
        test_data: pd.DataFrame,
        predictions: np.ndarray,
        columns: List[str],
        forecast_steps: int,
        metrics: Optional[Dict[str, float]] = None,
        optimization_metric: str = "mse",
        save_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot actual and predicted values for each feature in the time series.

        Args:
            dataset_name: Name of the dataset for organizing output files.
            model_name: Name of the forecasting model.
            test_data: DataFrame containing actual values with datetime index.
            predictions: Array of predicted values with shape (forecast_steps, num_features).
            columns: List of feature names corresponding to test_data columns.
            forecast_steps: Number of steps to forecast and plot.
            metrics: Optional dictionary of evaluation metrics to display in the plot title.
            save_dir: Optional directory path to save the plots. If None, plotting is skipped.

        Raises:
            ValueError: If inputs are invalid (e.g., empty data, mismatched shapes, invalid forecast_steps).
            RuntimeError: If plot saving fails due to I/O errors.
        """
        if test_data.empty:
            raise ValueError("test_data cannot be empty.")
        if not isinstance(test_data.index, pd.DatetimeIndex):
            raise ValueError("test_data must have a datetime index.")
        if not isinstance(predictions, np.ndarray):
            raise ValueError("predictions must be a numpy array.")
        if forecast_steps < 1:
            raise ValueError("forecast_steps must be positive.")
        if len(test_data) < forecast_steps:
            raise ValueError(f"test_data has {len(test_data)} rows, but forecast_steps is {forecast_steps}.")
        if predictions.shape[0] < forecast_steps:
            raise ValueError(f"predictions has {predictions.shape[0]} rows, but forecast_steps is {forecast_steps}.")
        if len(columns) != test_data.shape[1] or (predictions.ndim > 1 and len(columns) != predictions.shape[1]):
            raise ValueError("Number of columns must match test_data and predictions feature dimensions.")
        if np.any(np.isnan(test_data.values)) or np.any(np.isnan(predictions)):
            logger.warning("NaN values detected in inputs. Raising ValueError.")
            raise ValueError("test_data and predictions cannot contain NaN values.")

        num_cols = min(test_data.shape[1], predictions.shape[1] if predictions.ndim > 1 else 1)

        if not save_dir:
            logger.warning(f"No save_dir provided for plot_predictions ({model_name}). Skipping.")
            return

        output_dir = Path(save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Format metrics for the title if provided
        metrics_title = ""
        if metrics:
            metrics_title = f" (MAE: {metrics.get('mae', 'N/A'):.2f}, RMSE: {metrics.get('rmse', 'N/A'):.2f})"

        try:
            for i in range(num_cols):
                plt.figure(figsize=(10, 6))
                actual_values = test_data.iloc[:forecast_steps, i].values
                predicted_values = (
                    predictions[:forecast_steps, i] if predictions.ndim > 1
                    else predictions[:forecast_steps]
                )
                plt.plot(
                    test_data.index[:forecast_steps],
                    actual_values,
                    label="Actual",
                    marker="o",
                )
                plt.plot(
                    test_data.index[:forecast_steps],
                    predicted_values,
                    label="Predicted",
                    marker="x",
                )
                plt.title(f"{columns[i]} - {model_name}{metrics_title}")
                plt.xlabel("Date")
                plt.ylabel(columns[i])
                plt.legend()
                plt.grid(True)
                output_path = output_dir / f"{columns[i]}_{model_name}_predictions.png"
                plt.savefig(output_path)
                plt.close()
                logger.info(f"Saved prediction plot for {columns[i]} ({model_name}, {dataset_name}) to {output_path}")
        except OSError as e:
            logger.error(f"Failed to save prediction plot: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to save prediction plot: {str(e)}")

    @staticmethod
    def save_plot_predictions_data(
            dataset_name: str,
            model_name: str,
            actuals: pd.DataFrame,
            predictions: pd.DataFrame,
            save_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Saves plot data (actual values and predictions) to a CSV file.
        """
        if not save_dir:
            logger.warning(
                f"No save_dir provided for save_plot_predictions_data ({model_name}). Skipping.")
            return

        output_dir = Path(save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Iterate over columns (target variables)
        for col in actuals.columns:
            # Create a DataFrame for a single variable
            df_out = pd.DataFrame({
                "actual": actuals[col],
                "predicted": predictions[col] if col in predictions.columns else None
            }, index=actuals.index)

            # Calculate errors (optional, since they are on the accumulation chart)
            if df_out["predicted"] is not None:
                df_out["error"] = df_out["actual"] - df_out["predicted"]

            # Save to CSV
            output_path = output_dir / f"{col}_{model_name}_predictions_data.csv"
            df_out.to_csv(output_path)
            logger.info(f"Saved plot data CSV for {col} to {output_path}")

    @staticmethod
    def plot_error_accumulation(
        dataset_name: str,
        model_name: str,
        test_data: pd.DataFrame,
        predictions: np.ndarray,
        columns: List[str],
        forecast_steps: int,
        save_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot cumulative prediction errors for each feature in the time series.

        Args:
            dataset_name: Name of the dataset for organizing output files.
            model_name: Name of the forecasting model.
            test_data: DataFrame containing actual values with datetime index.
            predictions: Array of predicted values with shape (forecast_steps, num_features).
            columns: List of feature names corresponding to test_data columns.
            forecast_steps: Number of steps to forecast and plot.
            save_dir: Optional directory path to save the plots. If None, plotting is skipped.

        Raises:
            ValueError: If inputs are invalid (e.g., empty data, mismatched shapes, invalid forecast_steps).
            RuntimeError: If plot saving fails due to I/O errors.
        """
        if test_data.empty:
            raise ValueError("test_data cannot be empty.")
        if not isinstance(test_data.index, pd.DatetimeIndex):
            raise ValueError("test_data must have a datetime index.")
        if not isinstance(predictions, np.ndarray):
            raise ValueError("predictions must be a numpy array.")
        if forecast_steps < 1:
            raise ValueError("forecast_steps must be positive.")
        if len(test_data) < forecast_steps:
            raise ValueError(f"test_data has {len(test_data)} rows, but forecast_steps is {forecast_steps}.")
        if predictions.shape[0] < forecast_steps:
            raise ValueError(f"predictions has {predictions.shape[0]} rows, but forecast_steps is {forecast_steps}.")
        if len(columns) != test_data.shape[1] or (predictions.ndim > 1 and len(columns) != predictions.shape[1]):
            raise ValueError("Number of columns must match test_data and predictions feature dimensions.")
        if np.any(np.isnan(test_data.values)) or np.any(np.isnan(predictions)):
            logger.warning("NaN values detected in inputs. Raising ValueError.")
            raise ValueError("test_data and predictions cannot contain NaN values.")

        num_cols = min(test_data.shape[1], predictions.shape[1] if predictions.ndim > 1 else 1)

        if not save_dir:
            logger.warning(f"No save_dir provided for plot_error_accumulation ({model_name}). Skipping.")
            return

        output_dir = Path(save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            for i in range(num_cols):
                plt.figure(figsize=(10, 6))
                actual_values = test_data.iloc[:forecast_steps, i].values
                predicted_values = (
                    predictions[:forecast_steps, i] if predictions.ndim > 1
                    else predictions[:forecast_steps]
                )
                errors = np.abs(actual_values - predicted_values)
                if np.any(np.isinf(errors)):
                    logger.warning(f"Infinite errors detected for {columns[i]}. Raising ValueError.")
                    raise ValueError(f"Infinite errors detected for {columns[i]}.")
                cum_errors = np.cumsum(errors)
                plt.plot(test_data.index[:forecast_steps], cum_errors, label="Cumulative Error", marker="o")
                plt.title(f"{dataset_name} - {columns[i]} - {model_name} Error Accumulation (Horizon {forecast_steps})")
                plt.xlabel("Date")
                plt.ylabel("Cumulative Error")
                plt.legend()
                plt.grid(True)
                output_path = output_dir / f"{columns[i]}_{model_name}_error_accumulation.png"
                plt.savefig(output_path)
                plt.close()
                logger.info(
                    f"Saved error accumulation plot for {columns[i]} "
                    f"({model_name}, {dataset_name}) to {output_path}"
                )
        except OSError as e:
            logger.error(f"Failed to save error accumulation plot: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to save error accumulation plot: {str(e)}")

    @staticmethod
    def save_error_accumulation_data(
            dataset_name: str,
            model_name: str,
            actuals: pd.DataFrame,
            predictions: pd.DataFrame,
            save_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Calculates and saves Cumulative Absolute Error data to a CSV file.
        Mirrors the logic from plot_error_accumulation.
        """

        if not save_dir:
            logger.warning(
                f"No save_dir provided for save_error_accumulation_data ({model_name}). Skipping.")
            return

        output_dir = Path(save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for col in actuals.columns:
            if col not in predictions.columns:
                continue

            # Data preparation
            df_out = pd.DataFrame(index=actuals.index)

            # Calculations: Absolute error -> Cumulative sum
            # (This is exactly the same math used in the plot)
            abs_error = np.abs(actuals[col] - predictions[col])
            cumulative_error = abs_error.cumsum()

            df_out["cumulative_absolute_error"] = cumulative_error

            # Save
            output_path = output_dir / f"{col}_{model_name}_error_accumulation_data.csv"
            df_out.to_csv(output_path)
            logger.info(f"Saved cumulative error data CSV for {col} to {output_path}")

    @staticmethod
    def plot_training_history(
            experiment_name: str,
            model_name: str,
            history: Dict[str, List[float]],
            save_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """Plots training and validation loss curves."""
        if history is None or (isinstance(history, dict) and not history.get("train_loss")):
            return

        train_loss = history["train_loss"]
        val_loss = history.get("val_loss", [])
        val_loss_clean = [v for v in val_loss if v is not None]

        epochs = range(1, len(train_loss) + 1)

        if not save_dir:
            logger.warning(f"No save_dir provided for plot_training_history ({model_name}). Skipping.")
            return

        output_dir = Path(save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_loss, label="Training Loss", marker="o")
            if len(val_loss_clean) == len(train_loss):
                plt.plot(epochs, val_loss_clean, label="Validation Loss", marker="x")

            plt.title(f"Training History - {model_name}")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)

            output_path = output_dir / f"{model_name}_learning_curve.png"
            plt.savefig(output_path)
            plt.close()
            logger.info(f"Saved training history plot to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save training history plot: {e}")

    @staticmethod
    def save_training_history_data(
            experiment_name: str,
            model_name: str,
            history: Dict[str, List[float]],
            save_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Saves the loss function progression (train_loss, val_loss) from training history to a CSV file.
        """
        if history is None or (isinstance(history, dict) and "train_loss" not in history):
            return

        if not save_dir:
            logger.warning(
                f"No save_dir provided for save_training_history_data ({model_name}). Skipping.")
            return

        output_dir = Path(save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create DataFrame from history
        # Note: lists might have different lengths (e.g., if validation was less frequent),
        # but in this framework, they are aligned (saved every epoch).
        df_history = pd.DataFrame({
            "epoch": range(1, len(history["train_loss"]) + 1),
            "train_loss": history["train_loss"],
            "val_loss": history.get("val_loss", [None] * len(history["train_loss"]))
        })

        # Set epoch as index
        df_history.set_index("epoch", inplace=True)

        output_path = output_dir / f"{model_name}_learning_curve_data.csv"
        df_history.to_csv(output_path)
        logger.info(f"Saved training history data CSV to {output_path}")
