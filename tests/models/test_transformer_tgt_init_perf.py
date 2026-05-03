import pytest
import pandas as pd
import numpy as np
import torch
import logging

from models.factory import ModelFactory
from utils.dataset import TimeSeriesDataset
import models.transformer

logger = logging.getLogger(__name__)


class TestResearchTgtInit:
    """
    Research/Behavioral test suite to analyze the effectiveness of different
    target initialization strategies (`tgt_init`) across various data patterns.

    This suite performs PURE EVALUATION. It does not contain assertions on accuracy.
    It runs ALL strategies on ALL datasets to provide a complete performance landscape.
    """

    results = {}

    # Full list of strategies to test in every scenario
    ALL_STRATEGIES = [
        "zeros", "last_value", "mean", "median", "trend",
        "seasonal", "copy_history"
    ]

    @classmethod
    def teardown_class(cls):
        print("\n\n" + "=" * 100)
        print("RESEARCH SUMMARY: TGT_INIT PERFORMANCE EVALUATION (MSE)")
        print("=" * 100)

        if not cls.results:
            print("No results collected.")
            return

        df_results = pd.DataFrame(cls.results).T
        df_results = df_results.sort_index()
        cols = sorted(df_results.columns)
        df_results = df_results[cols]

        print(df_results.to_string(float_format="{:.4f}".format))
        print("-" * 100)
        print("NOTE: Lower MSE is better.")
        print("=" * 100 + "\n")

    # --- SYNTHETIC DATA GENERATORS ---

    @staticmethod
    def generate_linear_trend_data(n=200, slope=0.5, noise=0.1):
        """Generates data with a strong linear trend."""
        t = np.arange(n)
        y = slope * t + np.random.normal(0, noise, size=n)
        return pd.DataFrame({"y": y}, index=pd.date_range("2020-01-01", periods=n))

    @staticmethod
    def generate_step_data(n=200, step_prob=0.05, step_size=5.0):
        """Generates Random Walk / Step function data (sudden level shifts)."""
        y = np.zeros(n)
        val = 0
        for i in range(n):
            if np.random.rand() < step_prob:
                val += np.random.choice([-1, 1]) * step_size
            y[i] = val + np.random.normal(0, 0.1)
        return pd.DataFrame({"y": y}, index=pd.date_range("2020-01-01", periods=n))

    @staticmethod
    def generate_stationary_outlier_data(n=200, mean=10.0, outlier_prob=0.1, outlier_mag=20.0):
        """Generates stationary data with significant outliers (spikes)."""
        y = np.random.normal(mean, 1.0, size=n)
        mask = np.random.rand(n) < outlier_prob
        y[mask] += outlier_mag * np.random.choice([-1, 1], size=mask.sum())
        return pd.DataFrame({"y": y}, index=pd.date_range("2020-01-01", periods=n))

    @staticmethod
    def generate_seasonal_data(n=300, period=24, amplitude=10.0, noise=1.0):
        """Generates data with clear seasonality (Sine wave)."""
        t = np.arange(n)
        y = amplitude * np.sin(2 * np.pi * t / period) + np.random.normal(0, noise, size=n)
        return pd.DataFrame({"y": y}, index=pd.date_range("2020-01-01", periods=n, freq='h'))

    # --- EXPERIMENT RUNNER ---

    def run_experiment(self, data_df, tgt_init, use_revin, forecast_steps=10, window_size=20, epochs=5,
                       seasonal_period=1):
        """
        Trains a Transformer model with specific configuration.
        Returns the MSE on the test set.
        """
        torch.manual_seed(42)
        np.random.seed(42)

        ds = TimeSeriesDataset(
            "synth",
            {},
            num_features=1,
            data=data_df,
            columns=["y"],
            past_covariates=[],
            future_covariates=[]
        )
        ds.split_data(forecast_steps=forecast_steps)

        model_params = {
            "architecture": "encoder-decoder",
            "strategy": "direct",
            "hidden_size": 32,
            "num_heads": 4,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "dim_ff_multiplier": 2.0,
            "dropout": 0.0,
            "epochs": epochs,
            "batch_size": 8,
            "learning_rate": 0.01,
            "early_stopping_patience": 10,

            # Variables under test
            "tgt_init": tgt_init,
            "use_revin": use_revin,
            "revin_affine": True,
            "revin_eps": 1e-5,
            "preprocessing": {"preprocessing_groups": []},

            # Pass seasonal period for the 'seasonal' strategy
            "seasonal_period": seasonal_period
        }

        try:
            model = ModelFactory.create(
                "transformer", f"tf_{tgt_init}_{use_revin}", model_params,
                num_features=1, forecast_steps=forecast_steps, window_size=window_size, dataset=ds
            )

            model.fit(ds.development_data, is_final_fit=False, dataset=ds)

            # Evaluation on the held-out test set (the last window)
            history = ds.development_data.iloc[-window_size:]
            true_future = ds.test_data

            preds = model.predict(history)

            # Calculate MSE
            mse = np.mean((preds.values - true_future.values) ** 2)
            return mse

        except Exception as e:
            logger.error(f"Experiment failed for tgt_init={tgt_init}, RevIN={use_revin}. Error: {e}")
            return np.nan

    # --- TEST SCENARIOS ---

    @pytest.mark.integration
    def test_scenario_linear_trend(self):
        """
        Scenario 1: Strong Linear Trend.
        """
        data = self.generate_linear_trend_data(n=150, slope=2.0)
        revin_settings = [False, True]

        for revin in revin_settings:
            row_name = f"Linear Trend [RevIN={'ON' if revin else 'OFF'}]"
            self.results[row_name] = {}
            print(f"\nRunning: {row_name}")
            for strat in self.ALL_STRATEGIES:
                mse = self.run_experiment(data, strat, use_revin=revin, epochs=5)
                self.results[row_name][strat] = mse
                print(f"  > tgt_init='{strat}': MSE = {mse:.4f}")

    @pytest.mark.integration
    def test_scenario_step_function(self):
        """
        Scenario 2: Step Function / Random Walk.
        """
        data = self.generate_step_data(n=150)
        revin_settings = [False, True]

        for revin in revin_settings:
            row_name = f"Step Function [RevIN={'ON' if revin else 'OFF'}]"
            self.results[row_name] = {}
            print(f"\nRunning: {row_name}")
            for strat in self.ALL_STRATEGIES:
                mse = self.run_experiment(data, strat, use_revin=revin, epochs=5)
                self.results[row_name][strat] = mse
                print(f"  > tgt_init='{strat}': MSE = {mse:.4f}")

    @pytest.mark.integration
    def test_scenario_outliers(self):
        """
        Scenario 3: Stationary data with Outliers.
        """
        data = self.generate_stationary_outlier_data(n=200, outlier_prob=0.1)
        revin_settings = [False, True]

        for revin in revin_settings:
            row_name = f"Outliers [RevIN={'ON' if revin else 'OFF'}]"
            self.results[row_name] = {}
            print(f"\nRunning: {row_name}")
            for strat in self.ALL_STRATEGIES:
                mse = self.run_experiment(data, strat, use_revin=revin, epochs=5)
                self.results[row_name][strat] = mse
                print(f"  > tgt_init='{strat}': MSE = {mse:.4f}")

    @pytest.mark.integration
    def test_scenario_seasonality(self):
        """
        Scenario 4: Seasonality (Sine Wave).
        Here we ensure window_size > seasonal_period to allow 'seasonal' init to work.
        """
        period = 24
        # Generate 24h period data
        data = self.generate_seasonal_data(n=300, period=period, amplitude=10.0)
        revin_settings = [False, True]

        # CRITICAL: Window size MUST be larger than period for Seasonal Init to trigger
        window_size = 50  # > 24
        forecast_steps = 24  # Predict full cycle

        for revin in revin_settings:
            row_name = f"Seasonality [RevIN={'ON' if revin else 'OFF'}]"
            self.results[row_name] = {}
            print(f"\nRunning: {row_name}")
            for strat in self.ALL_STRATEGIES:
                mse = self.run_experiment(
                    data, strat, use_revin=revin, epochs=5,
                    window_size=window_size,
                    forecast_steps=forecast_steps,
                    seasonal_period=period
                )
                self.results[row_name][strat] = mse
                print(f"  > tgt_init='{strat}': MSE = {mse:.4f}")