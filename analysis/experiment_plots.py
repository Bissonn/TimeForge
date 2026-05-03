"""Statistical analysis and performance plots."""
from typing import Optional
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
from scipy import stats
import seaborn as sns

sns.set_style("whitegrid")

logger = logging.getLogger(__name__)


class ExperimentAnalyzer:
    """Analyze experimental results with statistical tests."""

    def __init__(self, results_dir: str = "results/experiments"):
        self.results_dir = Path(results_dir)

    def load_results(self, experiment_name: str) -> pd.DataFrame:
        """Load experiment results."""
        csv_path = self.results_dir / f"{experiment_name}_results.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path)

        json_path = self.results_dir / f"{experiment_name}_results.json"
        if json_path.exists():
            return pd.read_json(json_path)

        raise FileNotFoundError(f"Results not found: {experiment_name}")

    def plot_performance_vs_window_size(self,
                                        experiment_names: List[str],
                                        models: List[str],
                                        metric: str = 'mse',
                                        output_path: str = "results/plots/performance_vs_window.png"):
        """Plot performance vs window size."""
        results_by_model = {model: {'windows': [], 'means': [], 'stds': []}
                            for model in models}

        for exp_name in experiment_names:
            try:
                df = self.load_results(exp_name)
                window_size = int(exp_name.split('_w')[-1])

                for model in models:
                    model_data = df[df['model_name'] == model]

                    if len(model_data) > 0:
                        mean_metric = model_data[metric].mean()
                        std_metric = model_data[metric].std()

                        results_by_model[model]['windows'].append(window_size)
                        results_by_model[model]['means'].append(mean_metric)
                        results_by_model[model]['stds'].append(std_metric)

            except FileNotFoundError:
                logger.warning(f"Experiment {exp_name} not found")

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.tab10(np.arange(len(models)))

        for model, color in zip(models, colors):
            data = results_by_model[model]

            if len(data['windows']) > 0:
                sorted_idx = np.argsort(data['windows'])
                windows = np.array(data['windows'])[sorted_idx]
                means = np.array(data['means'])[sorted_idx]
                stds = np.array(data['stds'])[sorted_idx]

                ax.errorbar(windows, means, yerr=stds,
                            label=model, marker='o', markersize=8,
                            linewidth=2, capsize=5, color=color, alpha=0.8)

        ax.set_xlabel("Window Size", fontsize=12)
        ax.set_ylabel(f"{metric.upper()}", fontsize=12)
        ax.set_title("Performance vs Context Length", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot: {output_path}")
        plt.close()

    def plot_statistical_comparison(self,
                                    experiment_name: str,
                                    model1: str,
                                    model2: str,
                                    metric: str = 'mse',
                                    output_path: Optional[str] = None):
        """Plot statistical comparison between two models."""
        df = self.load_results(experiment_name)

        model1_results = df[df['model_name'] == model1][metric].values
        model2_results = df[df['model_name'] == model2][metric].values

        t_stat, p_value = stats.ttest_rel(model1_results, model2_results)

        # Cohen's d
        mean_diff = np.mean(model1_results) - np.mean(model2_results)
        pooled_std = np.sqrt((np.var(model1_results) + np.var(model2_results)) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std != 0 else 0

        fig, ax = plt.subplots(figsize=(8, 6))

        data_to_plot = [model1_results, model2_results]
        labels = [model1, model2]

        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                        showmeans=True, meanline=True)

        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_ylabel(f"{metric.upper()}", fontsize=12)
        ax.set_title(f"Statistical Comparison: {experiment_name}", fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')

        significance = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else "ns"))

        stats_text = (f"Paired t-test:\n"
                      f"t = {t_stat:.3f}\n"
                      f"p = {p_value:.4f} {significance}\n"
                      f"Cohen's d = {cohens_d:.3f}")

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        if output_path is None:
            output_path = f"results/plots/statistical_{experiment_name}.png"

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot: {output_path}")
        plt.close()

        logger.info(f"\n=== Statistical Test ===")
        logger.info(f"{model1}: {np.mean(model1_results):.4f} ± {np.std(model1_results):.4f}")
        logger.info(f"{model2}: {np.mean(model2_results):.4f} ± {np.std(model2_results):.4f}")
        logger.info(f"t = {t_stat:.3f}, p = {p_value:.4f} {significance}")
        logger.info(f"Cohen's d = {cohens_d:.3f}")