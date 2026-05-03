#!/usr/bin/env python
"""
Current-feature-only analysis pipeline with CLI.

Supported artifacts:
- gradient monitor CSV files:
    {model_name}_fold_{fold_idx}_w{window_size}_gradients.csv
- attention NPZ + metadata sidecar:
    {model_name}_fold_{fold_idx}_w{window_size}_attention.npz
    {model_name}_fold_{fold_idx}_w{window_size}_attention_metadata.json
- backtest metrics:
    backtest_metrics.json

Outputs:
- gradient_monitor_raw_data.csv
- gradient_monitor_step_summary.csv
- gradient_monitor_component_norms.csv
- gradient_monitor_epoch_summary.csv
- gradient_monitor_model_window_summary.csv
- gradient_monitor_stability_ranking.csv
- gradient_monitor_total_norm.png
- gradient_monitor_component_norms.png
- gradient_monitor_encoder_head_ratio.png
- attention_matrix_w{w}.csv
- attention_heatmap_w{w}.png
- attention_artifact_summary.csv
- effective_receptive_field.png
- aggregated_performance.csv
- performance_{mse,mae,rmse}.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# CLI
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze current gradient-monitor and attention artifacts."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="results",
        help="Base directory containing experiment result folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/analysis",
        help="Directory where analysis outputs will be saved.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model name filter, e.g. transformer_enc_only_direct",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="Optional window size filter, e.g. 96",
    )
    parser.add_argument(
        "--skip-gradients",
        action="store_true",
        help="Skip gradient monitor analysis.",
    )
    parser.add_argument(
        "--skip-attention",
        action="store_true",
        help="Skip attention analysis.",
    )
    parser.add_argument(
        "--skip-performance",
        action="store_true",
        help="Skip performance analysis.",
    )
    return parser.parse_args()


# =========================================================
# Scanning
# =========================================================

def scan_results_structure(base_dir: str = "results") -> Dict[str, Any]:
    """
    Traverse results directory structure and gather current-format artifact paths.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Directory {base_path} does not exist.")
        return {
            "models": [],
            "window_sizes": [],
            "experiments": [],
            "gradient_csv_files": [],
            "attention_files": [],
            "attention_meta_files": [],
            "metrics_data": [],
            "attn_lookup": {},
            "attn_meta_lookup": {},
        }

    detected_data = {
        "models": set(),
        "window_sizes": set(),
        "experiments": set(),
        "gradient_csv_files": [],
        "attention_files": [],
        "attention_meta_files": [],
        "metrics_data": [],
        "attn_lookup": {},       # (model_name, window_size) -> npz path
        "attn_meta_lookup": {},  # (model_name, window_size) -> metadata path
    }

    grad_csv_pattern = re.compile(r"(.+)_fold_(\d+)_w(\d+)_gradients\.csv")
    attn_pattern = re.compile(r"(.+)_fold_(\d+)_w(\d+)_attention\.npz")
    attn_meta_pattern = re.compile(r"(.+)_fold_(\d+)_w(\d+)_attention_metadata\.json")

    for exp_dir in base_path.iterdir():
        if not exp_dir.is_dir():
            continue

        detected_data["experiments"].add(exp_dir.name)

        exp_w_match = re.search(r"w(\d+)", exp_dir.name)
        exp_window = int(exp_w_match.group(1)) if exp_w_match else None

        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir():
                continue

            # --- 1. Metrics ---
            metrics_file = run_dir / "backtest_metrics.json"
            if metrics_file.exists():
                try:
                    with open(metrics_file, "r", encoding="utf-8") as f:
                        metrics = json.load(f)

                    match_date = re.search(r"_\d{8}_\d{6}_\d{3}$", run_dir.name)
                    if match_date:
                        model_name_guess = run_dir.name[:match_date.start()]
                    else:
                        model_name_guess = run_dir.name

                    detected_data["models"].add(model_name_guess)
                    if exp_window is not None:
                        detected_data["window_sizes"].add(exp_window)

                    row = {
                        "experiment": exp_dir.name,
                        "model": model_name_guess,
                        "window_size": exp_window,
                        **metrics,
                    }
                    detected_data["metrics_data"].append(row)

                except Exception as e:
                    print(f"Error reading metrics from {metrics_file}: {e}")

            # --- 2. Gradient monitor CSV ---
            grad_dir = run_dir / "gradients"
            if grad_dir.exists():
                for f in grad_dir.glob("*.csv"):
                    match = grad_csv_pattern.match(f.name)
                    if match:
                        model_name = match.group(1)
                        w_size = int(match.group(3))
                        detected_data["models"].add(model_name)
                        detected_data["window_sizes"].add(w_size)
                        detected_data["gradient_csv_files"].append(str(f))

            # --- 3. Attention artifacts ---
            attn_dir = run_dir / "attention"
            if attn_dir.exists():
                for f in attn_dir.glob("*.npz"):
                    match = attn_pattern.match(f.name)
                    if match:
                        model_name = match.group(1)
                        fold_idx = int(match.group(2))
                        w_size = int(match.group(3))

                        detected_data["models"].add(model_name)
                        detected_data["window_sizes"].add(w_size)
                        detected_data["attention_files"].append(str(f))

                        key = (model_name, w_size)
                        if key not in detected_data["attn_lookup"] or fold_idx == 1:
                            detected_data["attn_lookup"][key] = str(f)

                for f in attn_dir.glob("*_metadata.json"):
                    match = attn_meta_pattern.match(f.name)
                    if match:
                        model_name = match.group(1)
                        fold_idx = int(match.group(2))
                        w_size = int(match.group(3))

                        detected_data["attention_meta_files"].append(str(f))
                        key = (model_name, w_size)
                        if key not in detected_data["attn_meta_lookup"] or fold_idx == 1:
                            detected_data["attn_meta_lookup"][key] = str(f)

    detected_data["models"] = sorted(list(detected_data["models"]))
    detected_data["window_sizes"] = sorted(list(detected_data["window_sizes"]))
    detected_data["experiments"] = sorted(list(detected_data["experiments"]))
    return detected_data


def filter_scanned_data(
    data: Dict[str, Any],
    model_filter: Optional[str] = None,
    window_filter: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Filter scanned artifact registry by model and/or window.
    """
    out = dict(data)

    if model_filter is not None:
        out["models"] = [m for m in out["models"] if m == model_filter]
    if window_filter is not None:
        out["window_sizes"] = [w for w in out["window_sizes"] if w == window_filter]

    def _match_model_window(model_name: str, window: int) -> bool:
        model_ok = (model_filter is None) or (model_name == model_filter)
        window_ok = (window_filter is None) or (window == window_filter)
        return model_ok and window_ok

    grad_pattern = re.compile(r"(.+)_fold_(\d+)_w(\d+)_gradients\.csv")
    attn_pattern = re.compile(r"(.+)_fold_(\d+)_w(\d+)_attention\.npz")
    attn_meta_pattern = re.compile(r"(.+)_fold_(\d+)_w(\d+)_attention_metadata\.json")

    filtered_grad_csv = []
    for fpath in data["gradient_csv_files"]:
        fname = Path(fpath).name
        match = grad_pattern.match(fname)
        if match:
            model_name = match.group(1)
            window = int(match.group(3))
            if _match_model_window(model_name, window):
                filtered_grad_csv.append(fpath)
    out["gradient_csv_files"] = filtered_grad_csv

    filtered_attn = []
    for fpath in data["attention_files"]:
        fname = Path(fpath).name
        match = attn_pattern.match(fname)
        if match:
            model_name = match.group(1)
            window = int(match.group(3))
            if _match_model_window(model_name, window):
                filtered_attn.append(fpath)
    out["attention_files"] = filtered_attn

    filtered_attn_meta = []
    for fpath in data["attention_meta_files"]:
        fname = Path(fpath).name
        match = attn_meta_pattern.match(fname)
        if match:
            model_name = match.group(1)
            window = int(match.group(3))
            if _match_model_window(model_name, window):
                filtered_attn_meta.append(fpath)
    out["attention_meta_files"] = filtered_attn_meta

    out["attn_lookup"] = {
        k: v for k, v in data["attn_lookup"].items()
        if _match_model_window(k[0], k[1])
    }
    out["attn_meta_lookup"] = {
        k: v for k, v in data["attn_meta_lookup"].items()
        if _match_model_window(k[0], k[1])
    }

    metrics_filtered = []
    for row in data["metrics_data"]:
        model_name = row.get("model")
        window = row.get("window_size")
        model_ok = (model_filter is None) or (model_name == model_filter)
        window_ok = (window_filter is None) or (window == window_filter)
        if model_ok and window_ok:
            metrics_filtered.append(row)
    out["metrics_data"] = metrics_filtered

    return out


# =========================================================
# Gradient monitor CSV analysis
# =========================================================

def load_gradient_monitor_csv(files: List[str]) -> pd.DataFrame:
    rows = []
    grad_pattern = re.compile(r"(.+)_fold_(\d+)_w(\d+)_gradients\.csv")

    for fpath in files:
        fname = Path(fpath).name
        match = grad_pattern.match(fname)
        if not match:
            continue

        model_name = match.group(1)
        fold = int(match.group(2))
        window = int(match.group(3))

        try:
            df = pd.read_csv(fpath)
            if df.empty:
                continue

            required_cols = {
                "epoch",
                "step",
                "global_step",
                "batch_loss",
                "total_grad_norm",
                "encoder_grad_norm",
                "head_grad_norm",
            }
            if not required_cols.issubset(df.columns):
                continue

            df = df.copy()
            df["Model"] = model_name
            df["Window"] = window
            df["Fold"] = fold
            df["Source File"] = fname
            rows.append(df)

        except Exception as e:
            print(f"Failed to read {fpath}: {e}")

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def save_gradient_monitor_raw_csv(df: pd.DataFrame, output_csv: str):
    if df.empty:
        return
    df.to_csv(output_csv, index=False)
    print(f"  -> Saved gradient monitor raw data to {Path(output_csv).name}")


def add_gradient_monitor_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    eps = 1e-12

    out["encoder_head_ratio"] = out["encoder_grad_norm"] / (out["head_grad_norm"] + eps)
    out["head_encoder_ratio"] = out["head_grad_norm"] / (out["encoder_grad_norm"] + eps)
    out["encoder_share_of_total"] = out["encoder_grad_norm"] / (out["total_grad_norm"] + eps)
    out["head_share_of_total"] = out["head_grad_norm"] / (out["total_grad_norm"] + eps)

    out["is_vanishing_like"] = (out["total_grad_norm"] <= 1e-3).astype(int)
    out["is_exploding_like"] = (out["total_grad_norm"] >= 1.0).astype(int)

    return out


def save_gradient_monitor_step_summary(df: pd.DataFrame, output_csv: str):
    if df.empty:
        return
    df.to_csv(output_csv, index=False)
    print(f"  -> Saved enriched step summary to {Path(output_csv).name}")


def save_gradient_monitor_epoch_summary(df: pd.DataFrame, output_csv: str):
    if df.empty:
        return

    agg_cols = [
        "total_grad_norm",
        "encoder_grad_norm",
        "head_grad_norm",
        "batch_loss",
        "encoder_head_ratio",
        "head_encoder_ratio",
        "encoder_share_of_total",
        "head_share_of_total",
        "is_vanishing_like",
        "is_exploding_like",
    ]
    agg_cols = [c for c in agg_cols if c in df.columns]

    summary = (
        df.groupby(["Model", "Window", "Fold", "epoch"], as_index=False)[agg_cols]
        .agg(["mean", "max", "min", "std"])
    )

    summary.columns = [
        "_".join([str(x) for x in col if str(x) != ""])
        for col in summary.columns.to_flat_index()
    ]
    summary.to_csv(output_csv, index=False)
    print(f"  -> Saved epoch summary to {Path(output_csv).name}")


def save_gradient_monitor_model_window_summary(df: pd.DataFrame, output_csv: str):
    if df.empty:
        return

    agg_cols = [
        "total_grad_norm",
        "encoder_grad_norm",
        "head_grad_norm",
        "batch_loss",
        "encoder_head_ratio",
        "head_encoder_ratio",
        "encoder_share_of_total",
        "head_share_of_total",
        "is_vanishing_like",
        "is_exploding_like",
    ]
    agg_cols = [c for c in agg_cols if c in df.columns]

    summary = (
        df.groupby(["Model", "Window"], as_index=False)[agg_cols]
        .agg(["mean", "max", "min", "std"])
    )

    summary.columns = [
        "_".join([str(x) for x in col if str(x) != ""])
        for col in summary.columns.to_flat_index()
    ]
    summary.to_csv(output_csv, index=False)
    print(f"  -> Saved model-window summary to {Path(output_csv).name}")


def save_gradient_monitor_stability_ranking(df: pd.DataFrame, output_csv: str):
    if df.empty:
        return

    grouped = df.groupby(["Model", "Window"], as_index=False).agg(
        mean_total_grad_norm=("total_grad_norm", "mean"),
        std_total_grad_norm=("total_grad_norm", "std"),
        max_total_grad_norm=("total_grad_norm", "max"),
        mean_encoder_head_ratio=("encoder_head_ratio", "mean"),
        exploding_like_rate=("is_exploding_like", "mean"),
        vanishing_like_rate=("is_vanishing_like", "mean"),
    )

    grouped["std_total_grad_norm"] = grouped["std_total_grad_norm"].fillna(0.0)

    grouped["stability_score"] = (
        2.0 * grouped["exploding_like_rate"]
        + 1.5 * grouped["vanishing_like_rate"]
        + 0.5 * grouped["std_total_grad_norm"]
        + 0.25 * grouped["max_total_grad_norm"]
    )

    grouped = grouped.sort_values(["stability_score", "Model", "Window"], ascending=[True, True, True])
    grouped.to_csv(output_csv, index=False)
    print(f"  -> Saved stability ranking to {Path(output_csv).name}")


def plot_gradient_monitor_total_norm(df: pd.DataFrame, output_png: str):
    if df.empty:
        return

    df_plot = df.dropna(subset=["total_grad_norm"]).copy()
    if df_plot.empty:
        return

    x_col = "global_step" if "global_step" in df_plot.columns else "step"
    agg = (
        df_plot.groupby(["Model", "Window", x_col], as_index=False)["total_grad_norm"]
        .mean()
    )

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=agg,
        x=x_col,
        y="total_grad_norm",
        hue="Model",
        style="Window",
    )
    plt.yscale("log")
    plt.title("Total Gradient Norm (Avg over folds)")
    plt.xlabel(x_col.replace("_", " ").title())
    plt.ylabel("Total Gradient Norm (Log Scale)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()
    print(f"  -> Generated {Path(output_png).name}")


def plot_gradient_monitor_component_norms(df: pd.DataFrame, output_png: str, output_csv: str):
    if df.empty:
        return

    value_cols = ["encoder_grad_norm", "head_grad_norm", "total_grad_norm"]
    if not all(c in df.columns for c in value_cols):
        return

    x_col = "global_step" if "global_step" in df.columns else "step"

    long_df = df.melt(
        id_vars=[c for c in ["Model", "Window", "Fold", x_col, "epoch", "batch_loss", "Source File"] if c in df.columns],
        value_vars=value_cols,
        var_name="Component",
        value_name="Gradient Norm",
    ).dropna(subset=["Gradient Norm"])

    if long_df.empty:
        return

    long_df.to_csv(output_csv, index=False)
    print(f"  -> Saved component norms to {Path(output_csv).name}")

    agg = (
        long_df.groupby(["Model", "Window", x_col, "Component"], as_index=False)["Gradient Norm"]
        .mean()
    )

    plt.figure(figsize=(13, 7))
    sns.lineplot(
        data=agg,
        x=x_col,
        y="Gradient Norm",
        hue="Model",
        style="Component",
    )
    plt.yscale("log")
    plt.title("Encoder / Head / Total Gradient Norms")
    plt.xlabel(x_col.replace("_", " ").title())
    plt.ylabel("Gradient Norm (Log Scale)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()
    print(f"  -> Generated {Path(output_png).name}")


def plot_gradient_monitor_encoder_head_ratio(df: pd.DataFrame, output_png: str):
    if df.empty or "encoder_head_ratio" not in df.columns:
        return

    x_col = "global_step" if "global_step" in df.columns else "step"
    agg = (
        df.groupby(["Model", "Window", x_col], as_index=False)["encoder_head_ratio"]
        .mean()
    )

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=agg,
        x=x_col,
        y="encoder_head_ratio",
        hue="Model",
        style="Window",
    )
    plt.yscale("log")
    plt.title("Encoder-to-Head Gradient Ratio")
    plt.xlabel(x_col.replace("_", " ").title())
    plt.ylabel("Encoder / Head Ratio (Log Scale)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()
    print(f"  -> Generated {Path(output_png).name}")


# =========================================================
# Attention analysis
# =========================================================

def load_attention_metadata(meta_path: Optional[str]) -> Dict[str, Any]:
    if not meta_path:
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def choose_attention_key(npz_path: str, meta_path: Optional[str] = None) -> Tuple[Optional[str], Dict[str, Any]]:
    meta = load_attention_metadata(meta_path)

    try:
        with np.load(npz_path) as data:
            npz_keys = list(data.keys())
    except Exception:
        return None, meta

    primary = meta.get("primary_map")
    if primary and primary in npz_keys:
        return primary, meta

    meta_keys = meta.get("keys", [])
    for k in meta_keys:
        if k in npz_keys:
            return k, meta

    if npz_keys:
        return npz_keys[0], meta

    return None, meta


def load_attention_map(npz_path: str, meta_path: Optional[str] = None) -> Tuple[Optional[np.ndarray], Optional[str], Dict[str, Any]]:
    key_used, meta = choose_attention_key(npz_path, meta_path)
    if key_used is None:
        return None, None, meta

    try:
        with np.load(npz_path) as data:
            attn = data[key_used]

        if attn.ndim == 4:
            map_data = np.mean(attn, axis=(0, 1))
        elif attn.ndim == 3:
            map_data = np.mean(attn, axis=0)
        elif attn.ndim == 2:
            map_data = attn
        else:
            return None, key_used, meta

        return map_data, key_used, meta

    except Exception:
        return None, key_used, meta


def save_attention_to_csv(npz_path: str, meta_path: Optional[str], output_csv: str) -> Optional[Dict[str, Any]]:
    map_data, key_used, meta = load_attention_map(npz_path, meta_path)
    if map_data is None:
        return None

    pd.DataFrame(map_data).to_csv(output_csv, index=True, header=True)
    print(f"  -> Saved attention matrix CSV to {Path(output_csv).name}")

    summary = {
        "npz_file": Path(npz_path).name,
        "metadata_file": Path(meta_path).name if meta_path else "",
        "key_used": key_used,
        "architecture": meta.get("architecture"),
        "strategy": meta.get("strategy"),
        "window_size": meta.get("window_size"),
        "forecast_steps": meta.get("forecast_steps"),
        "attention_type": meta.get("attention_type"),
        "sampling_mode": meta.get("sampling_mode"),
        "primary_map": meta.get("primary_map"),
        "num_heads": meta.get("num_heads"),
        "num_encoder_layers": meta.get("num_encoder_layers"),
        "num_decoder_layers": meta.get("num_decoder_layers"),
        "q_len": int(map_data.shape[0]),
        "k_len": int(map_data.shape[1]),
    }
    return summary


def manual_plot_attention(npz_path: str, meta_path: Optional[str], output_png: str, title: str):
    map_data, key_used, _ = load_attention_map(npz_path, meta_path)
    if map_data is None:
        print(f"    Warning: could not load attention map from {Path(npz_path).name}")
        return

    pretty_title = title
    if key_used:
        pretty_title += f"\nMap: {key_used}"

    plt.figure(figsize=(10, 8))
    plt.imshow(map_data, cmap="Reds", aspect="auto", origin="upper", vmin=0)
    plt.colorbar(label="Avg Attention Weight")
    plt.title(pretty_title)
    plt.xlabel("Key Position (History Index)")
    plt.ylabel("Query Position")
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()
    print(f"  -> Generated {Path(output_png).name}")


def manual_plot_erf(
    model_name: str,
    window_sizes: List[int],
    attn_lookup: Dict[Tuple[str, int], str],
    attn_meta_lookup: Dict[Tuple[str, int], str],
    output_png: str,
):
    plt.figure(figsize=(12, 6))
    plotted = False

    for w in window_sizes:
        key = (model_name, w)
        if key not in attn_lookup:
            continue

        npz_path = attn_lookup[key]
        meta_path = attn_meta_lookup.get(key)

        map_data, _, _ = load_attention_map(npz_path, meta_path)
        if map_data is None:
            continue

        last_step_attn = map_data[-1, :]
        lags = np.arange(len(last_step_attn))[::-1]

        plt.plot(lags, last_step_attn, label=f"Window {w}", linewidth=2, alpha=0.8)
        plotted = True

    if plotted:
        plt.title(f"Attention Profile (Last Query Step) - {model_name}")
        plt.xlabel("Lag (Steps into Past)")
        plt.ylabel("Average Attention Weight")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, max(window_sizes) if window_sizes else 100)
        plt.tight_layout()
        plt.savefig(output_png)
        plt.close()
        print(f"  -> Generated {Path(output_png).name}")


# =========================================================
# Performance analysis
# =========================================================

def save_performance_outputs(metrics_data: List[Dict[str, Any]], output_dir: Path):
    if not metrics_data:
        print("  Performance analysis skipped (no metrics data)")
        return

    df = pd.DataFrame(metrics_data)
    df.to_csv(output_dir / "aggregated_performance.csv", index=False)
    print(f"  -> Saved aggregated_performance.csv")

    for metric in ["mse", "mae", "rmse"]:
        if metric in df.columns:
            try:
                plt.figure(figsize=(10, 6))
                sns.lineplot(
                    data=df,
                    x="window_size",
                    y=metric,
                    hue="model",
                    style="model",
                    markers=True,
                )
                plt.title(f"Performance vs Context Length ({metric.upper()})")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(str(output_dir / f"performance_{metric}.png"))
                plt.close()
                print(f"  -> Generated performance_{metric}.png")
            except Exception as e:
                print(f"Failed plotting {metric}: {e}")


# =========================================================
# Main
# =========================================================

def run_analysis(
    base_dir: str = "results",
    output_dir: str = "results/analysis",
    model_filter: Optional[str] = None,
    window_filter: Optional[int] = None,
    skip_gradients: bool = False,
    skip_attention: bool = False,
    skip_performance: bool = False,
):
    print("=" * 80)
    print("CURRENT-FEATURE ANALYSIS PIPELINE V5")
    print("=" * 80)

    print(f"\nScanning base directory: {base_dir}")
    data = scan_results_structure(base_dir=base_dir)
    data = filter_scanned_data(
        data,
        model_filter=model_filter,
        window_filter=window_filter,
    )

    if not data["models"] and not data["gradient_csv_files"] and not data["attention_files"]:
        print("ERROR: No matching artifacts detected after filtering.")
        return

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Gradient analysis
    if not skip_gradients:
        print("\n[1/3] Gradient monitor analysis...")
        if data["gradient_csv_files"]:
            grad_df = load_gradient_monitor_csv(data["gradient_csv_files"])
            if not grad_df.empty:
                grad_df = add_gradient_monitor_features(grad_df)

                save_gradient_monitor_raw_csv(
                    grad_df,
                    str(out_path / "gradient_monitor_raw_data.csv"),
                )
                save_gradient_monitor_step_summary(
                    grad_df,
                    str(out_path / "gradient_monitor_step_summary.csv"),
                )
                save_gradient_monitor_epoch_summary(
                    grad_df,
                    str(out_path / "gradient_monitor_epoch_summary.csv"),
                )
                save_gradient_monitor_model_window_summary(
                    grad_df,
                    str(out_path / "gradient_monitor_model_window_summary.csv"),
                )
                save_gradient_monitor_stability_ranking(
                    grad_df,
                    str(out_path / "gradient_monitor_stability_ranking.csv"),
                )

                plot_gradient_monitor_total_norm(
                    grad_df,
                    str(out_path / "gradient_monitor_total_norm.png"),
                )
                plot_gradient_monitor_component_norms(
                    grad_df,
                    str(out_path / "gradient_monitor_component_norms.png"),
                    str(out_path / "gradient_monitor_component_norms.csv"),
                )
                plot_gradient_monitor_encoder_head_ratio(
                    grad_df,
                    str(out_path / "gradient_monitor_encoder_head_ratio.png"),
                )
            else:
                print("  No readable gradient monitor CSV files found.")
        else:
            print("  No gradient monitor CSV files found.")
    else:
        print("\n[1/3] Gradient monitor analysis skipped.")

    # 2. Attention analysis
    if not skip_attention:
        print("\n[2/3] Attention analysis...")
        transformer_models = [m for m in data["models"] if "transformer" in m.lower()]
        attention_summary_rows = []

        if transformer_models and data["attn_lookup"]:
            transformer = transformer_models[0]

            for w in data["window_sizes"]:
                key = (transformer, w)
                if key not in data["attn_lookup"]:
                    continue

                npz_path = data["attn_lookup"][key]
                meta_path = data["attn_meta_lookup"].get(key)

                manual_plot_attention(
                    npz_path,
                    meta_path,
                    str(out_path / f"attention_heatmap_w{w}.png"),
                    f"Attention Map W={w} ({transformer})",
                )

                summary = save_attention_to_csv(
                    npz_path,
                    meta_path,
                    str(out_path / f"attention_matrix_w{w}.csv"),
                )
                if summary is not None:
                    attention_summary_rows.append(summary)

            manual_plot_erf(
                transformer,
                data["window_sizes"],
                data["attn_lookup"],
                data["attn_meta_lookup"],
                str(out_path / "effective_receptive_field.png"),
            )
        else:
            print("  No transformer attention artifacts found.")

        if attention_summary_rows:
            pd.DataFrame(attention_summary_rows).to_csv(
                out_path / "attention_artifact_summary.csv",
                index=False,
            )
            print("  -> Saved attention_artifact_summary.csv")
    else:
        print("\n[2/3] Attention analysis skipped.")

    # 3. Performance analysis
    if not skip_performance:
        print("\n[3/3] Performance analysis...")
        save_performance_outputs(data["metrics_data"], out_path)
    else:
        print("\n[3/3] Performance analysis skipped.")

    print("\nDONE.")
    print(f"Outputs saved to: {out_path.resolve()}")


if __name__ == "__main__":
    args = parse_args()
    run_analysis(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        model_filter=args.model,
        window_filter=args.window,
        skip_gradients=args.skip_gradients,
        skip_attention=args.skip_attention,
        skip_performance=args.skip_performance,
    )
