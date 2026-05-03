#!/usr/bin/env python
"""
Current-feature-only analysis pipeline with horizon-aware aggregation.

Supported artifacts:
- gradient monitor CSV files:
    {model_name}_fold_{fold_idx}_w{window_size}_gradients.csv
- attention NPZ + metadata sidecar:
    {model_name}_fold_{fold_idx}_w{window_size}_attention.npz
    {model_name}_fold_{fold_idx}_w{window_size}_attention_metadata.json
- backtest metrics:
    backtest_metrics.json

This version additionally parses:
- window
- horizon
from experiment directory names like:
    grad_test_w96_24
    grad_test_sarima_w336_96
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
        description="Analyze current gradient-monitor and attention artifacts with horizon-aware aggregation."
    )
    parser.add_argument("--base-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="results/analysis")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--skip-gradients", action="store_true")
    parser.add_argument("--skip-attention", action="store_true")
    parser.add_argument("--skip-performance", action="store_true")
    return parser.parse_args()


# =========================================================
# Helpers
# =========================================================

def parse_experiment_window_horizon(exp_name: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse names like:
    - grad_test_w96_24
    - grad_test_sarima_w336_96
    - anything_w1440_192
    """
    m = re.search(r"w(\d+)_(\d+)", exp_name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def safe_model_name_from_run_dir(run_dir_name: str) -> str:
    match_date = re.search(r"_\d{8}_\d{6}_\d{3}$", run_dir_name)
    if match_date:
        return run_dir_name[:match_date.start()]
    return run_dir_name


# =========================================================
# Scanning
# =========================================================

def scan_results_structure(base_dir: str = "results") -> Dict[str, Any]:
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Directory {base_path} does not exist.")
        return {
            "models": [],
            "window_sizes": [],
            "horizons": [],
            "experiments": [],
            "gradient_csv_files": [],
            "attention_files": [],
            "attention_meta_files": [],
            "metrics_data": [],
            "attn_lookup": {},
            "attn_meta_lookup": {},
        }

    detected = {
        "models": set(),
        "window_sizes": set(),
        "horizons": set(),
        "experiments": set(),
        "gradient_csv_files": [],
        "attention_files": [],
        "attention_meta_files": [],
        "metrics_data": [],
        # key = (model_name, window, horizon)
        "attn_lookup": {},
        "attn_meta_lookup": {},
    }

    grad_csv_pattern = re.compile(r"(.+)_fold_(\d+)_w(\d+)_gradients\.csv")
    attn_pattern = re.compile(r"(.+)_fold_(\d+)_w(\d+)_attention\.npz")
    attn_meta_pattern = re.compile(r"(.+)_fold_(\d+)_w(\d+)_attention_metadata\.json")

    for exp_dir in base_path.iterdir():
        if not exp_dir.is_dir():
            continue

        exp_name = exp_dir.name
        exp_window, exp_horizon = parse_experiment_window_horizon(exp_name)
        detected["experiments"].add(exp_name)

        if exp_window is not None:
            detected["window_sizes"].add(exp_window)
        if exp_horizon is not None:
            detected["horizons"].add(exp_horizon)

        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir():
                continue

            model_name_guess = safe_model_name_from_run_dir(run_dir.name)
            detected["models"].add(model_name_guess)

            # --- metrics ---
            metrics_file = run_dir / "backtest_metrics.json"
            if metrics_file.exists():
                try:
                    with open(metrics_file, "r", encoding="utf-8") as f:
                        metrics = json.load(f)
                    detected["metrics_data"].append({
                        "experiment": exp_name,
                        "model": model_name_guess,
                        "window_size": exp_window,
                        "horizon": exp_horizon,
                        **metrics,
                    })
                except Exception as e:
                    print(f"Error reading metrics from {metrics_file}: {e}")

            # --- gradients ---
            grad_dir = run_dir / "gradients"
            if grad_dir.exists():
                for f in grad_dir.glob("*.csv"):
                    match = grad_csv_pattern.match(f.name)
                    if not match:
                        continue
                    model_name = match.group(1)
                    file_window = int(match.group(3))

                    detected["models"].add(model_name)
                    detected["window_sizes"].add(file_window)
                    if exp_horizon is not None:
                        detected["horizons"].add(exp_horizon)

                    detected["gradient_csv_files"].append({
                        "path": str(f),
                        "experiment": exp_name,
                        "model": model_name,
                        "window_size": file_window,
                        "horizon": exp_horizon,
                    })

            # --- attention ---
            attn_dir = run_dir / "attention"
            if attn_dir.exists():
                for f in attn_dir.glob("*.npz"):
                    match = attn_pattern.match(f.name)
                    if not match:
                        continue
                    model_name = match.group(1)
                    fold_idx = int(match.group(2))
                    file_window = int(match.group(3))

                    detected["models"].add(model_name)
                    detected["window_sizes"].add(file_window)
                    if exp_horizon is not None:
                        detected["horizons"].add(exp_horizon)

                    detected["attention_files"].append({
                        "path": str(f),
                        "experiment": exp_name,
                        "model": model_name,
                        "window_size": file_window,
                        "horizon": exp_horizon,
                        "fold": fold_idx,
                    })

                    key = (model_name, file_window, exp_horizon)
                    if key not in detected["attn_lookup"] or fold_idx == 1:
                        detected["attn_lookup"][key] = str(f)

                for f in attn_dir.glob("*_metadata.json"):
                    match = attn_meta_pattern.match(f.name)
                    if not match:
                        continue
                    model_name = match.group(1)
                    fold_idx = int(match.group(2))
                    file_window = int(match.group(3))

                    detected["attention_meta_files"].append({
                        "path": str(f),
                        "experiment": exp_name,
                        "model": model_name,
                        "window_size": file_window,
                        "horizon": exp_horizon,
                        "fold": fold_idx,
                    })

                    key = (model_name, file_window, exp_horizon)
                    if key not in detected["attn_meta_lookup"] or fold_idx == 1:
                        detected["attn_meta_lookup"][key] = str(f)

    detected["models"] = sorted(detected["models"])
    detected["window_sizes"] = sorted(detected["window_sizes"])
    detected["horizons"] = sorted(detected["horizons"])
    detected["experiments"] = sorted(detected["experiments"])
    return detected


def filter_scanned_data(
    data: Dict[str, Any],
    model_filter: Optional[str] = None,
    window_filter: Optional[int] = None,
    horizon_filter: Optional[int] = None,
) -> Dict[str, Any]:
    out = dict(data)

    def ok(model_name: str, window: Optional[int], horizon: Optional[int]) -> bool:
        model_ok = model_filter is None or model_name == model_filter
        window_ok = window_filter is None or window == window_filter
        horizon_ok = horizon_filter is None or horizon == horizon_filter
        return model_ok and window_ok and horizon_ok

    out["models"] = [m for m in data["models"] if model_filter is None or m == model_filter]
    out["window_sizes"] = [w for w in data["window_sizes"] if window_filter is None or w == window_filter]
    out["horizons"] = [h for h in data["horizons"] if horizon_filter is None or h == horizon_filter]

    out["gradient_csv_files"] = [
        r for r in data["gradient_csv_files"]
        if ok(r["model"], r["window_size"], r["horizon"])
    ]
    out["attention_files"] = [
        r for r in data["attention_files"]
        if ok(r["model"], r["window_size"], r["horizon"])
    ]
    out["attention_meta_files"] = [
        r for r in data["attention_meta_files"]
        if ok(r["model"], r["window_size"], r["horizon"])
    ]
    out["metrics_data"] = [
        r for r in data["metrics_data"]
        if ok(r["model"], r["window_size"], r["horizon"])
    ]
    out["attn_lookup"] = {
        k: v for k, v in data["attn_lookup"].items()
        if ok(k[0], k[1], k[2])
    }
    out["attn_meta_lookup"] = {
        k: v for k, v in data["attn_meta_lookup"].items()
        if ok(k[0], k[1], k[2])
    }
    return out


# =========================================================
# Gradient analysis
# =========================================================

def load_gradient_monitor_csv(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rec in records:
        fpath = rec["path"]
        try:
            df = pd.read_csv(fpath)
            if df.empty:
                continue

            required = {
                "epoch", "step", "global_step", "batch_loss",
                "total_grad_norm", "encoder_grad_norm", "head_grad_norm"
            }
            if not required.issubset(df.columns):
                continue

            df = df.copy()
            df["Model"] = rec["model"]
            df["Window"] = rec["window_size"]
            df["Horizon"] = rec["horizon"]
            df["Experiment"] = rec["experiment"]
            df["Source File"] = Path(fpath).name

            m = re.search(r"_fold_(\d+)_w\d+_gradients\.csv$", Path(fpath).name)
            df["Fold"] = int(m.group(1)) if m else np.nan
            rows.append(df)
        except Exception as e:
            print(f"Failed to read {fpath}: {e}")

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


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


def save_csv(df: pd.DataFrame, output_csv: str, label: str):
    if df.empty:
        return
    df.to_csv(output_csv, index=False)
    print(f"  -> Saved {label} to {Path(output_csv).name}")


def save_gradient_epoch_summary(df: pd.DataFrame, output_csv: str):
    if df.empty:
        return
    agg_cols = [
        "total_grad_norm", "encoder_grad_norm", "head_grad_norm", "batch_loss",
        "encoder_head_ratio", "head_encoder_ratio",
        "encoder_share_of_total", "head_share_of_total",
        "is_vanishing_like", "is_exploding_like",
    ]
    summary = (
        df.groupby(["Model", "Window", "Horizon", "Fold", "epoch"], as_index=False)[agg_cols]
        .agg(["mean", "max", "min", "std"])
    )
    summary.columns = ["_".join([str(x) for x in col if str(x) != ""]) for col in summary.columns.to_flat_index()]
    summary.to_csv(output_csv, index=False)
    print(f"  -> Saved epoch summary to {Path(output_csv).name}")


def save_gradient_model_window_horizon_summary(df: pd.DataFrame, output_csv: str):
    if df.empty:
        return
    agg_cols = [
        "total_grad_norm", "encoder_grad_norm", "head_grad_norm", "batch_loss",
        "encoder_head_ratio", "head_encoder_ratio",
        "encoder_share_of_total", "head_share_of_total",
        "is_vanishing_like", "is_exploding_like",
    ]
    summary = (
        df.groupby(["Model", "Window", "Horizon"], as_index=False)[agg_cols]
        .agg(["mean", "max", "min", "std"])
    )
    summary.columns = ["_".join([str(x) for x in col if str(x) != ""]) for col in summary.columns.to_flat_index()]
    summary.to_csv(output_csv, index=False)
    print(f"  -> Saved model-window-horizon summary to {Path(output_csv).name}")


def save_gradient_stability_ranking(df: pd.DataFrame, output_csv: str):
    if df.empty:
        return
    grouped = df.groupby(["Model", "Window", "Horizon"], as_index=False).agg(
        mean_total_grad_norm=("total_grad_norm", "mean"),
        std_total_grad_norm=("total_grad_norm", "std"),
        max_total_grad_norm=("total_grad_norm", "max"),
        mean_encoder_head_ratio=("encoder_head_ratio", "mean"),
        exploding_like_rate=("is_exploding_like", "mean"),
        vanishing_like_rate=("is_vanishing_like", "mean"),
    )
    grouped["std_total_grad_norm"] = grouped["std_total_grad_norm"].fillna(0.0)
    grouped["stability_score"] = (
        2.0 * grouped["exploding_like_rate"] +
        1.5 * grouped["vanishing_like_rate"] +
        0.5 * grouped["std_total_grad_norm"] +
        0.25 * grouped["max_total_grad_norm"]
    )
    grouped = grouped.sort_values(["stability_score", "Model", "Window", "Horizon"])
    grouped.to_csv(output_csv, index=False)
    print(f"  -> Saved stability ranking to {Path(output_csv).name}")


def plot_gradient_total_norm(df: pd.DataFrame, output_png: str):
    if df.empty:
        return
    x_col = "global_step" if "global_step" in df.columns else "step"
    agg = df.groupby(["Model", "Window", "Horizon", x_col], as_index=False)["total_grad_norm"].mean()
    agg["Series"] = agg.apply(lambda r: f"{r['Model']} | w{int(r['Window'])} | h{int(r['Horizon'])}", axis=1)

    plt.figure(figsize=(14, 7))
    sns.lineplot(data=agg, x=x_col, y="total_grad_norm", hue="Series")
    plt.yscale("log")
    plt.title("Total Gradient Norm by Model / Window / Horizon")
    plt.xlabel(x_col.replace("_", " ").title())
    plt.ylabel("Total Gradient Norm (Log Scale)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()
    print(f"  -> Generated {Path(output_png).name}")


def plot_gradient_encoder_head_ratio(df: pd.DataFrame, output_png: str):
    if df.empty:
        return
    x_col = "global_step" if "global_step" in df.columns else "step"
    agg = df.groupby(["Model", "Window", "Horizon", x_col], as_index=False)["encoder_head_ratio"].mean()
    agg["Series"] = agg.apply(lambda r: f"{r['Model']} | w{int(r['Window'])} | h{int(r['Horizon'])}", axis=1)

    plt.figure(figsize=(14, 7))
    sns.lineplot(data=agg, x=x_col, y="encoder_head_ratio", hue="Series")
    plt.yscale("log")
    plt.title("Encoder / Head Gradient Ratio by Model / Window / Horizon")
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


def choose_attention_key(npz_path: str, meta_path: Optional[str]) -> Tuple[Optional[str], Dict[str, Any]]:
    meta = load_attention_metadata(meta_path)
    try:
        with np.load(npz_path) as data:
            keys = list(data.keys())
    except Exception:
        return None, meta

    primary = meta.get("primary_map")
    if primary and primary in keys:
        return primary, meta

    for k in meta.get("keys", []):
        if k in keys:
            return k, meta

    return (keys[0], meta) if keys else (None, meta)


def load_attention_map(npz_path: str, meta_path: Optional[str]) -> Tuple[Optional[np.ndarray], Optional[str], Dict[str, Any]]:
    key_used, meta = choose_attention_key(npz_path, meta_path)
    if key_used is None:
        return None, None, meta

    try:
        with np.load(npz_path) as data:
            attn = data[key_used]

        if attn.ndim == 4:
            mat = np.mean(attn, axis=(0, 1))
        elif attn.ndim == 3:
            mat = np.mean(attn, axis=0)
        elif attn.ndim == 2:
            mat = attn
        else:
            return None, key_used, meta

        return mat, key_used, meta
    except Exception:
        return None, key_used, meta


def save_attention_matrix_csv(npz_path: str, meta_path: Optional[str], output_csv: str) -> Optional[Dict[str, Any]]:
    mat, key_used, meta = load_attention_map(npz_path, meta_path)
    if mat is None:
        return None

    pd.DataFrame(mat).to_csv(output_csv, index=True, header=True)
    print(f"  -> Saved attention matrix to {Path(output_csv).name}")

    return {
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
        "q_len": int(mat.shape[0]),
        "k_len": int(mat.shape[1]),
    }


def plot_attention(npz_path: str, meta_path: Optional[str], output_png: str, title: str):
    mat, key_used, _ = load_attention_map(npz_path, meta_path)
    if mat is None:
        return

    plt.figure(figsize=(10, 8))
    plt.imshow(mat, cmap="Reds", aspect="auto", origin="upper", vmin=0)
    plt.colorbar(label="Avg Attention Weight")
    plt.title(f"{title}\nMap: {key_used}")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()
    print(f"  -> Generated {Path(output_png).name}")


def plot_erf(model_name: str, combos: List[Tuple[int, int]], attn_lookup: Dict, attn_meta_lookup: Dict, output_png: str):
    plt.figure(figsize=(12, 6))
    plotted = False

    for window, horizon in combos:
        key = (model_name, window, horizon)
        if key not in attn_lookup:
            continue
        npz_path = attn_lookup[key]
        meta_path = attn_meta_lookup.get(key)

        mat, _, _ = load_attention_map(npz_path, meta_path)
        if mat is None:
            continue

        last_row = mat[-1, :]
        lags = np.arange(len(last_row))[::-1]
        plt.plot(lags, last_row, label=f"w{window}, h{horizon}", linewidth=2, alpha=0.8)
        plotted = True

    if plotted:
        plt.title(f"Effective Receptive Field Proxy - {model_name}")
        plt.xlabel("Lag")
        plt.ylabel("Average Attention Weight")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_png)
        plt.close()
        print(f"  -> Generated {Path(output_png).name}")


# =========================================================
# Performance
# =========================================================

def save_performance_outputs(metrics_data: List[Dict[str, Any]], output_dir: Path):
    if not metrics_data:
        print("  Performance analysis skipped (no metrics data)")
        return

    df = pd.DataFrame(metrics_data)
    df.to_csv(output_dir / "aggregated_performance.csv", index=False)
    print("  -> Saved aggregated_performance.csv")

    for metric in ["mse", "mae", "rmse"]:
        if metric in df.columns:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=df, x="window_size", y=metric, hue="model", style="horizon", markers=True)
            plt.title(f"{metric.upper()} vs Window Size")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(output_dir / f"performance_{metric}.png")
            plt.close()
            print(f"  -> Generated performance_{metric}.png")


# =========================================================
# Main
# =========================================================

def run_analysis(
    base_dir: str = "results",
    output_dir: str = "results/analysis",
    model_filter: Optional[str] = None,
    window_filter: Optional[int] = None,
    horizon_filter: Optional[int] = None,
    skip_gradients: bool = False,
    skip_attention: bool = False,
    skip_performance: bool = False,
):
    print("=" * 80)
    print("CURRENT-FEATURE ANALYSIS PIPELINE V5 HORIZON-AWARE")
    print("=" * 80)

    data = scan_results_structure(base_dir=base_dir)
    data = filter_scanned_data(
        data,
        model_filter=model_filter,
        window_filter=window_filter,
        horizon_filter=horizon_filter,
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not skip_gradients:
        print("\n[1/3] Gradient monitor analysis...")
        grad_df = load_gradient_monitor_csv(data["gradient_csv_files"])
        if not grad_df.empty:
            grad_df = add_gradient_monitor_features(grad_df)
            save_csv(grad_df, str(out_path / "gradient_monitor_raw_data.csv"), "gradient monitor raw data")
            save_csv(grad_df, str(out_path / "gradient_monitor_step_summary.csv"), "gradient monitor step summary")
            save_gradient_epoch_summary(grad_df, str(out_path / "gradient_monitor_epoch_summary.csv"))
            save_gradient_model_window_horizon_summary(
                grad_df,
                str(out_path / "gradient_monitor_model_window_horizon_summary.csv"),
            )
            save_gradient_stability_ranking(
                grad_df,
                str(out_path / "gradient_monitor_stability_ranking.csv"),
            )
            plot_gradient_total_norm(grad_df, str(out_path / "gradient_monitor_total_norm.png"))
            plot_gradient_encoder_head_ratio(grad_df, str(out_path / "gradient_monitor_encoder_head_ratio.png"))
        else:
            print("  No readable gradient CSV files found.")
    else:
        print("\n[1/3] Gradient monitor analysis skipped.")

    if not skip_attention:
        print("\n[2/3] Attention analysis...")
        transformer_models = [m for m in data["models"] if "transformer" in m.lower()]
        attn_rows = []

        if transformer_models and data["attn_lookup"]:
            transformer = transformer_models[0]
            combos = sorted(
                [(w, h) for (m, w, h) in data["attn_lookup"].keys() if m == transformer],
                key=lambda x: (x[1], x[0]),
            )

            for window, horizon in combos:
                key = (transformer, window, horizon)
                npz_path = data["attn_lookup"][key]
                meta_path = data["attn_meta_lookup"].get(key)

                plot_attention(
                    npz_path,
                    meta_path,
                    str(out_path / f"attention_heatmap_w{window}_h{horizon}.png"),
                    f"Attention Map {transformer} | w{window} | h{horizon}",
                )
                row = save_attention_matrix_csv(
                    npz_path,
                    meta_path,
                    str(out_path / f"attention_matrix_w{window}_h{horizon}.csv"),
                )
                if row:
                    row["model"] = transformer
                    row["window"] = window
                    row["horizon"] = horizon
                    attn_rows.append(row)

            plot_erf(
                transformer,
                combos,
                data["attn_lookup"],
                data["attn_meta_lookup"],
                str(out_path / "effective_receptive_field.png"),
            )
        else:
            print("  No transformer attention artifacts found.")

        if attn_rows:
            pd.DataFrame(attn_rows).to_csv(out_path / "attention_artifact_summary.csv", index=False)
            print("  -> Saved attention_artifact_summary.csv")
    else:
        print("\n[2/3] Attention analysis skipped.")

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
        horizon_filter=args.horizon,
        skip_gradients=args.skip_gradients,
        skip_attention=args.skip_attention,
        skip_performance=args.skip_performance,
    )
