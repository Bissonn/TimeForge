# dyplom_new/scripts/generate_synthetic_data.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta

def setup_logging():
    """
    Konfiguruje logging do pliku i konsoli.
    """
    os.makedirs("results/logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("results/logs/synthetic_data.log"),
            logging.StreamHandler()
        ]
    )

def generate_trend(t: np.ndarray, trend_type: str, slope: float = 0.01, base: float = 0.0,
                   exp_rate: float = 0.001) -> np.ndarray:
    """
    Generuje komponent trendu.
    """
    if trend_type == 'linear':
        return base + slope * t
    elif trend_type == 'exponential':
        return base * np.exp(exp_rate * t)
    elif trend_type == 'none':
        return np.zeros_like(t)
    else:
        raise ValueError(f"Unsupported trend type: {trend_type}")

def generate_seasonality(t: np.ndarray, period: float, amplitude: float = 1.0,
                         phase_shift: float = 0.0) -> np.ndarray:
    """
    Generuje komponent sezonowy (sinusoidalny).
    """
    return amplitude * np.sin(2 * np.pi * t / period + phase_shift)

def generate_noise(length: int, noise_level: float = 0.1) -> np.ndarray:
    """
    Generuje szum Gaussowski.
    """
    return np.random.normal(0, noise_level, length)

def generate_synthetic_series(length: int, freq: str, start_date: str, base_level: float = 10.0,
                             trend_params: Dict = None, seasonality_params: Dict = None,
                             noise_level: float = 0.1) -> pd.Series:
    """
    Generuje pojedynczy syntetyczny szereg czasowy.
    """
    t = np.arange(length)
    series = base_level * np.ones(length)

    # Trend
    if trend_params:
        trend = generate_trend(
            t,
            trend_type=trend_params.get('type', 'none'),
            slope=trend_params.get('slope', 0.01),
            base=trend_params.get('base', base_level),
            exp_rate=trend_params.get('exp_rate', 0.001)
        )
        series += trend

    # Sezonowość
    if seasonality_params:
        seasonality = generate_seasonality(
            t,
            period=seasonality_params.get('period', 24.0),
            amplitude=seasonality_params.get('amplitude', 1.0),
            phase_shift=seasonality_params.get('phase_shift', 0.0)
        )
        series += seasonality

    # Szum
    noise = generate_noise(length, noise_level)
    series += noise

    # Indeks czasowy
    date_index = pd.date_range(start=start_date, periods=length, freq=freq)
    return pd.Series(series, index=date_index)

def generate_synthetic_dataset(config: Dict, output_dir: str = "data") -> pd.DataFrame:
    """
    Generuje syntetyczny dataset z wieloma szeregami.
    """
    dataset_config = config['dataset']
    length = dataset_config['length']
    freq = dataset_config['freq']
    start_date = dataset_config['start_date']
    series_configs = dataset_config['series']
    dataset_name = dataset_config['name']

    # Tworzymy DataFrame z kolumną date
    df = pd.DataFrame()
    df['date'] = pd.date_range(start=start_date, periods=length, freq=freq)

    # Generujemy szeregi i przypisujemy jako kolumny
    for series_conf in series_configs:
        name = series_conf['name']
        series = generate_synthetic_series(
            length=length,
            freq=freq,
            start_date=start_date,
            base_level=series_conf.get('base_level', 10.0),
            trend_params=series_conf.get('trend', {}),
            seasonality_params=series_conf.get('seasonality', {}),
            noise_level=series_conf.get('noise_level', 0.1)
        )
        logging.info(f"Generated series '{name}' with shape: {series.shape}, first few values: {series[:5]}")
        # Resetujemy indeks series, aby dopasować do df
        df[name] = series.reset_index(drop=True)

    # Zapisz dataset
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset_name}.csv")
    df.to_csv(output_path, index=False)
    logging.info(f"Saved synthetic dataset to {output_path}")

    return df

def plot_synthetic_dataset(df: pd.DataFrame, dataset_name: str, output_dir: str = "results/plots"):
    """
    Wizualizuje wygenerowany dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    for col in df.columns:
        if col != 'date':
            plt.plot(df['date'], df[col], label=col)
    plt.title(f"Synthetic Dataset: {dataset_name}")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    output_path = os.path.join(output_dir, f"{dataset_name}_plot.png")
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Saved plot to {output_path}")

def load_config(config_path: str = "synthetic_config.yaml") -> Dict:
    """
    Ładuje konfigurację z pliku YAML.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    """
    Główna funkcja skryptu generującego syntetyczne dane.
    """
    setup_logging()
    config = load_config()
    df = generate_synthetic_dataset(config)
    plot_synthetic_dataset(df, config['dataset']['name'])

if __name__ == "__main__":
    main()