# TS‑Forecaster: Advanced Time Series Forecasting Framework

A flexible, extensible Python framework for **time series forecasting**. Train, evaluate, and compare classical models (e.g., **ARIMA**, **VAR**) and deep learning architectures (e.g., **LSTM**, **Transformer**) through a unified interface and a configurable preprocessing pipeline.

<p align="center">
  <em>Unified interface • Powerful preprocessing • Config‑driven experiments • Reproducible results</em>
</p>

---

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
  - [Experiments](#experiments)
  - [Datasets](#datasets)
  - [Models](#models)
  - [Preprocessing](#preprocessing)
- [Built‑in Models](#built-in-models)
- [Add a New Model](#add-a-new-model)
- [Evaluation & Results](#evaluation--results)
- [Testing](#testing)
- [Tips on Differencing Without Losing Samples](#tips-on-differencing-without-losing-samples)
- [Contributing](#contributing)
- [License](#license)

---

## Features
- **Unified model interface** – swap models without changing training script.
- **Automated preprocessing** – scaling, log transforms, winsorization, differencing (incl. seasonal), and time features.
- **Config‑driven pipeline** – everything in one `config.yaml` (datasets, models, experiments, preprocessing, optimization).
- **Hyperparameter optimization** – `grid`, `random`, and `optuna`.
- **Walk‑forward validation** – consistent, reproducible evaluation.
- **Rich metrics & plots** – MAE, RMSE, SMAPE, MASE and forecast vs. actual visualizations.

---

## Project Structure
> This is the *recommended* layout. Your repository may use slightly different paths; update as needed.

```
.
├─ config.yaml                 # Global configuration for datasets, models, and experiments
├─ data/                       # Raw/processed datasets (CSV, Parquet, ...)
├─ models/                     # Forecasting models (ARIMA, VAR, LSTM, Transformer, ...)
│  ├─ base.py                  # Base forecaster classes (StatTSForecaster, NeuralTSForecaster)
│  └─ ...
├─ utils/                      # Core utilities
│  ├─ dataset.py               # Data loading & splitting
│  ├─ preprocessor.py          # Configurable preprocessing engine
│  └─ ...
├─ scripts/                    # Entry points / CLI
│  └─ train.py                 # Orchestrates fit → predict → evaluate
├─ tests/                      # Unit & integration tests mirroring src layout
└─ results/                    # Metrics, plots, and serialized models
```

---

## Installation
```bash
# 1) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -U pip
pip install -r requirements.txt
```

> **Python**: 3.10–3.12 recommended.

---

## Quick Start
Train a single model on a dataset:

```bash
python -m scripts.train --model arima --dataset ETTh1
```

Train multiple models:
```bash
python -m scripts.train --models arima lstm_direct --dataset ETTh1
```

Enable hyperparameter optimization (ensure the `optimize: true` flag and the `optimization` block are set in `config.yaml`):
```bash
python -m scripts.train --model lstm_direct --dataset my_dataset --optimized
```

Force retraining (ignore saved models):
```bash
python -m scripts.train --model var --dataset ETTh1 --force-train
```

All outputs (metrics, plots, artifacts) are saved under `results/`.

---

## Configuration
The entire pipeline is configured via `config.yaml`. It is validated against an internal schema in `utils/config_utils.py` (or equivalent).

### Experiments
```yaml
experiments:
  name: "Baseline"
  description: "Compare classical vs neural models"
  validation_setup:
    forecast_steps: 24         # horizon
    n_folds: 3                 # walk‑forward folds
    max_window_size: 720       # initial window in first fold
    early_stopping_validation_percentage: 0.2
```

### Datasets
```yaml
datasets:
  ETTh1:
    path: data/ETTh1.csv
    columns: ["target", "exog1", "exog2"]
    freq: "H"                   # optional; inferred if omitted
    preprocessing:              # optional; dataset‑level default preprocessing
      scaling:
        enabled: true
        method: standard
```

### Models
```yaml
models:
  arima:
    p: 5
    d: 1
    q: 0
    window_size: 30
    optimize: true
    optimization:
      method: grid
      params:
        p: [1, 2, 3]
        d: [1]
        q: [0, 1]
    preprocessing:
      scaling:
        enabled: true
        method: standard

  lstm_direct:
    window_size: 90
    hidden_size: 128
    num_layers: 2
    optimize: true
    optimization:
      method: optuna
      params:
        window_size: [30, 60, 90]
        hidden_size: {min: 64, max: 256, step: 32}
        learning_rate: {min: 1e-4, max: 1e-2, log: true}
        n_trials: 20
    preprocessing:
      scaling: {enabled: true, method: minmax}
      differencing: {enabled: true, auto: adf, max_d: 2}
```

### Preprocessing
Supported steps (toggle per dataset/model):
- `log_transform`: `enabled`, `method: log|log1p`
- `winsorize`: `enabled`, `limits: [lower_q, upper_q]`
- `scaling`: `enabled`, `method: minmax|standard`, `range: [0, 1]`
- `differencing`:
  - `enabled`
  - `auto: adf|kpss|none`
  - `order: int` (d)
  - `seasonal_order: int` (D)
  - `seasonal_period: int` (m)

> The framework keeps a per‑column **`pipeline_states`** dictionary with learned parameters (e.g., scalers, differencing orders) to ensure a correct `inverse_transform` after prediction.

---

## Built‑in Models
- **ARIMA / SARIMA** (statistical)
- **VAR** (multivariate)
- **LSTM** (direct, iterative)
- **Transformer** (configurable encoder‑decoder)

> Models adhere to a common base (see `models/base.py`) exposing `fit()`, `predict()`, and persistence utilities.

---

## Add a New Model
1. **Create a file** under `models/` (e.g., `my_model.py`).
2. **Inherit** from `StatTSForecaster` (classical) or `NeuralTSForecaster` (DL).
3. **Implement** required methods (`fit`, `predict`, constructor params).
4. **Register** the model with the registry (e.g., `@register_model("my_model", is_univariate=False)`).
5. **Expose** the import in `scripts/train.py` (so registration runs).
6. **Configure** the model in `config.yaml` under `models:`.

---

## Evaluation & Results
After each run the framework stores:
- **Metrics**: MAE, RMSE, SMAPE, MASE per fold and averaged.
- **Artifacts**: trained model files, preprocessing state.
- **Visualizations**: forecast vs. actual plots (PNG/HTML).

Results are grouped by experiment/model/dataset inside `results/`.

---

## Testing
Run the complete suite:
```bash
pytest -q
```

Guidelines:
- Tests mirror the source structure (`tests/utils/test_preprocessor*.py`, etc.).
- Prefer **unit tests** with mocking for speed and determinism; add **integration tests** for end‑to‑end flows.

---

## Tips on Differencing Without Losing Samples
Differencing normally drops the first `d + D·m` observations. This framework uses a **stateful lag‑buffer**:
- During `fit`, the raw history is stored per column.
- During `transform(test)`, a minimal tail of history (size `d + D·m`) is **prepended** to the test fragment, differencing is computed on the concatenation, and then the prepended tail is **sliced off**.
- This preserves the **length and index** of the fragment and avoids off‑by‑one issues during `inverse_transform`.

> For inverse differencing, continuity checks allow a start offset up to `d + D·m` steps and use historical anchors to reconstruct values in the correct order (seasonal → regular).

---

## Contributing
Contributions are welcome! Please:
1. Open an issue describing the change.
2. Create a feature branch and include tests.
3. Run `pytest` and ensure all checks pass.

---

## License
MIT
