# TimeForge: Advanced Time Series Forecasting Framework

A flexible, extensible Python framework for **time series forecasting**. Train, evaluate, and compare classical models (e.g., **ARIMA**, **VAR**) and deep learning architectures (e.g., **LSTM**, **Transformer**) through a unified interface and a configurable preprocessing pipeline.

<p align="center">
  <em>Unified interface • Powerful preprocessing • Config‑driven experiments • Reproducible results</em>
</p>

---

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Example](#quick-example)
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
- **Unified model interface** – swap models with minimal changes to training script.
- **Highly modular Transformer engine** - script-configurable selection of model backbones (encoder-only/decoder), forecasting strategies (direct/iterative), and SOTA components including RevIN/R2-IN (Robust) normalization, flexible readout, target initialization, and positional encoding.
- **Automated preprocessing** – scaling, log transforms, winsorization, differencing (incl. seasonal), and time features.
- **Config‑driven pipeline** – everything in one YAML configuration file (datasets, models, experiments, preprocessing, optimization).
- **Hyperparameter optimization** – `grid`, `random`, and `optuna`.
- **Walk‑forward validation** – consistent, reproducible evaluation.
- **Rich metrics & plots** – MAE, RMSE, SMAPE, MASE and forecast vs. actual visualizations.
- **Extensive diagnostics** - training diagnostics (gradient norms, self-attention patterns).

---

## Project Structure

```
├─ data/                # Raw and processed datasets (CSV, Parquet, etc.)
├─ models/              # Unified forecasting architectures (Statistical & Neural)
│  ├─ base.py           # Abstract base classes defining model contracts
│  └─ ...
├─ utils/               # Core logic for preprocessing, data loading, and metrics
│  ├─ dataset.py        # Data loading, handling, and sequence splitting
│  ├─ preprocessor.py   # Configurable engine for transformations (scaling, diff)
│  └─ ...
├─ analysis/            # Diagnostic tools for gradient stability and self-attention
├─ monitoring/          # Modules for real-time training and gradient tracking
├─ core/                # Internal experiment management, trainer, and runner logic
├─ scripts/             # Command-line interface and entry point scripts
│  └─ train.py          # Main script to orchestrate fit, predict, and evaluate
├─ tests/               # Extensive unit and integration test suite
├─ examples/            # YAML templates demonstrating framework-specific features
├─ experiments/         # Scientific research scenarios for hypothesis testing
└─ results/             # Persistent storage for metrics, plots, and model artifacts
```

---

## Prerequisites

### Hardware Acceleration (CUDA Support)

**TimeForge runs best when using CUDA.** While it is possible to run the application using a CPU, a single training session or experiment run can take significantly longer—up to several days. For this reason, using an NVIDIA GPU with CUDA support is highly recommended for any practical usage.

#### 1. Verify CUDA Installation
To check whether CUDA drivers are installed and recognized by your system, run the following command:

```bash
nvidia-smi
```
If CUDA is properly configured, you should see an output similar to this:
```bash
Tue May  5 18:18:16 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.163.01             Driver Version: 550.163.01     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3060 Ti     Off |   00000000:01:00.0 Off |                  N/A |
|  0%   44C    P8             11W /  240W |       9MiB /   8192MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```
To leverage GPU acceleration, you must install the version of PyTorch compiled with CUDA support.

## Installation
```bash

# Clone the repository
git clone https://github.com/Bissonn/TimeForge

# Enter the project directory
cd TimeForge

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -U pip
pip install -r requirements.txt
```

> **Python**: 3.10–3.12 recommended.

---

## Preparing Data

Before training a model, you need a dataset. You can generate a demonstration dataset using the built-in script:

```bash
python scripts/generate_demo_dataset.py
```

---

## Running Experiments

The project uses a unified script (`scripts.train`) for both optimization and evaluation. You need to provide an experiment name and a path to a configuration file.

### Optimization (Training)

To train a model with optimization enabled, use the `--optimize` flag:

```bash
python -m scripts.train --optimize \
  --experiment [EXPERIMENT_NAME] \
  --config-path examples/[CONFIG_FILE].yaml
```
This runs an optimalization, where the application chooses the set of parameters from the ranges specified in the [MODEL_NAME] in the [CONFIG_FILE], which lead to the predictions having the lowest score in the metric chosen to measure errors, in most cases this metric is the mean squared error (MSE).
This will return a **best_params.json** file located in results/[EXPERIMENT_NAME]/[MODEL_NAME]_{timestamp} directory. This will later be used if the same configuration is run with the `--evaluate` flag.
### Evaluation

To test the performance of a trained model, use the `--evaluate` flag:

```bash
python -m scripts.train --evaluate \
  --experiment [EXPERIMENT_NAME] \
  --config-path examples/[CONFIG_FILE].yaml
```
If this configuration has before been run with the `--optimize` flag, then it will be run according to the **best_params.json** file generated by this run. If it has been run several times with this flag, then it will either be run with the latest run or the one specified by the `--run_id` parameter.
If it has not been run with the `--optimize` flag before, then it will be run with the default parameters of the experiment and model specified in the [CONFIG_FILE]. However, it is recommended to run a given configuration first with `--optimize` and later with `--evaluate`.

---

## Command Arguments Explained

| Argument | Description |
|---|---|
| `--optimize` | Enables the training and hyperparameter optimization process. |
| `--evaluate` | Runs the evaluation metrics on a pre-trained model. |
| `--experiment` | A unique name for your project run (e.g., `my_first_test`). |
| `--config-path` | Path to the YAML configuration file (found in the `examples/` folder). |
| `--epochs EPOCHS` | Number of training epochs to run. |
| `--run-id RUN_ID` | Identifier for a specific run (useful for resuming or comparing runs). |
| `--log-level LOG_LEVEL` | Sets the logging verbosity (e.g., `DEBUG`, `INFO`, `WARNING`). |
| `--no-visualization` | Disables plot/chart generation during training or evaluation. |
| `--force-defaults` | Ignores config file overrides and uses default parameter values. |

---

## Quick Example

Run a basic example provided in the repository:

**1. Generate the demo data:**
```bash
python scripts/generate_demo_dataset.py
```

**2. Run optimization:**
```bash
python -m scripts.train --optimize \
  --experiment demo_quick_smoke_test \
  --config-path examples/demo_quick_test.yaml
```

All outputs (metrics, plots, artifacts) are saved under `results/`.

---

## Configuration
The entire pipeline is configured via YAML configuration file. File structure and configuration paramteres reference is available in [CONFIGURATION REFERENCE](docs/CONFIGURATION_REFERENCE.md)


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
Run the complete suite in `tests/` directory  :
```bash
pytest
```

Guidelines:
- Prefer **unit tests** with mocking for speed and determinism; add **integration tests** for end‑to‑end flows.

---

## Contributing
Contributions are welcome! Please:
1. Open an issue describing the change.
2. Create a feature branch and include tests.
3. Run `pytest` and ensure all checks pass.

---

## License
MIT
