# Configuration Parameter Reference

## Overview

This document describes the YAML configuration format accepted by the forecasting framework configuration parser.
The configuration file defines datasets, model configurations, optional logging and output paths, and optional experiments.

The reference is based on the parser implementation in `config_utils.py` and on the provided example YAML files. Fields listed here are parser-recognized fields. Runtime behavior outside the parser is documented only when it is directly implied by names, comments, examples, or validation logic.

> **Interpretation rule:** when a default, semantic dependency, or runtime effect is not explicitly encoded in the parser, this document says: `Not explicitly defined in the parser code.`

## Source Analysis Summary

| Item | Finding |
|---|---|
| Main parser file | `config_utils.py` |
| Main entry point | `load_config(config_path="config.yaml")` |
| Main validation function | `validate_config(config, config_with_line_info, config_path)` |
| Schema library | `schema.Schema`, `And`, `Or`, `Use`, `Optional` |
| YAML loaders | `ruamel.yaml.YAML(typ="rt")` for line information and `yaml.safe_load()` for actual config data |
| Top-level required sections | `datasets`, `models` |
| Top-level optional sections | `paths`, `logging`, `experiments` |
| Model types | `arima`, `sarima`, `var`, `lstm`, `transformer`, `simple_seasonal` |
| Dataset-level preprocessing format | Simple `preprocessing` pipeline: `log_transform`, `winsorize`, `scaling` |
| Model and experiment-model preprocessing format | Grouped `preprocessing.preprocessing_groups[]` format |
| Defaults injected by parser | Almost none during `load_config()`. `get_model_config()` injects `optimization: {method: grid, params: {}}` if missing for a selected model. |
| Error formatting | Schema errors are reformatted into compact messages with YAML path, approximate line number, and context snippet when available. |

## Configuration File Structure

At the top level, the YAML file has this shape:

```yaml
paths:       # optional
logging:     # optional
datasets:    # required
models:      # required
experiments: # optional
```

Two sections are mandatory: `datasets` and `models`. `experiments` is optional at schema level, although most training or evaluation workflows will normally need it.

Model configurations are stored under arbitrary user-defined keys:

```yaml
models:
  transformer_large:     # configuration name
    type: "transformer"  # model type used for schema selection
```

The model configuration key is not the model type. The `type` field selects the concrete model schema.

## Minimal Configuration Example

```yaml
logging:
  level: "INFO"

datasets:
  demo_energy:
    path: "data/demo_energy_demand.csv"
    columns: ["energy_demand"]
    freq: "h"

models:
  arima_baseline:
    type: "arima"
    p: 1
    d: 1
    q: 1

experiments:
  - name: "minimal_arima"
    dataset: "demo_energy"
    models: ["arima_baseline"]
    validation_setup:
      forecast_steps: 24
      n_folds: 3
      window_size: 168
      evaluation_metric: "mse"
```

## Full Configuration Example

```yaml
paths:
  model_save_path_template: "results/models/{model_name}_{dataset_name}.pkl"

logging:
  environment: "dev"
  console_level: "INFO"
  file_level: "DEBUG"
  file: "logs/{experiment_name}.log"
  use_context: true
  custom_levels:
    forecasting.training: "DEBUG"

datasets:
  demo_energy:
    path: "data/demo_energy_demand.csv"
    columns: ["energy_demand"]
    freq: "h"
    past_covariates: ["temperature", "humidity", "wind_speed"]
    future_covariates: ["is_weekend"]
    time_features: ["hour", "day_of_week"]
    preprocessing:
      scaling:
        enabled: true
        method: "standard"

models:
  transformer_hpo:
    type: "transformer"
    architecture: "encoder-only"
    strategy: "direct"
    hidden_size: 128
    num_heads: 4
    num_encoder_layers: 2
    dim_ff_multiplier: 4.0
    dropout: 0.1
    batch_size: 32
    learning_rate: 0.001
    epochs: 50
    optimizer: "adam"
    loss: "mse"
    use_exogenous: true
    scheduler_config:
      type: "onecycle"
      pct_start: 0.3
      div_factor: 25.0
      final_div_factor: 10000.0
      anneal_strategy: "cos"
    preprocessing:
      preprocessing_groups:
        - name: "target_scaling"
          apply_to: "__targets__"
          pipeline:
            scaling:
              enabled: true
              method: "standard"
    optimize: true
    optimization:
      method: "optuna"
      n_trials: 20
      pruner_config:
        type: "percentile"
        percentile: 25.0
        n_startup_trials: 5
        n_warmup_steps: 5
      params:
        hidden_size: [64, 128, 256]
        dropout: [0.05, 0.1, 0.2]
        learning_rate:
          min: 0.0001
          max: 0.01
          log: true
        scheduler_config:
          pct_start:
            min: 0.1
            max: 0.4
            step: 0.1

experiments:
  - name: "demo_transformer_hpo"
    dataset: "demo_energy"
    models:
      - name: "transformer_hpo"
        use_exogenous: true
    validation_setup:
      forecast_steps: 24
      n_folds: 3
      window_size: 168
      evaluation_metric: "mse"
      early_stopping_validation_percentage: 15
```

## Parameter Reference

### Top-Level, Paths, Logging, and Datasets

| YAML path | Type | Required | Default | Allowed values | Validation | Defined in |
|---|---|---|---|---|---|---|
| `paths` | `mapping` | No | Not explicitly defined in the parser code. | Not explicitly constrained beyond type validation. | If present, must contain only supported path options. | `validate_config.main_schema` |
| `paths.model_save_path_template` | `string` | No | Not explicitly defined in the parser code. | Not explicitly constrained beyond type validation. | No parser-level placeholder validation. | `validate_config.main_schema.paths` |
| `logging` | `mapping` | No | Not explicitly defined in the parser code. | Not explicitly constrained beyond type validation. | If present, must contain only supported logging fields. | `validate_config.main_schema` |
| `logging.level` | `string` | No | Not explicitly defined in the parser code. | DEBUG, INFO, WARNING, ERROR, CRITICAL | String is uppercased before membership check. | `validate_config.main_schema.logging` |
| `logging.environment` | `string` | No | Not explicitly defined in the parser code. | prod, dev, debug | String is lowercased before membership check. | `validate_config.main_schema.logging` |
| `logging.file` | `string` | No | Not explicitly defined in the parser code. | Any string | No path existence check. | `validate_config.main_schema.logging` |
| `logging.console_level` | `string` | No | Not explicitly defined in the parser code. | DEBUG, INFO, WARNING, ERROR, CRITICAL | String is uppercased before membership check. | `validate_config.main_schema.logging` |
| `logging.file_level` | `string` | No | Not explicitly defined in the parser code. | DEBUG, INFO, WARNING, ERROR, CRITICAL | String is uppercased before membership check. | `validate_config.main_schema.logging` |
| `logging.use_context` | `boolean` | No | Not explicitly defined in the parser code. | true, false | Must be YAML boolean. | `validate_config.main_schema.logging` |
| `logging.custom_levels.<module_name>` | `string` | No | Not explicitly defined in the parser code. | DEBUG, INFO, WARNING, ERROR, CRITICAL | Keys are arbitrary strings, values are validated log-level strings. | `validate_config.main_schema.logging` |
| `datasets` | `mapping` | Yes | Not explicitly defined in the parser code. | Not explicitly constrained beyond type validation. | Must be a mapping from dataset name to dataset config. | `validate_config.main_schema` |
| `datasets.<dataset_name>` | `mapping` | Yes | Not explicitly defined in the parser code. | Not explicitly constrained beyond type validation. | Each dataset config must match the dataset schema. | `validate_config.main_schema.datasets` |
| `datasets.<dataset_name>.path` | `string` | Yes | Not explicitly defined in the parser code. | Existing filesystem path | `os.path.exists(path)` must return true. | `validate_config.main_schema.datasets` |
| `datasets.<dataset_name>.columns` | `list[string]` | Yes | Not explicitly defined in the parser code. | Any list of strings | Must be a YAML sequence of strings. | `validate_config.main_schema.datasets` |
| `datasets.<dataset_name>.past_covariates` | `list[string]` | No | Not explicitly defined in the parser code. | Any list of strings | Must be a YAML sequence of strings. | `validate_config.main_schema.datasets` |
| `datasets.<dataset_name>.future_covariates` | `list[string]` | No | Not explicitly defined in the parser code. | Any list of strings | Must be a YAML sequence of strings. | `validate_config.main_schema.datasets` |
| `datasets.<dataset_name>.freq` | `string` | No | Not explicitly defined in the parser code. | Any pandas-compatible frequency accepted by `pandas.tseries.frequencies.to_offset` | `to_offset(freq)` must succeed. | `validate_config.main_schema.datasets` |
| `datasets.<dataset_name>.preprocessing` | `mapping` | No | Not explicitly defined in the parser code. | log_transform, winsorize, scaling | Uses the simple preprocessing pipeline schema, not preprocessing groups. | `validate_config.main_schema.datasets` |
| `datasets.<dataset_name>.time_features` | `list[string]` | No | Not explicitly defined in the parser code. | year, month, day, hour, minute, second, day_of_week, day_of_year, week, quarter, is_month_start, is_month_end, is_quarter_start, is_quarter_end, is_year_start, is_year_end, is_leap_year | Every item must belong to `ALLOWED_TIME_FEATURES`. | `validate_config.main_schema.datasets` |
| `datasets.<dataset_name>.differencing` | `mapping` | No | Not explicitly defined in the parser code. | enabled, order, seasonal_order, seasonal_period | Uses a dedicated differencing schema at dataset level. | `validate_config.main_schema.datasets` |

### Preprocessing and Differencing

The parser supports two preprocessing layouts:

1. Dataset-level preprocessing uses a simple pipeline directly under `datasets.<dataset_name>.preprocessing`.
2. Model-level and experiment-model-level preprocessing use grouped preprocessing under `preprocessing.preprocessing_groups[]`.

| YAML path | Type | Required | Default | Allowed values | Validation | Defined in |
|---|---|---|---|---|---|---|
| `datasets.<dataset_name>.preprocessing` | `mapping` | No | Not explicitly defined in the parser code. | log_transform, winsorize, scaling | Uses the simple preprocessing pipeline schema, not preprocessing groups. | `validate_config.main_schema.datasets` |
| `datasets.<dataset_name>.differencing` | `mapping` | No | Not explicitly defined in the parser code. | enabled, order, seasonal_order, seasonal_period | Uses a dedicated differencing schema at dataset level. | `validate_config.main_schema.datasets` |
| `datasets.<dataset_name>.preprocessing.log_transform` | `mapping` | No | Not explicitly defined in the parser code. | enabled, method, epsilon | Validated by `_define_log_transform_schema`. | `_define_pipeline_schema` |
| `datasets.<dataset_name>.preprocessing.log_transform.enabled` | `boolean` | Yes if log_transform is present | Not explicitly defined in the parser code. | true, false | Must be YAML boolean. | `_define_log_transform_schema` |
| `datasets.<dataset_name>.preprocessing.log_transform.method` | `string` | No | Not explicitly defined in the parser code. | log, log1p | Must be one of the allowed method strings. | `_define_log_transform_schema` |
| `datasets.<dataset_name>.preprocessing.log_transform.epsilon` | `float` | No | Not explicitly defined in the parser code. | float > 0 | Must be a float and strictly positive. | `_define_log_transform_schema` |
| `datasets.<dataset_name>.preprocessing.winsorize` | `mapping` | No | Not explicitly defined in the parser code. | enabled, limits | Validated by `_define_winsorize_schema`. | `_define_pipeline_schema` |
| `datasets.<dataset_name>.preprocessing.winsorize.enabled` | `boolean` | Yes if winsorize is present | Not explicitly defined in the parser code. | true, false | Must be YAML boolean. | `_define_winsorize_schema` |
| `datasets.<dataset_name>.preprocessing.winsorize.limits` | `list[float]` | No | Not explicitly defined in the parser code. | [lower, upper] | Exactly two floats with `0 <= lower < upper <= 1`. | `_define_winsorize_schema` |
| `datasets.<dataset_name>.preprocessing.scaling` | `mapping` | No | Not explicitly defined in the parser code. | enabled, method, range | Validated by `_define_scaling_schema`. | `_define_pipeline_schema` |
| `datasets.<dataset_name>.preprocessing.scaling.enabled` | `boolean` | Yes if scaling is present | Not explicitly defined in the parser code. | true, false | Must be YAML boolean. | `_define_scaling_schema` |
| `datasets.<dataset_name>.preprocessing.scaling.method` | `string` | No | Not explicitly defined in the parser code. | minmax, standard, robust | Must be one of the allowed method strings. | `_define_scaling_schema` |
| `datasets.<dataset_name>.preprocessing.scaling.range` | `list[number]` | No | Not explicitly defined in the parser code. | [min, max] | Exactly two int or float values with first value lower than second value. | `_define_scaling_schema` |
| `models.<model_name>.preprocessing.preprocessing_groups[].pipeline.log_transform` | `mapping` | No | Not explicitly defined in the parser code. | enabled, method, epsilon | Validated by `_define_log_transform_schema`. | `_define_pipeline_schema` |
| `models.<model_name>.preprocessing.preprocessing_groups[].pipeline.log_transform.enabled` | `boolean` | Yes if log_transform is present | Not explicitly defined in the parser code. | true, false | Must be YAML boolean. | `_define_log_transform_schema` |
| `models.<model_name>.preprocessing.preprocessing_groups[].pipeline.log_transform.method` | `string` | No | Not explicitly defined in the parser code. | log, log1p | Must be one of the allowed method strings. | `_define_log_transform_schema` |
| `models.<model_name>.preprocessing.preprocessing_groups[].pipeline.log_transform.epsilon` | `float` | No | Not explicitly defined in the parser code. | float > 0 | Must be a float and strictly positive. | `_define_log_transform_schema` |
| `models.<model_name>.preprocessing.preprocessing_groups[].pipeline.winsorize` | `mapping` | No | Not explicitly defined in the parser code. | enabled, limits | Validated by `_define_winsorize_schema`. | `_define_pipeline_schema` |
| `models.<model_name>.preprocessing.preprocessing_groups[].pipeline.winsorize.enabled` | `boolean` | Yes if winsorize is present | Not explicitly defined in the parser code. | true, false | Must be YAML boolean. | `_define_winsorize_schema` |
| `models.<model_name>.preprocessing.preprocessing_groups[].pipeline.winsorize.limits` | `list[float]` | No | Not explicitly defined in the parser code. | [lower, upper] | Exactly two floats with `0 <= lower < upper <= 1`. | `_define_winsorize_schema` |
| `models.<model_name>.preprocessing.preprocessing_groups[].pipeline.scaling` | `mapping` | No | Not explicitly defined in the parser code. | enabled, method, range | Validated by `_define_scaling_schema`. | `_define_pipeline_schema` |
| `models.<model_name>.preprocessing.preprocessing_groups[].pipeline.scaling.enabled` | `boolean` | Yes if scaling is present | Not explicitly defined in the parser code. | true, false | Must be YAML boolean. | `_define_scaling_schema` |
| `models.<model_name>.preprocessing.preprocessing_groups[].pipeline.scaling.method` | `string` | No | Not explicitly defined in the parser code. | minmax, standard, robust | Must be one of the allowed method strings. | `_define_scaling_schema` |
| `models.<model_name>.preprocessing.preprocessing_groups[].pipeline.scaling.range` | `list[number]` | No | Not explicitly defined in the parser code. | [min, max] | Exactly two int or float values with first value lower than second value. | `_define_scaling_schema` |
| `experiments[].models[].preprocessing.preprocessing_groups[].pipeline.log_transform` | `mapping` | No | Not explicitly defined in the parser code. | enabled, method, epsilon | Validated by `_define_log_transform_schema`. | `_define_pipeline_schema` |
| `experiments[].models[].preprocessing.preprocessing_groups[].pipeline.log_transform.enabled` | `boolean` | Yes if log_transform is present | Not explicitly defined in the parser code. | true, false | Must be YAML boolean. | `_define_log_transform_schema` |
| `experiments[].models[].preprocessing.preprocessing_groups[].pipeline.log_transform.method` | `string` | No | Not explicitly defined in the parser code. | log, log1p | Must be one of the allowed method strings. | `_define_log_transform_schema` |
| `experiments[].models[].preprocessing.preprocessing_groups[].pipeline.log_transform.epsilon` | `float` | No | Not explicitly defined in the parser code. | float > 0 | Must be a float and strictly positive. | `_define_log_transform_schema` |
| `experiments[].models[].preprocessing.preprocessing_groups[].pipeline.winsorize` | `mapping` | No | Not explicitly defined in the parser code. | enabled, limits | Validated by `_define_winsorize_schema`. | `_define_pipeline_schema` |
| `experiments[].models[].preprocessing.preprocessing_groups[].pipeline.winsorize.enabled` | `boolean` | Yes if winsorize is present | Not explicitly defined in the parser code. | true, false | Must be YAML boolean. | `_define_winsorize_schema` |
| `experiments[].models[].preprocessing.preprocessing_groups[].pipeline.winsorize.limits` | `list[float]` | No | Not explicitly defined in the parser code. | [lower, upper] | Exactly two floats with `0 <= lower < upper <= 1`. | `_define_winsorize_schema` |
| `experiments[].models[].preprocessing.preprocessing_groups[].pipeline.scaling` | `mapping` | No | Not explicitly defined in the parser code. | enabled, method, range | Validated by `_define_scaling_schema`. | `_define_pipeline_schema` |
| `experiments[].models[].preprocessing.preprocessing_groups[].pipeline.scaling.enabled` | `boolean` | Yes if scaling is present | Not explicitly defined in the parser code. | true, false | Must be YAML boolean. | `_define_scaling_schema` |
| `experiments[].models[].preprocessing.preprocessing_groups[].pipeline.scaling.method` | `string` | No | Not explicitly defined in the parser code. | minmax, standard, robust | Must be one of the allowed method strings. | `_define_scaling_schema` |
| `experiments[].models[].preprocessing.preprocessing_groups[].pipeline.scaling.range` | `list[number]` | No | Not explicitly defined in the parser code. | [min, max] | Exactly two int or float values with first value lower than second value. | `_define_scaling_schema` |
| `datasets.<dataset_name>.differencing.enabled` | `boolean` | Yes if differencing is present | Not explicitly defined in the parser code. | true, false | Must be YAML boolean. | `_define_differencing_schema` |
| `datasets.<dataset_name>.differencing.order` | `integer` | No | Not explicitly defined in the parser code. | integer >= 0 | Non-seasonal differencing order must be non-negative. | `_define_differencing_schema` |
| `datasets.<dataset_name>.differencing.seasonal_order` | `integer` | No | Not explicitly defined in the parser code. | integer >= 0 | Seasonal differencing order must be non-negative. | `_define_differencing_schema` |
| `datasets.<dataset_name>.differencing.seasonal_period` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Seasonal period must be strictly positive. | `_define_differencing_schema` |
| `models.<model_name>.preprocessing` | `mapping` | No | Not explicitly defined in the parser code. | time_features, preprocessing_groups | Validated by `_define_preprocessing_groups_schema`. | `_define_preprocessing_groups_schema` |
| `models.<model_name>.preprocessing.time_features` | `list[string]` | No | Not explicitly defined in the parser code. | Values from `ALLOWED_TIME_FEATURES` | Every item must belong to the allowed time feature list. | `_define_preprocessing_groups_schema` |
| `models.<model_name>.preprocessing.preprocessing_groups` | `list[mapping]` | Yes if preprocessing is present | Not explicitly defined in the parser code. | List of preprocessing group objects | Every group must define `name`, `apply_to`, and `pipeline`. | `_define_preprocessing_groups_schema` |
| `models.<model_name>.preprocessing.preprocessing_groups[].name` | `string` | Yes | Not explicitly defined in the parser code. | Any string | Must be a string. | `_define_preprocessing_groups_schema` |
| `models.<model_name>.preprocessing.preprocessing_groups[].apply_to` | `string or list[string]` | Yes | Not explicitly defined in the parser code. | `__targets__` or list of column names | A single string is valid only when it is exactly `__targets__`. For named columns, use a list of strings. | `_define_preprocessing_groups_schema` |
| `models.<model_name>.preprocessing.preprocessing_groups[].pipeline` | `mapping` | Yes | Not explicitly defined in the parser code. | log_transform, winsorize, scaling | Validated by `_define_pipeline_schema`. | `_define_preprocessing_groups_schema` |
| `experiments[].models[].preprocessing` | `mapping` | No | Not explicitly defined in the parser code. | time_features, preprocessing_groups | Validated by `_define_preprocessing_groups_schema`. | `_define_preprocessing_groups_schema` |
| `experiments[].models[].preprocessing.time_features` | `list[string]` | No | Not explicitly defined in the parser code. | Values from `ALLOWED_TIME_FEATURES` | Every item must belong to the allowed time feature list. | `_define_preprocessing_groups_schema` |
| `experiments[].models[].preprocessing.preprocessing_groups` | `list[mapping]` | Yes if preprocessing is present | Not explicitly defined in the parser code. | List of preprocessing group objects | Every group must define `name`, `apply_to`, and `pipeline`. | `_define_preprocessing_groups_schema` |
| `experiments[].models[].preprocessing.preprocessing_groups[].name` | `string` | Yes | Not explicitly defined in the parser code. | Any string | Must be a string. | `_define_preprocessing_groups_schema` |
| `experiments[].models[].preprocessing.preprocessing_groups[].apply_to` | `string or list[string]` | Yes | Not explicitly defined in the parser code. | `__targets__` or list of column names | A single string is valid only when it is exactly `__targets__`. For named columns, use a list of strings. | `_define_preprocessing_groups_schema` |
| `experiments[].models[].preprocessing.preprocessing_groups[].pipeline` | `mapping` | Yes | Not explicitly defined in the parser code. | log_transform, winsorize, scaling | Validated by `_define_pipeline_schema`. | `_define_preprocessing_groups_schema` |

### Experiments

| YAML path | Type | Required | Default | Allowed values | Validation | Defined in |
|---|---|---|---|---|---|---|
| `experiments` | `list[mapping]` | No | Not explicitly defined in the parser code. | Not explicitly constrained beyond type validation. | If present, every experiment must match `experiment_schema`. | `validate_config.experiment_schema` |
| `experiments[].name` | `string` | Yes | Not explicitly defined in the parser code. | Any non-empty string is recommended | Parser only checks string type. | `validate_config.experiment_schema` |
| `experiments[].description` | `string` | No | Not explicitly defined in the parser code. | Any string | Parser only checks string type. | `validate_config.experiment_schema` |
| `experiments[].dataset` | `string` | Yes | Not explicitly defined in the parser code. | Name of a dataset defined in `datasets` | Parser checks string type only. It does not verify dataset existence in this function. | `validate_config.experiment_schema` |
| `experiments[].models` | `list[string or mapping]` | Yes | Not explicitly defined in the parser code. | Model names as strings or model override objects | String entries are converted to `{name: <string>}` during validation. | `validate_config.experiment_schema` |
| `experiments[].models[].name` | `string` | Yes for mapping form | Not explicitly defined in the parser code. | Name of a model config defined in `models` | Cross-check verifies that this model name exists in the global `models` section. | `validate_config.experiment_schema` |
| `experiments[].models[].use_exogenous` | `boolean` | No | Not explicitly defined in the parser code. | true, false | Must be YAML boolean. | `validate_config.experiment_schema` |
| `experiments[].models[].past_covariates` | `list[string]` | No | Not explicitly defined in the parser code. | Any list of strings | No parser-level check that the columns exist in the dataset. | `validate_config.experiment_schema` |
| `experiments[].models[].future_covariates` | `list[string]` | No | Not explicitly defined in the parser code. | Any list of strings | No parser-level check that the columns exist in the dataset. | `validate_config.experiment_schema` |
| `experiments[].models[].use_raw_data_source` | `boolean` | No | Not explicitly defined in the parser code. | true, false | Must be YAML boolean. | `validate_config.experiment_schema` |
| `experiments[].validation_setup` | `mapping` | Yes | Not explicitly defined in the parser code. | forecast_steps, n_folds, window_size, early_stopping_validation_percentage, evaluation_metric | Required nested validation setup. | `validate_config.experiment_schema` |
| `experiments[].validation_setup.forecast_steps` | `integer` | Yes | Not explicitly defined in the parser code. | integer > 0 | Must be strictly positive. | `validate_config.experiment_schema` |
| `experiments[].validation_setup.n_folds` | `integer` | Yes | Not explicitly defined in the parser code. | integer > 0 | Must be strictly positive. | `validate_config.experiment_schema` |
| `experiments[].validation_setup.window_size` | `integer` | Yes | Not explicitly defined in the parser code. | integer > 0 | Must be strictly positive. | `validate_config.experiment_schema` |
| `experiments[].validation_setup.early_stopping_validation_percentage` | `int or float` | No | Not explicitly defined in the parser code. | 0 < value <= 100 | Must be in the open-closed interval `(0, 100]`. | `validate_config.experiment_schema` |
| `experiments[].validation_setup.evaluation_metric` | `string` | No | Not explicitly defined in the parser code. | mse, rmse, mae, smape, mase | Must be one of the allowed metric names. | `validate_config.experiment_schema` |

### Models: Common Structure and Optimization

| YAML path | Type | Required | Default | Allowed values | Validation | Defined in |
|---|---|---|---|---|---|---|
| `models` | `mapping` | Yes | Not explicitly defined in the parser code. | Not explicitly constrained beyond type validation. | Must be non-empty. Keys are arbitrary model configuration names, values are model configs. | `validate_config.main_schema.models` |
| `models.<model_name>` | `mapping` | Yes | Not explicitly defined in the parser code. | Not explicitly constrained beyond type validation. | Each model must contain `type` and is then validated against the type-specific schema. | `validate_each_model_against_its_schema` |
| `models.<model_name>.type` | `string` | Yes | Not explicitly defined in the parser code. | arima, sarima, var, lstm, transformer, simple_seasonal | Must match one of the keys in `MODEL_SCHEMAS`. | `validate_each_model_against_its_schema` |
| `models.<model_name>.optimize` | `boolean` | No | Not explicitly defined in the parser code. | true, false | Must be YAML boolean. | `MODEL_SCHEMAS[*].optimization` |
| `models.<model_name>.optimization` | `mapping` | No | Not explicitly defined in the parser code. | method, n_trials, params, pruner_config | If present, `method` is required. | `MODEL_SCHEMAS[*].optimization` |
| `models.<model_name>.optimization.method` | `string` | Yes if optimization is present | Not explicitly defined in the parser code. | grid, random, optuna | Must be one of the allowed method strings. | `MODEL_SCHEMAS[*].optimization` |
| `models.<model_name>.optimization.n_trials` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Must be strictly positive. | `MODEL_SCHEMAS[*].optimization` |
| `models.<model_name>.optimization.params` | `mapping` | No | Not explicitly defined in the parser code. | Model-specific search-space parameters | Validated against the model-specific optimization parameter schema. | `MODEL_SCHEMAS[*].optimization` |
| `models.<model_name>.optimization.pruner_config` | `mapping` | No | Not explicitly defined in the parser code. | See pruner_config reference | Allowed for arima, sarima, var, lstm, and transformer. Not present in simple_seasonal optimization schema. | `MODEL_SCHEMAS[*].optimization` |
| `models.<model_name>.optimization.params.<param>` | `list or range mapping` | No | Not explicitly defined in the parser code. | Categorical list or `{min, max, step?, log?}` depending on parameter schema | Integer parameters use integer ranges, float parameters use float ranges, selected categorical parameters use lists. | `*_opt_param_schema` |
| `models.<model_name>.optimization.params.<param>.min` | `integer or float` | Yes for range mapping | Not explicitly defined in the parser code. | Must match the expected numeric type | Must be `<= max`. Float ranges use Python float validation, so use `0.0` rather than `0` where needed. | `integer_range / float_range` |
| `models.<model_name>.optimization.params.<param>.max` | `integer or float` | Yes for range mapping | Not explicitly defined in the parser code. | Must match the expected numeric type | Must be `>= min`. | `integer_range / float_range` |
| `models.<model_name>.optimization.params.<param>.step` | `integer or float` | No | Not explicitly defined in the parser code. | step > 0 | Must match integer or float range type. | `integer_range / float_range` |
| `models.<model_name>.optimization.params.<param>.log` | `boolean` | No | Not explicitly defined in the parser code. | true, false | Available only in float range schema. | `float_range` |
| `models.<model_name>.optimization.params.scheduler_config.<scheduler_param>` | `list or range mapping` | No | Not explicitly defined in the parser code. | Categorical list or `{min, max, step?, log?}` depending on parameter schema | Integer parameters use integer ranges, float parameters use float ranges, selected categorical parameters use lists. | `*_opt_param_schema` |
| `models.<model_name>.optimization.params.scheduler_config.<scheduler_param>.min` | `integer or float` | Yes for range mapping | Not explicitly defined in the parser code. | Must match the expected numeric type | Must be `<= max`. Float ranges use Python float validation, so use `0.0` rather than `0` where needed. | `integer_range / float_range` |
| `models.<model_name>.optimization.params.scheduler_config.<scheduler_param>.max` | `integer or float` | Yes for range mapping | Not explicitly defined in the parser code. | Must match the expected numeric type | Must be `>= min`. | `integer_range / float_range` |
| `models.<model_name>.optimization.params.scheduler_config.<scheduler_param>.step` | `integer or float` | No | Not explicitly defined in the parser code. | step > 0 | Must match integer or float range type. | `integer_range / float_range` |
| `models.<model_name>.optimization.params.scheduler_config.<scheduler_param>.log` | `boolean` | No | Not explicitly defined in the parser code. | true, false | Available only in float range schema. | `float_range` |
| `models.<model_name>.optimization.pruner_config.type` | `string` | No | Not explicitly defined in the parser code. | median, percentile, hyperband, threshold, patient, none | Valid where `pruner_config_schema` is included. | `pruner_config_schema` |
| `models.<model_name>.optimization.pruner_config.n_startup_trials` | `integer` | No | Not explicitly defined in the parser code. | integer >= 0 | Valid where `pruner_config_schema` is included. | `pruner_config_schema` |
| `models.<model_name>.optimization.pruner_config.n_warmup_steps` | `integer` | No | Not explicitly defined in the parser code. | integer >= 0 | Valid where `pruner_config_schema` is included. | `pruner_config_schema` |
| `models.<model_name>.optimization.pruner_config.interval_steps` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Valid where `pruner_config_schema` is included. | `pruner_config_schema` |
| `models.<model_name>.optimization.pruner_config.n_min_trials` | `integer` | No | Not explicitly defined in the parser code. | integer >= 1 | Valid where `pruner_config_schema` is included. | `pruner_config_schema` |
| `models.<model_name>.optimization.pruner_config.percentile` | `float` | No | Not explicitly defined in the parser code. | 0 < value < 100 | Valid where `pruner_config_schema` is included. | `pruner_config_schema` |
| `models.<model_name>.optimization.pruner_config.min_resource` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Valid where `pruner_config_schema` is included. | `pruner_config_schema` |
| `models.<model_name>.optimization.pruner_config.max_resource` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Valid where `pruner_config_schema` is included. | `pruner_config_schema` |
| `models.<model_name>.optimization.pruner_config.reduction_factor` | `integer` | No | Not explicitly defined in the parser code. | integer >= 2 | Valid where `pruner_config_schema` is included. | `pruner_config_schema` |
| `models.<model_name>.optimization.pruner_config.lower` | `float` | No | Not explicitly defined in the parser code. | Any float | Valid where `pruner_config_schema` is included. | `pruner_config_schema` |
| `models.<model_name>.optimization.pruner_config.upper` | `float` | No | Not explicitly defined in the parser code. | Any float | Valid where `pruner_config_schema` is included. | `pruner_config_schema` |
| `models.<model_name>.optimization.pruner_config.wrapped_pruner` | `mapping` | No | Not explicitly defined in the parser code. | Any mapping | Valid where `pruner_config_schema` is included. | `pruner_config_schema` |
| `models.<model_name>.optimization.pruner_config.patience` | `integer` | No | Not explicitly defined in the parser code. | integer >= 0 | Valid where `pruner_config_schema` is included. | `pruner_config_schema` |

### Model Type: `arima`

| YAML path | Type | Required | Default | Allowed values | Validation | Defined in |
|---|---|---|---|---|---|---|
| `models.<model_name>.p` | `integer` | Yes | Not explicitly defined in the parser code. | integer >= 0 | Valid for model type `arima`. | `MODEL_SCHEMAS['arima']` |
| `models.<model_name>.d` | `integer` | Yes | Not explicitly defined in the parser code. | integer >= 0 | Valid for model type `arima`. | `MODEL_SCHEMAS['arima']` |
| `models.<model_name>.q` | `integer` | Yes | Not explicitly defined in the parser code. | integer >= 0 | Valid for model type `arima`. | `MODEL_SCHEMAS['arima']` |
| `models.<model_name>.trend` | `string` | No | Not explicitly defined in the parser code. | n, c, t, ct | Valid for model type `arima`. | `MODEL_SCHEMAS['arima']` |
| `models.<model_name>.use_exogenous` | `boolean` | No | Not explicitly defined in the parser code. | true, false | Valid for model type `arima`. | `MODEL_SCHEMAS['arima']` |
| `models.<model_name>.enforce_stationarity` | `boolean` | No | Not explicitly defined in the parser code. | true, false | Valid for model type `arima`. | `MODEL_SCHEMAS['arima']` |
| `models.<model_name>.enforce_invertibility` | `boolean` | No | Not explicitly defined in the parser code. | true, false | Valid for model type `arima`. | `MODEL_SCHEMAS['arima']` |
| `models.<model_name>.method` | `string` | No | Not explicitly defined in the parser code. | lbfgs, css-mle, bfgs, newton, nm, powell | Valid for model type `arima`. | `MODEL_SCHEMAS['arima']` |
| `models.<model_name>.maxiter` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Valid for model type `arima`. | `MODEL_SCHEMAS['arima']` |
| `models.<model_name>.remove_data_after_fit` | `boolean` | No | Not explicitly defined in the parser code. | true, false | Valid for model type `arima`. | `MODEL_SCHEMAS['arima']` |

### Model Type: `sarima`

| YAML path | Type | Required | Default | Allowed values | Validation | Defined in |
|---|---|---|---|---|---|---|
| `models.<model_name>.p` | `integer` | Yes | Not explicitly defined in the parser code. | integer >= 0 | Valid for model type `sarima`. | `MODEL_SCHEMAS['sarima']` |
| `models.<model_name>.d` | `integer` | Yes | Not explicitly defined in the parser code. | integer >= 0 | Valid for model type `sarima`. | `MODEL_SCHEMAS['sarima']` |
| `models.<model_name>.q` | `integer` | Yes | Not explicitly defined in the parser code. | integer >= 0 | Valid for model type `sarima`. | `MODEL_SCHEMAS['sarima']` |
| `models.<model_name>.P` | `integer` | Yes | Not explicitly defined in the parser code. | integer >= 0 | Valid for model type `sarima`. | `MODEL_SCHEMAS['sarima']` |
| `models.<model_name>.D` | `integer` | Yes | Not explicitly defined in the parser code. | integer >= 0 | Valid for model type `sarima`. | `MODEL_SCHEMAS['sarima']` |
| `models.<model_name>.Q` | `integer` | Yes | Not explicitly defined in the parser code. | integer >= 0 | Valid for model type `sarima`. | `MODEL_SCHEMAS['sarima']` |
| `models.<model_name>.trend` | `string` | No | Not explicitly defined in the parser code. | n, c, t, ct | Valid for model type `sarima`. | `MODEL_SCHEMAS['sarima']` |
| `models.<model_name>.seasonal_period` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Valid for model type `sarima`. | `MODEL_SCHEMAS['sarima']` |
| `models.<model_name>.use_exogenous` | `boolean` | No | Not explicitly defined in the parser code. | true, false | Valid for model type `sarima`. | `MODEL_SCHEMAS['sarima']` |
| `models.<model_name>.enforce_stationarity` | `boolean` | No | Not explicitly defined in the parser code. | true, false | Valid for model type `sarima`. | `MODEL_SCHEMAS['sarima']` |
| `models.<model_name>.enforce_invertibility` | `boolean` | No | Not explicitly defined in the parser code. | true, false | Valid for model type `sarima`. | `MODEL_SCHEMAS['sarima']` |
| `models.<model_name>.method` | `string` | No | Not explicitly defined in the parser code. | lbfgs, css-mle, bfgs, newton, nm, powell | Valid for model type `sarima`. | `MODEL_SCHEMAS['sarima']` |
| `models.<model_name>.maxiter` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Valid for model type `sarima`. | `MODEL_SCHEMAS['sarima']` |
| `models.<model_name>.remove_data_after_fit` | `boolean` | No | Not explicitly defined in the parser code. | true, false | Valid for model type `sarima`. | `MODEL_SCHEMAS['sarima']` |

### Model Type: `var`

| YAML path | Type | Required | Default | Allowed values | Validation | Defined in |
|---|---|---|---|---|---|---|
| `models.<model_name>.max_lags` | `integer` | Yes | Not explicitly defined in the parser code. | integer > 0 | Valid for model type `var`. | `MODEL_SCHEMAS['var']` |
| `models.<model_name>.ic` | `string` | No | Not explicitly defined in the parser code. | aic, bic, hqic | Valid for model type `var`. | `MODEL_SCHEMAS['var']` |
| `models.<model_name>.trend` | `string` | No | Not explicitly defined in the parser code. | n, c, t, ct | Valid for model type `var`. | `MODEL_SCHEMAS['var']` |
| `models.<model_name>.error_cov_type` | `string` | No | Not explicitly defined in the parser code. | unstructured, diagonal, scalar | Valid for model type `var`. | `MODEL_SCHEMAS['var']` |
| `models.<model_name>.maxiter` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Valid for model type `var`. | `MODEL_SCHEMAS['var']` |
| `models.<model_name>.use_exogenous` | `boolean` | No | Not explicitly defined in the parser code. | true, false | Valid for model type `var`. | `MODEL_SCHEMAS['var']` |
| `models.<model_name>.enforce_stationarity` | `boolean` | No | Not explicitly defined in the parser code. | true, false | Valid for model type `var`. | `MODEL_SCHEMAS['var']` |
| `models.<model_name>.enforce_invertibility` | `boolean` | No | Not explicitly defined in the parser code. | true, false | Valid for model type `var`. | `MODEL_SCHEMAS['var']` |
| `models.<model_name>.method` | `string` | No | Not explicitly defined in the parser code. | lbfgs, bfgs, newton, nm, powell | Valid for model type `var`. | `MODEL_SCHEMAS['var']` |
| `models.<model_name>.remove_data_after_fit` | `boolean` | No | Not explicitly defined in the parser code. | true, false | Valid for model type `var`. | `MODEL_SCHEMAS['var']` |

### Model Type: `simple_seasonal`

| YAML path | Type | Required | Default | Allowed values | Validation | Defined in |
|---|---|---|---|---|---|---|
| `models.<model_name>.seasonal_period` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Valid for model type `simple_seasonal`. | `MODEL_SCHEMAS['simple_seasonal']` |

### Model Type: `lstm`

| YAML path | Type | Required | Default | Allowed values | Validation | Defined in |
|---|---|---|---|---|---|---|
| `models.<model_name>.strategy` | `string` | Yes | Not explicitly defined in the parser code. | direct, iterative | Valid for model type `lstm`. | `MODEL_SCHEMAS['lstm']` |
| `models.<model_name>.hidden_size` | `integer` | Yes | Not explicitly defined in the parser code. | integer > 0 | Valid for model type `lstm`. | `MODEL_SCHEMAS['lstm']` |
| `models.<model_name>.num_layers` | `integer` | Yes | Not explicitly defined in the parser code. | integer > 0 | Valid for model type `lstm`. | `MODEL_SCHEMAS['lstm']` |
| `models.<model_name>.dropout` | `float` | No | Not explicitly defined in the parser code. | 0.0 <= value < 1.0 | Valid for model type `lstm`. | `MODEL_SCHEMAS['lstm']` |
| `models.<model_name>.batch_size` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Valid for model type `lstm`. | `MODEL_SCHEMAS['lstm']` |
| `models.<model_name>.learning_rate` | `float` | No | Not explicitly defined in the parser code. | float > 0 | Valid for model type `lstm`. | `MODEL_SCHEMAS['lstm']` |
| `models.<model_name>.epochs` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Valid for model type `lstm`. | `MODEL_SCHEMAS['lstm']` |
| `models.<model_name>.early_stopping_patience` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Valid for model type `lstm`. | `MODEL_SCHEMAS['lstm']` |
| `models.<model_name>.min_epochs` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Valid for model type `lstm`. | `MODEL_SCHEMAS['lstm']` |
| `models.<model_name>.weight_decay` | `float` | No | Not explicitly defined in the parser code. | float >= 0 | Valid for model type `lstm`. | `MODEL_SCHEMAS['lstm']` |
| `models.<model_name>.num_workers` | `integer` | No | Not explicitly defined in the parser code. | integer >= 0 | Valid for model type `lstm`. | `MODEL_SCHEMAS['lstm']` |
| `models.<model_name>.use_exogenous` | `boolean` | No | Not explicitly defined in the parser code. | true, false | Valid for model type `lstm`. | `MODEL_SCHEMAS['lstm']` |
| `models.<model_name>.past_covariate_policy` | `string` | No | Not explicitly defined in the parser code. | frozen, last_window, zero, custom | Valid for model type `lstm`. | `MODEL_SCHEMAS['lstm']` |
| `models.<model_name>.future_covariate_mode` | `string` | No | Not explicitly defined in the parser code. | none, global, stepwise | Valid for model type `lstm`. | `MODEL_SCHEMAS['lstm']` |
| `models.<model_name>.future_context_config.pooling` | `string` | No | Not explicitly defined in the parser code. | mean, last, learnable | Valid for model type `lstm`. | `MODEL_SCHEMAS['lstm']` |
| `models.<model_name>.future_context_config.compression_dim` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Valid for model type `lstm`. | `MODEL_SCHEMAS['lstm']` |
| `models.<model_name>.future_context_config.dropout` | `float` | No | Not explicitly defined in the parser code. | 0.0 <= value < 1.0 | Valid for model type `lstm`. | `MODEL_SCHEMAS['lstm']` |
| `models.<model_name>.iterative_stateful` | `boolean` | No | Not explicitly defined in the parser code. | true, false | Valid for model type `lstm`. | `MODEL_SCHEMAS['lstm']` |
| `models.<model_name>.fc_dropout` | `float` | No | Not explicitly defined in the parser code. | 0.0 <= value <= 1.0 | Valid for model type `lstm`. | `MODEL_SCHEMAS['lstm']` |
| `models.<model_name>.input_noise_injection.enabled` | `boolean` | Yes if input_noise_injection is present | Not explicitly defined in the parser code. | true, false | Valid for model type `lstm`. | `MODEL_SCHEMAS['lstm']` |
| `models.<model_name>.input_noise_injection.std` | `float` | No | Not explicitly defined in the parser code. | float > 0 | Valid for model type `lstm`. | `MODEL_SCHEMAS['lstm']` |
| `models.<model_name>.input_noise_injection.probability` | `float` | No | Not explicitly defined in the parser code. | 0.0 <= value <= 1.0 | Valid for model type `lstm`. | `MODEL_SCHEMAS['lstm']` |

### Model Type: `transformer`

| YAML path | Type | Required | Default | Allowed values | Validation | Defined in |
|---|---|---|---|---|---|---|
| `models.<model_name>.hidden_size` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.num_heads` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.num_encoder_layers` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.num_decoder_layers` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.architecture` | `string` | No | Not explicitly defined in the parser code. | encoder-only, encoder-decoder | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.activation` | `string` | No | Not explicitly defined in the parser code. | relu, gelu | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.norm_first` | `boolean` | No | Not explicitly defined in the parser code. | true, false | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.positional_encoding_config.type` | `string` | No | Not explicitly defined in the parser code. | sinusoidal, learnable, none | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.positional_encoding_config.max_len` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.positional_encoding_config.pe_dropout` | `float` | No | Not explicitly defined in the parser code. | 0.0 <= value <= 1.0 | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.positional_encoding_config.scale_with_sqrt_hidden_size` | `boolean` | No | Not explicitly defined in the parser code. | true, false | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.readout` | `string` | No | Not explicitly defined in the parser code. | last, mean, max, cls | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.attention_type` | `string` | No | Not explicitly defined in the parser code. | full, local | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.attention_window_size` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.strategy` | `string` | No | Not explicitly defined in the parser code. | direct, iterative | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.dim_ff_multiplier` | `float` | No | Not explicitly defined in the parser code. | float > 0 | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.tgt_init` | `string` | No | Not explicitly defined in the parser code. | seasonal, trend, last_value, mean, median, zeros | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.seasonal_period` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.use_revin` | `boolean` | No | Not explicitly defined in the parser code. | true, false | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.revin_affine` | `boolean` | No | Not explicitly defined in the parser code. | true, false | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.revin_eps` | `float` | No | Not explicitly defined in the parser code. | float > 0 | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.revin_robust` | `boolean` | No | Not explicitly defined in the parser code. | true, false | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.dropout` | `float` | No | Not explicitly defined in the parser code. | 0.0 <= value < 1.0 | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.batch_size` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.learning_rate` | `float` | No | Not explicitly defined in the parser code. | float > 0 | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.epochs` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.early_stopping_patience` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.min_epochs` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.weight_decay` | `float` | No | Not explicitly defined in the parser code. | float >= 0 | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.num_workers` | `integer` | No | Not explicitly defined in the parser code. | integer >= 0 | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.use_exogenous` | `boolean` | No | Not explicitly defined in the parser code. | true, false | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.auxiliary_loss.enabled` | `boolean` | Yes if auxiliary_loss is present | Not explicitly defined in the parser code. | true, false | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.auxiliary_loss.weight` | `float` | No | Not explicitly defined in the parser code. | 0.0 <= value <= 1.0 | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.auxiliary_loss.position_weighting` | `boolean` | No | Not explicitly defined in the parser code. | true, false | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.prediction_noise.enabled` | `boolean` | Yes if prediction_noise is present | Not explicitly defined in the parser code. | true, false | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.prediction_noise.std` | `float` | No | Not explicitly defined in the parser code. | float > 0 | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.prediction_noise.schedule` | `string` | No | Not explicitly defined in the parser code. | constant, curriculum | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.iterative_decoder_mode` | `string` | No | Not explicitly defined in the parser code. | concat, buffer, auto | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.head_type` | `string` | No | Not explicitly defined in the parser code. | linear, mlp | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.output_head_strategy` | `string` | No | Not explicitly defined in the parser code. | shared, multiple | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.nan_guard_enabled` | `boolean` | No | Not explicitly defined in the parser code. | true, false | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.device_safety_checks` | `boolean` | No | Not explicitly defined in the parser code. | true, false | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.use_amp_inference` | `boolean` | No | Not explicitly defined in the parser code. | true, false | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.amp_inference_dtype` | `string or null` | No | Not explicitly defined in the parser code. | Any string or null | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |
| `models.<model_name>.debug_amp_inference` | `boolean` | No | Not explicitly defined in the parser code. | true, false | Valid for model type `transformer`. | `MODEL_SCHEMAS['transformer']` |

### Shared Neural, Scheduler, Pruner, and HPO Helpers

| YAML path | Type | Required | Default | Allowed values | Validation | Defined in |
|---|---|---|---|---|---|---|
| `models.<model_name>.loss` | `string` | No | Not explicitly defined in the parser code. | mse, mae, l1, huber | Valid for LSTM and Transformer model-level schemas. LSTM HPO additionally allows rmse and smape. | `loss_schema and model HPO schemas` |
| `models.<model_name>.loss_params.delta` | `int or float` | No | Not explicitly defined in the parser code. | number > 0 | Only `delta` is explicitly validated under `loss_params`. | `loss_schema` |
| `models.<model_name>.optimizer` | `string` | No | Not explicitly defined in the parser code. | adam, adamw | String is lowercased before membership check. | `optimizer_schema` |
| `models.<model_name>.optimizer_config.eps` | `float` | No | Not explicitly defined in the parser code. | float > 0 | Must be strictly positive. | `optimizer_schema` |
| `models.<model_name>.optimization.params.<param>` | `list or range mapping` | No | Not explicitly defined in the parser code. | Categorical list or `{min, max, step?, log?}` depending on parameter schema | Integer parameters use integer ranges, float parameters use float ranges, selected categorical parameters use lists. | `*_opt_param_schema` |
| `models.<model_name>.optimization.params.<param>.min` | `integer or float` | Yes for range mapping | Not explicitly defined in the parser code. | Must match the expected numeric type | Must be `<= max`. Float ranges use Python float validation, so use `0.0` rather than `0` where needed. | `integer_range / float_range` |
| `models.<model_name>.optimization.params.<param>.max` | `integer or float` | Yes for range mapping | Not explicitly defined in the parser code. | Must match the expected numeric type | Must be `>= min`. | `integer_range / float_range` |
| `models.<model_name>.optimization.params.<param>.step` | `integer or float` | No | Not explicitly defined in the parser code. | step > 0 | Must match integer or float range type. | `integer_range / float_range` |
| `models.<model_name>.optimization.params.scheduler_config.<scheduler_param>` | `list or range mapping` | No | Not explicitly defined in the parser code. | Categorical list or `{min, max, step?, log?}` depending on parameter schema | Integer parameters use integer ranges, float parameters use float ranges, selected categorical parameters use lists. | `*_opt_param_schema` |
| `models.<model_name>.optimization.params.scheduler_config.<scheduler_param>.min` | `integer or float` | Yes for range mapping | Not explicitly defined in the parser code. | Must match the expected numeric type | Must be `<= max`. Float ranges use Python float validation, so use `0.0` rather than `0` where needed. | `integer_range / float_range` |
| `models.<model_name>.optimization.params.scheduler_config.<scheduler_param>.max` | `integer or float` | Yes for range mapping | Not explicitly defined in the parser code. | Must match the expected numeric type | Must be `>= min`. | `integer_range / float_range` |
| `models.<model_name>.optimization.params.scheduler_config.<scheduler_param>.step` | `integer or float` | No | Not explicitly defined in the parser code. | step > 0 | Must match integer or float range type. | `integer_range / float_range` |
| `models.<model_name>.scheduler_config.type` | `string` | No | Not explicitly defined in the parser code. | onecycle, cosine, step, exponential, plateau | Valid for LSTM and Transformer model-level `scheduler_config`. | `_define_scheduler_config_schema` |
| `models.<model_name>.scheduler_config.max_lr` | `float` | No | Not explicitly defined in the parser code. | float > 0 | Valid for LSTM and Transformer model-level `scheduler_config`. | `_define_scheduler_config_schema` |
| `models.<model_name>.scheduler_config.pct_start` | `float` | No | Not explicitly defined in the parser code. | 0 < value < 1 | Valid for LSTM and Transformer model-level `scheduler_config`. | `_define_scheduler_config_schema` |
| `models.<model_name>.scheduler_config.div_factor` | `float` | No | Not explicitly defined in the parser code. | float > 1 | Valid for LSTM and Transformer model-level `scheduler_config`. | `_define_scheduler_config_schema` |
| `models.<model_name>.scheduler_config.final_div_factor` | `float` | No | Not explicitly defined in the parser code. | float > 1 | Valid for LSTM and Transformer model-level `scheduler_config`. | `_define_scheduler_config_schema` |
| `models.<model_name>.scheduler_config.anneal_strategy` | `string` | No | Not explicitly defined in the parser code. | cos, linear | Valid for LSTM and Transformer model-level `scheduler_config`. | `_define_scheduler_config_schema` |
| `models.<model_name>.scheduler_config.T_max` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Valid for LSTM and Transformer model-level `scheduler_config`. | `_define_scheduler_config_schema` |
| `models.<model_name>.scheduler_config.eta_min` | `float` | No | Not explicitly defined in the parser code. | float >= 0 | Valid for LSTM and Transformer model-level `scheduler_config`. | `_define_scheduler_config_schema` |
| `models.<model_name>.scheduler_config.step_size` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Valid for LSTM and Transformer model-level `scheduler_config`. | `_define_scheduler_config_schema` |
| `models.<model_name>.scheduler_config.gamma` | `float` | No | Not explicitly defined in the parser code. | 0 < value < 1 | Valid for LSTM and Transformer model-level `scheduler_config`. | `_define_scheduler_config_schema` |
| `models.<model_name>.scheduler_config.mode` | `string` | No | Not explicitly defined in the parser code. | min, max | Valid for LSTM and Transformer model-level `scheduler_config`. | `_define_scheduler_config_schema` |
| `models.<model_name>.scheduler_config.factor` | `float` | No | Not explicitly defined in the parser code. | 0 < value < 1 | Valid for LSTM and Transformer model-level `scheduler_config`. | `_define_scheduler_config_schema` |
| `models.<model_name>.scheduler_config.patience` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Valid for LSTM and Transformer model-level `scheduler_config`. | `_define_scheduler_config_schema` |
| `models.<model_name>.scheduler_config.threshold` | `float` | No | Not explicitly defined in the parser code. | float > 0 | Valid for LSTM and Transformer model-level `scheduler_config`. | `_define_scheduler_config_schema` |
| `models.<model_name>.scheduler_config.threshold_mode` | `string` | No | Not explicitly defined in the parser code. | rel, abs | Valid for LSTM and Transformer model-level `scheduler_config`. | `_define_scheduler_config_schema` |
| `models.<model_name>.scheduler_config.cooldown` | `integer` | No | Not explicitly defined in the parser code. | integer >= 0 | Valid for LSTM and Transformer model-level `scheduler_config`. | `_define_scheduler_config_schema` |
| `models.<model_name>.scheduler_config.min_lr` | `float` | No | Not explicitly defined in the parser code. | float >= 0 | Valid for LSTM and Transformer model-level `scheduler_config`. | `_define_scheduler_config_schema` |
| `models.<model_name>.scheduler_config.eps` | `float` | No | Not explicitly defined in the parser code. | float > 0 | Valid for LSTM and Transformer model-level `scheduler_config`. | `_define_scheduler_config_schema` |
| `models.<model_name>.optimization.pruner_config.type` | `string` | No | Not explicitly defined in the parser code. | median, percentile, hyperband, threshold, patient, none | Valid where `pruner_config_schema` is included. | `pruner_config_schema` |
| `models.<model_name>.optimization.pruner_config.n_startup_trials` | `integer` | No | Not explicitly defined in the parser code. | integer >= 0 | Valid where `pruner_config_schema` is included. | `pruner_config_schema` |
| `models.<model_name>.optimization.pruner_config.n_warmup_steps` | `integer` | No | Not explicitly defined in the parser code. | integer >= 0 | Valid where `pruner_config_schema` is included. | `pruner_config_schema` |
| `models.<model_name>.optimization.pruner_config.interval_steps` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Valid where `pruner_config_schema` is included. | `pruner_config_schema` |
| `models.<model_name>.optimization.pruner_config.n_min_trials` | `integer` | No | Not explicitly defined in the parser code. | integer >= 1 | Valid where `pruner_config_schema` is included. | `pruner_config_schema` |
| `models.<model_name>.optimization.pruner_config.percentile` | `float` | No | Not explicitly defined in the parser code. | 0 < value < 100 | Valid where `pruner_config_schema` is included. | `pruner_config_schema` |
| `models.<model_name>.optimization.pruner_config.min_resource` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Valid where `pruner_config_schema` is included. | `pruner_config_schema` |
| `models.<model_name>.optimization.pruner_config.max_resource` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Valid where `pruner_config_schema` is included. | `pruner_config_schema` |
| `models.<model_name>.optimization.pruner_config.reduction_factor` | `integer` | No | Not explicitly defined in the parser code. | integer >= 2 | Valid where `pruner_config_schema` is included. | `pruner_config_schema` |
| `models.<model_name>.optimization.pruner_config.lower` | `float` | No | Not explicitly defined in the parser code. | Any float | Valid where `pruner_config_schema` is included. | `pruner_config_schema` |
| `models.<model_name>.optimization.pruner_config.upper` | `float` | No | Not explicitly defined in the parser code. | Any float | Valid where `pruner_config_schema` is included. | `pruner_config_schema` |
| `models.<model_name>.optimization.pruner_config.wrapped_pruner` | `mapping` | No | Not explicitly defined in the parser code. | Any mapping | Valid where `pruner_config_schema` is included. | `pruner_config_schema` |
| `models.<model_name>.optimization.pruner_config.patience` | `integer` | No | Not explicitly defined in the parser code. | integer >= 0 | Valid where `pruner_config_schema` is included. | `pruner_config_schema` |
| `models.<model_name>.use_amp` | `boolean` | No | Not explicitly defined in the parser code. | true, false | Valid for LSTM and Transformer. | `advanced_training_schema` |
| `models.<model_name>.max_grad_norm` | `float` | No | Not explicitly defined in the parser code. | float >= 0 | Valid for LSTM and Transformer. | `advanced_training_schema` |
| `models.<model_name>.save_horizon_csv` | `boolean` | No | Not explicitly defined in the parser code. | true, false | Valid for LSTM and Transformer. | `advanced_training_schema` |
| `models.<model_name>.auto_tune_horizon` | `boolean` | No | Not explicitly defined in the parser code. | true, false | Valid for LSTM and Transformer. | `advanced_training_schema` |
| `models.<model_name>.degradation_threshold` | `float` | No | Not explicitly defined in the parser code. | float > 1.0 | Valid for LSTM and Transformer. | `advanced_training_schema` |
| `models.<model_name>.save_scheduler_plot` | `boolean` | No | Not explicitly defined in the parser code. | true, false | Valid for LSTM and Transformer. | `advanced_training_schema` |
| `models.<model_name>.save_scheduler_csv` | `boolean` | No | Not explicitly defined in the parser code. | true, false | Valid for LSTM and Transformer. | `advanced_training_schema` |
| `models.<model_name>.gradient_monitor.enabled` | `boolean` | Yes if gradient_monitor is present | Not explicitly defined in the parser code. | true, false | Valid for LSTM and Transformer. | `gradient_monitor_schema` |
| `models.<model_name>.gradient_monitor.log_dir` | `string` | No | Not explicitly defined in the parser code. | Any string | Valid for LSTM and Transformer. | `gradient_monitor_schema` |
| `models.<model_name>.gradient_monitor.log_interval` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Valid for LSTM and Transformer. | `gradient_monitor_schema` |
| `models.<model_name>.attention_capture.enabled` | `boolean` | Yes if attention_capture is present | Not explicitly defined in the parser code. | true, false | Valid for Transformer only. | `attention_capture_schema` |
| `models.<model_name>.attention_capture.log_dir` | `string` | No | Not explicitly defined in the parser code. | Any string | Valid for Transformer only. | `attention_capture_schema` |
| `models.<model_name>.hpo_constraints.max_complexity_small` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Valid for LSTM and Transformer. | `hpo_constraints_schema / hpo_lr_scaling_schema` |
| `models.<model_name>.hpo_constraints.max_complexity_medium` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Valid for LSTM and Transformer. | `hpo_constraints_schema / hpo_lr_scaling_schema` |
| `models.<model_name>.hpo_constraints.max_complexity_large` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Valid for LSTM and Transformer. | `hpo_constraints_schema / hpo_lr_scaling_schema` |
| `models.<model_name>.hpo_constraints.max_complexity_very_large` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Valid for LSTM and Transformer. | `hpo_constraints_schema / hpo_lr_scaling_schema` |
| `models.<model_name>.hpo_lr_scaling.mode` | `string` | No | Not explicitly defined in the parser code. | sqrt, linear | Valid for LSTM and Transformer. | `hpo_constraints_schema / hpo_lr_scaling_schema` |
| `models.<model_name>.hpo_lr_scaling.ref_batch` | `integer` | No | Not explicitly defined in the parser code. | integer > 0 | Valid for LSTM and Transformer. | `hpo_constraints_schema / hpo_lr_scaling_schema` |
| `models.<model_name>.hpo_lr_scaling.lr0` | `float` | No | Not explicitly defined in the parser code. | float > 0 | Valid for LSTM and Transformer. | `hpo_constraints_schema / hpo_lr_scaling_schema` |

## Cross-Parameter Dependencies

### Experiment model references

Every model referenced in `experiments[].models` must exist as a key in the global `models` section. The parser accepts either string entries or mapping entries:

```yaml
experiments:
  - name: "demo"
    dataset: "demo_energy"
    models:
      - "transformer_baseline"
      - name: "lstm_baseline"
        use_exogenous: true
```

String entries are transformed during validation into dictionaries of the form `{name: <model_name>}`.

### Model schema selected by `type`

Each `models.<model_name>` entry must contain `type`. The value must be one of:

```text
arima, sarima, var, lstm, transformer, simple_seasonal
```

The parser then validates the rest of the model configuration against the type-specific schema.

### `apply_to` format in preprocessing groups

`apply_to` accepts either the exact string `"__targets__"` or a list of strings. A single column name as a bare string is not accepted.

Correct:

```yaml
apply_to: "__targets__"
```

Correct:

```yaml
apply_to: ["temperature", "humidity"]
```

Invalid according to the parser:

```yaml
apply_to: "temperature"
```

### Scheduler parameters depend on scheduler type at runtime

The parser accepts all scheduler parameters in one shared schema. It validates each provided parameter independently, but it does not enforce scheduler-specific required fields. For example, `scheduler_config.type: "onecycle"` does not force `pct_start`, `div_factor`, or `final_div_factor` at parser level.

### `optimization.params` depends on model type

The allowed HPO search-space keys depend on the model type:

| Model type | Allowed HPO parameter families |
|---|---|
| `arima` | `p`, `d`, `q`, `P`, `D`, `Q`, `seasonal_period`, `trend` |
| `sarima` | `p`, `d`, `q`, `P`, `D`, `Q`, `seasonal_period`, `trend` |
| `var` | `max_lags`, `maxiter`, `ic`, `trend`, `error_cov_type` |
| `simple_seasonal` | `seasonal_period` |
| `lstm` | architecture, training, optimizer, loss, `iterative_stateful`, `scheduler_config` |
| `transformer` | architecture, attention, training, RevIN, scheduler, safety flags |

## Validation Rules

### YAML parsing

The parser uses `ruamel.yaml.YAML(typ="rt")` to retain line information for better error messages and `yaml.safe_load()` to create the actual plain Python configuration dictionary.

### Required top-level sections

`datasets` and `models` are required. `paths`, `logging`, and `experiments` are optional.

### Dataset file paths

`datasets.<dataset_name>.path` is validated with `os.path.exists(path)`. A configuration can fail validation before training starts if the dataset file is not available at the configured path.

### Frequency strings

`datasets.<dataset_name>.freq` is validated by calling `pandas.tseries.frequencies.to_offset(freq)`. Any string accepted by pandas as a frequency offset passes parser validation.

### Numeric ranges

Optimization ranges use either integer or float range schemas:

```yaml
hidden_size:
  min: 64
  max: 256
  step: 64

learning_rate:
  min: 0.0001
  max: 0.01
  log: true
```

Integer ranges require integer `min` and `max`. Float ranges require float `min` and `max`. In YAML, prefer `0.0` instead of `0` for float fields.

### Lists as categorical search spaces

Several HPO parameters accept categorical lists:

```yaml
hidden_size: [64, 128, 256]
dropout: [0.1, 0.2, 0.3]
trend: ["n", "c"]
```

The allowed list element type and allowed values are model-specific.

## Defaults and Override Behavior

### Parser-level defaults

Most parameters do not receive defaults in `load_config()`. If a parameter is optional and missing, the returned validated dictionary simply does not contain that key.

### `get_model_config()` optimization default

The helper `get_model_config(config_name, config_path)` loads the full config, selects one model, and injects this default if `optimization` is absent:

```python
{"method": "grid", "params": {}}
```

This behavior is specific to `get_model_config()` and is not a general `load_config()` default.

### Docstring defaults are not parser defaults

Some docstrings mention defaults, for example `log_transform.method`, `scaling.method`, `differencing.order`, or `epsilon`. The schema validates these fields when they are present, but does not inject defaults when they are absent.

### Experiment model string normalization

Inside `experiments[].models`, a string model reference is normalized into a mapping:

```yaml
models: ["transformer_baseline"]
```

becomes conceptually:

```yaml
models:
  - name: "transformer_baseline"
```

## Common Configuration Patterns

### Standard target scaling for a model

```yaml
models:
  transformer_baseline:
    type: "transformer"
    preprocessing:
      preprocessing_groups:
        - name: "target_scaling"
          apply_to: "__targets__"
          pipeline:
            scaling:
              enabled: true
              method: "standard"
```

### Transformer with RevIN

```yaml
models:
  transformer_revin:
    type: "transformer"
    architecture: "encoder-only"
    strategy: "direct"
    use_revin: true
    revin_affine: true
    revin_eps: 0.00001
    revin_robust: false
```

### Optuna search over learning rate

```yaml
models:
  transformer_hpo:
    type: "transformer"
    optimize: true
    optimization:
      method: "optuna"
      n_trials: 20
      params:
        learning_rate:
          min: 0.0001
          max: 0.01
          log: true
```

### OneCycle scheduler

```yaml
scheduler_config:
  type: "onecycle"
  pct_start: 0.3
  div_factor: 25.0
  final_div_factor: 10000.0
  anneal_strategy: "cos"
```

### Experiment-level model override

```yaml
experiments:
  - name: "with_covariates"
    dataset: "demo_energy"
    models:
      - name: "transformer_baseline"
        use_exogenous: true
        past_covariates: ["temperature", "humidity"]
    validation_setup:
      forecast_steps: 24
      n_folds: 3
      window_size: 168
```

## Troubleshooting

### `Configuration file not found`

Cause: the path passed to `load_config()` does not exist.

Fix: pass the correct path to the YAML file.

### `Configuration file is empty`

Cause: YAML file is empty or parses as a false-like value.

Fix: add at least the required `datasets` and `models` sections.

### `Dataset file path does not exist`

Cause: `datasets.<dataset_name>.path` points to a missing file.

Fix: correct the dataset path or run validation from the repository root where relative paths resolve correctly.

### `Model configuration '<name>' is missing required field 'type'`

Cause: each global model configuration must declare its model type.

Fix:

```yaml
models:
  my_model:
    type: "transformer"
```

### `Invalid model type`

Cause: `models.<model_name>.type` is not one of the supported model types.

Fix: use one of `arima`, `sarima`, `var`, `lstm`, `transformer`, or `simple_seasonal`.

### `Model configuration '<name>' used in experiment '<experiment>' is not defined`

Cause: an experiment references a model name that is not present under `models`.

Fix: either define the model globally or update the experiment model reference.

### Invalid `apply_to`

Cause: using a bare string column name instead of a list.

Fix: use `apply_to: ["column_name"]` for named columns, or `apply_to: "__targets__"` for targets.

### Invalid time feature

Cause: a feature in `time_features` is not present in `ALLOWED_TIME_FEATURES`.

Fix: use only documented time feature names.

## Documentation and Parser Consistency Notes

- `_define_differencing_schema()` exists and dataset-level `differencing` is validated, but `_define_pipeline_schema()` does not include `differencing` even though its docstring example shows `differencing` inside `preprocessing`. Treat differencing as a separate dataset-level field unless the parser is changed.
- Several docstrings mention defaults, such as `log_transform.method`, `log_transform.epsilon`, `scaling.method`, and `differencing.order`. These defaults are not injected by the parser.
- `scaling.range` is described in the docstring as required for `method: "minmax"`, but this dependency is not enforced by the schema.
- `scheduler_config.type` is optional in the parser, although runtime scheduler construction probably needs it when a scheduler is intended.
- `pruner_config.type` is optional in the parser, although runtime pruner construction probably needs it except for implicit defaults handled elsewhere.
- `simple_seasonal` model-level `seasonal_period` requires `> 0`, while its HPO search schema allows integer values `>= 0` in categorical-list form.
- Transformer model-level `revin_eps` requires `> 0`, while the transformer HPO schema allows `>= 0` in range or list-like forms.
- LSTM model-level `loss` allows `mse`, `mae`, `l1`, and `huber`, while LSTM HPO `loss` additionally allows `rmse` and `smape`. This may be intentional, but it is a parser-level inconsistency to verify against runtime loss handling.
- Experiment-level `dataset` is checked as a string, but unlike model references, dataset existence is not explicitly cross-checked in `validate_experiments_and_models()`.
- Covariate names are validated only as strings. The parser does not check whether covariate columns are present in the dataset file.
- `MODEL_TYPES` is declared, but actual model validation uses `MODEL_SCHEMAS`.

## Appendix: Parameter Summary Table

| YAML path | Type | Required | Default | Defined in | Short description |
|---|---|---|---|---|---|
| `datasets` | `mapping` | Yes | Not explicitly defined in the parser code. | `validate_config.main_schema` | Global dataset registry. Dataset names are referenced by experiments. |
| `datasets.<dataset_name>` | `mapping` | Yes | Not explicitly defined in the parser code. | `validate_config.main_schema.datasets` | Named dataset configuration. |
| `datasets.<dataset_name>.columns` | `list[string]` | Yes | Not explicitly defined in the parser code. | `validate_config.main_schema.datasets` | Target column names used by the forecasting task. |
| `datasets.<dataset_name>.differencing` | `mapping` | No | Not explicitly defined in the parser code. | `validate_config.main_schema.datasets` | Dataset-level differencing configuration. It is not included in `_define_pipeline_schema`. |
| `datasets.<dataset_name>.differencing.enabled` | `boolean` | Yes if differencing is present | Not explicitly defined in the parser code. | `_define_differencing_schema` | Enables or disables dataset-level differencing. |
| `datasets.<dataset_name>.differencing.order` | `integer` | No | Not explicitly defined in the parser code. | `_define_differencing_schema` | Non-seasonal differencing order. |
| `datasets.<dataset_name>.differencing.seasonal_order` | `integer` | No | Not explicitly defined in the parser code. | `_define_differencing_schema` | Seasonal differencing order. |
| `datasets.<dataset_name>.differencing.seasonal_period` | `integer` | No | Not explicitly defined in the parser code. | `_define_differencing_schema` | Period used for seasonal differencing. |
| `datasets.<dataset_name>.freq` | `string` | No | Not explicitly defined in the parser code. | `validate_config.main_schema.datasets` | Sampling frequency of the time series. |
| `datasets.<dataset_name>.future_covariates` | `list[string]` | No | Not explicitly defined in the parser code. | `validate_config.main_schema.datasets` | Covariates expected to be known for future forecast steps. |
| `datasets.<dataset_name>.past_covariates` | `list[string]` | No | Not explicitly defined in the parser code. | `validate_config.main_schema.datasets` | Covariates known from past observations. |
| `datasets.<dataset_name>.path` | `string` | Yes | Not explicitly defined in the parser code. | `validate_config.main_schema.datasets` | Path to the dataset file. |
| `datasets.<dataset_name>.preprocessing` | `mapping` | No | Not explicitly defined in the parser code. | `validate_config.main_schema.datasets` | Dataset-level preprocessing pipeline. |
| `datasets.<dataset_name>.preprocessing.log_transform` | `mapping` | No | Not explicitly defined in the parser code. | `_define_pipeline_schema` | Optional log transformation step. |
| `datasets.<dataset_name>.preprocessing.log_transform.enabled` | `boolean` | Yes if log_transform is present | Not explicitly defined in the parser code. | `_define_log_transform_schema` | Enables or disables the log transformation step. |
| `datasets.<dataset_name>.preprocessing.log_transform.epsilon` | `float` | No | Not explicitly defined in the parser code. | `_define_log_transform_schema` | Small positive offset for numerical safety. The parser validates but does not inject it. |
| `datasets.<dataset_name>.preprocessing.log_transform.method` | `string` | No | Not explicitly defined in the parser code. | `_define_log_transform_schema` | Logarithm variant. The docstring mentions `log1p` as a default, but the parser does not inject a default. |
| `datasets.<dataset_name>.preprocessing.scaling` | `mapping` | No | Not explicitly defined in the parser code. | `_define_pipeline_schema` | Optional scaling step. |
| `datasets.<dataset_name>.preprocessing.scaling.enabled` | `boolean` | Yes if scaling is present | Not explicitly defined in the parser code. | `_define_scaling_schema` | Enables or disables scaling. |
| `datasets.<dataset_name>.preprocessing.scaling.method` | `string` | No | Not explicitly defined in the parser code. | `_define_scaling_schema` | Scaling method. The docstring mentions `minmax` as a default, but the parser does not inject a default. |
| `datasets.<dataset_name>.preprocessing.scaling.range` | `list[number]` | No | Not explicitly defined in the parser code. | `_define_scaling_schema` | Target range for min-max scaling. Parser does not require it when method is `minmax`, although the docstring says it is required. |
| `datasets.<dataset_name>.preprocessing.winsorize` | `mapping` | No | Not explicitly defined in the parser code. | `_define_pipeline_schema` | Optional winsorization step for clipping extremes. |
| `datasets.<dataset_name>.preprocessing.winsorize.enabled` | `boolean` | Yes if winsorize is present | Not explicitly defined in the parser code. | `_define_winsorize_schema` | Enables or disables winsorization. |
| `datasets.<dataset_name>.preprocessing.winsorize.limits` | `list[float]` | No | Not explicitly defined in the parser code. | `_define_winsorize_schema` | Lower and upper percentile limits for clipping. |
| `datasets.<dataset_name>.time_features` | `list[string]` | No | Not explicitly defined in the parser code. | `validate_config.main_schema.datasets` | Time features generated from a datetime index or date-like source. |
| `experiments` | `list[mapping]` | No | Not explicitly defined in the parser code. | `validate_config.experiment_schema` | Experiment definitions, each binding a dataset, models, and validation setup. |
| `experiments[].dataset` | `string` | Yes | Not explicitly defined in the parser code. | `validate_config.experiment_schema` | Dataset used by the experiment. |
| `experiments[].description` | `string` | No | Not explicitly defined in the parser code. | `validate_config.experiment_schema` | Human-readable experiment description. |
| `experiments[].models` | `list[string or mapping]` | Yes | Not explicitly defined in the parser code. | `validate_config.experiment_schema` | Models included in the experiment. |
| `experiments[].models[].future_covariates` | `list[string]` | No | Not explicitly defined in the parser code. | `validate_config.experiment_schema` | Experiment-level future covariate override. |
| `experiments[].models[].name` | `string` | Yes for mapping form | Not explicitly defined in the parser code. | `validate_config.experiment_schema` | Model configuration name referenced by the experiment. |
| `experiments[].models[].past_covariates` | `list[string]` | No | Not explicitly defined in the parser code. | `validate_config.experiment_schema` | Experiment-level past covariate override. |
| `experiments[].models[].preprocessing` | `mapping` | No | Not explicitly defined in the parser code. | `_define_preprocessing_groups_schema` | Grouped preprocessing configuration used at model level and experiment-model override level. |
| `experiments[].models[].preprocessing.preprocessing_groups` | `list[mapping]` | Yes if preprocessing is present | Not explicitly defined in the parser code. | `_define_preprocessing_groups_schema` | Ordered list of named preprocessing groups. |
| `experiments[].models[].preprocessing.preprocessing_groups[].apply_to` | `string or list[string]` | Yes | Not explicitly defined in the parser code. | `_define_preprocessing_groups_schema` | Target selector for the group. |
| `experiments[].models[].preprocessing.preprocessing_groups[].name` | `string` | Yes | Not explicitly defined in the parser code. | `_define_preprocessing_groups_schema` | Name of the preprocessing group. |
| `experiments[].models[].preprocessing.preprocessing_groups[].pipeline` | `mapping` | Yes | Not explicitly defined in the parser code. | `_define_preprocessing_groups_schema` | Transformation pipeline applied by the group. |
| `experiments[].models[].preprocessing.preprocessing_groups[].pipeline.log_transform` | `mapping` | No | Not explicitly defined in the parser code. | `_define_pipeline_schema` | Optional log transformation step. |
| `experiments[].models[].preprocessing.preprocessing_groups[].pipeline.log_transform.enabled` | `boolean` | Yes if log_transform is present | Not explicitly defined in the parser code. | `_define_log_transform_schema` | Enables or disables the log transformation step. |
| `experiments[].models[].preprocessing.preprocessing_groups[].pipeline.log_transform.epsilon` | `float` | No | Not explicitly defined in the parser code. | `_define_log_transform_schema` | Small positive offset for numerical safety. The parser validates but does not inject it. |
| `experiments[].models[].preprocessing.preprocessing_groups[].pipeline.log_transform.method` | `string` | No | Not explicitly defined in the parser code. | `_define_log_transform_schema` | Logarithm variant. The docstring mentions `log1p` as a default, but the parser does not inject a default. |
| `experiments[].models[].preprocessing.preprocessing_groups[].pipeline.scaling` | `mapping` | No | Not explicitly defined in the parser code. | `_define_pipeline_schema` | Optional scaling step. |
| `experiments[].models[].preprocessing.preprocessing_groups[].pipeline.scaling.enabled` | `boolean` | Yes if scaling is present | Not explicitly defined in the parser code. | `_define_scaling_schema` | Enables or disables scaling. |
| `experiments[].models[].preprocessing.preprocessing_groups[].pipeline.scaling.method` | `string` | No | Not explicitly defined in the parser code. | `_define_scaling_schema` | Scaling method. The docstring mentions `minmax` as a default, but the parser does not inject a default. |
| `experiments[].models[].preprocessing.preprocessing_groups[].pipeline.scaling.range` | `list[number]` | No | Not explicitly defined in the parser code. | `_define_scaling_schema` | Target range for min-max scaling. Parser does not require it when method is `minmax`, although the docstring says it is required. |
| `experiments[].models[].preprocessing.preprocessing_groups[].pipeline.winsorize` | `mapping` | No | Not explicitly defined in the parser code. | `_define_pipeline_schema` | Optional winsorization step for clipping extremes. |
| `experiments[].models[].preprocessing.preprocessing_groups[].pipeline.winsorize.enabled` | `boolean` | Yes if winsorize is present | Not explicitly defined in the parser code. | `_define_winsorize_schema` | Enables or disables winsorization. |
| `experiments[].models[].preprocessing.preprocessing_groups[].pipeline.winsorize.limits` | `list[float]` | No | Not explicitly defined in the parser code. | `_define_winsorize_schema` | Lower and upper percentile limits for clipping. |
| `experiments[].models[].preprocessing.time_features` | `list[string]` | No | Not explicitly defined in the parser code. | `_define_preprocessing_groups_schema` | Optional time-feature list inside grouped preprocessing. This is separate from dataset-level `time_features`. |
| `experiments[].models[].use_exogenous` | `boolean` | No | Not explicitly defined in the parser code. | `validate_config.experiment_schema` | Experiment-level override for exogenous variable usage. |
| `experiments[].models[].use_raw_data_source` | `boolean` | No | Not explicitly defined in the parser code. | `validate_config.experiment_schema` | Experiment-level flag for raw data source usage. |
| `experiments[].name` | `string` | Yes | Not explicitly defined in the parser code. | `validate_config.experiment_schema` | Experiment identifier. |
| `experiments[].validation_setup` | `mapping` | Yes | Not explicitly defined in the parser code. | `validate_config.experiment_schema` | Walk-forward or cross-validation configuration. |
| `experiments[].validation_setup.early_stopping_validation_percentage` | `int or float` | No | Not explicitly defined in the parser code. | `validate_config.experiment_schema` | Percentage of training data used for early-stopping validation, typically in optimization mode. |
| `experiments[].validation_setup.evaluation_metric` | `string` | No | Not explicitly defined in the parser code. | `validate_config.experiment_schema` | Metric used to evaluate forecasts. |
| `experiments[].validation_setup.forecast_steps` | `integer` | Yes | Not explicitly defined in the parser code. | `validate_config.experiment_schema` | Forecast horizon in time steps. |
| `experiments[].validation_setup.n_folds` | `integer` | Yes | Not explicitly defined in the parser code. | `validate_config.experiment_schema` | Number of validation folds. |
| `experiments[].validation_setup.window_size` | `integer` | Yes | Not explicitly defined in the parser code. | `validate_config.experiment_schema` | Input history window size. |
| `logging` | `mapping` | No | Not explicitly defined in the parser code. | `validate_config.main_schema` | Optional logging configuration. Supports a legacy single level and a newer centralized logging layout. |
| `logging.console_level` | `string` | No | Not explicitly defined in the parser code. | `validate_config.main_schema.logging` | Console log level. |
| `logging.custom_levels.<module_name>` | `string` | No | Not explicitly defined in the parser code. | `validate_config.main_schema.logging` | Per-module logging level override. |
| `logging.environment` | `string` | No | Not explicitly defined in the parser code. | `validate_config.main_schema.logging` | Named logging environment. |
| `logging.file` | `string` | No | Not explicitly defined in the parser code. | `validate_config.main_schema.logging` | Path to a log file. Parser comment says it supports `{experiment_name}` placeholder. |
| `logging.file_level` | `string` | No | Not explicitly defined in the parser code. | `validate_config.main_schema.logging` | File log level. |
| `logging.level` | `string` | No | Not explicitly defined in the parser code. | `validate_config.main_schema.logging` | Legacy global log level. |
| `logging.use_context` | `boolean` | No | Not explicitly defined in the parser code. | `validate_config.main_schema.logging` | Enable contextual logging tags such as experiment, fold, or epoch. |
| `models` | `mapping` | Yes | Not explicitly defined in the parser code. | `validate_config.main_schema.models` | Global model configuration registry. |
| `models.<model_name>` | `mapping` | Yes | Not explicitly defined in the parser code. | `validate_each_model_against_its_schema` | Named model configuration. The key is a user-defined configuration name, not necessarily the model type. |
| `models.<model_name>.D` | `integer` | Yes | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['sarima']` | Seasonal differencing order. |
| `models.<model_name>.P` | `integer` | Yes | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['sarima']` | Seasonal autoregressive order. |
| `models.<model_name>.Q` | `integer` | Yes | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['sarima']` | Seasonal moving-average order. |
| `models.<model_name>.activation` | `string` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Feed-forward activation function. |
| `models.<model_name>.amp_inference_dtype` | `string or null` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Parser does not restrict dtype names beyond string or null. |
| `models.<model_name>.architecture` | `string` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Transformer architecture variant. |
| `models.<model_name>.attention_capture.enabled` | `boolean` | Yes if attention_capture is present | Not explicitly defined in the parser code. | `attention_capture_schema` | Enable attention capture. |
| `models.<model_name>.attention_capture.log_dir` | `string` | No | Not explicitly defined in the parser code. | `attention_capture_schema` | Directory for attention logs. |
| `models.<model_name>.attention_type` | `string` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Attention mechanism variant. |
| `models.<model_name>.attention_window_size` | `integer` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Local attention window size. |
| `models.<model_name>.auto_tune_horizon` | `boolean` | No | Not explicitly defined in the parser code. | `advanced_training_schema` | Enable automatic horizon tuning. |
| `models.<model_name>.auxiliary_loss.enabled` | `boolean` | Yes if auxiliary_loss is present | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Enable auxiliary multi-step loss. |
| `models.<model_name>.auxiliary_loss.position_weighting` | `boolean` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Enable position weighting for auxiliary loss. |
| `models.<model_name>.auxiliary_loss.weight` | `float` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Auxiliary loss weight. |
| `models.<model_name>.batch_size` | `integer` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['lstm']` | Training batch size. |
| `models.<model_name>.batch_size` | `integer` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Training batch size. |
| `models.<model_name>.d` | `integer` | Yes | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['arima']` | Differencing order. |
| `models.<model_name>.d` | `integer` | Yes | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['sarima']` | Non-seasonal differencing order. |
| `models.<model_name>.debug_amp_inference` | `boolean` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Enable AMP inference debug mode. |
| `models.<model_name>.degradation_threshold` | `float` | No | Not explicitly defined in the parser code. | `advanced_training_schema` | Threshold for degradation-based logic. |
| `models.<model_name>.device_safety_checks` | `boolean` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Enable device consistency checks. |
| `models.<model_name>.dim_ff_multiplier` | `float` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Feed-forward hidden dimension multiplier. |
| `models.<model_name>.dropout` | `float` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['lstm']` | Dropout probability. |
| `models.<model_name>.dropout` | `float` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Dropout probability. |
| `models.<model_name>.early_stopping_patience` | `integer` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['lstm']` | Early stopping patience. |
| `models.<model_name>.early_stopping_patience` | `integer` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Early stopping patience. |
| `models.<model_name>.enforce_invertibility` | `boolean` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['arima']` | Invertibility enforcement flag. |
| `models.<model_name>.enforce_invertibility` | `boolean` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['sarima']` | Invertibility enforcement flag. |
| `models.<model_name>.enforce_invertibility` | `boolean` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['var']` | Invertibility enforcement flag. |
| `models.<model_name>.enforce_stationarity` | `boolean` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['arima']` | Stationarity enforcement flag. |
| `models.<model_name>.enforce_stationarity` | `boolean` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['sarima']` | Stationarity enforcement flag. |
| `models.<model_name>.enforce_stationarity` | `boolean` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['var']` | Stationarity enforcement flag. |
| `models.<model_name>.epochs` | `integer` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['lstm']` | Maximum number of training epochs. |
| `models.<model_name>.epochs` | `integer` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Maximum number of epochs. |
| `models.<model_name>.error_cov_type` | `string` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['var']` | Error covariance structure. |
| `models.<model_name>.fc_dropout` | `float` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['lstm']` | Additional dropout before the fully connected output layer. |
| `models.<model_name>.future_context_config.compression_dim` | `integer` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['lstm']` | Future context compression dimension. |
| `models.<model_name>.future_context_config.dropout` | `float` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['lstm']` | Future context dropout. |
| `models.<model_name>.future_context_config.pooling` | `string` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['lstm']` | Future context pooling method. |
| `models.<model_name>.future_covariate_mode` | `string` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['lstm']` | Future covariate handling mode. |
| `models.<model_name>.gradient_monitor.enabled` | `boolean` | Yes if gradient_monitor is present | Not explicitly defined in the parser code. | `gradient_monitor_schema` | Enable gradient monitor. |
| `models.<model_name>.gradient_monitor.log_dir` | `string` | No | Not explicitly defined in the parser code. | `gradient_monitor_schema` | Directory for gradient logs. |
| `models.<model_name>.gradient_monitor.log_interval` | `integer` | No | Not explicitly defined in the parser code. | `gradient_monitor_schema` | Logging interval. |
| `models.<model_name>.head_type` | `string` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Output head type. |
| `models.<model_name>.hidden_size` | `integer` | Yes | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['lstm']` | LSTM hidden state size. |
| `models.<model_name>.hidden_size` | `integer` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Transformer hidden size. |
| `models.<model_name>.hpo_constraints.max_complexity_large` | `integer` | No | Not explicitly defined in the parser code. | `hpo_constraints_schema / hpo_lr_scaling_schema` | Complexity bound for large configurations. |
| `models.<model_name>.hpo_constraints.max_complexity_medium` | `integer` | No | Not explicitly defined in the parser code. | `hpo_constraints_schema / hpo_lr_scaling_schema` | Complexity bound for medium configurations. |
| `models.<model_name>.hpo_constraints.max_complexity_small` | `integer` | No | Not explicitly defined in the parser code. | `hpo_constraints_schema / hpo_lr_scaling_schema` | Complexity bound for small configurations. |
| `models.<model_name>.hpo_constraints.max_complexity_very_large` | `integer` | No | Not explicitly defined in the parser code. | `hpo_constraints_schema / hpo_lr_scaling_schema` | Complexity bound for very large configurations. |
| `models.<model_name>.hpo_lr_scaling.lr0` | `float` | No | Not explicitly defined in the parser code. | `hpo_constraints_schema / hpo_lr_scaling_schema` | Base learning rate for scaling. |
| `models.<model_name>.hpo_lr_scaling.mode` | `string` | No | Not explicitly defined in the parser code. | `hpo_constraints_schema / hpo_lr_scaling_schema` | Learning-rate scaling mode. |
| `models.<model_name>.hpo_lr_scaling.ref_batch` | `integer` | No | Not explicitly defined in the parser code. | `hpo_constraints_schema / hpo_lr_scaling_schema` | Reference batch size for learning-rate scaling. |
| `models.<model_name>.ic` | `string` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['var']` | Information criterion. |
| `models.<model_name>.input_noise_injection.enabled` | `boolean` | Yes if input_noise_injection is present | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['lstm']` | Enable input noise injection. |
| `models.<model_name>.input_noise_injection.probability` | `float` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['lstm']` | Probability of applying input noise. |
| `models.<model_name>.input_noise_injection.std` | `float` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['lstm']` | Noise standard deviation. |
| `models.<model_name>.iterative_decoder_mode` | `string` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Memory optimization mode for iterative encoder-decoder prediction. |
| `models.<model_name>.iterative_stateful` | `boolean` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['lstm']` | Whether iterative prediction propagates LSTM state. |
| `models.<model_name>.learning_rate` | `float` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['lstm']` | Optimizer learning rate. |
| `models.<model_name>.learning_rate` | `float` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Optimizer learning rate. |
| `models.<model_name>.loss` | `string` | No | Not explicitly defined in the parser code. | `loss_schema and model HPO schemas` | Loss function name. |
| `models.<model_name>.loss_params.delta` | `int or float` | No | Not explicitly defined in the parser code. | `loss_schema` | Huber-like loss delta parameter. |
| `models.<model_name>.max_grad_norm` | `float` | No | Not explicitly defined in the parser code. | `advanced_training_schema` | Gradient clipping norm. Also appears in LSTM and Transformer HPO schemas. |
| `models.<model_name>.max_lags` | `integer` | Yes | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['var']` | Maximum lag order. |
| `models.<model_name>.maxiter` | `integer` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['arima']` | Maximum number of optimizer iterations. |
| `models.<model_name>.maxiter` | `integer` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['sarima']` | Maximum number of optimizer iterations. |
| `models.<model_name>.maxiter` | `integer` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['var']` | Maximum number of optimizer iterations. |
| `models.<model_name>.method` | `string` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['arima']` | Optimizer or estimation method. |
| `models.<model_name>.method` | `string` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['sarima']` | Optimizer or estimation method. |
| `models.<model_name>.method` | `string` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['var']` | Optimizer or estimation method. |
| `models.<model_name>.min_epochs` | `integer` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['lstm']` | Minimum number of epochs before early stopping can trigger. |
| `models.<model_name>.min_epochs` | `integer` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Minimum number of epochs before early stopping can trigger. |
| `models.<model_name>.nan_guard_enabled` | `boolean` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Enable NaN guard checks. |
| `models.<model_name>.norm_first` | `boolean` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Whether normalization is applied before sublayers. |
| `models.<model_name>.num_decoder_layers` | `integer` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Number of decoder layers. |
| `models.<model_name>.num_encoder_layers` | `integer` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Number of encoder layers. |
| `models.<model_name>.num_heads` | `integer` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Number of attention heads. |
| `models.<model_name>.num_layers` | `integer` | Yes | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['lstm']` | Number of recurrent layers. |
| `models.<model_name>.num_workers` | `integer` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['lstm']` | DataLoader worker count. |
| `models.<model_name>.num_workers` | `integer` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | DataLoader worker count. |
| `models.<model_name>.optimization` | `mapping` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS[*].optimization` | Hyperparameter optimization configuration. |
| `models.<model_name>.optimization.method` | `string` | Yes if optimization is present | Not explicitly defined in the parser code. | `MODEL_SCHEMAS[*].optimization` | HPO method. |
| `models.<model_name>.optimization.n_trials` | `integer` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS[*].optimization` | Number of HPO trials. |
| `models.<model_name>.optimization.params` | `mapping` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS[*].optimization` | Search space for optimized parameters. |
| `models.<model_name>.optimization.params.<param>` | `list or range mapping` | No | Not explicitly defined in the parser code. | `*_opt_param_schema` | Generic search-space value. Exact accepted form depends on the specific optimized parameter. |
| `models.<model_name>.optimization.params.<param>.log` | `boolean` | No | Not explicitly defined in the parser code. | `float_range` | Whether to sample the range on a log scale. |
| `models.<model_name>.optimization.params.<param>.max` | `integer or float` | Yes for range mapping | Not explicitly defined in the parser code. | `integer_range / float_range` | Upper bound for a search range. |
| `models.<model_name>.optimization.params.<param>.min` | `integer or float` | Yes for range mapping | Not explicitly defined in the parser code. | `integer_range / float_range` | Lower bound for a search range. |
| `models.<model_name>.optimization.params.<param>.step` | `integer or float` | No | Not explicitly defined in the parser code. | `integer_range / float_range` | Optional discretization step. |
| `models.<model_name>.optimization.params.scheduler_config.<scheduler_param>` | `list or range mapping` | No | Not explicitly defined in the parser code. | `*_opt_param_schema` | Generic search-space value. Exact accepted form depends on the specific optimized parameter. |
| `models.<model_name>.optimization.params.scheduler_config.<scheduler_param>.log` | `boolean` | No | Not explicitly defined in the parser code. | `float_range` | Whether to sample the range on a log scale. |
| `models.<model_name>.optimization.params.scheduler_config.<scheduler_param>.max` | `integer or float` | Yes for range mapping | Not explicitly defined in the parser code. | `integer_range / float_range` | Upper bound for a search range. |
| `models.<model_name>.optimization.params.scheduler_config.<scheduler_param>.min` | `integer or float` | Yes for range mapping | Not explicitly defined in the parser code. | `integer_range / float_range` | Lower bound for a search range. |
| `models.<model_name>.optimization.params.scheduler_config.<scheduler_param>.step` | `integer or float` | No | Not explicitly defined in the parser code. | `integer_range / float_range` | Optional discretization step. |
| `models.<model_name>.optimization.pruner_config` | `mapping` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS[*].optimization` | Optuna pruner configuration. |
| `models.<model_name>.optimization.pruner_config.interval_steps` | `integer` | No | Not explicitly defined in the parser code. | `pruner_config_schema` | Pruning check interval. |
| `models.<model_name>.optimization.pruner_config.lower` | `float` | No | Not explicitly defined in the parser code. | `pruner_config_schema` | ThresholdPruner lower threshold. |
| `models.<model_name>.optimization.pruner_config.max_resource` | `integer` | No | Not explicitly defined in the parser code. | `pruner_config_schema` | Hyperband maximum resource. |
| `models.<model_name>.optimization.pruner_config.min_resource` | `integer` | No | Not explicitly defined in the parser code. | `pruner_config_schema` | Hyperband minimum resource. |
| `models.<model_name>.optimization.pruner_config.n_min_trials` | `integer` | No | Not explicitly defined in the parser code. | `pruner_config_schema` | Minimum completed trials before pruning. |
| `models.<model_name>.optimization.pruner_config.n_startup_trials` | `integer` | No | Not explicitly defined in the parser code. | `pruner_config_schema` | Median or percentile startup trials. |
| `models.<model_name>.optimization.pruner_config.n_warmup_steps` | `integer` | No | Not explicitly defined in the parser code. | `pruner_config_schema` | Warmup steps before pruning. |
| `models.<model_name>.optimization.pruner_config.patience` | `integer` | No | Not explicitly defined in the parser code. | `pruner_config_schema` | PatientPruner patience. |
| `models.<model_name>.optimization.pruner_config.percentile` | `float` | No | Not explicitly defined in the parser code. | `pruner_config_schema` | Percentile cutoff for PercentilePruner. |
| `models.<model_name>.optimization.pruner_config.reduction_factor` | `integer` | No | Not explicitly defined in the parser code. | `pruner_config_schema` | Hyperband reduction factor. |
| `models.<model_name>.optimization.pruner_config.type` | `string` | No | Not explicitly defined in the parser code. | `pruner_config_schema` | Pruner type. Parser does not require it when pruner_config is present. |
| `models.<model_name>.optimization.pruner_config.upper` | `float` | No | Not explicitly defined in the parser code. | `pruner_config_schema` | ThresholdPruner upper threshold. |
| `models.<model_name>.optimization.pruner_config.wrapped_pruner` | `mapping` | No | Not explicitly defined in the parser code. | `pruner_config_schema` | PatientPruner wrapped pruner. Parser does not recursively validate its internal structure. |
| `models.<model_name>.optimize` | `boolean` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS[*].optimization` | Enable or mark model as eligible for hyperparameter optimization. |
| `models.<model_name>.optimizer` | `string` | No | Not explicitly defined in the parser code. | `optimizer_schema` | Optimizer name for neural models. |
| `models.<model_name>.optimizer_config.eps` | `float` | No | Not explicitly defined in the parser code. | `optimizer_schema` | Optimizer epsilon configuration. |
| `models.<model_name>.output_head_strategy` | `string` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Output head sharing strategy. |
| `models.<model_name>.p` | `integer` | Yes | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['arima']` | Autoregressive order. |
| `models.<model_name>.p` | `integer` | Yes | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['sarima']` | Non-seasonal autoregressive order. |
| `models.<model_name>.past_covariate_policy` | `string` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['lstm']` | Policy for past covariates in iterative mode. |
| `models.<model_name>.positional_encoding_config.max_len` | `integer` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Maximum positional encoding length. |
| `models.<model_name>.positional_encoding_config.pe_dropout` | `float` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Dropout used in positional encoding. |
| `models.<model_name>.positional_encoding_config.scale_with_sqrt_hidden_size` | `boolean` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Scale positional encoding with sqrt(hidden_size). |
| `models.<model_name>.positional_encoding_config.type` | `string` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Positional encoding type. |
| `models.<model_name>.prediction_noise.enabled` | `boolean` | Yes if prediction_noise is present | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Enable decoder prediction noise injection. |
| `models.<model_name>.prediction_noise.schedule` | `string` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Prediction noise schedule. |
| `models.<model_name>.prediction_noise.std` | `float` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Noise standard deviation. |
| `models.<model_name>.preprocessing` | `mapping` | No | Not explicitly defined in the parser code. | `_define_preprocessing_groups_schema` | Grouped preprocessing configuration used at model level and experiment-model override level. |
| `models.<model_name>.preprocessing.preprocessing_groups` | `list[mapping]` | Yes if preprocessing is present | Not explicitly defined in the parser code. | `_define_preprocessing_groups_schema` | Ordered list of named preprocessing groups. |
| `models.<model_name>.preprocessing.preprocessing_groups[].apply_to` | `string or list[string]` | Yes | Not explicitly defined in the parser code. | `_define_preprocessing_groups_schema` | Target selector for the group. |
| `models.<model_name>.preprocessing.preprocessing_groups[].name` | `string` | Yes | Not explicitly defined in the parser code. | `_define_preprocessing_groups_schema` | Name of the preprocessing group. |
| `models.<model_name>.preprocessing.preprocessing_groups[].pipeline` | `mapping` | Yes | Not explicitly defined in the parser code. | `_define_preprocessing_groups_schema` | Transformation pipeline applied by the group. |
| `models.<model_name>.preprocessing.preprocessing_groups[].pipeline.log_transform` | `mapping` | No | Not explicitly defined in the parser code. | `_define_pipeline_schema` | Optional log transformation step. |
| `models.<model_name>.preprocessing.preprocessing_groups[].pipeline.log_transform.enabled` | `boolean` | Yes if log_transform is present | Not explicitly defined in the parser code. | `_define_log_transform_schema` | Enables or disables the log transformation step. |
| `models.<model_name>.preprocessing.preprocessing_groups[].pipeline.log_transform.epsilon` | `float` | No | Not explicitly defined in the parser code. | `_define_log_transform_schema` | Small positive offset for numerical safety. The parser validates but does not inject it. |
| `models.<model_name>.preprocessing.preprocessing_groups[].pipeline.log_transform.method` | `string` | No | Not explicitly defined in the parser code. | `_define_log_transform_schema` | Logarithm variant. The docstring mentions `log1p` as a default, but the parser does not inject a default. |
| `models.<model_name>.preprocessing.preprocessing_groups[].pipeline.scaling` | `mapping` | No | Not explicitly defined in the parser code. | `_define_pipeline_schema` | Optional scaling step. |
| `models.<model_name>.preprocessing.preprocessing_groups[].pipeline.scaling.enabled` | `boolean` | Yes if scaling is present | Not explicitly defined in the parser code. | `_define_scaling_schema` | Enables or disables scaling. |
| `models.<model_name>.preprocessing.preprocessing_groups[].pipeline.scaling.method` | `string` | No | Not explicitly defined in the parser code. | `_define_scaling_schema` | Scaling method. The docstring mentions `minmax` as a default, but the parser does not inject a default. |
| `models.<model_name>.preprocessing.preprocessing_groups[].pipeline.scaling.range` | `list[number]` | No | Not explicitly defined in the parser code. | `_define_scaling_schema` | Target range for min-max scaling. Parser does not require it when method is `minmax`, although the docstring says it is required. |
| `models.<model_name>.preprocessing.preprocessing_groups[].pipeline.winsorize` | `mapping` | No | Not explicitly defined in the parser code. | `_define_pipeline_schema` | Optional winsorization step for clipping extremes. |
| `models.<model_name>.preprocessing.preprocessing_groups[].pipeline.winsorize.enabled` | `boolean` | Yes if winsorize is present | Not explicitly defined in the parser code. | `_define_winsorize_schema` | Enables or disables winsorization. |
| `models.<model_name>.preprocessing.preprocessing_groups[].pipeline.winsorize.limits` | `list[float]` | No | Not explicitly defined in the parser code. | `_define_winsorize_schema` | Lower and upper percentile limits for clipping. |
| `models.<model_name>.preprocessing.time_features` | `list[string]` | No | Not explicitly defined in the parser code. | `_define_preprocessing_groups_schema` | Optional time-feature list inside grouped preprocessing. This is separate from dataset-level `time_features`. |
| `models.<model_name>.q` | `integer` | Yes | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['arima']` | Moving-average order. |
| `models.<model_name>.q` | `integer` | Yes | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['sarima']` | Non-seasonal moving-average order. |
| `models.<model_name>.readout` | `string` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Encoder readout strategy. |
| `models.<model_name>.remove_data_after_fit` | `boolean` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['arima']` | Whether fitted model data should be removed after fitting. |
| `models.<model_name>.remove_data_after_fit` | `boolean` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['sarima']` | Whether fitted model data should be removed after fitting. |
| `models.<model_name>.remove_data_after_fit` | `boolean` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['var']` | Whether fitted model data should be removed after fitting. |
| `models.<model_name>.revin_affine` | `boolean` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Use affine parameters in RevIN. |
| `models.<model_name>.revin_eps` | `float` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | RevIN epsilon. |
| `models.<model_name>.revin_robust` | `boolean` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Use robust RevIN statistics. |
| `models.<model_name>.save_horizon_csv` | `boolean` | No | Not explicitly defined in the parser code. | `advanced_training_schema` | Save horizon-level CSV outputs. |
| `models.<model_name>.save_scheduler_csv` | `boolean` | No | Not explicitly defined in the parser code. | `advanced_training_schema` | Save learning-rate scheduler CSV. |
| `models.<model_name>.save_scheduler_plot` | `boolean` | No | Not explicitly defined in the parser code. | `advanced_training_schema` | Save learning-rate scheduler plot. |
| `models.<model_name>.scheduler_config.T_max` | `integer` | No | Not explicitly defined in the parser code. | `_define_scheduler_config_schema` | Cosine annealing period. |
| `models.<model_name>.scheduler_config.anneal_strategy` | `string` | No | Not explicitly defined in the parser code. | `_define_scheduler_config_schema` | Annealing strategy. |
| `models.<model_name>.scheduler_config.cooldown` | `integer` | No | Not explicitly defined in the parser code. | `_define_scheduler_config_schema` | Plateau cooldown period. |
| `models.<model_name>.scheduler_config.div_factor` | `float` | No | Not explicitly defined in the parser code. | `_define_scheduler_config_schema` | Initial learning rate division factor. |
| `models.<model_name>.scheduler_config.eps` | `float` | No | Not explicitly defined in the parser code. | `_define_scheduler_config_schema` | Minimal learning-rate change epsilon. |
| `models.<model_name>.scheduler_config.eta_min` | `float` | No | Not explicitly defined in the parser code. | `_define_scheduler_config_schema` | Minimum learning rate for cosine annealing. |
| `models.<model_name>.scheduler_config.factor` | `float` | No | Not explicitly defined in the parser code. | `_define_scheduler_config_schema` | Plateau reduction factor. |
| `models.<model_name>.scheduler_config.final_div_factor` | `float` | No | Not explicitly defined in the parser code. | `_define_scheduler_config_schema` | Final learning rate division factor. |
| `models.<model_name>.scheduler_config.gamma` | `float` | No | Not explicitly defined in the parser code. | `_define_scheduler_config_schema` | Multiplicative decay factor for step or exponential schedulers. |
| `models.<model_name>.scheduler_config.max_lr` | `float` | No | Not explicitly defined in the parser code. | `_define_scheduler_config_schema` | OneCycleLR maximum learning rate. |
| `models.<model_name>.scheduler_config.min_lr` | `float` | No | Not explicitly defined in the parser code. | `_define_scheduler_config_schema` | Minimum learning rate. |
| `models.<model_name>.scheduler_config.mode` | `string` | No | Not explicitly defined in the parser code. | `_define_scheduler_config_schema` | Plateau scheduler mode. |
| `models.<model_name>.scheduler_config.patience` | `integer` | No | Not explicitly defined in the parser code. | `_define_scheduler_config_schema` | Plateau patience. |
| `models.<model_name>.scheduler_config.pct_start` | `float` | No | Not explicitly defined in the parser code. | `_define_scheduler_config_schema` | Fraction of cycle spent increasing learning rate. |
| `models.<model_name>.scheduler_config.step_size` | `integer` | No | Not explicitly defined in the parser code. | `_define_scheduler_config_schema` | StepLR step size. |
| `models.<model_name>.scheduler_config.threshold` | `float` | No | Not explicitly defined in the parser code. | `_define_scheduler_config_schema` | Plateau improvement threshold. |
| `models.<model_name>.scheduler_config.threshold_mode` | `string` | No | Not explicitly defined in the parser code. | `_define_scheduler_config_schema` | Plateau threshold mode. |
| `models.<model_name>.scheduler_config.type` | `string` | No | Not explicitly defined in the parser code. | `_define_scheduler_config_schema` | Scheduler type. Parser does not require it even when scheduler_config is present. |
| `models.<model_name>.seasonal_period` | `integer` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['sarima']` | Seasonal period. Although semantically central for SARIMA, the parser marks it optional. |
| `models.<model_name>.seasonal_period` | `integer` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['simple_seasonal']` | Seasonal period used by the simple seasonal baseline. |
| `models.<model_name>.seasonal_period` | `integer` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Seasonal period used by seasonal target initialization. |
| `models.<model_name>.strategy` | `string` | Yes | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['lstm']` | Forecasting strategy. |
| `models.<model_name>.strategy` | `string` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Forecasting strategy. |
| `models.<model_name>.tgt_init` | `string` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Decoder target initialization strategy. |
| `models.<model_name>.trend` | `string` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['arima']` | Trend term passed to the statistical model. |
| `models.<model_name>.trend` | `string` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['sarima']` | Trend term passed to the statistical model. |
| `models.<model_name>.trend` | `string` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['var']` | Trend term. |
| `models.<model_name>.type` | `string` | Yes | Not explicitly defined in the parser code. | `validate_each_model_against_its_schema` | Model implementation type used to select the schema. |
| `models.<model_name>.use_amp` | `boolean` | No | Not explicitly defined in the parser code. | `advanced_training_schema` | Enable automatic mixed precision for training. |
| `models.<model_name>.use_amp_inference` | `boolean` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Enable AMP during inference. |
| `models.<model_name>.use_exogenous` | `boolean` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['arima']` | Whether to use exogenous variables. Parser accepts it, examples note ARIMA is effectively univariate in this framework. |
| `models.<model_name>.use_exogenous` | `boolean` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['lstm']` | Whether to use exogenous variables. |
| `models.<model_name>.use_exogenous` | `boolean` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['sarima']` | Whether to use exogenous variables. |
| `models.<model_name>.use_exogenous` | `boolean` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Whether to use exogenous variables. |
| `models.<model_name>.use_exogenous` | `boolean` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['var']` | Whether to use exogenous covariates. |
| `models.<model_name>.use_revin` | `boolean` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Enable Reversible Instance Normalization. |
| `models.<model_name>.weight_decay` | `float` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['lstm']` | Weight decay regularization. |
| `models.<model_name>.weight_decay` | `float` | No | Not explicitly defined in the parser code. | `MODEL_SCHEMAS['transformer']` | Weight decay regularization. |
| `paths` | `mapping` | No | Not explicitly defined in the parser code. | `validate_config.main_schema` | Optional paths section for runtime output templates. |
| `paths.model_save_path_template` | `string` | No | Not explicitly defined in the parser code. | `validate_config.main_schema.paths` | Template used by the application for saved model paths. Examples use placeholders such as `{model_name}` and `{dataset_name}`. |
