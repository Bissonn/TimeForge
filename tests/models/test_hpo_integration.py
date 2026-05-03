import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, ANY
from models.base import NeuralTSForecaster


# We use the factory to get a valid model instance easily
def test_fit_and_evaluate_fold_logic(model_factory):
    """
    Verify that _fit_and_evaluate_fold correctly delegates to fit() and predict()
    and calculates metrics.
    """
    # 1. Setup
    forecaster = model_factory(model_type="lstm", strategy="direct")

    # Mock internal methods to isolate the fold logic
    # We don't want to actually train a network here, just check the flow
    forecaster.fit = MagicMock(return_value=(0.5, {"loss": [0.5]}))  # Returns (loss, history)
    forecaster.predict = MagicMock(return_value=pd.DataFrame({"target_0": [1, 1, 1]}, index=[0, 1, 2]))
    forecaster.evaluate = MagicMock(return_value=0.25)

    # Mock Data
    train_fold = pd.DataFrame({"target_0": np.random.rand(50)})
    eval_fold = pd.DataFrame({"target_0": np.random.rand(5)})  # Length matches forecast_steps=5

    # 2. Execution
    loss, preds, history = forecaster._fit_and_evaluate_fold(
        train_fold=train_fold,
        eval_fold=eval_fold,
        validation_params={"early_stopping_validation_percentage": 20},
        dataset=MagicMock(target_columns=["target_0"]),
        is_final_fit=False
    )

    # 3. Verification
    # Check if fit was called with correct ES percentage
    forecaster.fit.assert_called_once()
    call_args = forecaster.fit.call_args[1]
    assert call_args["is_final_fit"] is False
    assert call_args["early_stopping_validation_percentage"] == 20

    # Check if predict was called
    forecaster.predict.assert_called_once()

    # Check if evaluate was called
    forecaster.evaluate.assert_called_once()

    # Check returns
    assert loss == 0.25
    assert isinstance(preds, pd.DataFrame)
    assert history == {"loss": [0.5]}


def test_fit_and_evaluate_fold_final_fit(model_factory):
    """
    Verify that is_final_fit=True disables validation in fit().
    """
    forecaster = model_factory(model_type="lstm")
    forecaster.fit = MagicMock(return_value=(0.1, {}))
    forecaster.predict = MagicMock(return_value=pd.DataFrame())
    forecaster.evaluate = MagicMock(return_value=0.0)

    forecaster._fit_and_evaluate_fold(
        train_fold=pd.DataFrame(),
        eval_fold=pd.DataFrame(),
        validation_params={},
        dataset=MagicMock(),
        is_final_fit=True
    )

    # Check arguments passed to fit
    call_args = forecaster.fit.call_args[1]
    assert call_args["is_final_fit"] is True


def test_optimize_hyperparameters_grid_flow(model_factory):
    """
    Verify the Grid Search loop flow in optimize_hyperparameters.
    """
    forecaster = model_factory(model_type="lstm")

    # Mock dependencies
    forecaster._fit_and_evaluate_fold = MagicMock(return_value=(0.5, None, {}))  # returns (loss, preds, history)

    dataset = MagicMock()
    folds = [(MagicMock(), MagicMock())]  # 1 fold

    config = {
        "optimization": {
            "method": "grid",
            "params": {"hidden_size": [10, 20]}
        },
        "hidden_size": 5  # Default
    }

    # Execution
    # We need to mock ModelFactory inside the method scope or use a real one.
    # Since base.py imports ModelFactory inside the method, we patch it.
    with patch("models.factory.ModelFactory") as mock_factory_class:
        # The factory creates candidates. We return a mock candidate.
        mock_candidate = MagicMock()
        mock_candidate._fit_and_evaluate_fold.return_value = (0.5, None, {})
        mock_factory_class.create.return_value = mock_candidate

        best_params, best_loss = forecaster.optimize_hyperparameters(
            dataset=dataset,
            model_config=config,
            validation_params={"n_folds": 1},
            folds=folds
        )

        # Verification
        # Factory should be called twice (for hidden_size=10 and 20)
        assert mock_factory_class.create.call_count == 2

        # Check if params were updated in calls
        calls = mock_factory_class.create.call_args_list
        params1 = calls[0][1]["model_params"]
        params2 = calls[1][1]["model_params"]

        assert params1["hidden_size"] == 10
        assert params2["hidden_size"] == 20


def test_optimize_hyperparameters_optuna_flow(model_factory):
    """
    Verify the Optuna integration: study creation, trial suggestion, reporting.
    """
    forecaster = model_factory(model_type="lstm")

    dataset = MagicMock()
    dataset.training_length = 1000
    dataset.target_columns = ["target"]
    dataset.past_covariates = []
    dataset.future_covariates = []

    folds = [(MagicMock(), MagicMock())]

    config = {
        "optimization": {
            "method": "optuna",
            "n_trials": 2,
            "params": {"dropout": {"min": 0.1, "max": 0.5}}
        }
    }

    # Mock Optuna
    with patch("models.base.optuna") as mock_optuna:
        class MockTrialPruned(BaseException): pass
        mock_optuna.TrialPruned = MockTrialPruned

        # Setup Study mock
        mock_study = MagicMock()
        mock_optuna.create_study.return_value = mock_study

        # Setup Trial mock
        mock_trial = MagicMock()
        mock_trial.should_prune.return_value = False
        mock_study.ask.return_value = mock_trial
        # Mock suggestions
        mock_trial.suggest_float.side_effect = [0.2, 0.4]

        # Mock Factory
        with patch("models.factory.ModelFactory") as mock_factory_class:
            mock_candidate = MagicMock()
            # Return different losses to check best selection
            mock_candidate._fit_and_evaluate_fold.side_effect = [
                (0.5, None, {}),  # Trial 1
                (0.3, None, {})  # Trial 2 (Better)
            ]
            mock_factory_class.create.return_value = mock_candidate

            best_params, best_loss = forecaster.optimize_hyperparameters(
                dataset=dataset,
                model_config=config,
                validation_params={"n_folds": 1},
                folds=folds
            )

            # Verification
            # 1. Study created
            mock_optuna.create_study.assert_called_once()

            # 2. Ask called n_trials times
            assert mock_study.ask.call_count == 2

            # 3. Tell called with correct values
            # trial 1 -> 0.5
            # trial 2 -> 0.3
            mock_study.tell.assert_any_call(mock_trial, 0.5)
            mock_study.tell.assert_any_call(mock_trial, 0.3)

            # 4. Best result returned
            assert best_loss == 0.3
            assert best_params["dropout"] == 0.4  # The second suggestion


def test_optuna_pruning_handling(model_factory):
    """
    Verify that TrialPruned exception is handled correctly (study notified).
    """
    forecaster = model_factory(model_type="lstm")
    folds = [(MagicMock(), MagicMock())]
    config = {"optimization": {"method": "optuna", "n_trials": 1, "params": {"a": [1]}}}

    dataset = MagicMock()
    dataset.training_length = 1000
    dataset.target_columns = ["target"]
    dataset.past_covariates = []
    dataset.future_covariates = []

    with patch("models.base.optuna") as mock_optuna:
        mock_study = MagicMock()
        class MockTrialPruned(BaseException): pass
        mock_optuna.TrialPruned = MockTrialPruned
        mock_optuna.create_study.return_value = mock_study
        mock_trial = MagicMock()
        mock_study.ask.return_value = mock_trial

        # Mock exception during evaluation
        with patch("models.factory.ModelFactory") as mock_factory:
            mock_candidate = MagicMock()
            # Raise Pruned exception
            mock_candidate._fit_and_evaluate_fold.side_effect = MockTrialPruned()
            mock_factory.create.return_value = mock_candidate

            # The optimization will fail because the only trial is pruned.
            # We expect ValueError, but we verify side effects (study.tell) before that.
            with pytest.raises(ValueError, match="No valid parameter combinations"):
                forecaster.optimize_hyperparameters(
                    dataset = dataset, model_config = config,
                    validation_params = {"n_folds": 1}, folds = folds
                )

            # Verify study.tell was called with PRUNED state
            mock_study.tell.assert_called_with(
                mock_trial,
                state=mock_optuna.trial.TrialState.PRUNED
            )