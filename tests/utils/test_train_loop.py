"""
Unit tests for the training loop utility (production-ready version).

This module provides a comprehensive suite of tests for the `run_train_loop`
function located in `utils/train_loop.py`. The tests cover standard
functionality, edge cases, error handling, and Automatic Mixed Precision (AMP)
support to ensure the training process is robust and reliable.

Dependencies like the PyTorch model, optimizer, and loss function are mocked
to isolate the training loop's logic for fast, deterministic testing.
"""

import logging
from unittest.mock import MagicMock, call

import pytest
import torch
import torch.nn as nn

from utils.train_loop import run_train_loop

pytestmark = pytest.mark.unit

# --- Constants for Test Configuration ---
NUM_TRAIN_SAMPLES = 100
NUM_VAL_SAMPLES = 20
SEQ_LEN = 10
NUM_FEATURES = 1
BATCH_SIZE = 32
NUM_TEST_EPOCHS_FULL = 3
NUM_TEST_EPOCHS_MAX = 10
NUM_TEST_EPOCHS_MIN = 1
EARLY_STOP_PATIENCE = 5
EARLY_STOP_PATIENCE_SHORT = 2
DUMMY_LOSS_VALUE_LOW = 0.1
DUMMY_LOSS_VALUE_MED = 0.2
DUMMY_LOSS_VALUE_HIGH = 0.3
# Ceiling division to calculate the number of batches per epoch
NUM_BATCHES_PER_EPOCH = (NUM_TRAIN_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE


# --- Fixtures and Mocks Setup ---

@pytest.fixture
def mock_model():
    """Provides a mock `torch.nn.Module` for testing."""
    model = MagicMock(spec=nn.Module)
    # Ensure that calling .to(device) on the mock returns the mock itself
    model.to.return_value = model
    # Return actual tensors from state_dict and parameters for torch.equal() checks
    model.state_dict.return_value = {'weight': torch.randn(1)}
    # parameters() must return fresh iterator each time (called multiple times in train_loop)
    # Use a lambda to create a new iterator on each call
    param_tensor = torch.randn(1, requires_grad=True)
    model.parameters = MagicMock(side_effect=lambda: iter([param_tensor]))
    model.train = MagicMock()
    model.eval = MagicMock()
    # The model must return a tensor to be compatible with the loss function
    model.return_value = torch.randn(BATCH_SIZE, NUM_FEATURES)
    return model


@pytest.fixture
def sample_tensors():
    """Provides a dictionary of sample tensors for training and validation."""
    return {
        "encoder_inputs_train": torch.randn(NUM_TRAIN_SAMPLES, SEQ_LEN, NUM_FEATURES),
        "decoder_inputs_train": None,
        "true_outputs_train": torch.randn(NUM_TRAIN_SAMPLES, NUM_FEATURES, NUM_FEATURES),
        "encoder_inputs_val": torch.randn(NUM_VAL_SAMPLES, SEQ_LEN, NUM_FEATURES),
        "decoder_inputs_val": None,
        "true_outputs_val": torch.randn(NUM_VAL_SAMPLES, NUM_FEATURES, NUM_FEATURES)
    }


@pytest.fixture
def mock_optimizer():
    """Provides a mock optimizer with `zero_grad` and `step` methods."""
    optimizer = MagicMock(spec=torch.optim.Optimizer)
    optimizer.zero_grad = MagicMock()
    optimizer.step = MagicMock()
    # GradScaler.unscale_() requires optimizer.param_groups
    optimizer.param_groups = [{'params': [torch.randn(1, requires_grad=True)]}]
    return optimizer


@pytest.fixture
def mock_loss_fn():
    """Provides a mock loss function returning a differentiable dummy loss tensor."""
    loss_fn = MagicMock(spec=nn.Module)
    loss_tensor = torch.tensor(DUMMY_LOSS_VALUE_LOW, requires_grad=True)
    loss_fn.return_value = loss_tensor
    return loss_fn

# Fixture merged from test_train_loop.py
@pytest.fixture
def mock_dependencies(mocker):
    """Mocks the logging utility functions to prevent actual logging during tests."""
    mocker.patch('utils.train_loop.log_training_start')
    mocker.patch('utils.train_loop.log_training_success')
    return mocker


# --- Success Case Tests ---

def test_run_train_loop_completes_full_run(
    mock_model, sample_tensors, mock_optimizer, mock_loss_fn, mock_dependencies
):
    """
    Scenario: A standard, successful training run without AMP.
    Verifies: The loop executes for the specified number of epochs, toggles model
    modes (`train`/`eval`), uses the optimizer, and returns a trained model.
    """
    # mock_dependencies added to suppress logging
    trained_model, history = run_train_loop(
        model=mock_model,
        **sample_tensors,
        loss_fn=mock_loss_fn,
        optimizer=mock_optimizer,
        epochs=NUM_TEST_EPOCHS_FULL,
        early_stopping_patience=EARLY_STOP_PATIENCE,
        device=torch.device('cpu'),
        model_name="test_model"
    )

    assert mock_model.train.call_count == NUM_TEST_EPOCHS_FULL
    assert mock_model.eval.call_count == NUM_TEST_EPOCHS_FULL
    assert mock_optimizer.step.call_count == NUM_BATCHES_PER_EPOCH * NUM_TEST_EPOCHS_FULL
    assert hasattr(trained_model, 'best_val_loss')


def test_early_stopping_triggers_correctly(
    mock_model, sample_tensors, mock_optimizer, mock_dependencies
):
    """
    Scenario: The early stopping mechanism terminates training.
    Verifies: Training stops after the patience window is exceeded when validation
    loss does not improve.
    """
    # mock_dependencies added to suppress logging
    mock_loss_fn = MagicMock(spec=nn.Module)
    # Simulate increasing validation loss: 0.1, 0.2, 0.3
    # Epoch 1: val_loss=0.1 (new best)
    # Epoch 2: val_loss=0.2 (no improvement, patience=1)
    # Epoch 3: val_loss=0.3 (no improvement, patience=2 -> stop)
    loss_sequence = [
        DUMMY_LOSS_VALUE_LOW, DUMMY_LOSS_VALUE_MED, DUMMY_LOSS_VALUE_HIGH
    ]
    # Create a side_effect list for all calls to loss_fn (train batches + 1 val)
    side_effects = []
    for val_loss in loss_sequence:
        side_effects.extend([torch.tensor(val_loss, requires_grad=True)] * NUM_BATCHES_PER_EPOCH)
        side_effects.append(torch.tensor(val_loss))
    mock_loss_fn.side_effect = side_effects

    run_train_loop(
        model=mock_model,
        **sample_tensors,
        loss_fn=mock_loss_fn,
        optimizer=mock_optimizer,
        epochs=NUM_TEST_EPOCHS_MAX,
        early_stopping_patience=EARLY_STOP_PATIENCE_SHORT,
        min_epochs=NUM_TEST_EPOCHS_MIN,
        device=torch.device('cpu'),
        model_name="early_stop_test"
    )

    # Expected: 1 epoch for initial best loss + 2 epochs for patience
    assert mock_model.train.call_count == 1 + EARLY_STOP_PATIENCE_SHORT


def test_no_validation_data_run(
    mock_model, sample_tensors, mock_optimizer, mock_loss_fn, mock_dependencies
):
    """
    Scenario: The training loop is run without validation data.
    Verifies: The loop executes for the full number of epochs, and the model's
    `eval()` method is never called.
    """
    # mock_dependencies added to suppress logging
    run_train_loop(
        model=mock_model,
        encoder_inputs_train=sample_tensors["encoder_inputs_train"],
        decoder_inputs_train=None,
        true_outputs_train=sample_tensors["true_outputs_train"],
        encoder_inputs_val=None,
        decoder_inputs_val=None,
        true_outputs_val=None,
        loss_fn=mock_loss_fn,
        optimizer=mock_optimizer,
        epochs=NUM_TEST_EPOCHS_FULL,
        early_stopping_patience=EARLY_STOP_PATIENCE,
        device=torch.device('cpu')
    )

    assert mock_model.train.call_count == NUM_TEST_EPOCHS_FULL
    mock_model.eval.assert_not_called()


# --- AMP-Specific Tests ---

def test_run_train_loop_with_amp_enabled_cuda_fp16(
    mocker, mock_model, sample_tensors, mock_optimizer, mock_loss_fn, mock_dependencies
):
    """
    Scenario: Training with AMP on a CUDA device that supports fp16 but not bfloat16.
    Verifies: GradScaler is instantiated and its methods (`scale`, `step`, `update`)
    are called.
    """
    # mock_dependencies added to suppress logging
    # 1. Create a REAL device object
    device = torch.device('cuda')

    # 2. Mock the environment checks
    mocker.patch('torch.cuda.is_available', return_value=True)
    mocker.patch('torch.cuda.is_bf16_supported', return_value=False)

    # 3. Mock GradScaler to verify it's used on CUDA
    # Create a properly configured mock scaler instance
    mock_scaler = MagicMock()
    scaled_loss_mock = MagicMock()
    scaled_loss_mock.backward.return_value = None
    mock_scaler.scale.return_value = scaled_loss_mock
    mock_scaler.step.return_value = None
    mock_scaler.update.return_value = None
    mock_scaler.unscale_.return_value = None
    # Patch GradScaler where it's imported in train_loop.py
    mock_scaler_class = mocker.patch('utils.train_loop.GradScaler', return_value=mock_scaler)

    run_train_loop(
        model=mock_model, **sample_tensors, loss_fn=mock_loss_fn, optimizer=mock_optimizer,
        epochs=NUM_TEST_EPOCHS_MIN, early_stopping_patience=EARLY_STOP_PATIENCE,
        device=device, model_name="amp_cuda_fp16_test", use_amp=True
    )

    # 4. Assertions: GradScaler should be instantiated and used on CUDA
    mock_scaler_class.assert_called_once()
    mock_scaler.scale.assert_called()
    mock_scaler.step.assert_called_with(mock_optimizer)
    mock_scaler.update.assert_called()


def test_run_train_loop_with_amp_enabled_cpu(
    mocker, mock_model, sample_tensors, mock_optimizer, mock_loss_fn, mock_dependencies
):
    """
    Scenario: Training with AMP on a CPU.
    Verifies: GradScaler is NOT used (CPU doesn't need gradient scaling).
    """
    # mock_dependencies added to suppress logging
    device = torch.device('cpu')
    # Only mock GradScaler to verify it's not used on CPU
    mock_scaler_class = mocker.patch('utils.train_loop.GradScaler')

    run_train_loop(
        model=mock_model, **sample_tensors, loss_fn=mock_loss_fn, optimizer=mock_optimizer,
        epochs=NUM_TEST_EPOCHS_MIN, early_stopping_patience=EARLY_STOP_PATIENCE,
        device=device, model_name="amp_cpu_test", use_amp=True
    )

    # Main assertion: GradScaler should NOT be instantiated on CPU
    mock_scaler_class.assert_not_called()


# --- Input Validation and Error Handling Tests ---

@pytest.mark.parametrize("invalid_args, error_msg", [
    ({"encoder_inputs_train": torch.Tensor()}, "encoder_inputs_train must be a non-empty torch.Tensor"),
    ({"batch_size": 0}, "batch_size must be a positive integer"),
    # ... Add more validation cases as needed
])
def test_run_train_loop_raises_value_error(
    mock_model, sample_tensors, mock_optimizer, mock_loss_fn, invalid_args, error_msg
):
    """
    Scenario: Parameterized test for various invalid inputs.
    Verifies: The function raises a `ValueError` with a descriptive message.
    """
    args = {
        "model": mock_model, "loss_fn": mock_loss_fn, "optimizer": mock_optimizer,
        "epochs": NUM_TEST_EPOCHS_MIN, "early_stopping_patience": EARLY_STOP_PATIENCE,
        "device": torch.device('cpu'), **sample_tensors
    }
    args.update(invalid_args)

    with pytest.raises(ValueError, match=error_msg):
        run_train_loop(**args)


# MERGED TEST: This now uses parameterization from test_train_loop2.py
# and adds the caplog assertion from test_train_loop.py.
@pytest.mark.parametrize("use_amp", [True, False], ids=["AMP_enabled", "AMP_disabled"])
def test_handles_non_finite_validation_loss(
    mock_model, sample_tensors, mock_optimizer, caplog, use_amp
):
    """
    Scenario: The loop encounters a non-finite (inf) validation loss.
    Verifies: The loop logs a warning, continues execution, and completes all epochs
    without crashing, both with and without AMP enabled.
    """
    mock_loss_fn = MagicMock(spec=nn.Module)
    # Per epoch: normal train loss, then 'inf' for validation
    side_effect_per_epoch = [torch.tensor(DUMMY_LOSS_VALUE_LOW, requires_grad=True)] * NUM_BATCHES_PER_EPOCH + [torch.tensor(float('inf'))]
    mock_loss_fn.side_effect = side_effect_per_epoch * NUM_TEST_EPOCHS_FULL

    with caplog.at_level(logging.WARNING):
        run_train_loop(
            model=mock_model, **sample_tensors, loss_fn=mock_loss_fn, optimizer=mock_optimizer,
            epochs=NUM_TEST_EPOCHS_FULL, early_stopping_patience=EARLY_STOP_PATIENCE,
            device=torch.device('cpu'), use_amp=use_amp
        )

    # Assertion added from test_train_loop.py
    assert "Non-finite validation loss" in caplog.text
    # Assertion from test_train_loop2.py
    assert mock_model.train.call_count == NUM_TEST_EPOCHS_FULL


def test_handles_non_finite_training_loss(mock_model, sample_tensors, mock_optimizer, caplog):
    """
    Scenario: The loop encounters a non-finite (NaN) training loss in a batch.
    Verifies: The loop logs a warning, skips the problematic batch (i.e., does
    not call optimizer.step), and correctly continues the rest of the epoch.
    """
    mock_loss_fn = MagicMock(spec=nn.Module)
    # First batch returns 'nan', the rest are normal
    bad_batch_loss = [torch.tensor(float('nan'))]
    rest_of_epoch_losses = [torch.tensor(DUMMY_LOSS_VALUE_LOW, requires_grad=True)] * (NUM_BATCHES_PER_EPOCH - 1)
    validation_loss = [torch.tensor(DUMMY_LOSS_VALUE_LOW)]
    mock_loss_fn.side_effect = bad_batch_loss + rest_of_epoch_losses + validation_loss

    with caplog.at_level(logging.WARNING):
        run_train_loop(
            model=mock_model, **sample_tensors, loss_fn=mock_loss_fn, optimizer=mock_optimizer,
            epochs=NUM_TEST_EPOCHS_MIN, early_stopping_patience=EARLY_STOP_PATIENCE,
            device=torch.device('cpu'), use_amp=False
        )

    assert "Non-finite loss detected" in caplog.text
    # Optimizer step should be called for all batches EXCEPT the first one
    assert mock_optimizer.step.call_count == (NUM_BATCHES_PER_EPOCH - 1)
    # The loop should complete the single epoch
    assert mock_model.train.call_count == NUM_TEST_EPOCHS_MIN


def test_run_train_loop_logs_grad_norm_when_clipping_enabled(
        mock_model, sample_tensors, mock_optimizer, mock_loss_fn, mock_dependencies, caplog, mocker
):
    """
    Scenario: Training with max_grad_norm > 0.
    Verifies: The loop correctly calculates and logs 'Avg Grad Norm'.
    """
    # Mock clip_grad_norm_ to return a fixed value (e.g. 0.5)
    # Note: clip_grad_norm_ returns the norm (Tensor), we mock it to return a tensor equal to 0.5
    mocker.patch('torch.nn.utils.clip_grad_norm_', return_value=torch.tensor(0.5))

    with caplog.at_level(logging.INFO):
        run_train_loop(
            model=mock_model,
            **sample_tensors,
            loss_fn=mock_loss_fn,
            optimizer=mock_optimizer,
            epochs=1,
            early_stopping_patience=5,
            device=torch.device('cpu'),
            max_grad_norm=1.0,  # Enable clipping
            log_every=1
        )

    # We expect to see "Avg Grad Norm" in the output
    assert "Grad: 0.5000" in caplog.text


def test_run_train_loop_skips_grad_norm_log_when_clipping_disabled(
        mock_model, sample_tensors, mock_optimizer, mock_loss_fn, mock_dependencies, caplog, mocker
):
    """
    Scenario: Training with max_grad_norm = 0.0 (disabled).
    Verifies: The loop DOES NOT log 'Avg Grad Norm'.
    """
    # Mock clip just in case, though it shouldn't be called
    mock_clip = mocker.patch('torch.nn.utils.clip_grad_norm_')

    with caplog.at_level(logging.INFO):
        run_train_loop(
            model=mock_model,
            **sample_tensors,
            loss_fn=mock_loss_fn,
            optimizer=mock_optimizer,
            epochs=1,
            early_stopping_patience=5,
            device=torch.device('cpu'),
            max_grad_norm=0.0,  # Disable clipping
            log_every=1
        )

    # Clip should not be called
    mock_clip.assert_not_called()
    # Log should contain loss but NOT Grad Norm
    assert "Train Loss:" in caplog.text
    assert "Avg Grad Norm" not in caplog.text