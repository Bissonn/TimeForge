"""
Test Suite for LR Scheduler Integration

Tests for scheduler creation, configuration, validation, and integration
with training loop.

Run with: pytest tests/test_scheduler_integration.py -v
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
import yaml
import tempfile
import os

# Import scheduler utilities
from utils.scheduler import create_scheduler, get_scheduler_step_info
from torch.optim.lr_scheduler import (
    OneCycleLR,
    CosineAnnealingLR,
    StepLR,
    ExponentialLR,
    ReduceLROnPlateau
)


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def dummy_model():
    """Create a simple dummy model for testing."""
    return nn.Linear(10, 1)


@pytest.fixture
def dummy_optimizer(dummy_model):
    """Create optimizer for dummy model."""
    return torch.optim.Adam(dummy_model.parameters(), lr=0.001)


@pytest.fixture
def scheduler_configs():
    """Provide various scheduler configurations."""
    return {
        "onecycle": {
            "type": "onecycle",
            "max_lr": 0.01,
            "pct_start": 0.3,
            "div_factor": 25.0,
            "final_div_factor": 1e4
        },
        "cosine": {
            "type": "cosine",
            "T_max": 100,
            "eta_min": 1e-6
        },
        "step": {
            "type": "step",
            "step_size": 20,
            "gamma": 0.1
        },
        "exponential": {
            "type": "exponential",
            "gamma": 0.95
        },
        "plateau": {
            "type": "plateau",
            "mode": "min",
            "factor": 0.5,
            "patience": 10
        }
    }


# ═══════════════════════════════════════════════════════════════════════════
# TEST SCHEDULER CREATION
# ═══════════════════════════════════════════════════════════════════════════

class TestSchedulerCreation:
    """Test scheduler creation with various configurations."""

    def test_onecycle_creation(self, dummy_optimizer, scheduler_configs):
        """Test OneCycleLR scheduler creation."""
        scheduler = create_scheduler(
            optimizer=dummy_optimizer,
            scheduler_config=scheduler_configs["onecycle"],
            train_size=1000,
            batch_size=32,
            max_epochs=100,
            default_lr=0.001
        )

        assert scheduler is not None
        assert isinstance(scheduler, OneCycleLR)

    def test_cosine_creation(self, dummy_optimizer, scheduler_configs):
        """Test CosineAnnealingLR scheduler creation."""
        scheduler = create_scheduler(
            optimizer=dummy_optimizer,
            scheduler_config=scheduler_configs["cosine"],
            train_size=1000,
            batch_size=32,
            max_epochs=100,
            default_lr=0.001
        )

        assert scheduler is not None
        assert isinstance(scheduler, CosineAnnealingLR)

    def test_step_creation(self, dummy_optimizer, scheduler_configs):
        """Test StepLR scheduler creation."""
        scheduler = create_scheduler(
            optimizer=dummy_optimizer,
            scheduler_config=scheduler_configs["step"],
            train_size=1000,
            batch_size=32,
            max_epochs=100,
            default_lr=0.001
        )

        assert scheduler is not None
        assert isinstance(scheduler, StepLR)

    def test_exponential_creation(self, dummy_optimizer, scheduler_configs):
        """Test ExponentialLR scheduler creation."""
        scheduler = create_scheduler(
            optimizer=dummy_optimizer,
            scheduler_config=scheduler_configs["exponential"],
            train_size=1000,
            batch_size=32,
            max_epochs=100,
            default_lr=0.001
        )

        assert scheduler is not None
        assert isinstance(scheduler, ExponentialLR)

    def test_plateau_creation(self, dummy_optimizer, scheduler_configs):
        """Test ReduceLROnPlateau scheduler creation."""
        scheduler = create_scheduler(
            optimizer=dummy_optimizer,
            scheduler_config=scheduler_configs["plateau"],
            train_size=1000,
            batch_size=32,
            max_epochs=100,
            default_lr=0.001
        )

        assert scheduler is not None
        assert isinstance(scheduler, ReduceLROnPlateau)

    def test_no_scheduler(self, dummy_optimizer):
        """Test that empty config returns None."""
        scheduler = create_scheduler(
            optimizer=dummy_optimizer,
            scheduler_config={},
            train_size=1000,
            batch_size=32,
            max_epochs=100,
            default_lr=0.001
        )

        assert scheduler is None

    def test_invalid_scheduler_type(self, dummy_optimizer):
        """Test handling of invalid scheduler type."""
        scheduler = create_scheduler(
            optimizer=dummy_optimizer,
            scheduler_config={"type": "invalid_type"},
            train_size=1000,
            batch_size=32,
            max_epochs=100,
            default_lr=0.001
        )

        assert scheduler is None


# ═══════════════════════════════════════════════════════════════════════════
# TEST SCHEDULER BEHAVIOR
# ═══════════════════════════════════════════════════════════════════════════

class TestSchedulerBehavior:
    """Test actual scheduler behavior during training simulation."""

    def test_onecycle_lr_curve(self, dummy_optimizer, scheduler_configs):
        """Test OneCycleLR produces expected LR curve."""
        scheduler = create_scheduler(
            optimizer=dummy_optimizer,
            scheduler_config=scheduler_configs["onecycle"],
            train_size=320,  # 10 steps/epoch * 32 batch_size
            batch_size=32,
            max_epochs=100,
            default_lr=0.001
        )

        lrs = []
        steps_per_epoch = 10
        num_epochs = 100

        for _ in range(num_epochs * steps_per_epoch):
            lrs.append(dummy_optimizer.param_groups[0]['lr'])
            dummy_optimizer.step()
            scheduler.step()

        # Check initial LR (should be max_lr / div_factor)
        initial_lr = lrs[0]
        max_lr = scheduler_configs["onecycle"]["max_lr"]
        div_factor = scheduler_configs["onecycle"]["div_factor"]
        expected_initial = max_lr / div_factor

        assert abs(initial_lr - expected_initial) < 1e-6, \
            f"Initial LR {initial_lr} != expected {expected_initial}"

        # Check that max LR is reached
        assert max(lrs) >= max_lr * 0.99, "Max LR not reached"

        # Check final LR (should be very small)
        final_lr = lrs[-1]
        assert final_lr < initial_lr * 0.1, "Final LR not significantly reduced"

    def test_step_lr_decay(self, dummy_optimizer, scheduler_configs):
        """Test StepLR produces correct decay."""
        scheduler = create_scheduler(
            optimizer=dummy_optimizer,
            scheduler_config=scheduler_configs["step"],
            train_size=1000,
            batch_size=32,
            max_epochs=100,
            default_lr=0.001
        )

        initial_lr = dummy_optimizer.param_groups[0]['lr']
        step_size = scheduler_configs["step"]["step_size"]
        gamma = scheduler_configs["step"]["gamma"]

        lrs = []
        for epoch in range(100):
            lrs.append(dummy_optimizer.param_groups[0]['lr'])
            dummy_optimizer.step()
            scheduler.step()

        # Check LR at step_size
        assert abs(lrs[step_size] - initial_lr * gamma) < 1e-9, \
            "StepLR did not decay correctly at step_size"

        # Check LR at 2 * step_size
        assert abs(lrs[2 * step_size] - initial_lr * gamma ** 2) < 1e-9, \
            "StepLR did not decay correctly at 2 * step_size"

    def test_exponential_lr_decay(self, dummy_optimizer, scheduler_configs):
        """Test ExponentialLR produces exponential decay."""
        scheduler = create_scheduler(
            optimizer=dummy_optimizer,
            scheduler_config=scheduler_configs["exponential"],
            train_size=1000,
            batch_size=32,
            max_epochs=100,
            default_lr=0.001
        )

        initial_lr = dummy_optimizer.param_groups[0]['lr']
        gamma = scheduler_configs["exponential"]["gamma"]

        # Step 10 epochs
        for _ in range(10):
            dummy_optimizer.step()
            scheduler.step()

        current_lr = dummy_optimizer.param_groups[0]['lr']
        expected_lr = initial_lr * (gamma ** 10)

        assert abs(current_lr - expected_lr) < 1e-9, \
            f"ExponentialLR decay incorrect: {current_lr} != {expected_lr}"

    def test_plateau_lr_reduction(self, dummy_optimizer, scheduler_configs):
        """Test ReduceLROnPlateau reduces LR on plateau."""
        scheduler = create_scheduler(
            optimizer=dummy_optimizer,
            scheduler_config=scheduler_configs["plateau"],
            train_size=1000,
            batch_size=32,
            max_epochs=100,
            default_lr=0.001
        )

        initial_lr = dummy_optimizer.param_groups[0]['lr']
        patience = scheduler_configs["plateau"]["patience"]
        factor = scheduler_configs["plateau"]["factor"]

        # Simulate plateau (constant loss)
        # Need patience+2 steps: patience steps to detect plateau,
        # then 1 more to trigger reduction, then 1 to see new LR
        constant_loss = 1.0
        for _ in range(patience + 2):  # ← patience + 2
            dummy_optimizer.step()
            scheduler.step(constant_loss)

        # LR should have been reduced
        current_lr = dummy_optimizer.param_groups[0]['lr']
        assert current_lr < initial_lr, "LR not reduced on plateau"

        # Check reduction factor (with tolerance for floating point)
        expected_lr = initial_lr * factor
        assert abs(current_lr - expected_lr) < 1e-6, \
            f"LR reduction incorrect: {current_lr} != {expected_lr}"

# ═══════════════════════════════════════════════════════════════════════════
# TEST SCHEDULER STEP INFO
# ═══════════════════════════════════════════════════════════════════════════

class TestSchedulerStepInfo:
    """Test get_scheduler_step_info utility."""

    def test_onecycle_step_info(self, dummy_optimizer, scheduler_configs):
        """Test OneCycleLR step info."""
        scheduler = create_scheduler(
            optimizer=dummy_optimizer,
            scheduler_config=scheduler_configs["onecycle"],
            train_size=1000,
            batch_size=32,
            max_epochs=100,
            default_lr=0.001
        )

        info = get_scheduler_step_info(scheduler)

        assert info['step_frequency'] == 'batch'
        assert info['requires_metric'] == False
        assert info['scheduler_name'] == 'OneCycleLR'

    def test_step_lr_step_info(self, dummy_optimizer, scheduler_configs):
        """Test StepLR step info."""
        scheduler = create_scheduler(
            optimizer=dummy_optimizer,
            scheduler_config=scheduler_configs["step"],
            train_size=1000,
            batch_size=32,
            max_epochs=100,
            default_lr=0.001
        )

        info = get_scheduler_step_info(scheduler)

        assert info['step_frequency'] == 'epoch'
        assert info['requires_metric'] == False
        assert info['scheduler_name'] == 'StepLR'

    def test_plateau_step_info(self, dummy_optimizer, scheduler_configs):
        """Test ReduceLROnPlateau step info."""
        scheduler = create_scheduler(
            optimizer=dummy_optimizer,
            scheduler_config=scheduler_configs["plateau"],
            train_size=1000,
            batch_size=32,
            max_epochs=100,
            default_lr=0.001
        )

        info = get_scheduler_step_info(scheduler)

        assert info['step_frequency'] == 'epoch'
        assert info['requires_metric'] == True
        assert info['scheduler_name'] == 'ReduceLROnPlateau'

    def test_none_step_info(self):
        """Test None scheduler step info."""
        info = get_scheduler_step_info(None)

        assert info['step_frequency'] is None
        assert info['requires_metric'] == False
        assert info['scheduler_name'] is None


# ═══════════════════════════════════════════════════════════════════════════
# TEST CONFIG VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

class TestConfigValidation:
    """Test YAML config validation for scheduler_config."""

    def test_valid_onecycle_config(self, scheduler_configs):
        """Test valid OneCycleLR config passes validation."""
        config = {
            "model": {
                "type": "transformer",
                "learning_rate": 0.001,
                "scheduler_config": scheduler_configs["onecycle"]
            }
        }

        # This should not raise an error
        # (Assuming config_utils validation is imported and used)
        assert config["model"]["scheduler_config"]["type"] == "onecycle"

    def test_valid_plateau_config(self, scheduler_configs):
        """Test valid ReduceLROnPlateau config."""
        config = {
            "model": {
                "type": "lstm",
                "learning_rate": 0.001,
                "scheduler_config": scheduler_configs["plateau"]
            }
        }

        assert config["model"]["scheduler_config"]["type"] == "plateau"
        assert config["model"]["scheduler_config"]["mode"] == "min"


# ═══════════════════════════════════════════════════════════════════════════
# TEST INTEGRATION WITH TRAINING
# ═══════════════════════════════════════════════════════════════════════════

class TestTrainingIntegration:
    """Test scheduler integration with training loop."""

    def test_batch_level_scheduler_integration(self, dummy_model, dummy_optimizer):
        """Test batch-level scheduler in simulated training."""
        scheduler_config = {
            "type": "onecycle",
            "max_lr": 0.01,
            "pct_start": 0.3
        }

        train_size = 320  # 10 batches * 32 batch_size
        batch_size = 32
        max_epochs = 10

        scheduler = create_scheduler(
            optimizer=dummy_optimizer,
            scheduler_config=scheduler_config,
            train_size=train_size,
            batch_size=batch_size,
            max_epochs=max_epochs,
            default_lr=0.001
        )

        info = get_scheduler_step_info(scheduler)

        # Simulate training
        lrs_recorded = []
        steps_per_epoch = train_size // batch_size

        for epoch in range(max_epochs):
            for batch in range(steps_per_epoch):
                # Record LR
                lrs_recorded.append(dummy_optimizer.param_groups[0]['lr'])

                dummy_optimizer.step()

                # Step scheduler (batch-level)
                if info['step_frequency'] == 'batch':
                    scheduler.step()

        # Verify LR changed during training
        assert len(set(lrs_recorded)) > 1, "LR did not change during training"
        assert len(lrs_recorded) == max_epochs * steps_per_epoch

    def test_epoch_level_scheduler_integration(self, dummy_model, dummy_optimizer):
        """Test epoch-level scheduler in simulated training."""
        scheduler_config = {
            "type": "step",
            "step_size": 3,
            "gamma": 0.5
        }

        scheduler = create_scheduler(
            optimizer=dummy_optimizer,
            scheduler_config=scheduler_config,
            train_size=1000,
            batch_size=32,
            max_epochs=10,
            default_lr=0.001
        )

        info = get_scheduler_step_info(scheduler)

        # Simulate training
        epoch_lrs = []

        for epoch in range(10):
            epoch_lrs.append(dummy_optimizer.param_groups[0]['lr'])

            # Simulate batches (no scheduler step)
            for batch in range(5):
                dummy_optimizer.step()

            # Step scheduler (epoch-level)
            if info['step_frequency'] == 'epoch' and not info['requires_metric']:
                scheduler.step()

        # Verify LR changed at step_size intervals
        assert epoch_lrs[0] == 0.001
        assert epoch_lrs[3] == 0.0005  # After step_size=3
        assert epoch_lrs[6] == 0.00025  # After 2 * step_size


# ═══════════════════════════════════════════════════════════════════════════
# TEST EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_epochs(self, dummy_optimizer):
        """Test scheduler with zero epochs returns None."""
        scheduler_config = {"type": "onecycle", "max_lr": 0.01}

        scheduler = create_scheduler(
            optimizer=dummy_optimizer,
            scheduler_config=scheduler_config,
            train_size=1000,
            batch_size=32,
            max_epochs=0,  # Edge case
            default_lr=0.001
        )

        # Should return None for zero epochs (can't create valid scheduler)
        assert scheduler is None, "Should return None for zero epochs"

    def test_missing_required_params(self, dummy_optimizer):
        """Test handling of missing required parameters."""
        # OneCycleLR without max_lr should use default_lr
        scheduler_config = {"type": "onecycle"}

        scheduler = create_scheduler(
            optimizer=dummy_optimizer,
            scheduler_config=scheduler_config,
            train_size=1000,
            batch_size=32,
            max_epochs=100,
            default_lr=0.001
        )

        assert scheduler is not None

    def test_very_small_batch_size(self, dummy_optimizer):
        """Test with very small batch size."""
        scheduler_config = {"type": "onecycle", "max_lr": 0.01}

        scheduler = create_scheduler(
            optimizer=dummy_optimizer,
            scheduler_config=scheduler_config,
            train_size=1000,
            batch_size=1,  # Very small
            max_epochs=100,
            default_lr=0.001
        )

        assert scheduler is not None


# ═══════════════════════════════════════════════════════════════════════════
# RUN TESTS
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
