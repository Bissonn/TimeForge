# tests/conftest.py
"""
Shared pytest fixtures for RunContext tests.

These fixtures reduce boilerplate in tests and provide commonly used contexts.
"""

import pytest
from pathlib import Path
from core.context import RunContext


@pytest.fixture
def base_context(tmp_path):
    """
    Create a basic RunContext for testing.

    Provides a minimal context with standard directory structure.
    Use this fixture when you need a clean context without metadata.

    Example:
        def test_something(base_context):
            path = base_context.get_artifact_path(...)
            assert path.exists()
    """
    return RunContext.from_base_path(
        base_path=tmp_path / "test_run",
        run_id="test_run",
        experiment_name="test_experiment"
    )


@pytest.fixture
def base_context_with_dirs(base_context):
    """
    Create a base context with physical directories already created.

    Use this when your test needs directories to exist upfront.

    Example:
        def test_saving(base_context_with_dirs):
            # Directories already exist
            base_context_with_dirs.save_metadata()
    """
    base_context.create_directories()
    return base_context


@pytest.fixture
def fold_context(base_context):
    """
    Create a fold-specific context with typical metadata.

    Provides a context configured for fold 0 with transformer model.
    Use this for tests that need a realistic fold configuration.

    Example:
        def test_artifact_naming(fold_context):
            path = fold_context.get_artifact_path("gradients", "gradients", "json")
            assert "transformer_fold_0_w96" in path.name
    """
    return base_context.with_metadata(
        model_name="transformer",
        model_type="Transformer",
        fold_idx=0,
        window_size=96
    )


@pytest.fixture
def fold_context_with_dirs(fold_context):
    """
    Create a fold-specific context with physical directories.

    Combines fold_context with directory creation.
    Use this for integration tests that save actual files.

    Example:
        def test_full_flow(fold_context_with_dirs):
            fold_context_with_dirs.save_metadata()
            # File is actually saved
    """
    fold_context.create_directories()
    return fold_context


@pytest.fixture
def multi_fold_contexts(base_context):
    """
    Create multiple fold contexts for testing multi-fold scenarios.

    Returns a list of 3 fold contexts (fold 0, 1, 2).
    Use this for tests that need to verify behavior across multiple folds.

    Example:
        def test_no_overwriting(multi_fold_contexts):
            for ctx in multi_fold_contexts:
                ctx.save_metadata()
            # Verify 3 separate metadata files exist
    """
    return [
        base_context.with_metadata(
            model_name="transformer",
            fold_idx=fold_idx,
            window_size=96
        )
        for fold_idx in range(3)
    ]


# Optional: Add markers for different test categories
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests (no filesystem I/O)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (with filesystem I/O)"
    )
    config.addinivalue_line(
        "markers", "edge_case: Edge case tests"
    )
