# tests/test_context.py
"""
Comprehensive tests for RunContext.

Tests cover:
- Directory creation and initialization
- Artifact path generation with various edge cases
- Metadata immutability
- Metadata serialization
- PathJSONEncoder
- Error handling
"""

import json
from pathlib import Path

import numpy as np
import pytest

from core.context import RunContext, PathJSONEncoder


# --------
# Helpers
# --------

def _make_base_context(tmp_path: Path, run_id: str = "run_001") -> RunContext:
    """Helper to create a minimal base context for tests."""
    return RunContext.from_base_path(
        base_path=tmp_path / run_id,
        run_id=run_id,
        experiment_name="test_experiment",
    )


# -------------------------
# from_base_path + mkdir
# -------------------------

def test_from_base_path_initializes_directories(tmp_path):
    """Test that from_base_path sets up standard directory structure."""
    ctx = _make_base_context(tmp_path)

    assert ctx.base_dir == tmp_path / "run_001"
    assert ctx.gradients_dir == ctx.base_dir / "gradients"
    assert ctx.attention_dir == ctx.base_dir / "attention"
    assert ctx.plots_dir == ctx.base_dir / "plots"
    assert ctx.data_dir == ctx.base_dir / "data"
    assert ctx.checkpoints_dir == ctx.base_dir / "checkpoints"


@pytest.mark.parametrize("with_checkpoints", [True, False])
def test_create_directories_creates_all_known_dirs(tmp_path, with_checkpoints):
    """Test that create_directories creates all configured directories."""
    base_dir = tmp_path / "run_001"
    checkpoints_dir = base_dir / "checkpoints" if with_checkpoints else None

    ctx = RunContext(
        run_id="run_001",
        experiment_name="test_experiment",
        base_dir=base_dir,
        gradients_dir=base_dir / "gradients",
        attention_dir=base_dir / "attention",
        plots_dir=base_dir / "plots",
        data_dir=base_dir / "data",
        checkpoints_dir=checkpoints_dir,
    )

    ctx.create_directories()

    expected_dirs = [
        ctx.gradients_dir,
        ctx.attention_dir,
        ctx.plots_dir,
        ctx.data_dir,
    ]

    # checkpoints_dir is optional - create only if defined
    if with_checkpoints:
        expected_dirs.append(ctx.checkpoints_dir)

    for d in expected_dirs:
        assert d.is_dir(), f"Expected directory to exist: {d}"


def test_create_directories_raises_on_permission_error(tmp_path):
    """Test that OSError is propagated when directory creation fails."""
    ctx = _make_base_context(tmp_path)

    # Make base_dir read-only to trigger permission error
    ctx.base_dir.mkdir(parents=True)
    ctx.base_dir.chmod(0o444)  # Read-only

    try:
        with pytest.raises(OSError):
            ctx.create_directories()
    finally:
        # Cleanup - restore permissions
        ctx.base_dir.chmod(0o755)


# -------------------------
# get_artifact_path
# -------------------------

@pytest.mark.parametrize(
    (
        "category,"
        "checkpoints_defined,"
        "include_fold,"
        "include_window,"
        "fold_idx,"
        "window_size,"
        "suffix,"
        "extension,"
        "expected_filename,"
        "use_fallback"
    ),
    [
        # 1) Known category: gradients, with fold + window
        (
            "gradients",
            True,          # checkpoints_dir exists (irrelevant for gradients)
            True,          # include_fold
            True,          # include_window
            1,             # fold_idx
            96,            # window_size
            "grads",       # suffix
            "npz",         # extension
            "transformer_fold_1_w96_grads.npz",
            False,         # no fallback
        ),
        # 2) Known category: data, no fold, with window
        (
            "data",
            True,
            False,         # include_fold=False
            True,          # include_window=True
            0,             # fold_idx (ignored)
            96,
            "metadata",
            "json",
            "transformer_w96_metadata.json",
            False,
        ),
        # 3) Unknown category: embeddings (fallback to base_dir/embeddings)
        (
            "embeddings",
            True,
            True,
            True,
            2,
            None,          # window_size=None -> excluded from name
            "vectors",
            "npz",
            "transformer_fold_2_vectors.npz",
            True,          # uses fallback
        ),
        # 4) Category checkpoints, but checkpoints_dir=None -> fallback to base_dir/checkpoints
        (
            "checkpoints",
            False,         # checkpoints_dir=None
            True,
            False,         # include_window=False
            3,
            128,           # window_size (ignored)
            "best",
            "pt",
            "transformer_fold_3_best.pt",
            True,          # uses fallback because checkpoints_dir is None
        ),
        # 5) No suffix
        (
            "plots",
            True,
            True,
            False,
            5,
            256,
            "",            # suffix=""
            "png",
            "transformer_fold_5.png",
            False,
        ),
    ],
)
def test_get_artifact_path_parametrized(
    tmp_path,
    caplog,
    category,
    checkpoints_defined,
    include_fold,
    include_window,
    fold_idx,
    window_size,
    suffix,
    extension,
    expected_filename,
    use_fallback,
):
    """Test get_artifact_path with various configuration combinations."""
    base_dir = tmp_path / "run_ctx"
    checkpoints_dir = base_dir / "checkpoints" if checkpoints_defined else None

    ctx = RunContext(
        run_id="run_ctx",
        experiment_name="experiment",
        base_dir=base_dir,
        gradients_dir=base_dir / "gradients",
        attention_dir=base_dir / "attention",
        plots_dir=base_dir / "plots",
        data_dir=base_dir / "data",
        checkpoints_dir=checkpoints_dir,
        model_name="transformer",
        fold_idx=fold_idx,
        window_size=window_size,
    )

    with caplog.at_level("DEBUG"):
        path = ctx.get_artifact_path(
            category=category,
            suffix=suffix,
            extension=extension,
            include_fold=include_fold,
            include_window=include_window,
        )

    # Check filename convention
    assert path.name == expected_filename

    # Check expected parent directory
    if not use_fallback:
        # Known categories defined in context
        if category == "gradients":
            expected_parent = ctx.gradients_dir
        elif category == "attention":
            expected_parent = ctx.attention_dir
        elif category == "plots":
            expected_parent = ctx.plots_dir
        elif category == "data":
            expected_parent = ctx.data_dir
        elif category == "checkpoints":
            # Only valid here if checkpoints_defined=True
            assert checkpoints_defined, "Non-fallback checkpoints_dir must be defined"
            expected_parent = ctx.checkpoints_dir
        else:
            pytest.fail(f"Unexpected known category without fallback: {category}")
    else:
        # Fallback -> base_dir/category
        expected_parent = base_dir / category

    assert path.parent == expected_parent

    # Check DEBUG log presence for fallback
    msg = f"Using fallback directory for category '{category}'"
    if use_fallback:
        assert msg in caplog.text
    else:
        assert msg not in caplog.text


def test_get_artifact_path_defaults_to_model_when_name_is_none(tmp_path):
    """Test that filename uses 'model' when model_name is None."""
    ctx = _make_base_context(tmp_path).with_metadata(
        model_name=None,  # Explicitly None
        fold_idx=0
    )

    path = ctx.get_artifact_path(
        category="gradients",
        suffix="test",
        extension="json"
    )

    # Should default to "model" when model_name is None
    assert path.name == "model_fold_0_test.json"


# -------------------------
# with_metadata
# -------------------------

@pytest.mark.parametrize(
    "initial_kwargs, update_kwargs, expected_model_name, expected_fold_idx",
    [
        (
            {},  # No initial metadata
            {"model_name": "transformer", "fold_idx": 1},
            "transformer",
            1,
        ),
        (
            {"model_name": "base_model", "fold_idx": 0},
            {"fold_idx": 2},  # Override fold_idx
            "base_model",     # Original model_name preserved
            2,
        ),
    ],
)
def test_with_metadata_returns_new_instance_and_keeps_original_intact(
    tmp_path,
    initial_kwargs,
    update_kwargs,
    expected_model_name,
    expected_fold_idx,
):
    """Test that with_metadata creates new instance and preserves original."""
    ctx = _make_base_context(tmp_path).with_metadata(**initial_kwargs)

    new_ctx = ctx.with_metadata(**update_kwargs)

    # Original context remains unchanged
    for k, v in initial_kwargs.items():
        assert getattr(ctx, k) == v

    # New context has updated fields
    assert new_ctx.model_name == expected_model_name
    assert new_ctx.fold_idx == expected_fold_idx

    # Base_dir should be identical but object identity different
    assert new_ctx.base_dir == ctx.base_dir
    assert new_ctx is not ctx


# -------------------------
# save_metadata
# -------------------------

def test_save_metadata_creates_json_with_serializable_fields(tmp_path):
    """Test that save_metadata creates valid JSON with all fields."""
    ctx = _make_base_context(tmp_path).with_metadata(
        model_name="transformer",
        model_type="Transformer",
        fold_idx=0,
        window_size=96,
        metadata={"note": "test_run"},
    )

    ctx.create_directories()
    ctx.save_metadata()

    expected_path = ctx.get_artifact_path(
        category="data",
        suffix="metadata",
        extension="json",
    )
    assert expected_path.is_file()

    with open(expected_path, "r") as f:
        data = json.load(f)

    # Consistency checks
    assert data["run_id"] == "run_001"
    assert data["experiment_name"] == "test_experiment"
    assert data["model_name"] == "transformer"
    assert data["model_type"] == "Transformer"
    assert data["fold_idx"] == 0
    assert data["window_size"] == 96
    assert data["metadata"]["note"] == "test_run"

    # Paths must be serialized to strings (PathJSONEncoder)
    assert isinstance(data["base_dir"], str)
    assert isinstance(data["gradients_dir"], str)
    assert isinstance(data["data_dir"], str)


def test_save_metadata_creates_parent_for_fallback_category(tmp_path):
    """
    Test that save_metadata creates parent dir for fallback categories.

    This tests the critical safety mechanism: parent.mkdir() in save_metadata()
    ensures fallback category directories are created even if create_directories()
    wasn't called or didn't know about that category.
    """
    ctx = _make_base_context(tmp_path).with_metadata(
        model_name="test",
        fold_idx=0,
        window_size=96
    )

    # DON'T call create_directories() - data/ won't exist
    # But save_metadata() should create it via parent.mkdir()

    ctx.save_metadata()

    metadata_path = ctx.get_artifact_path("data", "metadata", "json")
    assert metadata_path.exists()
    assert metadata_path.parent.exists()  # data/ was created by save_metadata()


# -------------------------
# PathJSONEncoder
# -------------------------

@pytest.mark.parametrize(
    "obj, expected",
    [
        (Path("/tmp/test.txt"), "/tmp/test.txt"),
        (np.int32(5), 5),
        (np.int64(42), 42),
        (np.float32(3.14), 3.14),  # ← Regular float
        (np.array([1, 2, 3]), [1, 2, 3]),
        (np.array([[1, 2], [3, 4]]), [[1, 2], [3, 4]]),
    ],
)
def test_path_json_encoder_handles_supported_types(obj, expected):
    payload = {"value": obj}
    encoded = json.dumps(payload, cls=PathJSONEncoder)
    decoded = json.loads(encoded)

    if isinstance(expected, list):
        assert decoded["value"] == expected
    elif isinstance(decoded["value"], float):  # ← Fixed!
        assert decoded["value"] == pytest.approx(expected, rel=1e-5)
    else:
        assert decoded["value"] == expected

def test_path_json_encoder_handles_nested_structures(tmp_path):
    """Test PathJSONEncoder with nested dicts and lists containing ML types."""
    complex_data = {
        "paths": {
            "gradients": tmp_path / "gradients",
            "data": tmp_path / "data",
        },
        "arrays": {
            "losses": np.array([0.5, 0.3, 0.1]),
            "metrics": {
                "train": np.float32(0.95),
                "val": np.float32(0.92),
            }
        },
        "scalars": [np.int64(1), np.int64(2), np.int64(3)],
    }

    encoded = json.dumps(complex_data, cls=PathJSONEncoder)
    decoded = json.loads(encoded)

    # Verify paths serialized to strings
    assert isinstance(decoded["paths"]["gradients"], str)
    assert isinstance(decoded["paths"]["data"], str)

    # Verify arrays converted to lists
    assert decoded["arrays"]["losses"] == [0.5, 0.3, 0.1]

    # Verify numpy scalars converted to native types
    assert isinstance(decoded["arrays"]["metrics"]["train"], float)
    assert decoded["scalars"] == [1, 2, 3]


# -------------------------
# run_name property
# -------------------------

def test_run_name_property_generates_correct_identifier(tmp_path):
    """Test that run_name property formats correctly."""
    ctx = RunContext.from_base_path(
        base_path=tmp_path / "test",
        run_id="run123",
        experiment_name="vanishing_gradient"
    )

    assert ctx.run_name == "vanishing_gradient_run123"


# -------------------------
# Edge cases
# -------------------------

def test_context_with_all_none_metadata_fields(tmp_path):
    """Test context behavior when all metadata fields are None."""
    ctx = RunContext.from_base_path(
        base_path=tmp_path / "test",
        run_id="test",
        experiment_name="test"
    )
    # All metadata fields should be None by default
    assert ctx.model_name is None
    assert ctx.model_type is None
    assert ctx.fold_idx is None
    assert ctx.window_size is None

    # get_artifact_path should still work with defaults
    path = ctx.get_artifact_path("gradients", "test", "json")
    assert "model_test.json" in path.name  # Uses default "model"


def test_multiple_contexts_dont_interfere(tmp_path):
    """Test that multiple contexts can coexist without interference."""
    ctx1 = RunContext.from_base_path(
        tmp_path / "run1",
        run_id="run1",
        experiment_name="exp1"
    ).with_metadata(model_name="transformer", fold_idx=0)

    ctx2 = RunContext.from_base_path(
        tmp_path / "run2",
        run_id="run2",
        experiment_name="exp2"
    ).with_metadata(model_name="lstm", fold_idx=1)

    # Contexts should be independent
    assert ctx1.model_name == "transformer"
    assert ctx2.model_name == "lstm"
    assert ctx1.fold_idx == 0
    assert ctx2.fold_idx == 1
    assert ctx1.base_dir != ctx2.base_dir

def test_with_metadata_deep_copies_nested_structures(base_context):
    """Verify nested structures in metadata are deep copied."""
    base_context.metadata["hyperparams"] = {"lr": 0.001, "layers": [128, 64]}

    ctx2 = base_context.with_metadata(fold_idx=0)

    # Modify nested structures in ctx2
    ctx2.metadata["hyperparams"]["lr"] = 0.01
    ctx2.metadata["hyperparams"]["layers"].append(32)

    # base_context should NOT be affected
    assert base_context.metadata["hyperparams"]["lr"] == 0.001
    assert base_context.metadata["hyperparams"]["layers"] == [128, 64]

    # ctx2 should have modifications
    assert ctx2.metadata["hyperparams"]["lr"] == 0.01
    assert ctx2.metadata["hyperparams"]["layers"] == [128, 64, 32]
