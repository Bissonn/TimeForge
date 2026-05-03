import json
from dataclasses import asdict

import pytest

from utils.splitting import (
    BacktestConfig,
    DevTestSplitConfig,
    DevelopmentCVConfig,
    SplittingProfile,
    FoldSpec,
    SplitPlan,
    compute_train_range,
    build_split_plan_from_profile,
    split_plan_to_dict,
)

# NEW: Hypothesis imports
from hypothesis import given, settings, strategies as st, assume

# ======================================================================
# Basic profile construction tests
# ======================================================================


def test_splitting_profile_from_dict_defaults():
    cfg = {
        "horizon": 24,
        "shift": 24,
        # no dev_test_split / development_cv / backtest -> defaults should kick in
    }

    profile = SplittingProfile.from_dict("ili_default", cfg)

    assert profile.name == "ili_default"
    assert profile.method == "rolling_origin"
    assert profile.horizon == 24
    assert profile.shift == 24

    # Dev/test split defaults
    assert isinstance(profile.dev_test_split, DevTestSplitConfig)
    assert profile.dev_test_split.mode == "tail_test_folds"
    assert profile.dev_test_split.n_test_folds == 3
    assert profile.dev_test_split.min_dev_points == 200

    # Dev CV defaults
    assert isinstance(profile.development_cv, DevelopmentCVConfig)
    assert profile.development_cv.enabled is True
    assert profile.development_cv.n_folds == 3
    assert profile.development_cv.window_size == 104
    assert profile.development_cv.train_window == "expanding"
    assert profile.development_cv.align_to_dev_end is True

    # Backtest defaults
    assert isinstance(profile.backtest, BacktestConfig)
    assert profile.backtest.enabled is True
    assert profile.backtest.n_folds == 3
    assert profile.backtest.train_window == "expanding"
    assert profile.backtest.start_from_dev_end is True


def test_splitting_profile_validation_rejects_invalid_method():
    cfg = {
        "method": "unknown_method",
        "horizon": 24,
        "shift": 24,
    }
    with pytest.raises(ValueError):
        SplittingProfile.from_dict("bad_profile", cfg)


# ======================================================================
# Dev/test split: simple deterministic scenario
# ======================================================================


def test_dev_test_split_tail_test_folds_simple():
    """
    Simple deterministic test for dev_test_split with tail_test_folds:

    dataset_len = 400
    horizon     = 24
    n_test_folds= 3

    => test_points = 3 * 24 = 72
       test_start  = 400 - 72 = 328
       test_end    = 400
       dev_start   = 0
       dev_end     = 328
    """
    dataset_len = 400
    cfg = {
        "horizon": 24,
        "shift": 24,
        "dev_test_split": {
            "mode": "tail_test_folds",
            "n_test_folds": 3,
            "min_dev_points": 200,
        },
        "development_cv": {
            "enabled": False,
        },
        "backtest": {
            "enabled": False,
        },
    }

    profile = SplittingProfile.from_dict("ili_simple", cfg)
    profile.validate()

    # Build a plan with only dev/test partitioning (no folds)
    plan = build_split_plan_from_profile(
        dataset_name="dummy",
        profile=profile,
        dataset_len=dataset_len,
    )

    # No folds because both dev_cv and backtest are disabled
    assert plan.folds == []


# ======================================================================
# compute_train_range unit tests
# ======================================================================


def test_compute_train_range_expanding():
    """
    For expanding mode, train_start should be global_start, train_end = origin.
    """
    origin = 120
    global_start = 0
    train_start, train_end = compute_train_range(
        origin=origin,
        train_window_mode="expanding",
        train_window_size=999,  # ignored
        global_start=global_start,
        min_train_points=1,
    )
    assert train_start == global_start
    assert train_end == origin
    assert train_end - train_start == 120


def test_compute_train_range_sliding_window():
    """
    For sliding mode, the train window should have fixed length unless constrained
    by global_start.
    """
    origin = 150
    global_start = 0
    window_size = 50

    train_start, train_end = compute_train_range(
        origin=origin,
        train_window_mode="sliding",
        train_window_size=window_size,
        global_start=global_start,
        min_train_points=window_size,
    )
    assert train_end == origin
    assert train_end - train_start == window_size
    assert train_start == origin - window_size


def test_compute_train_range_sliding_respects_global_start():
    """
    If origin - window_size < global_start, the start must be clamped to global_start.
    """
    origin = 10
    global_start = 0
    window_size = 50

    train_start, train_end = compute_train_range(
        origin=origin,
        train_window_mode="sliding",
        train_window_size=window_size,
        global_start=global_start,
        min_train_points=1,
    )
    assert train_start == global_start
    assert train_end == origin
    assert train_end - train_start == 10  # smaller than window_size, but allowed


# ======================================================================
# Integration: build_split_plan_from_profile (expanding)
# ======================================================================


def test_build_split_plan_expanding_dev_and_backtest():
    """
    Full plan for:
      - method: rolling_origin
      - horizon: 24
      - shift: 24
      - dev_test_split: 3 test folds
      - dev CV: 3 folds, expanding
      - backtest: 3 folds, expanding (start_from_dev_end=True)

    This roughly matches your ILI setup in spirit (not exact dataset length).
    """
    dataset_len = 400  # arbitrary but large enough

    cfg = {
        "horizon": 24,
        "shift": 24,
        "dev_test_split": {
            "mode": "tail_test_folds",
            "n_test_folds": 3,
            "min_dev_points": 200,
        },
        "development_cv": {
            "enabled": True,
            "n_folds": 3,
            "window_size": 104,
            "train_window": "expanding",
            "align_to_dev_end": True,
        },
        "backtest": {
            "enabled": True,
            "n_folds": 3,
            "train_window": "expanding",
            "start_from_dev_end": True,
        },
    }
    profile = SplittingProfile.from_dict("ili_like", cfg)

    plan = build_split_plan_from_profile(
        dataset_name="illness",
        profile=profile,
        dataset_len=dataset_len,
    )

    # There should be:
    #   dev:   3 folds -> each fold: train + val -> 6 specs
    #   test:  3 folds -> each fold: train + test -> 6 specs
    # Total: 12 FoldSpec
    assert len(plan.folds) == (3 * 2 + 3 * 2)

    dev_train = plan.folds_by_role(partition_type="dev", role="train")
    dev_val = plan.folds_by_role(partition_type="dev", role="val")
    test_train = plan.folds_by_role(partition_type="test", role="train")
    test_test = plan.folds_by_role(partition_type="test", role="test")

    assert len(dev_train) == 3
    assert len(dev_val) == 3
    assert len(test_train) == 3
    assert len(test_test) == 3

    horizon = profile.horizon

    # Basic consistency checks for dev CV folds
    for fold_idx in range(3):
        t = [f for f in dev_train if f.fold_index == fold_idx][0]
        v = [f for f in dev_val if f.fold_index == fold_idx][0]

        # Train must end where val starts
        assert t.end_idx == v.start_idx
        # Val length must equal horizon
        assert v.end_idx - v.start_idx == horizon
        # Train is non-empty
        assert t.end_idx > t.start_idx

    # Basic consistency checks for backtest folds
    for fold_idx in range(3):
        t = [f for f in test_train if f.fold_index == fold_idx][0]
        te = [f for f in test_test if f.fold_index == fold_idx][0]

        assert t.end_idx == te.start_idx
        assert te.end_idx - te.start_idx == horizon
        assert t.end_idx > t.start_idx

    # Fold indices are 0..2 for each (partition_type, role)
    assert sorted({f.fold_index for f in dev_train}) == [0, 1, 2]
    assert sorted({f.fold_index for f in dev_val}) == [0, 1, 2]
    assert sorted({f.fold_index for f in test_train}) == [0, 1, 2]
    assert sorted({f.fold_index for f in test_test}) == [0, 1, 2]


# ======================================================================
# Integration: build_split_plan_from_profile (sliding)
# ======================================================================


def test_build_split_plan_sliding_dev_only():
    """
    Plan with sliding train window on dev segment only.
    Backtest disabled to keep the test simple.

    We mainly check that:
      - each train fold uses sliding window semantics,
      - val folds are immediately after train,
      - horizon is respected.
    """
    dataset_len = 400
    window_size = 50
    horizon = 12

    cfg = {
        "horizon": horizon,
        "shift": horizon,
        "dev_test_split": {
            "mode": "tail_test_folds",
            "n_test_folds": 2,
            "min_dev_points": 150,
        },
        "development_cv": {
            "enabled": True,
            "n_folds": 3,
            "window_size": window_size,
            "train_window": "sliding",
            "align_to_dev_end": True,
        },
        "backtest": {
            "enabled": False,
        },
    }

    profile = SplittingProfile.from_dict("sliding_dev", cfg)
    plan = build_split_plan_from_profile(
        dataset_name="dummy",
        profile=profile,
        dataset_len=dataset_len,
    )

    dev_train = plan.folds_by_role(partition_type="dev", role="train")
    dev_val = plan.folds_by_role(partition_type="dev", role="val")

    assert len(dev_train) == 3
    assert len(dev_val) == 3

    for fold_idx in range(3):
        t = [f for f in dev_train if f.fold_index == fold_idx][0]
        v = [f for f in dev_val if f.fold_index == fold_idx][0]

        # Val is directly after train
        assert t.end_idx == v.start_idx
        # Val length = horizon
        assert v.end_idx - v.start_idx == horizon

        # In sliding mode we aim for fixed-length train windows,
        # but they may be shorter for the earliest folds (due to global_start).
        train_len = t.end_idx - t.start_idx
        assert train_len > 0
        assert train_len <= window_size


# ======================================================================
# JSON export roundtrip
# ======================================================================


def test_split_plan_to_dict_and_json_roundtrip(tmp_path):
    dataset_len = 300
    cfg = {
        "horizon": 24,
        "shift": 24,
        "dev_test_split": {
            "mode": "tail_test_folds",
            "n_test_folds": 2,
            "min_dev_points": 100,
        },
        "development_cv": {
            "enabled": True,
            "n_folds": 2,
            "window_size": 60,
            "train_window": "expanding",
            "align_to_dev_end": True,
        },
        "backtest": {
            "enabled": True,
            "n_folds": 2,
            "train_window": "expanding",
            "start_from_dev_end": True,
        },
    }

    profile = SplittingProfile.from_dict("roundtrip", cfg)
    plan = build_split_plan_from_profile(
        dataset_name="roundtrip_ds",
        profile=profile,
        dataset_len=dataset_len,
    )

    data = split_plan_to_dict(plan, profile=profile, extra_metadata={"run_id": "abc123"})
    assert data["dataset_name"] == "roundtrip_ds"
    assert "folds" in data
    assert len(data["folds"]) > 0
    assert data["profile"]["name"] == "roundtrip"
    assert data["metadata"]["run_id"] == "abc123"

    # JSON roundtrip
    json_path = tmp_path / "splits.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f)

    with json_path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["dataset_name"] == data["dataset_name"]
    assert loaded["profile"]["horizon"] == data["profile"]["horizon"]
    assert loaded["metadata"]["run_id"] == "abc123"


# ======================================================================
# Hypothesis property tests for splitting
# ======================================================================


@st.composite
def splitting_scenarios(draw):
    """
    Generate a wide range of consistent splitting configurations
    plus a dataset length.

    We keep the space relatively constrained to avoid constant
    ValueErrors from obviously impossible setups.
    """
    dataset_len = draw(st.integers(min_value=300, max_value=1500))

    # Number of test folds and horizon
    n_test_folds = draw(st.integers(min_value=1, max_value=5))
    max_h = max(4, min(52, dataset_len // (n_test_folds + 2)))
    horizon = draw(st.integers(min_value=4, max_value=max_h))

    shift = horizon  # keep it simple and consistent

    # Dev/test split config
    min_dev_points = draw(
        st.integers(
            min_value=horizon * 2,
            max_value=max(horizon * 3, dataset_len // 4),
        )
    )

    dev_cv_enabled = draw(st.booleans())
    backtest_enabled = draw(st.booleans())

    dev_n_folds = draw(st.integers(min_value=1, max_value=4))
    backtest_n_folds = draw(st.integers(min_value=1, max_value=4))

    window_size = draw(
        st.integers(
            min_value=horizon * 2,
            max_value=min(dataset_len // 2, horizon * 8),
        )
    )

    dev_train_window = draw(st.sampled_from(["expanding", "sliding"]))
    backtest_train_window = draw(st.sampled_from(["expanding", "sliding"]))

    cfg = {
        "horizon": horizon,
        "shift": shift,
        "dev_test_split": {
            "mode": "tail_test_folds",
            "n_test_folds": n_test_folds,
            "min_dev_points": min_dev_points,
        },
        "development_cv": {
            "enabled": dev_cv_enabled,
            "n_folds": dev_n_folds,
            "window_size": window_size,
            "train_window": dev_train_window,
            "align_to_dev_end": True,
        },
        "backtest": {
            "enabled": backtest_enabled,
            "n_folds": backtest_n_folds,
            "train_window": backtest_train_window,
            "start_from_dev_end": True,
        },
    }

    return cfg, dataset_len


@settings(max_examples=50, deadline=None)
@given(splitting_scenarios())
def test_split_plan_indices_are_within_bounds(scenario):
    """
    For any valid split plan, all folds must:
      - stay within [0, dataset_len],
      - have non-empty ranges.
    """
    cfg, dataset_len = scenario

    # Build profile & plan; skip impossible configs with ValueError
    try:
        profile = SplittingProfile.from_dict("prop", cfg)
        plan = build_split_plan_from_profile(
            dataset_name="ds",
            profile=profile,
            dataset_len=dataset_len,
        )
    except ValueError:
        assume(False)

    for f in plan.folds:
        assert 0 <= f.start_idx < f.end_idx <= dataset_len
        assert f.end_idx - f.start_idx > 0


@settings(max_examples=50, deadline=None)
@given(splitting_scenarios())
def test_split_plan_val_and_test_have_horizon_length(scenario):
    """
    For any valid split plan:
      - dev/val folds must have length == horizon,
      - test/test folds must have length == horizon.
    """
    cfg, dataset_len = scenario

    try:
        profile = SplittingProfile.from_dict("prop", cfg)
        plan = build_split_plan_from_profile(
            dataset_name="ds",
            profile=profile,
            dataset_len=dataset_len,
        )
    except ValueError:
        assume(False)

    horizon = profile.horizon

    dev_val = plan.folds_by_role(partition_type="dev", role="val")
    test_test = plan.folds_by_role(partition_type="test", role="test")

    for f in dev_val + test_test:
        assert f.end_idx - f.start_idx == horizon


@settings(max_examples=50, deadline=None)
@given(splitting_scenarios())
def test_split_plan_train_val_test_are_contiguous_per_fold(scenario):
    """
    For any valid split plan, for a given (partition_type, fold_index),
    train and val/test folds must be contiguous and non-overlapping:
      dev:  train -> val
      test: train -> test
    """
    cfg, dataset_len = scenario

    try:
        profile = SplittingProfile.from_dict("prop", cfg)
        plan = build_split_plan_from_profile(
            dataset_name="ds",
            profile=profile,
            dataset_len=dataset_len,
        )
    except ValueError:
        assume(False)

    # dev: train -> val
    dev_train = plan.folds_by_role(partition_type="dev", role="train")
    dev_val = plan.folds_by_role(partition_type="dev", role="val")

    fold_indices = sorted({f.fold_index for f in dev_train + dev_val})
    for idx in fold_indices:
        t = [f for f in dev_train if f.fold_index == idx]
        v = [f for f in dev_val if f.fold_index == idx]
        if t and v:
            t = t[0]
            v = v[0]
            assert t.end_idx == v.start_idx
            assert t.start_idx < t.end_idx <= v.start_idx < v.end_idx

    # test: train -> test
    test_train = plan.folds_by_role(partition_type="test", role="train")
    test_test = plan.folds_by_role(partition_type="test", role="test")

    fold_indices = sorted({f.fold_index for f in test_train + test_test})
    for idx in fold_indices:
        t = [f for f in test_train if f.fold_index == idx]
        te = [f for f in test_test if f.fold_index == idx]
        if t and te:
            t = t[0]
            te = te[0]
            assert t.end_idx == te.start_idx
            assert t.start_idx < t.end_idx <= te.start_idx < te.end_idx


@settings(max_examples=50, deadline=None)
@given(splitting_scenarios())
def test_split_plan_folds_are_monotonic_within_role(scenario):
    """
    For any valid split plan, folds of the same (partition_type, role)
    and different fold_index must form a monotonic sequence in time:

      - when sorted by (fold_index, start_idx), both start_idx and end_idx
        are non-decreasing,
      - end_idx is strictly increasing (each fold covers strictly more
        or at least later data than the previous one).

    Overlap is allowed (and expected) for expanding windows; this test
    only checks time monotonicity, not disjointness.
    """
    cfg, dataset_len = scenario

    try:
        profile = SplittingProfile.from_dict("prop", cfg)
        plan = build_split_plan_from_profile(
            dataset_name="ds",
            profile=profile,
            dataset_len=dataset_len,
        )
    except ValueError:
        assume(False)

    def check_monotonic(partition_type: str, role: str):
        folds = plan.folds_by_role(partition_type=partition_type, role=role)
        if not folds:
            return

        # Sort by (fold_index, start_idx) to get a consistent temporal order
        folds_sorted = sorted(
            folds,
            key=lambda f: (f.fold_index, f.start_idx, f.end_idx),
        )

        prev = folds_sorted[0]
        for curr in folds_sorted[1:]:
            # Same role and partition -> later folds should not move backward in time
            assert curr.start_idx >= prev.start_idx
            assert curr.end_idx > prev.end_idx  # strictly later / wider
            prev = curr

    for pt in ["dev", "test"]:
        for role in ["train", "val", "test"]:
            check_monotonic(partition_type=pt, role=role)


@settings(max_examples=50, deadline=None)
@given(splitting_scenarios())
def test_split_plan_sliding_train_window_respects_max_size(scenario):
    """
    For any plan using sliding train windows, ensure that
    train_len <= window_size for those segments.
    """
    cfg, dataset_len = scenario

    try:
        profile = SplittingProfile.from_dict("prop", cfg)
        plan = build_split_plan_from_profile(
            dataset_name="ds",
            profile=profile,
            dataset_len=dataset_len,
        )
    except ValueError:
        assume(False)

    dev_cfg = profile.development_cv
    back_cfg = profile.backtest

    # dev segment
    if dev_cfg.enabled and dev_cfg.train_window == "sliding":
        dev_train = plan.folds_by_role(partition_type="dev", role="train")
        for f in dev_train:
            train_len = f.end_idx - f.start_idx
            assert train_len <= dev_cfg.window_size
            assert train_len > 0

    # test segment (backtest)
    if back_cfg.enabled and back_cfg.train_window == "sliding":
        test_train = plan.folds_by_role(partition_type="test", role="train")
        # We don't have an explicit window_size in backtest config,
        # but we still expect strictly positive length.
        for f in test_train:
            train_len = f.end_idx - f.start_idx
            assert train_len > 0
