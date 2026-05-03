# utils/splitting.py

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import json

PartitionType = Literal["dev", "test"]
FoldRole = Literal["train", "val", "test"]
TrainWindowMode = Literal["expanding", "sliding"]
SplitMethod = Literal["rolling_origin"]


# ======================================================================
# Core data structures
# ======================================================================


@dataclass(frozen=True)
class FoldSpec:
    """
    A single contiguous segment of the time axis, with a role and a fold index.

    Indices are integer offsets into the dataset after any global filtering /
    alignment has been applied. The [start_idx, end_idx) convention is used
    (end-exclusive), so length = end_idx - start_idx.

    Example:
        partition_type = "dev"
        role           = "train"
        fold_index     = 1
        start_idx      = 120
        end_idx        = 224
    """

    partition_type: PartitionType
    role: FoldRole
    fold_index: int
    start_idx: int
    end_idx: int

    def length(self) -> int:
        return max(0, self.end_idx - self.start_idx)


@dataclass(frozen=True)
class SplitPlan:
    """
    Full description of how a dataset is split into development and test
    segments, and then into folds for CV and backtesting.

    The same dataset_name can be used with multiple SplitPlans (for different
    experiments or profiles), but within a single plan, `fold_index` is
    interpreted separately per (partition_type, role) pair.
    """

    dataset_name: str
    folds: List[FoldSpec] = field(default_factory=list)

    # Convenience accessors ------------------------------------------------

    def dev_folds(self) -> List[FoldSpec]:
        return [f for f in self.folds if f.partition_type == "dev"]

    def test_folds(self) -> List[FoldSpec]:
        return [f for f in self.folds if f.partition_type == "test"]

    def folds_by_role(
        self,
        partition_type: Optional[PartitionType] = None,
        role: Optional[FoldRole] = None,
    ) -> List[FoldSpec]:
        folds = self.folds
        if partition_type is not None:
            folds = [f for f in folds if f.partition_type == partition_type]
        if role is not None:
            folds = [f for f in folds if f.role == role]
        return folds


# ======================================================================
# Config dataclasses for splitting profiles
# ======================================================================


@dataclass
class DevTestSplitConfig:
    """
    How to split the full dataset into a development segment (for HPO / CV)
    and a test segment (for backtesting / final evaluation).

    Currently we support one mode:

      - mode = "tail_test_folds":
          The last (n_test_folds * horizon) points are reserved for test.
          The rest is used as development, subject to min_dev_points.

    All indices are computed in "global" dataset coordinates: [0, N).
    """

    mode: Literal["tail_test_folds"] = "tail_test_folds"
    n_test_folds: int = 3
    min_dev_points: int = 200

    def validate(self) -> None:
        if self.n_test_folds <= 0:
            raise ValueError("DevTestSplitConfig.n_test_folds must be > 0.")
        if self.min_dev_points <= 0:
            raise ValueError("DevTestSplitConfig.min_dev_points must be > 0.")


@dataclass
class DevelopmentCVConfig:
    """
    Cross-validation configuration for the development (HPO) segment.

    - enabled:
        Whether to perform CV on development data.

    - n_folds:
        Number of CV folds.

    - window_size:
        Size of the conditioning window (history) used by the model.
        This is used only to compute the earliest possible origin.

    - train_window:
        "expanding" or "sliding":
          * expanding:
              train_start = dev_start
              train_end   = origin
          * sliding:
              train_end   = origin
              train_start = max(dev_start, train_end - train_window_size)

    - align_to_dev_end:
        If True, origins are chosen so that the *last* fold ends exactly at
        dev_end. Otherwise, we still enforce that folds do not cross dev_end,
        but we do not force exact alignment.
    """

    enabled: bool = True
    n_folds: int = 3
    window_size: int = 104
    train_window: TrainWindowMode = "expanding"
    align_to_dev_end: bool = True

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.n_folds <= 0:
            raise ValueError("DevelopmentCVConfig.n_folds must be > 0.")
        if self.window_size <= 0:
            raise ValueError("DevelopmentCVConfig.window_size must be > 0.")
        if self.train_window not in ("expanding", "sliding"):
            raise ValueError(
                f"DevelopmentCVConfig.train_window must be 'expanding' or 'sliding', "
                f"got {self.train_window!r}"
            )


@dataclass
class BacktestConfig:
    """
    Backtesting configuration for the test segment.

    - enabled:
        Whether to perform backtesting on the test data.

    - n_folds:
        Number of rolling-origin folds on the test segment.

    - train_window:
        "expanding" or "sliding" (same semantics as in DevelopmentCVConfig).

    - start_from_dev_end:
        If True, the training ranges for test folds are allowed to extend
        into the dev segment (i.e. train can start at dev_start).
        If False, training uses only the test segment as its time axis.
    """

    enabled: bool = True
    n_folds: int = 3
    train_window: TrainWindowMode = "expanding"
    start_from_dev_end: bool = True

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.n_folds <= 0:
            raise ValueError("BacktestConfig.n_folds must be > 0.")
        if self.train_window not in ("expanding", "sliding"):
            raise ValueError(
                f"BacktestConfig.train_window must be 'expanding' or 'sliding', "
                f"got {self.train_window!r}"
            )


@dataclass
class SplittingProfile:
    """
    High-level splitting profile for a dataset / experiment.

    This is meant to be constructed from a YAML config section, e.g.:

        splitting_profiles:
          ili_default:
            method: "rolling_origin"
            horizon: 24
            shift: 24

            dev_test_split:
              mode: "tail_test_folds"
              n_test_folds: 3
              min_dev_points: 200

            development_cv:
              enabled: true
              n_folds: 3
              window_size: 104
              train_window: "expanding"
              align_to_dev_end: true

            backtest:
              enabled: true
              n_folds: 3
              train_window: "expanding"
              start_from_dev_end: true
    """

    name: str
    method: SplitMethod = "rolling_origin"
    horizon: int = 1
    shift: int = 1
    dev_test_split: DevTestSplitConfig = field(default_factory=DevTestSplitConfig)
    development_cv: DevelopmentCVConfig = field(default_factory=DevelopmentCVConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "SplittingProfile":
        """
        Build a SplittingProfile from a plain config dict (e.g. Hydra / YAML).

        Unknown keys are ignored at the top level. Missing subsections are
        filled with defaults.
        """
        method = data.get("method", "rolling_origin")
        horizon = data.get("horizon", 1)
        shift = data.get("shift", horizon)

        dts_cfg = data.get("dev_test_split", {}) or {}
        dev_test_split = DevTestSplitConfig(
            mode=dts_cfg.get("mode", "tail_test_folds"),
            n_test_folds=dts_cfg.get("n_test_folds", 3),
            min_dev_points=dts_cfg.get("min_dev_points", 200),
        )

        dev_cv_cfg = data.get("development_cv", {}) or {}
        development_cv = DevelopmentCVConfig(
            enabled=dev_cv_cfg.get("enabled", True),
            n_folds=dev_cv_cfg.get("n_folds", 3),
            window_size=dev_cv_cfg.get("window_size", 104),
            train_window=dev_cv_cfg.get("train_window", "expanding"),
            align_to_dev_end=dev_cv_cfg.get("align_to_dev_end", True),
        )

        bt_cfg = data.get("backtest", {}) or {}
        backtest = BacktestConfig(
            enabled=bt_cfg.get("enabled", True),
            n_folds=bt_cfg.get("n_folds", 3),
            train_window=bt_cfg.get("train_window", "expanding"),
            start_from_dev_end=bt_cfg.get("start_from_dev_end", True),
        )

        profile = cls(
            name=name,
            method=method,  # type: ignore[arg-type]
            horizon=horizon,
            shift=shift,
            dev_test_split=dev_test_split,
            development_cv=development_cv,
            backtest=backtest,
        )
        profile.validate()
        return profile

    def validate(self) -> None:
        if self.method != "rolling_origin":
            raise ValueError(
                f"Only 'rolling_origin' method is supported at the moment, "
                f"got {self.method!r}"
            )
        if self.horizon <= 0:
            raise ValueError("SplittingProfile.horizon must be > 0.")
        if self.shift <= 0:
            raise ValueError("SplittingProfile.shift must be > 0.")

        self.dev_test_split.validate()
        self.development_cv.validate()
        self.backtest.validate()


# ======================================================================
# Helper for computing train ranges (expanding / sliding)
# ======================================================================


def compute_train_range(
    origin: int,
    train_window_mode: TrainWindowMode,
    train_window_size: int,
    global_start: int,
    min_train_points: int = 1,
) -> Tuple[int, int]:
    """
    Compute [train_start, train_end) given the origin index and train_window mode.

    Parameters
    ----------
    origin:
        The index (in global coordinates) where the forecast window starts.
        Train always ends at `origin`.

    train_window_mode:
        "expanding" or "sliding".

    train_window_size:
        Used only for "sliding" mode. For "expanding" it can be any positive
        integer (ignored).

    global_start:
        The earliest allowable start index (e.g. dev_start or test_start).

    min_train_points:
        Minimum required number of training points in the range.

    Returns
    -------
    (train_start, train_end):
        A pair of indices (inclusive start, exclusive end).

    Raises
    ------
    ValueError:
        If the resulting training segment would have fewer than min_train_points
        elements.
    """
    if origin <= global_start:
        raise ValueError(
            f"Origin {origin} must be > global_start {global_start} "
            "to allow a non-empty training segment."
        )

    if train_window_mode == "expanding":
        train_start = global_start
        train_end = origin
    elif train_window_mode == "sliding":
        if train_window_size <= 0:
            raise ValueError(
                "train_window_size must be > 0 in 'sliding' mode."
            )
        train_end = origin
        train_start = max(global_start, train_end - train_window_size)
    else:
        raise ValueError(
            f"Unknown train_window_mode: {train_window_mode!r}"
        )

    if train_end - train_start < min_train_points:
        raise ValueError(
            f"Insufficient train length: got {train_end - train_start}, "
            f"required >= {min_train_points} (origin={origin}, "
            f"global_start={global_start}, mode={train_window_mode!r})."
        )

    return train_start, train_end


# ======================================================================
# Rolling-origin split builders
# ======================================================================


def _compute_dev_test_ranges(
    profile: SplittingProfile,
    dataset_len: int,
) -> Tuple[int, int, int, int]:
    """
    Compute (dev_start, dev_end, test_start, test_end) in [0, dataset_len)
    given the DevTestSplitConfig.

    dev_* and test_* are in global coordinates and use [start, end) convention.
    """
    if dataset_len <= 0:
        raise ValueError("dataset_len must be > 0.")

    dts = profile.dev_test_split
    horizon = profile.horizon

    if dts.mode != "tail_test_folds":
        raise ValueError(
            f"Unsupported DevTestSplitConfig.mode: {dts.mode!r}"
        )

    test_points = dts.n_test_folds * horizon
    if test_points >= dataset_len:
        raise ValueError(
            f"Not enough points ({dataset_len}) to allocate "
            f"{dts.n_test_folds} test folds with horizon={horizon}."
        )

    test_start = dataset_len - test_points
    test_end = dataset_len

    dev_start = 0
    dev_end = test_start

    if dev_end - dev_start < dts.min_dev_points:
        raise ValueError(
            f"Development segment too short: {dev_end - dev_start} points, "
            f"required >= {dts.min_dev_points}."
        )

    return dev_start, dev_end, test_start, test_end


def _build_rolling_origins(
    segment_start: int,
    segment_end: int,
    horizon: int,
    shift: int,
    n_folds: int,
) -> List[int]:
    """
    Compute rolling-origin indices for a given segment [segment_start, segment_end).

    Each origin defines a forecast window [origin, origin + horizon), which
    must be contained within [segment_start, segment_end).

    We generate all possible origins with step=shift, then, if there are more
    than n_folds, we keep the last n_folds (most recent folds).
    """
    max_origin = segment_end - horizon
    if max_origin <= segment_start:
        raise ValueError(
            f"Segment [{segment_start}, {segment_end}) is too short for "
            f"horizon={horizon}."
        )

    origins: List[int] = []
    origin = segment_start
    while origin <= max_origin:
        origins.append(origin)
        origin += shift

    if not origins:
        raise ValueError(
            f"No valid origins generated for segment "
            f"[{segment_start}, {segment_end}), horizon={horizon}, shift={shift}."
        )

    if len(origins) > n_folds:
        origins = origins[-n_folds:]

    if len(origins) < n_folds:
        # We choose to be strict here: if config asks for n_folds, we enforce it.
        raise ValueError(
            f"Requested {n_folds} folds but only {len(origins)} origins "
            f"are possible for horizon={horizon}, shift={shift}, "
            f"segment=[{segment_start}, {segment_end})."
        )

    return origins


def build_split_plan_from_profile(
    dataset_name: str,
    profile: SplittingProfile,
    dataset_len: int,
) -> SplitPlan:
    """
    Build a SplitPlan for a single dataset using a SplittingProfile.

    This is the high-level entry point that:

      1. Splits the time axis into dev/test segments (dev_test_split).
      2. Builds CV folds on the dev segment (development_cv).
      3. Builds backtest folds on the test segment (backtest).

    This function does *not* interact with any model code – it only operates
    on integer indices [0, dataset_len).

    Parameters
    ----------
    dataset_name:
        Name of the dataset (used for logging / JSON export only).

    profile:
        SplittingProfile describing method, horizon, shift and segment configs.

    dataset_len:
        Number of time steps in the dataset after any global preprocessing.

    Returns
    -------
    SplitPlan:
        An immutable plan that can be consumed by higher-level training /
        evaluation code.
    """
    profile.validate()
    dev_start, dev_end, test_start, test_end = _compute_dev_test_ranges(
        profile, dataset_len
    )

    folds: List[FoldSpec] = []

    horizon = profile.horizon
    shift = profile.shift

    # ------------------------------------------------------------------
    # Development CV folds
    # ------------------------------------------------------------------
    dev_cfg = profile.development_cv
    if dev_cfg.enabled:
        origins = _build_rolling_origins(
            segment_start=dev_start,
            segment_end=dev_end,
            horizon=horizon,
            shift=shift,
            n_folds=dev_cfg.n_folds,
        )

        for fold_idx, origin in enumerate(origins):
            train_start, train_end = compute_train_range(
                origin=origin,
                train_window_mode=dev_cfg.train_window,
                train_window_size=dev_cfg.window_size,
                global_start=dev_start,
                min_train_points=dev_cfg.window_size
                if dev_cfg.train_window == "sliding"
                else 1,
            )
            val_start = origin
            val_end = origin + horizon

            folds.append(
                FoldSpec(
                    partition_type="dev",
                    role="train",
                    fold_index=fold_idx,
                    start_idx=train_start,
                    end_idx=train_end,
                )
            )
            folds.append(
                FoldSpec(
                    partition_type="dev",
                    role="val",
                    fold_index=fold_idx,
                    start_idx=val_start,
                    end_idx=val_end,
                )
            )

    # ------------------------------------------------------------------
    # Backtest folds
    # ------------------------------------------------------------------
    bt_cfg = profile.backtest
    if bt_cfg.enabled:
        origins = _build_rolling_origins(
            segment_start=test_start,
            segment_end=test_end,
            horizon=horizon,
            shift=shift,
            n_folds=bt_cfg.n_folds,
        )

        # Decide the global_start for training ranges in backtest
        if bt_cfg.start_from_dev_end:
            global_start = dev_start
        else:
            global_start = test_start

        for fold_idx, origin in enumerate(origins):
            train_start, train_end = compute_train_range(
                origin=origin,
                train_window_mode=bt_cfg.train_window,
                train_window_size=dev_cfg.window_size,  # reuse same window_size
                global_start=global_start,
                min_train_points=1,
            )
            test_fold_start = origin
            test_fold_end = origin + horizon

            folds.append(
                FoldSpec(
                    partition_type="test",
                    role="train",
                    fold_index=fold_idx,
                    start_idx=train_start,
                    end_idx=train_end,
                )
            )
            folds.append(
                FoldSpec(
                    partition_type="test",
                    role="test",
                    fold_index=fold_idx,
                    start_idx=test_fold_start,
                    end_idx=test_fold_end,
                )
            )

    return SplitPlan(dataset_name=dataset_name, folds=folds)


# ======================================================================
# Legacy compatibility hook (Phase 0 safety)
# ======================================================================


def build_split_plan_legacy(
    dataset_name: str,
    dataset_len: int,
    forecast_steps: int,
    n_folds: int,
    window_size: int,
) -> SplitPlan:
    """
    Legacy split builder placeholder.

    This function is meant to wrap the *existing* splitting logic in your
    framework and expose it as a SplitPlan, without changing the behavior.

    For Phase 0 you can leave this as a stub and only implement it when
    you start refactoring the old code. For now it raises to prevent
    accidental use.
    """
    raise NotImplementedError(
        "build_split_plan_legacy is a placeholder. "
        "Wire your existing splitting logic here during refactoring."
    )


# ======================================================================
# JSON export helpers (for visualization / debugging)
# ======================================================================


def split_plan_to_dict(
    plan: SplitPlan,
    profile: Optional[SplittingProfile] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convert a SplitPlan (and optionally its SplittingProfile) to a JSON-
    serializable dict. This is the format expected by visualize_splits.py.

    The structure is intentionally simple and self-explanatory.
    """
    data: Dict[str, Any] = {
        "dataset_name": plan.dataset_name,
        "folds": [
            {
                "partition_type": f.partition_type,
                "role": f.role,
                "fold_index": f.fold_index,
                "start_idx": f.start_idx,
                "end_idx": f.end_idx,
            }
            for f in plan.folds
        ],
    }

    if profile is not None:
        data["profile"] = {
            "name": profile.name,
            "method": profile.method,
            "horizon": profile.horizon,
            "shift": profile.shift,
            "dev_test_split": asdict(profile.dev_test_split),
            "development_cv": asdict(profile.development_cv),
            "backtest": asdict(profile.backtest),
        }

    if extra_metadata:
        data["metadata"] = extra_metadata

    return data


def save_split_plan(
    path: Union[str, Path],
    plan: SplitPlan,
    profile: Optional[SplittingProfile] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a SplitPlan + optional profile/metadata to a JSON file.

    Parameters
    ----------
    path:
        Target path (str or Path). Parent directories are created if needed.

    plan:
        The SplitPlan to serialize.

    profile:
        Optional SplittingProfile used to produce this plan.

    extra_metadata:
        Optional dict with arbitrary extra metadata (experiment name, run id,
        git commit hash, etc.).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = split_plan_to_dict(plan, profile=profile, extra_metadata=extra_metadata)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
