"""Leakage-resistant validation and accept/pool/abstain decisions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import norm

from .models import CompositeRegressor, PredictionResult


@dataclass(frozen=True)
class ValidationSplit:
    scheme: str
    fold_id: str
    train_index: np.ndarray
    test_index: np.ndarray
    buffered_out_index: np.ndarray


@dataclass(frozen=True)
class MetricSummary:
    hole_balanced_mae: float
    hole_balanced_rmse: float
    hole_balanced_bias: float
    length_weighted_mae: float
    length_weighted_rmse: float
    length_weighted_bias: float
    interval_coverage: float
    failure_rate: float
    n_rows: int
    n_holes: int


@dataclass(frozen=True)
class PairedInterval:
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    n_holes: int


@dataclass(frozen=True)
class SchemeEvidence:
    scheme: str
    mae_difference: PairedInterval
    coverage_difference: float
    conditioned_failure_rate: float
    comparator_failure_rate: float
    conditioned_interval_coverage: float = float("nan")
    comparator_interval_coverage: float = float("nan")
    paired_rows: int = 0
    paired_holes: int = 0
    joint_success_rate: float = float("nan")


@dataclass(frozen=True)
class DecisionOutcome:
    action: str
    selected_policy: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class TuningResult:
    best_parameters: Mapping[str, object]
    best_score: float
    candidate_scores: tuple[tuple[Mapping[str, object], float], ...]


def leave_one_hole_out_splits(
    data: pd.DataFrame, hole_col: str = "BHID"
) -> tuple[ValidationSplit, ...]:
    """Return deterministic whole-hole folds."""

    if hole_col not in data:
        raise ValueError(f"missing hole column {hole_col!r}")
    holes = data[hole_col].astype(str).to_numpy()
    splits: list[ValidationSplit] = []
    for hole in sorted(pd.unique(holes)):
        test = np.flatnonzero(holes == hole)
        train = np.flatnonzero(holes != hole)
        if len(train) and len(test):
            splits.append(
                ValidationSplit(
                    scheme="leave_one_hole_out",
                    fold_id=str(hole),
                    train_index=train,
                    test_index=test,
                    buffered_out_index=np.asarray([], dtype=int),
                )
            )
    return tuple(splits)


def _hole_centres(
    data: pd.DataFrame,
    hole_col: str,
    coordinate_cols: Sequence[str],
) -> pd.DataFrame:
    required = [hole_col, *coordinate_cols]
    missing = [column for column in required if column not in data]
    if missing:
        raise ValueError(f"missing columns for spatial splitting: {missing}")
    centres = (
        data.loc[:, required]
        .assign(**{hole_col: data[hole_col].astype(str)})
        .groupby(hole_col, sort=True)[list(coordinate_cols)]
        .median()
    )
    if not np.all(np.isfinite(centres.to_numpy(dtype=float))):
        raise ValueError("hole centres contain non-finite coordinates")
    return centres


def median_nearest_hole_distance(
    data: pd.DataFrame,
    *,
    hole_col: str = "BHID",
    coordinate_cols: Sequence[str] = ("mid_easting", "mid_northing"),
) -> float:
    centres = _hole_centres(data, hole_col, coordinate_cols)
    if len(centres) < 2:
        return 0.0
    distances = cdist(centres.to_numpy(), centres.to_numpy())
    np.fill_diagonal(distances, np.inf)
    return float(np.median(np.min(distances, axis=1)))


def spatial_block_splits(
    data: pd.DataFrame,
    *,
    n_blocks: int = 5,
    hole_col: str = "BHID",
    northing_col: str = "mid_northing",
    easting_col: str = "mid_easting",
    block_col: str | None = "northing_block",
    buffer_distance: float | str | None = "median_nn",
) -> tuple[ValidationSplit, ...]:
    """Create contiguous northing blocks with an optional hole-centre buffer."""

    if n_blocks < 2:
        raise ValueError("n_blocks must be at least 2")
    centres = _hole_centres(data, hole_col, (easting_col, northing_col))
    hole_names = centres.index.to_numpy(dtype=str)

    if block_col is not None and block_col in data:
        per_hole = (
            data.assign(**{hole_col: data[hole_col].astype(str)})
            .groupby(hole_col, sort=True)[block_col]
            .first()
            .reindex(hole_names)
        )
        if per_hole.isna().any():
            raise ValueError("precomputed spatial block is missing for some holes")
        labels = per_hole.astype(str).to_numpy()
    else:
        order = np.argsort(
            centres[northing_col].to_numpy(dtype=float), kind="mergesort"
        )
        labels = np.empty(len(centres), dtype=object)
        for block_number, positions in enumerate(np.array_split(order, n_blocks)):
            labels[positions] = f"N{block_number + 1}"
        labels = labels.astype(str)

    if buffer_distance == "median_nn":
        resolved_buffer = median_nearest_hole_distance(
            data,
            hole_col=hole_col,
            coordinate_cols=(easting_col, northing_col),
        )
    elif buffer_distance is None:
        resolved_buffer = 0.0
    else:
        resolved_buffer = float(buffer_distance)
        if resolved_buffer < 0:
            raise ValueError("buffer_distance must be nonnegative")

    row_holes = data[hole_col].astype(str).to_numpy()
    centre_array = centres.loc[hole_names].to_numpy(dtype=float)
    centre_distances = cdist(centre_array, centre_array)
    splits: list[ValidationSplit] = []
    for label in sorted(pd.unique(labels)):
        test_holes = hole_names[labels == label]
        test_hole_mask = np.isin(hole_names, test_holes)
        if resolved_buffer > 0:
            close_to_test = np.any(
                centre_distances[:, test_hole_mask] <= resolved_buffer,
                axis=1,
            )
        else:
            close_to_test = test_hole_mask.copy()
        train_holes = hole_names[~close_to_test]
        buffered_holes = hole_names[close_to_test & ~test_hole_mask]
        train = np.flatnonzero(np.isin(row_holes, train_holes))
        test = np.flatnonzero(np.isin(row_holes, test_holes))
        buffered = np.flatnonzero(np.isin(row_holes, buffered_holes))
        if len(train) and len(test):
            splits.append(
                ValidationSplit(
                    scheme="northing_block_buffered",
                    fold_id=str(label),
                    train_index=train,
                    test_index=test,
                    buffered_out_index=buffered,
                )
            )
    return tuple(splits)


def leave_one_batch_out_splits(
    data: pd.DataFrame, batch_col: str = "BATCH_NUMBER"
) -> tuple[ValidationSplit, ...]:
    if batch_col not in data:
        raise ValueError(f"missing batch column {batch_col!r}")
    groups = data[batch_col].astype(str).to_numpy()
    splits: list[ValidationSplit] = []
    for batch in sorted(pd.unique(groups)):
        test = np.flatnonzero(groups == batch)
        train = np.flatnonzero(groups != batch)
        if len(train) and len(test):
            splits.append(
                ValidationSplit(
                    scheme="leave_one_batch_out",
                    fold_id=str(batch),
                    train_index=train,
                    test_index=test,
                    buffered_out_index=np.asarray([], dtype=int),
                )
            )
    return tuple(splits)


def _parent_component_labels(values: Sequence[object]) -> np.ndarray:
    """Return connected components for rows sharing any parent-assay token."""

    count = len(values)
    roots = np.arange(count, dtype=int)

    def find(index: int) -> int:
        while roots[index] != index:
            roots[index] = roots[roots[index]]
            index = int(roots[index])
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            roots[right_root] = left_root

    token_owner: dict[str, int] = {}
    for row_index, value in enumerate(values):
        raw = "" if value is None else str(value).strip()
        tokens = (
            []
            if not raw or raw.lower() in {"nan", "none", "<na>"}
            else [token.strip() for token in raw.split("|") if token.strip()]
        )
        if not tokens:
            tokens = [f"__row_{row_index}"]
        for token in tokens:
            if token in token_owner:
                union(row_index, token_owner[token])
            else:
                token_owner[token] = row_index

    return np.asarray([find(index) for index in range(count)], dtype=int)

def random_parent_splits(
    data: pd.DataFrame,
    *,
    n_splits: int = 5,
    parent_col: str = "parent_assay_ids",
    random_state: int = 20260728,
) -> tuple[ValidationSplit, ...]:
    """Random-row optimism comparator that still keeps each parent in one fold."""

    if parent_col not in data:
        raise ValueError(f"missing parent column {parent_col!r}")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    parent = _parent_component_labels(data[parent_col].to_numpy())
    unique = np.unique(parent)
    rng = np.random.default_rng(random_state)
    rng.shuffle(unique)
    splits: list[ValidationSplit] = []
    for number, test_parent in enumerate(np.array_split(unique, n_splits), start=1):
        test = np.flatnonzero(np.isin(parent, test_parent))
        train = np.flatnonzero(~np.isin(parent, test_parent))
        if len(train) and len(test):
            splits.append(
                ValidationSplit(
                    scheme="random_parent_optimism",
                    fold_id=str(number),
                    train_index=train,
                    test_index=test,
                    buffered_out_index=np.asarray([], dtype=int),
                )
            )
    return tuple(splits)


def grouped_inner_splits(
    data: pd.DataFrame,
    *,
    group_col: str = "BHID",
    n_splits: int = 5,
) -> tuple[ValidationSplit, ...]:
    """Deterministic balanced group folds for inner tuning."""

    if group_col not in data:
        raise ValueError(f"missing group column {group_col!r}")
    groups = data[group_col].astype(str).to_numpy()
    unique = np.asarray(sorted(pd.unique(groups)), dtype=object)
    if len(unique) < 2:
        raise ValueError("at least two groups are required for grouped tuning")
    fold_count = min(n_splits, len(unique))
    # Greedy assignment balances rows while keeping complete holes together.
    sizes = {group: int(np.sum(groups == group)) for group in unique}
    assignments: list[list[str]] = [[] for _ in range(fold_count)]
    loads = np.zeros(fold_count, dtype=int)
    for group in sorted(unique, key=lambda item: (-sizes[item], str(item))):
        fold = int(np.argmin(loads))
        assignments[fold].append(str(group))
        loads[fold] += sizes[group]
    result: list[ValidationSplit] = []
    for number, test_groups in enumerate(assignments, start=1):
        test = np.flatnonzero(np.isin(groups, test_groups))
        train = np.flatnonzero(~np.isin(groups, test_groups))
        if len(train) and len(test):
            result.append(
                ValidationSplit(
                    scheme="grouped_inner",
                    fold_id=str(number),
                    train_index=train,
                    test_index=test,
                    buffered_out_index=np.asarray([], dtype=int),
                )
            )
    return tuple(result)


def _parameter_grid(
    parameters: Mapping[str, Sequence[object]],
) -> tuple[dict[str, object], ...]:
    names = sorted(parameters)
    values = [parameters[name] for name in names]
    return tuple(dict(zip(names, combination)) for combination in product(*values))


def tune_grouped(
    data: pd.DataFrame,
    *,
    model_factory: Callable[[Mapping[str, object]], CompositeRegressor],
    parameter_grid: Mapping[str, Sequence[object]],
    target_col: str = "tgc_pct",
    group_col: str = "BHID",
    weight_col: str = "support_m",
    n_splits: int = 5,
    failure_penalty: float | None = None,
) -> TuningResult:
    """Tune only inside deterministic grouped folds."""

    splits = grouped_inner_splits(data, group_col=group_col, n_splits=n_splits)
    y = data[target_col].to_numpy(dtype=float)
    if failure_penalty is None:
        failure_penalty = max(float(np.nanstd(y)), np.finfo(float).eps)
    scored: list[tuple[Mapping[str, object], float]] = []
    for parameters in _parameter_grid(parameter_grid):
        fold_scores: list[float] = []
        for split in splits:
            train = data.iloc[split.train_index]
            test = data.iloc[split.test_index]
            model = model_factory(parameters)
            model.fit(train, target_col)
            prediction = model.predict(test)
            truth = test[target_col].to_numpy(dtype=float)
            weights = (
                test[weight_col].to_numpy(dtype=float)
                if weight_col in test
                else np.ones(len(test), dtype=float)
            )
            success = prediction.success & np.isfinite(prediction.mean)
            if np.any(success):
                mae = float(
                    np.average(
                        np.abs(prediction.mean[success] - truth[success]),
                        weights=weights[success],
                    )
                )
            else:
                mae = float(failure_penalty)
            score = mae + float(failure_penalty) * (1.0 - np.mean(success))
            fold_scores.append(score)
        scored.append((parameters, float(np.mean(fold_scores))))
    scored.sort(key=lambda item: (item[1], repr(sorted(item[0].items()))))
    return TuningResult(
        best_parameters=dict(scored[0][0]),
        best_score=float(scored[0][1]),
        candidate_scores=tuple((dict(params), score) for params, score in scored),
    )


def evaluate_model(
    data: pd.DataFrame,
    splits: Sequence[ValidationSplit],
    *,
    model_factory: Callable[[], CompositeRegressor],
    target_col: str = "tgc_pct",
    hole_col: str = "BHID",
    weight_col: str = "support_m",
) -> pd.DataFrame:
    """Fit a fresh model in every outer fold and return row-level predictions."""

    frames: list[pd.DataFrame] = []
    for split in splits:
        train = data.iloc[split.train_index]
        test = data.iloc[split.test_index]
        model = model_factory()
        model.fit(train, target_col)
        prediction = model.predict(test)
        frame = pd.DataFrame(
            {
                "row_index": test.index.to_numpy(),
                "scheme": split.scheme,
                "fold_id": split.fold_id,
                "truth": test[target_col].to_numpy(dtype=float),
                "prediction": prediction.mean,
                "variance": prediction.variance,
                "success": prediction.success,
                "hole": test[hole_col].astype(str).to_numpy(),
                "weight": (
                    test[weight_col].to_numpy(dtype=float)
                    if weight_col in test
                    else np.ones(len(test), dtype=float)
                ),
            }
        )
        frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=[
                "row_index",
                "scheme",
                "fold_id",
                "truth",
                "prediction",
                "variance",
                "success",
                "hole",
                "weight",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def prediction_metrics(
    truth: Sequence[float],
    prediction: Sequence[float],
    holes: Sequence[object],
    *,
    weights: Sequence[float] | None = None,
    variance: Sequence[float] | None = None,
    success: Sequence[bool] | None = None,
    confidence_level: float = 0.95,
) -> MetricSummary:
    """Compute hole-balanced and support-weighted validation metrics."""

    y = np.asarray(truth, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    hole = np.asarray(holes, dtype=object)
    if len(y) != len(pred) or len(y) != len(hole):
        raise ValueError("truth, prediction, and holes must have equal length")
    w = np.ones(len(y), dtype=float) if weights is None else np.asarray(weights, float)
    if len(w) != len(y) or not np.all(np.isfinite(w) & (w > 0)):
        raise ValueError("weights must be finite, positive, and match truth")
    ok = (
        np.ones(len(y), dtype=bool)
        if success is None
        else np.asarray(success, dtype=bool)
    )
    ok &= np.isfinite(y) & np.isfinite(pred)
    failure_rate = float(1.0 - np.mean(ok)) if len(ok) else float("nan")
    if not np.any(ok):
        return MetricSummary(
            *(float("nan"),) * 7,
            failure_rate=failure_rate,
            n_rows=len(y),
            n_holes=len(pd.unique(hole)),
        )

    error = pred - y
    per_hole: list[tuple[float, float, float]] = []
    for group in sorted(pd.unique(hole[ok]).astype(str)):
        use = ok & (hole.astype(str) == group)
        local_w = w[use]
        local_error = error[use]
        per_hole.append(
            (
                float(np.average(np.abs(local_error), weights=local_w)),
                float(np.sqrt(np.average(local_error**2, weights=local_w))),
                float(np.average(local_error, weights=local_w)),
            )
        )
    per_hole_array = np.asarray(per_hole)
    hb_mae, hb_rmse, hb_bias = np.mean(per_hole_array, axis=0)
    lw_mae = float(np.average(np.abs(error[ok]), weights=w[ok]))
    lw_rmse = float(np.sqrt(np.average(error[ok] ** 2, weights=w[ok])))
    lw_bias = float(np.average(error[ok], weights=w[ok]))

    coverage = float("nan")
    if variance is not None:
        var = np.asarray(variance, dtype=float)
        if len(var) != len(y):
            raise ValueError("variance must match truth")
        interval_ok = ok & np.isfinite(var) & (var >= 0)
        if np.any(interval_ok):
            z = float(norm.ppf(0.5 + confidence_level / 2.0))
            half_width = z * np.sqrt(var[interval_ok])
            coverage = float(
                np.mean(
                    (y[interval_ok] >= pred[interval_ok] - half_width)
                    & (y[interval_ok] <= pred[interval_ok] + half_width)
                )
            )

    return MetricSummary(
        hole_balanced_mae=float(hb_mae),
        hole_balanced_rmse=float(hb_rmse),
        hole_balanced_bias=float(hb_bias),
        length_weighted_mae=lw_mae,
        length_weighted_rmse=lw_rmse,
        length_weighted_bias=lw_bias,
        interval_coverage=coverage,
        failure_rate=failure_rate,
        n_rows=len(y),
        n_holes=len(per_hole),
    )


def summarize_prediction_frame(
    predictions: pd.DataFrame, confidence_level: float = 0.95
) -> MetricSummary:
    required = {
        "truth",
        "prediction",
        "hole",
        "weight",
        "variance",
        "success",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"prediction frame is missing {sorted(missing)}")
    return prediction_metrics(
        predictions["truth"],
        predictions["prediction"],
        predictions["hole"],
        weights=predictions["weight"],
        variance=predictions["variance"],
        success=predictions["success"],
        confidence_level=confidence_level,
    )


def paired_hole_bootstrap_mae(
    truth: Sequence[float],
    conditioned_prediction: Sequence[float],
    comparator_prediction: Sequence[float],
    holes: Sequence[object],
    *,
    weights: Sequence[float] | None = None,
    conditioned_success: Sequence[bool] | None = None,
    comparator_success: Sequence[bool] | None = None,
    replicates: int = 2000,
    confidence_level: float = 0.95,
    random_state: int = 20260728,
) -> PairedInterval:
    """Paired whole-hole bootstrap of conditioned minus comparator MAE."""

    if replicates < 1:
        raise ValueError("replicates must be positive")
    y = np.asarray(truth, dtype=float)
    first = np.asarray(conditioned_prediction, dtype=float)
    second = np.asarray(comparator_prediction, dtype=float)
    hole = np.asarray(holes, dtype=object).astype(str)
    w = np.ones(len(y), dtype=float) if weights is None else np.asarray(weights, float)
    ok_first = (
        np.ones(len(y), dtype=bool)
        if conditioned_success is None
        else np.asarray(conditioned_success, dtype=bool)
    )
    ok_second = (
        np.ones(len(y), dtype=bool)
        if comparator_success is None
        else np.asarray(comparator_success, dtype=bool)
    )
    if not (len(y) == len(first) == len(second) == len(hole) == len(w)):
        raise ValueError("all bootstrap inputs must have equal length")
    valid = (
        np.isfinite(y)
        & np.isfinite(first)
        & np.isfinite(second)
        & np.isfinite(w)
        & (w > 0)
        & ok_first
        & ok_second
    )
    differences: list[float] = []
    labels: list[str] = []
    for label in sorted(pd.unique(hole[valid])):
        use = valid & (hole == label)
        if np.any(use):
            first_mae = np.average(np.abs(first[use] - y[use]), weights=w[use])
            second_mae = np.average(np.abs(second[use] - y[use]), weights=w[use])
            differences.append(float(first_mae - second_mae))
            labels.append(str(label))
    if not differences:
        return PairedInterval(
            estimate=float("nan"),
            lower=float("nan"),
            upper=float("nan"),
            confidence_level=confidence_level,
            n_holes=0,
        )
    diff = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(random_state)
    draw = rng.integers(0, len(diff), size=(replicates, len(diff)))
    sampled = np.mean(diff[draw], axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(sampled, [alpha, 1.0 - alpha])
    return PairedInterval(
        estimate=float(np.mean(diff)),
        lower=float(lower),
        upper=float(upper),
        confidence_level=confidence_level,
        n_holes=len(labels),
    )


def decide_accept_pool_abstain(
    *,
    variogram_gate_passed: bool,
    scheme_evidence: Iterable[SchemeEvidence],
    gate_failure_reasons: Sequence[str] | None = None,
    maximum_coverage_loss: float = 0.0,
    maximum_failure_rate: float = 0.05,
    pooled_policy_available: bool = True,
) -> DecisionOutcome:
    """Apply the frozen accept/pool/abstain rule.

    Negative MAE differences favour the conditioned model.
    """

    evidence = tuple(scheme_evidence)
    if not variogram_gate_passed:
        return DecisionOutcome(
            action="abstain",
            selected_policy="regression_only",
            reasons=(
                tuple(str(reason) for reason in gate_failure_reasons)
                if gate_failure_reasons
                else ("combined kriging gate failed",)
            ),
        )
    if not evidence:
        return DecisionOutcome(
            action="abstain",
            selected_policy="regression_only",
            reasons=("no primary-scheme evidence supplied",),
        )
    reasons: list[str] = []
    for item in evidence:
        interval = item.mae_difference
        if not np.all(np.isfinite([interval.estimate, interval.lower, interval.upper])):
            reasons.append(f"{item.scheme}: paired MAE interval is unavailable")
        if item.conditioned_failure_rate > maximum_failure_rate:
            reasons.append(
                f"{item.scheme}: failure rate "
                f"{item.conditioned_failure_rate:.3g} exceeds "
                f"{maximum_failure_rate:.3g}"
            )
    if reasons:
        return DecisionOutcome(
            action="abstain",
            selected_policy="regression_only",
            reasons=tuple(reasons),
        )

    improves = all(item.mae_difference.upper < 0 for item in evidence)
    coverage_diagnostic = all(
        item.coverage_difference >= -maximum_coverage_loss for item in evidence
    )
    if improves:
        diagnostic_reason = (
            "nominal uncalibrated coverage did not degrade beyond the descriptive limit"
            if coverage_diagnostic
            else "nominal uncalibrated coverage degraded; retained as a secondary diagnostic only"
        )
        return DecisionOutcome(
            action="accept",
            selected_policy="geology_conditioned",
            reasons=(
                "paired hole-bootstrap MAE intervals favour conditioning in "
                "every primary scheme",
                diagnostic_reason,
            ),
        )

    if pooled_policy_available:
        reasons.append(
            "conditioning lacks paired MAE improvement in every primary scheme"
        )
        return DecisionOutcome(
            action="pool",
            selected_policy="pooled_geology",
            reasons=tuple(reasons),
        )
    return DecisionOutcome(
        action="abstain",
        selected_policy="regression_only",
        reasons=(
            "conditioning was not accepted and no predeclared pooled policy is available",
        ),
    )
