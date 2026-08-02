"""Additional post-analysis statistical evaluation for the active Tanga study.

This module is deliberately downstream of the frozen analysis.  It does not
replace the prospective threshold, alter the central abstention, fit an SGS,
or infer directional continuity.  Its outputs are diagnostic/reconciliation
records intended to make calibration, spatial support, and reporting limits
auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import binomtest

from .information_sensitivity import model_factory
from .models import GeologyRegressionRegressor
from .synthetic import (
    SyntheticBenchmarkConfig,
    _balanced_between_hole_pair_indices,
    generate_synthetic_matrix,
)
from .validation import ValidationSplit


SEED = 20260728
ROTATIONS_DEG = (0.0, 45.0, 90.0, 135.0)
ORIGIN_FRACTIONS = (-0.4, -0.2, 0.0, 0.2, 0.4)
BUFFER_MULTIPLIERS = (0.5, 1.0, 1.5)


@dataclass(frozen=True)
class PairScoreDesign:
    first: np.ndarray
    second: np.ndarray
    inverse: np.ndarray
    overall_counts: np.ndarray
    short_mask: np.ndarray
    short_counts: np.ndarray
    pair_count_requested: int
    pair_count_used: int
    eligible_hole_pairs: int
    short_lag_quantile: float
    random_state: int


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% only here)."""

    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("successes/trials are invalid")
    if not np.isclose(confidence, 0.95):
        raise ValueError("this audit fixes confidence at 0.95")
    z = 1.959963984540054
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return float(centre - half), float(centre + half)


def binomial_intervals(successes: int, trials: int) -> dict[str, float]:
    wilson_low, wilson_high = wilson_interval(successes, trials)
    exact = binomtest(successes, trials).proportion_ci(confidence_level=0.95, method="exact")
    return {
        "wilson_lower_95": wilson_low,
        "wilson_upper_95": wilson_high,
        "exact_lower_95": float(exact.low),
        "exact_upper_95": float(exact.high),
    }


def calibrate_higher_threshold(scores: Sequence[float], false_pass_rate: float = 0.05) -> float:
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise ValueError("at least two finite calibration scores are required")
    boundary = np.quantile(values, 1.0 - false_pass_rate, method="higher")
    return float(np.nextafter(boundary, np.inf))


def make_pair_score_design(
    coordinates: np.ndarray,
    holes: Sequence[object],
    domains: Sequence[object],
    *,
    pair_count: int = 20_000,
    short_lag_quantile: float = 0.20,
    random_state: int = SEED,
) -> PairScoreDesign:
    """Prepare the frozen same-domain, equal-between-hole-pair score design."""

    coords = np.asarray(coordinates, dtype=float)
    hole_array = np.asarray(holes, dtype=object).astype(str)
    domain_array = np.asarray(domains, dtype=object).astype(str)
    first, second = _balanced_between_hole_pair_indices(
        tuple(hole_array.tolist()), tuple(domain_array.tolist()), int(pair_count), int(random_state)
    )
    distance = np.linalg.norm(coords[first] - coords[second], axis=1)
    keep = np.isfinite(distance) & (distance > 0)
    first, second, distance = first[keep], second[keep], distance[keep]
    if len(first) < 10:
        raise ValueError("fewer than ten eligible pairs")
    keys = np.asarray(
        [f"{a}|{b}" if a <= b else f"{b}|{a}" for a, b in zip(hole_array[first], hole_array[second])],
        dtype=object,
    )
    _, inverse = np.unique(keys, return_inverse=True)
    group_count = int(np.max(inverse)) + 1
    overall_counts = np.bincount(inverse, minlength=group_count).astype(float)
    cutoff = np.quantile(distance, short_lag_quantile)
    short_mask = distance <= cutoff
    short_counts = np.bincount(inverse[short_mask], minlength=group_count).astype(float)
    return PairScoreDesign(
        first=first,
        second=second,
        inverse=inverse,
        overall_counts=overall_counts,
        short_mask=short_mask,
        short_counts=short_counts,
        pair_count_requested=int(pair_count),
        pair_count_used=len(first),
        eligible_hole_pairs=group_count,
        short_lag_quantile=float(short_lag_quantile),
        random_state=int(random_state),
    )


def vectorized_short_lag_scores(matrix: np.ndarray, design: PairScoreDesign) -> np.ndarray:
    """Score response columns exactly as the scalar balanced-pair implementation."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("matrix must be finite and two dimensional")
    semivariance = 0.5 * (values[design.first] - values[design.second]) ** 2
    n_groups = len(design.overall_counts)
    overall_sums = np.zeros((n_groups, values.shape[1]), dtype=float)
    np.add.at(overall_sums, design.inverse, semivariance)
    overall = np.mean(overall_sums / design.overall_counts[:, None], axis=0)
    short_inverse = design.inverse[design.short_mask]
    short_sums = np.zeros((n_groups, values.shape[1]), dtype=float)
    np.add.at(short_sums, short_inverse, semivariance[design.short_mask])
    supported = design.short_counts > 0
    short = np.mean(short_sums[supported] / design.short_counts[supported, None], axis=0)
    result = np.zeros(values.shape[1], dtype=float)
    positive = overall > np.finfo(float).eps
    result[positive] = 1.0 - short[positive] / overall[positive]
    return result


def vectorized_pair_universe_score_sensitivities(
    matrix: np.ndarray, design: PairScoreDesign
) -> dict[str, np.ndarray]:
    """Return legacy and matched-hole-pair short-lag score variants.

    The legacy variant reproduces the frozen implementation. The
    same-supported variant averages numerator and denominator over only hole
    pairs represented at short lags. The within-pair variant first calculates
    the short/overall ratio inside each supported hole pair and then weights
    those ratios equally. These diagnostics do not replace the frozen score.
    """

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("matrix must be finite and two dimensional")
    semivariance = 0.5 * (
        values[design.first] - values[design.second]
    ) ** 2
    n_groups = len(design.overall_counts)
    overall_sums = np.zeros((n_groups, values.shape[1]), dtype=float)
    np.add.at(overall_sums, design.inverse, semivariance)
    overall_by_pair = overall_sums / design.overall_counts[:, None]
    short_inverse = design.inverse[design.short_mask]
    short_sums = np.zeros((n_groups, values.shape[1]), dtype=float)
    np.add.at(
        short_sums,
        short_inverse,
        semivariance[design.short_mask],
    )
    supported = design.short_counts > 0
    if not np.any(supported):
        nan = np.full(values.shape[1], np.nan, dtype=float)
        return {
            "legacy_all_pair_denominator": nan.copy(),
            "same_supported_hole_pairs": nan.copy(),
            "within_hole_pair_ratio": nan.copy(),
        }
    short_by_pair = (
        short_sums[supported] / design.short_counts[supported, None]
    )
    legacy_overall = np.mean(overall_by_pair, axis=0)
    supported_overall = np.mean(overall_by_pair[supported], axis=0)
    supported_short = np.mean(short_by_pair, axis=0)

    legacy = np.zeros(values.shape[1], dtype=float)
    same_supported = np.zeros(values.shape[1], dtype=float)
    legacy_positive = legacy_overall > np.finfo(float).eps
    supported_positive = supported_overall > np.finfo(float).eps
    legacy[legacy_positive] = (
        1.0
        - supported_short[legacy_positive]
        / legacy_overall[legacy_positive]
    )
    same_supported[supported_positive] = (
        1.0
        - supported_short[supported_positive]
        / supported_overall[supported_positive]
    )

    pair_denominator = overall_by_pair[supported]
    valid_ratio = pair_denominator > np.finfo(float).eps
    ratios = np.full_like(short_by_pair, np.nan)
    ratios[valid_ratio] = (
        short_by_pair[valid_ratio] / pair_denominator[valid_ratio]
    )
    with np.errstate(invalid="ignore"):
        within_pair = 1.0 - np.nanmean(ratios, axis=0)
    return {
        "legacy_all_pair_denominator": legacy,
        "same_supported_hole_pairs": same_supported,
        "within_hole_pair_ratio": within_pair,
    }


def trend_residualizer() -> GeologyRegressionRegressor:
    return GeologyRegressionRegressor(
        categorical_cols=("canonical_lithology", "grsc_subtype", "weathering"),
        numeric_cols=("depth_within_hole_m", "hole_mean_depth_m", "mid_rl"),
        alpha=1.0,
    )


def simulate_score_stream(
    data: pd.DataFrame,
    *,
    scenario: str,
    n_simulations: int,
    random_seed: int,
    practical_range: float,
    structured_variance: float,
    nugget_variance: float,
    boundary_effect: float,
    score_designs: Mapping[str, PairScoreDesign],
    n_features: int = 64,
    batch_size: int = 500,
) -> dict[str, np.ndarray]:
    """Generate and score an independent stream in bounded-memory batches."""

    coordinates = data[["mid_easting", "mid_northing", "mid_rl"]].to_numpy(float)
    domains = data["canonical_lithology"].astype(str).to_numpy()
    outputs = {name: [] for name in score_designs}
    for batch_index, start in enumerate(range(0, int(n_simulations), int(batch_size))):
        count = min(int(batch_size), int(n_simulations) - start)
        config = SyntheticBenchmarkConfig(
            simulations_per_scenario=count,
            random_seed=int(random_seed + 100_003 * batch_index),
            n_features=int(n_features),
            practical_range=float(practical_range),
            structured_variance=float(structured_variance),
            nugget_variance=float(nugget_variance),
            boundary_effect=float(boundary_effect),
        )
        generated = generate_synthetic_matrix(
            coordinates,
            scenario=scenario,
            config=config,
            domains=domains if scenario == "hard_boundary" else None,
        )
        residuals = trend_residualizer().residualize_matrix(data, generated)
        for name, design in score_designs.items():
            outputs[name].append(vectorized_short_lag_scores(residuals, design))
    return {name: np.concatenate(parts) for name, parts in outputs.items()}


def evaluation_record(
    *,
    stream_role: str,
    scenario_label: str,
    effect_size_label: str,
    generator_scenario: str,
    scores: np.ndarray,
    threshold: float,
    frozen_threshold: float,
    frozen_observed_score: float,
    practical_range: float,
    median_nn: float,
    structured_variance: float,
    nugget_variance: float,
    boundary_effect: float,
    basis_count: int,
    design: PairScoreDesign,
) -> dict[str, object]:
    finite = np.asarray(scores, dtype=float)
    finite = finite[np.isfinite(finite)]
    passes = int(np.sum(finite >= threshold))
    return {
        "stream_role": stream_role,
        "basis_id": f"batched_independent_basis_mixture_{basis_count}",
        "lag_fraction": design.short_lag_quantile,
        "pair_sample_fraction": design.pair_count_requested / 20_000.0,
        "pair_count_requested": design.pair_count_requested,
        "pair_count_used": design.pair_count_used,
        "eligible_hole_pairs": design.eligible_hole_pairs,
        "scenario_label": scenario_label,
        "effect_size_label": effect_size_label,
        "generator_scenario": generator_scenario,
        "range_multiplier": practical_range / median_nn,
        "practical_range_m": practical_range,
        "structured_variance": structured_variance,
        "nugget_variance": nugget_variance,
        "boundary_effect": boundary_effect,
        "n_simulations": len(finite),
        "passes": passes,
        "pass_rate": passes / len(finite),
        **binomial_intervals(passes, len(finite)),
        "evaluation_threshold_source": "independent_reviewer_calibration_stream",
        "evaluation_threshold": threshold,
        "frozen_threshold": frozen_threshold,
        "frozen_observed_score": frozen_observed_score,
        "frozen_gate_decision": "abstain",
        "operating_qualification": "failed",
        "portability_claim": False,
    }


def independent_synthetic_evaluation(
    data: pd.DataFrame,
    gate: Mapping[str, object],
    *,
    calibration_count: int,
    null_evaluation_count: int,
    positive_count: int,
    batch_size: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Separate threshold calibration from independent null/positive evaluation."""

    coordinates = data[["mid_easting", "mid_northing", "mid_rl"]].to_numpy(float)
    holes = data["BHID"].astype(str).to_numpy()
    domains = data["canonical_lithology"].astype(str).to_numpy()
    median_nn = float(data["spatial_buffer_m"].median())
    base_design = make_pair_score_design(coordinates, holes, domains)
    base_map = {"primary": base_design}

    calibration_scores = simulate_score_stream(
        data,
        scenario="null",
        n_simulations=calibration_count,
        random_seed=31_000_001,
        practical_range=2.0 * median_nn,
        structured_variance=0.0,
        nugget_variance=0.25,
        boundary_effect=0.0,
        score_designs=base_map,
        batch_size=batch_size,
    )["primary"]
    threshold = calibrate_higher_threshold(calibration_scores)
    frozen_threshold = float(gate["synthetic"]["threshold"])
    frozen_observed = float(gate["actual_residual_short_lag_score"])

    records: list[dict[str, object]] = []
    records.append(
        evaluation_record(
            stream_role="threshold_calibration_only",
            scenario_label="independent_noise_null_calibration",
            effect_size_label="null",
            generator_scenario="null",
            scores=calibration_scores,
            threshold=threshold,
            frozen_threshold=frozen_threshold,
            frozen_observed_score=frozen_observed,
            practical_range=2.0 * median_nn,
            median_nn=median_nn,
            structured_variance=0.0,
            nugget_variance=0.25,
            boundary_effect=0.0,
            basis_count=max(1, int(np.ceil(calibration_count / batch_size))),
            design=base_design,
        )
    )
    null_scores = simulate_score_stream(
        data,
        scenario="null",
        n_simulations=null_evaluation_count,
        random_seed=41_000_001,
        practical_range=2.0 * median_nn,
        structured_variance=0.0,
        nugget_variance=0.25,
        boundary_effect=0.0,
        score_designs=base_map,
        batch_size=batch_size,
    )["primary"]
    records.append(
        evaluation_record(
            stream_role="independent_null_evaluation",
            scenario_label="independent_noise_null_evaluation",
            effect_size_label="null",
            generator_scenario="null",
            scores=null_scores,
            threshold=threshold,
            frozen_threshold=frozen_threshold,
            frozen_observed_score=frozen_observed,
            practical_range=2.0 * median_nn,
            median_nn=median_nn,
            structured_variance=0.0,
            nugget_variance=0.25,
            boundary_effect=0.0,
            basis_count=max(1, int(np.ceil(null_evaluation_count / batch_size))),
            design=base_design,
        )
    )

    # These are bounded observation-location benchmarks, not geological-contact
    # simulations.  The categorical contrast uses logged lithology labels only
    # to test a context discontinuity; it is not a folded-contact analogue.
    positive_specs = (
        ("within_context_continuity_strong_2nn", "strong", "stationary", 2.0, 1.0, 0.25, 0.0),
        ("within_context_continuity_moderate_1nn", "moderate", "stationary", 1.0, 0.5, 0.5, 0.0),
        ("categorical_context_contrast_strong_2nn", "strong", "hard_boundary", 2.0, 1.0, 0.25, 1.0),
        ("categorical_context_contrast_moderate_1nn", "moderate", "hard_boundary", 1.0, 0.5, 0.5, 0.5),
    )
    sensitivity_parts: list[pd.DataFrame] = []
    for spec_index, (label, effect, scenario, range_mult, structured, nugget, boundary) in enumerate(positive_specs):
        scores = simulate_score_stream(
            data,
            scenario=scenario,
            n_simulations=positive_count,
            random_seed=51_000_001 + 1_000_003 * spec_index,
            practical_range=range_mult * median_nn,
            structured_variance=structured,
            nugget_variance=nugget,
            boundary_effect=boundary,
            score_designs=base_map,
            batch_size=batch_size,
        )["primary"]
        records.append(
            evaluation_record(
                stream_role="independent_positive_evaluation",
                scenario_label=label,
                effect_size_label=effect,
                generator_scenario=scenario,
                scores=scores,
                threshold=threshold,
                frozen_threshold=frozen_threshold,
                frozen_observed_score=frozen_observed,
                practical_range=range_mult * median_nn,
                median_nn=median_nn,
                structured_variance=structured,
                nugget_variance=nugget,
                boundary_effect=boundary,
                basis_count=max(1, int(np.ceil(positive_count / batch_size))),
                design=base_design,
            )
        )

    # Basis/lag/pair-sampling sensitivity uses a smaller but independent bank.
    sensitivity_n_per_basis = 400 if positive_count >= 5_000 else max(200, positive_count // 10)
    lag_values = (0.10, 0.20, 0.30)
    pair_counts = (5_000, 10_000, 20_000)
    designs = {
        f"lag_{lag:.2f}_pairs_{pairs}": make_pair_score_design(
            coordinates,
            holes,
            domains,
            pair_count=pairs,
            short_lag_quantile=lag,
            random_state=SEED + pairs + int(lag * 1000),
        )
        for lag in lag_values
        for pairs in pair_counts
    }
    for basis_index in range(5):
        bank = simulate_score_stream(
            data,
            scenario="stationary",
            n_simulations=sensitivity_n_per_basis,
            random_seed=61_000_001 + 1_000_003 * basis_index,
            practical_range=2.0 * median_nn,
            structured_variance=1.0,
            nugget_variance=0.25,
            boundary_effect=0.0,
            score_designs=designs,
            batch_size=sensitivity_n_per_basis,
        )
        for name, scores in bank.items():
            design = designs[name]
            passes = int(np.sum(scores >= threshold))
            sensitivity_parts.append(
                pd.DataFrame(
                    [{
                        "basis_id": basis_index + 1,
                        "lag_fraction": design.short_lag_quantile,
                        "pair_sample_fraction": design.pair_count_requested / 20_000.0,
                        "pair_count_requested": design.pair_count_requested,
                        "pair_count_used": design.pair_count_used,
                        "eligible_hole_pairs": design.eligible_hole_pairs,
                        "scenario_label": "within_context_continuity_strong_2nn",
                        "n_simulations": len(scores),
                        "passes": passes,
                        "pass_rate": passes / len(scores),
                        **binomial_intervals(passes, len(scores)),
                        "evaluation_threshold": threshold,
                        "role": "basis_lag_pair_sampling_sensitivity_only",
                        "portability_claim": False,
                    }]
                )
            )
    summary = {
        "calibration_seed": 31_000_001,
        "evaluation_null_seed": 41_000_001,
        "positive_seed_start": 51_000_001,
        "calibration_count": calibration_count,
        "independent_null_evaluation_count": null_evaluation_count,
        "positive_count_per_scenario": positive_count,
        "independent_calibration_threshold": threshold,
        "frozen_threshold": frozen_threshold,
        "frozen_observed_score": frozen_observed,
        "frozen_decision_retained": "abstain",
        "operating_qualification": "failed",
        "transitional_contact_scenario_removed": True,
        "reason": "no defensible signed geological contact distance was supplied to the frozen benchmark",
        "portability_claim": False,
    }
    return pd.DataFrame(records), pd.concat(sensitivity_parts, ignore_index=True), summary


def _hole_centres(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.assign(BHID=data["BHID"].astype(str))
        .groupby("BHID", sort=True)[["mid_easting", "mid_northing"]]
        .median()
    )


def rotated_origin_splits(
    data: pd.DataFrame,
    *,
    rotation_deg: float,
    origin_fraction: float,
    buffer_m: float,
    n_blocks: int = 5,
) -> tuple[ValidationSplit, ...]:
    """Create five contiguous, grade-blind blocks on a rotated projection."""

    centres = _hole_centres(data)
    angle = np.deg2rad(rotation_deg)
    projection = (
        centres["mid_easting"].to_numpy(float) * np.cos(angle)
        + centres["mid_northing"].to_numpy(float) * np.sin(angle)
    )
    holes = centres.index.to_numpy(dtype=str)
    order = np.lexsort((holes, projection))
    n_holes = len(holes)
    nominal = np.arange(1, n_blocks) * n_holes / n_blocks
    shift = origin_fraction * n_holes / n_blocks
    cuts = np.rint(nominal + shift).astype(int)
    cuts = np.clip(cuts, 1, n_holes - 1)
    if np.any(np.diff(cuts) < 1):
        raise ValueError("origin shift collapsed a spatial block")
    ordered_labels = np.empty(n_holes, dtype=int)
    for block_index, positions in enumerate(np.split(order, cuts), start=1):
        ordered_labels[positions] = block_index

    distances = cdist(centres.to_numpy(float), centres.to_numpy(float))
    row_holes = data["BHID"].astype(str).to_numpy()
    splits: list[ValidationSplit] = []
    for block_index in range(1, n_blocks + 1):
        test_holes = holes[ordered_labels == block_index]
        test_mask = np.isin(holes, test_holes)
        close = np.any(distances[:, test_mask] <= buffer_m, axis=1)
        train_holes = holes[~close]
        buffered_holes = holes[close & ~test_mask]
        splits.append(
            ValidationSplit(
                scheme="rotated_origin_buffered_block",
                fold_id=f"B{block_index}",
                train_index=np.flatnonzero(np.isin(row_holes, train_holes)),
                test_index=np.flatnonzero(np.isin(row_holes, test_holes)),
                buffered_out_index=np.flatnonzero(np.isin(row_holes, buffered_holes)),
            )
        )
    return tuple(splits)


def _hole_balanced_mae(truth: np.ndarray, prediction: np.ndarray, holes: np.ndarray, weights: np.ndarray) -> float:
    frame = pd.DataFrame(
        {"truth": truth, "prediction": prediction, "hole": holes.astype(str), "weight": weights}
    )
    frame["weighted_abs_error"] = np.abs(frame["truth"] - frame["prediction"]) * frame["weight"]
    per_hole = frame.groupby("hole", sort=True).agg(
        error_mass=("weighted_abs_error", "sum"), support=("weight", "sum")
    )
    return float(np.mean(per_hole["error_mass"] / per_hole["support"]))


def descriptive_block_resampling(values: Sequence[float], *, replicates: int = 10_000, seed: int = SEED) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(int(replicates), len(array)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def spatial_robustness(
    data: pd.DataFrame,
    *,
    median_nn: float,
    rotations: Sequence[float] = ROTATIONS_DEG,
    origins: Sequence[float] = ORIGIN_FRACTIONS,
    buffers: Sequence[float] = BUFFER_MULTIPLIERS,
    block_resamples: int = 10_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for rotation in rotations:
        for origin_index, origin in enumerate(origins, start=1):
            for buffer_multiplier in buffers:
                buffer_m = float(buffer_multiplier) * float(median_nn)
                splits = rotated_origin_splits(
                    data,
                    rotation_deg=float(rotation),
                    origin_fraction=float(origin),
                    buffer_m=buffer_m,
                )
                for split in splits:
                    train = data.iloc[split.train_index]
                    test = data.iloc[split.test_index]
                    lith = model_factory("lithology_only")().fit(train, "tgc_pct").predict(test)
                    weather = model_factory("geology_only")().fit(train, "tgc_pct").predict(test)
                    truth = test["tgc_pct"].to_numpy(float)
                    holes = test["BHID"].astype(str).to_numpy()
                    weights = test["support_m"].to_numpy(float)
                    lith_mae = _hole_balanced_mae(truth, lith.mean, holes, weights)
                    weather_mae = _hole_balanced_mae(truth, weather.mean, holes, weights)
                    rows.append(
                        {
                            "rotation_deg": float(rotation),
                            "origin_index": origin_index,
                            "origin_fraction": float(origin),
                            "buffer_multiplier": float(buffer_multiplier),
                            "buffer_m": buffer_m,
                            "block_id": split.fold_id,
                            "test_interval_count": len(test),
                            "test_hole_count": int(test["BHID"].nunique()),
                            "training_interval_count": len(train),
                            "training_hole_count": int(train["BHID"].nunique()),
                            "buffered_out_interval_count": len(split.buffered_out_index),
                            "buffered_out_hole_count": int(data.iloc[split.buffered_out_index]["BHID"].nunique()),
                            "prediction_count": int(np.sum(lith.success & weather.success)),
                            "lithology_mae": lith_mae,
                            "weathering_mae": weather_mae,
                            "delta_mae_weathering_minus_lithology": weather_mae - lith_mae,
                            "sign_favours_weathering": bool(weather_mae < lith_mae),
                            "partition_basis": "grade_blind_rotated_hole_centre_projection_with_rank_shifted_contiguous_cuts",
                        }
                    )
    per_block = pd.DataFrame(rows)
    summaries: list[dict[str, object]] = []
    group_cols = ["rotation_deg", "origin_index", "origin_fraction", "buffer_multiplier", "buffer_m"]
    for key, group in per_block.groupby(group_cols, sort=True):
        deltas = group["delta_mae_weathering_minus_lithology"].to_numpy(float)
        leave_one_out = np.asarray([np.mean(np.delete(deltas, i)) for i in range(len(deltas))])
        low, high = descriptive_block_resampling(
            deltas,
            replicates=block_resamples,
            seed=SEED + int(key[0] * 10) + int(key[1] * 100) + int(key[3] * 1000),
        )
        summaries.append(
            {
                "rotation_deg": key[0],
                "origin_index": key[1],
                "origin_fraction": key[2],
                "buffer_multiplier": key[3],
                "buffer_m": key[4],
                "n_blocks": len(group),
                "n_effective_blocks": int(group["delta_mae_weathering_minus_lithology"].notna().sum()),
                "n_predictions": int(group["prediction_count"].sum()),
                "n_test_intervals": int(group["test_interval_count"].sum()),
                "n_test_holes": int(group["test_hole_count"].sum()),
                "equal_block_mean_delta": float(np.mean(deltas)),
                "median_block_delta": float(np.median(deltas)),
                "sign_concordance": float(np.mean(deltas < 0)),
                "leave_one_block_out_min": float(np.min(leave_one_out)),
                "leave_one_block_out_max": float(np.max(leave_one_out)),
                "block_resample_ci_lower_95": low,
                "block_resample_ci_upper_95": high,
                "block_resampling_replicates": int(block_resamples),
                "inference_label": "descriptive dependent-partition sensitivity; not a precise five-block confidence interval",
            }
        )
    return per_block, pd.DataFrame(summaries)


def cohort_reconciliation(data_dir: Path, analysis_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    assay = pd.read_csv(data_dir / "assay_clean.csv")
    fragments = pd.read_csv(data_dir / "assay_geology_fragments.csv")
    native = pd.read_csv(data_dir / "composites_native.csv")
    composites = pd.read_csv(data_dir / "composites_2m.csv")
    primary = pd.read_csv(analysis_dir / "primary_analysis_cohort.csv")
    eligible_assay_ids = set(
        assay.loc[assay["analysis_eligible"].astype(bool), "assay_id"].astype(str)
    )
    fragments = fragments.loc[
        fragments["assay_id"].astype(str).isin(eligible_assay_ids)
    ].copy()
    fragment_counts = fragments.groupby("assay_id", sort=False).size()
    boundary = fragments.groupby("assay_id", sort=False)[
        ["lithology_difference", "weathering_difference"]
    ].max()
    stages = [
        (1, "source_assays", "assay_clean.csv", len(assay), "assay_rows", "", 0, "supplied analytical rows", "reported"),
        (2, "analysis_eligible_assays", "assay_clean.csv", int(assay["analysis_eligible"].astype(bool).sum()), "assay_rows", "source_assays", -int((~assay["analysis_eligible"].astype(bool)).sum()), "quarantine exclusion", "derived_exact"),
        (3, "geology_fragments", "assay_geology_fragments.csv", len(fragments), "fragment_rows", "analysis_eligible_assays", len(fragments) - int(assay["analysis_eligible"].astype(bool).sum()), "split at authoritative geology interval boundaries", "derived_exact"),
        (4, "assays_with_multiple_fragments", "assay_geology_fragments.csv", int((fragment_counts > 1).sum()), "assay_ids", "analysis_eligible_assays", 0, "fragment multiplicity diagnostic", "derived_exact"),
        (5, "extra_fragment_rows", "assay_geology_fragments.csv", int((fragment_counts - 1).clip(lower=0).sum()), "fragment_rows", "analysis_eligible_assays", 0, "rows created beyond one per assay", "derived_exact"),
        (6, "assays_with_source_logged_context_difference", "assay_geology_fragments.csv", int((boundary["lithology_difference"].astype(bool) | boundary["weathering_difference"].astype(bool)).sum()), "assay_ids", "analysis_eligible_assays", 0, "source assay label differs from authoritative logged context; not necessarily a crossed boundary", "derived_exact"),
        (7, "native_geology_fragment_composites", "composites_native.csv", len(native), "composite_rows", "geology_fragments", len(native) - len(fragments), "native support is geology-fragmented, not necessarily one source assay", "derived_exact"),
        (8, "native_primary_spatial_eligible", "composites_native.csv", int(native["primary_spatial_eligible"].astype(bool).sum()), "composite_rows", "native_geology_fragment_composites", -int((~native["primary_spatial_eligible"].astype(bool)).sum()), "measured-survey support gate", "derived_exact"),
        (9, "all_2m_composites", "composites_2m.csv", len(composites), "composite_rows", "geology_fragments", len(composites) - len(fragments), "fixed-support construction without bridging geology/weathering boundaries", "derived_exact"),
        (10, "incomplete_2m_composites", "composites_2m.csv", int((~composites["support_complete"].astype(bool)).sum()), "composite_rows", "all_2m_composites", 0, "incomplete assayed coverage", "derived_exact"),
        (11, "complete_2m_composites", "composites_2m.csv", int(composites["support_complete"].astype(bool).sum()), "composite_rows", "all_2m_composites", -int((~composites["support_complete"].astype(bool)).sum()), "complete support only", "derived_exact"),
        (12, "complete_but_spatially_ineligible_2m", "composites_2m.csv", int((composites["support_complete"].astype(bool) & ~composites["primary_spatial_eligible"].astype(bool)).sum()), "composite_rows", "complete_2m_composites", 0, "survey support exclusion", "derived_exact"),
        (13, "final_primary_2m", "primary_analysis_cohort.csv", len(primary), "composite_rows", "complete_2m_composites", len(primary) - int(composites["support_complete"].astype(bool).sum()), "complete and primary spatial eligible", "derived_exact"),
        (14, "final_primary_holes", "primary_analysis_cohort.csv", int(primary["BHID"].nunique()), "holes", "final_primary_2m", 0, "unique holes in final primary cohort", "derived_exact"),
    ]
    reconciliation = pd.DataFrame(
        stages,
        columns=["stage_order", "stage", "source_file", "count", "unit", "parent_stage", "change_from_parent", "reason", "status"],
    )
    reconciliation["scope"] = "active Tanga study; exact row lineage"
    mass = pd.read_csv(data_dir / "composite_mass_balance.csv").copy()
    mass["support_conserved"] = mass["support_difference_m"].abs() <= 1e-8
    mass["tgc_mass_conserved"] = mass["tgc_mass_difference"].abs() <= 1e-8
    mass["tolerance"] = 1e-8
    mass["tgc_mass_unit"] = "percent_metres"
    return reconciliation, mass


def qaqc_tables(assay_clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = "restricted professional report, section 3.3.3 and sections 3.4.4-3.4.8, Tables 3.2-3.9/Figs 3.22-3.23"
    rows: list[dict[str, object]] = []
    def add(category: str, metric: str, value: object, unit: str, status: str = "reported", limitation: str = "aggregate report fact; row-level control data unavailable") -> None:
        rows.append({"category": category, "metric": metric, "value": value, "unit": unit, "reporting_status": status, "source": source, "limitation": limitation})
    add("all_controls", "count", 373, "samples")
    add("all_controls", "insertion_frequency", "1 in 10", "field samples")
    for category, count in (("crm", 93), ("blank", 94), ("coarse_duplicate", 93), ("pulp_duplicate", 93)):
        add(category, "count", count, "samples")
    for crm, count, grade, sd in (("GGC-08", 18, 0.39, 0.06), ("GGC-09", 19, 2.41, 0.27), ("GGC-11", 19, 4.977, 0.158), ("GGC-13", 19, 7.99, 0.57), ("GGC-14", 18, 9.23, 0.53)):
        add(crm, "certified_grade", grade, "TGC_pct")
        add(crm, "certified_standard_deviation", sd, "TGC_pct")
        add(crm, "count", count, "samples")
        add(crm, "action_limit_failures", 0, "samples")
        for metric in ("observed_mean", "observed_bias", "warning_count"):
            add(crm, metric, "not reported", "not reported", "not_reported")
    add("blank", "minimum", "<0.05", "TGC_pct")
    add("blank", "maximum", "<0.05", "TGC_pct")
    add("blank", "mean", "<0.05", "TGC_pct")
    add("blank", "median", "<0.05", "TGC_pct")
    add("coarse_duplicate", "reported_precision", 12.0, "plus_or_minus_pct")
    add("coarse_duplicate", "reported_correlation_lower_bound", 0.98, "correlation")
    add("pulp_duplicate", "reported_precision", 4.6, "plus_or_minus_pct")
    add("pulp_duplicate", "reported_correlation_lower_bound", 0.995, "correlation")
    for category in ("coarse_duplicate", "pulp_duplicate"):
        add(category, "precision_metric_definition", "not reported", "not reported", "not_reported")
    add("preparation", "crush_size", 2, "mm_max")
    add("preparation", "pulverisation_passing", 85, "pct_passing_75_micrometres")
    add("analysis", "method", "infrared combustion / LECO stated elsewhere in report", "text")
    add("analysis", "nominal_lower_reporting_limit", 0.05, "TGC_pct")
    for metric in ("reassay_count", "corrective_action_count", "batch_control_counts"):
        add("all_controls", metric, "not reported", "not reported", "not_reported")
    qaqc = pd.DataFrame(rows)

    assay = assay_clean.loc[assay_clean["analysis_eligible"].astype(bool)].copy()
    per_hole_batch = assay.groupby("BHID")["BATCH_NUMBER"].nunique()
    per_batch = assay.groupby("BATCH_NUMBER").agg(assay_rows=("assay_id", "size"), holes=("BHID", "nunique")).reset_index()
    context = per_batch.assign(
        all_holes_single_batch=bool((per_hole_batch == 1).all()),
        number_of_batches=int(assay["BATCH_NUMBER"].nunique()),
        batch_effect_identifiable_separately_from_hole=False,
        limitation="each hole occurs in one batch; batch and hole are confounded",
    )
    return qaqc, context


def model_specification() -> pd.DataFrame:
    rows = [
        ("outcome", "all regression/IDW/contact models", "raw TGC percentage; no response transform", "run_analysis.py: fixed model families"),
        ("global_mean", "baseline", "support-weighted training mean; weighted residual variance", "models.py: GlobalMeanRegressor"),
        ("lithology_only", "fixed post-analysis comparator", "one-hot canonical_lithology + grsc_subtype; Ridge alpha=1e-6; support weights", "information_sensitivity.py:model_factory"),
        ("geology_only", "fixed primary model", "one-hot canonical_lithology + grsc_subtype + weathering; Ridge alpha=1e-6; support weights", "run_analysis.py:fixed_factories"),
        ("partial_pooling", "penalized context model, not hierarchical partial pooling", "same geology categories plus depth_within_hole_m, hole_mean_depth_m, mid_rl; median imputation, standardization; Ridge alpha tuned in {0.1,1,10} within grouped inner folds", "run_analysis.py:partial_factory"),
        ("idw", "distance comparator", "power tuned in {1,2}; max_neighbors in {16,32}; radius=None; min_neighbors=1; no zero-distance/shared-parent leakage", "run_analysis.py:idw_factory; models.py:IDWRegressor"),
        ("contact_pooled", "contact-policy comparator", "global support-weighted mean", "run_analysis.py:contact policies"),
        ("contact_hard", "contact-policy sensitivity", "one-hot canonical_lithology; Ridge alpha=1e-6", "run_analysis.py:contact policies"),
        ("contact_soft_partial_pool", "contact-policy sensitivity", "one-hot canonical_lithology; Ridge alpha=10", "run_analysis.py:contact policies"),
        ("contact_transitional", "along-hole contact-policy sensitivity", "canonical_lithology + signed and absolute along-hole distance; Ridge alpha=1", "run_analysis.py:contact policies"),
        ("nominal_intervals", "uncalibrated diagnostic", "mean +/- 1.96*sqrt(variance); training residual MSE for regression/global models; local weighted dispersion for IDW", "models.py prediction methods; validation.py metrics"),
        ("diagnostic_kriging", "post-gate non-decision sensitivity only", "fold-local OK/RK; 10 lags; maxlag 0.5 diagonal; classical exponential; radius=1 fitted range; max/min neighbours=32/3", "config/study_config.json:post_gate_diagnostic_kriging"),
        ("simple_kriging", "not implemented", "no Simple Kriging model", "active code audit"),
        ("sgs_lva", "prohibited and not implemented", "no SGS, no no-domain SGS, no locally varying anisotropy", "AGENTS.md and active code audit"),
    ]
    return pd.DataFrame(rows, columns=["model_or_component", "role", "exact_implementation", "implementation_source"])


def _per_hole_deltas(predictions: pd.DataFrame, conditioned: str, comparator: str, scheme: str, model_col: str = "model") -> np.ndarray:
    left = predictions.loc[(predictions[model_col] == conditioned) & (predictions["scheme"] == scheme)].copy()
    right = predictions.loc[(predictions[model_col] == comparator) & (predictions["scheme"] == scheme)].copy()
    joined = left.merge(right, on=["row_index", "scheme", "fold_id"], suffixes=("_conditioned", "_comparator"), validate="one_to_one")
    joined = joined.loc[joined["success_conditioned"].astype(bool) & joined["success_comparator"].astype(bool)].copy()
    joined["conditioned_error_mass"] = np.abs(joined["truth_conditioned"] - joined["prediction_conditioned"]) * joined["weight_conditioned"]
    joined["comparator_error_mass"] = np.abs(joined["truth_conditioned"] - joined["prediction_comparator"]) * joined["weight_conditioned"]
    holes = joined.groupby("hole_conditioned", sort=True).agg(conditioned=("conditioned_error_mass", "sum"), comparator=("comparator_error_mass", "sum"), support=("weight_conditioned", "sum"))
    return ((holes["conditioned"] - holes["comparator"]) / holes["support"]).to_numpy(float)


def bootstrap_effect_interval(
    values: Sequence[float], *, replicates: int = 20_000, seed: int = SEED
) -> tuple[float, float, float]:
    """Return a descriptive two-sided percentile interval for a mean delta."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        array, size=(int(replicates), len(array)), replace=True
    ).mean(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return float(np.mean(array)), float(lower), float(upper)


def post_analysis_effect_intervals(
    post_predictions: pd.DataFrame, contact_predictions: pd.DataFrame
) -> pd.DataFrame:
    """Summarize post-analysis contrasts without directional p-values."""
    rows: list[dict[str, object]] = []
    comparisons = (
        ("secondary_model_increments", "lithology_minus_global_mean", "lithology_only", "global_mean"),
        ("secondary_model_increments", "weathering_minus_lithology", "geology_only", "lithology_only"),
        ("secondary_model_increments", "penalized_context_minus_geology", "partial_pooling", "geology_only"),
    )
    for family, hypothesis, conditioned, comparator in comparisons:
        for scheme in ("leave_one_hole_out", "northing_block_buffered"):
            deltas = _per_hole_deltas(
                post_predictions, conditioned, comparator, scheme
            )
            estimate, lower, upper = bootstrap_effect_interval(
                deltas, seed=SEED + len(rows) * 101
            )
            rows.append({
                "family": family,
                "hypothesis": hypothesis,
                "scheme": scheme,
                "conditioned_code_identifier": conditioned,
                "comparator_code_identifier": comparator,
                "estimate_mae_delta": estimate,
                "bootstrap_ci_low": lower,
                "bootstrap_ci_high": upper,
                "n_independent_hole_units": len(deltas),
                "bootstrap_replicates": 20_000,
            })
    if not contact_predictions.empty:
        contact = contact_predictions.rename(columns={"policy": "model"})
        labels = {
            "hard": "low_penalty_minus_pooled",
            "soft_partial_pool": "high_penalty_minus_pooled",
            "transitional": "distance_augmented_minus_pooled",
        }
        for policy, hypothesis in labels.items():
            deltas = _per_hole_deltas(
                contact, policy, "pooled", "leave_one_hole_out"
            )
            estimate, lower, upper = bootstrap_effect_interval(
                deltas, seed=SEED + len(rows) * 101
            )
            rows.append({
                "family": "secondary_contact_policy_family",
                "hypothesis": hypothesis,
                "scheme": "leave_one_hole_out",
                "conditioned_code_identifier": policy,
                "comparator_code_identifier": "pooled",
                "estimate_mae_delta": estimate,
                "bootstrap_ci_low": lower,
                "bootstrap_ci_high": upper,
                "n_independent_hole_units": len(deltas),
                "bootstrap_replicates": 20_000,
            })
    table = pd.DataFrame(rows)
    table["role"] = (
        "post-analysis descriptive effect-size interval; not preregistered"
    )
    table["limitation"] = (
        "percentile intervals resample held-out holes; five spatial blocks and "
        "alternative partitions remain dependent"
    )
    return table


def contact_balanced_sensitivity(
    primary: pd.DataFrame,
    contact_predictions: pd.DataFrame,
    *,
    window_m: float = 10.0,
    replicates: int = 20_000,
    seed: int = SEED,
) -> pd.DataFrame:
    """Post hoc held-hole sensitivity balancing sides, contacts, then holes."""
    contact_data = primary.loc[
        primary["signed_contact_distance_m"].notna()
    ].reset_index(drop=True)
    context = contact_data[[
        "BHID",
        "canonical_lithology",
        "nearest_qfr_grsc_contact_id",
        "abs_contact_distance_m",
    ]].copy()
    context["row_index"] = context.index
    low = contact_predictions.loc[
        contact_predictions["policy"].eq("hard")
    ].merge(context, on="row_index", how="left", validate="many_to_one")
    pooled = contact_predictions.loc[
        contact_predictions["policy"].eq("pooled")
    ]
    joined = low.merge(
        pooled,
        on=["row_index", "scheme", "fold_id"],
        suffixes=("_low", "_pooled"),
        validate="one_to_one",
    )
    joined = joined.loc[
        joined["success_low"].astype(bool)
        & joined["success_pooled"].astype(bool)
        & (joined["abs_contact_distance_m"] <= float(window_m))
    ].copy()
    joined["absolute_error_delta"] = (
        np.abs(joined["truth_low"] - joined["prediction_low"])
        - np.abs(joined["truth_low"] - joined["prediction_pooled"])
    )
    joined["weighted_delta"] = (
        joined["absolute_error_delta"] * joined["weight_low"]
    )
    side = joined.groupby(
        ["BHID", "nearest_qfr_grsc_contact_id", "canonical_lithology"],
        sort=True,
        observed=True,
    ).agg(
        weighted_delta=("weighted_delta", "sum"),
        support=("weight_low", "sum"),
        rows=("row_index", "size"),
    ).reset_index()
    side["side_delta"] = side["weighted_delta"] / side["support"]
    contact = side.groupby(
        ["BHID", "nearest_qfr_grsc_contact_id"], sort=True, observed=True
    ).agg(
        contact_delta=("side_delta", "mean"),
        represented_sides=("canonical_lithology", "nunique"),
        contact_side_units=("canonical_lithology", "size"),
    ).reset_index()
    two_sided_keys = set(
        map(
            tuple,
            contact.loc[
                contact["represented_sides"].eq(2),
                ["BHID", "nearest_qfr_grsc_contact_id"],
            ].to_numpy(),
        )
    )
    joined["two_sided_contact"] = [
        (hole, contact_id) in two_sided_keys
        for hole, contact_id in zip(
            joined["BHID"], joined["nearest_qfr_grsc_contact_id"]
        )
    ]
    rows: list[dict[str, object]] = []
    for cohort, require_two_sides in (
        ("all_represented_contacts", False),
        ("two_sided_contacts_only", True),
    ):
        selected_contacts = contact.loc[
            contact["represented_sides"].eq(2)
            if require_two_sides
            else np.ones(len(contact), dtype=bool)
        ].copy()
        hole_means = selected_contacts.groupby(
            "BHID", sort=True, observed=True
        )["contact_delta"].mean()
        estimate, lower, upper = bootstrap_effect_interval(
            hole_means.to_numpy(float), replicates=replicates,
            seed=seed + len(rows) * 409,
        )
        local = joined.loc[
            joined["two_sided_contact"]
            if require_two_sides
            else np.ones(len(joined), dtype=bool)
        ]
        local_sides = side.merge(
            selected_contacts[["BHID", "nearest_qfr_grsc_contact_id"]],
            on=["BHID", "nearest_qfr_grsc_contact_id"],
            how="inner",
            validate="many_to_one",
        )
        rows.append({
            "cohort": cohort,
            "window_m": float(window_m),
            "rows": int(len(local)),
            "holes": int(hole_means.size),
            "unique_contacts": int(len(selected_contacts)),
            "contact_side_units": int(len(local_sides)),
            "qfr_rows": int(local["canonical_lithology"].eq("qfr").sum()),
            "graphitic_schist_rows": int(
                local["canonical_lithology"].eq("graphitic_schist").sum()
            ),
            "low_penalty_minus_pooled_mae": estimate,
            "bootstrap_ci_low": lower,
            "bootstrap_ci_high": upper,
            "bootstrap_replicates": int(replicates),
            "weighting_hierarchy": (
                "support within contact side; equal sides within contact; "
                "equal contacts within hole; equal holes"
            ),
            "status": "reviewer_requested_post_analysis_sensitivity",
            "limitation": (
                "sparse logged QFR support; along-hole distance only; "
                "does not establish a hard contact or directional continuity"
            ),
        })
    return pd.DataFrame(rows)


def geological_contrast_preservation(
    primary: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    minimum_support_m: float = 4.0,
    observed_contrast_thresholds: Sequence[float] = (0.0, 0.25, 0.5, 1.0),
    replicates: int = 20_000,
    seed: int = SEED,
    directional_zero_tolerance_tgc_pct: float = 1e-12,
    models: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Audit whether held-out predictions preserve within-hole lithology contrasts.

    Each lithology mean is support weighted. Lithology-pair contrasts are weighted
    equally within a hole and holes are the independent resampling units. The
    analysis is an additional post-analysis evaluation and is not preregistered.
    """
    context = primary.reset_index(drop=True)[
        ["BHID", "canonical_lithology", "support_m"]
    ].copy()
    context["row_index"] = context.index
    data = predictions.merge(
        context, on="row_index", how="left", validate="many_to_one"
    )
    data = data.loc[
        data["success"].astype(bool)
        & data["canonical_lithology"].notna()
        & data["scheme"].isin(
            ["leave_one_hole_out", "northing_block_buffered"]
        )
    ].copy()
    data["analysis_weight"] = pd.to_numeric(
        data.get("weight", data["support_m"]), errors="coerce"
    )
    data["truth_mass"] = data["truth"] * data["analysis_weight"]
    data["prediction_mass"] = data["prediction"] * data["analysis_weight"]
    grouped = data.groupby(
        ["model", "scheme", "hole", "canonical_lithology"],
        sort=True, observed=True,
    ).agg(
        truth_mass=("truth_mass", "sum"),
        prediction_mass=("prediction_mass", "sum"),
        support_m=("analysis_weight", "sum"),
        rows=("row_index", "size"),
    ).reset_index()
    grouped = grouped.loc[
        grouped["support_m"].ge(float(minimum_support_m))
        & grouped["rows"].ge(2)
    ].copy()
    grouped["truth_mean"] = grouped["truth_mass"] / grouped["support_m"]
    grouped["prediction_mean"] = (
        grouped["prediction_mass"] / grouped["support_m"]
    )

    pair_rows: list[dict[str, object]] = []
    for (model, scheme, hole), local in grouped.groupby(
        ["model", "scheme", "hole"], sort=True, observed=True
    ):
        records = local.sort_values("canonical_lithology").to_dict("records")
        for i, left in enumerate(records):
            for right in records[i + 1:]:
                observed = float(right["truth_mean"] - left["truth_mean"])
                predicted = float(
                    right["prediction_mean"] - left["prediction_mean"]
                )
                predicted_for_sign = (
                    0.0
                    if abs(predicted) <= directional_zero_tolerance_tgc_pct
                    else predicted
                )
                pair_rows.append({
                    "model": model,
                    "scheme": scheme,
                    "hole": hole,
                    "lithology_pair": (
                        f"{left['canonical_lithology']} vs "
                        f"{right['canonical_lithology']}"
                    ),
                    "observed_contrast": observed,
                    "predicted_contrast": predicted,
                    "absolute_contrast_error": abs(predicted - observed),
                    "sign_agreement": float(
                        np.sign(predicted_for_sign) == np.sign(observed)
                    ),
                })
    pairs = pd.DataFrame(pair_rows)
    if pairs.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    models = tuple(models) if models is not None else (
        "global_mean", "lithology_only", "geology_only",
        "partial_pooling", "idw",
    )
    for threshold in observed_contrast_thresholds:
        selected = pairs.loc[
            pairs["observed_contrast"].abs().ge(float(threshold))
        ].copy()
        for scheme in ("leave_one_hole_out", "northing_block_buffered"):
            local_scheme = selected.loc[selected["scheme"].eq(scheme)]
            for model in models:
                local = local_scheme.loc[local_scheme["model"].eq(model)]
                hole_metrics = local.groupby("hole", sort=True).agg(
                    absolute_contrast_error=("absolute_contrast_error", "mean"),
                    sign_agreement=("sign_agreement", "mean"),
                    lithology_pairs=("lithology_pair", "size"),
                )
                if hole_metrics.empty:
                    continue
                error_est, error_low, error_high = bootstrap_effect_interval(
                    hole_metrics["absolute_contrast_error"].to_numpy(float),
                    replicates=replicates,
                    seed=seed + len(rows) * 101,
                )
                sign_est, sign_low, sign_high = bootstrap_effect_interval(
                    hole_metrics["sign_agreement"].to_numpy(float),
                    replicates=replicates,
                    seed=seed + len(rows) * 101 + 17,
                )
                rows.append({
                    "record_type": "model_summary",
                    "scheme": scheme,
                    "model_code_identifier": model,
                    "comparator_code_identifier": "",
                    "minimum_support_per_lithology_m": minimum_support_m,
                    "minimum_absolute_observed_contrast": threshold,
                    "holes": int(len(hole_metrics)),
                    "lithology_pair_units": int(
                        hole_metrics["lithology_pairs"].sum()
                    ),
                    "mean_absolute_contrast_error": error_est,
                    "absolute_contrast_error_ci_low": error_low,
                    "absolute_contrast_error_ci_high": error_high,
                    "mean_sign_agreement": sign_est,
                    "sign_agreement_ci_low": sign_low,
                    "sign_agreement_ci_high": sign_high,
                    "paired_error_delta": np.nan,
                    "paired_error_delta_ci_low": np.nan,
                    "paired_error_delta_ci_high": np.nan,
                })
            for comparator in ("global_mean", "idw"):
                conditioned = local_scheme.loc[
                    local_scheme["model"].eq("lithology_only")
                ][["hole", "lithology_pair", "absolute_contrast_error"]]
                baseline = local_scheme.loc[
                    local_scheme["model"].eq(comparator)
                ][["hole", "lithology_pair", "absolute_contrast_error"]]
                joined = conditioned.merge(
                    baseline,
                    on=["hole", "lithology_pair"],
                    suffixes=("_lithology", "_comparator"),
                    validate="one_to_one",
                )
                if joined.empty:
                    continue
                joined["delta"] = (
                    joined["absolute_contrast_error_lithology"]
                    - joined["absolute_contrast_error_comparator"]
                )
                hole_delta = joined.groupby("hole", sort=True)["delta"].mean()
                estimate, lower, upper = bootstrap_effect_interval(
                    hole_delta.to_numpy(float), replicates=replicates,
                    seed=seed + len(rows) * 101 + 31,
                )
                rows.append({
                    "record_type": "paired_comparison",
                    "scheme": scheme,
                    "model_code_identifier": "lithology_only",
                    "comparator_code_identifier": comparator,
                    "minimum_support_per_lithology_m": minimum_support_m,
                    "minimum_absolute_observed_contrast": threshold,
                    "holes": int(len(hole_delta)),
                    "lithology_pair_units": int(len(joined)),
                    "mean_absolute_contrast_error": np.nan,
                    "absolute_contrast_error_ci_low": np.nan,
                    "absolute_contrast_error_ci_high": np.nan,
                    "mean_sign_agreement": np.nan,
                    "sign_agreement_ci_low": np.nan,
                    "sign_agreement_ci_high": np.nan,
                    "paired_error_delta": estimate,
                    "paired_error_delta_ci_low": lower,
                    "paired_error_delta_ci_high": upper,
                })
    output = pd.DataFrame(rows)
    output["weighting_hierarchy"] = (
        "support within lithology; equal lithology pairs within hole; equal holes"
    )
    output["role"] = (
        "additional post-analysis geology-informed descriptive validation"
    )
    output["directional_zero_tolerance_tgc_pct"] = (
        directional_zero_tolerance_tgc_pct
    )
    output["limitation"] = (
        "within-hole logged-lithology contrasts; no hard-boundary, directional-continuity, or causal claim"
    )
    return output


def decision_hierarchy() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (1, "central", "RQ5 accept/pool/abstain gate", "abstain retained", "conjunctive fail-safe; not altered by reviewer revision"),
            (2, "secondary", "RQ2 geological conditioning and RQ3 contact policies", "descriptive/secondary", "effect sizes use post-analysis two-sided hole-resampling intervals"),
            (3, "exploratory", "RQ1 support checks; RQ4 components; post-gate kriging; RQ6 sparse evidence", "diagnostic/exploratory", "cannot establish local prediction"),
        ],
        columns=["rank", "tier", "decision_family", "status", "limitation"],
    )


def parameter_ledger(
    *, calibration_count: int, null_evaluation_count: int, positive_count: int, median_nn: float
) -> pd.DataFrame:
    rows = [
        ("null_threshold_calibration_stream", calibration_count, "Monte Carlo precision at a nominal 0.05 false-pass target", "independent threshold-calibration stream plus Wilson/exact intervals", "the frozen 500-score stream reused for evaluation", "high for Monte Carlo precision", "synthetic design validity and portability remain unproven"),
        ("independent_null_evaluation_stream", null_evaluation_count, "separate evaluation is needed because the frozen null rate was calculated on its calibration set", "independent-seed false-pass rate with Wilson/exact intervals", "reuse of calibration scores", "high for Monte Carlo precision", "does not validate other deposits or alternative sampling geometries"),
        ("positive_evaluation_per_scenario", positive_count, "effect-size uncertainty rather than a single strong scenario", "four geologically bounded observation-location scenarios with binomial intervals", "500 per scenario; false transitional-contact analogue", "moderate", "scenario labels are bounded benchmarks, not geological replicas"),
        ("random_feature_basis_sensitivity", "5 bases x 400 fields" if positive_count >= 5000 else "5 bases x >=200 fields", "detection must not depend on one Fourier basis", "between-basis detection spread", "one basis only", "moderate", "basis family remains synthetic"),
        ("short_lag_quantile_sensitivity", "0.10, 0.20, 0.30", "short-lag score has no unique geological scale", "lag-fraction detection sensitivity", "0.20 only", "moderate", "quantiles are pair-distribution fractions, not variogram ranges"),
        ("pair_sampling_sensitivity", "5,000; 10,000; 20,000", "same-domain equal-hole-pair Monte Carlo support", "detection sensitivity across 0.25x, 0.5x, 1x frozen pair count", "one pair sample only", "moderate", "pair samples share holes and are not independent data"),
        ("spatial_projection_rotations", "0, 45, 90, 135 degrees", "four grade-blind axes test dependence on the northing-only partition", "equal-block delta and sign concordance", "northing-only partition", "moderate", "rotations are validation partitions, not anisotropy axes"),
        ("spatial_partition_origins", "-0.4, -0.2, 0, 0.2, 0.4 block widths", "boundary-placement sensitivity without grade use", "leave-one-block-out and origin spread", "one origin", "moderate", "rank-shifted blocks are dependent partitions"),
        ("spatial_buffers_m", f"{0.5*median_nn:.6f}; {median_nn:.6f}; {1.5*median_nn:.6f}", f"0.5x, 1x, 1.5x the frozen median nearest-hole distance ({median_nn:.6f} m)", "training/buffered/test interval and hole counts", "zero buffer or one buffer only", "moderate", "spacing-derived buffers have no economic or geological-continuity meaning"),
        ("descriptive_block_resamples", 10_000, "stabilize a descriptive five-value resampling summary", "equal-block resampling plus leave-one-block-out", "hole bootstrap presented as block inference", "low for inference; high for reproducibility", "five dependent blocks cannot support a precise inferential claim"),
        ("descriptive_effect_interval_replicates", 20_000, "post-analysis contrasts require uncertainty without directional testing", "two-sided percentile hole-resampling intervals", "directional tests and post hoc multiplicity claims", "moderate", "descriptive intervals are not retroactive preregistration"),
    ]
    return pd.DataFrame(rows, columns=["parameter", "number", "geology_or_data_evidence", "diagnostic_used", "rejected_alternatives", "confidence", "limitation"])

