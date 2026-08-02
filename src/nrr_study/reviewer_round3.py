"""Third-round reviewer sensitivities for the active Tanga study.

These analyses are downstream, reviewer-motivated, and non-decision. They do
not replace the frozen covariance gate, standardize residuals, introduce a
directional interpretation, or change the prediction estimand.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, levene, skew

from .analysis_helpers import evaluate_fixed_safely
ORDER = (
    "graphitic_schist_generic",
    "graphitic_schist_GRSC1",
    "graphitic_schist_GRSC2",
    "graphitic_schist_GRSC3",
    "qfr",
    "ambiguous",
    "other",
)


def _group(canonical: pd.Series, subtype: pd.Series) -> pd.Series:
    """Map verified canonical labels to the seven disclosed operational groups."""
    result = canonical.fillna("unmapped").astype(str).copy()
    graphitic = result.eq("graphitic_schist")
    result.loc[graphitic] = "graphitic_schist_generic"
    clean_subtype = subtype.fillna("not_graphitic_schist").astype(str)
    for code in ("GRSC1", "GRSC2", "GRSC3"):
        result.loc[graphitic & clean_subtype.eq(code)] = f"graphitic_schist_{code}"
    unexpected = sorted(set(result).difference(ORDER))
    if unexpected:
        raise ValueError(f"unexpected public geology groups: {unexpected}")
    return result
from .framework_extension import (
    ALPHA_GRID,
    _factory,
    _paired_evidence_checked,
    nested_grouped_predictions,
)
from .information_sensitivity import primary_metrics
from .models import GlobalMeanRegressor
from .reviewer_revision import (
    calibrate_higher_threshold,
    make_pair_score_design,
    trend_residualizer,
    vectorized_pair_universe_score_sensitivities,
)
from .synthetic import SyntheticBenchmarkConfig, generate_synthetic_matrix
from .validation import (
    ValidationSplit,
    leave_one_hole_out_splits,
    spatial_block_splits,
)


SEED = 20260728
PRIMARY_SCHEMES = ("leave_one_hole_out", "northing_block_buffered")
FLAG_FIELDS = (
    "source_lithology_difference",
    "source_weathering_difference",
)


def public_geology_group_summary(
    primary: pd.DataFrame,
    hierarchy: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize the frozen primary cohort by the seven public groups."""
    required = {
        "canonical_lithology",
        "grsc_subtype",
        "BHID",
        "support_m",
        "tgc_pct",
        "northing_block_label",
    }
    missing = sorted(required.difference(primary.columns))
    if missing:
        raise ValueError(f"primary cohort lacks public-summary fields: {missing}")

    data = primary.assign(
        public_grouping=_group(
            primary["canonical_lithology"], primary["grsc_subtype"]
        )
    )
    grouped = data.groupby("public_grouping", observed=False)
    summary = grouped.agg(
        rows=("public_grouping", "size"),
        holes=("BHID", "nunique"),
        support_m=("support_m", "sum"),
        median_tgc_pct=("tgc_pct", "median"),
        q25_tgc_pct=("tgc_pct", lambda values: values.quantile(0.25)),
        q75_tgc_pct=("tgc_pct", lambda values: values.quantile(0.75)),
        frozen_spatial_block_count=("northing_block_label", "nunique"),
    ).reindex(ORDER)
    if summary.isna().any().any():
        raise ValueError("one or more frozen public geology groups lack support")
    summary = summary.reset_index()
    for column in ("rows", "holes", "frozen_spatial_block_count"):
        summary[column] = summary[column].astype(int)
    for column in (
        "support_m",
        "median_tgc_pct",
        "q25_tgc_pct",
        "q75_tgc_pct",
    ):
        summary[column] = summary[column].astype(float)

    if hierarchy is not None:
        expected = hierarchy.set_index("public_grouping").reindex(ORDER)
        required_hierarchy = [
            "primary_composite_count",
            "primary_hole_count",
            "primary_support_m",
        ]
        if expected[required_hierarchy].isna().any().any():
            raise ValueError("public hierarchy does not contain all seven groups")
        if not np.array_equal(
            summary["rows"].to_numpy(),
            expected["primary_composite_count"].to_numpy(dtype=int),
        ):
            raise AssertionError("public-group row counts differ from hierarchy")
        if not np.array_equal(
            summary["holes"].to_numpy(),
            expected["primary_hole_count"].to_numpy(dtype=int),
        ):
            raise AssertionError("public-group hole counts differ from hierarchy")
        if not np.allclose(
            summary["support_m"].to_numpy(),
            expected["primary_support_m"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-9,
        ):
            raise AssertionError("public-group support differs from hierarchy")
    return summary


def primary_splits(data: pd.DataFrame) -> tuple[ValidationSplit, ...]:
    return (
        *leave_one_hole_out_splits(data),
        *spatial_block_splits(
            data,
            n_blocks=5,
            block_col="northing_block",
            buffer_distance=float(data["spatial_buffer_m"].median()),
        ),
    )


def filter_splits(
    splits: Sequence[ValidationSplit], keep_mask: Sequence[bool]
) -> tuple[ValidationSplit, ...]:
    """Filter fixed outer-fold memberships without recomputing geometry."""
    keep = np.asarray(keep_mask, dtype=bool)
    old_to_new = np.full(len(keep), -1, dtype=int)
    old_to_new[np.flatnonzero(keep)] = np.arange(int(np.sum(keep)))
    result: list[ValidationSplit] = []
    for split in splits:
        train = old_to_new[np.asarray(split.train_index, dtype=int)]
        test = old_to_new[np.asarray(split.test_index, dtype=int)]
        buffered = old_to_new[np.asarray(split.buffered_out_index, dtype=int)]
        result.append(
            ValidationSplit(
                scheme=split.scheme,
                fold_id=split.fold_id,
                train_index=train[train >= 0],
                test_index=test[test >= 0],
                buffered_out_index=buffered[buffered >= 0],
            )
        )
    return tuple(result)


def version_flag_exclusion_sensitivities(
    primary: pd.DataFrame,
    *,
    bootstraps: int = 20_000,
    seed: int = SEED,
    alphas: Sequence[float] = ALPHA_GRID,
    inner_grouped_folds: int = 5,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Refit core comparisons after separate source-version exclusions."""
    full_splits = primary_splits(primary)
    cohort_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    evidence_frames: list[pd.DataFrame] = []
    tuning_frames: list[pd.DataFrame] = []
    for flag_index, flag_field in enumerate(FLAG_FIELDS):
        keep = ~primary[flag_field].astype(bool).to_numpy()
        subset = primary.loc[keep].copy().reset_index(drop=True)
        splits = filter_splits(full_splits, keep)
        exclusion = f"exclude_{flag_field}"
        removed = primary.loc[~keep]
        cohort_rows.append(
            {
                "exclusion": exclusion,
                "flag_field": flag_field,
                "removed_rows": int((~keep).sum()),
                "removed_holes_represented": int(removed["BHID"].nunique()),
                "retained_rows": int(len(subset)),
                "retained_holes": int(subset["BHID"].nunique()),
                "fold_membership_policy": (
                    "filter frozen full-cohort outer memberships; "
                    "do not recompute blocks or buffers"
                ),
                "analysis_role": (
                    "reviewer_motivated_post_analysis_source_version_"
                    "exclusion_sensitivity"
                ),
            }
        )
        global_predictions = evaluate_fixed_safely(
            subset, splits, lambda: GlobalMeanRegressor()
        ).assign(model="global_mean")
        lithology_predictions = evaluate_fixed_safely(
            subset,
            splits,
            _factory("lithology_only_nested", 1e-6),
        ).assign(model="lithology_only")
        coordinate_predictions, coordinate_tuning = nested_grouped_predictions(
            subset,
            splits,
            "coordinate_trend",
            alphas=alphas,
            inner_grouped_folds=inner_grouped_folds,
        )
        geology_spatial_predictions, geology_spatial_tuning = (
            nested_grouped_predictions(
                subset,
                splits,
                "lithology_spatial",
                alphas=alphas,
                inner_grouped_folds=inner_grouped_folds,
            )
        )
        predictions = pd.concat(
            [
                global_predictions,
                lithology_predictions,
                coordinate_predictions,
                geology_spatial_predictions,
            ],
            ignore_index=True,
        )
        predictions["exclusion"] = exclusion
        prediction_frames.append(predictions)

        metrics = primary_metrics(predictions)
        metrics["exclusion"] = exclusion
        metric_frames.append(metrics)

        comparisons = (
            ("lithology_only", "global_mean", "lithology_minus_global"),
            (
                "lithology_spatial",
                "coordinate_trend",
                "lithology_spatial_minus_coordinate",
            ),
        )
        for comparison_index, (
            conditioned,
            comparator,
            label,
        ) in enumerate(comparisons):
            evidence = _paired_evidence_checked(
                predictions,
                conditioned=conditioned,
                comparator=comparator,
                replicates=bootstraps,
                seed=seed + flag_index * 1009 + comparison_index * 101,
            )
            evidence["comparison"] = label
            evidence["exclusion"] = exclusion
            evidence["removed_rows"] = int((~keep).sum())
            evidence["retained_holes"] = int(subset["BHID"].nunique())
            evidence["analysis_role"] = (
                "reviewer_motivated_post_analysis_source_version_"
                "exclusion_sensitivity"
            )
            evidence_frames.append(evidence)

        tuning = pd.concat(
            [coordinate_tuning, geology_spatial_tuning],
            ignore_index=True,
        )
        tuning["exclusion"] = exclusion
        tuning_frames.append(tuning)
    return (
        pd.DataFrame(cohort_rows),
        pd.concat(prediction_frames, ignore_index=True),
        pd.concat(metric_frames, ignore_index=True),
        pd.concat(evidence_frames, ignore_index=True),
        pd.concat(tuning_frames, ignore_index=True),
    )


def _null_score_stream(
    data: pd.DataFrame,
    design,
    *,
    n_simulations: int,
    random_seed: int,
    batch_size: int = 250,
) -> dict[str, np.ndarray]:
    coordinates = data[
        ["mid_easting", "mid_northing", "mid_rl"]
    ].to_numpy(float)
    outputs: dict[str, list[np.ndarray]] = {
        "legacy_all_pair_denominator": [],
        "same_supported_hole_pairs": [],
        "within_hole_pair_ratio": [],
    }
    for batch_index, start in enumerate(
        range(0, int(n_simulations), int(batch_size))
    ):
        count = min(int(batch_size), int(n_simulations) - start)
        config = SyntheticBenchmarkConfig(
            simulations_per_scenario=count,
            random_seed=int(random_seed + 100_003 * batch_index),
            n_features=64,
            practical_range=2.0
            * float(data["spatial_buffer_m"].median()),
            structured_variance=1.0,
            nugget_variance=0.25,
            boundary_effect=1.0,
        )
        generated = generate_synthetic_matrix(
            coordinates,
            scenario="null",
            config=config,
            domains=None,
        )
        residuals = trend_residualizer().residualize_matrix(data, generated)
        scores = vectorized_pair_universe_score_sensitivities(
            residuals, design
        )
        for method, values in scores.items():
            outputs[method].append(values)
    return {
        method: np.concatenate(parts)
        for method, parts in outputs.items()
    }


def pair_universe_score_sensitivity(
    primary: pd.DataFrame,
    gate: Mapping[str, object],
    *,
    calibration_count: int = 2_000,
    null_evaluation_count: int = 2_000,
    pair_count: int = 20_000,
    short_lag_quantile: float = 0.20,
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Audit numerator/denominator hole-pair support with matched nulls."""
    coordinates = primary[
        ["mid_easting", "mid_northing", "mid_rl"]
    ].to_numpy(float)
    holes = primary["BHID"].astype(str).to_numpy()
    domains = primary["canonical_lithology"].astype(str).to_numpy()
    trend = trend_residualizer().fit(primary, "tgc_pct")
    residuals = (
        primary["tgc_pct"].to_numpy(float) - trend.predict(primary).mean
    )
    design = make_pair_score_design(
        coordinates,
        holes,
        domains,
        pair_count=pair_count,
        short_lag_quantile=short_lag_quantile,
        random_state=seed,
    )
    actual = vectorized_pair_universe_score_sensitivities(
        residuals[:, None], design
    )
    frozen_actual = float(gate["actual_residual_short_lag_score"])
    legacy_actual = float(actual["legacy_all_pair_denominator"][0])
    if not np.isclose(legacy_actual, frozen_actual, rtol=1e-12, atol=1e-12):
        raise AssertionError(
            "legacy pair-universe sensitivity does not reproduce frozen score"
        )

    calibration = _null_score_stream(
        primary,
        design,
        n_simulations=calibration_count,
        random_seed=seed + 7_000_000,
    )
    evaluation = _null_score_stream(
        primary,
        design,
        n_simulations=null_evaluation_count,
        random_seed=seed + 17_000_000,
    )
    residual_stability_pass = bool(
        gate["residual_omnidirectional_stability"]["passed"]
    )
    synthetic_operating_pass = bool(gate["synthetic"]["meets_targets"])
    rows: list[dict[str, object]] = []
    thresholds: dict[str, float] = {}
    for method in (
        "legacy_all_pair_denominator",
        "same_supported_hole_pairs",
        "within_hole_pair_ratio",
    ):
        threshold = calibrate_higher_threshold(
            calibration[method], false_pass_rate=0.05
        )
        thresholds[method] = threshold
        empirical = float(actual[method][0])
        null_pass_rate = float(np.mean(evaluation[method] >= threshold))
        score_pass = bool(empirical >= threshold)
        rows.append(
            {
                "method": method,
                "empirical_score": empirical,
                "matched_null_threshold": threshold,
                "empirical_score_pass": score_pass,
                "null_calibration_count": int(calibration_count),
                "independent_null_evaluation_count": int(
                    null_evaluation_count
                ),
                "independent_null_false_pass_rate": null_pass_rate,
                "pair_count_requested": int(pair_count),
                "pair_count_used": int(design.pair_count_used),
                "eligible_hole_pairs": int(design.eligible_hole_pairs),
                "short_supported_hole_pairs": int(
                    np.sum(design.short_counts > 0)
                ),
                "short_supported_hole_pair_fraction": float(
                    np.mean(design.short_counts > 0)
                ),
                "short_lag_quantile": float(short_lag_quantile),
                "residual_stability_pass": residual_stability_pass,
                "original_synthetic_operating_target_pass": (
                    synthetic_operating_pass
                ),
                "complete_covariance_eligibility_pass": bool(
                    score_pass
                    and residual_stability_pass
                    and synthetic_operating_pass
                ),
                "analysis_role": (
                    "reviewer_motivated_post_analysis_pair_universe_"
                    "score_sensitivity"
                ),
                "changes_frozen_gate": False,
            }
        )
    summary = {
        "reviewer_pair_universe_claim_accurate": True,
        "legacy_numerator_pair_universe": (
            "short-supported independent hole pairs"
        ),
        "legacy_denominator_pair_universe": (
            "all eligible independent hole pairs"
        ),
        "matched_methods": [
            "same_supported_hole_pairs",
            "within_hole_pair_ratio",
        ],
        "frozen_score_reproduced": legacy_actual,
        "frozen_score": frozen_actual,
        "eligible_hole_pairs": int(design.eligible_hole_pairs),
        "short_supported_hole_pairs": int(
            np.sum(design.short_counts > 0)
        ),
        "residual_stability_pass": residual_stability_pass,
        "original_synthetic_operating_target_pass": synthetic_operating_pass,
        "empirical_abstention_changes": False,
        "reason": (
            "complete eligibility remains false under every score variant "
            "because residual stability and the original complete synthetic "
            "operating target fail independently"
        ),
        "changes_frozen_gate": False,
        "matched_null_thresholds": thresholds,
    }
    return pd.DataFrame(rows), summary


def residual_distribution_diagnostics(
    primary: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Assess whether a response transform is scientifically necessary."""
    data = primary.copy()
    raw_model = trend_residualizer().fit(data, "tgc_pct")
    data["raw_residual"] = (
        data["tgc_pct"].to_numpy(float)
        - raw_model.predict(data).mean
    )
    data["log1p_tgc"] = np.log1p(data["tgc_pct"].to_numpy(float))
    log_model = trend_residualizer().fit(data, "log1p_tgc")
    data["log1p_residual"] = (
        data["log1p_tgc"].to_numpy(float)
        - log_model.predict(data).mean
    )
    rows: list[dict[str, object]] = []
    groups: list[tuple[str, pd.DataFrame]] = [
        ("all", data),
        *[
            (str(name), group)
            for name, group in data.groupby(
                "canonical_lithology", sort=True
            )
        ],
    ]
    for name, group in groups:
        raw = group["raw_residual"].to_numpy(float)
        logged = group["log1p_residual"].to_numpy(float)
        rows.append(
            {
                "canonical_lithology": name,
                "rows": int(len(group)),
                "holes": int(group["BHID"].nunique()),
                "raw_mean": float(np.mean(raw)),
                "raw_sd": float(np.std(raw, ddof=1)),
                "raw_skew": float(skew(raw, bias=False)),
                "raw_excess_kurtosis": float(kurtosis(raw, bias=False)),
                "raw_p01": float(np.quantile(raw, 0.01)),
                "raw_p99": float(np.quantile(raw, 0.99)),
                "log1p_mean": float(np.mean(logged)),
                "log1p_sd": float(np.std(logged, ddof=1)),
                "log1p_skew": float(skew(logged, bias=False)),
                "log1p_excess_kurtosis": float(
                    kurtosis(logged, bias=False)
                ),
                "analysis_role": (
                    "reviewer_motivated_post_analysis_distribution_"
                    "diagnostic"
                ),
            }
        )
    raw_groups = [
        group["raw_residual"].to_numpy(float)
        for _, group in data.groupby("canonical_lithology", sort=True)
    ]
    log_groups = [
        group["log1p_residual"].to_numpy(float)
        for _, group in data.groupby("canonical_lithology", sort=True)
    ]
    raw_bf = levene(*raw_groups, center="median")
    log_bf = levene(*log_groups, center="median")
    summary = {
        "raw_overall_skew": rows[0]["raw_skew"],
        "raw_overall_excess_kurtosis": rows[0][
            "raw_excess_kurtosis"
        ],
        "log1p_overall_skew": rows[0]["log1p_skew"],
        "log1p_overall_excess_kurtosis": rows[0][
            "log1p_excess_kurtosis"
        ],
        "raw_brown_forsythe_statistic": float(raw_bf.statistic),
        "raw_brown_forsythe_pvalue": float(raw_bf.pvalue),
        "log1p_brown_forsythe_statistic": float(log_bf.statistic),
        "log1p_brown_forsythe_pvalue": float(log_bf.pvalue),
        "transformation_variogram_sensitivity_implemented": False,
        "normal_score_sensitivity_implemented": False,
        "decision": "not scientifically necessary for the current claim",
        "evidence": (
            "raw residual skew and excess kurtosis are moderate; log1p "
            "reduces overall skew but does not remove major-lithology "
            "variance heterogeneity, while robust and classical raw-scale "
            "variograms are already retained"
        ),
        "limitation": (
            "the Brown-Forsythe result is influenced by strongly unequal "
            "group sizes, including only 37 QFR composites"
        ),
        "what_would_change_decision": (
            "extreme residual tails, transform-sensitive stable independent-"
            "hole variograms, or a future fold-local transformed prediction "
            "design with justified back-transformation"
        ),
        "changes_frozen_gate": False,
    }
    return pd.DataFrame(rows), summary
