"""Analysis orchestration helpers for the active empirical study.

These utilities operate only at observed assay/composite supports.  They do
not create blocks, grids, resources, simulations of the deposit, or drilling
recommendations.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr

from .models import CompositeRegressor, FoldLocalKrigingRegressor
from .validation import (
    ValidationSplit,
    evaluate_model,
    summarize_prediction_frame,
    tune_grouped,
)


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def directional_structure_support(raw_geology: pd.DataFrame) -> dict[str, int]:
    """Count all beta entries separately from S1 foliation entries."""

    beta_columns = [
        column for column in raw_geology.columns if "STRUCT-BETA" in column
    ]
    type_columns = [
        column
        for column in raw_geology.columns
        if "TYPE OF STRUCTURE" in column
    ]
    if len(beta_columns) != 1 or len(type_columns) != 1:
        raise ValueError(
            "geology table must contain one STRUCT-BETA and one structure-type column"
        )
    beta_values = pd.to_numeric(raw_geology[beta_columns[0]], errors="coerce")
    valid_beta = beta_values.notna()
    structure_type = (
        raw_geology[type_columns[0]]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    s1 = valid_beta & structure_type.eq("S1")
    return {
        "beta_measurements": int(valid_beta.sum()),
        "beta_holes": int(
            raw_geology.loc[valid_beta, "BHID"].astype(str).nunique()
        ),
        "s1_beta_measurements": int(s1.sum()),
        "s1_beta_holes": int(
            raw_geology.loc[s1, "BHID"].astype(str).nunique()
        ),
    }


def freeze_manifest(paths: Sequence[str | Path], root: str | Path) -> pd.DataFrame:
    root_path = Path(root).resolve()
    records: list[dict[str, object]] = []
    for item in paths:
        path = Path(item).resolve()
        records.append(
            {
                "path": str(path.relative_to(root_path)).replace("\\", "/")
                if path.is_relative_to(root_path)
                else str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "frozen_before_analysis": True,
            }
        )
    return pd.DataFrame(records).sort_values("path").reset_index(drop=True)


def prepare_primary_composites(
    composites: pd.DataFrame, fold_registry: pd.DataFrame
) -> pd.DataFrame:
    """Apply the frozen primary-support gate and attach grade-blind folds."""

    required = {
        "composite_id",
        "BHID",
        "MIDPOINT",
        "support_m",
        "support_complete",
        "tgc_pct",
        "parent_assay_ids",
        "canonical_lithology",
        "weathering",
        "BATCH_NUMBER",
        "primary_spatial_eligible",
        "mid_easting",
        "mid_northing",
        "mid_rl",
        "mid_tvd",
    }
    missing = sorted(required - set(composites.columns))
    if missing:
        raise ValueError(f"primary composite table is missing {missing}")
    fold_required = {
        "BHID",
        "northing_block_label",
        "spatial_buffer_m",
        "grade_used",
    }
    missing_fold = sorted(fold_required - set(fold_registry.columns))
    if missing_fold:
        raise ValueError(f"fold registry is missing {missing_fold}")
    if fold_registry["BHID"].astype(str).duplicated().any():
        raise ValueError("fold registry contains duplicate holes")
    if fold_registry["grade_used"].astype(bool).any():
        raise ValueError("fold registry is not grade blind")

    data = composites.loc[
        composites["support_complete"].astype(bool)
        & composites["primary_spatial_eligible"].astype(bool)
    ].copy()
    finite = np.ones(len(data), dtype=bool)
    for column in ("tgc_pct", "support_m", "mid_easting", "mid_northing", "mid_rl"):
        finite &= np.isfinite(pd.to_numeric(data[column], errors="coerce"))
    data = data.loc[finite].copy()
    data["BHID"] = data["BHID"].astype(str)
    data = data.merge(
        fold_registry[
            [
                "BHID",
                "northing_block_label",
                "spatial_buffer_m",
                "loho_fold",
            ]
        ].assign(BHID=lambda frame: frame["BHID"].astype(str)),
        on="BHID",
        how="left",
        validate="many_to_one",
    )
    if data["northing_block_label"].isna().any():
        missing_holes = sorted(data.loc[data["northing_block_label"].isna(), "BHID"].unique())
        raise ValueError(f"fold registry lacks holes: {missing_holes}")
    data["northing_block"] = data["northing_block_label"].astype(str)
    data["hole_mean_depth_m"] = data.groupby("BHID")["MIDPOINT"].transform("mean")
    data["depth_within_hole_m"] = data["MIDPOINT"] - data["hole_mean_depth_m"]
    data["abs_rl_m"] = data["mid_rl"]
    return data.sort_values(["BHID", "FROM", "TO", "composite_id"]).reset_index(drop=True)


def support_sensitivity(
    composite_tables: Mapping[str, pd.DataFrame],
    *,
    censored_base: float = 0.025,
    censor_values: Sequence[float] = (0.0, 0.025, 0.05),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for support_name, raw in composite_tables.items():
        data = raw.loc[
            raw["support_complete"].astype(bool)
            & raw["primary_spatial_eligible"].astype(bool)
        ].copy()
        grade = pd.to_numeric(data["tgc_pct"], errors="coerce").to_numpy(float)
        support = pd.to_numeric(data["support_m"], errors="coerce").to_numpy(float)
        censor_fraction = pd.to_numeric(
            data.get("censored_support_fraction", 0.0), errors="coerce"
        ).fillna(0.0).to_numpy(float)
        valid = np.isfinite(grade) & np.isfinite(support) & (support > 0)
        grade, support, censor_fraction = grade[valid], support[valid], censor_fraction[valid]
        hole = data.loc[valid, "BHID"].astype(str)
        hole_mean = (
            pd.DataFrame({"hole": hole, "grade": grade, "support": support})
            .assign(mass=lambda frame: frame["grade"] * frame["support"])
            .groupby("hole", sort=True)
            .agg(mass=("mass", "sum"), support=("support", "sum"))
        )
        hole_mean["grade"] = hole_mean["mass"] / hole_mean["support"]
        for censor_value in censor_values:
            adjusted = grade + censor_fraction * (float(censor_value) - censored_base)
            q99, q995 = np.quantile(adjusted, [0.99, 0.995])
            rows.append(
                {
                    "support": support_name,
                    "censor_value_pct": float(censor_value),
                    "rows": len(adjusted),
                    "holes": hole.nunique(),
                    "total_support_m": float(support.sum()),
                    "length_weighted_mean_tgc_pct": float(np.average(adjusted, weights=support)),
                    "equal_hole_mean_tgc_pct": float(hole_mean["grade"].mean()),
                    "uncapped_mean_tgc_pct": float(np.mean(adjusted)),
                    "p99_cap_pct": float(q99),
                    "p99_capped_mean_tgc_pct": float(np.mean(np.minimum(adjusted, q99))),
                    "p99_5_cap_pct": float(q995),
                    "p99_5_capped_mean_tgc_pct": float(np.mean(np.minimum(adjusted, q995))),
                    "topcut_base_policy": "none",
                }
            )
    return pd.DataFrame(rows)


def cell_declustering_sensitivity(
    data: pd.DataFrame, base_spacing_m: float
) -> pd.DataFrame:
    """Cell-declustered means across spacing-derived sizes and two origins."""

    rows: list[dict[str, object]] = []
    x = data["mid_easting"].to_numpy(float)
    y = data["mid_northing"].to_numpy(float)
    grade = data["tgc_pct"].to_numpy(float)
    for multiplier in (0.5, 1.0, 2.0):
        size = max(float(base_spacing_m) * multiplier, np.finfo(float).eps)
        for origin_fraction in (0.0, 0.5):
            ox = float(np.min(x) + origin_fraction * size)
            oy = float(np.min(y) + origin_fraction * size)
            ix = np.floor((x - ox) / size).astype(int)
            iy = np.floor((y - oy) / size).astype(int)
            keys = pd.Series(list(zip(ix, iy)))
            counts = keys.value_counts()
            weights = np.asarray([1.0 / counts[key] for key in keys], dtype=float)
            rows.append(
                {
                    "cell_size_m": size,
                    "cell_size_over_median_hole_nn": multiplier,
                    "origin_fraction": origin_fraction,
                    "occupied_cells": int(len(counts)),
                    "cell_declustered_mean_tgc_pct": float(np.average(grade, weights=weights)),
                    "weight_cv": float(np.std(weights) / np.mean(weights)),
                    "interpretation": "sampling-cluster sensitivity, not an estimation grid",
                }
            )
    return pd.DataFrame(rows)


def domain_summary(data: pd.DataFrame) -> pd.DataFrame:
    grouped = data.groupby(
        ["canonical_lithology", "grsc_subtype", "weathering"],
        dropna=False,
        sort=True,
    )
    rows: list[dict[str, object]] = []
    for key, group in grouped:
        grade = group["tgc_pct"].to_numpy(float)
        support = group["support_m"].to_numpy(float)
        hole_means = (
            group.assign(mass=group["tgc_pct"] * group["support_m"])
            .groupby("BHID")
            .agg(mass=("mass", "sum"), support=("support_m", "sum"))
        )
        hole_means["grade"] = hole_means["mass"] / hole_means["support"]
        rows.append(
            {
                "canonical_lithology": key[0],
                "grsc_subtype": key[1],
                "weathering": key[2],
                "rows": len(group),
                "holes": group["BHID"].nunique(),
                "support_m": float(support.sum()),
                "length_weighted_mean_tgc_pct": float(np.average(grade, weights=support)),
                "median_tgc_pct": float(np.median(grade)),
                "equal_hole_mean_tgc_pct": float(hole_means["grade"].mean()),
                "p10_tgc_pct": float(np.quantile(grade, 0.10)),
                "p90_tgc_pct": float(np.quantile(grade, 0.90)),
            }
        )
    return pd.DataFrame(rows)


def depth_associations(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    strata: list[tuple[str, pd.DataFrame]] = [("all", data)]
    strata.extend(
        (f"lithology={name}", group)
        for name, group in data.groupby("canonical_lithology", sort=True)
    )
    for stratum, group in strata:
        for variable in (
            "MIDPOINT",
            "mid_rl",
            "depth_within_hole_m",
            "hole_mean_depth_m",
        ):
            use = group[[variable, "tgc_pct", "BHID"]].dropna()
            if len(use) < 3 or use[variable].nunique() < 2:
                rho = p_value = np.nan
            else:
                rho, p_value = spearmanr(use[variable], use["tgc_pct"])
            rows.append(
                {
                    "stratum": stratum,
                    "variable": variable,
                    "rows": len(use),
                    "holes": use["BHID"].nunique(),
                    "spearman_rho": float(rho),
                    "spearman_p": float(p_value),
                    "causal_interpretation_permitted": False,
                }
            )
    return pd.DataFrame(rows)


def add_signed_contact_distance(
    composites: pd.DataFrame, geology: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach nearest verified along-hole QFR–graphitic-schist contact distance."""

    contacts: list[dict[str, object]] = []
    for hole, group in geology.sort_values(["BHID", "FROM", "TO"]).groupby("BHID"):
        records = group.reset_index(drop=True)
        for index in range(len(records) - 1):
            left = records.iloc[index]
            right = records.iloc[index + 1]
            if not np.isclose(float(left["TO"]), float(right["FROM"]), atol=1e-6):
                continue
            pair = {str(left["canonical_lithology"]), str(right["canonical_lithology"])}
            if pair != {"qfr", "graphitic_schist"}:
                continue
            contacts.append(
                {
                    "contact_id": f"{hole}_QFR_GRSC_{len(contacts) + 1:04d}",
                    "BHID": str(hole),
                    "contact_md_m": float((left["TO"] + right["FROM"]) / 2.0),
                    "shallower_lithology": str(left["canonical_lithology"]),
                    "deeper_lithology": str(right["canonical_lithology"]),
                    "boundary_source": "adjacent_authoritative_logged_intervals",
                }
            )
    contact_frame = pd.DataFrame(contacts)
    result = composites.copy()
    result["nearest_qfr_grsc_contact_id"] = pd.NA
    result["signed_contact_distance_m"] = np.nan
    result["abs_contact_distance_m"] = np.nan
    if contact_frame.empty:
        return result, contact_frame
    per_hole = {
        hole: group
        for hole, group in contact_frame.groupby("BHID", sort=False)
    }
    for row_index, row in result.iterrows():
        lith = str(row["canonical_lithology"])
        if lith not in {"qfr", "graphitic_schist"} or str(row["BHID"]) not in per_hole:
            continue
        candidates = per_hole[str(row["BHID"])]
        distances = row["MIDPOINT"] - candidates["contact_md_m"].to_numpy(float)
        selected_position = int(np.argmin(np.abs(distances)))
        selected = candidates.iloc[selected_position]
        absolute = float(abs(distances[selected_position]))
        signed = absolute if lith == "graphitic_schist" else -absolute
        result.at[row_index, "nearest_qfr_grsc_contact_id"] = selected["contact_id"]
        result.at[row_index, "signed_contact_distance_m"] = signed
        result.at[row_index, "abs_contact_distance_m"] = absolute
    return result, contact_frame


def contact_window_summary(
    data: pd.DataFrame, windows: Sequence[float] = (1.0, 2.0, 5.0, 10.0)
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    linked = data.loc[data["signed_contact_distance_m"].notna()].copy()
    for window in windows:
        local = linked.loc[linked["abs_contact_distance_m"] <= float(window)]
        for side, group in local.groupby("canonical_lithology", sort=True):
            if group.empty:
                continue
            rows.append(
                {
                    "window_m": float(window),
                    "side": side,
                    "rows": len(group),
                    "holes": group["BHID"].nunique(),
                    "support_m": float(group["support_m"].sum()),
                    "length_weighted_mean_tgc_pct": float(
                        np.average(group["tgc_pct"], weights=group["support_m"])
                    ),
                    "distance_definition": "signed along-hole distance; not perpendicular distance or true thickness",
                }
            )
    return pd.DataFrame(rows)


def _prediction_failure_frame(
    test: pd.DataFrame, split: ValidationSplit, message: str
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_index": test.index.to_numpy(),
            "scheme": split.scheme,
            "fold_id": split.fold_id,
            "truth": test["tgc_pct"].to_numpy(float),
            "prediction": np.nan,
            "variance": np.nan,
            "success": False,
            "hole": test["BHID"].astype(str).to_numpy(),
            "weight": test["support_m"].to_numpy(float),
            "error_message": message,
        }
    )


def evaluate_fixed_safely(
    data: pd.DataFrame,
    splits: Sequence[ValidationSplit],
    model_factory: Callable[[], CompositeRegressor],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split in splits:
        try:
            frames.append(
                evaluate_model(
                    data,
                    [split],
                    model_factory=model_factory,
                    target_col="tgc_pct",
                    hole_col="BHID",
                    weight_col="support_m",
                ).assign(error_message="")
            )
        except Exception as exc:  # fold-level abstention must be auditable
            frames.append(
                _prediction_failure_frame(
                    data.iloc[split.test_index], split, f"{type(exc).__name__}: {exc}"
                )
            )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def evaluate_kriging_diagnostic_safely(
    data: pd.DataFrame,
    splits: Sequence[ValidationSplit],
    model_factory: Callable[[], FoldLocalKrigingRegressor],
    *,
    model_name: str,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate post-gate fold-local kriging and retain a fold audit.

    This helper is deliberately separate from the decision-eligible prediction
    path.  It never changes the prospective gate and accepts only models whose
    explicit execution policy is ``diagnostic_sensitivity``.
    """

    if n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer")
    if n_jobs > 1 and len(splits) > 1:
        results = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            batch_size=1,
            pre_dispatch=n_jobs,
        )(
            delayed(evaluate_kriging_diagnostic_safely)(
                data,
                [split],
                model_factory,
                model_name=model_name,
                n_jobs=1,
            )
            for split in splits
        )
        predictions = pd.concat(
            [item[0] for item in results], ignore_index=True
        )
        audits = pd.concat([item[1] for item in results], ignore_index=True)
        return predictions, audits

    frames: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for split in splits:
        train = data.iloc[split.train_index]
        test = data.iloc[split.test_index]
        model = model_factory()
        if model.execution_policy != "diagnostic_sensitivity":
            raise ValueError(
                "diagnostic evaluator requires execution_policy="
                "diagnostic_sensitivity"
            )
        audit: dict[str, object] = {
            "model": model_name,
            "scheme": split.scheme,
            "fold_id": split.fold_id,
            "train_rows": len(train),
            "test_rows": len(test),
            "train_holes": int(train["BHID"].nunique()),
            "test_holes": int(test["BHID"].nunique()),
            "execution_policy": model.execution_policy,
            "fit_sensitivity_family": model.fit_sensitivity_family,
            "pair_mode": model.pair_mode,
            "pair_domain_policy": model.pair_domain_policy,
            "operational_n_lags": model.operational_n_lags,
            "operational_maxlag_fraction": model.operational_maxlag_fraction,
            "operational_estimator": model.operational_estimator,
            "operational_model": model.operational_model,
            "min_unique_hole_pairs": model.min_unique_hole_pairs,
            "min_supported_bins": model.stability_thresholds.min_supported_bins,
            "max_neighbors": model.max_neighbors,
            "min_neighbors": model.min_neighbors,
            "search_radius_multiplier": model.search_radius_multiplier,
            "acceptance_role": "post_gate_non_decision_sensitivity",
            "public_release_approved": False,
        }
        try:
            model.fit(train, "tgc_pct")
            result = model.predict(test)
            frames.append(
                pd.DataFrame(
                    {
                        "row_index": test.index.to_numpy(),
                        "scheme": split.scheme,
                        "fold_id": split.fold_id,
                        "truth": test["tgc_pct"].to_numpy(float),
                        "prediction": result.mean,
                        "variance": result.variance,
                        "success": result.success,
                        "hole": test["BHID"].astype(str).to_numpy(),
                        "weight": test["support_m"].to_numpy(float),
                        "error_message": "",
                        "model": model_name,
                        "acceptance_role": "post_gate_non_decision_sensitivity",
                        "public_release_approved": False,
                    }
                )
            )
            gate = model.fold_gate_
            variogram = model.fold_variogram_
            empirical = model.operational_empirical_
            supported = (
                empirical.supported
                & (
                    empirical.unique_hole_pairs
                    >= model.min_unique_hole_pairs
                )
            )
            audit.update(
                {
                    "status": "complete",
                    "fold_gate_passed": gate.passed,
                    "fold_gate_reasons": "; ".join(gate.reasons),
                    "continued_past_failed_stability": (
                        model.continued_past_failed_stability_
                    ),
                    "minimum_support_passed": model.minimum_support_passed_,
                    "successful_fits": gate.successful_fits,
                    "supported_bins": gate.supported_bins,
                    "minimum_unique_hole_pairs_in_supported_bins": (
                        int(np.min(empirical.unique_hole_pairs[supported]))
                        if np.any(supported)
                        else 0
                    ),
                    "range_m": variogram.range,
                    "operational_range": variogram.range,
                    "partial_sill": variogram.sill,
                    "operational_sill": variogram.sill,
                    "nugget": variogram.nugget,
                    "operational_nugget": variogram.nugget,
                    "nugget_fraction": (
                        variogram.nugget / variogram.total_sill
                        if variogram.total_sill > 0
                        else 1.0
                    ),
                    "nugget_ratio": (
                        variogram.nugget / variogram.total_sill
                        if variogram.total_sill > 0
                        else 1.0
                    ),
                    "normalized_rmse": variogram.normalized_rmse,
                    "search_radius_m": (
                        model.search_radius_multiplier * variogram.range
                    ),
                    "estimated_rows": int(np.sum(result.success)),
                    "failed_rows": int(np.sum(~result.success)),
                    "fit_success": 1,
                    "prediction_failure_rate": float(np.mean(~result.success)),
                    "error_message": "",
                }
            )
        except Exception as exc:  # diagnostic failures remain auditable
            message = f"{type(exc).__name__}: {exc}"
            frames.append(_prediction_failure_frame(test, split, message).assign(
                model=model_name,
                acceptance_role="post_gate_non_decision_sensitivity",
                public_release_approved=False,
            ))
            gate = getattr(model, "fold_gate_", None)
            audit.update(
                {
                    "status": "failed",
                    "fold_gate_passed": getattr(gate, "passed", False),
                    "fold_gate_reasons": "; ".join(
                        getattr(gate, "reasons", ())
                    ),
                    "continued_past_failed_stability": getattr(
                        model, "continued_past_failed_stability_", False
                    ),
                    "minimum_support_passed": getattr(
                        model, "minimum_support_passed_", False
                    ),
                    "estimated_rows": 0,
                    "failed_rows": len(test),
                    "fit_success": 0,
                    "prediction_failure_rate": 1.0,
                    "error_message": message,
                }
            )
        audits.append(audit)
    predictions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return predictions, pd.DataFrame(audits)


def evaluate_nested_safely(
    data: pd.DataFrame,
    splits: Sequence[ValidationSplit],
    *,
    model_factory: Callable[[Mapping[str, object]], CompositeRegressor],
    parameter_grid: Mapping[str, Sequence[object]],
    tuning_folds: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nested whole-hole tuning; a failed fold becomes an explicit non-estimate."""

    predictions: list[pd.DataFrame] = []
    tuning: list[dict[str, object]] = []
    for split in splits:
        train = data.iloc[split.train_index]
        test = data.iloc[split.test_index]
        try:
            tuned = tune_grouped(
                train,
                model_factory=model_factory,
                parameter_grid=parameter_grid,
                target_col="tgc_pct",
                group_col="BHID",
                weight_col="support_m",
                n_splits=min(tuning_folds, max(2, train["BHID"].nunique())),
            )
            model = model_factory(tuned.best_parameters).fit(train, "tgc_pct")
            result = model.predict(test)
            predictions.append(
                pd.DataFrame(
                    {
                        "row_index": test.index.to_numpy(),
                        "scheme": split.scheme,
                        "fold_id": split.fold_id,
                        "truth": test["tgc_pct"].to_numpy(float),
                        "prediction": result.mean,
                        "variance": result.variance,
                        "success": result.success,
                        "hole": test["BHID"].astype(str).to_numpy(),
                        "weight": test["support_m"].to_numpy(float),
                        "error_message": "",
                    }
                )
            )
            tuning.append(
                {
                    "scheme": split.scheme,
                    "outer_fold": split.fold_id,
                    "best_parameters": repr(dict(tuned.best_parameters)),
                    "best_score": tuned.best_score,
                    "candidate_scores": repr(tuned.candidate_scores),
                    "status": "success",
                    **{f"best_{key}": value for key, value in tuned.best_parameters.items()},
                }
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            predictions.append(_prediction_failure_frame(test, split, message))
            tuning.append(
                {
                    "scheme": split.scheme,
                    "outer_fold": split.fold_id,
                    "best_parameters": "",
                    "best_score": np.nan,
                    "candidate_scores": "",
                    "status": "abstain",
                    "error_message": message,
                }
            )
    return (
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
        pd.DataFrame(tuning),
    )


def consensus_parameters(tuning: pd.DataFrame) -> dict[str, object]:
    if tuning.empty or "best_parameters" not in tuning:
        return {}
    valid = tuning.loc[tuning["status"].eq("success"), "best_parameters"].astype(str)
    if valid.empty:
        return {}
    winner = Counter(valid).most_common(1)[0][0]
    for _, row in tuning.loc[
        tuning["best_parameters"].astype(str).eq(winner)
    ].iterrows():
        return {
            key.removeprefix("best_"): value
            for key, value in row.items()
            if key.startswith("best_")
            and key not in {"best_parameters", "best_score"}
            and pd.notna(value)
        }
    return {}


def metrics_record(model: str, scheme: str, predictions: pd.DataFrame) -> dict[str, object]:
    local = predictions.loc[
        predictions["model"].eq(model) & predictions["scheme"].eq(scheme)
    ]
    if local.empty:
        return {
            "model": model,
            "scheme": scheme,
            "status": "not_executed",
        }
    summary = summarize_prediction_frame(local)
    return {
        "model": model,
        "scheme": scheme,
        "status": "complete",
        **asdict(summary),
    }

