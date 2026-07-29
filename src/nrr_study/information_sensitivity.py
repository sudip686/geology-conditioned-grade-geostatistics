"""Post-analysis information and selection sensitivity utilities.

All calculations are at observed composite support. Hole thinning is a
grade-blind information-sensitivity experiment, not a spacing optimisation or
a drilling recommendation. Selected specimens are a targeted cohort and are
never treated as representative of the full assay population.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .analysis_helpers import evaluate_fixed_safely
from .models import (
    GeologyRegressionRegressor,
    GlobalMeanRegressor,
    IDWRegressor,
)
from .validation import (
    MetricSummary,
    PairedInterval,
    ValidationSplit,
    paired_hole_bootstrap_mae,
    summarize_prediction_frame,
)


PRIMARY_SCHEMES = ("leave_one_hole_out", "northing_block_buffered")


def parent_sample_set(value: object) -> frozenset[str]:
    if value is None or pd.isna(value):
        return frozenset()
    return frozenset(
        token.strip()
        for token in str(value).split("|")
        if token.strip() and token.strip().lower() not in {"nan", "none", "<na>"}
    )


def reconcile_selected_master(
    master: pd.DataFrame, composites: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reconcile the selected-specimen register and flag wholly selected support."""

    required_master = {"sample_entity_key", "sample_id", "assay_sample_id", "bhid"}
    missing = sorted(required_master - set(master.columns))
    if missing:
        raise ValueError(f"sample master is missing {missing}")
    if "parent_sample_ids" not in composites:
        raise ValueError("composites are missing parent_sample_ids")

    exact = master.loc[master["assay_sample_id"].notna()].copy()
    exact["assay_sample_id"] = exact["assay_sample_id"].astype(str).str.strip()
    exact = exact.loc[exact["assay_sample_id"].ne("")]
    unique_ids = frozenset(exact["assay_sample_id"].unique())
    duplicate_mask = exact["assay_sample_id"].duplicated(keep=False)
    duplicate_rows = exact.loc[
        duplicate_mask,
        [
            "sample_entity_key",
            "sample_id",
            "bhid",
            "assay_sample_id",
            "from_m",
            "to_m",
        ],
    ].sort_values(["assay_sample_id", "sample_entity_key"])

    flagged = composites.copy()
    flagged["parent_sample_id_count"] = flagged["parent_sample_ids"].map(
        lambda value: len(parent_sample_set(value))
    )
    flagged["selected_endpoint"] = flagged["parent_sample_ids"].map(
        lambda value: bool(parent_sample_set(value))
        and parent_sample_set(value).issubset(unique_ids)
    )
    flagged["selected_cohort_boundary"] = np.where(
        flagged["selected_endpoint"],
        (
            "all nonempty parent sample IDs occur in the 293-ID selected register; "
            "targeted selection sensitivity only"
        ),
        "not wholly inside selected-ID register",
    )
    summary = pd.DataFrame(
        [
            {
                "master_rows": len(master),
                "exact_linked_rows": len(exact),
                "unique_exact_assay_ids": len(unique_ids),
                "duplicate_assay_id_linkages": exact.loc[
                    duplicate_mask, "assay_sample_id"
                ].nunique(),
                "duplicate_linkage_rows": int(duplicate_mask.sum()),
                "selected_2m_composites": int(flagged["selected_endpoint"].sum()),
                "selected_endpoint_holes": int(
                    flagged.loc[flagged["selected_endpoint"], "BHID"].nunique()
                ),
                "selection_rule": (
                    "nonempty parent sample-ID set must be wholly contained in "
                    "the unique exact-linked assay-ID register"
                ),
                "representativeness_claim_permitted": False,
            }
        ]
    )
    return summary, duplicate_rows.reset_index(drop=True), flagged


def model_factory(name: str) -> Callable[[], object]:
    if name == "global_mean":
        return lambda: GlobalMeanRegressor()
    if name == "lithology_only":
        return lambda: GeologyRegressionRegressor(
            categorical_cols=("canonical_lithology", "grsc_subtype"),
            numeric_cols=(),
            alpha=1e-6,
        )
    if name == "geology_only":
        return lambda: GeologyRegressionRegressor(
            categorical_cols=(
                "canonical_lithology",
                "grsc_subtype",
                "weathering",
            ),
            numeric_cols=(),
            alpha=1e-6,
        )
    if name == "idw":
        return lambda: IDWRegressor(
            power=2.0,
            max_neighbors=32,
            search_radius=None,
            min_neighbors=1,
        )
    raise ValueError(f"unsupported post-analysis model: {name}")


def whole_hole_selection_predictions(
    flagged: pd.DataFrame,
    *,
    model_names: Sequence[str] = ("geology_only", "idw"),
) -> pd.DataFrame:
    """Compare full and selected-information training at whole-hole holdout."""

    selected = flagged.loc[flagged["selected_endpoint"]].copy()
    if selected.empty:
        return pd.DataFrame()
    full_holes = sorted(flagged["BHID"].astype(str).unique())
    selected_holes = sorted(selected["BHID"].astype(str).unique())
    frames: list[pd.DataFrame] = []
    for model_name in model_names:
        factory = model_factory(model_name)
        regimes = (
            ("full_cohort", full_holes, False, False),
            (
                "selected_endpoints_full_training",
                selected_holes,
                False,
                True,
            ),
            ("selected_only_training", selected_holes, True, True),
        )
        for regime, test_holes, selected_train, selected_test in regimes:
            for hole in test_holes:
                test_mask = flagged["BHID"].astype(str).eq(hole)
                if selected_test:
                    test_mask &= flagged["selected_endpoint"]
                train_mask = ~flagged["BHID"].astype(str).eq(hole)
                if selected_train:
                    train_mask &= flagged["selected_endpoint"]
                train = flagged.loc[train_mask]
                test = flagged.loc[test_mask]
                if train.empty or test.empty:
                    continue
                split = ValidationSplit(
                    scheme="whole_hole_selection_sensitivity",
                    fold_id=hole,
                    train_index=np.arange(len(train), dtype=int),
                    test_index=np.arange(len(train), len(train) + len(test), dtype=int),
                    buffered_out_index=np.asarray([], dtype=int),
                )
                combined = pd.concat([train, test], ignore_index=True)
                prediction = evaluate_fixed_safely(
                    combined, [split], factory
                ).assign(
                    model=model_name,
                    information_regime=regime,
                    composite_id=test["composite_id"].astype(str).to_numpy(),
                    selected_endpoint=selected_test,
                    training_rows=len(train),
                    training_holes=train["BHID"].nunique(),
                    selection_limitation=(
                        "targeted selected-specimen cohort; no representativeness claim"
                    ),
                )
                frames.append(prediction)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def selection_prediction_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, regime), group in predictions.groupby(
        ["model", "information_regime"], sort=True
    ):
        metric = summarize_prediction_frame(group)
        rows.append(
            {
                "model": model,
                "information_regime": regime,
                **asdict(metric),
                "interpretation": (
                    "information/selection sensitivity only; selected specimens "
                    "are not assumed representative"
                ),
            }
        )
    return pd.DataFrame(rows)


def selection_training_paired_ci(
    predictions: pd.DataFrame,
    *,
    replicates: int = 2000,
    seed: int = 20260728,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in sorted(predictions["model"].unique()):
        full = predictions.loc[
            predictions["model"].eq(model)
            & predictions["information_regime"].eq(
                "selected_endpoints_full_training"
            )
        ]
        selected = predictions.loc[
            predictions["model"].eq(model)
            & predictions["information_regime"].eq("selected_only_training")
        ]
        joined = selected.merge(
            full,
            on=["composite_id", "fold_id"],
            suffixes=("_selected", "_full"),
            validate="one_to_one",
        )
        interval = paired_hole_bootstrap_mae(
            joined["truth_selected"],
            joined["prediction_selected"],
            joined["prediction_full"],
            joined["hole_selected"],
            weights=joined["weight_selected"],
            conditioned_success=joined["success_selected"],
            comparator_success=joined["success_full"],
            replicates=replicates,
            random_state=seed,
        )
        rows.append(
            {
                "model": model,
                "comparison": "selected_only_minus_full_training_at_selected_endpoints",
                **asdict(interval),
                "bootstrap_replicates": int(replicates),
                "interpretation": (
                    "information-loss sensitivity; not a claim that selected "
                    "specimens represent the full cohort"
                ),
            }
        )
    return pd.DataFrame(rows)


def farthest_point_order(
    holes: pd.DataFrame,
    *,
    hole_col: str = "BHID",
    x_col: str = "collar_easting",
    y_col: str = "collar_northing",
) -> list[str]:
    """Return a grade-blind, deterministic, spatially spreading order."""

    required = {hole_col, x_col, y_col}
    missing = sorted(required - set(holes.columns))
    if missing:
        raise ValueError(f"hole table is missing {missing}")
    frame = holes.loc[:, [hole_col, x_col, y_col]].copy()
    frame[hole_col] = frame[hole_col].astype(str)
    frame = frame.sort_values(hole_col).reset_index(drop=True)
    coords = frame[[x_col, y_col]].to_numpy(float)
    if len(frame) == 0:
        return []
    centroid = np.mean(coords, axis=0)
    radial = np.linalg.norm(coords - centroid, axis=1)
    first = int(
        sorted(
            range(len(frame)),
            key=lambda index: (-radial[index], frame.at[index, hole_col]),
        )[0]
    )
    selected = [first]
    remaining = set(range(len(frame))) - {first}
    while remaining:
        chosen_coords = coords[selected]
        ranked = []
        for index in remaining:
            minimum_distance = float(
                np.min(np.linalg.norm(chosen_coords - coords[index], axis=1))
            )
            ranked.append(
                (-minimum_distance, frame.at[index, hole_col], index)
            )
        index = sorted(ranked)[0][2]
        selected.append(index)
        remaining.remove(index)
    return frame.iloc[selected][hole_col].tolist()


def hole_thinning_information_sensitivity(
    data: pd.DataFrame,
    fold_registry: pd.DataFrame,
    *,
    fractions: Sequence[float] = (1.0, 0.8, 0.6, 0.5),
    model_names: Sequence[str] = ("geology_only", "idw"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Five grade-blind validation rotations with nested farthest-point subsets."""

    registry = fold_registry.copy()
    registry["BHID"] = registry["BHID"].astype(str)
    if registry["grade_used"].astype(bool).any():
        raise ValueError("hole-thinning registry is not grade blind")
    blocks = sorted(registry["northing_block_label"].astype(str).unique())
    if len(blocks) != 5:
        raise ValueError(f"expected five frozen northing blocks, found {len(blocks)}")
    data = data.copy()
    data["BHID"] = data["BHID"].astype(str)
    prediction_frames: list[pd.DataFrame] = []
    hole_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    coordinates = registry.set_index("BHID")[
        ["collar_easting", "collar_northing"]
    ]

    for rotation, block in enumerate(blocks, start=1):
        validation_holes = sorted(
            registry.loc[
                registry["northing_block_label"].astype(str).eq(block), "BHID"
            ]
        )
        pool = registry.loc[~registry["BHID"].isin(validation_holes)].copy()
        order = farthest_point_order(pool)
        for rank, hole in enumerate(order, start=1):
            selection_rows.append(
                {
                    "rotation": rotation,
                    "validation_block": block,
                    "training_hole": hole,
                    "farthest_point_rank": rank,
                    "selection_used_grade": False,
                }
            )
        for fraction in fractions:
            if not 0 < fraction <= 1:
                raise ValueError("training fractions must be in (0, 1]")
            count = (
                len(order)
                if np.isclose(fraction, 1.0)
                else max(1, int(np.floor(len(order) * fraction + 0.5)))
            )
            training_holes = order[:count]
            train = data.loc[data["BHID"].isin(training_holes)]
            test = data.loc[data["BHID"].isin(validation_holes)]
            combined = pd.concat([train, test], ignore_index=True)
            split = ValidationSplit(
                scheme="hole_thinning_information_sensitivity",
                fold_id=f"rotation_{rotation}_{fraction:.2f}",
                train_index=np.arange(len(train), dtype=int),
                test_index=np.arange(len(train), len(combined), dtype=int),
                buffered_out_index=np.asarray([], dtype=int),
            )
            for model_name in model_names:
                prediction = evaluate_fixed_safely(
                    combined, [split], model_factory(model_name)
                ).assign(
                    model=model_name,
                    rotation=rotation,
                    validation_block=block,
                    training_fraction=fraction,
                    available_training_holes=len(order),
                    retained_training_holes=len(training_holes),
                    information_sensitivity_only=True,
                )
                prediction_frames.append(prediction)
                for hole, group in prediction.groupby("hole", sort=True):
                    valid = group["success"].astype(bool) & np.isfinite(
                        group["prediction"]
                    )
                    if valid.any():
                        mae = float(
                            np.average(
                                np.abs(
                                    group.loc[valid, "prediction"]
                                    - group.loc[valid, "truth"]
                                ),
                                weights=group.loc[valid, "weight"],
                            )
                        )
                    else:
                        mae = np.nan
                    validation_xy = coordinates.loc[str(hole)].to_numpy(float)
                    training_xy = coordinates.loc[training_holes].to_numpy(float)
                    nearest = float(
                        np.min(np.linalg.norm(training_xy - validation_xy, axis=1))
                    )
                    hole_rows.append(
                        {
                            "rotation": rotation,
                            "validation_block": block,
                            "model": model_name,
                            "training_fraction": fraction,
                            "available_training_holes": len(order),
                            "retained_training_holes": len(training_holes),
                            "validation_hole": hole,
                            "validation_rows": len(group),
                            "hole_mae_tgc_pct": mae,
                            "nearest_training_hole_distance_m": nearest,
                            "failure_rate": float(1.0 - valid.mean()),
                            "interpretation": (
                                "information sensitivity only; no preferred "
                                "fraction or spacing recommendation"
                            ),
                        }
                    )
    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    return (
        predictions,
        pd.DataFrame(hole_rows),
        pd.DataFrame(selection_rows),
    )


def thinning_summary(hole_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = hole_metrics.groupby(
        ["rotation", "validation_block", "model", "training_fraction"],
        sort=True,
    )
    for key, group in grouped:
        rows.append(
            {
                "rotation": key[0],
                "validation_block": key[1],
                "model": key[2],
                "training_fraction": key[3],
                "validation_holes": group["validation_hole"].nunique(),
                "mean_hole_mae_tgc_pct": float(group["hole_mae_tgc_pct"].mean()),
                "median_hole_mae_tgc_pct": float(
                    group["hole_mae_tgc_pct"].median()
                ),
                "mean_nearest_training_hole_distance_m": float(
                    group["nearest_training_hole_distance_m"].mean()
                ),
                "median_nearest_training_hole_distance_m": float(
                    group["nearest_training_hole_distance_m"].median()
                ),
                "interpretation": (
                    "information sensitivity only; fractions are not ranked "
                    "as preferred and distances are not recommendations"
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_model_evidence(
    predictions: pd.DataFrame,
    *,
    conditioned: str,
    comparator: str,
    schemes: Sequence[str] = PRIMARY_SCHEMES,
    replicates: int = 2000,
    seed: int = 20260728,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scheme in schemes:
        first = predictions.loc[
            predictions["model"].eq(conditioned)
            & predictions["scheme"].eq(scheme)
        ]
        second = predictions.loc[
            predictions["model"].eq(comparator)
            & predictions["scheme"].eq(scheme)
        ]
        joined = first.merge(
            second,
            on=["row_index", "scheme", "fold_id"],
            suffixes=("_conditioned", "_comparator"),
            validate="one_to_one",
        )
        interval = paired_hole_bootstrap_mae(
            joined["truth_conditioned"],
            joined["prediction_conditioned"],
            joined["prediction_comparator"],
            joined["hole_conditioned"],
            weights=joined["weight_conditioned"],
            conditioned_success=joined["success_conditioned"],
            comparator_success=joined["success_comparator"],
            replicates=replicates,
            random_state=seed,
        )
        rows.append(
            {
                "conditioned_model": conditioned,
                "comparator_model": comparator,
                "scheme": scheme,
                **asdict(interval),
                "bootstrap_replicates": int(replicates),
                "support_rule": (
                    "supported only when the upper 95% CI is below zero in "
                    "both primary held-out schemes"
                ),
            }
        )
    return pd.DataFrame(rows)


def both_schemes_support(evidence: pd.DataFrame) -> bool:
    if evidence.empty:
        return False
    available = set(evidence["scheme"].astype(str))
    return set(PRIMARY_SCHEMES).issubset(available) and bool(
        (
            evidence.loc[
                evidence["scheme"].isin(PRIMARY_SCHEMES), "upper"
            ]
            < 0
        ).all()
    )


def format_evidence(evidence: pd.DataFrame) -> str:
    parts: list[str] = []
    for _, row in evidence.sort_values("scheme").iterrows():
        parts.append(
            f"{row['scheme']}: delta MAE {row['estimate']:.4g} "
            f"(95% CI {row['lower']:.4g}, {row['upper']:.4g})"
        )
    return "; ".join(parts)


def primary_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, scheme), group in predictions.groupby(
        ["model", "scheme"], sort=True
    ):
        metric: MetricSummary = summarize_prediction_frame(group)
        rows.append({"model": model, "scheme": scheme, **asdict(metric)})
    return pd.DataFrame(rows)


def contact_window_validation(
    primary: pd.DataFrame,
    contact_predictions: pd.DataFrame,
    *,
    windows: Sequence[float] = (1.0, 2.0, 5.0, 10.0),
    replicates: int = 2000,
    seed: int = 20260728,
) -> pd.DataFrame:
    """Filter existing held-hole policy predictions by along-hole contact window."""

    contact_data = primary.loc[
        primary["signed_contact_distance_m"].notna()
    ].reset_index(drop=True)
    context = contact_data[
        ["BHID", "abs_contact_distance_m", "signed_contact_distance_m"]
    ].copy()
    context["row_index"] = context.index
    predictions = contact_predictions.merge(
        context,
        on="row_index",
        how="left",
        validate="many_to_one",
    )
    rows: list[dict[str, object]] = []
    for window in windows:
        local = predictions.loc[
            predictions["abs_contact_distance_m"] <= float(window)
        ]
        hard = local.loc[local["policy"].eq("hard")]
        pooled = local.loc[local["policy"].eq("pooled")]
        joined = hard.merge(
            pooled,
            on=["row_index", "scheme", "fold_id"],
            suffixes=("_hard", "_pooled"),
            validate="one_to_one",
        )
        interval = paired_hole_bootstrap_mae(
            joined["truth_hard"],
            joined["prediction_hard"],
            joined["prediction_pooled"],
            joined["hole_hard"],
            weights=joined["weight_hard"],
            conditioned_success=joined["success_hard"],
            comparator_success=joined["success_pooled"],
            replicates=replicates,
            random_state=seed,
        )
        for policy, group in local.groupby("policy", sort=True):
            metric = summarize_prediction_frame(group)
            rows.append(
                {
                    "window_m": float(window),
                    "policy": policy,
                    **asdict(metric),
                    "hard_minus_pooled_mae_estimate": interval.estimate,
                    "hard_minus_pooled_mae_ci_low": interval.lower,
                    "hard_minus_pooled_mae_ci_high": interval.upper,
                    "bootstrap_replicates": int(replicates),
                    "evidence_status": (
                        "held-hole comparison"
                        if metric.n_holes >= 8
                        else "insufficient independent holes / abstain"
                    ),
                    "distance_definition": (
                        "near verified logged contacts; signed along-hole "
                        "distance only, not perpendicular distance or true thickness"
                    ),
                    "training_definition": (
                        "existing models trained on contact-bearing holes; "
                        "predictions filtered post hoc by predeclared window"
                    ),
                }
            )
    return pd.DataFrame(rows)


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def refresh_analysis_freeze(
    freeze_path: str | Path,
    workspace: str | Path,
    *,
    metadata_only_paths: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refresh hashes while allowing only declared metadata-status changes."""

    freeze_file = Path(freeze_path)
    root = Path(workspace)
    frame = pd.read_csv(freeze_file)
    declared = {str(Path(item)).replace("\\", "/") for item in metadata_only_paths}
    audit: list[dict[str, object]] = []
    updated = frame.copy()
    updated["sha256"] = updated["sha256"].astype(str)
    unexpected: list[str] = []
    for index, row in frame.iterrows():
        relative = str(row["path"]).replace("\\", "/")
        path = root / Path(relative)
        if not path.exists():
            raise FileNotFoundError(path)
        current_hash = file_sha256(path)
        current_size = path.stat().st_size
        changed = str(row["sha256"]).upper() != current_hash
        if changed and relative not in declared:
            unexpected.append(relative)
        updated.at[index, "sha256"] = current_hash
        updated.at[index, "size_bytes"] = current_size
        audit.append(
            {
                "path": relative,
                "old_sha256": str(row["sha256"]).upper(),
                "new_sha256": current_hash,
                "hash_changed": changed,
                "change_class": (
                    "metadata_only_status_cleanup"
                    if changed and relative in declared
                    else "unchanged"
                ),
                "numerical_value_change": False,
                "verification": (
                    "declared status/provenance metadata cleanup; analytical "
                    "grade remains sourced only from the unchanged primary assay "
                    "and composite pipeline"
                    if changed
                    else "hash unchanged"
                ),
            }
        )
    if unexpected:
        raise ValueError(f"unexpected changed frozen inputs: {unexpected}")
    if any(bool(row["hash_changed"]) for row in audit):
        updated.to_csv(freeze_file, index=False)
    return updated, pd.DataFrame(audit)

