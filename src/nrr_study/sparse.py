"""Exact-linked sparse petrography, flake and Raman/XRD pilot.

The full assay table is always the grade authority. Report-derived descriptors
are never interpolated and are evaluated only with whole-hole holdout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


INDEX_MINERALS = (
    "sillimanite",
    "kyanite",
    "staurolite",
    "cordierite",
    "garnet",
    "orthopyroxene",
)


def _strip_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(column).strip() for column in result.columns]
    return result


def load_full_assay_grade(path: str | Path, censored_value: float = 0.025) -> pd.DataFrame:
    """Return the sole authorised sample-grade mapping."""

    assay = _strip_columns(pd.read_csv(path)).rename(
        columns={
            "BHID": "bhid",
            "SAMPLE NO": "assay_sample_id",
            "FROM": "assay_from_m",
            "TO": "assay_to_m",
            "GRAPHITIC CARBON": "tgc_raw",
            "BATCH_NUMBER": "batch_number",
        }
    )
    assay["bhid"] = assay["bhid"].astype(str).str.strip()
    assay["assay_sample_id"] = assay["assay_sample_id"].astype(str).str.strip()
    raw = assay["tgc_raw"].astype(str).str.strip()
    assay["tgc_censored"] = raw.str.fullmatch(r"<\s*0\.05", na=False)
    assay["tgc_pct"] = pd.to_numeric(
        raw.where(~assay["tgc_censored"], str(censored_value)),
        errors="coerce",
    )
    assay["assay_midpoint_md_m"] = (
        pd.to_numeric(assay["assay_from_m"], errors="coerce")
        + pd.to_numeric(assay["assay_to_m"], errors="coerce")
    ) / 2.0
    keep = [
        "assay_sample_id",
        "bhid",
        "assay_from_m",
        "assay_to_m",
        "assay_midpoint_md_m",
        "tgc_pct",
        "tgc_censored",
        "batch_number",
    ]
    if assay["assay_sample_id"].duplicated().any():
        raise ValueError("Full assay authority contains duplicate sample IDs")
    return assay.loc[:, keep]


def _join_grade(
    cohort: pd.DataFrame,
    assay: pd.DataFrame,
    key: str = "assay_sample_id",
) -> pd.DataFrame:
    merged = cohort.merge(
        assay,
        on=key,
        how="left",
        suffixes=("", "_assay"),
        validate="many_to_one",
    )
    merged["exact_grade_link"] = merged["tgc_pct"].notna()
    return merged


def _concatenate_report_text(frame: pd.DataFrame) -> pd.Series:
    fields = [
        "report_rock_type",
        "report_field_description",
        "reported_ppl_observation",
        "reported_xpl_observation",
        "reported_rl_observation",
        "reported_texture_microstructure",
        "reported_modal_mineralogy_text",
        "reported_paragenesis_or_remarks",
    ]
    available = [field for field in fields if field in frame.columns]
    return (
        frame[available]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )


def build_sparse_cohorts(
    *,
    assay_path: str | Path,
    observation_path: str | Path,
    modal_path: str | Path,
    flake_path: str | Path,
    analytical_path: str | Path,
    master_path: str | Path,
    interval_context_path: str | Path | None = None,
    censored_value: float = 0.025,
) -> dict[str, pd.DataFrame]:
    """Build exact-linked cohorts without flattening unmatched full assays."""

    assay = load_full_assay_grade(assay_path, censored_value=censored_value)
    observations = _strip_columns(pd.read_csv(observation_path))
    modal = _strip_columns(pd.read_csv(modal_path))
    flake = _strip_columns(pd.read_csv(flake_path))
    analytical = _strip_columns(pd.read_csv(analytical_path))
    master = _strip_columns(pd.read_csv(master_path))

    petro = observations.merge(
        modal[
            [
                "observation_record_id",
                "reported_graphite_modal_min_pct",
                "reported_graphite_modal_max_pct",
                "reported_graphite_modal_midpoint_pct",
            ]
        ],
        on="observation_record_id",
        how="left",
        validate="one_to_one",
    )
    report_text = _concatenate_report_text(petro)
    for mineral in INDEX_MINERALS:
        petro[f"reported_{mineral}_present"] = report_text.str.contains(
            rf"\b{mineral}\b", regex=True
        )
    petro["reported_index_mineral_any"] = petro[
        [f"reported_{name}_present" for name in INDEX_MINERALS]
    ].any(axis=1)
    petro["reported_index_mineral_count"] = petro[
        [f"reported_{name}_present" for name in INDEX_MINERALS]
    ].sum(axis=1)
    petro = _join_grade(petro, assay)

    master_link = master[
        [
            "sample_id",
            "assay_sample_id",
            "bhid",
            "from_m",
            "to_m",
            "workbook_lithology_code",
            "workbook_weathering_code",
        ]
    ].drop_duplicates("sample_id")
    flake = flake.merge(
        master_link,
        left_on="petrography_sample_id",
        right_on="sample_id",
        how="left",
        validate="one_to_one",
    )
    parts = flake[
        ["fine_small_medium_closed_pct", "large_closed_pct", "jumbo_closed_pct"]
    ].apply(pd.to_numeric, errors="coerce")
    parts = parts.clip(lower=0.0) + 0.5
    parts = parts.div(parts.sum(axis=1), axis=0)
    flake["ilr_fine_vs_large_jumbo"] = np.sqrt(2.0 / 3.0) * np.log(
        parts["fine_small_medium_closed_pct"]
        / np.sqrt(parts["large_closed_pct"] * parts["jumbo_closed_pct"])
    )
    flake = _join_grade(flake, assay)

    analytical = _join_grade(analytical, assay)
    analytical = analytical.rename(
        columns={
            "reported_raman_id_ig": "reported_id_ig",
            "reported_raman_g_fwhm_cm1": "reported_g_fwhm_cm1",
        }
    )

    if interval_context_path is not None and Path(interval_context_path).exists():
        context = _strip_columns(pd.read_csv(interval_context_path))
        context_key = next(
            (
                name
                for name in ("assay_sample_id", "sample_id", "SAMPLE_ID", "SAMPLE NO")
                if name in context.columns
            ),
            None,
        )
        if context_key:
            if context_key != "assay_sample_id":
                context = context.rename(columns={context_key: "assay_sample_id"})
            context_columns = [
                column
                for column in (
                    "assay_sample_id",
                    "canonical_lithology",
                    "canonical_domain",
                    "weathering",
                    "canonical_weathering",
                    "mid_rl",
                    "midpoint_rl_m",
                )
                if column in context.columns
            ]
            context = context[context_columns].drop_duplicates("assay_sample_id")
            for name, frame in {
                "petrography": petro,
                "flake": flake,
                "analytical": analytical,
            }.items():
                frame = frame.merge(
                    context,
                    on="assay_sample_id",
                    how="left",
                    validate="many_to_one",
                )
                if name == "petrography":
                    petro = frame
                elif name == "flake":
                    flake = frame
                else:
                    analytical = frame

    return {
        "petrography": petro,
        "modal": petro.loc[
            petro["reported_graphite_modal_midpoint_pct"].notna()
            & petro["exact_grade_link"]
        ].copy(),
        "flake": flake.loc[
            flake["ilr_fine_vs_large_jumbo"].notna() & flake["exact_grade_link"]
        ].copy(),
        "analytical": analytical.loc[analytical["exact_grade_link"]].copy(),
    }


def _prepare_features(
    frame: pd.DataFrame,
    descriptor_columns: list[str],
) -> tuple[list[str], list[str]]:
    numeric = ["assay_midpoint_md_m"]
    for candidate in ("mid_rl", "midpoint_rl_m"):
        if candidate in frame.columns and frame[candidate].notna().any():
            numeric.append(candidate)
            break
    numeric.extend(descriptor_columns)
    categorical: list[str] = []
    for candidates in (
        ("canonical_domain", "canonical_lithology", "verified_class_or_classes", "workbook_lithology_code"),
        ("canonical_weathering", "weathering", "verified_weathering_proxy_or_classes", "workbook_weathering_code"),
    ):
        chosen = next(
            (name for name in candidates if name in frame.columns and frame[name].notna().any()),
            None,
        )
        if chosen:
            categorical.append(chosen)
    return list(dict.fromkeys(numeric)), categorical


def _ridge_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            )
        )
    return Pipeline(
        [
            ("features", ColumnTransformer(transformers, remainder="drop")),
            ("regression", Ridge(alpha=1.0)),
        ]
    )


def leave_one_hole_out_predictions(
    frame: pd.DataFrame,
    descriptor_columns: Iterable[str],
    minimum_holes: int = 8,
) -> tuple[pd.DataFrame, str]:
    """Compare baseline and descriptor-augmented TGC models by held-out hole."""

    descriptors = list(descriptor_columns)
    data = frame.loc[
        frame["tgc_pct"].notna() & frame["bhid"].notna()
    ].copy()
    holes = sorted(data["bhid"].astype(str).unique())
    if len(holes) < minimum_holes:
        return pd.DataFrame(), "insufficient_independent_holes"
    baseline_num, categorical = _prepare_features(data, [])
    augmented_num, augmented_cat = _prepare_features(data, descriptors)
    rows: list[dict[str, object]] = []
    for hole in holes:
        test_mask = data["bhid"].astype(str).eq(hole)
        train = data.loc[~test_mask]
        test = data.loc[test_mask]
        if train.empty or test.empty:
            continue
        baseline = _ridge_pipeline(baseline_num, categorical)
        augmented = _ridge_pipeline(augmented_num, augmented_cat)
        baseline.fit(train, train["tgc_pct"])
        augmented.fit(train, train["tgc_pct"])
        pred_base = baseline.predict(test)
        pred_aug = augmented.predict(test)
        for index, actual, base_value, aug_value in zip(
            test.index, test["tgc_pct"], pred_base, pred_aug
        ):
            rows.append(
                {
                    "row_index": int(index),
                    "bhid": hole,
                    "actual_tgc_pct": float(actual),
                    "baseline_prediction": float(base_value),
                    "augmented_prediction": float(aug_value),
                    "baseline_abs_error": float(abs(actual - base_value)),
                    "augmented_abs_error": float(abs(actual - aug_value)),
                }
            )
    return pd.DataFrame(rows), "complete"


@dataclass(frozen=True)
class SparseFamilyResult:
    family: str
    records: int
    holes: int
    spearman_rho: float
    spearman_p: float
    baseline_mae: float
    augmented_mae: float
    delta_mae: float
    delta_ci_low: float
    delta_ci_high: float
    permutation_p: float
    conclusion: str


def _hole_bootstrap_delta(
    predictions: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> np.ndarray:
    by_hole = predictions.groupby("bhid").agg(
        baseline=("baseline_abs_error", "mean"),
        augmented=("augmented_abs_error", "mean"),
    )
    delta = by_hole["augmented"] - by_hole["baseline"]
    rng = np.random.default_rng(seed)
    values = delta.to_numpy(float)
    picks = rng.integers(0, len(values), size=(replicates, len(values)))
    return values[picks].mean(axis=1)


def _sign_flip_pvalue(values: np.ndarray, seed: int, replicates: int = 10000) -> float:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    observed = float(values.mean())
    signs = rng.choice((-1.0, 1.0), size=(replicates, len(values)))
    null = (signs * values).mean(axis=1)
    return float((1 + np.sum(null <= observed)) / (replicates + 1))


def evaluate_sparse_family(
    family: str,
    frame: pd.DataFrame,
    descriptor_columns: list[str],
    *,
    primary_descriptor: str,
    minimum_holes: int = 8,
    bootstrap_replicates: int = 2000,
    seed: int = 20260728,
) -> tuple[SparseFamilyResult, pd.DataFrame]:
    """Return a prospective whole-hole descriptor-family decision."""

    data = frame.loc[
        frame[descriptor_columns].notna().all(axis=1)
        & frame["tgc_pct"].notna()
        & frame["bhid"].notna()
    ].copy()
    records = len(data)
    holes = data["bhid"].nunique()
    if records >= 3 and data[primary_descriptor].nunique() > 1:
        correlation = spearmanr(
            data["tgc_pct"], data[primary_descriptor], nan_policy="omit"
        )
        rho, rho_p = float(correlation.statistic), float(correlation.pvalue)
    else:
        rho, rho_p = np.nan, np.nan
    predictions, state = leave_one_hole_out_predictions(
        data, descriptor_columns, minimum_holes=minimum_holes
    )
    if state != "complete" or predictions.empty:
        result = SparseFamilyResult(
            family=family,
            records=records,
            holes=holes,
            spearman_rho=rho,
            spearman_p=rho_p,
            baseline_mae=np.nan,
            augmented_mae=np.nan,
            delta_mae=np.nan,
            delta_ci_low=np.nan,
            delta_ci_high=np.nan,
            permutation_p=np.nan,
            conclusion="insufficient evidence / abstain",
        )
        return result, predictions
    by_hole = predictions.groupby("bhid").agg(
        baseline=("baseline_abs_error", "mean"),
        augmented=("augmented_abs_error", "mean"),
    )
    deltas = (by_hole["augmented"] - by_hole["baseline"]).to_numpy(float)
    bootstrap = _hole_bootstrap_delta(
        predictions, replicates=bootstrap_replicates, seed=seed
    )
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    p_value = _sign_flip_pvalue(deltas, seed=seed + 1)
    delta = float(deltas.mean())
    if high < 0.0 and p_value < 0.05:
        conclusion = "supported"
    else:
        conclusion = "unsupported"
    result = SparseFamilyResult(
        family=family,
        records=records,
        holes=holes,
        spearman_rho=rho,
        spearman_p=rho_p,
        baseline_mae=float(predictions["baseline_abs_error"].mean()),
        augmented_mae=float(predictions["augmented_abs_error"].mean()),
        delta_mae=delta,
        delta_ci_low=float(low),
        delta_ci_high=float(high),
        permutation_p=p_value,
        conclusion=conclusion,
    )
    return result, predictions


def run_sparse_module(
    cohorts: dict[str, pd.DataFrame],
    *,
    bootstrap_replicates: int = 2000,
    seed: int = 20260728,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Evaluate one predeclared descriptor family at a time."""

    specifications = [
        (
            "modal_graphite",
            cohorts["modal"],
            ["reported_graphite_modal_midpoint_pct"],
            "reported_graphite_modal_midpoint_pct",
        ),
        (
            "flake_ilr",
            cohorts["flake"],
            ["ilr_fine_vs_large_jumbo"],
            "ilr_fine_vs_large_jumbo",
        ),
        (
            "reported_index_mineral_assemblage",
            cohorts["petrography"],
            ["reported_index_mineral_any", "reported_index_mineral_count"],
            "reported_index_mineral_count",
        ),
        (
            "raman_xrd_reported_metrics",
            cohorts["analytical"],
            ["reported_d002_nm", "reported_id_ig"],
            "reported_id_ig",
        ),
    ]
    results: list[dict[str, object]] = []
    predictions: dict[str, pd.DataFrame] = {}
    for offset, (family, frame, descriptors, primary) in enumerate(specifications):
        available = [column for column in descriptors if column in frame.columns]
        if primary not in available:
            result = SparseFamilyResult(
                family=family,
                records=len(frame),
                holes=frame["bhid"].nunique() if "bhid" in frame else 0,
                spearman_rho=np.nan,
                spearman_p=np.nan,
                baseline_mae=np.nan,
                augmented_mae=np.nan,
                delta_mae=np.nan,
                delta_ci_low=np.nan,
                delta_ci_high=np.nan,
                permutation_p=np.nan,
                conclusion="insufficient evidence / abstain",
            )
            pred = pd.DataFrame()
        else:
            result, pred = evaluate_sparse_family(
                family,
                frame,
                available,
                primary_descriptor=primary,
                bootstrap_replicates=bootstrap_replicates,
                seed=seed + offset * 101,
            )
        results.append(result.__dict__)
        predictions[family] = pred
    return pd.DataFrame(results), predictions
