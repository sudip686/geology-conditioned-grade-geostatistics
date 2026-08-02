"""Additional post-analysis tests for method-first publication framing.

All model selection is nested inside the outer grouped folds. These analyses are
transparent post-analysis sensitivities and do not alter the frozen covariance
gate or retroactively become preregistered hypotheses.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from .analysis_helpers import evaluate_fixed_safely
from .information_sensitivity import paired_model_evidence, primary_metrics
from .models import GeologyRegressionRegressor, GlobalMeanRegressor, PredictionResult
from .reviewer_revision import (
    BUFFER_MULTIPLIERS,
    ORIGIN_FRACTIONS,
    ROTATIONS_DEG,
    rotated_origin_splits,
)
from .validation import ValidationSplit, leave_one_batch_out_splits, leave_one_hole_out_splits, spatial_block_splits

ALPHA_GRID = (0.1, 1.0, 10.0)
QUADRATIC_COORDINATE_FIELDS = ("mid_easting", "mid_northing", "mid_rl")
QUADRATIC_DEGREE = 2


class BoundedQuadraticRegressionRegressor:
    """Support-weighted ridge model with a fixed quadratic coordinate basis.

    This is a reviewer-motivated, post-analysis non-spatial-covariance
    sensitivity. The degree is fixed at two and only the ridge penalty is
    selected inside grouped training folds. It is not a variogram, kriging,
    anisotropy, or directional-continuity model.
    """

    def __init__(
        self,
        *,
        categorical_cols: Sequence[str] = (),
        coordinate_cols: Sequence[str] = QUADRATIC_COORDINATE_FIELDS,
        alpha: float = 1.0,
        degree: int = QUADRATIC_DEGREE,
        weight_col: str | None = "support_m",
    ) -> None:
        if degree != QUADRATIC_DEGREE:
            raise ValueError("bounded reviewer sensitivity requires degree=2")
        if alpha < 0:
            raise ValueError("alpha must be nonnegative")
        self.categorical_cols = tuple(categorical_cols)
        self.coordinate_cols = tuple(coordinate_cols)
        self.alpha = float(alpha)
        self.degree = int(degree)
        self.weight_col = weight_col

    def _make_pipeline(self) -> Pipeline:
        numeric = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "quadratic",
                    PolynomialFeatures(degree=self.degree, include_bias=False),
                ),
            ]
        )
        transformers: list[tuple[str, Pipeline, list[str]]] = [
            ("coordinates", numeric, list(self.coordinate_cols))
        ]
        if self.categorical_cols:
            categorical = Pipeline(
                [
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    (
                        "encode",
                        OneHotEncoder(
                            handle_unknown="ignore", sparse_output=False
                        ),
                    ),
                ]
            )
            transformers.insert(
                0, ("geology", categorical, list(self.categorical_cols))
            )
        return Pipeline(
            [
                (
                    "features",
                    ColumnTransformer(transformers, remainder="drop"),
                ),
                ("regressor", Ridge(alpha=max(self.alpha, 1e-12))),
            ]
        )

    def fit(
        self, data: pd.DataFrame, target_col: str = "tgc_pct"
    ) -> "BoundedQuadraticRegressionRegressor":
        features = self.categorical_cols + self.coordinate_cols
        missing = [
            column for column in (target_col, *features) if column not in data
        ]
        if missing:
            raise ValueError(f"missing required columns: {missing}")
        target = data[target_col].to_numpy(float)
        if not np.all(np.isfinite(target)):
            raise ValueError("training target contains non-finite values")
        self.pipeline_ = self._make_pipeline()
        fit_kwargs: dict[str, np.ndarray] = {}
        if self.weight_col is not None and self.weight_col in data:
            weights = data[self.weight_col].to_numpy(float)
            if not np.all(np.isfinite(weights) & (weights > 0)):
                raise ValueError("sample weights must be finite and positive")
            fit_kwargs["regressor__sample_weight"] = weights
        feature_frame = data.loc[:, list(features)]
        self.pipeline_.fit(feature_frame, target, **fit_kwargs)
        fitted = np.asarray(self.pipeline_.predict(feature_frame), dtype=float)
        residual = target - fitted
        if fit_kwargs:
            self.residual_variance_ = float(
                np.average(
                    residual * residual,
                    weights=fit_kwargs["regressor__sample_weight"],
                )
            )
        else:
            self.residual_variance_ = float(np.mean(residual * residual))
        coordinate_count = len(self.coordinate_cols)
        self.quadratic_term_count_ = int(
            coordinate_count + coordinate_count * (coordinate_count + 1) / 2
        )
        return self

    def predict(self, data: pd.DataFrame) -> PredictionResult:
        if not hasattr(self, "pipeline_"):
            raise RuntimeError("model is not fitted")
        features = self.categorical_cols + self.coordinate_cols
        missing = [column for column in features if column not in data]
        if missing:
            raise ValueError(f"missing required columns: {missing}")
        mean = np.asarray(
            self.pipeline_.predict(data.loc[:, list(features)]), dtype=float
        )
        return PredictionResult(
            mean=mean,
            variance=np.full(len(data), self.residual_variance_, dtype=float),
            success=np.isfinite(mean),
        )


def _factory(model: str, alpha: float) -> Callable[[], object]:
    if model == "coordinate_trend":
        return lambda: GeologyRegressionRegressor(
            categorical_cols=(), numeric_cols=("mid_easting","mid_northing","mid_rl"),
            alpha=alpha,
        )
    if model == "lithology_only_nested":
        return lambda: GeologyRegressionRegressor(
            categorical_cols=("canonical_lithology","grsc_subtype"), numeric_cols=(),
            alpha=alpha,
        )
    if model == "lithology_spatial":
        return lambda: GeologyRegressionRegressor(
            categorical_cols=("canonical_lithology","grsc_subtype"),
            numeric_cols=("mid_easting","mid_northing","mid_rl"), alpha=alpha,
        )
    if model == "quadratic_coordinate_trend":
        return lambda: BoundedQuadraticRegressionRegressor(alpha=alpha)
    if model == "lithology_quadratic_coordinate":
        return lambda: BoundedQuadraticRegressionRegressor(
            categorical_cols=("canonical_lithology", "grsc_subtype"),
            alpha=alpha,
        )
    raise ValueError(model)


def _inner_splits(train: pd.DataFrame, n_splits: int=5) -> tuple[ValidationSplit,...]:
    groups=train["BHID"].astype(str).to_numpy()
    unique=np.unique(groups)
    if len(unique)<2:
        return ()
    splitter=GroupKFold(n_splits=min(n_splits,len(unique)))
    result=[]
    dummy=np.zeros(len(train))
    for i,(tr,te) in enumerate(splitter.split(dummy,groups=groups),start=1):
        result.append(ValidationSplit("inner_grouped",f"G{i}",tr,te,np.asarray([],dtype=int)))
    return tuple(result)


def _hole_balanced_mae(predictions: pd.DataFrame) -> float:
    ok=predictions.loc[predictions["success"].astype(bool)].copy()
    if ok.empty:
        return float("inf")
    ok["ae"]=(ok["prediction"]-ok["truth"]).abs()
    ok["weighted_ae"]=ok["ae"]*ok["weight"]
    by=ok.groupby("hole",sort=True).agg(mass=("weighted_ae","sum"),support=("weight","sum"))
    return float((by["mass"]/by["support"]).mean())


def nested_grouped_predictions(
    data: pd.DataFrame,
    outer_splits: Sequence[ValidationSplit],
    model: str,
    alphas: Sequence[float] = ALPHA_GRID,
    inner_grouped_folds: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames=[]; audits=[]
    for outer in outer_splits:
        train=data.iloc[outer.train_index].copy().reset_index(drop=True)
        inner=_inner_splits(train, n_splits=inner_grouped_folds)
        scores=[]
        for alpha in alphas:
            pred=evaluate_fixed_safely(train,inner,_factory(model,float(alpha)))
            scores.append((float(alpha),_hole_balanced_mae(pred)))
        best_alpha,best_score=min(scores,key=lambda item:(item[1],item[0]))
        outer_pred=evaluate_fixed_safely(data,[outer],_factory(model,best_alpha)).assign(model=model)
        frames.append(outer_pred)
        for alpha,score in scores:
            audits.append({
                "model":model,"scheme":outer.scheme,"fold_id":outer.fold_id,
                "alpha":alpha,"inner_hole_balanced_mae":score,
                "selected":bool(alpha==best_alpha),"inner_group_count":len(inner),
                "outer_train_rows":len(outer.train_index),"outer_test_rows":len(outer.test_index),
            })
    return pd.concat(frames,ignore_index=True),pd.DataFrame(audits)



def _paired_evidence_checked(
    predictions: pd.DataFrame, *, conditioned: str, comparator: str,
    replicates: int, seed: int,
) -> pd.DataFrame:
    """Run paired evidence only after exact prediction-universe checks."""
    frames = {}
    for model in (conditioned, comparator):
        frame = predictions.loc[predictions["model"].eq(model)].copy()
        keys = ["scheme", "row_index"]
        if frame.duplicated(keys).any():
            raise AssertionError(f"duplicate paired keys for {model}")
        frames[model] = frame
    left, right = frames[conditioned], frames[comparator]
    key_cols = ["scheme", "row_index"]
    check = left.merge(right, on=key_cols, how="outer", suffixes=("_conditioned", "_comparator"), indicator=True)
    if not check["_merge"].eq("both").all():
        raise AssertionError("paired prediction universes differ")
    for field in ("truth", "weight", "hole"):
        if field == "hole":
            equal = check[f"{field}_conditioned"].astype(str).eq(check[f"{field}_comparator"].astype(str))
        else:
            equal = np.isclose(check[f"{field}_conditioned"], check[f"{field}_comparator"], equal_nan=True)
        if not bool(np.all(equal)):
            raise AssertionError(f"paired {field} mismatch")
    out = paired_model_evidence(
        predictions, conditioned=conditioned, comparator=comparator,
        replicates=replicates, seed=seed,
    )
    diagnostics = []
    for scheme, group in check.groupby("scheme", sort=True):
        diagnostics.append({
            "scheme": scheme,
            "paired_rows": len(group),
            "paired_holes": group["hole_conditioned"].nunique(),
            "joint_success_rate": float((group["success_conditioned"].astype(bool) & group["success_comparator"].astype(bool)).mean()),
            "conditioned_failure_rate": float(1.0 - group["success_conditioned"].astype(bool).mean()),
            "comparator_failure_rate": float(1.0 - group["success_comparator"].astype(bool).mean()),
        })
    return out.merge(pd.DataFrame(diagnostics), on="scheme", how="left", validate="one_to_one")

def coordinate_model_analysis(primary: pd.DataFrame, *, bootstraps:int=2000, seed:int=20260728):
    splits=(*leave_one_hole_out_splits(primary),*spatial_block_splits(
        primary,n_blocks=5,block_col="northing_block",
        buffer_distance=float(primary["spatial_buffer_m"].median()),
    ))
    prediction_frames=[]; tuning_frames=[]
    for model in ("coordinate_trend","lithology_only_nested","lithology_spatial"):
        pred,audit=nested_grouped_predictions(primary,splits,model)
        prediction_frames.append(pred);tuning_frames.append(audit)
    predictions=pd.concat(prediction_frames,ignore_index=True)
    comparisons=(
        ("lithology_only_nested","coordinate_trend","lithology_minus_coordinate"),
        ("lithology_spatial","coordinate_trend","lithology_spatial_minus_coordinate"),
        ("lithology_spatial","lithology_only_nested","spatial_increment_given_lithology"),
    )
    evidence=[]
    for conditioned,comparator,label in comparisons:
        frame=_paired_evidence_checked(
            predictions, conditioned=conditioned, comparator=comparator,
            replicates=bootstraps, seed=seed,
        )
        frame["comparison"]=label;evidence.append(frame)
    return predictions,primary_metrics(predictions),pd.concat(evidence,ignore_index=True),pd.concat(tuning_frames,ignore_index=True)


def quadratic_coordinate_model_analysis(
    primary: pd.DataFrame,
    *,
    bootstraps: int = 20_000,
    seed: int = 20260728,
    alphas: Sequence[float] = ALPHA_GRID,
    inner_grouped_folds: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate the fixed degree-two coordinate sensitivity in primary folds."""
    splits = (
        *leave_one_hole_out_splits(primary),
        *spatial_block_splits(
            primary,
            n_blocks=5,
            block_col="northing_block",
            buffer_distance=float(primary["spatial_buffer_m"].median()),
        ),
    )
    prediction_frames: list[pd.DataFrame] = []
    tuning_frames: list[pd.DataFrame] = []
    for model in (
        "quadratic_coordinate_trend",
        "lithology_quadratic_coordinate",
    ):
        predictions, tuning = nested_grouped_predictions(
            primary,
            splits,
            model,
            alphas=alphas,
            inner_grouped_folds=inner_grouped_folds,
        )
        prediction_frames.append(predictions)
        tuning_frames.append(tuning)
    combined = pd.concat(prediction_frames, ignore_index=True)
    evidence = _paired_evidence_checked(
        combined,
        conditioned="lithology_quadratic_coordinate",
        comparator="quadratic_coordinate_trend",
        replicates=bootstraps,
        seed=seed,
    )
    evidence["comparison"] = (
        "lithology_quadratic_coordinate_minus_quadratic_coordinate"
    )
    evidence["analysis_role"] = (
        "reviewer_motivated_post_analysis_non_decision_sensitivity"
    )
    evidence["polynomial_degree"] = QUADRATIC_DEGREE
    evidence["coordinate_term_count"] = 9
    evidence["changes_frozen_covariance_gate"] = False
    return (
        combined,
        primary_metrics(combined),
        evidence,
        pd.concat(tuning_frames, ignore_index=True),
    )


def _paired_fold_metrics(
    predictions: pd.DataFrame,
    *,
    conditioned: str,
    comparator: str,
) -> dict[str, object]:
    """Calculate paired hole-balanced metrics on one outer fold."""
    keys = ["scheme", "fold_id", "row_index"]
    frames: dict[str, pd.DataFrame] = {}
    for model in (conditioned, comparator):
        frame = predictions.loc[predictions["model"].eq(model)].copy()
        if frame.duplicated(keys).any():
            raise AssertionError(f"duplicate prediction keys for {model}")
        frames[model] = frame
    joined = frames[conditioned].merge(
        frames[comparator],
        on=keys,
        how="outer",
        suffixes=("_conditioned", "_comparator"),
        indicator=True,
        validate="one_to_one",
    )
    if not joined["_merge"].eq("both").all():
        raise AssertionError("paired prediction universes differ")
    if not np.all(
        np.isclose(
            joined["truth_conditioned"],
            joined["truth_comparator"],
            equal_nan=True,
        )
    ):
        raise AssertionError("paired truth mismatch")
    if not np.all(
        np.isclose(
            joined["weight_conditioned"],
            joined["weight_comparator"],
            equal_nan=True,
        )
    ):
        raise AssertionError("paired weight mismatch")
    if not joined["hole_conditioned"].astype(str).eq(
        joined["hole_comparator"].astype(str)
    ).all():
        raise AssertionError("paired hole mismatch")
    success = (
        joined["success_conditioned"].astype(bool)
        & joined["success_comparator"].astype(bool)
        & np.isfinite(joined["prediction_conditioned"])
        & np.isfinite(joined["prediction_comparator"])
    )
    paired = joined.loc[success].copy()
    if paired.empty:
        return {
            "conditioned_mae": float("nan"),
            "comparator_mae": float("nan"),
            "delta_mae": float("nan"),
            "paired_prediction_count": 0,
            "attempted_prediction_count": int(len(joined)),
            "paired_hole_count": 0,
            "joint_success_rate": 0.0,
        }
    paired["conditioned_error_mass"] = (
        np.abs(
            paired["prediction_conditioned"] - paired["truth_conditioned"]
        )
        * paired["weight_conditioned"]
    )
    paired["comparator_error_mass"] = (
        np.abs(
            paired["prediction_comparator"] - paired["truth_conditioned"]
        )
        * paired["weight_conditioned"]
    )
    per_hole = paired.groupby("hole_conditioned", sort=True).agg(
        conditioned_error_mass=("conditioned_error_mass", "sum"),
        comparator_error_mass=("comparator_error_mass", "sum"),
        support=("weight_conditioned", "sum"),
    )
    conditioned_mae = float(
        np.mean(per_hole["conditioned_error_mass"] / per_hole["support"])
    )
    comparator_mae = float(
        np.mean(per_hole["comparator_error_mass"] / per_hole["support"])
    )
    return {
        "conditioned_mae": conditioned_mae,
        "comparator_mae": comparator_mae,
        "delta_mae": conditioned_mae - comparator_mae,
        "paired_prediction_count": int(len(paired)),
        "attempted_prediction_count": int(len(joined)),
        "paired_hole_count": int(len(per_hole)),
        "joint_success_rate": float(success.mean()),
    }


def quadratic_spatial_design_robustness(
    primary: pd.DataFrame,
    *,
    rotations: Sequence[float] = ROTATIONS_DEG,
    origins: Sequence[float] = ORIGIN_FRACTIONS,
    buffer_multipliers: Sequence[float] = BUFFER_MULTIPLIERS,
    alphas: Sequence[float] = ALPHA_GRID,
    inner_grouped_folds: int = 5,
    n_blocks: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply the matched quadratic comparison to grade-blind spatial designs.

    The repeated designs reuse the same holes and are reported as dependent
    validation-geometry sensitivities. Rotations are fold-construction devices
    only and carry no directional or anisotropy interpretation.
    """
    median_nn = float(primary["spatial_buffer_m"].median())
    block_rows: list[dict[str, object]] = []
    design_rows: list[dict[str, object]] = []
    tuning_frames: list[pd.DataFrame] = []
    for rotation in rotations:
        for origin_index, origin in enumerate(origins, start=1):
            for buffer_multiplier in buffer_multipliers:
                buffer_m = float(buffer_multiplier) * median_nn
                design_id = (
                    f"R{float(rotation):g}_O{origin_index}_"
                    f"B{float(buffer_multiplier):g}"
                )
                splits = rotated_origin_splits(
                    primary,
                    rotation_deg=float(rotation),
                    origin_fraction=float(origin),
                    buffer_m=buffer_m,
                    n_blocks=n_blocks,
                )
                model_predictions: list[pd.DataFrame] = []
                selected_alphas: dict[tuple[str, str], float] = {}
                for model in (
                    "quadratic_coordinate_trend",
                    "lithology_quadratic_coordinate",
                ):
                    predictions, tuning = nested_grouped_predictions(
                        primary,
                        splits,
                        model,
                        alphas=alphas,
                        inner_grouped_folds=inner_grouped_folds,
                    )
                    predictions["design_id"] = design_id
                    model_predictions.append(predictions)
                    tuning = tuning.assign(
                        design_id=design_id,
                        rotation_deg=float(rotation),
                        origin_index=int(origin_index),
                        origin_fraction=float(origin),
                        buffer_multiplier=float(buffer_multiplier),
                        buffer_m=buffer_m,
                    )
                    tuning_frames.append(tuning)
                    selected = tuning.loc[tuning["selected"].astype(bool)]
                    for record in selected.itertuples(index=False):
                        selected_alphas[(model, str(record.fold_id))] = float(
                            record.alpha
                        )
                combined = pd.concat(model_predictions, ignore_index=True)
                local_block_rows: list[dict[str, object]] = []
                for split in splits:
                    fold_predictions = combined.loc[
                        combined["fold_id"].eq(split.fold_id)
                    ].copy()
                    metrics = _paired_fold_metrics(
                        fold_predictions,
                        conditioned="lithology_quadratic_coordinate",
                        comparator="quadratic_coordinate_trend",
                    )
                    record = {
                        "design_id": design_id,
                        "rotation_deg": float(rotation),
                        "origin_index": int(origin_index),
                        "origin_fraction": float(origin),
                        "buffer_multiplier": float(buffer_multiplier),
                        "buffer_m": buffer_m,
                        "block_id": split.fold_id,
                        "n_blocks": int(n_blocks),
                        "training_interval_count": int(len(split.train_index)),
                        "test_interval_count": int(len(split.test_index)),
                        "buffered_out_interval_count": int(
                            len(split.buffered_out_index)
                        ),
                        "training_hole_count": int(
                            primary.iloc[split.train_index]["BHID"].nunique()
                        ),
                        "test_hole_count": int(
                            primary.iloc[split.test_index]["BHID"].nunique()
                        ),
                        "buffered_out_hole_count": int(
                            primary.iloc[split.buffered_out_index][
                                "BHID"
                            ].nunique()
                        ),
                        "coordinate_alpha": selected_alphas[
                            ("quadratic_coordinate_trend", split.fold_id)
                        ],
                        "geology_coordinate_alpha": selected_alphas[
                            ("lithology_quadratic_coordinate", split.fold_id)
                        ],
                        **metrics,
                        "sign_favours_geology": bool(
                            np.isfinite(metrics["delta_mae"])
                            and float(metrics["delta_mae"]) < 0
                        ),
                        "analysis_role": (
                            "reviewer_motivated_post_analysis_"
                            "dependent_validation_geometry_sensitivity"
                        ),
                        "partition_basis": (
                            "grade_blind_rotated_hole_centre_projection_"
                            "with_rank_shifted_contiguous_cuts"
                        ),
                        "directional_or_anisotropy_interpretation": False,
                        "changes_frozen_covariance_gate": False,
                    }
                    block_rows.append(record)
                    local_block_rows.append(record)
                deltas = np.asarray(
                    [row["delta_mae"] for row in local_block_rows], dtype=float
                )
                finite = deltas[np.isfinite(deltas)]
                if len(finite) != n_blocks:
                    leave_one_out = np.asarray([], dtype=float)
                else:
                    leave_one_out = np.asarray(
                        [
                            np.mean(np.delete(finite, index))
                            for index in range(len(finite))
                        ]
                    )
                design_rows.append(
                    {
                        "design_id": design_id,
                        "rotation_deg": float(rotation),
                        "origin_index": int(origin_index),
                        "origin_fraction": float(origin),
                        "buffer_multiplier": float(buffer_multiplier),
                        "buffer_m": buffer_m,
                        "n_blocks": int(n_blocks),
                        "n_effective_blocks": int(len(finite)),
                        "equal_block_mean_delta": (
                            float(np.mean(finite)) if len(finite) else float("nan")
                        ),
                        "median_block_delta": (
                            float(np.median(finite))
                            if len(finite)
                            else float("nan")
                        ),
                        "sign_concordance": (
                            float(np.mean(finite < 0))
                            if len(finite)
                            else float("nan")
                        ),
                        "leave_one_block_out_min": (
                            float(np.min(leave_one_out))
                            if len(leave_one_out)
                            else float("nan")
                        ),
                        "leave_one_block_out_max": (
                            float(np.max(leave_one_out))
                            if len(leave_one_out)
                            else float("nan")
                        ),
                        "joint_success_rate": float(
                            np.average(
                                [
                                    row["joint_success_rate"]
                                    for row in local_block_rows
                                ],
                                weights=[
                                    row["attempted_prediction_count"]
                                    for row in local_block_rows
                                ],
                            )
                        ),
                        "inference_label": (
                            "descriptive dependent-partition sensitivity; "
                            "designs are not independent experiments"
                        ),
                        "directional_or_anisotropy_interpretation": False,
                        "changes_frozen_covariance_gate": False,
                    }
                )
    per_block = pd.DataFrame(block_rows)
    per_design = pd.DataFrame(design_rows)
    effects = per_design["equal_block_mean_delta"].to_numpy(float)
    finite_effects = effects[np.isfinite(effects)]
    expected_designs = (
        len(tuple(rotations))
        * len(tuple(origins))
        * len(tuple(buffer_multipliers))
    )
    overall = pd.DataFrame(
        [
            {
                "design_count_expected": int(expected_designs),
                "design_count_completed": int(len(per_design)),
                "effective_design_count": int(len(finite_effects)),
                "median_design_delta": (
                    float(np.median(finite_effects))
                    if len(finite_effects)
                    else float("nan")
                ),
                "minimum_design_delta": (
                    float(np.min(finite_effects))
                    if len(finite_effects)
                    else float("nan")
                ),
                "maximum_design_delta": (
                    float(np.max(finite_effects))
                    if len(finite_effects)
                    else float("nan")
                ),
                "fraction_designs_favouring_geology": (
                    float(np.mean(finite_effects < 0))
                    if len(finite_effects)
                    else float("nan")
                ),
                "median_reference_distance_m": median_nn,
                "polynomial_degree": QUADRATIC_DEGREE,
                "coordinate_term_count": 9,
                "ridge_alpha_grid": ";".join(str(value) for value in alphas),
                "inner_grouped_folds": int(inner_grouped_folds),
                "analysis_role": (
                    "reviewer_motivated_post_analysis_"
                    "dependent_validation_geometry_sensitivity"
                ),
                "inference_label": (
                    "descriptive only; 60 designs reuse the same 100 holes"
                ),
                "directional_or_anisotropy_interpretation": False,
                "changes_frozen_covariance_gate": False,
            }
        ]
    )
    tuning_audit = (
        pd.concat(tuning_frames, ignore_index=True)
        if tuning_frames
        else pd.DataFrame()
    )
    return per_block, per_design, overall, tuning_audit


def paired_lithology_vs_idw(lithology_predictions: pd.DataFrame, idw_predictions: pd.DataFrame, *, bootstraps:int=2000, seed:int=20260728):
    joined=pd.concat([
        lithology_predictions.assign(model="lithology_only"),
        idw_predictions.loc[
            idw_predictions["model"].eq("idw")
            & idw_predictions["scheme"].isin(lithology_predictions["scheme"].unique())
        ],
    ],ignore_index=True)
    out=_paired_evidence_checked(
        joined, conditioned="lithology_only", comparator="idw",
        replicates=bootstraps, seed=seed,
    )
    out["comparison"]="lithology_minus_idw"
    return out


def covariance_threshold_sensitivity(gate: dict) -> pd.DataFrame:
    base = {"range_ratio": 3.0, "range_cv": 0.5, "nugget_fraction": 0.8, "normalized_rmse": 0.35}
    observed = float(gate["actual_residual_short_lag_score"])
    frozen_score_threshold = float(gate["synthetic"]["threshold"])
    independent_score_threshold = 0.0392879997
    rows = []
    for label, maximum_factor, minimum_factor in (("strict", 0.8, 1.2), ("primary", 1.0, 1.0), ("relaxed", 1.2, 0.8)):
        limits = {k: v * maximum_factor for k, v in base.items()}
        components = {}
        for family, key in (("downhole", "downhole_stability"), ("residual_omnidirectional", "residual_omnidirectional_stability")):
            obs = gate[key]
            components[family] = (
                obs["range_ratio"] <= limits["range_ratio"]
                and obs["range_cv"] <= limits["range_cv"]
                and obs["worst_nugget_fraction"] <= limits["nugget_fraction"]
                and obs["worst_normalized_rmse"] <= limits["normalized_rmse"]
                and obs["successful_fits"] >= 4 and obs["supported_bins"] >= 4
            )
        frozen_threshold = frozen_score_threshold * minimum_factor
        independent_threshold = independent_score_threshold * minimum_factor
        frozen_score_pass = observed >= frozen_threshold
        independent_score_pass = observed >= independent_threshold
        synthetic_pass = bool(gate["synthetic"]["meets_targets"])
        rows.append({
            "threshold_set": label, "maximum_criterion_factor": maximum_factor,
            "minimum_score_factor": minimum_factor,
            **{f"max_{k}": v for k, v in limits.items()},
            "observed_short_lag_score": observed,
            "frozen_score_threshold": frozen_threshold,
            "independent_score_threshold": independent_threshold,
            "frozen_score_pass": frozen_score_pass,
            "independent_score_pass": independent_score_pass,
            "downhole_stability_pass": components["downhole"],
            "residual_stability_pass": components["residual_omnidirectional"],
            "synthetic_target_pass": synthetic_pass,
            "complete_covariance_eligibility_pass": bool(
                components["residual_omnidirectional"] and frozen_score_pass
                and independent_score_pass and synthetic_pass
            ),
            "interpretation": "post-analysis robustness sensitivity; thresholds were not optimized and operating targets were held fixed",
        })
    return pd.DataFrame(rows)

def batch_sensitivity(primary: pd.DataFrame, *, bootstraps:int=2000, seed:int=20260728):
    summary=(primary.groupby(["BATCH_NUMBER","canonical_lithology"],sort=True)
        .apply(lambda g:pd.Series({"rows":len(g),"holes":g["BHID"].nunique(),"support_m":g["support_m"].sum(),"length_weighted_mean_tgc_pct":np.average(g["tgc_pct"],weights=g["support_m"])}),include_groups=False).reset_index())
    counts=primary.groupby("BATCH_NUMBER",sort=True).size()
    largest=str(counts.idxmax())
    subset=primary.loc[~primary["BATCH_NUMBER"].astype(str).eq(largest)].copy().reset_index(drop=True)
    splits=(*leave_one_hole_out_splits(subset),*spatial_block_splits(subset,n_blocks=5,block_col="northing_block",buffer_distance=float(primary["spatial_buffer_m"].median())))
    global_pred=evaluate_fixed_safely(subset,splits,lambda:GlobalMeanRegressor()).assign(model="global_mean")
    lith_pred=evaluate_fixed_safely(subset,splits,_factory("lithology_only_nested",1e-6)).assign(model="lithology_only")
    predictions=pd.concat([global_pred,lith_pred],ignore_index=True)
    paired=paired_model_evidence(predictions,conditioned="lithology_only",comparator="global_mean",replicates=bootstraps,seed=seed)
    paired["comparison"]="lithology_minus_global_excluding_largest_batch"
    paired["excluded_batch"]=largest;paired["remaining_rows"]=len(subset);paired["remaining_holes"]=subset["BHID"].nunique()
    return summary,predictions,primary_metrics(predictions),paired


def leave_one_batch_out_evidence(
    primary: pd.DataFrame, *, bootstraps: int = 20_000, seed: int = 20260728,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate lithology while withholding complete analytical batches."""
    splits = leave_one_batch_out_splits(primary)
    global_pred = evaluate_fixed_safely(primary, splits, lambda: GlobalMeanRegressor()).assign(model="global_mean")
    lith_pred = evaluate_fixed_safely(primary, splits, _factory("lithology_only_nested", 1e-6)).assign(model="lithology_only")
    predictions = pd.concat([global_pred, lith_pred], ignore_index=True)
    if len(splits) != primary["BATCH_NUMBER"].nunique():
        raise AssertionError("leave-one-batch-out fold count mismatch")
    required = predictions.loc[predictions["success"].astype(bool)].copy()
    batch_by_row = primary[["BATCH_NUMBER"]].reset_index().rename(columns={"index": "row_index"})
    required = required.merge(batch_by_row, on="row_index", how="left", validate="many_to_one")
    required["absolute_error"] = (required["prediction"] - required["truth"]).abs()
    required["weighted_absolute_error"] = required["absolute_error"] * required["weight"]
    per_hole = required.groupby(["BATCH_NUMBER", "model", "hole"], sort=True).agg(
        error_mass=("weighted_absolute_error", "sum"), support=("weight", "sum")
    ).reset_index()
    per_hole["hole_mae"] = per_hole["error_mass"] / per_hole["support"]
    per_batch = per_hole.groupby(["BATCH_NUMBER", "model"], sort=True).agg(
        mae=("hole_mae", "mean"), holes=("hole", "nunique")
    ).reset_index()
    wide = per_batch.pivot(index="BATCH_NUMBER", columns="model", values="mae").dropna().reset_index()
    wide["lithology_minus_global_mae"] = wide["lithology_only"] - wide["global_mean"]
    wide["holes"] = wide["BATCH_NUMBER"].map(per_batch.groupby("BATCH_NUMBER")["holes"].max())
    values = wide["lithology_minus_global_mae"].to_numpy(float)
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, len(values), size=(bootstraps, len(values)))].mean(axis=1)
    summary = pd.DataFrame([{
        "comparison": "lithology_minus_global_leave_one_batch_out",
        "estimate": float(values.mean()), "lower": float(np.quantile(boot, 0.025)),
        "upper": float(np.quantile(boot, 0.975)), "confidence_level": 0.95,
        "batches": int(len(values)), "batches_favouring_lithology": int((values < 0).sum()),
        "bootstrap_replicates": int(bootstraps),
        "weighting": "support within hole; equal holes within batch; equal batches",
        "limitation": "batch, hole, and location are confounded; no laboratory effect is identified",
    }])
    return wide.sort_values("BATCH_NUMBER").reset_index(drop=True), summary, predictions

