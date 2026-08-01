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
from sklearn.model_selection import GroupKFold

from .analysis_helpers import evaluate_fixed_safely
from .information_sensitivity import paired_model_evidence, primary_metrics
from .models import GeologyRegressionRegressor, GlobalMeanRegressor
from .validation import ValidationSplit, leave_one_batch_out_splits, leave_one_hole_out_splits, spatial_block_splits

ALPHA_GRID = (0.1, 1.0, 10.0)


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
    data: pd.DataFrame, outer_splits: Sequence[ValidationSplit], model: str,
    alphas: Sequence[float]=ALPHA_GRID,
) -> tuple[pd.DataFrame,pd.DataFrame]:
    frames=[]; audits=[]
    for outer in outer_splits:
        train=data.iloc[outer.train_index].copy().reset_index(drop=True)
        inner=_inner_splits(train)
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

