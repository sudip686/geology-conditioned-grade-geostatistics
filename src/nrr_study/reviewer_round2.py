"""Verified second-round extensions for operating-region interpretation."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from .analysis_helpers import evaluate_fixed_safely
from .framework_extension import _paired_evidence_checked
from .information_sensitivity import primary_metrics
from .models import GeologyRegressionRegressor
from .reviewer_revision import geological_contrast_preservation
from .validation import leave_one_hole_out_splits, spatial_block_splits

SCHEMES=("leave_one_hole_out","northing_block_buffered")
def primary_splits(data):
 return (*leave_one_hole_out_splits(data),*spatial_block_splits(data,n_blocks=5,block_col="northing_block",buffer_distance=float(data["spatial_buffer_m"].median())))
def factory(categories,numeric=(),alpha=1e-6):
 return lambda:GeologyRegressionRegressor(categorical_cols=tuple(categories),numeric_cols=tuple(numeric),alpha=alpha)

def hierarchy_ablation(primary,existing_predictions,*,bootstraps=20_000,seed=20260728):
 data=primary.copy().reset_index(drop=True)
 data["graphitic_binary"]=np.where(data["canonical_lithology"].astype(str).eq("graphitic_schist"),"graphitic","non_graphitic")
 splits=primary_splits(data)
 frames=[]
 specs=(("graphitic_binary",("graphitic_binary",)),("canonical_lithology_only",("canonical_lithology",)))
 for name,cats in specs:
  frames.append(evaluate_fixed_safely(data,splits,factory(cats)).assign(model=name))
 new=pd.concat(frames,ignore_index=True)
 keep=existing_predictions.loc[existing_predictions["scheme"].isin(SCHEMES)&existing_predictions["model"].isin(["lithology_only","geology_only","partial_pooling"])].copy()
 combined=pd.concat([new,keep],ignore_index=True)
 comparisons=(("canonical_lithology_only","graphitic_binary","canonical_minus_binary"),("lithology_only","canonical_lithology_only","subtype_increment"),("geology_only","lithology_only","weathering_increment"),("partial_pooling","geology_only","context_increment"))
 evidence=[]
 for conditioned,comparator,label in comparisons:
  out=_paired_evidence_checked(combined,conditioned=conditioned,comparator=comparator,replicates=bootstraps,seed=seed+len(evidence)*101)
  out["comparison"]=label;evidence.append(out)
 support=[]
 for name,cols in (("graphitic_binary",["graphitic_binary"]),("canonical_lithology_only",["canonical_lithology"]),("lithology_only",["canonical_lithology","grsc_subtype"]),("geology_only",["canonical_lithology","grsc_subtype","weathering"]),("partial_pooling",["canonical_lithology","grsc_subtype","weathering"])):
  levels=sum(data[c].astype(str).nunique() for c in cols)
  support.append({"model":name,"categorical_fields":";".join(cols),"total_levels":levels,"minimum_level_holes":min(data.groupby(c)["BHID"].nunique().min() for c in cols),"role":"post-analysis hierarchy ablation"})
 return new,primary_metrics(combined),pd.concat(evidence,ignore_index=True),pd.DataFrame(support)

def label_perturbation(primary,coordinate_predictions,*,rates=(0.05,0.10,0.20),repeats=10,seed=20260802):
 base=primary.copy().reset_index(drop=True);splits=primary_splits(base);rng=np.random.default_rng(seed);rows=[]
 coordinate=coordinate_predictions.loc[coordinate_predictions["model"].eq("coordinate_trend")].copy()
 for rate in rates:
  for repeat in range(repeats):
   data=base.copy();n=max(2,int(round(rate*len(data))));idx=np.sort(rng.choice(len(data),size=n,replace=False));perm=rng.permutation(idx)
   for col in ("canonical_lithology","grsc_subtype"):
    data.loc[idx,col]=base.loc[perm,col].to_numpy()
   pred=evaluate_fixed_safely(data,splits,factory(("canonical_lithology","grsc_subtype"),numeric=("mid_easting","mid_northing","mid_rl"),alpha=1.0)).assign(model="perturbed_lithology_spatial")
   joined=pd.concat([pred,coordinate],ignore_index=True)
   ev=_paired_evidence_checked(joined,conditioned="perturbed_lithology_spatial",comparator="coordinate_trend",replicates=1000,seed=seed+repeat+int(rate*1000))
   for _,r in ev.iterrows(): rows.append({"perturbation_rate":rate,"repeat":repeat,"scheme":r["scheme"],"mae_delta":r["estimate"],"paired_rows":r["paired_rows"],"paired_holes":r["paired_holes"],"stress_test_role":"grade-blind frequency-preserving label permutation; not an empirical logging-error model"})
 detail=pd.DataFrame(rows)
 summary=detail.groupby(["perturbation_rate","scheme"],sort=True).agg(repeats=("repeat","nunique"),median_delta=("mae_delta","median"),minimum_delta=("mae_delta","min"),maximum_delta=("mae_delta","max"),fraction_favouring_geology=("mae_delta",lambda x:float((x<0).mean()))).reset_index()
 return detail,summary

def nearest_training_hole_registry(primary):
 data=primary.reset_index(drop=True);centres=data.groupby("BHID")[["mid_easting","mid_northing","mid_rl"]].mean();rows=[]
 for split in primary_splits(data):
  train_holes=sorted(data.iloc[split.train_index]["BHID"].astype(str).unique());test_holes=sorted(data.iloc[split.test_index]["BHID"].astype(str).unique())
  tc=centres.loc[test_holes].to_numpy(float);rc=centres.loc[train_holes].to_numpy(float);d=cdist(tc,rc)
  for i,hole in enumerate(test_holes): rows.append({"scheme":split.scheme,"fold_id":split.fold_id,"hole":hole,"nearest_training_hole_distance_m":float(d[i].min()),"training_holes":len(train_holes)})
 return pd.DataFrame(rows)

def distance_conditioned_performance(primary,predictions):
 reg=nearest_training_hole_registry(primary)
 use=predictions.loc[predictions["scheme"].isin(SCHEMES)].merge(reg,on=["scheme","fold_id","hole"],how="left",validate="many_to_one")
 bins=[0,50,100,200,400,np.inf];labels=["0-50","50-100","100-200","200-400",">400"]
 use["distance_bin_m"]=pd.cut(use["nearest_training_hole_distance_m"],bins=bins,labels=labels,right=False,include_lowest=True)
 use["absolute_error"]=(use["prediction"]-use["truth"]).abs();use["weighted_absolute_error"]=use["absolute_error"]*use["weight"]
 holes=use.loc[use["success"].astype(bool)].groupby(["model","scheme","distance_bin_m","hole"],observed=True,sort=True).agg(error_mass=("weighted_absolute_error","sum"),support=("weight","sum"),distance=("nearest_training_hole_distance_m","first")).reset_index();holes["hole_mae"]=holes["error_mass"]/holes["support"]
 summary=holes.groupby(["model","scheme","distance_bin_m"],observed=True,sort=True).agg(holes=("hole","nunique"),mae=("hole_mae","mean"),median_distance_m=("distance","median")).reset_index()
 attempted=use.groupby(["model","scheme","distance_bin_m"],observed=True,sort=True).agg(rows=("row_index","size"),failure_rate=("success",lambda x:float(1-x.astype(bool).mean()))).reset_index();summary=summary.merge(attempted,on=["model","scheme","distance_bin_m"],how="outer")
 summary["rank_within_bin"]=summary.groupby(["scheme","distance_bin_m"],observed=True)["mae"].rank(method="min")
 return reg,summary

def extended_contrast(primary,predictions):
 return geological_contrast_preservation(primary,predictions,observed_contrast_thresholds=(0.0,0.25,0.5,1.0),replicates=20_000,seed=20260802,models=("global_mean","coordinate_trend","idw","lithology_only","lithology_spatial","geology_only","partial_pooling"))
