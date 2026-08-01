"""Bounded independent-hole positive control for information-layer selection.

This post-analysis benchmark uses actual hole centroids and buffered block folds.
It does not alter or claim to be the frozen composite-scale empirical gate.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.distance import cdist,pdist,squareform

def hole_design(primary):
 rows=[]
 for hole,g in primary.groupby("BHID",sort=True):
  mode=g["canonical_lithology"].astype(str).mode();lith=mode.iloc[0] if len(mode) else "unknown"
  rows.append({"BHID":hole,"easting":g["mid_easting"].mean(),"northing":g["mid_northing"].mean(),"rl":g["mid_rl"].mean(),"lithology":lith,"northing_block":g["northing_block"].iloc[0]})
 return pd.DataFrame(rows)
def _binary(labels): return (np.asarray(labels).astype(str)=="graphitic_schist").astype(float)
def _trend_fit_predict(y_train,label_train,label_test):
 x=np.c_[np.ones(len(y_train)),_binary(label_train)];beta=np.linalg.solve(x.T@x+np.diag([1e-8,1e-3]),x.T@y_train)
 return x@beta,np.c_[np.ones(len(label_test)),_binary(label_test)]@beta

def _score(coords,residual,short_distance):
 d=pdist(coords);r=np.asarray(residual,float);a,b=np.triu_indices(len(r),1);mask=d<=short_distance
 if mask.sum()<5 or np.std(r)<=1e-12:return 0.0
 return float(np.corrcoef(r[a[mask]],r[b[mask]])[0,1]) if np.std(r[a[mask]])>0 and np.std(r[b[mask]])>0 else 0.0

def _gp_predict(train_coords,test_coords,residual,ranges,noise_fraction_grid=(0.2,0.5,0.8)):
 d=squareform(pdist(train_coords));dt=cdist(test_coords,train_coords);var=max(float(np.var(residual)),1e-6);best=None
 for practical_range in ranges:
  scale=max(practical_range/3.0,1e-6)
  for noise_fraction in noise_fraction_grid:
   structured=var*(1-noise_fraction);noise=var*noise_fraction
   k=structured*np.exp(-d/scale);k.flat[::len(k)+1]+=noise+1e-8
   try:
    cf=cho_factor(k,lower=True,check_finite=False);alpha=cho_solve(cf,residual,check_finite=False)
    logdet=2*np.log(np.diag(cf[0])).sum();nll=0.5*(residual@alpha+logdet+len(residual)*np.log(2*np.pi))
   except Exception:continue
   if best is None or nll<best[0]:best=(nll,scale,structured,alpha)
 if best is None:return np.zeros(len(test_coords)),False,np.nan
 _,scale,structured,alpha=best;cross=structured*np.exp(-dt/scale)
 return cross@alpha,True,scale*3.0

def _folds(design,buffer_m):
 coords=design[["easting","northing","rl"]].to_numpy(float);out=[]
 for block in sorted(design["northing_block"].unique()):
  test=np.flatnonzero(design["northing_block"].eq(block).to_numpy());candidate=np.flatnonzero(~design["northing_block"].eq(block).to_numpy());dist=cdist(coords[candidate],coords[test]);train=candidate[dist.min(axis=1)>buffer_m]
  out.append((str(block),train,test))
 return out

def run_positive_control(primary,simulations=100,calibration_simulations=500,seed=20260802):
 design=hole_design(primary);coords=design[["easting","northing","rl"]].to_numpy(float);labels=design["lithology"].to_numpy(str);buffer_m=float(primary["spatial_buffer_m"].median());folds=_folds(design,buffer_m)
 nn=np.partition(squareform(pdist(coords))+np.eye(len(coords))*1e12,0,axis=1)[:,0];median_nn=float(np.median(nn));short=2*median_nn
 rng=np.random.default_rng(seed);thresholds={}
 for fold,train,test in folds:
  c=coords[train];noise=rng.normal(size=(len(train),calibration_simulations));scores=np.array([_score(c,noise[:,j],short) for j in range(calibration_simulations)])
  geo=[]
  for j in range(calibration_simulations):
   y=noise[:,j];g=_binary(labels[train]);geo.append(abs(y[g==1].mean()-y[g==0].mean())/(np.std(y)+1e-9) if len(np.unique(g))>1 else 0)
  thresholds[fold]=(float(np.quantile(scores,0.95)),float(np.quantile(geo,0.95)))
 scenarios=(
  ("null",0.0,0.0,1.0),("geology_only",2.0,0.0,1.0),("moderate_covariance",2.0,0.6,1.0),("strong_covariance",5.0,3.0,0.4),("covariance_dominant",0.2,3.0,0.4),("label_noise",5.0,3.0,0.4))
 distances=squareform(pdist(coords));rows=[]
 for scenario,geo_effect,structured_var,nugget_var in scenarios:
  scale=4*median_nn/3.0;cov=structured_var*np.exp(-distances/max(scale,1e-6))+np.eye(len(coords))*nugget_var
  fields=rng.multivariate_normal(np.zeros(len(coords)),cov,size=simulations).T
  true_labels=labels.copy()
  for sim in range(simulations):
   used_labels=true_labels.copy()
   if scenario=="label_noise":
    idx=rng.choice(len(used_labels),size=max(2,int(0.2*len(used_labels))),replace=False);used_labels[idx]=used_labels[rng.permutation(idx)]
   y=4.0+geo_effect*_binary(true_labels)+fields[:,sim]
   for fold,train,test in folds:
    global_pred=np.repeat(y[train].mean(),len(test));train_trend,geo_pred=_trend_fit_predict(y[train],used_labels[train],used_labels[test]);resid_geo=y[train]-train_trend
    global_resid=y[train]-y[train].mean();ranges=(2*median_nn,4*median_nn,8*median_nn)
    spatial_inc,sp_ok,sp_range=_gp_predict(coords[train],coords[test],global_resid,ranges);rk_inc,rk_ok,rk_range=_gp_predict(coords[train],coords[test],resid_geo,ranges)
    spatial_pred=global_pred+spatial_inc;rk_pred=geo_pred+rk_inc
    g=_binary(used_labels[train]);geo_score=abs(y[train][g==1].mean()-y[train][g==0].mean())/(np.std(y[train])+1e-9) if len(np.unique(g))>1 else 0
    cov_score=_score(coords[train],resid_geo,short);cov_pass=bool(cov_score>thresholds[fold][0] and rk_ok);geo_pass=bool(geo_score>thresholds[fold][1])
    selected="regression_kriging" if geo_pass and cov_pass else "geology_only" if geo_pass else "spatial_covariance" if cov_pass and sp_ok else "global_mean"
    preds={"global_mean":global_pred,"geology_only":geo_pred,"spatial_covariance":spatial_pred,"regression_kriging":rk_pred};mae={k:float(np.mean(np.abs(v-y[test]))) for k,v in preds.items()};oracle=min(mae,key=mae.get)
    rows.append({"scenario":scenario,"simulation":sim,"fold":fold,"test_holes":len(test),"geology_gate_pass":geo_pass,"covariance_gate_pass":cov_pass,"selected_policy":selected,"oracle_policy":oracle,"selected_mae":mae[selected],"oracle_mae":mae[oracle],"selection_regret":mae[selected]-mae[oracle],**{f"{k}_mae":v for k,v in mae.items()},"residual_score":cov_score,"geology_score":geo_score,"fitted_rk_range_m":rk_range,"role":"bounded post-analysis independent-hole positive control; not the frozen composite-scale empirical gate"})
 detail=pd.DataFrame(rows)
 summary=detail.groupby("scenario",sort=False).agg(simulation_fold_units=("fold","size"),geology_gate_pass_rate=("geology_gate_pass","mean"),covariance_gate_pass_rate=("covariance_gate_pass","mean"),mean_selected_mae=("selected_mae","mean"),mean_oracle_mae=("oracle_mae","mean"),mean_selection_regret=("selection_regret","mean")).reset_index()
 policy=detail.groupby(["scenario","selected_policy"],sort=True).size().rename("count").reset_index();policy["selection_rate"]=policy["count"]/policy.groupby("scenario")["count"].transform("sum")
 return detail,summary,policy,pd.DataFrame([{"holes":len(design),"folds":len(folds),"simulations_per_scenario":simulations,"calibration_nulls_per_fold":calibration_simulations,"median_hole_nnd_m":median_nn,"short_lag_distance_m":short,"buffer_m":buffer_m,"limitation":"independent-hole benchmark; actual composite support and frozen empirical gate are not replaced"}])
