"""09_v7_inference.py — fixes the two statistical-inference problems raised in the round-6 review.
Major 2: learning-curve uncertainty must respect clustering (60 updates share 10 outer splits).
         -> report per-split success rate AND a split-level (cluster) bootstrap CI.
Major 5: the paired bootstrap used a single fixed split; add ACROSS-SPLIT variability of
         (retrain - recalibrate) for AUC and Brier from v5_strategy.csv (10 splits),
         so the two estimand sources (test-sampling vs update-sampling) are separated.
Writes outputs/v7_inference.json. Deterministic.
"""
import os, json
import numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
rng=np.random.RandomState(2026)
R={}
rd=pd.read_csv("outputs/v5_repeated_curve.csv"); SIZES=[100,200,500,1000,2000,5000]

# ---- per-split success rate at each size (out of 6 sub-updates) ----
per_split={}
for n in SIZES:
    rates=[int(g[g.n_pat==n].stable.sum()) for _,g in rd.groupby("rep")]  # 6 each
    per_split[n]={"successes_of_6_per_split":rates}
# ---- cluster bootstrap over 10 splits for the pooled success probability ----
split_ids=sorted(rd.rep.unique())
cluster={}
for n in SIZES:
    by_split={s:rd[(rd.rep==s)&(rd.n_pat==n)].stable.values for s in split_ids}
    pooled=float(np.mean([v.mean() for v in by_split.values()]))
    boots=[]
    for _ in range(4000):
        pick=rng.choice(len(split_ids),len(split_ids),replace=True)
        boots.append(np.mean([by_split[split_ids[k]].mean() for k in pick]))
    lo,hi=np.percentile(boots,[2.5,97.5])
    cluster[n]={"pooled_success_pct":round(100*pooled,1),
                "cluster_bootstrap_ci":[round(100*lo,1),round(100*hi,1)],
                "naive_wilson_like_se":round(100*np.sqrt(pooled*(1-pooled)/60),2)}
R["learning_curve_cluster"]=cluster

# ---- Major 5: across-split variability of (retrain - recalibrate) from v5_strategy ----
sd=pd.read_csv("outputs/v5_strategy.csv")
d_auc=[]; d_brier=[]
for rep,g in sd.groupby("rep"):
    r=g[g.strategy=="recalibrate_2par"].iloc[0]; f=g[g.strategy=="full_refit"].iloc[0]
    d_auc.append(f.AUC-r.AUC); d_brier.append(r.Brier-f.Brier)   # positive = retrain better
d_auc=np.array(d_auc); d_brier=np.array(d_brier)
def t_ci(x):
    m=x.mean(); se=x.std(ddof=1)/np.sqrt(len(x)); return [round(m-1.96*se,4),round(m+1.96*se,4)]
R["across_split_delta"]={
  "n_splits":int(len(d_auc)),
  "dAUC_retrain_minus_recalibrate":{"mean":round(float(d_auc.mean()),4),"ci95":t_ci(d_auc),"range":[round(float(d_auc.min()),4),round(float(d_auc.max()),4)]},
  "dBrier_retrain_minus_recalibrate":{"mean":round(float(d_brier.mean()),4),"ci95":t_ci(d_brier),"range":[round(float(d_brier.min()),4),round(float(d_brier.max()),4)]},
  "note":"These CIs reflect UPDATE-SAMPLE (split-to-split) variability. The v6 paired patient bootstrap (single fixed split) reflects TEST-SET sampling variability only; the two are different estimands and are now reported separately."}
json.dump(R,open("outputs/v7_inference.json","w"),indent=2)
print(json.dumps(R,indent=2))
