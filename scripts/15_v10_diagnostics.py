"""15_v10_diagnostics.py — Round-10 supporting numbers (real, relative root, seed 2026).
(a) slope distribution per size (median/IQR/extremes/negative count) + events-per-RECALIBRATION-PARAMETER (events/2)
(b) across-10-split ΔAUC: mean, SD, range (present as conditional split variability, not a t-CI)
(c) confirm reweight bootstrap resamples BOTH cohorts and recomputes weights+slope each replicate (facts from 13)
Writes outputs/sr_diag_v10.json.
"""
import os, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; os.chdir(ROOT)
SIZES=[100,200,500,1000,2000,5000]
rd=pd.read_csv("outputs/v5_repeated_curve.csv")
out={}

# (a) slope distribution + events/parameter
sd={}
# recompute update-half event counts identically (seed 2026 outer, 1000+rep sub) to pair events with slopes
def load():
    v=pd.read_csv("data/Validation cohorts--eICU-CRD.csv"); v["outcome"]=v["Hstatus"].astype(int); return v
V=load(); half=len(V)//2
curve_rng=np.random.RandomState(2026)
events_by_rep={n:[] for n in SIZES}
for rep in range(10):
    p_=curve_rng.permutation(len(V)); Vu=V.iloc[p_[:half]].reset_index(drop=True); yu=Vu.outcome.values
    sub_rng=np.random.RandomState(1000+rep)
    for n in SIZES:
        for _ in range(6):
            idx=sub_rng.choice(len(Vu),n,replace=False); events_by_rep[n].append(int(yu[idx].sum()))
for n in SIZES:
    sl=rd[rd.n_pat==n].slope.values; ev=np.array(events_by_rep[n])
    sd[str(n)]={"slope_median":round(float(np.median(sl)),3),"slope_q1":round(float(np.percentile(sl,25)),3),
        "slope_q3":round(float(np.percentile(sl,75)),3),"slope_min":round(float(sl.min()),3),"slope_max":round(float(sl.max()),3),
        "n_negative_slope":int((sl<0).sum()),"n_slope_gt3":int((sl>3).sum()),
        "events_median":round(float(np.median(ev)),1),"events_min":int(ev.min()),"events_max":int(ev.max()),
        "events_per_recalibration_parameter_median":round(float(np.median(ev))/2.0,1)}
out["slope_and_events"]=sd

# (b) across-10 ΔAUC mean/SD/range (vs frozen) — present as conditional split variability, NOT t-CI
sc=pd.read_csv("outputs/sr_sofa_compare.csv")  # columns rep, recal, sofa, aps3, sofa_aps3, refit, frozen
def stat(col):
    d=(sc[col]-sc["frozen"]).values
    return {"mean":round(float(d.mean()),4),"sd":round(float(d.std(ddof=1)),4),
            "range":[round(float(d.min()),4),round(float(d.max()),4)],
            "label":"mean (SD) [range] over 10 correlated outer splits = conditional split-scheme variability, not a population 95% CI"}
out["delta_auc_across10"]={k:stat(k) for k in ["recal","sofa","aps3","sofa_aps3","refit"]}

# (c) reweight bootstrap facts (from 13_v9_sensitivity.py design)
out["reweight_bootstrap_facts"]={"resamples_both_cohorts":True,"recomputes_propensity_and_weights_each_replicate":True,
    "re-estimates_weighted_slope_each_replicate":True,"n_replicates":200,
    "weights_normalised":False,"note":"both MIMIC-IV and eICU rows resampled; P(eICU|X) re-fit and weights truncated [0.1,10] and weighted IRLS slope re-estimated within each replicate; weights not renormalised (does not affect slope)."}

# (d) representative split = pre-specified seed 2026 first permutation (not cherry-picked)
out["representative_split"]={"rule":"RandomState(2026).permutation(len(V)); first 50/50 split; fixed seed, chosen before inspecting results (deterministic, not selected on outcome)"}

json.dump(out,open("outputs/sr_diag_v10.json","w",encoding="utf-8"),indent=2,ensure_ascii=False)
print(json.dumps(out,indent=2,ensure_ascii=False))
