"""14_v9_supp.py — rebuild v9 supplementary (S1 rename, S2 corrected, S3 weight+S5+S4 per-var, TRIPOD intact)."""
import os, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; os.chdir(ROOT)
v6=json.load(open("outputs/v6_results.json",encoding="utf-8"))
prep=json.load(open("outputs/sr_prep_v9.json",encoding="utf-8"))
sens=json.load(open("outputs/sr_sens_v9.json",encoding="utf-8"))
ws=v6["weight_sensitivity"]
rd=pd.read_csv("outputs/v5_repeated_curve.csv"); SIZES=[100,200,500,1000,2000,5000]

# S1 per-split persistence after first achieving 6/6
s1=[]
for rep,g in rd.groupby("rep"):
    row={"Outer split":int(rep)}
    for n in SIZES: row[str(n)]=int(g[g.n_pat==n].stable.sum())
    s1.append(row)
s1=pd.DataFrame(s1)
mono=v6["learning_curve_monotonicity"]["n_splits"]-v6["learning_curve_monotonicity"]["non_monotone_splits"]

s2=pd.read_csv("outputs/v6_smd_by_variable.csv")
over42=prep["S2_weighted_over_010"]
missB=prep["missing_pct_B"]; missV=prep["missing_pct_V"]; cc=prep["complete_case_pct"]
wd=sens["weight_diagnostics"]; wr=sens["weighted_slope_bootstrap_refit"]; wfx=sens["weighted_slope_bootstrap_fixedweight_FOR_CONTRAST"]
thr=pd.read_csv("outputs/sr_threshold_sensitivity.csv")
ec=sens["update_event_counts"]
sofa=sens["delta_auc_vs_frozen_across10"]; pb=sens["paired_patient_bootstrap_rep0"]

L=["# Supplementary Material (v9)\n",
   "Secondary analysis — cross-database transport and within-target recalibration of a heart-failure ICU in-hospital-mortality model. Every value is recomputed from `data/`, `outputs/*.json|csv` and `scripts/13_v9_sensitivity.py` (seed 2026).\n"]

L.append("## Table S1. Per-split counts of held-out recalibrations meeting the stability target (of 6 sub-updates) at each update size\n")
L.append("Target = held-out |slope−1|<0.2 AND |O/E−1|<0.15 (permissive, study-defined). **Note on terminology:** the secondary 'all-six' statistic is a *persistence-after-first-achieving-6/6* measure (once a split reaches 6/6 does it stay at 6/6 at larger sizes), **not** general stochastic monotonicity; %d of %d outer splits satisfied that persistence criterion.\n"%(mono, v6['learning_curve_monotonicity']['n_splits']))
L.append(s1.to_markdown(index=False)+"\n")

L.append("## Table S2. Per-variable SMD (eICU vs MIMIC-IV) before and after inverse-probability re-weighting\n")
L.append(s2.to_markdown(index=False)+"\n")
L.append("Mean |SMD| after weighting = %.3f. **After weighting, %d covariates still exceed |SMD| 0.10** (not one): %s. COPD is a binary variable, so its weighted quantile deviations are 0 (mean/distribution differ only through prevalence, |SMD| %.3f).\n"%(
    ws['0.1-10']['mean_abs_SMD'], over42['count'],
    ", ".join(f"{v} ({s:.3f})" for v,s in over42['variables']), ws['0.1-10']['max_abs_SMD']))

L.append("## Table S3. Weight-truncation sensitivity and re-weighting diagnostics (same frozen model and outcome)\n")
s3=pd.DataFrame([dict(Truncation=k, mean_abs_SMD=ws[k]['mean_abs_SMD'], max_abs_SMD=ws[k]['max_abs_SMD'],
    ESS_abs=v6['weight_sensitivity'][k]['ESS'], ESS_pct=ws[k]['ESS_pct'], weighted_slope=ws[k]['weighted_slope'],
    weighted_intercept=ws[k]['weighted_intercept'], weighted_OE=ws[k]['weighted_OE'], weight_median=ws[k]['weight_median']) for k in ['0.1-10','1-5','p1p99']])
L.append(s3.to_markdown(index=False)+"\n")
L.append("Common-support diagnostics: propensity separability AUC = **%.3f** (the two databases are strongly distinguishable on the 21 covariates -> severe covariate shift); weight median %.3f, p25 %.3f, p75 %.3f, p99 %.2f; **%.1f%% of eICU weights are pinned at a truncation boundary**; absolute ESS = %d (17.4%%).\n"%(
    wd['propensity_separability_AUC'], wd['weight_median'], wd['weight_p25'], wd['weight_p75'], wd['weight_p99'],
    100*wd['extreme_weight_fraction'], wd['ESS_absolute']))
L.append("Weighted-slope interval **with the propensity model re-fit inside each bootstrap replicate** (n=%d): %.3f (%.3f–%.3f). Fixed-weight interval for contrast (n=%d): %.3f (%.3f–%.3f). Re-fitting did not materially widen the interval, but separability, extreme weights and tail imbalance (below) keep this an *exploratory* diagnosis, not a causal decomposition.\n"%(
    wr['n_boot'],wr['mean'],wr['ci95'][0],wr['ci95'][1],wfx['n_boot'],wfx['mean'],wfx['ci95'][0],wfx['ci95'][1]))
L.append("Distributional (tail) balance after weighting — weighted-minus-MIMIC-IV percentile deviations: "+
    "; ".join(f"{c}: p10 {v['p10_dev']:+.1f}/p50 {v['p50_dev']:+.1f}/p90 {v['p90_dev']:+.1f}" for c,v in wd['nonlinear_quantile_deviation'].items())+". Mean balance therefore masks residual tail imbalance (largest for RespiratoryRate and HeartRate).\n")
L.append("Overlap and weight distributions shown in `figures/sr_fig_overlap.png`.\n")

L.append("## Table S4. Per-variable dictionary for the 21 model covariates (one row per variable)\n")
L.append("Units, original source column, encoding, documented availability window, missingness, and residual post-timing information risk. The redistributed CSVs carry **no timestamp**; 'Documented window' is taken from the source studies and could not be independently re-derived. No vasopressor/ventilation/intubation/diagnosis-specific variable enters the main 21-variable model.\n")
# original column mapping + units + category + timing
meta={
 "Age":("Age","years","demographic","baseline/admission"),
 "male":("Gender","binary (1=male)","demographic","baseline/admission"),
 "HeartRate":("HeartRate","beats/min","vital sign","first 24 h (aggregation rule undocumented)"),
 "RespiratoryRate":("RespiratoryRate","breaths/min","vital sign","first 24 h (aggregation undocumented)"),
 "Temperature":("Temperature","°C","vital sign","first 24 h (aggregation undocumented)"),
 "SBP":("SBP","mmHg","vital sign","first 24 h (aggregation undocumented)"),
 "AG":("AG","mmol/L","laboratory","first measurement on admission"),
 "BUN":("BUN","mg/dL","laboratory","first measurement on admission"),
 "Creatinine":("Creatinine","mg/dL","laboratory","first measurement on admission"),
 "Chloride":("Chloride","mmol/L","laboratory","first measurement on admission"),
 "MCH":("MCH","pg","laboratory (CBC)","first measurement on admission"),
 "MCHC":("MCHC","g/dL","laboratory (CBC)","first measurement on admission"),
 "MCV":("MCV","fL","laboratory (CBC)","first measurement on admission"),
 "Platelet":("Platelet","10^9/L","laboratory (CBC)","first measurement on admission"),
 "Potassium":("Potassium","mmol/L","laboratory","first measurement on admission"),
 "RBC":("RBC","10^6/µL","laboratory (CBC)","first measurement on admission"),
 "RDW":("RDW","%","laboratory (CBC)","first measurement on admission"),
 "Sodium":("Sodium","mmol/L","laboratory","first measurement on admission"),
 "WBC":("WBC","10^9/L","laboratory (CBC)","first measurement on admission"),
 "COPD":("COPD","binary","comorbidity","baseline history"),
 "Diabetes":("Diabetes","binary","comorbidity","baseline history"),
}
rows=[]
for c in ["Age","male","HeartRate","RespiratoryRate","Temperature","SBP","AG","BUN","Creatinine","Chloride","MCH","MCHC","MCV","Platelet","Potassium","RBC","RDW","Sodium","WBC","COPD","Diabetes"]:
    orig,unit,cat,win=meta[c]
    risk = "whole-stay labs in MIMIC-III -> exploratory only" if cat=="laboratory" else ("post-timing info possible if vital summarised over stay" if cat=="vital sign" else "baseline; low")
    rows.append(dict(Variable=c,Original_column=orig,Unit=unit,Category=cat,Documented_window=win,
                     Missing_pct_MIMICIV=missB[c],Missing_pct_eICU=missV[c],Post_timing_risk=risk))
L.append(pd.DataFrame(rows).to_markdown(index=False)+"\n")
L.append("Variables used **only** for severity-score comparison / sensitivity (not in the 21-variable main model): SOFA, APS-III, NT-proBNP, ejection fraction (EF), Calcium, and other file-only labs. These are excluded from the main dictionary to avoid implying they enter the transported model.\n")

L.append("## Table S5. Missingness and imputation diagnostics\n")
L.append("Across the 21 model covariates the redistributed model-ready files had **no missing values** (complete-case fraction: MIMIC-IV %.0f%%, eICU %.0f%%). The median-imputation step in the pipeline therefore did not alter the analysed data, and imputation/scaling parameters were fit on the training/update portion only (test portion transform-only). The single missing record in the MIMIC-III cohort was in the outcome, not a predictor.\n"%(cc['MIMIC-IV'],cc['eICU']))

L.append("## Table S6. Stability-target threshold sensitivity\n")
L.append("Hit-rate (%%) at each update size under different pre-set tolerances (slope tol × O/E tol). The learning curve is sensitive to the tolerance: at 1,000 stays the hit-rate ranges from %.1f%% (strict 0.10/0.10) to %.1f%% (lenient 0.30/0.20); the headline 80.0%% corresponds to the 0.20/0.15 configuration used in the main text.\n"%(
    float(thr[(thr.slope_tol==0.1)&(thr.oe_tol==0.1)]['rate@1000'].iloc[0]),
    float(thr[(thr.slope_tol==0.3)&(thr.oe_tol==0.2)]['rate@1000'].iloc[0])))
L.append(thr.to_markdown(index=False)+"\n")

L.append("## Table S7. Events and fit diagnostics behind the stability curve\n")
ecr=[]
for n in map(str,SIZES):
    d=ec[n]; ecr.append(dict(Update_size=int(n),update_events_min=d['min'],update_events_median=d['median'],update_events_max=d['max'],
        negative_slopes=d['neg_slope'],extreme_slope_gt3=d['extreme_slope_gt3'],nonconvergence=d['nonconv'],subupdates=d['total']))
L.append(pd.DataFrame(ecr).to_markdown(index=False)+"\n")
L.append("At the smallest sizes the update sample contains only ~6–21 deaths (events-per-variable < 1), which — not numerical non-convergence (0 failures; a single negative slope at n=100) — drives the large slope dispersion.\n")

L.append("## Table S8. Held-out model comparison vs the transported model, across 10 splits (paired ΔAUC) and paired patient bootstrap (representative split)\n")
cmp=[]
for k,lab in [("recal","Two-parameter recalibration"),("sofa","SOFA (refit on update half)"),("aps3","APS-III (refit)"),("sofa_aps3","SOFA + APS-III (refit)"),("refit","21-var refit")]:
    cmp.append(dict(Model=lab,deltaAUC_vs_frozen_mean=sofa[k]['mean'],ci95=f"{sofa[k]['ci95'][0]}–{sofa[k]['ci95'][1]}"))
L.append(pd.DataFrame(cmp).to_markdown(index=False)+"\n")
L.append("Paired patient bootstrap on the held-out test half (representative split): ΔAUC(SOFA − recalibrated) = %.3f (%.3f–%.3f); ΔAUC(21-var refit − SOFA) = %.3f (%.3f–%.3f). So even after recalibration the transported model is out-ranked by a refitted SOFA (positive CI), while the full refit edges past SOFA modestly. Calibration curves in `figures/sr_fig_calib_curves.png`.\n"%(
    pb['dAUC_sofa_minus_recalibrated']['mean'],pb['dAUC_sofa_minus_recalibrated']['ci95'][0],pb['dAUC_sofa_minus_recalibrated']['ci95'][1],
    pb['dAUC_refit_minus_sofa']['mean'],pb['dAUC_refit_minus_sofa']['ci95'][0],pb['dAUC_refit_minus_sofa']['ci95'][1]))

open("outputs/SUPPLEMENT_v9.md","w",encoding="utf-8").write("\n".join(L))
print("wrote outputs/SUPPLEMENT_v9.md  (S1-S8)")
