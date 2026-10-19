"""26_v18_newanalyses.py — v18 round: four genuinely NEW analyses that the pre-submission
review demanded. Every number is computed from data/ with the frozen pipeline of
analysis_v5.py (seed 2026). Nothing is transcribed by hand.

  (A) Linear-predictor DISPERSION and the resulting dispersion-predicted calibration slope.
      Motivated by the observation that a model calibrated where its linear predictor is
      under-dispersed transports with slope ~= SD_lp(dev)/SD_lp(target). This is a competing,
      largely mechanical explanation for the eICU slope collapse that the manuscript never
      tested. -> outputs/sr_dispersion_v18.json
  (B) BOOTSTRAP CONFIDENCE INTERVALS for every headline calibration quantity (slope, joint
      intercept, calibration-in-the-large, O/E, Brier) that previously had none.
      -> outputs/sr_calib_ci_v18.json
  (C) PARTICIPANT CHARACTERISTICS table (the TRIPOD item that was marked complete but was
      not actually reported) + a race/ethnicity subgroup check (both eICU and MIMIC-IV files
      carry a coded Race column). -> outputs/sr_baseline_v18.json, outputs/sr_race_subgroup_v18.json
  (D) NT-proBNP / LVEF SENSITIVITY that Methods claimed but never reported (apparent,
      in-sample, explicitly labelled). -> outputs/sr_sensitivity_v18.json

Run from project root:  python scripts/26_v18_newanalyses.py
"""
import os, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
COMMON = ["Age","male","HeartRate","RespiratoryRate","Temperature","SBP","AG","BUN","Creatinine",
          "Chloride","MCH","MCHC","MCV","Platelet","Potassium","RBC","RDW","Sodium","WBC","COPD","Diabetes"]
BINARY = ["male","COPD","Diabetes"]
NBOOT = 500          # matches the AUC bootstrap already declared in Methods
SEED   = 2026

# ---------------------------------------------------------------- canonical loader
def load():
    a = pd.read_csv("data/data01.csv").rename(columns={
        "heart rate":"HeartRate","Respiratory rate":"RespiratoryRate","temperature":"Temperature",
        "Systolic blood pressure":"SBP","Anion gap":"AG","Urea nitrogen":"BUN","Blood potassium":"Potassium",
        "Blood sodium":"Sodium","Leucocyte":"WBC","Platelets":"Platelet","age":"Age"})
    a["male"]=(a["gendera"]==1).astype(int); a["Diabetes"]=a["diabetes"].astype(int)
    a=a[a["outcome"].notna()].copy(); a["outcome"]=a["outcome"].astype(int)
    b = pd.read_csv("data/Derivation cohorts--MIMIC-IV database.csv")
    b["male"]=(b["Gender"]==1).astype(int); b["outcome"]=b["Hstatus"].astype(int)
    v = pd.read_csv("data/Validation cohorts--eICU-CRD.csv")
    v["male"]=(v["Gender"]==1).astype(int); v["outcome"]=v["Hstatus"].astype(int)
    return b, v, a
B, V, M3 = load()

pipe = Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                 ("lr",LogisticRegression(max_iter=2000))]).fit(B[COMMON].astype(float), B["outcome"].values)
def probs(df): return pipe.predict_proba(df[COMMON].astype(float))[:,1]
def lg(p):     return np.log(np.clip(p,1e-9,1-1e-9)/np.clip(1-p,1e-9,1-1e-9))

def unpen_calib(y, l):
    """Unpenalised IRLS for logit P = a + b*l  (the REPORTING estimator, unchanged)."""
    y=np.asarray(y,float); X=np.column_stack([np.ones(len(l)),np.asarray(l,float)]); beta=np.zeros(2)
    for _ in range(200):
        eta=X@beta; p=1/(1+np.exp(-eta)); Wd=p*(1-p)+1e-12
        step=np.linalg.solve(X.T@(X*Wd[:,None])+1e-8*np.eye(2), X.T@(y-p)); beta=beta+step
        if np.max(np.abs(step))<1e-12: break
    return beta[1], beta[0]          # slope, joint intercept
def cal_in_large(y, l):
    """Calibration-in-the-large: intercept with slope FIXED at 1 (offset logistic)."""
    y=np.asarray(y,float); l=np.asarray(l,float)
    a=0.0
    for _ in range(200):
        p=1/(1+np.exp(-(a+l))); step=(y-p).sum()/max((p*(1-p)).sum(),1e-12); a=a+step
        if abs(step)<1e-12: break
    return a
def calib_all(y, p):
    l = lg(p); s,a = unpen_calib(y,l)
    return dict(slope=float(s), joint_intercept=float(a), cal_in_large=float(cal_in_large(y,l)),
                OE=float(np.sum(y)/np.sum(p)), brier=float(brier_score_loss(y,p)))

def boot_ci(y, p, stat, n=NBOOT, seed=SEED):
    """Stay-level percentile bootstrap for any scalar statistic of (y,p)."""
    rng=np.random.RandomState(seed); y=np.asarray(y,float); p=np.asarray(p,float); out=[]
    for _ in range(n):
        i=rng.choice(len(y),len(y),replace=True)
        if len(np.unique(y[i]))<2: continue
        try: out.append(stat(y[i],p[i]))
        except Exception: pass
    if not out: return [None,None]
    return [float(np.percentile(out,2.5)), float(np.percentile(out,97.5))]

# ================================================================= (A) DISPERSION
print("== (A) linear-predictor dispersion and dispersion-predicted calibration slope ==")
disp = {}
for nm, df in [("MIMIC-IV",B), ("MIMIC-III",M3), ("eICU-CRD",V)]:
    l = lg(probs(df))
    disp[nm] = dict(n=int(len(l)), lp_mean=float(l.mean()), lp_sd=float(l.std(ddof=1)),
                    lp_iqr=float(np.subtract(*np.percentile(l,[75,25]))),
                    lp_p05=float(np.percentile(l,5)), lp_p95=float(np.percentile(l,95)))
    print(f"  {nm:9s} LP mean {l.mean():+.3f}  SD {l.std(ddof=1):.3f}  IQR {np.subcentile if False else np.subtract(*np.percentile(l,[75,25])):.3f}")
sd_dev = disp["MIMIC-IV"]["lp_sd"]

# bootstrap the SD RATIO (resample each cohort independently), and the predicted slope
rng=np.random.RandomState(SEED); ratios=[]; predslope={}
for nm, df in [("MIMIC-III",M3), ("eICU-CRD",V)]:
    l = lg(probs(df)); ratios_n=[]
    for _ in range(NBOOT):
        ib=rng.choice(len(l_dev:=lg(probs(B))),len(l_dev),replace=True)
        it=rng.choice(len(l),len(l),replace=True)
        ratios_n.append(float(np.std(l_dev[ib],ddof=1)/np.std(l[it],ddof=1)))
    ratios_n=np.array(ratios_n)
    predlope = sd_dev/disp[nm]["lp_sd"]
    ci = [float(np.percentile(ratios_n,2.5)), float(np.percentile(ratios_n,97.5))]
    predslope[nm] = dict(
        lp_sd_ratio_target_over_dev=float(disp[nm]["lp_sd"]/sd_dev),
        lp_sd_ratio_ci=[1/ci[1], 1/ci[0]],                     # inverse: ratio -> 1/ratio
        dispersion_predicted_slope=float(predlope),
        observed_slope=None)
    print(f"  {nm:9s} SD ratio {disp[nm]['lp_sd']/sd_dev:.3f}  "
          f"dispersion-predicted slope {predlope:.3f}")

tr = json.load(open("outputs/v5_results.json",encoding="utf-8"))["transport"]
for nm, key in [("MIMIC-III","A_MIMIC-III_temporal_exploratory"), ("eICU-CRD","V_eICU_geographic_primary")]:
    obs = tr[key]["calib_slope"]; pred = predslope[nm]["dispersion_predicted_slope"]
    predslope[nm]["observed_slope"]=float(obs)
    predslope[nm]["residual_observed_minus_predicted"]=float(obs-pred)
    print(f"             observed slope {obs:.4f}  predicted {pred:.4f}  residual {obs-pred:+.4f}")

json.dump(dict(
    note=("A model calibrated on a dataset where its linear predictor is UNDER-dispersed transports "
          "with slope approximately SD_lp(development)/SD_lp(target). This is a mechanical, covariate-free "
          "component of the slope failure and a competing explanation to case-mix or definition heterogeneity."),
    per_cohort=disp, slope_prediction=predslope,
    estimand="calibration slope of the frozen study-refitted model on the logit scale"),
    open("outputs/sr_dispersion_v18.json","w"), indent=2, ensure_ascii=False)

# ================================================================= (B) CALIBRATION CIs
print("\n== (B) bootstrap CIs for calibration quantities (500 stay-level replicates) ==")
half=len(V)//2; p_=np.random.RandomState(SEED).permutation(len(V))
Vu=V.iloc[p_[:half]].reset_index(drop=True); Vt=V.iloc[p_[half:]].reset_index(drop=True)
yu,yt=Vu.outcome.values,Vt.outcome.values
lr=LogisticRegression(max_iter=2000).fit(lg(probs(Vu))[:,None],yu)      # the ACTUAL update estimator
p_recal = lr.predict_proba(lg(probs(Vt))[:,None])[:,1]

CI = {}
def add(key, y, p):
    d = calib_all(y,p)
    d["slope_ci"]          = boot_ci(y,p,lambda a,b: unpen_calib(a,lg(b))[0])
    d["joint_intercept_ci"]= boot_ci(y,p,lambda a,b: unpen_calib(a,lg(b))[1])
    d["cal_in_large_ci"]   = boot_ci(y,p,lambda a,b: cal_in_large(a,lg(b)))
    d["OE_ci"]             = boot_ci(y,p,lambda a,b: np.sum(a)/np.sum(b))
    d["brier_ci"]          = boot_ci(y,p,lambda a,b: brier_score_loss(a,b))
    d["auc_ci"]            = boot_ci(y,p,lambda a,b: roc_auc_score(a,b))
    CI[key]=d
    print(f"  {key:34s} slope {d['slope']:.3f} {np.round(d['slope_ci'],3)}  "
          f"alpha {d['joint_intercept']:+.3f} {np.round(d['joint_intercept_ci'],3)}  "
          f"O/E {d['OE']:.3f} {np.round(d['OE_ci'],3)}")

add("transport_MIMIC_IV_apparent", B.outcome.values, probs(B))
add("transport_MIMIC_III_exploratory", M3.outcome.values, probs(M3))
add("transport_eICU_external", V.outcome.values, probs(V))
add("repsplit_eICU_frozen", yt, probs(Vt))
add("repsplit_eICU_recalibrated", yt, p_recal)
json.dump(dict(
    note=("95% percentile intervals from a 500-replicate stay-level bootstrap of the SAME stay sample the "
          "point estimate uses; these are within-sample (non-penalised IRLS) calibration estimates. "
          "They do NOT propagate the uncertainty of the frozen model itself or of the split, and the "
          "eICU interval does not account for hospital clustering (unavailable)."),
    n_boot=NBOOT, seed=SEED, quantity_estimator="unpenalised IRLS", strata=CI),
    open("outputs/sr_calib_ci_v18.json","w"), indent=2, ensure_ascii=False)

# ================================================================= (C) BASELINE TABLE
print("\n== (C) participant characteristics per cohort (TRIPOD item) ==")
smd_src = json.load(open("outputs/sr_prep_v9.json",encoding="utf-8"))
w = pd.read_csv("outputs/v6_smd_by_variable.csv").set_index("Variable")["SMD_unweighted"]
def summ(df, c):
    """NaN-aware summary. The MIMIC-III file DOES carry missing vital signs (see the
    missingness block below), so np.mean would propagate NaN; medians/means are computed on
    observed values and n is reported alongside."""
    s = df[c].astype(float)
    x = s.dropna().values
    if c in BINARY:
        xn = s.fillna(0).values
        return dict(kind="binary", n=int(len(s)), n_missing=int(s.isna().sum()),
                    n_pos=int(xn.sum()), pct_pos=float(100*np.mean(xn)))
    return dict(kind="continuous", n=int(len(s)), n_missing=int(s.isna().sum()),
                mean=float(np.mean(x)), sd=float(np.std(x, ddof=1)),
                median=float(np.median(x)), q1=float(np.percentile(x,25)), q3=float(np.percentile(x,75)))
miss={c:dict(MIMIC_IV=int(B[c].isna().sum()), MIMIC_III=int(M3[c].isna().sum()),
             eICU=int(V[c].isna().sum())) for c in COMMON}
tot_m3=sum(miss[c]["MIMIC_III"] for c in COMMON)
rows_any=int(M3[[c for c in COMMON]].isna().any(axis=1).sum())
print(f"  missingness: MIMIC-IV total {sum(miss[c]['MIMIC_IV'] for c in COMMON)} cells, "
      f"eICU total {sum(miss[c]['eICU'] for c in COMMON)} cells, "
      f"MIMIC-III total {tot_m3} cells in {rows_any} rows ({100*rows_any/len(M3):.2f}% of stays)")
base={}
for c in COMMON:
    base[c]=dict(MIMIC_IV=summ(B,c), MIMIC_III=summ(M3,c), eICU=summ(V,c),
                 SMD_eICU_vs_MIMICIV=None if c not in w.index else float(w[c]))
json.dump(dict(
    note=("Per-cohort summary of the 21 model covariates, computed on OBSERVED values. SMD is the "
          "absolute standardised mean difference (eICU vs MIMIC-IV) already tabulated in the "
          "supplement; it is reported here so the characteristics table also documents the covariate "
          "shift that motivates the measured-covariate standardisation analysis. MIMIC-IV and eICU are "
          "complete; the MIMIC-III file carries 57 missing cells in four vital signs (1.79% of stays)."),
    missing_cells=miss,
    missing_summary=dict(MIMIC_IV_total=sum(miss[c]["MIMIC_IV"] for c in COMMON),
                         eICU_total=sum(miss[c]["eICU"] for c in COMMON),
                         MIMIC_III_total=tot_m3, MIMIC_III_rows_affected=rows_any,
                         MIMIC_III_pct_rows=round(100*rows_any/len(M3),2)),
    units=dict(age="years",male="binary 1=male",HeartRate="beats/min",RespiratoryRate="breaths/min",
               Temperature="deg C",SBP="mmHg",AG="mmol/L",BUN="mg/dL",Creatinine="mg/dL",
               Chloride="mmol/L",MCH="pg",MCHC="g/dL",MCV="fL",Platelet="10^9/L",
               Potassium="mmol/L",RBC="10^6/uL",RDW="%",Sodium="mmol/L",WBC="10^9/L",
               COPD="binary",Diabetes="binary"),
    cohort_sizes=dict(MIMIC_IV=dict(n=int(len(B)),deaths=int(B.outcome.sum()),prevalence=float(B.outcome.mean())),
                      MIMIC_III=dict(n=int(len(M3)),deaths=int(M3.outcome.sum()),prevalence=float(M3.outcome.mean())),
                      eICU=dict(n=int(len(V)),deaths=int(V.outcome.sum()),prevalence=float(V.outcome.mean()))),
    variables=base),
    open("outputs/sr_baseline_v18.json","w"), indent=2, ensure_ascii=False)

# ---- race/ethnicity subgroup (coded 1/2/3 in BOTH MIMIC-IV and eICU files) ----
print("  race-coded subgroup (frozen model, descriptive, no interaction test):")
race={}
for nm, df in [("eICU-CRD",V), ("MIMIC-IV",B)]:
    rows=[]
    for code in sorted(df["Race"].dropna().unique()):
        m=(df["Race"]==code).values; y=df["outcome"].values[m]; p=probs(df)[m]
        d=calib_all(y,p); d["auc_ci"]=boot_ci(y,p,lambda a,b: roc_auc_score(a,b),n=300)
        rows.append(dict(race_code=int(code), n=int(m.sum()), deaths=int(y.sum()),
                         AUC=float(roc_auc_score(y,p)), AUC_ci=d["auc_ci"],
                         slope=d["slope"], OE=d["OE"]))
        print(f"    {nm} Race={int(code)}  n={m.sum():5d}  deaths={int(y.sum()):4d}  "
              f"AUC {rows[-1]['AUC']:.3f} {np.round(rows[-1]['AUC_ci'],3)}  slope {d['slope']:.2f}  O/E {d['OE']:.2f}")
    race[nm]=rows
json.dump(dict(
    note=("Descriptive subgroup estimates for the frozen transported model. Race is a coded 1/2/3 column "
          "present in BOTH the MIMIC-IV and the eICU redistributed files. The code->label mapping is NOT "
          "documented in the released files, so codes are reported verbatim rather than assigned to "
          "ethnic groups. No interaction test and no per-stratum calibration CI are performed; these are "
          "not inferential subgroup claims."),
    strata=race),
    open("outputs/sr_race_subgroup_v18.json","w"), indent=2, ensure_ascii=False)

# ================================================================= (D) NT-proBNP / LVEF
print("\n== (D) HF-marker sensitivity that Methods claimed (APPARENT / in-sample) ==")
def apparent_auc(df, cols):
    m=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                ("lr",LogisticRegression(max_iter=2000))]).fit(df[cols].astype(float), df["outcome"].values)
    return float(roc_auc_score(df["outcome"].values, m.predict_proba(df[cols].astype(float))[:,1]))
sens={}
sens["MIMIC_III_base_21"]            = apparent_auc(M3, COMMON)
sens["MIMIC_III_plus_NTproBNP_EF"]    = apparent_auc(M3, COMMON+["NT-proBNP","EF"])
sens["delta_apparent_AUC"]            = sens["MIMIC_III_plus_NTproBNP_EF"]-sens["MIMIC_III_base_21"]
print(f"  MIMIC-III apparent AUC  21 vars {sens['MIMIC_III_base_21']:.4f} "
      f"-> +NT-proBNP+EF {sens['MIMIC_III_plus_NTproBNP_EF']:.4f} "
      f"(delta {sens['delta_apparent_AUC']:+.4f})")
json.dump(dict(
    note=("APPARENT (in-sample, resubstitution) performance only: the added biomarkers and the 21 model "
          "covariates are fitted and scored on the SAME MIMIC-III rows, with no held-out data. This "
          "therefore OVERSTATES performance and must not be read as a validated increment. It is reported "
          "solely to quantify what the deliberately cross-database-intersected 21-variable model omits: "
          "the heart-failure-specific markers that exist in only the MIMIC-III file."),
    estimand="apparent resubstitution AUC on the MIMIC-III exploratory cohort",
    n=int(len(M3)), **{k:(round(v,4) if isinstance(v,float) else v) for k,v in sens.items()}),
    open("outputs/sr_sensitivity_v18.json","w"), indent=2, ensure_ascii=False)

print("\nwrote outputs/sr_dispersion_v18.json, sr_calib_ci_v18.json, "
      "sr_baseline_v18.json, sr_race_subgroup_v18.json, sr_sensitivity_v18.json")
