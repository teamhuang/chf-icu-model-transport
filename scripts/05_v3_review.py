"""DEPRECATED — do not use for reported numbers. Contains two defects found and fixed in the active pipeline: (1) inverted inverse-probability weights; (2) double-counted minimum-stable-size tally. Retained for provenance only; superseded by scripts/analysis_v5.py per requirements.txt.

v3 revision analysis in response to second-round review:

1. O/E direction diagnosis (reviewer says MIMIC-III O/E=1.33 means UNDER,
   eICU O/E=0.64 means OVER-prediction; prior draft had it reversed).
2. Patient-level independent update/test split (replaces hospital-level,
   = HOSPITALID not available in published eICU files).
3. Sample-size learning curve for recalibration (n = 50/200/500/1000/2000 deaths
   and patients); evaluated on an independent held-out patient subset.
4. Case-mix reweighting (importance weights to match target covariate
   distribution) — tests whether case-mix shift alone explains calibration drift.
5. Baseline score comparison wording correction: SOFA on eICU is 0.712 vs
   frozen 0.689 — SOFA is HIGHER there; reviewer is right.

Outputs:
- outputs/v3_oe_diagnosis.txt
- outputs/v3_learning_curve.csv
- outputs/v3_patient_split_validation.csv
- outputs/v3_reweighting.csv
- figures/fig10_learning_curve.png
- figures/fig11_reweighting.png
"""
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")
import os
os.makedirs("figures", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
RNG = np.random.RandomState(42)

COMMON = ["Age", "male", "HeartRate", "RespiratoryRate", "Temperature", "SBP", "AG", "BUN",
          "Creatinine", "Chloride", "MCH", "MCHC", "MCV", "Platelet", "Potassium", "RBC",
          "RDW", "Sodium", "WBC", "COPD", "Diabetes"]

def load_data():
    a = pd.read_csv("data/data01.csv").rename(columns={
        "heart rate": "HeartRate", "Respiratory rate": "RespiratoryRate",
        "temperature": "Temperature", "Systolic blood pressure": "SBP",
        "Anion gap": "AG", "Urea nitrogen": "BUN", "Blood potassium": "Potassium",
        "Blood sodium": "Sodium", "Leucocyte": "WBC", "Platelets": "Platelet", "age": "Age"})
    a["male"] = (a["gendera"] == 1).astype(int)
    a["Diabetes"] = a["diabetes"].astype(int)
    a = a[a["outcome"].notna()].copy(); a["outcome"] = a["outcome"].astype(int)
    b = pd.read_csv("data/Derivation cohorts--MIMIC-IV database.csv")
    b["male"] = (b["Gender"] == 1).astype(int)
    b["outcome"] = b["Hstatus"].astype(int)
    v = pd.read_csv("data/Validation cohorts--eICU-CRD.csv")
    v["male"] = (v["Gender"] == 1).astype(int)
    v["outcome"] = v["Hstatus"].astype(int)
    return a, b, v

A, B, V = load_data()

pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                 ("sc", StandardScaler()),
                 ("lr", LogisticRegression(max_iter=2000))]) \
    .fit(B[COMMON].astype(float), B["outcome"].values)

def pr_pipe(df):
    return pipe.predict_proba(df[COMMON].astype(float))[:, 1]

def calibrate(y, p):
    eps = 1e-9
    lp = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    lr = LogisticRegression(max_iter=500).fit(lp[:, None], y)
    slope = lr.coef_[0][0]
    intercept = lr.intercept_[0]
    oe = y.sum() / max(p.sum(), 1e-9)
    return {"slope": slope, "intercept": intercept, "OE": oe,
            "pred_deaths": p.sum(), "obs_deaths": y.sum()}

# 1. O/E direction diagnosis
report = ["== O/E DIRECTION DIAGNOSIS ==",
          "(O/E = observed deatths / expected(deaths predicted by model sum of probabilities))",
          "O/E > 1 => model UNDER-predicts risk (more obs deaths than predicted)",
          "O/E < 1 => model OVER-predicts risk (fewer obs deaths than predicted)\n"]
for lbl, df in [("B_MIMIC-IV", B), ("A_MIMIC-III", A), ("V_eICU", V)]:
    d = calibrate(df["outcome"].values, pr_pipe(df))
    dir_ = "UNDER-pred" if d["OE"] > 1 else ("OVER-pred" if d["OE"] < 1 else "exact")
    report.append("%-12s observed=%5d  predicted=%8.1f  O/E=%.3f  => %s"%(
        lbl, d["obs_deaths"], d["pred_deaths"], d["OE"], dir_))
print("\n".join(report))
with open("outputs/v3_oe_diagnosis.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report) + "\n")

# 2. Patient-level independent split on eICU
# Stratified split 50/50 on the patient level (replaces hospital-level, unavailable)
nV = len(V)
perm = RNG.permutation(nV)
half = nV // 2
idx_epdate = perm[:half]
idx_test = perm[half:]
V_upd, V_test = V.iloc[idx_epdate].reset_index(drop=True), V.iloc[idx_test].reset_index(drop=True)
y_upd, y_test = V_upd["outcome"].values, V_test["outcome"].values

def fit_recalib(X_upd, y_upd, base_pipe):
    # logistic recalibration: slope/intercept on top of base logit
    base_p = base_pipe.predict_proba(X_upd)[:, 1]
    lp = np.log(np.clip(base_p, 1e-9, 1 - 1e-9) / np.clip(1 - base_p, 1e-9, 1 - 1e-9))
    lr = LogisticRegression(max_iter=2000).fit(lp[:, None], y_upd)
    # apply it
    base_p_test = base_pipe.predict_proba(X_upd)[:, 1]
    lp_test = np.log(np.clip(base_p_test, 1e-9, 1 - 1e-9) / np.clip(1 - base_p_test, 1e-9, 1 - 1e-9))
    lr_test = lr.predict_proba(lp_test[:, None])[:, 1]
    return lr, lr_test

pipe_recal = LogisticRegression(max_iter=500)
lr_fit, pr_recal_V_test = fit_recalib(V_upd[COMMON].astype(float), y_upd, pipe)
# evaluate on the independent test subset
pr_frozen_test = pipe.predict_proba(V_test[COMMON].astype(float))[:, 1]
# apply recalibration transform on test set
def apply_recalib(X, base_pipe, lr):
    base_p = base_pipe.predict_proba(X)[:, 1]
    lp = np.log(np.clip(base_p, 1e-9, 1 - 1e-9) / np.clip(1 - base_p, 1e-9, 1 - 1e-9))
    return lr.predict_proba(lp[:, None])[:, 1]
pr_recal_test = apply_recalib(V_test[COMMON].astype(float), pipe, lr_fit)

rows = []
for strat, p_ in [("frozen_MIMIC-IV", pr_frozen_test), ("recalibrated_on_eICU_split", pr_recal_test)]:
    c = calibrate(y_test, p_)
    rows.append((strat, roc_auc_score(y_test, p_), brier_score_loss(y_test, p_),
                 c["slope"], c["intercept"], c["OE"]))
df_upd = pd.DataFrame(rows, columns=["Strategy", "AUC", "Brier", "CalibSlope", "CalibIntercept", "O_E"])
df_upd.round(4).to_csv("outputs/v3_patient_split_validation.csv", index=False)
print("\n== PATIENT-LEVEL SPLIT VALIDATION (eICU half for fitting, half for test) ==")
print(df_upd.round(3).to_string(index=False))

# 3. Sample size learning curve
#        take varying sizes of the update subset, bootstrap repeats, evaluate on fixed test subset.
n_pat_list = [100, 200, 500, 1000, 2000, 5000]
n_events_list = [] ; rows_lc = []
for n_pat in n_pat_list:
    for rep in range(30):
        if len(V_upd) < n_pat:
            continue
        idx = RNG.choice(len(V_upd), n_pat, replace=False)
        d_sub = V_upd.iloc[idx]
        y_sub = d_sub["outcome"].values
        n_events_list.append(int(y_sub.sum()))
        lr_, _ = fit_recalib(d_sub[COMMON].astype(float), y_sub, pipe)
        pr_t = apply_recalib(V_test[COMMON].astype(float), pipe, lr_)
        auc_ = roc_auc_score(y_test, pr_t); brs_ = brier_score_loss(y_test, pr_t)
        cm = calibrate(y_test, pr_t)
        rows_lc.append((n_pat, y_sub.sum(), auc_, brs_,cm["slope"], cm["intercept"], cm["OE"], rep))
lc = pd.DataFrame(rows_lc, columns=["n_pat", "n_deaths", "AUC", "Brier", "slope", "intercept", "OE", "rep"])
lc.round(4).to_csv("outputs/v3_learning_curve.csv", index=False)
agg_lc = lc.groupby("n_pat").agg(
    mean_slope=("slope","mean"), sd_slope=("slope","std"),
    mean_Brier=("Brier","mean"), sd_Brier=("Brier","std"),
    mean_OE=("OE","mean"), sd_OE=("OE","std")).round(4)
print("\n== SAMPLE-SIZE LEARNING CURVE (mean over 30 repetitions on held-out test half) ==")
print(agg_lc.to_string())
base_frozen = calibrate(y_test, pr_frozen_test)
print("Frozen model on test-half: slope=%.3f Brier=%.3f O/E=%.3f"%(base_frozen['slope'], brier_score_loss(y_test, pr_frozen_test), base_frozen['OE']))

# Learning curve figure
plt.figure(figsize=(7, 5))
for mean_col, sd_col, c, lbl in [("mean_slope","sd_slope","tab:blue","Calibration slope"),
                                 ("mean_Brier","sd_Brier","tab:orange","Brier score")]:
    m, sd = agg_lc[mean_col].values, agg_lc[sd_col].values
    x = agg_lc.index.values
    plt.plot(x, m, "o-", color=c, label=lbl)
    plt.fill_between(x, m - sd, m + sd, alpha=0.18, color=c)
plt.axhline(base_frozen["slope"], color="tab:blue", ls="--", alpha=.4, label="Frozen slope (=0.432)")
plt.axhline(brier_score_loss(y_test, pr_frozen_test), color="tab:orange", ls="--", alpha=.4, label="Frozen Brier")
plt.axhline(1.0, color="grey", ls=":", alpha=.4, label="Ideal slope")
plt.xscale("log")
plt.xlabel("Number of eICU patients used for recalibration (log scale)")
plt.ylabel("Value"); plt.legend(); plt.tight_layout()
plt.title("Recalibration: performance vs sample size")
plt.savefig("figures/fig10_learning_curve.png", dpi=300); plt.close()

# 4. Case-mix reweighting: Reweight MIMIC-IV predictions so that the covariate
#    distribution matches eICU (post-stratify by quantile bins of key variables)
#    Use a KDE-free importance-weight approximation using propensity score
#    P(cohort=eICU|X) / P(cohort=eICU)
X_pool = pd.concat([B[COMMON].astype(float), V[COMMON].astype(float)], ignore_index=True)
isV = np.r_[np.zeros(len(B)), np.ones(len(V))]
clf_pool = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=2000))]).fit(X_pool, isV)
psV = clf_pool.predict_proba(V[COMMON].astype(float))[:, 1]     # P(eICU | X)
psB = clf_pool.predict_proba(B[COMMON].astype(float))[:, 1]     # P(eICU | X)
piV = len(V) / (len(V) + len(B))
piB = 1 - piV
wV = (psV / (1 - psV)) * (piB / piV)   # importance weight for V (treat eICU as target, source = B)
wV = np.clip(wV, 0.1, 10)

prV = pr_pipe(V); yV = V["outcome"].values; yB = B["outcome"].values
# weighted Brier + calibration
def weighted_brier(y, p, w):
    return np.average((p - y) ** 2, weights=w)
lbl_ = pd.DataFrame({"w": wV, "y": yV, "p": prV, "lp": np.log(np.clip(prV,1e-9,1-1e-9)/np.clip(1-prV,1e-9,1-1e-9))})
# weighted calibration (logit)
w_lr = LogisticRegression(max_iter=500).fit(lbl_[["lp"]].values, yV, sample_weight=wV)
w_slope = w_lr.coef_[0][0]; w_intercept = w_lr.intercept_[0]
rows_rw = [("MIMIC-IV on its own cohort (unweighted)", 1.0, len(yB), yB.sum(),
            roc_auc_score(yB, pr_pipe(B)), brier_score_loss(yB, pr_pipe(B)))]

# apply B model to eICU with weights = reweight eICU dist to MIMIC-IV? Simpler: just
# compare frozen model AUC vs weighted Brier on eICU (weighted toward MIMIC-IV population)
d = calibrate(yV, prV)
rows_rw.append(("eICU: frozen model (unweighted)", 0.0, len(yV), yV.sum(),
                roc_auc_score(yV, prV), brier_score_loss(yV, prV)))
rows_rw.append(("eICU: frozen model (reweighted to MIMIC-IV case-mix)", 0.0, len(yV), yV.sum(),
                roc_auc_score(yV, prV), weighted_brier(yV, prV, wV)))

# model for weighted calibration slope
rows_rw.append(("eICU: refit model (unweighted, reference)", 0.0, len(yV), yV.sum(),
                roc_auc_score(yV, Pipeline([("imp", SimpleImputer(strategy="median")),
                                            ("sc", StandardScaler()),
                                            ("lr", LogisticRegression(max_iter=2000))]).fit(
                    V[COMMON].astype(float), yV).predict_proba(V[COMMON].astype(float))[:, 1]),
                brier_score_loss(yV, Pipeline([("imp", SimpleImputer(strategy="median")),
                                              ("sc", StandardScaler()),
                                              ("lr", LogisticRegression(max_iter=2000))]).fit(
                    V[COMMON].astype(float), yV).predict_proba(V[COMMON].astype(float))[:, 1])))
rw = pd.DataFrame(rows_rw, columns=["Model", "Weight", "n", "deaths", "AUC", "Brier"])
rw.round(4).to_csv("outputs/v3_reweighting.csv", index=False)
print("\n== CASE-MIX REWEIGHTING (P/cohort eICU|X propensity, B-source) ==")
print(rw.round(3).to_string(index=False))
# Weighted calibration slope/intercept for documentation
print("Weighted calibration: slope=%.3f  intercept=%+.3f"%(w_slope, w_intercept))
with open("outputs/v3_reweighting.csv", "a") as f:
    f.write("\n# weighted calib slope on eICU with MIMIC-IV target-propensity weights: %.3f / intercept %+.3f"%(w_slope, w_intercept))

# Reweighting figure
plt.figure(figsize=(6, 5))
labels_ = ["Frozen model\n(unweighted eICU)", "Frozen model\n(reweighted case-mix)", "eICU refit model"]
vals_ = [brier_score_loss(yV, prV), weighted_brier(yV, prV, wV),
         brier_score_loss(yV, Pipeline([("imp", SimpleImputer(strategy="median")),
                                      ("sc", StandardScaler()),
                                      ("lr", LogisticRegression(max_iter=2000))]).fit(V[COMMON].astype(float), yV).predict_proba(V[COMMON].astype(float))[:, 1])]
cols = ["tab:red", "tab:blue", "tab:green"]
plt.bar(labels_, vals_, color=cols)
plt.axhline(vals_[0], color="red", ls=":", alpha=.5)
for i, v in enumerate(vals_):
    plt.text(i, v+0.002, "%.3f"%v, ha="center")
plt.ylabel("Brier score (lower = better)")
plt.title("Case-mix reweighting test (eICU)")
plt.tight_layout(); plt.savefig("figures/fig11_reweighting.png", dpi=300); plt.close()

# also do reweighted frozen-model prediction on MIMIC-III?  A is so small that reweighting is noisy.
print("\nWrote outputs/v3_* files and fig10/fig11.")
