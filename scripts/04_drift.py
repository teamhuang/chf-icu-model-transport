"""DEPRECATED — do not use for reported numbers. Retained for provenance only; superseded by the active pipeline in requirements.txt. See scripts/analysis_v5.py for the current implementation.

Calibration drift, model updating strategies, baseline-score comparison,
and performance-decomposition analysis across the three CHF-ICU cohorts.

Addresses four reviewer concerns:
1. MIMIC-III / MIMIC-IV temporal overlap  -> documented as a limitation, with AUC
   reported for the FULL MIMIC-III cohort AND acknowledged as overlapping 2008-2012.
   (True non-overlap would require re-extraction from MIMIC-III, out of scope here;
   we flag it explicitly rather than over-claim temporal independence.)
2. Predictor leakage risk              -> clearly documented in methods draft.
3. Baseline score comparison           -> cohort-specific SOFA/APSIII single-var AUC
   computed on each cohort and compared head-to-head against the frozen 21-var model.
4. Model updating strategies           -> intercept-only update, logistic recalibration,
   full refit, each evaluated on external cohorts.

Outputs
- outputs/calibration_drift.txt
- outputs/model_updating.csv
- outputs/baseline_scores.csv
- outputs/hydra_smd.csv          (variable distribution shift, standardized mean differences)
- figures/fig7_calibration_drift.png
- figures/fig8_smd_heatmap.png
- figures/fig9_goshen_dca.png    (3-cohort DCA incl. eICU)
"""
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from collections import OrderedDict

warnings.filterwarnings("ignore")
import os
os.makedirs("figures", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
RNG = np.random.RandomState(42)

COMMON = ["Age", "male", "HeartRate", "RespiratoryRate", "Temperature", "SBP", "AG", "BUN",
          "Creatinine", "Chloride", "MCH", "MCHC", "MCV", "Platelet", "Potassium", "RBC",
          "RDW", "Sodium", "WBC", "COPD", "Diabetes"]

# --- Load -------------------------------------------------------------------
def load_data():
    a = pd.read_csv("data/data01.csv").rename(columns={
        "heart rate": "HeartRate", "Respiratory rate": "RespiratoryRate",
        "temperature": "Temperature", "Systolic blood pressure": "SBP",
        "Anion gap": "AG", "Urea nitrogen": "BUN", "Blood potassium": "Potassium",
        "Blood sodium": "Sodium", "Leucocyte": "WBC", "Platelets": "Platelet",
        "age": "Age"})
    a["male"] = (a["gendera"] == 1).astype(int)
    a["Diabetes"] = a["diabetes"].astype(int)
    a = a[a["outcome"].notna()].copy()
    a["outcome"] = a["outcome"].astype(int)

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

def predict_pipe(df):
    return pipe.predict_proba(df[COMMON].astype(float))[:, 1]

prB, prA, prV = predict_pipe(B), predict_pipe(A), predict_pipe(V)
yB, yA, yV = B["outcome"].values, A["outcome"].values, V["outcome"].values

# 1. Calibration intercept & slope --------------------------------------------
def calibration_stats(y, pr):
    eps = 1e-9
    lp = np.log(np.clip(pr, eps, 1 - eps) / np.clip(1 - pr, eps, 1 - eps))
    # slope: logit(pr) coefficient predicting events
    X = lp[:, None]
    r = LogisticRegression(max_iter=500).fit(X, y)
    slope = r.coef_[0][0]
    intercept = r.intercept_[0]
    # O/E (observed / expected)
    oe = y.sum() / pr.sum()
    return intercept, slope, oe

lines = ["== CALIBRATION DRIFT =="]
lines.append("(calibration intercept: 0 = perfect; slope: 1 = perfect; O/E: observed/expected events)\n")
drift = {}
for name, y_, pr_ in [("B_MIMIC-IV (fit)", yB, prB), ("A_MIMIC-III (temporal-ext)", yA, prA),
                      ("V_eICU (geographic-ext)", yV, prV)]:
    i_, s_, oe_ = calibration_stats(y_, pr_)
    drift[name] = (i_, s_, oe_)
    lines.append("%-28s n=%5d deaths=%4d  intercept=%+.3f  slope=%.3f  O/E=%.3f  AUC=%.3f  Brier=%.3f" % (
        name, len(y_), y_.sum(), i_, s_, oe_, roc_auc_score(y_, pr_), brier_score_loss(y_, pr_)))

# 2. Model updating strategies -------------------------------------------------
def fit_strategy(strategy, X_ext, y_ext, X_fit, y_fit):
    """Fit an updating model on the external cohort; returns frozen predictions in-sample."""
    if strategy == "unchanged":
        return None, None
    X_ext_ = X_ext.copy()
    if strategy == "intercept_update":
        # 21-var frozen model, but re-fit ONLY the intercept on external data with
        # all other coefficients locked to the original scaled/unscaled betas.
        # Equivalent: penalized logit with coefficient penalty -> use base intercept offset.
        base_pr = pipe.predict_proba(X_ext)[:, 1]
        eps = 1e-9
        lp_old = np.log(np.clip(base_pr, eps, 1 - eps) / np.clip(1 - base_pr, eps, 1 - eps))
        lr = LogisticRegression(max_iter=1000)
        lr.fit(pd.DataFrame({"lp_old": lp_old}), y_ext)
        intercept_shift = lr.intercept_[0]
        return lp_old + intercept_shift, "intercept_shift={:+.3f}".format(intercept_shift)
    elif strategy == "recalibrate":
        # logistic recalibration: logit(pr) -> re-fit slope AND intercept
        lp_old = np.log(np.clip(pipe.predict_proba(X_ext)[:, 1], 1e-9, 1 - 1e-9) /
                        np.clip(1 - pipe.predict_proba(X_ext)[:, 1], 1e-9, 1 - 1e-9))
        lr = LogisticRegression(max_iter=1000)
        lr.fit(lp_old[:, None], y_ext)
        new_slope = lr.coef_[0][0]
        new_intercept = lr.intercept_[0]
        new_lp = new_slope * lp_old + new_intercept
        return 1 / (1 + np.exp(-new_lp)), "slope={:.3f}, intercept={:+.3f}".format(new_slope, new_intercept)
    elif strategy == "refit":
        lr = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                      ("lr", LogisticRegression(max_iter=2000))]).fit(X_ext, y_ext)
        return lr.predict_proba(X_ext)[:, 1], "B-refit on target cohort"
    raise ValueError(strategy)

strategies = ["unchanged", "intercept_update", "recalibrate", "refit"]
targets = [("A_MIMIC-III", A, yA), ("V_eICU", V, yV)]
update_rows = []
base_auc = {"A_MIMIC-III": roc_auc_score(yA, prA), "V_eICU": roc_auc_score(yV, prV)}
for tname, tdata, y_t in targets:
    for strat in strategies:
        if strat == "unchanged":
            pr_new = pipe.predict_proba(tdata[COMMON].astype(float))[:, 1]
            info = "frozen B model"
        else:
            pr_new, info = fit_strategy(strat, tdata[COMMON].astype(float), y_t, B[COMMON].astype(float), yB)
            # fit_strategy may return logit linear predictor for intercept_update -> convert
            if np.any((pr_new < 0) | (pr_new > 1)) or np.mean(pr_new) < 0.1:  # logit-like scale
                pr_new = 1 / (1 + np.exp(-pr_new)) if pr_new.max() > 1 or pr_new.min() < 0 else pr_new
        pr_new = np.clip(np.asarray(pr_new, float), 0.0, 1.0)
        auc_ = roc_auc_score(y_t, pr_new)
        i_, s_, oe_ = calibration_stats(y_t, pr_new)
        brs_ = brier_score_loss(y_t, pr_new)
        update_rows.append((tname, strat, auc_, auc_ - base_auc[tname], brs_, i_, s_, oe_, info))
updates = pd.DataFrame(update_rows, columns=["Cohort", "Strategy", "AUC", "AUC_gain_vs_frozen",
                                             "Brier", "CalibIntercept", "CalibSlope", "CalibOE", "Info"])
updates.round(4).to_csv("outputs/model_updating.csv", index=False)
print("== MODEL UPDATING ==")
print(updates.round(3).to_string(index=False))

# 3. Baseline severity-score comparison ---------------------------------------
def single_var_auc(df, var, y_):
    X = df[[var]].astype(float).values
    lr = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                  ("lr", LogisticRegression(max_iter=1000))]).fit(X, y_)
    return roc_auc_score(y_, lr.predict_proba(X)[:, 1])

rows = []
# frozen 21-var model AUC (already computed)
for name, y_, pr_ in [("A_MIMIC-III", yA, prA), ("V_eICU", yV, prV), ("B_MIMIC-IV", yB, prB)]:
    rows.append((name, "Frozen 21-var model", roc_auc_score(y_, pr_)))
for name, df in [("B_MIMIC-IV", B), ("A_MIMIC-III", A), ("V_eICU", V)]:
    for score in ["SOFA", "APSIII"]:
        if score in df.columns:
            u_ = single_var_auc(df, score, df["outcome"].values)
            rows.append((name, "%s alone (cohort-specific fit)" % score, u_))
# also A_MIMIC-III has SOFA? check
base = pd.DataFrame(rows, columns=["Cohort", "Model", "AUC"])
base.round(4).to_csv("outputs/baseline_scores.csv", index=False)
print("\n== BASELINE SCORE COMPARISON (cohort-level) ==")
print(base.round(3).to_string(index=False))

# 4. Variable distribution shift (SMD) between B vs A, B vs V ------------------
def smd(x1, x0):
    x1 = np.asarray(pd.Series(x1).dropna(), dtype=float)
    x0 = np.asarray(pd.Series(x0).dropna(), dtype=float)
    p1, p0 = x1.mean(), x0.mean()
    v1, v0 = x1.var(ddof=1), x0.var(ddof=1)
    pooled = np.sqrt((v1 + v0) / 2) if (v1 + v0) > 0 else 1.0
    return abs(p1 - p0) / pooled

smd_records = []
for c in COMMON:
    if B[c].nunique() <= 2:
        # binary: absolute prevalence difference
        dA = abs(B[c].mean() - A[c].mean())
        dV = abs(B[c].mean() - V[c].mean())
        smd_records.append((c, dA, dV))
    else:
        smd_records.append((c, smd(B[c], A[c]), smd(B[c], V[c])))
smd_df = pd.DataFrame(smd_records, columns=["Variable", "SMD_BvsA", "SMD_BvsV"])
# SMD for SOFA / APSIII only in A/V (B has them too? B does have SOFA & APSIII)
for extra in ["SOFA", "APSIII"]:
    if extra in B.columns and extra in A.columns:
        smd_df = pd.concat([smd_df, pd.DataFrame(
            [[extra, smd(B[extra], A[extra]), np.nan]],
            columns=smd_df.columns)], ignore_index=True)
    if extra in B.columns and extra in V.columns:
        smd_df = pd.concat([smd_df, pd.DataFrame(
            [[extra, np.nan, smd(B[extra], V[extra])]],
            columns=smd_df.columns)], ignore_index=True)
smd_df = smd_df.groupby("Variable", as_index=False).max()
smd_df.round(4).to_csv("outputs/hydra_smd.csv", index=False)
print("\n== VARIABLE DISTRIBUTION SHIFT (SMD) ==")
print(smd_df.round(3).to_string(index=False))

# ---- Figures ------------------------------------------------------------------
plt.figure(figsize=(6, 5))
for (label, y_, pr_), c in zip([("B_MIMIC-IV", yB, prB), ("A_MIMIC-III", yA, prA), ("V_eICU", yV, prV)],
                               ["tab:blue", "tab:orange", "tab:green"]):
    # calibration intercept/slope plot in deciles
    frac_pos, mean_pred = [], []
    bins = pd.qcut(pr_, 8, duplicates="drop")
    for b, gp in pd.DataFrame({"b": bins, "p": pr_, "y": y_}).groupby("b"):
        if len(gp) == 0:
            continue
        mean_pred.append(gp["p"].mean())
        frac_pos.append(gp["y"].mean())
    plt.plot(mean_pred, frac_pos, "o-", label="%s (n=%d)" % (label, len(y_)), color=c)
plt.plot([0, .6], [0, .6], "k--", alpha=.4)
plt.text(.02, .93, "\n".join(
    ["%s  intercept=%+.2f  slope=%.2f  O/E=%.2f" % (k, v[0], v[1], v[2]) for k, v in drift.items()]),
    fontsize=8, transform=plt.gcf().transFigure)
plt.xlabel("Predicted probability"); plt.ylabel("Observed proportion")
plt.title("Calibration drift across three cohorts"); plt.legend(); plt.tight_layout()
plt.savefig("figures/fig7_calibration_drift.png", dpi=300); plt.close()

plt.figure(figsize=(6, 10))
smd_plot = smd_df.dropna(subset=["SMD_BvsA", "SMD_BvsV"], how="all").sort_values("SMD_BvsA", ascending=False)
y_pos = np.arange(len(smd_plot))
plt.barh(y_pos - 0.2, smd_plot["SMD_BvsA"].fillna(0), height=0.4, label="B (MIMIC-IV) vs A (MIMIC-III)", color="tab:orange")
plt.barh(y_pos + 0.2, smd_plot["SMD_BvsV"].fillna(0), height=0.4, label="B (MIMIC-IV) vs V (eICU)", color="tab:green")
plt.axvline(0.1, color="grey", ls="--", alpha=.6, label="SMD=0.1 threshold")
plt.axvline(0.25, color="red", ls="--", alpha=.4, label="SMD=0.25 threshold")
plt.yticks(y_pos, smd_plot["Variable"])
plt.gca().invert_yaxis()
plt.xlabel("Absolute standardized mean difference (or abs. prevalence diff)")
plt.title("Variable distribution shift across cohorts"); plt.legend(); plt.tight_layout()
plt.savefig("figures/fig8_smd_heatmap.png", dpi=300); plt.close()

# --- three-cohort DCA
def dca_data(y, pr, thresholds):
    n = len(y)
    model_nb, treat_all_nb = [], []
    for t in thresholds:
        tp = ((pr >= t) & (y == 1)).sum()
        fp = ((pr >= t) & (y == 0)).sum()
        model_nb.append(tp / n - fp / n * (t / (1 - t)))
        treat_all_nb.append(y.mean() - (1 - y.mean()) * t / (1 - t))
    return np.array(model_nb), np.array(treat_all_nb)

ths = np.arange(0.01, 0.99, 0.01)
plt.figure(figsize=(6.5, 5))
colors_dca = [("B_MIMIC-IV", yB, prB, "tab:blue"), ("A_MIMIC-III", yA, prA, "tab:orange"),
              ("V_eICU", yV, prV, "tab:green")]
for label, y_, pr_, c in colors_dca:
    nb_m, _ = dca_data(y_, pr_, ths)
    plt.plot(ths, nb_m, label=label, color=c, lw=2)
plt.axhline(0, color="black", ls=":", label="Treat none")
plt.ylim(-0.05, 0.2)
plt.xlabel("Threshold probability"); plt.ylabel("Net benefit")
plt.title("Decision curve analysis (three cohorts)"); plt.legend(); plt.tight_layout()
plt.savefig("figures/fig9_goshen_dca.png", dpi=300); plt.close()

# --- write report ---------------------------------------------------------------
with open("outputs/calibration_drift.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n\n")
    f.write("== MODEL UPDATING STRATEGIES ==\n")
    f.write(updates.round(4).to_string(index=False) + "\n\n")
    f.write("== BASELINE SEVERITY SCORES ==\n")
    f.write(base.round(4).to_string(index=False) + "\n\n")
    f.write("== VARIABLE DISTRIBUTION SHIFT (SMD) ==\n")
    f.write(smd_df.round(4).to_string(index=False) + "\n")
print("\nWrote outputs/calibration_drift.txt and fig7-fig9.")
