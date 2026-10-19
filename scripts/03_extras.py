"""DEPRECATED — do not use for reported numbers. Retained for provenance only; superseded by the active pipeline in requirements.txt. See scripts/analysis_v5.py for the current implementation.

Enhanced secondary analysis (3 additions):
1. Third external validation on the truly geographically independent eICU
   validation cohort (14,428 CHF ICU stays from 208 US hospitals, no MIMIC overlap).
2. Sensitivity analyses: (a) cohort A (MIMIC-III) base model + NT-proBNP + LVEF;
   (b) cohort B (MIMIC-IV) base model + SOFA + APSIII.
3. Nomogram built from the multivariable logistic coefficients (frozen B model).

Outputs
- outputs/third_validation.txt
- outputs/sensitivity_analysis.csv
- figures/fig5_nomogram.png
- figures/fig6_multicohort_roc.png
"""
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss

warnings.filterwarnings("ignore")
os_dir = __import__("os")
os_dir.makedirs("figures", exist_ok=True)
os_dir.makedirs("outputs", exist_ok=True)
RNG = np.random.RandomState(42)

COMMON = ["Age", "male", "HeartRate", "RespiratoryRate", "Temperature", "SBP", "AG", "BUN",
          "Creatinine", "Chloride", "MCH", "MCHC", "MCV", "Platelet", "Potassium", "RBC",
          "RDW", "Sodium", "WBC", "COPD", "Diabetes"]

def load_and_harmonize():
    # A: MIMIC-III (n=1177 raw; 1 NaN outcome dropped)
    a = pd.read_csv("data/data01.csv").rename(columns={
        "heart rate": "HeartRate", "Respiratory rate": "RespiratoryRate",
        "temperature": "Temperature", "Systolic blood pressure": "SBP",
        "Anion gap": "AG", "Urea nitrogen": "BUN", "Blood potassium": "Potassium",
        "Blood sodium": "Sodium", "Leucocyte": "WBC", "Platelets": "Platelet", "age": "Age"})
    a["male"] = (a["gendera"] == 1).astype(int)
    a["Diabetes"] = a["diabetes"].astype(int)
    a = a[a["outcome"].notna()].copy()
    a["outcome"] = a["outcome"].astype(int)
    # B: MIMIC-IV derivation (n=15983)
    b = pd.read_csv(glob.glob("data/Derivation cohorts--MIMIC-IV database.csv")[0])
    b["male"] = (b["Gender"] == 1).astype(int)
    b["outcome"] = b["Hstatus"].astype(int)
    # V: eICU validation (n=14428)
    v = pd.read_csv(glob.glob("data/Validation cohorts--eICU-CRD.csv")[0])
    v["male"] = (v["Gender"] == 1).astype(int)
    v["outcome"] = v["Hstatus"].astype(int)
    return a, b, v

A, B, V = load_and_harmonize()

pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                 ("sc", StandardScaler()),
                 ("lr", LogisticRegression(max_iter=2000))]).fit(B[COMMON].astype(float), B["outcome"].values)

aucs = {}
for name, d in [("B_MIMIC-IV (fit)", B), ("A_MIMIC-III (external)", A), ("V_eICU (external-geo)", V)]:
    pr = pipe.predict_proba(d[COMMON].astype(float))[:, 1]
    aucs[name] = (roc_auc_score(d["outcome"], pr), brier_score_loss(d["outcome"], pr), len(d), int(d["outcome"].sum()))

# Bootstrap CIs on each cohort
def boot_ci(y, pr_draw_fn, n_boot=500):
    y = np.asarray(y); aucs_, brs_ = [], []
    for _ in range(n_boot):
        idx = RNG.choice(len(y), len(y), replace=True)
        p = pr_draw_fn(idx)
        aucs_.append(roc_auc_score(y[idx], p))
        brs_.append(brier_score_loss(y[idx], p))
    a_lo, a_hi = np.percentile(aucs_, [2.5, 97.5])
    b_lo, b_hi = np.percentile(brs_, [2.5, 97.5])
    return (a_lo, a_hi), (b_lo, b_hi)

def pr_B(idx): return pipe.predict_proba(B[COMMON].astype(float).iloc[idx])[:, 1]
def pr_A(idx): return pipe.predict_proba(A[COMMON].astype(float).iloc[idx])[:, 1]
def pr_V(idx): return pipe.predict_proba(V[COMMON].astype(float).iloc[idx])[:, 1]
ci = {}
ci["B_MIMIC-IV (fit)"] = boot_ci(B["outcome"].values, pr_B)
ci["A_MIMIC-III (external)"] = boot_ci(A["outcome"].values, pr_A)
ci["V_eICU (external-geo)"] = boot_ci(V["outcome"].values, pr_V)

# Additional: V_COPD/HF subgroup info
print("== Third validation (eICU) ==")
for k, (a_, b_, n_, d_) in aucs.items():
    (alo, ahi), (blo, bhi) = ci[k]
    print("%-25s n=%5d deaths=%4d (%.1f%%) AUC=%.3f [%.3f-%.3f] Brier=%.3f [%.3f-%.3f]" % (
        k, n_, d_, 100 * d_ / n_, a_, alo, ahi, b_, blo, bhi))

# Sensitivity analyses --------------------------------------------------------------------------------
rows = []
# (a) Cohort A + NT-proBNP + LVEF
EXTRA_A = COMMON + ["NT-proBNP", "EF"]
Ae = A[EXTRA_A].astype(float).copy()
pe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
               ("lr", LogisticRegression(max_iter=2000))]).fit(Ae, A["outcome"].values)
pr = pe.predict_proba(Ae)[:, 1]
auc_ext = roc_auc_score(A["outcome"], pr)
rows.append(("A_MIMIC-III", "Base (21 vars)", aucs["A_MIMIC-III (external)"][0]))
rows.append(("A_MIMIC-III", "Base + NT-proBNP + LVEF", auc_ext))

# in-cohort A apparent (not frozen B model) for clarity
prA_local = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                      ("lr", LogisticRegression(max_iter=2000))]
                     ).fit(A[COMMON].astype(float), A["outcome"].values)
auc_apparent_A = roc_auc_score(A["outcome"], prA_local.predict_proba(A[COMMON].astype(float))[:, 1])
rows.append(("A_MIMIC-III", "Apparent (cohort-specific fit, 21 vars)", auc_apparent_A))

# (b) Cohort B + SOFA + APSIII
EXTRA_B = COMMON + ["SOFA", "APSIII"]
Be = B[EXTRA_B].astype(float).copy()
pe2 = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                ("lr", LogisticRegression(max_iter=2000))]).fit(Be, B["outcome"].values)
auc_sofa = roc_auc_score(B["outcome"], pe2.predict_proba(Be)[:, 1])
rows.append(("B_MIMIC-IV", "Base (21 vars, in-sample)", aucs["B_MIMIC-IV (fit)"][0]))
rows.append(("B_MIMIC-IV", "Base + SOFA + APSIII (in-sample)", auc_sofa))

# (c) eICU + SOFA/APSIII + HF
EXTRA_V = COMMON + ["SOFA", "APSIII"]
Ve = V[EXTRA_V].astype(float).copy()
pe3 = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                ("lr", LogisticRegression(max_iter=2000))]).fit(Ve, V["outcome"].values)
auc_v_sofa = roc_auc_score(V["outcome"], pe3.predict_proba(Ve)[:, 1])
rows.append(("V_eICU", "Base (21 vars, in-sample)", None))
rows.append(("V_eICU", "Base + SOFA + APSIII (in-sample)", auc_v_sofa))

sens = pd.DataFrame(rows, columns=["Cohort", "Model", "AUC"])
sens.to_csv("outputs/sensitivity_analysis.csv", index=False)
print("\n== Sensitivity analysis ==")
print(sens.to_string(index=False))

# Third-validation text report ------------------------------------------------------------------------
lines = []
lines.append("THIRD EXTERNAL VALIDATION — eICU Collaborative Research Database cohort")
lines.append("(geographically distinct: 208 hospitals, no MIMIC patient overlap)\n")
for k, (a_, b_, n_, d_) in aucs.items():
    (alo, ahi), (blo, bhi) = ci[k]
    lines.append("%-25s n=%5d deaths=%4d (%.1f%%)  AUC=%.3f [95%%CI %.3f-%.3f]  Brier=%.3f [95%%CI %.3f-%.3f]" % (
        k, n_, d_, 100 * d_ / n_, a_, alo, ahi, b_, blo, bhi))
lines.append("\nSensitivity analyses (AUC, in-sample where noted):")
for c, m, x in rows:
    lines.append("%-13s %-38s %s" % (c, m, "%.3f" % x if x is not None else "(see table)"))
with open("outputs/third_validation.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\nWrote outputs/third_validation.txt")

# Nomogram -----------------------------------------------------------------------------------------------
# Points scale: 100 points at max positive contribution (per IQR standardized beta).
LR = pipe.named_steps["lr"]
coefs = LR.coef_[0]
sc = pipe.named_steps["sc"]
imp = pipe.named_steps["imp"].statistics_
# Impute A/B/V for the ones used; use linear-scaled coefficient x variable-IQR for display
iqr = {c: float(np.nanpercentile(B[c], 75) - np.nanpercentile(B[c], 25)) for c in COMMON}
intercept = LR.intercept_[0]

# Points per variable: beta_iqr * (X - median)/IQR -->  contribution range in IQR units
fig, axes = plt.subplots(len(COMMON) + 1, 1, figsize=(11, 16))
label_map = {"Age": "Age (per IQR)", "male": "Male sex",
             "HeartRate": "Heart rate (per IQR)", "RespiratoryRate": "Respiratory rate (per IQR)",
             "Temperature": "Temperature (per IQR)", "SBP": "SBP (per IQR)",
             "AG": "Anion gap (per IQR)", "BUN": "BUN (per IQR)", "Creatinine": "Creatinine (per IQR)",
             "Chloride": "Chloride (per IQR)", "MCH": "MCH (per IQR)", "MCHC": "MCHC (per IQR)",
             "MCV": "MCV (per IQR)", "Platelet": "Platelet (per IQR)", "Potassium": "Potassium (per IQR)",
             "RBC": "RBC (per IQR)", "RDW": "RDW (per IQR)", "Sodium": "Sodium (per IQR)",
             "WBC": "WBC (per IQR)", "COPD": "COPD", "Diabetes": "Diabetes"}
# max absolute beta scaled in 0-100 points
max_abs_beta = max(abs(coefs))
scale_factor = 100.0 / max_abs_beta
for i, c in enumerate(COMMON):
    ax = axes[i]
    # decide if binary (unique values <=2 in B) or continuous
    uniq = B[c].dropna().unique()
    if len(uniq) <= 2:
        vals = sorted([v for v in uniq if not np.isnan(v)])[:2]
        contribs = [coefs[i] * (v - sc.mean_[i]) / sc.scale_[i] for v in vals]
        # make binary axis: low -> high
        axis_vals = list(contribs)
    else:
        q_lo, q_hi = np.nanpercentile(B[c], [5, 95])
        xs = np.linspace(q_lo, q_hi, 100)
        axis_vals = coefs[i] * (xs - sc.mean_[i]) / sc.scale_[i]
    pts = (axis_vals - min(min(axis_vals), 0)) * scale_factor
    ax.plot(axis_vals, pts, lw=2, color="tab:blue")
    ax.axhline(0, color="grey", lw=.5); ax.axhline(100, color="grey", lw=.5)
    ax.set_ylabel(label_map[c], fontsize=9, rotation=0, ha="right", va="center")
    ax.set_yticks([]); ax.set_xticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
# total points axis
ax = axes[-1]
# compute total point range over the full B range
idx_range = np.linspace(0, 1, 50)
# approximate total via base contributions at q5-q95
all_pts = []
bq95 = {c: np.nanpercentile(B[c], 95) for c in COMMON}
bq5 = {c: np.nanpercentile(B[c], 5) for c in COMMON}
all_pts_lohi = []
for i, c in enumerate(COMMON):
    if B[c].dropna().nunique() <= 2:
        vals = sorted(B[c].dropna().unique())[:2]
        los = [coefs[i] * (v - sc.mean_[i]) / sc.scale_[i] for v in vals]
        all_pts_lohi.append((min(los), max(los)))
    else:
        lo = coefs[i] * (bq5[c] - sc.mean_[i]) / sc.scale_[i]
        hi = coefs[i] * (bq95[c] - sc.mean_[i]) / sc.scale_[i]
        all_pts_lohi.append((min(lo, hi), max(lo, hi)))
lo_total = sum(p[0] for p in all_pts_lohi)
hi_total = sum(p[1] for p in all_pts_lohi)
# logistic probability curve
xs = np.linspace(lo_total, hi_total, 200)
prob = 1 / (1 + np.exp(-(xs + intercept)))
ax.plot(xs, prob, lw=2, color="tab:red")
ax.set_xlabel("Total contribution (sum of scaled beta × value)")
ax.set_ylabel("Predicted\nprobability", fontsize=9, rotation=0, ha="right", va="center")
ax.spines[["top", "right"]].set_visible(False)
plt.suptitle("Nomogram: multivariable logistic model, fit on B_MIMIC-IV (per-IQR standardized)", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("figures/fig5_nomogram.png", dpi=300)
plt.close()

# Multi-cohort ROC plot
plt.figure(figsize=(6, 5))
for (name, d), col in zip([("B_MIMIC-IV (fit)", B), ("A_MIMIC-III (external)", A), ("V_eICU (external-geo)", V)],
                          ["tab:blue", "tab:orange", "tab:green"]):
    pr = pipe.predict_proba(d[COMMON].astype(float))[:, 1]
    y = d["outcome"].values
    fpr, tpr, _ = roc_curve(y, pr)
    plt.plot(fpr, tpr, color=col, lw=2, label="%s  AUC=%.3f" % (name, roc_auc_score(y, pr)))
plt.plot([0, 1], [0, 1], "k--", alpha=.4)
plt.xlabel("1 - Specificity"); plt.ylabel("Sensitivity")
plt.legend(loc="lower right"); plt.tight_layout()
plt.savefig("figures/fig6_multicohort_roc.png", dpi=300); plt.close()
print("Wrote figures/fig5_nomogram.png and fig6_multicohort_roc.png")
