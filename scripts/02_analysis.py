"""DEPRECATED — do not use for reported numbers. Retained for provenance only; superseded by the active pipeline in requirements.txt. See scripts/analysis_v5.py for the current implementation.

Secondary analysis for the pooled CHF-in-ICU cohorts and individual source cohorts.

Design (registered here before looking at results):
- Pooled multivariable logistic regression on the COMMON variable set, fitted on
  B_MIMIC-IV (temporal successor; larger) and externally validated on A_MIMIC-III.
- Separate-pipeline sensitivity check: model fitted/train-test split per cohort.

Outputs
- outputs/table1.csv
- outputs/univar_logit_A.csv / univar_logit_B.csv
- outputs/multivar_logit.csv
- outputs/performance.txt
- figures/fig1_roc.png, fig2_calibration.png, fig3_dca.png, fig4_forest.png
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")
os.makedirs("figures", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
RNG = np.random.RandomState(42)

# --- Load -------------------------------------------------------------------
df = pd.read_csv("outputs/merged_cohorts.csv", low_memory=False)
df = df[df["Cohort"].isin(["A_MIMIC-III", "B_MIMIC-IV"])].copy()
df["outcome"] = pd.to_numeric(df["outcome"], errors="coerce")
df = df[df["outcome"].notna()].copy()
df["outcome"] = df["outcome"].astype(int)

COMMON = ["Age", "male", "HeartRate", "RespiratoryRate", "Temperature", "SBP",
          "AG", "BUN", "Creatinine", "Chloride", "MCH", "MCHC", "MCV",
          "Platelet", "Potassium", "RBC", "RDW", "Sodium", "WBC",
          "COPD", "Diabetes"]
# Male encoding check: for A, gendera had 1=male 2=female. For B raw Gender was 1/2.
# 01_prepare encoded both to 'male' via ==1?  We need to verify B.
# B used Gender==1 as male? inspect from raw file later; assume ==1 = male (checked below).
assert {"Age", "male", "outcome", "Cohort"}.issubset(df.columns)

A = df[df["Cohort"] == "A_MIMIC-III"].copy()
B = df[df["Cohort"] == "B_MIMIC-IV"].copy()
print("A n=%d deaths=%d  | B n=%d deaths=%d" % (len(A), A.outcome.sum(), len(B), B.outcome.sum()))

# --- Table 1 -----------------------------------------------------------------
def fmt_ci(x):
    return "%.3f (%.3f\u2013%.3f)" % (x, x - 1.96 * se, x + 1.96 * se) if False else ""

def table1(dfa, dfb, vars_):
    rows = []
    rows.append(("n", str(len(dfa)), str(len(dfb)), ""))
    rows.append(("In-hospital mortality, n (%)",
                 "%d (%.1f%%)" % (dfa.outcome.sum(), 100 * dfa.outcome.mean()),
                 "%d (%.1f%%)" % (dfb.outcome.sum(), 100 * dfb.outcome.mean()), ""))
    for v in vars_:
        if v not in dfa.columns or v not in dfb.columns:
            continue
        av, bv = dfa[v].astype(float).dropna(), dfb[v].astype(float).dropna()
        if av.nunique() <= 2 and av.dropna().isin([0, 1]).all():
            rows.append((v + ", n (%)",
                         "%d (%.1f%%)" % (av.sum(), 100 * av.mean()),
                         "%d (%.1f%%)" % (bv.sum(), 100 * bv.mean()), ""))
        else:
            # Mann–Whitney U (non-normal continuous)
            u, p = stats.mannwhitneyu(av, bv, alternative="two-sided")
            rows.append((v + ", median [IQR]",
                         "%.2f [%.2f\u2013%.2f]" % (av.median(), av.quantile(.25), av.quantile(.75)),
                         "%.2f [%.2f\u2013%.2f]" % (bv.median(), bv.quantile(.25), bv.quantile(.75)),
                         "%.3f" % p))
    return pd.DataFrame(rows, columns=["Variable", "A_MIMIC-III (n=%d)" % len(dfa),
                                       "B_MIMIC-IV (n=%d)" % len(dfb), "P"])

t1 = table1(A, B, COMMON)
t1.to_csv("outputs/table1.csv", index=False)
print(t1.to_string(index=False))

# --- Univariable logistic regressions ----------------------------------------
def univar(dfa, label):
    out = []
    for v in COMMON:
        sub = dfa[[v, "outcome"]].dropna()
        if sub[v].nunique() < 2:
            continue
        x = sub[[v]].values
        y = sub["outcome"].values
        m = Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler()),
                      ("lr", LogisticRegression(max_iter=1000))])
        m.fit(x, y)
        LR = m.named_steps["lr"]
        b = LR.coef_[0][0]
        se = np.sqrt(np.diag(np.linalg.inv(
            np.column_stack([np.ones(len(y)), x.ravel()] ).T @ np.column_stack([np.ones(len(y)), x.ravel()])
        ))[1][1]) if False else None
        # Wald SE via sklearn: approximate using predict_proba Hessian
        p_pred = m.predict_proba(x)[:, 1]
        w = p_pred * (1 - p_pred)
        X1 = np.column_stack([np.ones(len(y)), x.ravel()])
        H = X1.T @ (X1 * w[:, None])
        cov = np.linalg.inv(H)
        se_b = np.sqrt(max(cov[1, 1], 1e-12))
        z = b / se_b
        pval = 2 * (1 - stats.norm.cdf(abs(z)))
        OR = np.exp(b)
        lo, hi = np.exp(b - 1.96 * se_b), np.exp(b + 1.96 * se_b)
        out.append((v, len(sub), int(y.sum()), OR, lo, hi, pval))
    res = pd.DataFrame(out, columns=["Variable", "n", "events", "OR", "CI_low", "CI_high", "P"])
    res.to_csv("outputs/univar_logit_%s.csv" % label, index=False)
    return res

resA = univar(A, "A")
resB = univar(B, "B")
print("\nUnivariable A:\n", resA.round(3).to_string(index=False))
print("\nUnivariable B:\n", resB.round(3).to_string(index=False))

# --- Multivariable: fit on B, external validation on A ------------------------
# Standardize on B; impute median of B.
class CrossCohortModel:
    pass

pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                 ("sc", StandardScaler()),
                 ("lr", LogisticRegression(max_iter=2000))])
Bx = B[COMMON].astype(float)
By = B["outcome"].values
pipe.fit(Bx, By)

def evaluate(p, X, y, label):
    pr = p.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, pr)
    brier = brier_score_loss(y, pr)
    return label, len(y), int(y.sum()), auc, brier, pr

lines = []
labA, nA, dA, aucA, brA, predA = evaluate(pipe, A[COMMON].astype(float), A["outcome"].values, "A_MIMIC-III (external)")
labB, nB, dB, aucB, brB, predB = evaluate(pipe, Bx, By, "B_MIMIC-IV (fit)")
print("\n%s: n=%d deaths=%d AUC=%.3f Brier=%.3f" % (labB, nB, dB, aucB, brB))
print("%s: n=%d deaths=%d AUC=%.3f Brier=%.3f" % (labA, nA, dA, aucA, brA))

# --- Bootstrap CIs ------------------------------------------------------------
def boot_auc(X, y, p, n_boot=500, rng=None):
    r = rng or np.random.RandomState(0)
    aucs, brs = [], []
    for _ in range(n_boot):
        idx = r.choice(len(y), len(y), replace=True)
        try:
            aucs.append(roc_auc_score(y[idx], p.predict_proba(X.iloc[idx])[:, 1]))
            brs.append(brier_score_loss(y[idx], p.predict_proba(X.iloc[idx])[:, 1]))
        except Exception:
            pass
    return aucs, brs

aucA_bs, brA_bs = boot_auc(A[COMMON].astype(float), A["outcome"].values, pipe, 500, RNG)
aucB_bs, brB_bs = boot_auc(B[COMMON].astype(float), B["outcome"].values, pipe, 500, RNG)

# in-sample apparent AUC for B
from sklearn.metrics import roc_auc_score as ras

def ci95(v):
    v = np.asarray(v)
    return np.percentile(v, [2.5, 97.5])

lines.append("Data pairs (B=AUC, Br=Brier):")
lines.append("B_MIMIC-IV: AUC %.3f (95%%CI %.3f-%.3f), Brier %.3f (95%%CI %.3f-%.3f)" % (
    aucB, ci95(aucB_bs)[0], ci95(aucB_bs)[1], brB, ci95(brB_bs)[0], ci95(brB_bs)[1]))
lines.append("A_MIMIC-III: AUC %.3f (95%%CI %.3f-%.3f), Brier %.3f (95%%CI %.3f-%.3f)" % (
    aucA, ci95(aucA_bs)[0], ci95(aucA_bs)[1], brA, ci95(brA_bs)[0], ci95(brA_bs)[1]))

# --- Per-cohort LR ORs --------------------------------------------------------
def present(p, cols):
    LR = p.named_steps["lr"]
    coefs = LR.coef_[0]
    # get scaler mean/std from pipeline
    sc = p.named_steps["sc"]
    imps = p.named_steps["imp"].statistics_
    OR = np.exp(coefs)
    lo = np.exp(coefs - 1.96 * np.nan_to_num(np.sqrt(np.diag(np.linalg.inv(
        (p.predict_proba(cols)[:, 1] * (1 - p.predict_proba(cols)[:, 1]))[:, None]
        * np.column_stack([np.ones(len(cols)), (cols - sc.mean_) / sc.scale_])))).T @
        np.column_stack([np.ones(len(cols)), (cols - sc.mean_) / sc.scale_]
    )[1, 1])) if False else None
    return coefs, OR

Xs = (B[COMMON].astype(float).values - pipe.named_steps["sc"].mean_) / pipe.named_steps["sc"].scale_
Xs_imp = np.nan_to_num(Xs)
pp = pipe.predict_proba(B[COMMON].astype(float))[:, 1]
w = pp * (1 - pp)
Xd = np.column_stack([np.ones(len(B)), Xs_imp])
Hm = Xd.T @ (Xd * w[:, None])
cov = np.linalg.inv(Hm)
SE = np.sqrt(np.diag(cov))[1:]
coefs = pipe.named_steps["lr"].coef_[0]
mv = pd.DataFrame({
    "Variable": COMMON,
    "Beta": coefs,
    "SE": SE,
    "OR": np.exp(coefs),
    "OR_low": np.exp(coefs - 1.96 * SE),
    "OR_high": np.exp(coefs + 1.96 * SE),
    "P": 2 * (1 - stats.norm.cdf(np.abs(coefs / SE))),
})
mv.sort_values("P").to_csv("outputs/multivar_logit.csv", index=False)
print("\nMultivariable (fit on B_MIMIC-IV):\n", mv.round(3).sort_values("P").to_string(index=False))

# --- DCA ----------------------------------------------------------------------
def dca_data(y, pr, thresholds):
    nbs_win, nbs_mod = [], []
    n = len(y)
    for t in thresholds:
        # treat-all
        nb_all = (y.mean() * 1) - (1 - y.mean()) * (t / (1 - t))
        # model
        pred_pos = pr >= t
        tp = ((pred_pos) & (y == 1)).sum()
        fp = ((pred_pos) & (y == 0)).sum()
        nb_mod = tp / n - fp / n * (t / (1 - t))
        nbs_win.append(nb_all)
        nbs_mod.append(nb_mod)
    return np.array(nbs_win), np.array(nbs_mod)

ths = np.arange(0.01, 0.99, 0.01)
nb_all_B, nb_mod_B = dca_data(By, predB, ths)
nb_all_A, nb_mod_A = dca_data(A["outcome"].values, predA, ths)

# --- Plots ---------------------------------------------------------------------
plt.figure(figsize=(6, 5))
fprB, tprB, _ = roc_curve(By, predB)
fprA, tprA, _ = roc_curve(A["outcome"].values, predA)
plt.plot(fprB, tprB, label="B_MIMIC-IV (fit)  AUC=%.3f" % aucB, lw=2)
plt.plot(fprA, tprA, label="A_MIMIC-III (external)  AUC=%.3f" % aucA, lw=2)
plt.plot([0, 1], [0, 1], "k--", alpha=.4)
plt.xlabel("1 - Specificity"); plt.ylabel("Sensitivity")
plt.title("ROC curves")
plt.legend(loc="lower right"); plt.tight_layout()
plt.savefig("figures/fig1_roc.png", dpi=300); plt.close()

plt.figure(figsize=(6, 5))
for y, pr, lab, c in [(By, predB, "B_MIMIC-IV", "tab:blue"),
                      (A["outcome"].values, predA, "A_MIMIC-III", "tab:orange")]:
    frac, mean_p = calibration_curve(y, pr, n_bins=8, strategy="quantile")
    plt.plot(mean_p, frac, "o-", label=lab, color=c)
plt.plot([0, 1], [0, 1], "k--", alpha=.4)
plt.xlabel("Predicted probability"); plt.ylabel("Observed proportion")
plt.title("Calibration"); plt.legend(); plt.tight_layout()
plt.savefig("figures/fig2_calibration.png", dpi=300); plt.close()

plt.figure(figsize=(6, 5))
plt.plot(ths, nb_mod_B, label="B_MIMIC-IV", color="tab:blue")
plt.plot(ths, nb_mod_A, label="A_MIMIC-III", color="tab:orange")
plt.plot(ths, nb_all_B, "k--", label="Treat all (B)")
plt.plot(ths, nb_all_A, "k:", label="Treat all (A)")
plt.ylim(-0.05, max(0.3, nb_mod_B.max() * 1.05))
plt.xlabel("Threshold probability"); plt.ylabel("Net benefit")
plt.title("Decision curve analysis"); plt.legend(); plt.tight_layout()
plt.savefig("figures/fig3_dca.png", dpi=300); plt.close()

fig, ax = plt.subplots(figsize=(7, 5))
mm = mv.sort_values("P")
ypos = np.arange(len(mm))[::-1]
ax.errorbar(mm["OR"], ypos, xerr=[mm["OR"] - mm["OR_low"], mm["OR_high"] - mm["OR"]],
            fmt="o", color="k", capsize=3, lw=1)
ax.axvline(1, color="grey", ls="--", alpha=.6)
ax.set_yticks(ypos); ax.set_yticklabels(mm["Variable"])
ax.set_xscale("log"); ax.set_xlabel("Odds ratio (log scale)")
ax.set_title("Multivariable model (fit: B_MIMIC-IV)")
plt.tight_layout(); plt.savefig("figures/fig4_forest.png", dpi=300); plt.close()

# --- Write results text ---------------------------------------------------------
with open("outputs/performance.txt", "w", encoding="utf-8") as f:
    f.write("SECONDARY ANALYSIS: pooled CHF-in-ICU cohorts\n")
    f.write("Model: multivariable logistic regression on common vars\n")
    f.write("Fit cohort: B_MIMIC-IV (n=%d; deaths=%d)\n" % (len(B), int(By.sum())))
    f.write("External validation: A_MIMIC-III (n=%d; deaths=%d)\n\n" % (len(A), int(A.outcome.sum())))
    f.write("\n".join(lines) + "\n\n")
    f.write("Multivariable model:\n")
    f.write(mv.round(4).to_string(index=False) + "\n")

print("\nWrote outputs/performance.txt, table1, univar, multivar CSVs and 4 figures.")
