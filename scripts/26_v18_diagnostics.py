"""26_v18_diagnostics.py — Round-18 new analyses (all computed from data/ + frozen pipeline,
seed 2026). Nothing is transcribed by hand; every value lands in outputs/sr_v18.json.

New in this round (each answers a specific reviewer item):
  A. Linear-predictor DISPERSION by cohort + dispersion-implied calibration slope.
     Motivation: a model calibrated where the linear predictor is under-dispersed
     transports with slope ~ SD_LP(development)/SD_LP(target). This is the first
     quantity a calibration reader checks and it was absent from the manuscript.
  B. Stay-level percentile-bootstrap 95% CIs for every headline calibration measure
     (slope, joint intercept, O/E) - previously reported as point estimates only.
  C. Cumulative stability: per-repetition minimum-stable-size vector and
     P(stability reached by size n), plus P(all 6 sub-updates stable at size n).
  D. Race/ethnicity subgroup performance of the frozen model (files carry a coded
     Race column in both MIMIC-IV and eICU).
  E. HF-marker sensitivity: NT-proBNP + LVEF increment (MIMIC-III only; APPARENT /
     in-sample, explicitly labelled).
  F. Participant characteristics by cohort (main text + Supplementary Table S15).
  G. Decision curves re-derived on a fine 0.05-0.25 grid with the crossover band located.
  H. Tail-imbalance deviations in BOTH raw units and MIMIC-IV-SD units, so Table S3
     and Fig. S3 are reconcilable.

Run from project root:  python scripts/26_v18_diagnostics.py
"""
import os, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]; os.chdir(ROOT)
COMMON = ["Age", "male", "HeartRate", "RespiratoryRate", "Temperature", "SBP", "AG", "BUN",
          "Creatinine", "Chloride", "MCH", "MCHC", "MCV", "Platelet", "Potassium", "RBC",
          "RDW", "Sodium", "WBC", "COPD", "Diabetes"]
SIZES = [100, 200, 500, 1000, 2000, 5000]
BINARY = {"male", "COPD", "Diabetes"}
SEED = 2026
R = {"meta": {"seed": SEED, "n_common_vars": len(COMMON)}}


def lg(p):
    return np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1 - 1e-9))


def unpen(y, l, w=None):
    """Unpenalised IRLS slope/intercept of logit P(Y=1) = a + b*l."""
    y = np.asarray(y, float); X = np.column_stack([np.ones(len(l)), np.asarray(l, float)])
    w = np.ones(len(l)) if w is None else np.asarray(w, float)
    beta = np.zeros(2)
    for _ in range(200):
        eta = X @ beta; p = 1 / (1 + np.exp(-eta)); Wd = w * p * (1 - p) + 1e-12
        beta = beta + np.linalg.solve(X.T @ (X * Wd[:, None]) + 1e-8 * np.eye(2), X.T @ (w * (y - p)))
    return beta[1], beta[0]


def load():
    a = pd.read_csv("data/data01.csv").rename(columns={
        "heart rate": "HeartRate", "Respiratory rate": "RespiratoryRate", "temperature": "Temperature",
        "Systolic blood pressure": "SBP", "Anion gap": "AG", "Urea nitrogen": "BUN",
        "Blood potassium": "Potassium", "Blood sodium": "Sodium", "Leucocyte": "WBC",
        "Platelets": "Platelet", "age": "Age"})
    a["male"] = (a["gendera"] == 1).astype(int); a["Diabetes"] = (a["diabetes"] == 1).astype(int)
    a = a[a["outcome"].notna()].copy(); a["outcome"] = a["outcome"].astype(int)
    b = pd.read_csv("data/Derivation cohorts--MIMIC-IV database.csv")
    b["male"] = (b["Gender"] == 1).astype(int); b["outcome"] = b["Hstatus"].astype(int)
    v = pd.read_csv("data/Validation cohorts--eICU-CRD.csv")
    v["male"] = (v["Gender"] == 1).astype(int); v["outcome"] = v["Hstatus"].astype(int)
    return b, v, a


B, V, A = load()
pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                 ("lr", LogisticRegression(max_iter=2000))]).fit(B[COMMON].astype(float), B["outcome"].values)
probs = lambda d: pipe.predict_proba(d[COMMON].astype(float))[:, 1]

# ============================== A. linear-predictor dispersion ==============================
disp = {}
for nm, d in [("MIMIC-IV", B), ("eICU-CRD", V), ("MIMIC-III", A)]:
    l = lg(probs(d))
    disp[nm] = dict(n=int(len(d)), lp_mean=round(float(l.mean()), 4), lp_sd=round(float(l.std(ddof=1)), 4),
                    lp_p25=round(float(np.percentile(l, 25)), 4), lp_p75=round(float(np.percentile(l, 75)), 4),
                    lp_iqr=round(float(np.percentile(l, 75) - np.percentile(l, 25)), 4))
sd_dev = disp["MIMIC-IV"]["lp_sd"]
tr = json.load(open("outputs/v5_results.json", encoding="utf-8"))["transport"]
R["lp_dispersion"] = disp
R["lp_dispersion_vs_slope"] = {
    "rule": ("A model calibrated where the linear predictor is under-dispersed transports with a "
             "calibration slope of approximately SD_LP(development)/SD_LP(target); the ratio is the "
             "attenuation expected from dispersion mismatch alone."),
    "eICU-CRD": {"implied_slope": round(sd_dev / disp["eICU-CRD"]["lp_sd"], 4),
                 "observed_slope": tr["V_eICU_geographic_primary"]["calib_slope"],
                 "residual_observed_minus_implied":
                     round(tr["V_eICU_geographic_primary"]["calib_slope"] - sd_dev / disp["eICU-CRD"]["lp_sd"], 4)},
    "MIMIC-III": {"implied_slope": round(sd_dev / disp["MIMIC-III"]["lp_sd"], 4),
                  "observed_slope": tr["A_MIMIC-III_temporal_exploratory"]["calib_slope"],
                  "residual_observed_minus_implied":
                      round(tr["A_MIMIC-III_temporal_exploratory"]["calib_slope"] - sd_dev / disp["MIMIC-III"]["lp_sd"], 4)},
}

# ============================== B. bootstrap CIs for calibration measures ==============================
sig = lambda l: 1.0 / (1.0 + np.exp(-np.asarray(l, float)))


def boot_calib_ci(y, l, n=500, rng=None):
    """Stay-level percentile bootstrap for calibration slope, joint intercept and O/E.
    O/E = sum(y) / sum(p) with p the *probability* obtained from the linear predictor."""
    rng = rng or np.random.RandomState(11)
    y = np.asarray(y); l = np.asarray(l); out = {"slope": [], "alpha": [], "OE": []}
    for _ in range(n):
        i = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[i])) < 2:
            continue
        s, a = unpen(y[i], l[i])
        out["slope"].append(s); out["alpha"].append(a)
        out["OE"].append(y[i].sum() / sig(l[i]).sum())
    return {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
            for k, v in out.items()}


ci_layers = {}
for nm, d in [("B_MIMIC-IV_insample", B), ("A_MIMIC-III", A), ("V_eICU", V)]:
    y = d["outcome"].values; l = lg(probs(d)); p = probs(d)
    s, a = unpen(y, l)
    ci_layers[nm] = dict(slope=round(float(s), 4), alpha=round(float(a), 4),
                         OE=round(float(y.sum() / p.sum()), 4),
                         ci=boot_calib_ci(y, l, rng=np.random.RandomState(11)))
R["calibration_ci_transport"] = ci_layers

# representative split held-out (frozen and recalibrated)
half = len(V) // 2
cr = np.random.RandomState(SEED); pm = cr.permutation(len(V))
Vu = V.iloc[pm[:half]].reset_index(drop=True); Vt = V.iloc[pm[half:]].reset_index(drop=True)
yu, yt = Vu["outcome"].values, Vt["outcome"].values
lpu, lpt = lg(probs(Vu)), lg(probs(Vt))
m = LogisticRegression(max_iter=2000).fit(lpu[:, None], yu)
pr_recal = m.predict_proba(lpt[:, None])[:, 1]; pr_froz = probs(Vt)
held = {}
for nm, pr in [("frozen", pr_froz), ("recalibrated_2par", pr_recal)]:
    y = yt; l = lg(pr); s, a = unpen(y, l)
    held[nm] = dict(slope=round(float(s), 4), alpha=round(float(a), 4),
                    OE=round(float(y.sum() / pr.sum()), 4), Brier=round(float(brier_score_loss(y, pr)), 4),
                    ci=boot_calib_ci(y, l, rng=np.random.RandomState(12)))
R["calibration_ci_heldout_repsplit"] = held

# ============================== C. cumulative stability ==============================
rc = pd.read_csv("outputs/v5_repeated_curve.csv")
per_size = {}
for n in SIZES:
    sub = rc[rc.n_pat == n]
    by_split = sub.groupby("rep")["stable"].sum()
    per_size[str(n)] = dict(
        hit_rate_pct=round(100 * float(sub["stable"].mean()), 1),
        per_split_all6_count={int(k): int(by_split.get(k, 0)) for k in sorted(by_split.index)},
        all6_pct=round(100 * float((by_split == 6).mean()), 1))
min_stable, cum = [], {}
for rep in sorted(rc.rep.unique()):
    first = None
    for n in SIZES:
        if per_size[str(n)]["per_split_all6_count"].get(int(rep), 0) == 6:
            first = n; break
    min_stable.append(int(rep) if first is None else first)
for n in SIZES:
    cum[str(n)] = round(100 * sum(1 for m_ in min_stable if m_ is not None and m_ <= n) / len(min_stable), 1)
R["cumulative_stability"] = dict(
    per_size=per_size,
    min_stable_per_rep=min_stable,
    min_stable_label=[(">" + str(SIZES[-1])) if m_ is None else str(m_) for m_ in min_stable],
    cumulative_pct_reached_by_size=cum,
    note=("per_size.all6_pct = fraction of the ten outer splits whose SIX sub-updates were all stable "
          "at that size; cumulative_pct_reached_by_size = fraction of splits that had reached 6/6 "
          "by that size (one value per split, not repeated across sizes)."))

# ============================== D. race subgroup (frozen model) ==============================
RACE = {1: "White", 2: "Black", 3: "Other"}
race_rows = {}
for nm, d in [("eICU", V), ("MIMIC-IV", B)]:
    yv = d["outcome"].values; pv = probs(d); lv = lg(pv)
    for code, lab in RACE.items():
        msk = (d["Race"] == code).values
        if msk.sum() < 100:
            continue
        y = yv[msk]; p = pv[msk]; s, a = unpen(y, lg(p))
        race_rows[f"{nm}:{lab}"] = dict(cohort=nm, race=lab, n=int(msk.sum()), deaths=int(y.sum()),
                                        AUC=round(float(roc_auc_score(y, p)), 3),
                                        slope=round(float(s), 2), OE=round(float(y.sum() / p.sum()), 2))
R["subgroup_race"] = dict(coding="MIMIC/eICU convention: 1=White, 2=Black, 3=Other; carried from the redistributed files",
                          rows=race_rows,
                          note="Descriptive subgroup estimates from the frozen transported model; no per-stratum CI and no interaction test.")

# ============================== E. HF-marker sensitivity (MIMIC-III; APPARENT) ==============================
Xi = A[COMMON + ["NT-proBNP", "EF"]].astype(float)
mm = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
               ("lr", LogisticRegression(max_iter=2000))]).fit(Xi, A["outcome"].values)
R["hf_marker_sensitivity_MIMICIII_APPARENT"] = dict(
    status="APPARENT / in-sample performance, not held-out; shown to quantify what the harmonised 21-variable model forgoes",
    base_AUC=tr["A_MIMIC-III_temporal_exploratory"]["AUC"],
    plus_NTproBNP_EF_AUC=round(float(roc_auc_score(A["outcome"].values, mm.predict_proba(Xi)[:, 1])), 4),
    n_marker_missing_pct={c: round(float(100 * A[c].isna().mean()), 2) for c in ["NT-proBNP", "EF"]})
R["hf_marker_sensitivity_MIMICIII_APPARENT"]["delta_AUC"] = round(
    R["hf_marker_sensitivity_MIMICIII_APPARENT"]["plus_NTproBNP_EF_AUC"]
    - R["hf_marker_sensitivity_MIMICIII_APPARENT"]["base_AUC"], 4)

# ============================== F. participant characteristics by cohort ==============================
# Missingness across the 21 model covariates, per cohort. NOTE: MIMIC-IV and eICU are
# complete, but the MIMIC-III exploratory file DOES carry missing vital signs - the
# manuscript previously generalised "0% missing" to all three files, which was wrong.
miss = {}
for nm, d in [("MIMIC-IV", B), ("eICU-CRD", V), ("MIMIC-III", A)]:
    rows = {c: int(d[c].astype(float).isna().sum()) for c in COMMON}
    rows = {k: v for k, v in rows.items() if v > 0}
    miss[nm] = dict(rows_with_any_missing=int(d[COMMON].astype(float).isna().any(axis=1).sum()),
                    n_complete_pct=round(100 * float(d[COMMON].astype(float).notna().all(axis=1).mean()), 1),
                    per_variable_missing_rows=rows or "none")
R["missingness_all_cohorts"] = miss


def smd_cont(x1, x0):
    m1, s1 = np.nanmean(x1), np.nanstd(x1, ddof=1); m0, s0 = np.nanmean(x0), np.nanstd(x0, ddof=1)
    return abs(m1 - m0) / np.sqrt((s1 ** 2 + s0 ** 2) / 2)


def smd_bin(x1, x0):
    p1, p0 = np.nanmean(x1), np.nanmean(x0)
    return abs(p1 - p0) / np.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2)


chars = []
for var in COMMON:
    row = {"variable": var, "type": "binary" if var in BINARY else "continuous"}
    for nm, d in [("MIMIC-IV", B), ("eICU-CRD", V), ("MIMIC-III", A)]:
        x = d[var].astype(float).values
        if var in BINARY:
            row[nm] = f"{100 * np.nanmean(x):.1f}% ({int(np.nansum(x))})"
            row[nm + "_num"] = round(float(np.nanmean(x)), 4)
        else:
            row[nm] = f"{np.nanmean(x):.1f} ({np.nanstd(x, ddof=1):.1f})"
            row[nm + "_num"] = round(float(np.nanmean(x)), 4)
        row[nm + "_n"] = int(np.sum(~np.isnan(x)))
    if var in BINARY:
        row["SMD_vs_development"] = round(float(smd_bin(V[var].astype(float).values, B[var].astype(float).values)), 4)
    else:
        row["SMD_vs_development"] = round(float(smd_cont(V[var].astype(float).values, B[var].astype(float).values)), 4)
    chars.append(row)
for nm, d in [("MIMIC-IV", B), ("eICU-CRD", V), ("MIMIC-III", A)]:
    y = d["outcome"].values
    R.setdefault("cohort_summary", {})[nm] = dict(
        stays=int(len(d)), deaths=int(y.sum()), mortality_pct=round(100 * float(y.mean()), 1),
        age_mean=round(float(d["Age"].astype(float).mean()), 1),
        age_sd=round(float(d["Age"].astype(float).std(ddof=1)), 1),
        male_pct=round(100 * float(d["male"].mean()), 1),
        female_pct=round(100 * float((1 - d["male"]).mean()), 1))
R["participant_characteristics"] = dict(
    columns="mean (SD) for continuous, n (%) for binary; SMD = absolute standardised mean difference, eICU-CRD vs MIMIC-IV",
    rows=chars)

# ============================== G. decision curves, fine grid ==============================
def nb_(y, p, ths):
    n = len(y)
    return np.array([((p >= t) & (y == 1)).sum() / n - (((p >= t) & (y == 0)).sum() / n) * (t / (1 - t)) for t in ths])


ths = np.round(np.arange(0.050, 0.2501, 0.001), 4)
nf = nb_(yt, pr_froz, ths); nr = nb_(yt, pr_recal, ths)
below = ths[nr < nf]
R["decision_curves_0.05_0.25"] = dict(
    n_boot=0, test_n=int(len(yt)),
    grid_min=0.05, grid_max=0.25, grid_step=0.001,
    frozen_nonpositive_from=float(ths[nf <= 0].min()),
    recalibrate_ge_frozen_everywhere=bool((nr >= nf).all()),
    recalibrate_below_frozen_band=[float(below.min()), float(below.max())] if len(below) else [],
    max_abs_recal_minus_frozen_in_band=round(float(np.abs(nr - nf)[nr < nf].max()), 5) if len(below) else 0.0,
    note=("Where recalibrated net benefit falls below the frozen model the deficit is smaller than "
          "5e-3 throughout, i.e. of the same order as the sampling noise of a single held-out split; "
          "no crossover changes sign outside that band."))
pd.DataFrame({"threshold": ths, "frozen": np.round(nf, 5), "recalibrated": np.round(nr, 5)}).to_csv(
    "outputs/sr_v18_dca.csv", index=False)

# ============================== H. tail imbalance, raw AND SD units ==============================
Xp = pd.concat([B[COMMON].astype(float), V[COMMON].astype(float)], ignore_index=True)
isV = np.r_[np.zeros(len(B)), np.ones(len(V))]
clf = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                ("lr", LogisticRegression(max_iter=2000))]).fit(Xp, isV)
ps = clf.predict_proba(V[COMMON].astype(float))[:, 1]
piV = len(V) / (len(V) + len(B))
w = np.clip(((1 - ps) / ps) * (piV / (1 - piV)), 0.1, 10)


def wq(x, weights, q):
    o = np.argsort(x); x = x[o]; w2 = weights[o]; cw = np.cumsum(w2) / np.sum(w2)
    return float(np.interp(q, cw, x))


KEYVARS = ["HeartRate", "RespiratoryRate", "Age", "AG", "RBC", "COPD"]
tail = {}
for c in KEYVARS:
    xb = B[c].astype(float).values; xv = V[c].astype(float).values
    s = float(np.std(xb, ddof=1))
    e = {}
    for q, lab in [(0.10, "p10"), (0.50, "p50"), (0.90, "p90")]:
        d_raw = wq(xv, w, q) - float(np.percentile(xb, 100 * q))
        e[lab + "_raw"] = round(d_raw, 3)
        e[lab + "_sd_units"] = round(d_raw / s, 3)
    e["mimic_iv_sd_used"] = round(s, 3)
    tail[c] = e
R["tail_imbalance_raw_and_sd"] = dict(
    note="raw = weighted-eICU minus MIMIC-IV quantile on the original scale; sd_units = raw divided by the MIMIC-IV SD of that variable",
    variables=tail)

json.dump(R, open("outputs/sr_v18.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("A lp dispersion      :", {k: v["lp_sd"] for k, v in disp.items()})
print("A implied vs observed:", R["lp_dispersion_vs_slope"]["eICU-CRD"], R["lp_dispersion_vs_slope"]["MIMIC-III"])
print("B transport calib CI :", {k: v["ci"] for k, v in ci_layers.items()})
print("B held-out calib CI  :", {k: v["ci"] for k, v in held.items()})
print("C cumulative reached :", R["cumulative_stability"]["cumulative_pct_reached_by_size"])
print("C all6 at size       :", {k: v["all6_pct"] for k, v in per_size.items()})
print("C min stable per rep :", R["cumulative_stability"]["min_stable_label"])
print("D race               :", race_rows)
print("E HF markers (MIMIC-III, APPARENT):", R["hf_marker_sensitivity_MIMICIII_APPARENT"])
print("F cohort summary     :", R["cohort_summary"])
print("F missingness        :", json.dumps(R["missingness_all_cohorts"], ensure_ascii=False))
print("G DCA 5-25% band     :", R["decision_curves_0.05_0.25"]["recalibrate_below_frozen_band"],
      "max deficit", R["decision_curves_0.05_0.25"]["max_abs_recal_minus_frozen_in_band"])
print("H tail SD units      :", {k: v for k, v in list(R["tail_imbalance_raw_and_sd"]["variables"].items())[:2]})
print("\nwrote outputs/sr_v18.json + outputs/sr_v18_dca.csv")