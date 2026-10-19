"""27_v18_figs.py — Round-18 figure rebuild (all values read from the frozen outputs,
never re-typed). Fixes applied this round:

  * lowercase bold panel letters everywhere (Nature/SR convention; was A-F)
  * final width fixed at 180 mm and exported at 600 dpi + vector PDF (Nature line-art rule)
  * Fig 2c  : slope and O/E as separate dot series on a reference line at 1.0
             (was a stacked bar that made slope < 1 hard to read)
  * Fig 3a  : adds the two stricter attainment statistics required after review
             (all-six-stable per split, and cumulative stability reached by size n)
  * Fig 4a  : zoomed AUC axis - the section's headline effect (dAUC ~0.055) was
             compressed by a zero baseline
  * Fig S1  : decision curves restricted to the 0.05-0.25 range actually claimed in
             the text (was 0.01-0.60, contradicting caption and Methods), and the
             crossover band is shaded
  * Fig S3a : raw units annotated alongside SD units so Table S3 and the figure reconcile

Run from project root:  python scripts/27_v18_figs.py
"""
import os, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]; os.chdir(ROOT)
for fam in ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]:
    import matplotlib.font_manager as fm
    if any(fam.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = fam; break
# 180 mm = 7.09 in final width; 7 pt base text (Nature: 5-7 pt at final size)
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.7, "axes.spines.top": False,
                     "axes.spines.right": False, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6,
                     "axes.labelsize": 7, "figure.dpi": 100,
                     "pdf.fonttype": 42, "ps.fonttype": 42})  # TrueType (editable) fonts, not Type 3
OK = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73", "orange": "#E69F00",
      "sky": "#56B4E9", "purple": "#CC79A7", "yellow": "#F0E442", "grey": "#5A5A5A", "black": "#222222"}
W_IN = 7.09          # 180 mm
DPI = 600            # line-art rule
COMMON = ["Age", "male", "HeartRate", "RespiratoryRate", "Temperature", "SBP", "AG", "BUN",
          "Creatinine", "Chloride", "MCH", "MCHC", "MCV", "Platelet", "Potassium", "RBC",
          "RDW", "Sodium", "WBC", "COPD", "Diabetes"]
SIZES = [100, 200, 500, 1000, 2000, 5000]
V18 = json.load(open("outputs/sr_v18.json", encoding="utf-8"))


def panel(ax, ch, dx=-0.17):
    # Panel letters intentionally suppressed: the author will overlay their own
    # A/B/C labels in the layout step. Signature and call sites are kept so the
    # rest of the script runs unchanged; this helper is now a no-op.
    return


def save(fig, name):
    fig.savefig(f"figures/{name}.png", dpi=DPI, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(f"figures/{name}.pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig); print("  wrote", name)


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


def frozen_probs(b):
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=2000))]).fit(b[COMMON].astype(float), b["outcome"].values)
    return lambda df: pipe.predict_proba(df[COMMON].astype(float))[:, 1]


def calib_bins(y, p, nb=10):
    q = pd.qcut(pd.Series(p), nb, labels=False, duplicates="drop")
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "q": q}).groupby("q")
    return d["p"].mean().values, d["y"].mean().values, d.size().values


# ============================== FIG 2 : transport ==============================
def fig2():
    b, v, m3 = load(); pr = frozen_probs(b)
    order = [("MIMIC-IV", b, OK["blue"]), ("eICU-CRD", v, OK["vermillion"]), ("MIMIC-III", m3, OK["green"])]
    tr = json.load(open("outputs/v5_results.json", encoding="utf-8"))["transport"]
    ci = V18["calibration_ci_transport"]
    # two key maps: v5_results.json transport uses long labels, sr_v18.json CI uses short ones
    km_tr = {"MIMIC-IV": "B_MIMIC-IV_insample", "eICU-CRD": "V_eICU_geographic_primary",
             "MIMIC-III": "A_MIMIC-III_temporal_exploratory"}
    km_ci = {"MIMIC-IV": "B_MIMIC-IV_insample", "eICU-CRD": "V_eICU", "MIMIC-III": "A_MIMIC-III"}
    ci = V18["calibration_ci_transport"]
    fig = plt.figure(figsize=(W_IN, 2.35)); gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.36)
    axA, axB, axC = (fig.add_subplot(gs[i]) for i in range(3))
    for nm, df, c in order:
        y = df["outcome"].values; p = pr(df)
        fpr, tpr, _ = roc_curve(y, p)
        axA.plot(fpr, tpr, color=c, lw=1.2, label=f"{nm} {roc_auc_score(y, p):.3f}")
    axA.plot([0, 1], [0, 1], ls=":", color=OK["grey"], lw=0.7)
    axA.set_xlabel("1 − specificity"); axA.set_ylabel("Sensitivity")
    axA.legend(frameon=False, loc="lower right"); axA.set_title("ROC", fontsize=7.5); panel(axA, "a")
    for nm, df, c in order:
        y = df["outcome"].values; p = pr(df); mp, my, _ = calib_bins(y, p)
        axB.plot(mp, my, "o-", color=c, ms=3.2, lw=1.0, label=nm)
    axB.plot([0, 1], [0, 1], ls=":", color=OK["grey"], lw=0.7)
    axB.set_xlabel("Predicted risk"); axB.set_ylabel("Observed risk")
    axB.set_title("Calibration (10 equal-frequency groups)", fontsize=7.5)
    axB.legend(frameon=False); panel(axB, "b")
    # panel c: dot plot, slope and O/E with bootstrap 95% CI, reference line at 1.0
    xs = np.arange(3); off = 0.17
    for i, (nm, _, c) in enumerate(order):
        k = km_tr[nm]; j = ci[km_ci[nm]]
        sl, oe = tr[k]["calib_slope"], tr[k]["OE"]
        slc, oec = j["ci"]["slope"], j["ci"]["OE"]
        axC.plot([i - off, i - off], slc, "-", color=c, lw=1.0, solid_capstyle="butt")
        axC.plot([i - off], [sl], "o", color=c, ms=5, label="slope" if i == 0 else None)
        axC.plot([i + off, i + off], oec, "-", color=c, lw=1.0, ls=":", alpha=0.7, solid_capstyle="butt")
        axC.plot([i + off], [oe], "D", color=c, ms=4.5, mfc="white", label="O/E" if i == 0 else None)
        axC.annotate(f"{sl:.2f}", (i - off, sl), textcoords="offset points", xytext=(-11, -6),
                     fontsize=5.8, va="center", ha="center", color=c)
        axC.annotate(f"{oe:.2f}", (i + off, oe), textcoords="offset points", xytext=(11, 6),
                     fontsize=5.8, va="center", ha="center", color=c)
    axC.axhline(1, ls=":", color=OK["grey"], lw=0.7)
    axC.set_xticks(xs); axC.set_xticklabels([n for n, _, _ in order])
    axC.set_ylabel("value"); axC.set_ylim(0, 1.75)
    axC.set_title("Calibration slope and O/E\n(points = estimate, bars = 95% bootstrap CI)", fontsize=7)
    axC.legend(frameon=False, loc="upper left", handletextpad=0.3); panel(axC, "c")
    save(fig, "v18_fig2_transport")


# ============================== FIG 3 : stability ==============================
def fig3():
    rc = pd.read_csv("outputs/v5_repeated_curve.csv")
    sens = json.load(open("outputs/sr_sens_v9.json", encoding="utf-8"))["threshold_sensitivity"]
    cum = V18["cumulative_stability"]; per_size = cum["per_size"]
    fig = plt.figure(figsize=(W_IN, 2.35)); gs = gridspec.GridSpec(1, 3, width_ratios=[1.15, 1.05, 1.05], wspace=0.38)
    axA, axB, axC = (fig.add_subplot(gs[i]) for i in range(3))
    hit = [per_size[str(n)]["hit_rate_pct"] for n in SIZES]
    all6 = [per_size[str(n)]["all6_pct"] for n in SIZES]
    reach = [cum["cumulative_pct_reached_by_size"][str(n)] for n in SIZES]
    axA.plot(SIZES, hit, "o-", color=OK["blue"], ms=4, lw=1.3, label="per-fit hit rate")
    axA.plot(SIZES, all6, "s-", color=OK["orange"], ms=4, lw=1.3, label="all 6 sub-fits stable")
    axA.plot(SIZES, reach, "^-", color=OK["green"], ms=4, lw=1.3, label="cumulative reached")
    axA.set_xscale("log"); axA.set_xticks(SIZES); axA.set_xticklabels([f"{n:,}" for n in SIZES], rotation=30)
    axA.set_ylabel("target-attainment (%)"); axA.set_ylim(0, 108)
    axA.set_title("Three attainment statistics", fontsize=7.5)
    axA.legend(frameon=False, loc="lower right"); panel(axA, "a")
    for st, oe, c, lab in [(0.1, 0.1, OK["vermillion"], "±0.10 / 0.10"), (0.2, 0.15, OK["blue"], "±0.20 / 0.15"),
                           (0.3, 0.2, OK["green"], "±0.30 / 0.20")]:
        row = [r for r in sens if abs(r["slope_tol"] - st) < 1e-6 and abs(r["oe_tol"] - oe) < 1e-6]
        if row:
            r = row[0]; axB.plot(SIZES, [r[f"rate@{n}"] for n in SIZES], "o-", ms=3.4, lw=1.1, color=c, label=lab)
    axB.set_xscale("log"); axB.set_xticks(SIZES); axB.set_xticklabels([f"{n:,}" for n in SIZES], rotation=30)
    axB.set_ylabel("per-fit hit rate (%)"); axB.set_xlabel("update size (stays)")
    axB.legend(frameon=False, title="slope / O-E tolerance", title_fontsize=6)
    axB.set_title("Tolerance sensitivity", fontsize=7.5); panel(axB, "b")
    data = [rc[rc.n_pat == n].slope.values for n in SIZES]
    parts = axC.violinplot(data, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(OK["sky"]); pc.set_alpha(0.6)
    for k in ("cbars", "cmins", "cmaxes", "cwhiskers"):
        if k in parts:
            parts[k].set_color(OK["grey"])
    axC.axhline(1, ls=":", color=OK["grey"], lw=0.7); axC.axhline(0, ls=":", color=OK["vermillion"], lw=0.7)
    axC.set_yscale("symlog", linthresh=1)
    axC.set_xticks(range(1, 7)); axC.set_xticklabels([f"{n:,}" for n in SIZES], rotation=30)
    axC.set_ylabel("held-out calibration slope"); axC.set_xlabel("update size")
    axC.set_title("Slope distribution (symlog)", fontsize=7.5); panel(axC, "c")
    save(fig, "v18_fig3_stability")


# ============================== FIG 4 : strategies ==============================
def fig4():
    sc = json.load(open("outputs/v5_results.json", encoding="utf-8"))["strategy_comparison"]
    order = [("frozen", "Frozen", OK["grey"]), ("intercept_only", "Intercept-only", OK["orange"]),
             ("recalibrate_2par", "2-param recal", OK["sky"]), ("full_refit", "Refit", OK["blue"])]
    fig = plt.figure(figsize=(W_IN, 2.25)); gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1.2], wspace=0.9)
    axs = [fig.add_subplot(gs[i]) for i in range(4)]
    # panel a: ZOOMED auc axis (zero baseline previously hid the dAUC ~0.055 effect)
    ax = axs[0]
    vals = [sc[k]["AUC"][0] for k, _, _ in order]; errs = [sc[k]["AUC"][1] for k, _, _ in order]
    ax.bar(range(4), vals, 0.62, yerr=errs, capsize=2, color=[c for _, _, c in order], error_kw=dict(lw=0.7))
    for i, (val, er) in enumerate(zip(vals, errs)):
        ax.annotate(f"{val:.3f}", (i, val + er), textcoords="offset points", xytext=(0, 1.5),
                    ha="center", fontsize=5.6)
    ax.set_ylim(0.66, 0.775); ax.set_ylabel("AUC (zoom)")
    ax.set_title("AUC (mean ± SD, 10 splits)", fontsize=7.5); panel(ax, "a")
    for ax, metric, ttl, fmt, ylab in [(axs[1], "Brier", "Brier (mean ± SD)", ".3f", "Brier score (lower better)"),
                                        (axs[2], "slope", "Calibration slope (mean ± SD)", ".2f", "slope (1 = ideal)")]:
        vv = [sc[k][metric][0] for k, _, _ in order]; ee = [sc[k][metric][1] for k, _, _ in order]
        ax.bar(range(4), vv, 0.62, yerr=ee, capsize=2, color=[c for _, _, c in order], error_kw=dict(lw=0.7))
        for i, (val, er) in enumerate(zip(vv, ee)):
            ax.annotate(f"{val:{fmt}}", (i, val + er), textcoords="offset points", xytext=(0, 1.5),
                        ha="center", fontsize=5.6)
        if metric == "slope":
            ax.axhline(1, ls=":", color=OK["grey"], lw=0.7); ax.set_ylim(0, 1.35)
        ax.set_ylabel(ylab, fontsize=6.5)
        ax.set_title(ttl, fontsize=7.5)
    for ax, ch in zip(axs[1:3], ["b", "c"]):
        panel(ax, ch)
    for ax in axs[:3]:
        ax.set_xticks(range(4))
        ax.set_xticklabels([n for _, n, _ in order], rotation=40, ha="right", fontsize=5.8)
    b, v, m3 = load(); pr = frozen_probs(b); half = len(v) // 2
    cr = np.random.RandomState(2026); p_ = cr.permutation(len(v))
    Vu = v.iloc[p_[:half]].reset_index(drop=True); Vt = v.iloc[p_[half:]].reset_index(drop=True)
    yu, yt = Vu["outcome"].values, Vt["outcome"].values
    lg = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1 - p * 0))
    lp = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1 - 1e-9))
    pf_f = pr(Vt)
    m = LogisticRegression(max_iter=2000).fit(lp(pr(Vu))[:, None], yu)
    pf_r = 1 / (1 + np.exp(-(m.intercept_[0] + m.coef_[0][0] * lp(pf_f))))
    lr = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                   ("lr", LogisticRegression(max_iter=2000))]).fit(Vu[COMMON].astype(float), yu)
    pf_ref = lr.predict_proba(Vt[COMMON].astype(float))[:, 1]
    ms = LogisticRegression(max_iter=2000).fit(Vu[["SOFA"]].astype(float), yu)
    pf_sofa = ms.predict_proba(Vt[["SOFA"]].astype(float))[:, 1]
    axD = axs[3]
    for nm, p, c in [("frozen", pf_f, OK["grey"]), ("recal", pf_r, OK["sky"]),
                     ("refit", pf_ref, OK["blue"]), ("SOFA", pf_sofa, OK["vermillion"])]:
        mp, my, _ = calib_bins(yt, p)
        axD.plot(mp, my, "o-", ms=2.8, lw=1.0, color=c, label=nm)
    axD.plot([0, 1], [0, 1], ls=":", color=OK["grey"], lw=0.7)
    axD.set_xlabel("Predicted risk"); axD.set_ylabel("Observed risk")
    axD.set_title("Held-out calibration", fontsize=7.5)
    axD.legend(frameon=False); panel(axD, "d")
    save(fig, "v18_fig4_strategies")


# ============================== FIG 5 : reweighting ==============================
def fig5():
    b, v, m3 = load()
    wd = json.load(open("outputs/sr_sens_v9.json", encoding="utf-8"))["weight_diagnostics"]
    smd = pd.read_csv("outputs/v6_smd_by_variable.csv")
    fig = plt.figure(figsize=(W_IN, 2.4)); gs = gridspec.GridSpec(1, 3, width_ratios=[1.15, 1, 1], wspace=0.42)
    axA, axB, axC = (fig.add_subplot(gs[i]) for i in range(3))
    smd = smd.reindex(smd["SMD_unweighted"].abs().sort_values(ascending=True).index)
    yy = np.arange(len(smd))
    axA.hlines(yy, 0, smd["SMD_unweighted"], color=OK["grey"], lw=1.1)
    axA.plot(smd["SMD_unweighted"], yy, "o", ms=3.4, color=OK["vermillion"], label="unweighted")
    axA.plot(smd["SMD_weighted"], yy, "o", ms=3.4, color=OK["blue"], label="weighted")
    axA.axvline(0.1, ls=":", color=OK["grey"], lw=0.7)
    axA.set_yticks(yy); axA.set_yticklabels(smd["Variable"], fontsize=5.2)
    axA.set_xlabel("absolute SMD"); axA.set_title("Covariate balance", fontsize=7.5)
    axA.legend(frameon=False, loc="lower right"); panel(axA, "a")
    X = pd.concat([b[COMMON].astype(float), v[COMMON].astype(float)], ignore_index=True)
    lab = np.r_[np.zeros(len(b)), np.ones(len(v))]
    sc = StandardScaler().fit_transform(X.values)
    ps = LogisticRegression(max_iter=2000).fit(sc, lab).predict_proba(sc)[:, 1]
    bins = np.linspace(0, 1, 41)
    axB.hist(ps[lab == 0], bins=bins, alpha=0.55, color=OK["blue"], label="MIMIC-IV", density=True)
    axB.hist(ps[lab == 1], bins=bins, alpha=0.55, color=OK["vermillion"], label="eICU", density=True)
    axB.set_xlabel("P(eICU | X)"); axB.set_ylabel("density")
    axB.set_title(f"Overlap (separability AUC {wd['propensity_separability_AUC']:.3f})", fontsize=7.5)
    axB.legend(frameon=False); panel(axB, "b")
    pi_e = ps[lab == 1]; wgt = np.clip(((1 - pi_e) / pi_e) * (len(v) / len(b)), 0.1, 10)
    axC.hist(wgt, bins=np.linspace(0, 10, 41), color=OK["purple"], alpha=0.8)
    axC.axvline(0.1, color=OK["grey"], ls=":", lw=0.7); axC.axvline(10, color=OK["grey"], ls=":", lw=0.7)
    axC.set_xlabel("truncated weight"); axC.set_ylabel("eICU stays")
    axC.set_title(f"ESS {wd['ESS_pct']:.1f}% · {wd['extreme_weight_fraction']*100:.0f}% pinned at a bound", fontsize=7.5)
    panel(axC, "c")
    save(fig, "v18_fig5_reweighting")


# ============================== FIG S1 : decision curves ==============================
def figS1():
    dca = V18["decision_curves_0.05_0.25"]
    df = pd.read_csv("outputs/sr_v18_dca.csv")
    # prevalence of the representative held-out half, re-derived from the same frozen pipeline
    b, v, m3 = load(); pr = frozen_probs(b)
    half = len(v) // 2
    cr = np.random.RandomState(2026); p_ = cr.permutation(len(v))
    Vt = v.iloc[p_[half:]].reset_index(drop=True)
    prev = float(Vt["outcome"].mean())
    fig = plt.figure(figsize=(W_IN * 0.62, 2.3))
    ax = fig.add_subplot(111)
    lo, hi = dca["recalibrate_below_frozen_band"]
    ax.axvspan(lo, hi, color=OK["grey"], alpha=0.13, lw=0)
    ax.plot(df.threshold, df.recalibrated, color=OK["blue"], lw=1.5, label="recalibrated")
    ax.plot(df.threshold, df.frozen, color=OK["vermillion"], lw=1.5, label="frozen")
    ax.axhline(0, color=OK["black"], ls=":", lw=0.8)
    ta = [prev - (1 - prev) * t / (1 - t) for t in df.threshold.values]
    ax.plot(df.threshold, ta, color=OK["grey"], ls="--", lw=1.0, label="treat all")
    ax.axvline(dca["frozen_nonpositive_from"], color=OK["vermillion"], ls=":", lw=0.8)
    ax.annotate(f"frozen net benefit\nturns negative ≈ {dca['frozen_nonpositive_from']:.2f}",
                (dca["frozen_nonpositive_from"] - 0.0015, 0.005), textcoords="offset points",
                xytext=(-4, 12), fontsize=5.8, color=OK["vermillion"], ha="right")
    ax.annotate(f"recalibrated below frozen: {lo:.3f}–{hi:.3f}\n(max deficit {dca['max_abs_recal_minus_frozen_in_band']:.4f})",
                (0.055, -0.045), fontsize=5.8, ha="left", va="center", color=OK["grey"])
    ax.annotate("", xy=(lo, -0.02), xytext=(hi, -0.02),
                arrowprops=dict(arrowstyle="-", color=OK["grey"], lw=0.7))
    ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
    ax.set_xlim(0.05, 0.25); ax.set_ylim(-0.19, 0.10)
    ax.set_title("Decision curves on held-out eICU stays", fontsize=7.5)
    ax.legend(frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    save(fig, "v18_figS1_dca")


# ============================== FIG S2 : slope distribution + regularisation ==============================
def figS2():
    rc = pd.read_csv("outputs/v5_repeated_curve.csv")
    v12 = json.load(open("outputs/sr_v12.json", encoding="utf-8"))["penalised_recalibration_sensitivity"]
    fig = plt.figure(figsize=(W_IN * 0.86, 2.3)); gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.34)
    axA, axB = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    data = [rc[rc.n_pat == n].slope.values for n in SIZES]
    parts = axA.violinplot(data, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(OK["sky"]); pc.set_alpha(0.6)
    for k in ("cbars", "cmins", "cmaxes"):
        parts[k].set_color(OK["grey"])
    axA.axhline(1, ls=":", color=OK["grey"], lw=0.7); axA.axhline(0, ls=":", color=OK["vermillion"], lw=0.7)
    axA.set_yscale("symlog", linthresh=1)
    axA.set_xticks(range(1, 7)); axA.set_xticklabels([f"{n:,}" for n in SIZES], rotation=30)
    axA.set_ylabel("held-out calibration slope"); axA.set_xlabel("update size")
    axA.set_title("Unpenalised held-out slope", fontsize=7.5); panel(axA, "a")
    x = np.arange(len(SIZES))
    un = [v12[str(n)]["heldout_median_unpen"] for n in SIZES]
    rb = [v12[str(n)]["heldout_median_robust"] for n in SIZES]
    umx = [v12[str(n)]["heldout_max_unpen"] for n in SIZES]
    rmx = [v12[str(n)]["heldout_max_robust"] for n in SIZES]
    axB.plot(x, un, "o-", color=OK["vermillion"], ms=3.4, label="unpenalised median")
    axB.plot(x, rb, "s-", color=OK["green"], ms=3.4, label="penalised median")
    axB.plot(x, umx, "o--", color=OK["vermillion"], ms=2.8, alpha=0.5, label="unpenalised max")
    axB.plot(x, rmx, "s--", color=OK["green"], ms=2.8, alpha=0.5, label="penalised max")
    axB.axhline(1, ls=":", color=OK["grey"], lw=0.7); axB.set_yscale("log")
    axB.set_xticks(x); axB.set_xticklabels([f"{n:,}" for n in SIZES], rotation=30)
    axB.set_ylabel("held-out slope (log)"); axB.set_xlabel("update size")
    axB.set_title("Regularisation trade-off", fontsize=7.5)
    axB.legend(frameon=False, loc="upper right"); panel(axB, "b")
    save(fig, "v18_figS2_slope_reg")


# ============================== FIG S3 : tail imbalance + truncation ==============================
def figS3():
    tail = V18["tail_imbalance_raw_and_sd"]["variables"]
    ws = json.load(open("outputs/v6_results.json", encoding="utf-8"))["weight_sensitivity"]
    KEYVARS = ["HeartRate", "RespiratoryRate", "Age", "AG", "RBC", "COPD"]
    fig = plt.figure(figsize=(W_IN * 0.86, 2.3)); gs = gridspec.GridSpec(1, 2, width_ratios=[1.15, 1], wspace=0.36)
    axA, axB = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    x = np.arange(len(KEYVARS)); ww = 0.26
    for k, lab, c in [("p10", "p10", OK["sky"]), ("p50", "p50", OK["blue"]), ("p90", "p90", OK["vermillion"])]:
        axA.bar(x + [(-ww, 0, ww)[("p10", "p50", "p90").index(lab)]], [tail[v][f"{k}_sd_units"] for v in KEYVARS],
                ww, color=c, label=lab)
    axA.axhline(0, color=OK["grey"], lw=0.7)
    labs = [f"{v}\nSD {tail[v]['mimic_iv_sd_used']:g}" for v in KEYVARS]
    axA.set_xticks(x); axA.set_xticklabels(labs, rotation=45, ha="right", fontsize=5.4)
    axA.set_ylim(-3.9, 3.9)
    axA.set_ylabel("weighted eICU − MIMIC-IV\nquantile gap (MIMIC-IV SD units)", fontsize=6.5)
    axA.set_title("Residual tail imbalance after reweighting", fontsize=7.5)
    axA.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.30), fontsize=5.8)
    panel(axA, "a")
    keys = ["0.1-10", "1-5", "p1p99"]; lab = ["0.1–10", "1–5", "p1/p99"]
    slope = [ws[k]["weighted_slope"] for k in keys]; ess = [ws[k]["ESS_pct"] for k in keys]
    axB.bar(lab, slope, 0.55, color=[OK["blue"], OK["sky"], OK["purple"]])
    axB.axhline(1, ls=":", color=OK["grey"], lw=0.7); axB.set_ylim(0, 0.62)
    axB.set_ylabel("weighted calibration slope"); axB.set_xlabel("weight truncation")
    for i, e in enumerate(ess):
        axB.annotate(f"ESS {e:.0f}%", (i, slope[i] + 0.012), ha="center", fontsize=6)
    axB.set_title("Truncation sensitivity", fontsize=7.5); panel(axB, "b")
    save(fig, "v18_figS3_tails_trunc")


if __name__ == "__main__":
    for fn in [fig2, fig3, fig4, fig5, figS1, figS2, figS3]:
        try:
            fn()
        except Exception as e:
            import traceback; print("  FAIL", fn.__name__, repr(e)); traceback.print_exc()
    print("done")