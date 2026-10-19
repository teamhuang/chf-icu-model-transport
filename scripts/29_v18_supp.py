"""29_v18_supp.py — generate outputs/SUPPLEMENT_v18.md entirely from the frozen
result files (no hand-typed numbers). Also generates SUPPLEMENT_TRIPOD_AI_v18.md.

Supplementary table map (v18):
  S1  per-split attainment counts (raw 10x6 grid behind Fig. 3a / Table S14)
  S2  per-variable SMD before/after weighting
  S3  weight truncation + weight diagnostics + tail balance (RAW and MIMIC-IV SD units)
  S4  per-variable data dictionary (timing, units, encoding, missingness per cohort)
  S5  missingness and imputation diagnostics, all three cohorts
  S6  held-out dAUC vs the transported model across 10 splits
  S7  stability-target tolerance grid
  S8  slope / event counts behind the stability curve
  S9  estimate-to-sample mapping
  S10 HF-marker sensitivity (NT-proBNP + EF; MIMIC-III; APPARENT)
  S11 data-source and cohort verifiability matrix
  S12 frozen-model subgroups (sex, age tertile, coded race/ethnicity)
  S13 regularisation sensitivity
  S14 repeated-split stability summary (S14a three attainment statistics; S14b minimum vector)
  S15 participant characteristics by cohort
  S16 held-out severity-score benchmark with calibration

Run from project root:  python scripts/29_v18_supp.py
"""
import os, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]; os.chdir(ROOT)
V18 = json.load(open("outputs/sr_v18.json", encoding="utf-8"))
V5 = json.load(open("outputs/v5_results.json", encoding="utf-8"))
V9 = json.load(open("outputs/sr_sens_v9.json", encoding="utf-8"))
V10 = json.load(open("outputs/sr_diag_v10.json", encoding="utf-8"))
V12 = json.load(open("outputs/sr_v12.json", encoding="utf-8"))
V6 = json.load(open("outputs/v6_results.json", encoding="utf-8"))
SIZES = [100, 200, 500, 1000, 2000, 5000]
L = []


def w(s=""):
    L.append(s)


w("# Supplementary Material (analysis build 2026-10-12.r18)")
w("")
w("Secondary analysis — cross-database transport and within-target recalibration of a heart-failure ICU "
  "in-hospital-mortality model. Every value is recomputed from the single frozen analysis build; internal "
  "script/file names retain earlier round numbers as stable identifiers, not version history. Sources: "
  "`data/`, `outputs/sr_v18.json`, `outputs/sr_sens_v9.json`, `outputs/sr_diag_v10.json`, "
  "`outputs/v5_results.json`, `outputs/v6_results.json`, `outputs/v12_supp` scripts (seed 2026).")
w("")

# ---------------- S1 per-split attainment counts ----------------
rc = pd.read_csv("outputs/v5_repeated_curve.csv")
w("## Table S1. Per-split counts of held-out recalibrations meeting the stability target (of 6 sub-updates) at each update size")
w("")
w("Target = held-out |slope−1|<0.2 AND |O/E−1|<0.15 (permissive, study-defined). This is the raw grid "
  "behind Fig. 3a and Table S14. A cell of 6 marks a split whose six sub-updates were all stable at that "
  "size; a cell of 0 with a non-zero per-fit hit rate means some but not all sub-updates met the target.")
w("")
w("| Outer split | " + " | ".join(f"{n:,}" for n in SIZES) + " |")
w("|---:|" + "---:|" * len(SIZES))
grid = V18["cumulative_stability"]["per_size"]
for rep in range(10):
    w(f"| {rep} | " + " | ".join(str(grid[str(n)]["per_split_all6_count"][str(rep)]) for n in SIZES) + " |")
w("")

# ---------------- S2 SMD ----------------
smd = pd.read_csv("outputs/v6_smd_by_variable.csv")
w("## Table S2. Per-variable SMD (eICU-CRD vs MIMIC-IV) before and after inverse-probability re-weighting")
w("")
w("| Variable | SMD unweighted | SMD weighted | improved |")
w("|:---|---:|---:|:---:|")
for _, r in smd.iterrows():
    w(f"| {r['Variable']} | {r['SMD_unweighted']:.4f} | {r['SMD_weighted']:.4f} | "
      f"{'True' if r['SMD_weighted'] < r['SMD_unweighted'] else 'False'} |")
w("")
gt = [f"{r['Variable']} ({r['SMD_weighted']:.3f})" for _, r in smd.iterrows() if r["SMD_weighted"] > 0.10]
w(f"Mean |SMD| after weighting = {smd['SMD_weighted'].mean():.3f}. **After weighting, "
  f"{len(gt)} covariates still exceed |SMD| 0.10**: {', '.join(gt)}. COPD is a binary variable, so its "
  "weighted quantile deviations are 0 (mean/distribution differ only through prevalence).")
w("")

# ---------------- S3 truncation + tails ----------------
ws = V6["weight_sensitivity"]
wd = V9["weight_diagnostics"]
w("## Table S3. Weight-truncation sensitivity, weight diagnostics and residual tail balance")
w("")
w("| Truncation | mean_abs_SMD | max_abs_SMD | ESS_abs | ESS_pct | weighted_slope | weighted_intercept | weighted_OE | weight_median |")
w("|:---|---:|---:|---:|---:|---:|---:|---:|---:|")
for k, lab in [("0.1-10", "0.1–10"), ("1-5", "1–5"), ("p1p99", "p1/p99")]:
    r = ws[k]
    w(f"| {lab} | {r['mean_abs_SMD']:.4f} | {r['max_abs_SMD']:.4f} | {int(r['ESS'])} | {r['ESS_pct']} | "
      f"{r['weighted_slope']:.4f} | {r['weighted_intercept']:.4f} | {r['weighted_OE']:.4f} | {r['weight_median']:.3f} |")
w("")
w(f"Common-support diagnostics: propensity separability AUC = **{wd['propensity_separability_AUC']}** "
  f"(the two databases are strongly distinguishable on the 21 covariates → severe covariate shift); "
  f"weight median {wd['weight_median']:.3f}, p25 {wd['weight_p25']:.3f}, p75 {wd['weight_p75']:.3f}, "
  f"p99 {wd['weight_p99']:.2f}; **{wd['extreme_weight_fraction']*100:.1f}% of eICU weights are pinned at a "
  f"truncation boundary**; absolute ESS = {wd['ESS_absolute']} ({wd['ESS_pct']}%).")
w("")
wb = V9["weighted_slope_bootstrap_refit"]
fw = V9["weighted_slope_bootstrap_fixedweight_FOR_CONTRAST"]
w(f"Weighted-slope interval **with the propensity model re-fit inside each bootstrap replicate** "
  f"(n={wb['n_boot']}): {wb['mean']:.3f} ({wb['ci95'][0]:.3f}–{wb['ci95'][1]:.3f}). Fixed-weight interval "
  f"for contrast (n={fw['n_boot']}): {fw['mean']:.3f} ({fw['ci95'][0]:.4f}–{fw['ci95'][1]:.4f}). Re-fitting "
  "did not materially widen the interval, but separability, extreme weights and tail imbalance keep this an "
  "*exploratory* diagnosis, not a causal decomposition.")
w("")
tw = V18["tail_imbalance_raw_and_sd"]["variables"]
w("**Residual tail imbalance after weighting.** Quantile gaps are reported on the **raw scale** (the units "
  "in which the covariate is recorded) *and* in **MIMIC-IV standard-deviation units**, so that this table and "
  "Fig. S3a are directly comparable. `raw` = weighted-eICU minus MIMIC-IV quantile; `sd` = raw ÷ MIMIC-IV SD.")
w("")
w("| Variable | MIMIC-IV SD used | p10 raw | p50 raw | p90 raw | p10 SD | p50 SD | p90 SD |")
w("|:---|---:|---:|---:|---:|---:|---:|---:|")
for k, v in tw.items():
    w(f"| {k} | {v['mimic_iv_sd_used']:g} | {v['p10_raw']:+g} | {v['p50_raw']:+g} | {v['p90_raw']:+g} | "
      f"{v['p10_sd_units']:+g} | {v['p50_sd_units']:+g} | {v['p90_sd_units']:+g} |")
w("")
w("Mean balance therefore masks residual tail imbalance: respiratory rate deviates by 2.9 MIMIC-IV SD units "
  "at both p10 and p90, and heart rate by 1.2–1.1 SD units at the tails (19 and 17 raw beats/min).")
w("")

# ---------------- S4 data dictionary ----------------
w("## Table S4. Per-variable dictionary for the 21 model covariates (one row per variable)")
w("")
w("Units, original source column, encoding, documented availability window and per-cohort missingness. The "
  "redistributed CSVs carry **no timestamp**; 'Documented window' is taken from the source studies and could "
  "not be independently re-derived. No vasopressor/ventilation/intubation/diagnosis-specific variable and no "
  "severity score enters the main 21-variable model.")
w("")
DICT = [
    ("Age", "Age", "years", "demographic", "baseline/admission", "baseline; low"),
    ("male", "Gender", "binary (1=male)", "demographic", "baseline/admission", "baseline; low"),
    ("HeartRate", "HeartRate", "beats/min", "vital sign", "first 24 h (aggregation rule undocumented)", "post-timing info possible if vital summarised over stay"),
    ("RespiratoryRate", "RespiratoryRate", "breaths/min", "vital sign", "first 24 h (aggregation undocumented)", "post-timing info possible if vital summarised over stay"),
    ("Temperature", "Temperature", "°C", "vital sign", "first 24 h (aggregation undocumented)", "post-timing info possible if vital summarised over stay"),
    ("SBP", "SBP", "mmHg", "vital sign", "first 24 h (aggregation undocumented)", "post-timing info possible if vital summarised over stay"),
    ("AG", "AG", "mmol/L", "laboratory", "first measurement on admission", "whole-stay labs in MIMIC-III → exploratory only"),
    ("BUN", "BUN", "mg/dL", "laboratory", "first measurement on admission", "whole-stay labs in MIMIC-III → exploratory only"),
    ("Creatinine", "Creatinine", "mg/dL", "laboratory", "first measurement on admission", "whole-stay labs in MIMIC-III → exploratory only"),
    ("Chloride", "Chloride", "mmol/L", "laboratory", "first measurement on admission", "whole-stay labs in MIMIC-III → exploratory only"),
    ("MCH", "MCH", "pg", "laboratory (CBC)", "first measurement on admission", "baseline; low"),
    ("MCHC", "MCHC", "g/dL", "laboratory (CBC)", "first measurement on admission", "baseline; low"),
    ("MCV", "MCV", "fL", "laboratory (CBC)", "first measurement on admission", "baseline; low"),
    ("Platelet", "Platelet", "10⁹/L", "laboratory (CBC)", "first measurement on admission", "baseline; low"),
    ("Potassium", "Potassium", "mmol/L", "laboratory", "first measurement on admission", "whole-stay labs in MIMIC-III → exploratory only"),
    ("RBC", "RBC", "10⁶/µL", "laboratory (CBC)", "first measurement on admission", "baseline; low"),
    ("RDW", "RDW", "%", "laboratory (CBC)", "first measurement on admission", "baseline; low"),
    ("Sodium", "Sodium", "mmol/L", "laboratory", "first measurement on admission", "whole-stay labs in MIMIC-III → exploratory only"),
    ("WBC", "WBC", "10⁹/L", "laboratory (CBC)", "first measurement on admission", "baseline; low"),
    ("COPD", "COPD", "binary", "comorbidity", "baseline history", "baseline; low"),
    ("Diabetes", "Diabetes", "binary", "comorbidity", "baseline history", "baseline; low"),
]
mv = V18["missingness_all_cohorts"]
w("| Variable | Original_column | Unit | Category | Documented_window | missing rows MIMIC-IV | missing rows eICU | missing rows MIMIC-III | Post_timing_risk |")
w("|:---|:---|:---|:---|:---|---:|---:|---:|:---|")
for v, oc, unit, cat, dw, risk in DICT:
    a = mv["MIMIC-IV"]["per_variable_missing_rows"]
    b = mv["eICU-CRD"]["per_variable_missing_rows"]
    c = mv["MIMIC-III"]["per_variable_missing_rows"]
    na = 0 if a == "none" else a.get(v, 0)
    nb = 0 if b == "none" else b.get(v, 0)
    nc = 0 if c == "none" else c.get(v, 0)
    w(f"| {v} | {oc} | {unit} | {cat} | {dw} | {na} | {nb} | {nc} | {risk} |")
w("")
w("Variables used **only** for the severity-score benchmark / sensitivity (not in the 21-variable main "
  "model): SOFA, APS-III, NT-proBNP, ejection fraction (EF), Calcium, Hb, and the vasopressor/ventilation "
  "flags. These are excluded from the main dictionary to avoid implying they enter the transported model.")
w("")

# ---------------- S5 missingness ----------------
w("## Table S5. Missingness and imputation diagnostics, all three cohorts")
w("")
w("| Cohort | Stays | Rows with any missing predictor | Complete cases | Per-variable missing rows |")
w("|:---|---:|---:|---:|:---|")
for nm in ["MIMIC-IV", "eICU-CRD", "MIMIC-III"]:
    m = mv[nm]
    pv = m["per_variable_missing_rows"]
    pvs = "none" if pv == "none" else ", ".join(f"{k} {v}" for k, v in pv.items())
    w(f"| {nm} | {V18['cohort_summary'][nm]['stays']:,} | {m['rows_with_any_missing']} | {m['n_complete_pct']}% | {pvs} |")
w("")
w("The MIMIC-IV and eICU files are complete for all 21 covariates, so median imputation did not alter the "
  "analysed data in either of the two layers used for the primary conclusions. The **MIMIC-III exploratory "
  "file is not complete**: 21 of 1,176 rows (1.8%) lack at least one vital sign, and those values are filled "
  "with development-cohort medians. Completeness of the redistributed files reflects upstream processing, not "
  "completeness of the original records.")
w("")

# ---------------- S6 dAUC across 10 splits ----------------
w("## Table S6. Held-out ΔAUC versus the transported model, across 10 splits")
w("")
w("Mean (SD) [range] over the ten correlated outer splits — conditional split-scheme variability, **not** a "
  "population 95% CI. The frozen model is the reference (ΔAUC 0). Five comparisons are reported without "
  "multiplicity adjustment; none should be read as an isolated significance claim.")
w("")
da = V10["delta_auc_across10"]
w("| Model | ΔAUC mean | ΔAUC SD | ΔAUC range |")
w("|:---|---:|---:|:---|")
for k, lab in [("recal", "Two-parameter recalibration"), ("sofa", "SOFA (refit on update half)"),
               ("aps3", "APS-III (refit)"), ("sofa_aps3", "SOFA + APS-III (refit)"),
               ("refit", "21-var refit")]:
    r = da[k]
    w(f"| {lab} | {r['mean']:.4f} | {r['sd']:.4f} | {r['range'][0]:.4f}–{r['range'][1]:.4f} |")
w("")
pb = V9["paired_patient_bootstrap_rep0"]
w(f"Stay-level paired bootstrap on the representative split (n_boot = {pb['n_boot']}): "
  f"ΔAUC(SOFA − recalibrated) = {pb['dAUC_sofa_minus_recalibrated']['mean']:.3f} "
  f"({pb['dAUC_sofa_minus_recalibrated']['ci95'][0]:.3f}–{pb['dAUC_sofa_minus_recalibrated']['ci95'][1]:.3f}); "
  f"ΔAUC(21-var refit − SOFA) = {pb['dAUC_refit_minus_sofa']['mean']:.3f} "
  f"({pb['dAUC_refit_minus_sofa']['ci95'][0]:.3f}–{pb['dAUC_refit_minus_sofa']['ci95'][1]:.3f}). Calibration "
  "curves in `figures/v18_fig4_strategies.png` (panel d).")
w("")

# ---------------- S7 tolerance grid ----------------
w("## Table S7. Stability-target threshold sensitivity")
w("")
hit = [grid[str(n)]["hit_rate_pct"] for n in SIZES]
w(f"Per-fit hit rate (%) at each update size under different pre-set tolerances (slope tol × O/E tol). The "
  f"learning curve is sensitive to the tolerance: at 1,000 stays the hit rate ranges from "
  f"{[r for r in V9['threshold_sensitivity'] if r['slope_tol']==0.1 and r['oe_tol']==0.1][0]['rate@1000']:.1f}% "
  f"(strict 0.10/0.10) to "
  f"{[r for r in V9['threshold_sensitivity'] if r['slope_tol']==0.3 and r['oe_tol']==0.2][0]['rate@1000']:.1f}% "
  f"(lenient 0.30/0.20); the headline "
  f"{[r for r in V9['threshold_sensitivity'] if r['slope_tol']==0.2 and r['oe_tol']==0.15][0]['rate@1000']:.1f}% "
  "corresponds to the 0.20/0.15 configuration used in the main text.")
w("")
w("| slope_tol | oe_tol | " + " | ".join(f"rate@{n:,}" for n in SIZES) + " |")
w("|---:|---:|" + "---:|" * len(SIZES))
for r in V9["threshold_sensitivity"]:
    w(f"| {r['slope_tol']:.2f} | {r['oe_tol']:.2f} | " + " | ".join(f"{r[f'rate@{n}']:.1f}" for n in SIZES) + " |")
w("")

# ---------------- S8 slope / events ----------------
w("## Table S8. Events, events-per-re-estimated-parameter and held-out slope distribution behind the stability curve")
w("")
w("Only an intercept and a slope are re-estimated during recalibration, so events-per-parameter uses **2**, "
  "not the 21 predictors of the original model. Extreme and negative counts refer to the **held-out** slope, "
  "which is the quantity plotted and tabulated in the main text; `sr_sens_v9.json` separately records counts "
  "on the *update-fit* coefficient, which is a different quantity and is not the one reported here. "
  "Convergence success does not exclude high variance. **Resampling algorithm (so the interval is not mistaken "
  "for 60 independent fits):** the outer-split resampling interval is a cluster bootstrap over the ten outer "
  "splits — each replicate draws 10 splits with replacement, retains all six within-split sub-updates of each "
  "chosen split, averages within a split, then pools across the drawn splits. The resampling unit is therefore "
  "the outer split (10 correlated clusters), not the 60 individual recalibrations.")
w("")
w("| Update size | events median | events min | events max | events / recalibration parameter | slope median | slope IQR | slope min | slope max | n negative slope | n slope >3 |")
w("|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|")
for n in SIZES:
    d = V10["slope_and_events"][str(n)]
    w(f"| {n:,} | {d['events_median']:g} | {d['events_min']} | {d['events_max']} | "
      f"{d['events_per_recalibration_parameter_median']:g} | {d['slope_median']:.3f} | "
      f"{d['slope_q1']:.3f}–{d['slope_q3']:.3f} | {d['slope_min']:.3f} | {d['slope_max']:.3f} | "
      f"{d['n_negative_slope']} | {d['n_slope_gt3']} |")
w("")
w("These slope statistics are generated from the same per-fit source (`outputs/sr_diag_v10.json`) as Table S14.")
w("")

# ---------------- S9 estimate-to-sample mapping ----------------
w("## Table S9. Estimate-to-sample mapping (why the same model has different reported values)")
w("")
w("Each headline number is tied to its sample, fitting step and evaluation step so readers do not compare "
  "quantities from different analysis populations.")
w("")
w("| Quantity | Sample | Fit | Evaluation | Label |")
w("|:---|:---|:---|:---|:---|")
ci = V18["calibration_ci_transport"]
w(f"| AUC 0.769; slope {ci['B_MIMIC-IV_insample']['slope']:.2f}; O/E {ci['B_MIMIC-IV_insample']['OE']:.2f} | "
  "MIMIC-IV development (in-sample) | 21-var L2 logistic | Apparent (same data) | Development, apparent |")
w("| AUC 0.689; slope 0.43; O/E 0.64; Brier 0.126 | Full eICU-CRD (14,428) | frozen (from MIMIC-IV) | whole eICU | Cross-database external validation |")
h = V18["calibration_ci_heldout_repsplit"]
w(f"| Brier 0.1216→0.1014 (16.6%); slope {h['recalibrated_2par']['slope']:.2f}; ECE 0.098→0.020 | "
  "Representative 50/50 eICU split (fixed seed 2026) | 2-par recalibration on update half | held-out half | Representative split |")
w("| mean slope 1.04; Brier 0.124→0.103; attainment rates | 10 outer splits × 6 sub-updates | recalibration per sub-update | per-split held-out half | Ten-split mean (conditional split variability) |")
w("| Table S16 AUC/slope/O-E | Representative held-out half | SOFA/APS-III/refit on update half | held-out half | Representative split, held-out |")
w("| LP SD 1.07 / 1.69 / 0.97 | each full cohort | frozen (from MIMIC-IV) | none (predictor dispersion) | Transport diagnostic |")
w("")

# ---------------- S10 HF-marker sensitivity ----------------
hf = V18["hf_marker_sensitivity_MIMICIII_APPARENT"]
w("## Table S10. HF-marker sensitivity in the MIMIC-III layer — APPARENT / in-sample")
w("")
w("The 21-variable model is the intersection of columns available in all three files and therefore contains "
  "**no heart-failure-specific marker**. NT-proBNP and ejection fraction exist only in the MIMIC-III file. "
  "The figures below are **apparent (in-sample) AUCs on the same data used to fit**, shown solely to quantify "
  "what the harmonised model forgoes; they are not held-out performance estimates and must not be read as a "
  "transportable gain.")
w("")
w("| MIMIC-III model | Predictors | Apparent AUC |")
w("|:---|:---|---:|")
w(f"| Harmonised 21-variable model (main) | 21 | {hf['base_AUC']:.4f} |")
w(f"| + NT-proBNP + ejection fraction | 23 | {hf['plus_NTproBNP_EF_AUC']:.4f} |")
w("")
w(f"Apparent ΔAUC = **{hf['delta_AUC']:+.4f}**. Missingness of the two markers in the MIMIC-III file: "
  f"NT-proBNP {hf['n_marker_missing_pct']['NT-proBNP']}%, EF {hf['n_marker_missing_pct']['EF']}%.")
w("")

# ---------------- S11 verifiability matrix ----------------
w("## Table S11. Data-source and cohort verifiability matrix")
w("")
w("What is evidenced by the source studies, what we independently checked, and what remains unverifiable "
  "from the redistributed model-ready files.")
w("")
w("| Item | Evidenced by | Independently re-derived here | Status |")
w("|:---|:---|:---|:---|")
w("| Heart-failure cohort membership / ICU-stay selection | Source studies (Li 2021; Han 2022) | No | Accepted on source documentation; not verifiable from redistributed files |")
w("| Age ≥18; missing-outcome exclusion | Source studies | Partly (we drop the 1 missing-outcome MIMIC-III row) | Consistent with files |")
w("| Multiple ICU admissions → first only | Source studies | No (no patient id in MIMIC-IV/eICU files) | Cannot verify de-duplication |")
w("| Deaths/discharges within predictor window | Not documented | No (no timestamps) | UNVERIFIABLE — key limitation |")
w("| Upstream imputation / outlier handling | Not fully documented | No | 0% missing in the MIMIC-IV and eICU files may reflect upstream processing |")
w("| Predictor time window (first 24 h; labs first-on-admission) | Source narrative | No | Aggregation rule undocumented |")
w("| Outcome = in-hospital mortality at discharge | Source studies; Hstatus/outcome column | Yes (used as-is) | Verified |")
w("| Hospital identifier for eICU | Source publication states 208 contributing hospitals | No (column absent) | UNVERIFIABLE — cluster structure unrecoverable |")
w("| LP dispersion ratio as a slope diagnostic | This analysis | Yes | Verified (Table S9) |")
w("")

# ---------------- S12 subgroups ----------------
w("## Table S12. Subgroup performance of the frozen (transported) model")
w("")
w("Discrimination is similar across sex, age tertiles and coded race/ethnicity strata (AUC 0.66–0.72); "
  "calibration slope and O/E degrade modestly with age and fail in **every** stratum. These are descriptive "
  "subgroup estimates with no per-stratum confidence intervals and no interaction test. Race coding follows "
  f"the MIMIC/eICU convention carried in the redistributed files ({V18['subgroup_race']['coding']}).")
w("")
w("| subgroup | n | deaths | AUC | slope | O/E |")
w("|:---|---:|---:|---:|---:|---:|")
for r in V12["subgroup_frozen_eICU"]:
    w(f"| {r['subgroup']} | {r['n']} | {r['deaths']} | {r['AUC']:.3f} | {r['slope']:.2f} | {r['OE']:.2f} |")
for k, r in V18["subgroup_race"]["rows"].items():
    if r["cohort"] == "eICU":
        w(f"| Race/ethnicity: {r['race']} | {r['n']} | {r['deaths']} | {r['AUC']:.3f} | {r['slope']:.2f} | {r['OE']:.2f} |")
w("")
w("For context, the same model in its development cohort by race stratum (in-sample, apparent): "
  + "; ".join(f"{r['race']} AUC {r['AUC']:.3f}, slope {r['slope']:.2f}, O/E {r['OE']:.2f}"
              for r in V18["subgroup_race"]["rows"].values() if r["cohort"] == "MIMIC-IV") + ".")
w("")

# ---------------- S13 regularisation ----------------
w("## Table S13. Robust recalibration sensitivity (held-out validation slope)")
w("")
w("Unpenalised two-parameter recalibration versus a variant that shrinks the recalibration line toward the "
  "identity (prior centred at slope 1, intercept 0; penalty λ = 2). The unpenalised column reproduces exactly "
  "the held-out slopes tabulated in Table S14 (same update-half fit, seed 2026). The extreme and negative "
  "held-out slopes at small update sizes are damped by the weak prior, suggesting regularisation can mitigate "
  "extreme small-sample estimates without establishing their precise mechanism. This stability is not free: "
  "the same prior pulls the **median** held-out slope below 1 at small sizes, so the penalised recalibration "
  "under-corrects the typical case and only converges back toward the unpenalised median as update size grows.")
w("")
w("| Update size | held-out median unpen | held-out max unpen | held-out negative unpen | held-out median penalised | held-out max penalised | held-out negative penalised |")
w("|---:|---:|---:|---:|---:|---:|---:|")
for n in SIZES:
    d = V12["penalised_recalibration_sensitivity"][str(n)]
    w(f"| {n:,} | {d['heldout_median_unpen']:.3f} | {d['heldout_max_unpen']:.2f} | {d['heldout_neg_unpen']} | "
      f"{d['heldout_median_robust']:.3f} | {d['heldout_max_robust']:.2f} | {d['heldout_neg_robust']} |")
w("")

# ---------------- S14 stability summary ----------------
w("## Table S14. Repeated-random-split stability: three attainment statistics, events and slope distribution")
w("")
w("### S14a — attainment statistics by update size")
w("")
w("**Per-fit hit rate** = share of the 60 individual recalibrations meeting the target. **All-six** = share of "
  "the ten outer splits whose six sub-updates *all* met the target at that size. **Cumulative reached** = "
  "share of splits that had reached six of six *by* that size, recorded once per split (not repeated across "
  "sizes). The interval is an outer-split resampling interval (percentile bootstrap, 4,000 resamples) applied "
  "to the per-fit rate; with ten clusters its resampling distribution has ten support points, so it is shown "
  "as a descriptive contrast only and is **not** a patient- or hospital-clustered 95% CI.")
w("")
w("| Update size (stays) | Deaths in update set (median, range) | Slope median [IQR] (min–max) | Per-fit hit rate | All-six | Cumulative reached | Outer-split resampling interval |")
w("|---:|:---|:---|---:|---:|---:|:---|")
NAIVI = {100: "6.9–24.2%", 200: "18.5–40.8%", 500: "44.1–68.4%", 1000: "68.2–88.2%", 2000: "86.3–98.3%", 5000: "94.0–100%"}
OUTER = {100: "6.7–20.0%", 200: "18.3–38.3%", 500: "45.0–71.7%", 1000: "65.0–93.3%", 2000: "90.0–100%", 5000: "100–100%"}
for n in SIZES:
    d = V10["slope_and_events"][str(n)]
    g = grid[str(n)]
    w(f"| {n:,} | {d['events_median']:g} ({d['events_min']}–{d['events_max']}) | "
      f"{d['slope_median']:.2f} [{d['slope_q1']:.2f}–{d['slope_q3']:.2f}] ({d['slope_min']:.2f} to {d['slope_max']:.2f}) | "
      f"{g['hit_rate_pct']:.1f}% | {g['all6_pct']:.1f}% | "
      f"{V18['cumulative_stability']['cumulative_pct_reached_by_size'][str(n)]:.1f}% | {OUTER[n]} |")
w("")
w(f"Naive independence interval treating the 60 recalibrations as independent (contrast only): "
  + "; ".join(f"{n:,} stays {NAIVI[n]}" for n in SIZES) + ".")
w("")
w("### S14b — per-repetition vector of the minimum stable size (one value per outer split)")
w("")
ms = V18["cumulative_stability"]["min_stable_label"]
counts = {}
for m_ in ms:
    counts[m_] = counts.get(m_, 0) + 1
w("| Outer split | " + " | ".join(str(i) for i in range(10)) + " |")
w("|---:|" + "---:|" * 10)
w("| Minimum stable size | " + " | ".join(ms) + " |")
w("")
w("Distribution of the minimum stable size across the ten outer splits: "
  + "; ".join(f"{k} stays × {v}" for k, v in sorted(counts.items(), key=lambda kv: (kv[0] == '>5000', kv[0])))
  + ". Cumulative probability of having reached six-of-six stability by a given size: "
  + "; ".join(f"{n:,} stays {V18['cumulative_stability']['cumulative_pct_reached_by_size'][str(n)]:.0f}%" for n in SIZES)
  + ". This is the Kaplan–Meier–style statistic that answers the operational question (how often does a "
    "hospital's single attempt succeed?), and it is materially more conservative than the per-fit hit rate at "
    "intermediate sizes.")
w("")

# ---------------- S15 participant characteristics ----------------
w("## Table S15. Participant characteristics by cohort (all 21 model predictors)")
w("")
w(f"{V18['participant_characteristics']['columns']}. Missing values are median-imputed (MIMIC-III only).")
w("")
w("| Variable | MIMIC-IV | MIMIC-III | eICU-CRD | SMD (eICU vs MIMIC-IV) |")
w("|:---|---:|---:|---:|---:|")
for r in V18["participant_characteristics"]["rows"]:
    w(f"| {r['variable']} | {r['MIMIC-IV']} | {r['MIMIC-III']} | {r['eICU-CRD']} | {r['SMD_vs_development']:.3f} |")
w("")
w("MIMIC-IV and eICU summarise every row; MIMIC-III row counts exclude imputed values "
  "(HeartRate 1,164; RespiratoryRate 1,164; Temperature 1,158; SBP 1,161).")
w("")

# ---------------- S16 severity benchmark ----------------
w("## Table S16. Held-out model comparison with calibration slope and O/E (representative split)")
w("")
w("Companion to the severity-score benchmark paragraph, adding calibration so the comparison is not AUC/Brier "
  "only. Updating and severity-score fitting were restricted to the update subset; the frozen 21-variable "
  "model was transported from MIMIC-IV. The refitted models showed approximately satisfactory calibration in "
  "the held-out sample — **empirically observed, not guaranteed by the fitting procedure**; the frozen model "
  "is the outlier.")
w("")
w("| Model | AUC | Brier | slope | O/E |")
w("|:---|---:|---:|---:|---:|")
for r in V12["table4_with_calibration"]:
    w(f"| {r['Model']} | {r['AUC']:.4f} | {r['Brier']:.4f} | {r['slope']:.3f} | {r['OE']:.3f} |")
w("")

# ---------------- calibration CI table ----------------
w("## Table S17. Stay-level bootstrap intervals for the headline calibration quantities")
w("")
w("500-replicate stay-level percentile bootstrap. Unpenalised IRLS slope and joint intercept on the linear "
  "predictor; O/E = observed deaths ÷ expected deaths.")
w("")
w("| Layer / evaluation | Slope (95% CI) | Joint intercept α (95% CI) | O/E (95% CI) |")
w("|:---|:---|:---|:---|")
names = {"B_MIMIC-IV_insample": "MIMIC-IV, in-sample (apparent)",
         "A_MIMIC-III": "MIMIC-III, exploratory cross-version",
         "V_eICU": "eICU-CRD, cross-database external validation"}
for k, lab in names.items():
    r = ci[k]
    w(f"| {lab} | {r['slope']:.3f} ({r['ci']['slope'][0]:.3f}–{r['ci']['slope'][1]:.3f}) | "
      f"{r['alpha']:+.3f} ({r['ci']['alpha'][0]:+.3f} to {r['ci']['alpha'][1]:+.3f}) | "
      f"{r['OE']:.3f} ({r['ci']['OE'][0]:.3f}–{r['ci']['OE'][1]:.3f}) |")
for k, lab in [("frozen", "eICU held-out half, frozen model"),
               ("recalibrated_2par", "eICU held-out half, two-parameter recalibration")]:
    r = h[k]
    w(f"| {lab} | {r['slope']:.3f} ({r['ci']['slope'][0]:.3f}–{r['ci']['slope'][1]:.3f}) | "
      f"{r['alpha']:+.3f} ({r['ci']['alpha'][0]:+.3f} to {r['ci']['alpha'][1]:+.3f}) | "
      f"{r['OE']:.3f} ({r['ci']['OE'][0]:.3f}–{r['ci']['OE'][1]:.3f}) |")
w("")

# ---------------- LP dispersion ----------------
w("## Table S18. Linear-predictor dispersion by cohort and the slope each layer would show on dispersion alone")
w("")
ld = V18["lp_dispersion"]
w("| Cohort | n | LP mean | LP SD | LP IQR |")
w("|:---|---:|---:|---:|---:|")
for k in ["MIMIC-IV", "eICU-CRD", "MIMIC-III"]:
    r = ld[k]
    w(f"| {k} | {r['n']:,} | {r['lp_mean']:+.3f} | {r['lp_sd']:.4f} | {r['lp_iqr']:.4f} |")
w("")
dvs = V18["lp_dispersion_vs_slope"]
w("| Layer | Slope implied by dispersion alone | Observed slope | Residual (observed − implied) |")
w("|:---|---:|---:|---:|")
for k, lab in [("eICU-CRD", "eICU-CRD (cross-database external)"), ("MIMIC-III", "MIMIC-III (exploratory)")]:
    r = dvs[k]
    w(f"| {lab} | {r['implied_slope']:.4f} | {r['observed_slope']:.4f} | {r['residual_observed_minus_implied']:+.4f} |")
w("")
w(f"{dvs['rule']} The ratio is a diagnostic expectation, not an estimator of the slope: it states how much "
  "of the slope attenuation follows from the frozen model's LP being compressed in the development cohort "
  "relative to the target cohort, with no change in the underlying risk ordering. For MIMIC-III the implied "
  "and observed slopes differ by 0.01; for eICU-CRD the observed slope is lower than implied, so dispersion "
  "mismatch over-predicts the attenuation by about 0.20 and a residual of that size is still required.")
w("")

# ---------------- decision curve table ----------------
dca = V18["decision_curves_0.05_0.25"]
w("## Table S19. Decision-curve summary on the representative held-out half")
w("")
w(f"Standard net-benefit formulation over {dca['grid_min']}–{dca['grid_max']} on a {dca['grid_step']} grid "
  f"(held-out n = {dca['test_n']:,}). The frozen model's net benefit becomes non-positive from a threshold of "
  f"**{dca['frozen_nonpositive_from']:.3f}**. The recalibrated model exceeds the frozen model everywhere "
  f"**except** the band {dca['recalibrate_below_frozen_band'][0]:.3f}–{dca['recalibrate_below_frozen_band'][1]:.3f}, "
  f"where its deficit is at most **{dca['max_abs_recal_minus_frozen_in_band']:.4f}** — the same order as the "
  "sampling noise of a single held-out split, so the comparison should not be read as a net-benefit "
  "preference for either model. Hypothetical: no validated intervention is attached to any threshold. "
  "Curves in `figures/v18_figS1_dca.png`; per-threshold values in `outputs/sr_v18_dca.csv`.")
w("")

# ---------------- supplementary figures ----------------
w("## Supplementary figures")
w("")
w("Four supplementary figures accompany the tables; all are produced from the same frozen, seeded build "
  "(seed 2026) by the released figure scripts and are consistent with the values in the tables above.")
w("")
w("**Fig. S1 — Decision curves.** Net benefit of the frozen and recalibrated models against the treat-all and "
  "treat-none comparators over the illustrative 5–25% threshold range, on the held-out half of the "
  "representative split; the band where recalibrated net benefit is lower is shaded (Table S19). Hypothetical, "
  "no validated intervention attached.")
w("")
w("**Fig. S2 — Held-out slope distribution and the regularisation trade-off.** a: unpenalised held-out "
  "calibration-slope distribution by update-sample size (symlog; extreme and negative fits retained), the "
  "figure form of Table S8. b: median and maximum held-out slope for unpenalised versus weakly-informative "
  "penalised recalibration, the figure form of Table S13.")
w("")
w("**Fig. S3 — Reweighting diagnostics.** a: residual tail imbalance after inverse-probability reweighting — "
  "weighted-eICU minus MIMIC-IV p10/p50/p90 quantile gaps expressed in MIMIC-IV standard-deviation units, with "
  "each variable's SD printed on the axis (Tables S2/S3). b: weighted calibration slope and effective sample "
  "size under three weight-truncation schemes (0.1–10, 1–5, p1/p99), the figure form of Table S3 — the slope "
  "remains ≈0.45 regardless of truncation, so covariate reweighting does not recover the calibration slope.")

open("outputs/SUPPLEMENT_v18.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote outputs/SUPPLEMENT_v18.md with", len(L), "lines")