"""16_v10_supp.py — regenerate SUPPLEMENT_v10.md from v9 supplement, replacing S7 and S8 with
round-10 diagnostics (events/parameter, slope distribution; ΔAUC mean(SD)[range])."""
import os, json, re
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; os.chdir(ROOT)
diag=json.load(open("outputs/sr_diag_v10.json",encoding="utf-8"))
sup=open("outputs/SUPPLEMENT_v9.md",encoding="utf-8").read()

SIZES=["100","200","500","1000","2000","5000"]
sd=diag["slope_and_events"]
def r2(x): return round(float(x),2)
rows=[]
for n in SIZES:
    d=sd[n]; rows.append(dict(Update_size=int(n),events_median=d["events_median"],events_min=d["events_min"],events_max=d["events_max"],
        events_per_recalibration_param=d["events_per_recalibration_parameter_median"],
        slope_median=r2(d["slope_median"]),slope_IQR=f"{r2(d['slope_q1'])}–{r2(d['slope_q3'])}",slope_min=r2(d["slope_min"]),slope_max=r2(d["slope_max"]),
        n_negative_slope=d["n_negative_slope"],n_slope_gt3=d["n_slope_gt3"]))
s7=pd.DataFrame(rows)
# emit ready-to-paste Table 2 slope/events cells (same source, 2dp)
t2=[]
for n in SIZES:
    d=sd[n]
    t2.append(f"| {int(n):,} | {d['events_median']:g} ({d['events_min']}–{d['events_max']}) | {r2(d['slope_median']):.2f} [{r2(d['slope_q1']):.2f}–{r2(d['slope_q3']):.2f}] ({r2(d['slope_min']):.2f} to {r2(d['slope_max']):.2f}) |")
print("TABLE2_CELLS:\n"+"\n".join(t2))
S7=("## Table S7. Events, events-per-re-estimated-parameter and held-out slope distribution behind the stability curve\n\n"
    "Only an intercept and a slope are re-estimated during recalibration, so events-per-parameter uses **2**, not the 21 predictors of the original model. The median slope is near 1 even at small sizes; a few extreme/negative fits inflate the mean and SD, potentially reflecting sparse-data instability or near-separation (a dedicated separation diagnostic was not run). Convergence success does not exclude high variance. **These slope statistics are generated from the same per-fit source (`outputs/sr_diag_v10.json`) as main-text Table 2.**\n\n"
    + s7.to_markdown(index=False)+"\n")

da=diag["delta_auc_across10"]
rows=[]
for k,lab in [("recal","Two-parameter recalibration"),("sofa","SOFA (refit on update half)"),("aps3","APS-III (refit)"),("sofa_aps3","SOFA + APS-III (refit)"),("refit","21-var refit")]:
    d=da[k]; rows.append(dict(Model=lab,deltaAUC_mean=d["mean"],deltaAUC_SD=d["sd"],deltaAUC_range=f"{d['range'][0]}–{d['range'][1]}"))
s8=pd.DataFrame(rows)
pb=json.load(open("outputs/sr_sens_v9.json",encoding="utf-8"))["paired_patient_bootstrap_rep0"]
S8=("## Table S8. Held-out ΔAUC versus the transported model, across 10 splits\n\n"
    "Reported as mean (SD) [range] over the ten correlated outer splits — conditional split-scheme variability, **not** a population 95% CI. The frozen model is the reference (ΔAUC 0).\n\n"
    + s8.to_markdown(index=False)+"\n\n"
    + "Stay-level paired bootstrap on the representative split: ΔAUC(SOFA − recalibrated) = %.3f (%.3f–%.3f); ΔAUC(21-var refit − SOFA) = %.3f (%.3f–%.3f). Calibration curves in `figures/sr_fig_calib_curves.png`.\n"%(
        pb['dAUC_sofa_minus_recalibrated']['mean'],pb['dAUC_sofa_minus_recalibrated']['ci95'][0],pb['dAUC_sofa_minus_recalibrated']['ci95'][1],
        pb['dAUC_refit_minus_sofa']['mean'],pb['dAUC_refit_minus_sofa']['ci95'][0],pb['dAUC_refit_minus_sofa']['ci95'][1]))

# replace S7 and S8 blocks (each runs from its header to the next '## Table' or end)
def replace_block(text, header, newblock):
    i=text.find(header)
    if i<0: return text+newblock
    # find next '## Table' after i+len(header)
    j=text.find("\n## Table", i+len(header))
    if j<0: j=len(text)
    return text[:i]+newblock+("\n" if j<len(text) else "")+text[j+1:]
sup=sup.replace("# Supplementary Material (v9)","# Supplementary Material (v10)")
sup=replace_block(sup,"## Table S7.",S7)
sup=replace_block(sup,"## Table S8.",S8)
open("outputs/SUPPLEMENT_v10.md","w",encoding="utf-8").write(sup)
print("wrote outputs/SUPPLEMENT_v10.md")
print("S7:\n",s7.to_string(index=False))
print("S8:\n",s8.to_string(index=False))
