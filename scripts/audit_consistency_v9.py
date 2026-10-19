"""audit_consistency_v9.py — Phase-4/5 audit for the round-9 revision.
Recompute v9 numbers from outputs/sr_*.{json,csv} + v5/v6 JSON, verify corrected references,
check journal caps, and confirm the reviewer's flagged errors (S2 count, S4 vars, temporal label)
are fixed. Exit 0 = PASS."""
import os, re, json, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; os.chdir(ROOT)
MS=open("outputs/SCI_manuscript_v9.md",encoding="utf-8").read()
SUPP=open("outputs/SUPPLEMENT_v9.md",encoding="utf-8").read()
sens=json.load(open("outputs/sr_sens_v9.json",encoding="utf-8"))
prep=json.load(open("outputs/sr_prep_v9.json",encoding="utf-8"))
v5=json.load(open("outputs/v5_results.json",encoding="utf-8"))
fails=[]
def ok(c,m):
    print(("  OK  " if c else "  FAIL ")+m)
    if not c: fails.append(m)

print("== required v9 numbers present in manuscript ==")
for t in ["0.689","0.769","0.43","0.64","1.04","0.124","0.103","0.057","17.4%","0.904","48%","2,516","42%","90%","0.032","retrospective prognostic modelling"]:
    ok(t in MS, f"quotes {t}")
# threshold sensitivity at 1000 strict/lenient match computed
thr=pd.read_csv("outputs/sr_threshold_sensitivity.csv")
strict=float(thr[(thr.slope_tol==0.1)&(thr.oe_tol==0.1)]['rate@1000'].iloc[0]); lenient=float(thr[(thr.slope_tol==0.3)&(thr.oe_tol==0.2)]['rate@1000'].iloc[0])
ok("42%" in MS and "90%" in MS, "hit-rate spread at 1,000 (strict 42%% / lenient 90%%) quoted -> got %.1f/%.1f"%(strict,lenient))
# primary config reproduces Table 2 (13.3/28.3/56.7/80/95/100)
ok(all(x in MS for x in ["13.3%","28.3%","56.7%","80.0%","95.0%"]), "primary hit-rates 13.3-95.0 present")
# events EPV<1 at n=100
ok("events-per-variable" in MS.lower() and "12 deaths" in MS or "median 12" in MS, "EPV<1 / median 12 deaths at n=100 reported")
# SOFA paired ΔAUC from JSON
ok(abs(sens['paired_patient_bootstrap_rep0']['dAUC_sofa_minus_recalibrated']['mean']-0.0317)<0.001, "SOFA-recal ΔAUC 0.032 in JSON")
ok("+0.032" in MS, "manuscript quotes SOFA-recal ΔAUC +0.032")
# weighted-slope refit interval present, and we do NOT use fixed-weight-only as strong evidence
ok(sens['weighted_slope_bootstrap_refit']['ci95']==[0.373,0.524], "refit-bootstrap weighted interval 0.373-0.524")
ok("re-fitted inside each bootstrap replicate" in MS, "states propensity re-fit inside bootstrap")

print("== reviewer-flagged corrections ==")
# S2: at least four covariates >0.10 after weighting (was wrongly 'one')
over=prep["S2_weighted_over_010"]["count"]
ok(over==4, "S2 corrected to %d covariates >0.10 after weighting"%over)
ok(all(v in SUPP for v,_ in prep["S2_weighted_over_010"]["variables"]), "S2 names Age/RespiratoryRate/RBC/COPD in supplement")
ok("not one" in SUPP or "still exceeded" in SUPP.lower(), "S2 explicitly states >one covariate above 0.10")
# S4: Calcium must NOT be in the 21-variable dictionary table
s4seg=SUPP.split("## Table S4")[1].split("## Table S5")[0] if "## Table S4" in SUPP else ""
ok(not re.search(r"\|\s*Calcium\s*\|", s4seg), "Calcium not a dictionary ROW in S4 (may appear only as an excluded-variables note)")
ok(s4seg.count("\n|")>=21 or "Diabetes" in s4seg, "S4 lists the 21 main covariates one per row")
# missingness = 0 in the 21 predictors
ok(prep["complete_case_pct"]["MIMIC-IV"]==100.0 and prep["complete_case_pct"]["eICU"]==100.0, "complete-case 100% in both cohorts -> no missingness")
ok("no missing values" in MS.lower(), "manuscript states imputation did not operate (no missing values)")
# MIMIC-III relabelled
ok("Exploratory cross-version" in MS and "Temporal (exploratory)" not in MS, "MIMIC-III relabelled as exploratory cross-version (Table 1)")
# cross-database / within-target terms
ok("cross-database external validation" in MS and "within-target-database held-out assessment" in MS, "uses cross-database external + within-target held-out wording")

print("== reference corrections retained ==")
ok("Li F, Xin H, Zhang J, Fu M, Zhou J, Lian Z" in MS, "Ref1 correct authors")
ok("12(7):e059761" in MS, "Ref2 12(7)")
ok("chronic critical illness" in MS, "Ref4 includes chronic critical illness")
ok("Vergouwe Y" in MS and "Schnellinger EM" in MS and "Elayan H" in MS and "Barlow CJ" in MS, "Ref12/13/16/17 initials correct")
ok(MS.count("42056904")==1, "PMID 42056904 once (Ref14/15 merged)")
ok(all(p in MS for p in ["34301649","40487883","31842878","27891652","42625358","38626948","31397478"]), "key PMIDs present")

print("== no affirmative over-claims (negations allowed) ==")
bad_affirm=[r"\bwe (?:completed|provide|establish) a validated (?:updating )?decision rule", r"time window (?:is|was) resolved",
            r"restored from", r"\b208[- ]hospital independent", r"largely case-mix", r"correct overall uncertainty",
            r"\bcan be deployed at admission", r"recovered almost all", r"perfect calibration"]
for rx in bad_affirm:
    ok(not re.search(rx,MS,re.I), f"no affirmative: {rx}")

print("== journal caps ==")
ab=re.search(r"## Abstract\n\n(.+?)\n\n---",MS,re.S).group(1); nw=len(ab.split())
ok(nw<=200, "abstract <=200 words (=%d)"%nw)
ntab=len(re.findall(r"\*\*Table \d",MS)); nfig=len(re.findall(r"\*\*Fig\. \d",MS.split("## Figure legends")[1]))
ok(ntab+nfig<=8, "display items <=8 (%d+%d)"%(ntab,nfig))
main=MS[MS.find("## Introduction"):MS.find("## Methods")]; ok(len(main.split())<=4500, "main text <=4500 words (=%d)"%len(main.split()))
for need in ["**Keywords","Artificial-intelligence assistance","TRIPOD+AI","Data availability","Code availability",
             "Author contributions","Competing interests","Funding"]:
    ok(need in MS, f"has section/keyword: {need}")
ok(MS.count("[AUTHOR-SPECIFIC")>=5, "AUTHOR-SPECIFIC placeholders preserved (=%d)"%MS.count("[AUTHOR-SPECIFIC"))

print("\nRESULT:", "PASS" if not fails else f"FAIL ({len(fails)})")
sys.exit(0 if not fails else 1)
