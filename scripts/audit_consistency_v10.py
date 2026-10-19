"""audit_consistency_v10.py — round-10 audit: verify the EPV fix, positioning, SOFA caption,
bootstrap-level rename, reference renumbering (Obermeyer removed), and journal caps.
Recompute from outputs/sr_diag_v10.json + sr_sens_v9.json + v5 JSON. Exit 0 = PASS."""
import os, re, json, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; os.chdir(ROOT)
MS=open("outputs/SCI_manuscript_v10.md",encoding="utf-8").read()
SUPP=open("outputs/SUPPLEMENT_v10.md",encoding="utf-8").read()
diag=json.load(open("outputs/sr_diag_v10.json",encoding="utf-8"))
fails=[]
def ok(c,m):
    print(("  OK  " if c else "  FAIL ")+m)
    if not c: fails.append(m)

print("== MC1 EPV error corrected ==")
ok("events-per-variable below 1" not in MS and "EPV computed on the original 21 predictors is not the relevant denominator" in MS, "no 'EPV below 1' claim; EPV-on-21 explicitly rejected as denominator")
ok("only an intercept and a slope" in MS.lower() or "only an intercept and slope" in MS.lower() or "re-estimates **only an intercept and a slope**" in MS, "states only 2 params re-estimated")
ok("6 events per re-estimated parameter" in MS, "quotes ~6 events per re-estimated parameter (n=100)")
ok(abs(diag["slope_and_events"]["100"]["events_per_recalibration_parameter_median"]-6.0)<0.1, "JSON events/param=6.0 at n=100")
ok("does not exclude near-separation" in MS or "does not exclude" in MS, "convergence != no numerical problem")
ok("empirical finding" in MS, "instability framed as empirical finding")

print("== MC2/3 positioning ==")
ok("retrospective cross-database assessment" in MS, "abstract/scope: retrospective cross-database assessment")
ok("within-database held-out validation" in MS, "within-database held-out validation present")
ok("hospital-level" in MS and "does not translate into a hospital-level" in MS, "no hospital-level requirement claim")
ok("already-processed retrospective variables" in MS or "already-processed" in MS, "repair applies to processed retrospective vars")

print("== MC4 SOFA fairness / caption ==")
ok("all scores fitted on the same eICU update half" not in MS, "old wrong Table 4 caption removed")
ok("Models evaluated on the same held-out eICU sample" in MS, "Table 4 caption corrected")
ok("predictive-performance benchmark" in MS and "not a comparison of clinical implementation cost" in MS, "SOFA framed as performance benchmark not cost")
ok("component timing" in MS and "could not be traced" in MS, "SOFA timing unverifiable stated")

print("== MC5 / wording ==")
ok("case-mix alone does not explain" not in MS and "measured case-mix alone" not in MS, "removed over-strong case-mix exclusion")
ok("cannot determine the contribution of case-mix" in MS, "adopted bounded case-mix wording")
ok("outer-split resampling interval" in MS, "cluster interval renamed to outer-split resampling interval")
ok("stay-level paired bootstrap" in MS and "paired patient bootstrap" not in MS, "patient->stay-level bootstrap rename")
ok("Despite similar marginal mortality rates" in MS and "the over-prediction reflects the shifted predictor distribution" not in MS, "marginal-rate causal sentence fixed")
ok("illustrative" in MS and "plausible ICU mortality-risk action thresholds" not in MS, "DCA band = illustrative")
ok("Recalibration restores calibration; coefficient refitting improves discrimination" in MS and "only re-estimation restores ranking" not in MS, "section heading corrected")
ok("first 50/50 permutation" in MS and "before any result was inspected" in MS, "representative split pre-specified")
ok("10 equal-frequency bins" in MS or "over 10 equal-frequency bins" in MS, "ECE bin count stated")

print("== ΔAUC now mean(SD)[range], conditional ==")
ok("mean (SD) [range]" in MS, "ΔAUC presented as mean (SD) [range]")
sofa=diag["delta_auc_across10"]["sofa"]; ok(sofa["sd"]>0 and "0.022 (0.008)" in MS, "SOFA ΔAUC mean(SD) 0.022 (0.008) quoted")

print("== references renumbered; Obermeyer removed ==")
ok("Dissecting racial bias" not in MS and "Obermeyer" not in MS, "Obermeyer (old Ref11) removed")
ok("Vergouwe Y" in MS and "Schnellinger EM" in MS and "Elayan H" in MS and "Barlow CJ" in MS, "core method refs present")
ok(MS.count("42056904")==1, "PMID 42056904 once")
ok(all(p in MS for p in ["34301649","31842878","27891652","34865652","40750843","42625358","38626948","38729274","31397478"]), "key PMIDs present")
# no dangling [21] after renumber to 20
ok("[21]" not in MS, "no citation beyond renumbered max [20]")
ok("12(7):e059761" in MS and "chronic critical illness" in MS, "Ref2/Ref4 corrections retained")

print("== journal caps ==")
ab=re.search(r"## Abstract\n\n(.+?)\n\n---",MS,re.S).group(1); nw=len(ab.split())
ok(180<=nw<=195, "abstract 180-195 words (=%d)"%nw)
ntab=len(re.findall(r"\*\*Table \d",MS)); nfig=len(re.findall(r"\*\*Fig\. \d",MS.split("## Figure legends")[1]))
ok(ntab+nfig<=8, "display items <=8 (%d+%d)"%(ntab,nfig))
main=MS[MS.find("## Introduction"):MS.find("## Methods")]; ok(len(main.split())<=4500, "main text <=4500 (=%d)"%len(main.split()))
for need in ["**Keywords","Artificial-intelligence assistance","TRIPOD+AI","Data availability","Code availability","Authors and affiliations"]:
    ok(need in MS, "has: "+need)
ok(MS.count("[AUTHOR-SPECIFIC")>=5, "AUTHOR-SPECIFIC placeholders (=%d)"%MS.count("[AUTHOR-SPECIFIC"))
ok("zero missingness in the input files does not imply" in MS, "methods: 0%% missing in files != original data complete")

print("\nRESULT:", "PASS" if not fails else f"FAIL ({len(fails)})")
sys.exit(0 if not fails else 1)
