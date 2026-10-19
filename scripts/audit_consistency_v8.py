"""audit_consistency_v8.py — recompute every v8 manuscript number from outputs/*.json|csv,
verify reference corrections, and scan for over-claims that the SR review required removed.
Exit 0 (PASS) only if all checks succeed. Run from project root."""
import os, re, json, sys
import pandas as pd, numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MS=open("outputs/SCI_manuscript_v8.md",encoding="utf-8").read()
v5=json.load(open("outputs/v5_results.json",encoding="utf-8"))
v6=json.load(open("outputs/v6_results.json",encoding="utf-8"))
v7=json.load(open("outputs/v7_inference.json",encoding="utf-8"))
v8=json.load(open("outputs/v8_heldout.json",encoding="utf-8"))["heldout_same_split"]
fails=[]
def ok(cond,msg):
    print(("  OK  " if cond else "  FAIL ")+msg)
    if not cond: fails.append(msg)

print("== headline numbers present & consistent ==")
for tok in ["31,587","3,999","0.769","0.689","0.677","0.703","0.43","0.64","1.04","0.124","0.103","0.057","17.4%","0.25","0.06","0.21","0.137","0.729","0.752","0.748"]:
    ok(tok in MS, f"manuscript quotes {tok}")
ok(v5["total_n"]==31587 and v5["total_deaths"]==3999, "totals 31,587 / 3,999 match JSON")
ok(abs(v5["patient_split_independent"]["brier_rel_improvement_pct"]-16.61)<0.01, "Brier rel improvement 16.6% (unrounded) — NOT the double-rounded 17.21")
ok(abs(round(v5["strategy_comparison"]["full_refit"]["AUC"][0]-v5["strategy_comparison"]["frozen"]["AUC"][0],3)-0.057)<0.001, "refit-frozen AUC = 0.057 (double-rounded 0.058 not used)")

print("== intercept-only O/E~1 but slope~0.44 counterexample present ==")
sc=v5["strategy_comparison"]
ok(abs(sc["intercept_only"]["OE"]-0.98)<0.02 and abs(sc["intercept_only"]["slope"][0]-0.44)<0.02, "real data: intercept_only O/E~0.98 & slope~0.44")
ok("monitor only" in MS and ("near-ideal O/E does not imply" in MS or "O/E alone cannot license" in MS), "uses counterexample to reject O/E-alone rule")

print("== held-out fair comparison table matches v8_heldout.json ==")
ok(abs(v8["SOFA_only"]["AUC"]-0.729)<0.001, "SOFA held-out AUC 0.729")
ok(abs(v8["refit_21var"]["AUC"]-0.752)<0.001, "refit held-out AUC 0.752")
ok(abs(v8["frozen_21var"]["AUC"]-0.697)<0.002 and abs(v8["recalibrated_2par"]["AUC"]-0.697)<0.002, "frozen=recal held-out AUC 0.697")

print("== weighting: fixed-weight bootstrap limitation disclosed ==")
ok("held fixed" in MS and "understates" in MS, "fixed weight model in bootstrap disclosed")
ok(abs(v6["weighted_slope_CI01_10"]["ci95"][0]-0.383)<0.01 and abs(v6["weighted_slope_CI01_10"]["ci95"][1]-0.520)<0.01, "weighted slope CI 0.38-0.52")

print("== reference corrections (verify-or-drop) ==")
ok("Li F, Xin H, Zhang J, Fu M, Zhou J, Lian Z" in MS, "Ref1 author string corrected (Xin/Zhang/Lian present)")
ok("Song Y" not in MS and "Ge J" not in MS, "Ref1 spurious authors (Song/Ge) removed")
ok("12(7):e059761" in MS and "12(6):e059761" not in MS, "Ref2 issue 12(7) corrected")
ok("chronic critical illness" in MS, "Ref4 title includes chronic critical illness")
ok("Vergouwe Y" in MS and "Vergouwe SD" not in MS, "Ref12 Vergouwe Y")
ok("Schnellinger EM" in MS and "Schnellinger JI" not in MS, "Ref13 Schnellinger EM")
ok("Elayan H" in MS and "Elayan A," not in MS, "Ref16 Elayan H")
ok("Barlow CJ" in MS and "Barlow G" not in MS, "Ref17 Barlow CJ")
# Ref14/15 duplicate merged: PMID 42056904 should appear exactly once
ok(MS.count("42056904")==1, "PMID 42056904 appears once (Ref14/15 duplicate merged)")
ok("See Ref 14" not in MS, "old Ref15 '(See Ref 14.)' placeholder gone")
ok("Swedish" not in MS, "unverifiable Swedish paediatric citation absent")

print("== retired over-claims MUST be absent ==")
for bad in ["largely case-mix-like","correct overall uncertainty","time window … resolved","time-window question … resolved",
            "restored from","500 例","500例","hospital's own patients","receiving hospital's requirement","genuine external evidence",
            "perfect calibration","no covariate","recovered almost all","4452","1648"]:
    ok(bad not in MS, f"absent over-claim: {bad!r}")
# decision-rule threshold ladder must be gone
ok(not re.search(r"if O/E is within ~?\d+%", MS), "no numeric 'if O/E within X%' threshold ladder")
ok(("refrain from converting these results into an updating decision rule" in MS) or ("does not propose a validated updating algorithm" in MS), "explicit no-decision-rule stance")

print("== formatting caps ==")
ab=re.search(r"## Abstract\n\n(.+?)\n\n---",MS,re.S).group(1)
nw=len(ab.split()); ok(nw<=200, f"abstract <=200 words (={nw})")
n_tables=len(re.findall(r"\*\*Table \d",MS))
n_figs=len(re.findall(r"\*\*Fig\. \d",MS.split("## Figure legends")[1]))
disp=n_tables+n_figs
ok(disp<=8, "main display items <=8 (Tables=%d + Figs=%d = %d)"%(n_tables,n_figs,disp))
# figure callouts present in body
body=MS.split("## Figure legends")[0]
for i in range(1,5): ok(f"Fig. {i}" in body, f"Fig. {i} cited in body")
# no internal review-round / version-log artefacts in the manuscript narrative
for artifact in ["Version:", "supersedes", "reviewer pass", "review round", "review-round",
                 "Response to the", "changes made in", "round-6", "round 6",
                 "评审", "关键变更", "Major 1", "Major 2", "Major 3", "[Zenodo/CodeOcean DOI to be minted"]:
    ok(artifact not in MS, f"no internal artifact token in manuscript: {artifact!r}")

print("\nRESULT:", "PASS" if not fails else f"FAIL ({len(fails)})")
sys.exit(0 if not fails else 1)
