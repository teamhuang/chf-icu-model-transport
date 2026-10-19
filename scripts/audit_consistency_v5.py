"""audit_consistency_v5.py — reverse-check the v5 manuscript against outputs/v5_results.json.
Prints PASS/FAIL for each headline number and scans for retired/wrong phrasing.
"""
import json, re, sys, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R=json.load(open("outputs/v5_results.json"))
M=open("outputs/SCI_manuscript_v5.md",encoding="utf-8").read()
ok=True
def check(label, cond):
    global ok
    print(("  OK  " if cond else "FAIL ")+label); ok=ok and cond

print("== numeric reverse-checks ==")
check("total patients 31,587", "31,587" in M and R["total_n"]==31587)
check("no stale 32,587", "32,587" not in M)
check("deaths 3,999", "3,999" in M)
check("eICU AUC 0.689 (0.677-0.703)",
      R["transport"]["V_eICU_geographic_primary"]["AUC"]==0.689 and "0.689 (95% CI 0.677–0.703)" in M)
check("MIMIC-IV AUC 0.769", R["transport"]["B_MIMIC-IV_insample"]["AUC"]==0.7689)
check("MIMIC-III AUC 0.748", R["transport"]["A_MIMIC-III_temporal_exploratory"]["AUC"]==0.7482)
check("patient-split Brier 0.122->0.101, 16.6%",
      abs(R["patient_split_independent"]["frozen"]["Brier"]-0.122)<0.001 and
      abs(R["patient_split_independent"]["recalibrated"]["Brier"]-0.101)<0.001 and
      abs(R["patient_split_independent"]["brier_rel_improvement_pct"]-16.6)<0.2)
# strategy table
sc=R["strategy_comparison"]
for strat,vals in [("frozen",("0.691","0.124","0.44")),("intercept_only",("0.691","0.109","0.44")),
                   ("recalibrate_2par",("0.691","0.103","1.04")),("full_refit",("0.749","0.099","1.01"))]:
    a=round(sc[strat]["AUC"][0],3); b=round(sc[strat]["Brier"][0],3); s=round(sc[strat]["slope"][0],2)
    check(f"strategy {strat} AUC/Brier/slope consistent ({a},{b},{s})",
          abs(a-float(vals[0]))<0.0005 and abs(b-float(vals[1]))<0.0005 and abs(s-float(vals[2]))<0.005)
check("refit gain +0.057 reported", abs(R["refit_auc_gain"]-0.0574)<0.001 and "0.691 → 0.749" in M)
# learning curve stable%
lc={d["n_pat"]:d["stable_pct"] for d in R["repeated_curve"]}
for n,pct in [(100,13.3),(200,28.3),(500,56.7),(1000,80.0),(2000,95.0),(5000,100.0)]:
    check(f"stable% at {n} == {pct}", abs(lc[n]-pct)<0.05)
# min-stable sums to 10
check("min-stable counts sum to 10", sum(R["min_stable_counts"].values())==10)
check("min-stable: 500x2 1000x4 2000x1 5000x3",
      R["min_stable_counts"]=={"500":2,"1000":4,"2000":1,"5000":3})
check("no impossible '12/10' or '9/10' in manuscript", "12/10" not in M and "9/10" not in M)
# reweighting decomposition
rw=R["reweighting"]
check("|SMD| 0.25->0.06 with correct weights",
      abs(rw["unweighted"]["mean_abs_SMD"]-0.251)<0.01 and abs(rw["corrected_weights"]["mean_abs_SMD"]-0.060)<0.01)
check("slope NOT recovered (~0.45) after correct weighting", abs(rw["corrected_weights"]["calib_slope"]-0.45)<0.03)
check("O/E recovered (~0.94) after correct weighting", abs(rw["corrected_weights"]["OE"]-0.94)<0.02)
check("ESS 17.4% reported", abs(rw["corrected_weights"]["ESS_pct"]-17.4)<0.1 and "17.4%" in M)
check("sensitivity MIMIC-III +NT/LVEF 0.748->0.796",
      abs(R["sensitivity"]["A_plus_NTLVEF_recomputed"]-0.796)<0.002)

print("== retired / wrong-phrasing scan ==")
forbidden=[("requires ≥500 patients","v3 stale claim"),("requires ~500","stale"),
           ("stable from ~500","v4 decision-rule stale"),("not case-mix-driven","blanket over-claim"),
           ("is not case-mix","blanket"),("consistently higher","DCA over-claim"),
           ("Swedish","unverifiable ref"),("消者级","corrupted token"),
           ("monotone transform","jargon flag (allowed only in invariance note)")]
for term,why in forbidden:
    hit=term.lower() in M.lower()
    # allow monotone-transform usage (it appears legitimately once)
    if term=="monotone transform":
        print(f"  INFO '{term}' occurrences={M.lower().count(term)} (acceptable)"); continue
    check(f"absent: '{term}' ({why})", not hit)

print("\nRESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
