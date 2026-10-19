"""audit_consistency_v6.py — reverse-check v6 manuscript against v5_results.json + v6_results.json,
and scan for the over-claims retired this round. PASS required before ship.
"""
import json,sys,os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V5=json.load(open("outputs/v5_results.json")); V6=json.load(open("outputs/v6_results.json"))
M=open("outputs/SCI_manuscript_v6.md",encoding="utf-8").read()
ok=True
def chk(lbl,c):
    global ok; print(("  OK  " if c else "FAIL ")+lbl); ok=ok and c
print("== v6 numeric checks ==")
chk("total 31,587 patients / 3,999 deaths", "31,587" in M and "3,999" in M)
chk("no stale 32,587", "32,587" not in M)
t=V5["transport"]
chk("eICU AUC 0.689 (95% CI 0.677-0.703)", t["V_eICU_geographic_primary"]["AUC"]==0.689 and "0.689, 95% CI 0.677–0.703" in M)
chk("slope 0.43 / O/E 0.64 eICU", round(t["V_eICU_geographic_primary"]["calib_slope"],2)==0.43 and round(t["V_eICU_geographic_primary"]["OE"],2)==0.64)
ps=V5["patient_split_independent"]
chk("patient-split Brier 0.122->0.101 (16.6%)", abs(ps["frozen"]["Brier"]-0.122)<0.001 and abs(ps["recalibrated"]["Brier"]-0.101)<0.001)
fc=V6["flexible_calibration"]
chk("ECE frozen 0.098 -> recal 0.020", abs(fc["frozen"]["ECE"]-0.098)<0.002 and abs(fc["recalibrated"]["ECE"]-0.020)<0.002 and "0.098 → 0.020" in M)
pr=V6["paired_recal_vs_refit"]
chk("paired ΔAUC 0.055 (0.043-0.069)", abs(pr["dAUC_refit_minus_recal"]["mean"]-0.055)<0.002 and "0.055 (95% CI 0.043–0.069)" in M)
chk("paired ΔBrier 0.004 (0.003-0.006)", abs(abs(pr["dBrier_refit_minus_recal"]["mean"])-0.004)<0.001)
chk("ΔNB@0.10 ~0.012 (0.009-0.016)", abs(pr["dNetBenefit_refit_minus_recal"]["0.1"]["mean"]-0.0124)<0.001)
lc=V6["learning_curve_monotonicity"]
sw={d["n_pat"]:d for d in lc["per_update_success_with_CI"]}
chk("success 500 -> 57% (44-68)", sw[500]["success_frac"]==56.7 and "57% (95% CI 44–68)" in M)
chk("success 1000 -> 80% (68-88)", sw[1000]["success_frac"]==80.0 and "80% (68–88)" in M)
chk("success 2000 -> 95% (86-98)", sw[2000]["success_frac"]==95.0 and "95% (86–98)" in M)
chk("min-stable secondary dist 500x2/1000x4/2000x1/5000x3", V5["min_stable_counts"]=={"500":2,"1000":4,"2000":1,"5000":3})
chk("monotonicity 9/10 monotone (1 reversal)", lc["non_monotone_splits"]==1 and "nine reached stability monotonically" in M)
rw=V6["weight_sensitivity"]["0.1-10"]
chk("|SMD| 0.25->0.06, max 0.21", abs(rw["mean_abs_SMD"]-0.060)<0.005 and "largest residual per-variable imbalance 0.21" in M)
wsci=V6["weighted_slope_CI01_10"]
chk("weighted slope 0.45 CI (0.38-0.52)", abs(wsci["mean"]-0.45)<0.01 and "0.45 (95% CI 0.38–0.52)" in M)
chk("ESS 17.4% disclosed", abs(rw["ESS_pct"]-17.4)<0.1 and "17.4%" in M)
ws=V6["weight_sensitivity"]; slopes=[ws[k]["weighted_slope"] for k in ws]
chk("truncation slope stays 0.43-0.45", min(slopes)>=0.42 and max(slopes)<=0.46)

print("== retired over-claim scan ==")
banned=[("one sixth","data-cost over-claim MC3"),("1/6","data-cost"),
        ("perfect balance","MC1 over-claim"),("no covariate-shift method","MC1 over-claim"),
        ("largely preserved","discrimination over-claim"),("recovered almost all","MC4 over-claim"),
        ("hospital's own patients","MC7 over-reach"),("receiving hospital","MC7"),
        ("genuine external evidence","secondary-split wording"),("12/10","impossible count"),
        ("requires ≥500","stale"),("stable from ~500","stale"),("Swedish","unverifiable ref")]
for term,why in banned:
    chk(f"absent '{term}' ({why})", term.lower() not in M.lower())

print("== required hedge present ==")
chk("deployment rule labelled exploratory/unvalidated", "prospectively unvalidated protocol" in M)
chk("prediction-window limitation present", "prediction time window" in M)
chk("method precedents cited (closed testing / dynamic updating)", "PMID 27891652" in M and "PMID 34865652" in M)
chk("case-mix cited as precedent (40750843)", "40750843" in M)

print("\nRESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
