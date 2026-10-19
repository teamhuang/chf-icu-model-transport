"""audit_consistency_v7.py — v7 reverse-check + retired-phrase scan. Must print PASS."""
import json,sys,os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V5=json.load(open("outputs/v5_results.json"));V6=json.load(open("outputs/v6_results.json"));V7=json.load(open("outputs/v7_inference.json"))
M=open("outputs/SCI_manuscript_v7.md",encoding="utf-8").read()
ok=True
def chk(l,c):
    global ok;print(("  OK  " if c else "FAIL ")+l);ok=ok and c

print("== v7 numeric/consistency ==")
chk("31,587 / 3,999 present; no 32,587","31,587" in M and "3,999" in M and "32,587" not in M)
chk("abstract slope uses 10-split mean 1.04","mean held-out calibration slope of 1.04" in M)
chk("single-split 1.11 labelled illustrative and reconciled","(This single illustrative split's slope is 1.11" in M)
lc=V7["learning_curve_cluster"]
chk("cluster CI 1000 = 65-93 quoted",abs(lc["1000"]["cluster_bootstrap_ci"][0]-65)<0.1 and "65–93" in M)
chk("cluster CI 500 = 45-71.7 quoted","45.0–71.7%" in M)
chk("cluster CI 2000 = 90-100 quoted",lc["2000"]["cluster_bootstrap_ci"]==[90.0,100.0] and "90.0–100.0%" in M)
chk("naive vs cluster contrast present","Naive (independence)" in M)
chk("across-split ΔAUC 0.057 (0.055-0.060)",abs(V7["across_split_delta"]["dAUC_retrain_minus_recalibrate"]["mean"]-0.0574)<0.001 and "0.057 (95% CI 0.055–0.060" in M)
chk("single-split ΔAUC 0.055 (0.043-0.069)","0.055 (95% CI 0.043–0.069)" in M)
chk("two estimands named (test-set vs update-sample)","reflects *test-set*" in M and "reflects *update-sample*" in M)
chk("min-stable 500x2/1000x4/2000x1/5000x3",V5["min_stable_counts"]=={"500":2,"1000":4,"2000":1,"5000":3} and "500 in two splits, 1,000 in four, 2,000 in one, and 5,000 in three" in M)
chk("monotonicity 9/10","Nine of ten splits were monotone" in M)
rw=V6["weight_sensitivity"]["0.1-10"]
chk("max residual SMD 0.21 disclosed","largest residual per-variable imbalance was 0.21" in M)
chk("ESS 17.4% disclosed","effective sample size under the balance-achieving weights was only 17.4%" in M)
wci=V6["weighted_slope_CI01_10"];chk("weighted slope 0.45 CI 0.38-0.52","0.45 (95% CI 0.38–0.52)" in M and abs(wci["mean"]-0.45)<0.01)
chk("Brier rel 16.6% from unrounded stated","16.6% reduction computed from unrounded values" in M)
chk("ECE formula present","ECE = Σ_b" in M)
chk("calibration formula explicit","logit P(Y=1) = α + β·logit(p_frozen)" in M or "logit P(Y=1) = α + β·logit(p_frozen)" in M or "α + β·logit(p_frozen)" in M)
chk("unit-of-analysis (no subject id dedup) disclosed","carry no patient identifier" in M or "no subject identifier" in M)
chk("prediction-window documented (first-24h/early-prognostic)","first 24 hours" in M and "early-prognostic" in M)
chk("no stale 'cannot verify window' claim","could not verify whether the predictors" not in M)
chk("source PMCID anchors present","PMC9315914" in M and "PMC8311359" in M)
chk("refit called fixed-predictor refit","Fixed-predictor refit" in M or "fixed-predictor refit" in M)
chk("DCA harm softened","can yield net benefit no better than taking no model-informed action" in M)
chk("case-mix standardisation (not decomposition)","standardisation analysis, not a causal decomposition" in M)

print("== references corrected ==")
for s,why in [("4529–4539","Vergouwe real pages"),("27891652","closed testing"),("5(1):20","Schnellinger vol:page"),
              ("34865652","dynamic updating"),("1448–1457","JAMIA pages"),("42625358","Barlow Pilcher 2026"),
              ("40750843","Elayan case-mix"),("10.64898/2026.05.03.26352335","medRxiv competitor")]:
    chk(f"contains {s} ({why})", s in M)
for bad in ["4452–4462","1648–1657","Swedish"]:
    chk(f"old-wrong/retired absent: {bad}", bad not in M)

print("== retired over-claims scan ==")
for t,why in [("one sixth","v5 data-cost"),("1/6","data-cost"),("perfect balance","over-claim"),
              ("no covariate-shift","over-claim"),("largely preserved","discrimination over-claim"),
              ("recovered almost all","MC4"),("hospital's own patients","MC7"),
              ("receiving hospital","MC7"),("repeated independent patient","non-independence"),
              ("full redevelopment","mislabel"),("12/10","impossible count"),("genuine external evidence","overclaim")]:
    chk(f"absent '{t}' ({why})", t.lower() not in M.lower())
chk("'full model redevelopment' negated correctly","not a full model redevelopment" in M)

print("\nRESULT:","PASS" if ok else "FAIL");sys.exit(0 if ok else 1)
