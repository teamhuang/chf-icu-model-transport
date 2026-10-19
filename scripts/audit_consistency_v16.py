"""audit_consistency_v16.py — round-16 audit (figure evidence-chain reorganisation).

Carries over every v15 numeric lock (Table2=S7=S11=JSON; Table4=S13=JSON; subgroup=JSON;
model-spec betas == sklearn L2; round-13/14/15 content checks) and ADDS the round-16
figure checks:
  * five-figure evidence chain present (Fig. 1 roadmap + Fig. 2-5 renumbered), 4 main tables;
  * the new Figure 1 is a study-design/analytical-workflow ROADMAP that visually separates
    the frozen model's cross-database external assessment from the within-eICU update;
  * every in-text figure citation renumbered consistently (no stale old numbers);
  * supplementary Fig. S2/S3 legends present and their source figures exist on disk;
  * every main/supplementary figure file referenced in the legends exists;
  * display-item cap relaxed to 5 figures + 4 tables (=9);
  * version/build tags updated to r16 / audit_consistency_v16.py.
Exit 0 = PASS.
"""
import os, re, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
ROOT=Path(__file__).resolve().parents[1]; os.chdir(ROOT)
MS=open("outputs/SCI_manuscript_v16.md",encoding="utf-8").read()
SUPP=open("outputs/SUPPLEMENT_v16.md",encoding="utf-8").read()
TRI=open("outputs/SUPPLEMENT_TRIPOD_AI_v16.md",encoding="utf-8").read()
diag=json.load(open("outputs/sr_diag_v10.json",encoding="utf-8"))["slope_and_events"]
v12=json.load(open("outputs/sr_v12.json",encoding="utf-8"))
fails=[]
def ok(c,m):
    print(("  OK  " if c else "  FAIL ")+m)
    if not c: fails.append(m)
def num(s): return float(str(s).replace("−","-"))

# ---------- parse main Table 2 ----------
t2={}
for line in MS.splitlines():
    mm=re.match(r"\|\s*([\d,]+)\s*\|\s*([\d.]+)\s*\((\d+)[–-](\d+)\)\s*\|\s*(-?[\d.]+)\s*\[(-?[\d.]+)[–-](-?[\d.]+)\]\s*\(([-−]?[\d.]+)\s*to\s*([-−]?[\d.]+)\)", line)
    if mm:
        size=int(mm.group(1).replace(",",""))
        t2[size]={"events_median":num(mm.group(2)),"slope_median":num(mm.group(5)),"slope_min":num(mm.group(8)),"slope_max":num(mm.group(9))}
s7={}; s7seg=SUPP.split("## Table S7")[1].split("## Table S8")[0]
for line in s7seg.splitlines():
    cells=[c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells)>=6 and re.match(r"^\d+$", cells[0]):
        s7[int(cells[0])]={"events_median":num(cells[1]),"epp":num(cells[4]),"slope_median":num(cells[5])}
s11={}; s11seg=SUPP.split("## Table S11")[1].split("## Table S12")[0]
for line in s11seg.splitlines():
    cells=[c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells)>=10 and re.match(r"^\d+$", cells[0]):
        s11[int(cells[0])]={"unpen_med":num(cells[1]),"unpen_max":num(cells[2]),"unpen_neg":int(cells[3]),"rob_max":num(cells[5]),"ev_all":num(cells[9])}
print("== (1-3) Table 2 == S7 == S11 == JSON (single source) ==")
ok(len(t2)==6, f"Table 2 parsed 6 sizes (got {len(t2)})")
for n in [100,200,500,1000,2000,5000]:
    jm=round(diag[str(n)]["slope_median"],2); jmax=round(diag[str(n)]["slope_max"],2); je=round(diag[str(n)]["events_median"],1)
    ok(abs(t2[n]["slope_median"]-jm)<0.005 and abs(s7[n]["slope_median"]-jm)<0.005, f"n={n} slope Table2/S7 == JSON {jm}")
    ok(abs(s11[n]["unpen_med"]-jm)<0.005 and abs(s11[n]["unpen_max"]-jmax)<0.02, f"n={n} S11 unpen med/max == JSON {jm}/{jmax}")
    ok(abs(t2[n]["events_median"]-je)<0.06 and abs(s11[n]["ev_all"]-je)<0.06, f"n={n} events Table2/S11 == JSON {je}")

# ---------- (4) Table 4 == S13 == JSON ----------
print("== (4) Table 4 slope/OE == S13 == JSON ==")
t4map={}
t4seg=MS.split("**Table 4.")[1].split("### ")[0]
for line in t4seg.splitlines():
    cells=[c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells)==5 and re.match(r"^\d?\.\d", cells[1]):
        t4map[cells[0]]={"slope":num(cells[3]),"oe":num(cells[4])}
s13={}; s13seg=SUPP.split("## Table S13")[1].split("## Supplementary figures")[0]
for line in s13seg.splitlines():
    cells=[c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells)==5 and re.match(r"^\d", cells[1]):
        s13[cells[0]]={"slope":num(cells[3]),"oe":num(cells[4])}
jt={r["Model"]:r for r in v12["table4_with_calibration"]}
pairs=[("Frozen 21-variable (transported)","Frozen 21-var"),("Two-parameter recalibration","Two-parameter recalibration"),
       ("SOFA only (refit on update half)","SOFA (refit)"),("APS-III only (refit on update half)","APS-III (refit)"),
       ("SOFA + APS-III (refit on update half)","SOFA+APS-III (refit)"),("21-variable refit (update half)","21-var refit")]
ok(len(t4map)==6, f"Table 4 parsed 6 models (got {len(t4map)})")
for mn,sn in pairs:
    j=jt[sn]
    ok(abs(t4map[mn]["slope"]-round(j["slope"],2))<0.005 and abs(t4map[mn]["oe"]-round(j["OE"],2))<0.005, f"T4 {sn} slope/OE == JSON")
    ok(abs(s13[sn]["slope"]-j["slope"])<0.0005 and abs(s13[sn]["oe"]-j["OE"])<0.0005, f"S13 {sn} == JSON (unrounded)")

# ---------- (5) subgroup S12 == JSON ----------
print("== (5) S12 subgroup == JSON ==")
sg={r["subgroup"]:r for r in v12["subgroup_frozen_eICU"]}
s12seg=SUPP.split("## Table S12")[1].split("## Table S13")[0]; s12={}
for line in s12seg.splitlines():
    cells=[c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells)==6 and cells[0].startswith(("Sex","Age")):
        s12[cells[0]]={"auc":num(cells[3]),"slope":num(cells[4])}
for k,r in sg.items():
    ok(k in s12 and abs(s12[k]["auc"]-r["AUC"])<0.0005, f"S12 {k} AUC == JSON")

# ---------- (6) model_specification betas match sklearn L2 ----------
print("== (6) model_specification.csv betas == fitted sklearn L2 model ==")
COMMON=["Age","male","HeartRate","RespiratoryRate","Temperature","SBP","AG","BUN","Creatinine","Chloride","MCH","MCHC","MCV","Platelet","Potassium","RBC","RDW","Sodium","WBC","COPD","Diabetes"]
ok(os.path.exists("outputs/model_specification.csv"), "model_specification.csv exists")
spec=pd.read_csv("outputs/model_specification.csv")
ok(set(spec["variable"])==set(COMMON) and len(spec)==21, "spec has all 21 variables")
b=pd.read_csv("data/Derivation cohorts--MIMIC-IV database.csv"); b["male"]=(b["Gender"]==1).astype(int); b["outcome"]=b["Hstatus"].astype(int)
X=b[COMMON].astype(float)
imp=SimpleImputer(strategy="median").fit(X); sc=StandardScaler().fit(imp.transform(X))
lr=LogisticRegression(max_iter=2000).fit(sc.transform(imp.transform(X)), b["outcome"].values)
skmap=dict(zip(COMMON, lr.coef_[0]))
sd=pd.Series(dict(zip(spec["variable"], spec["beta_standardised"])))
maxd=max(abs(skmap[v]-sd[v]) for v in COMMON)
ok(maxd<1e-4, f"spec betas match sklearn L2 (max|diff|={maxd:.2e})")
ok(abs(lr.intercept_[0]-(-2.339))<0.002, f"intercept −2.339 cited == fitted {lr.intercept_[0]:.3f}")
ok("−2.339" in MS or "-2.339" in MS, "manuscript cites the intercept value")

# ---------- (7) round-13 + round-14 + round-15 content ----------
print("== (7) round-13/14/15 content checks ==")
ok("Two-parameter recalibration improved held-out calibration within the receiving database" not in MS, "abstract duplicate recalibration/refit/SOFA sentence removed")
ok("whole-population phenomenon" not in MS, "subgroup 'whole-population phenomenon' over-claim removed")
ok("subgroup uncertainty and heterogeneity were not formally assessed" in MS, "subgroup uncertainty/heterogeneity caveat present")
ok("calibrated in-sample by construction" not in MS, "P1-5b: 'calibrated in-sample by construction' removed")
ok("apparent calibration close to the ideal" in MS, "P1-5b: development-set calibration reworded present")
ok("almost perfectly separable" not in MS, "P1-7a: 'almost perfectly separable' removed")
ok("strongly distinguishable" in MS, "P1-7a: 'strongly distinguishable' present")
ok("stability-versus-typical-calibration trade-off" in MS, "P1-7b: trade-off reported in main text")
ok("median 1.009" in MS and "0.742" in MS, "P1-7b: median-slope trade-off numbers cited")
ok("recovery of absolute-risk estimates from an existing transported model without re-estimating all predictor coefficients" in MS, "P1-6: recalibration-contribution sentence corrected")
ok("a refitted SOFA model also outputs a probability" in MS, "P1-6: acknowledges score models output probability")
ok("fixed-predictor refit" in MS and "no feature re-selection" in MS, "P2: fixed-predictor refit defined")
s13seg2=SUPP.split("## Table S13")[1].split("## Supplementary figures")[0]
ok("by construction" not in s13seg2, "P1-1: S13 'by construction' removed")
ok("empirically observed rather than guaranteed" in s13seg2, "P1-1: S13 empirically-observed wording present")
ok("0.4483" in MS and "same estimand under different resampling procedures" in MS, "P1-2: three weighted-slope values reconciled in main text")
ok("joint empirical study of three linked phenomena" in MS, "P1-3: joint-empirical-study contribution framing present")

# ---------- (8) NEW round-16: figure evidence chain ----------
print("== (8) round-16 figure evidence chain ==")
LEG=MS.split("## Figure legends")[1]
# 5 main figure legends present, in order
for i in range(1,6):
    ok(f"**Fig. {i}.**" in LEG, f"main Fig. {i} legend present")
ok("**Fig. 1.** Study design and analytical workflow." in LEG, "Fig. 1 legend titled 'Study design and analytical workflow'")
ok("roadmap" in LEG.lower() or "workflow" in LEG.lower(), "Fig. 1 described as workflow/roadmap")
ok("Solid paths denote" in LEG and "dashed paths denote" in LEG, "Fig. 1 legend states solid=external vs dashed=within-database line convention")
ok("within-database assessments rather than independent hospital-level validations" in LEG, "Fig. 1 legend keeps the within-vs-external boundary")
for s in ["Fig. S1","Fig. S2","Fig. S3"]:
    ok(s in LEG, f"supplementary {s} legend present")
ok("regularisation trade-off" in LEG or "Regularisation trade-off" in LEG, "Fig. S2 regularisation trade-off described")
ok("Truncation" in LEG or "truncation" in LEG, "Fig. S3 truncation sensitivity described")
# in-text renumbering: no stale old numbers
ok("(Fig. 2a,b; Table 1)" in MS, "transport cited as Fig. 2a,b (renumbered)")
ok("(Fig. 3; Table 2)" in MS, "stability cited as Fig. 3 (renumbered)")
ok("(Fig. 4a–c; Table 3)" in MS, "strategy comparison cited as Fig. 4a–c")
ok("Fig. 4d" in MS, "held-out calibration curves cited as Fig. 4d")
ok("(Fig. 5a,b" in MS or "Fig. 5a,b" in MS, "reweighting cited as Fig. 5a,b")
ok("Fig. 1a,b" not in MS and "(Fig. 2; Table 2)" not in MS, "no stale old Fig.1/Fig.2 transport/stability citations")
ok("Supplementary Fig. S2" in MS and "Supplementary Fig. S3" in MS, "in-text references to new supplementary Fig. S2/S3 present")
ok("summarised in Fig. 1" in MS and "not a new external validation" in MS, "Methods points to Fig. 1 roadmap and states the update is not new external validation")
# figure files exist on disk
for f in ["v15_fig1_roadmap","v15_fig2_transport","v15_fig3_stability","v15_fig4_strategies","v15_fig5_reweighting","v15_figS2_slope_reg","v15_figS3_tails_trunc"]:
    ok(os.path.exists(f"figures/{f}.png") and os.path.exists(f"figures/{f}.pdf"), f"figure file exists: {f}.png/.pdf")
ok("figures/v15_fig1_roadmap.png" in LEG and "figures/v15_fig5_reweighting.png" in LEG, "legends reference the regenerated figure files")

# ---------- (9) version/build tags ----------
print("== (9) version / build-tag consistency ==")
ok("# Supplementary Material (analysis build 2026-10-12.r16)" in SUPP, "supplement header r16")
ok("analysis build 2026-10-12.r16" in TRI, "TRIPOD header r16")
ok("audit_consistency_v16.py" in MS and "audit_consistency_v15.py" not in MS.split("## Declarations")[0], "manuscript references v16 audit, not v15")
ok("24_reproduce_all_v16.py" in MS, "v16 reproduction driver declared in Code availability")
ok("22_v15_fig1_roadmap.py" in MS and "23_v15_data_figs.py" in MS, "figure-generation scripts declared in Code availability")
ok("model_specification.csv" in MS, "manuscript cites model_specification.csv")
ok("scripts/analysis_v5.py" not in MS.split("## Declarations")[0], "P2: main body (pre-Declarations) free of per-script enumeration")

# ---------- (10) TRIPOD figure-map consistency ----------
print("== (10) TRIPOD+AI figure-map consistency ==")
ok("Fig. 1 is a study-design/analytical-workflow roadmap" in TRI, "TRIPOD describes Fig. 1 as the workflow roadmap (not ROC/calibration)")
ok("Fig. 2b, 4" in TRI, "TRIPOD model-performance row cites renumbered Fig. 2b, 4")
ok("Fig. 1 is ROC/calibration" not in TRI, "stale 'Fig. 1 is ROC/calibration' removed from TRIPOD")
for it in ["Sample size","Data-splitting","Model updating","Protocol & registration","Patient & public involvement","Ethics","AI statement"]:
    ok(it in TRI, f"TRIPOD covers item: {it}")

# ---------- (11) banned + caps ----------
print("== (11) banned-phrase sweep + caps ==")
for b_ in ["before any result was inspected","events-per-variable below 1","all scores fitted on the same eICU update half",
           "case-mix alone does not explain","paired patient bootstrap","only re-estimation restores ranking",
           "viable statistical repair","whole-population phenomenon","(v10)","audit_consistency_v10.py"]:
    ok(b_ not in MS, f"absent banned: {b_!r}")
ab=re.search(r"## Abstract\n\n(.+?)\n\n---",MS,re.S).group(1)
ws=len(ab.split())
ok(ws<=200, f"abstract words <=200 (={ws}) [SR word count]")
ntab=len(re.findall(r"\*\*Table \d",MS))
nfig=len(re.findall(r"\*\*Fig\. \d\.\*\*",LEG))
ok(nfig==5, f"exactly 5 main figures (got {nfig})")
ok(ntab==4, f"exactly 4 main tables (got {ntab})")
ok(nfig+ntab<=9, f"display items <=9 ({ntab}+{nfig})")
main=MS[MS.find("## Introduction"):MS.find("## Methods")]; ok(len(main.split())<=4800, f"main<=4800 (={len(main.split())})")
ok(MS.count("[AUTHOR-SPECIFIC")>=5, f"AUTHOR-SPECIFIC >=5 (={MS.count('[AUTHOR-SPECIFIC')})")

print("\nRESULT:", "PASS" if not fails else f"FAIL ({len(fails)})")
for f in fails: print("   -",f)
sys.exit(0 if not fails else 1)
