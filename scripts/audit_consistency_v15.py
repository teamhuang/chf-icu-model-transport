"""audit_consistency_v15.py — round-15 audit.
Carries over all v14 numeric locks (Table2=S7=S11=JSON; Table4=S13=JSON; subgroup=JSON;
model-spec betas == sklearn L2) and adds round-15 checks:
  * S13 "by construction" removed -> "empirically observed rather than guaranteed";
  * the three weighted-slope values (0.4483 / 0.450 / 0.451) reconciled in the main text;
  * joint-empirical-study contribution framing present in the Introduction;
  * version/build tags updated to r15 / audit_consistency_v15.py.
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
MS=open("outputs/SCI_manuscript_v15.md",encoding="utf-8").read()
SUPP=open("outputs/SUPPLEMENT_v15.md",encoding="utf-8").read()
TRI=open("outputs/SUPPLEMENT_TRIPOD_AI_v15.md",encoding="utf-8").read()
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
s13={}; s13seg=SUPP.split("## Table S13")[1]
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

# ---------- (6) NEW: model_specification betas match sklearn L2 ----------
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

# ---------- (7) round-13 + round-14 content ----------
print("== (7) round-13/14 content checks ==")
ok("Two-parameter recalibration improved held-out calibration within the receiving database" not in MS, "abstract duplicate recalibration/refit/SOFA sentence removed")
ok("whole-population phenomenon" not in MS, "subgroup 'whole-population phenomenon' over-claim removed")
ok("subgroup uncertainty and heterogeneity were not formally assessed" in MS, "subgroup uncertainty/heterogeneity caveat present")
ok("finite-sample variance rather than algorithm failure" not in MS and "finite-sample variance rather than solver failure" not in MS, "MS mechanism over-claim softened")
ok("finite-sample variance rather than algorithm failure" not in SUPP, "S11 supplement mechanism over-claim softened")
# round-14 specifics
ok("calibrated in-sample by construction" not in MS, "P1-5b: 'calibrated in-sample by construction' removed")
ok("apparent calibration close to the ideal" in MS, "P1-5b: development-set calibration reworded present")
ok("almost perfectly separable" not in MS, "P1-7a: 'almost perfectly separable' removed")
ok("strongly distinguishable" in MS, "P1-7a: 'strongly distinguishable' present")
ok("stability-versus-typical-calibration trade-off" in MS, "P1-7b: trade-off reported in main text")
ok("median 1.009" in MS and "0.742" in MS, "P1-7b: median-slope trade-off numbers cited")
ok("stability-versus-typical-calibration trade-off" in SUPP, "P1-7b: trade-off reported in S11 supplement")
ok("recovery of absolute-risk estimates from an existing transported model without re-estimating all predictor coefficients" in MS, "P1-6: recalibration-contribution sentence corrected")
ok("a refitted SOFA model also outputs a probability" in MS, "P1-6: acknowledges score models output probability")
ok("across the ten splits the mean held-out slope was 1.04" in MS and "in the illustrative split ECE decreased from 0.098 to 0.020" in MS, "P1-5c: Discussion separates ten-split slope from illustrative-split ECE")
ok("cluster bootstrap over the ten outer splits" in SUPP and "resampling unit is therefore the outer split" in SUPP, "P1-5a: S7 states resampling algorithm (unit = outer split)")
ok("fixed-predictor refit" in MS and "no feature re-selection" in MS, "P2: fixed-predictor refit defined")
# round-15 specifics
s13seg=SUPP.split("## Table S13")[1]
ok("by construction" not in s13seg, "P1-1: S13 'by construction' removed")
ok("empirically observed rather than guaranteed" in s13seg, "P1-1: S13 empirically-observed wording present")
ok("0.4483" in MS and "same estimand under different resampling procedures" in MS, "P1-2: three weighted-slope values reconciled in main text")
ok("joint empirical study of three linked phenomena" in MS, "P1-3: joint-empirical-study contribution framing present")

# ---------- (8) version/build tags ----------
print("== (8) version / build-tag consistency ==")
ok("# Supplementary Material (analysis build 2026-10-12.r15)" in SUPP, "supplement header r15")
ok("audit_consistency_v15.py" in MS and "audit_consistency_v14.py" not in MS.split("## Declarations")[0], "manuscript references v15 audit, not v14")
ok("20_reproduce_all.py" in MS or "21_reproduce_all_v15.py" in MS, "reproduction driver declared in Code availability")
ok("model_specification.csv" in MS, "manuscript cites model_specification.csv")
ok("scripts/analysis_v5.py" not in MS.split("## Declarations")[0], "P2: main body (pre-Declarations) free of per-script enumeration")

# ---------- (9) TRIPOD v13 fixes ----------
print("== (9) TRIPOD+AI v13 fixes ==")
ok("No formal eligibility/screening flow diagram" in TRI and "not** a participant flow" in TRI.replace("not** a participant flow","not** a participant flow"), "TRIPOD flow-diagram mislabel corrected")
ok("Fig. 1 is ROC/calibration" in TRI, "TRIPOD states Fig 1 is ROC/calibration not flow")
ok("model_specification" in TRI and "verified to reproduce" in TRI, "TRIPOD model-accessibility cites verified spec file")
ok("official TRIPOD+AI checklist PDF" in TRI or "official numbered TRIPOD+AI checklist" in TRI, "TRIPOD instructs transcription onto official numbered checklist")
for it in ["Sample size","Data-splitting","Model updating","Protocol & registration","Patient & public involvement","Ethics","AI statement"]:
    ok(it in TRI, f"TRIPOD covers item: {it}")

# ---------- (10) banned + caps ----------
print("== (10) banned-phrase sweep + caps ==")
for b_ in ["before any result was inspected","events-per-variable below 1","all scores fitted on the same eICU update half",
           "case-mix alone does not explain","paired patient bootstrap","only re-estimation restores ranking",
           "viable statistical repair","whole-population phenomenon","(v10)","audit_consistency_v10.py"]:
    ok(b_ not in MS, f"absent banned: {b_!r}")
ab=re.search(r"## Abstract\n\n(.+?)\n\n---",MS,re.S).group(1)
ws=len(ab.split())
ok(ws<=200, f"abstract words <=200 (={ws}) [SR word count]")
ntab=len(re.findall(r"\*\*Table \d",MS)); nfig=len(re.findall(r"\*\*Fig\. \d",MS.split("## Figure legends")[1]))
ok(ntab+nfig<=8, f"display items <=8 ({ntab}+{nfig})")
main=MS[MS.find("## Introduction"):MS.find("## Methods")]; ok(len(main.split())<=4800, f"main<=4800 (={len(main.split())})")
ok(MS.count("[AUTHOR-SPECIFIC")>=5, f"AUTHOR-SPECIFIC >=5 (={MS.count('[AUTHOR-SPECIFIC')})")

print("\nRESULT:", "PASS" if not fails else f"FAIL ({len(fails)})")
for f in fails: print("   -",f)
sys.exit(0 if not fails else 1)
