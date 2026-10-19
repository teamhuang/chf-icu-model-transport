"""audit_consistency_v17.py — round-17 audit (SR submission build: reduced defensive
language, references renumbered by order of appearance in Nature/SR style and verified
online, figures/tables numbered by appearance, figures embedded, Word output produced).

Keeps every numeric lock from v16 (Table 2 = S7 = S11 = JSON; Table 4 = S13 = JSON;
subgroup = JSON; model-spec betas == sklearn L2) and adds v17-specific checks:
  * no leftover [[..]] citation tokens or @@REFERENCES@@ marker;
  * exactly 19 references, numbered 1..19 in strict order of first appearance;
  * the unverifiable old "Steyerberg systematic review of updating methods" is GONE,
    Davis (nonparametric drift correction) and the verified medRxiv preprint are present;
  * figures and tables appear in ascending order (Fig 1 first; Fig 1..5, Table 1..4);
  * every embedded figure file exists; the .docx submission exists and embeds 5 images;
  * the oversized 'Scope statement.' block was folded into concise prose (defensive
    language reduced) while the essential honesty markers remain;
  * abstract <= 200 words; >= 5 [AUTHOR-SPECIFIC] placeholders preserved.
Exit 0 = PASS.
"""
import os, re, json, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
ROOT=Path(__file__).resolve().parents[1]; os.chdir(ROOT)
MS=open("outputs/SCI_manuscript_v17.md",encoding="utf-8").read()
SUPP=open("outputs/SUPPLEMENT_v17.md",encoding="utf-8").read()
TRI=open("outputs/SUPPLEMENT_TRIPOD_AI_v17.md",encoding="utf-8").read()
diag=json.load(open("outputs/sr_diag_v10.json",encoding="utf-8"))["slope_and_events"]
v12=json.load(open("outputs/sr_v12.json",encoding="utf-8"))
fails=[]
def ok(c,m):
    print(("  OK  " if c else "  FAIL ")+m)
    if not c: fails.append(m)
def num(s): return float(str(s).replace("−","-"))

# ---------- numeric locks (as v16) ----------
t2={}
for line in MS.splitlines():
    mm=re.match(r"\|\s*([\d,]+)\s*\|\s*([\d.]+)\s*\((\d+)[–-](\d+)\)\s*\|\s*(-?[\d.]+)\s*\[(-?[\d.]+)[–-](-?[\d.]+)\]\s*\(([-−]?[\d.]+)\s*to\s*([-−]?[\d.]+)\)", line)
    if mm:
        size=int(mm.group(1).replace(",",""))
        t2[size]={"events_median":num(mm.group(2)),"slope_median":num(mm.group(5)),"slope_max":num(mm.group(9))}
s7={}; s7seg=SUPP.split("## Table S7")[1].split("## Table S8")[0]
for line in s7seg.splitlines():
    cells=[c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells)>=6 and re.match(r"^\d+$", cells[0]):
        s7[int(cells[0])]={"events_median":num(cells[1]),"slope_median":num(cells[5])}
s11={}; s11seg=SUPP.split("## Table S11")[1].split("## Table S12")[0]
for line in s11seg.splitlines():
    cells=[c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells)>=10 and re.match(r"^\d+$", cells[0]):
        s11[int(cells[0])]={"unpen_med":num(cells[1]),"unpen_max":num(cells[2]),"rob_med":num(cells[4])}
print("== (1) Table 2 == S7 == S11 == JSON ==")
ok(len(t2)==6, f"Table 2 parsed 6 sizes (got {len(t2)})")
for n in [100,200,500,1000,2000,5000]:
    jm=round(diag[str(n)]["slope_median"],2); jmax=round(diag[str(n)]["slope_max"],2); je=round(diag[str(n)]["events_median"],1)
    ok(abs(t2[n]["slope_median"]-jm)<0.005 and abs(s7[n]["slope_median"]-jm)<0.005, f"n={n} slope Table2/S7 == JSON {jm}")
    ok(abs(s11[n]["unpen_med"]-jm)<0.005 and abs(s11[n]["unpen_max"]-jmax)<0.02, f"n={n} S11 unpen med/max == JSON")
    ok(abs(t2[n]["events_median"]-je)<0.06, f"n={n} events Table2 == JSON {je}")
# S11 penalised medians cited in text (1.009->0.742, 1.059->0.904)
ok("1.009" in MS and "0.742" in MS and "1.059" in MS and "0.904" in MS, "regularisation trade-off medians cited")

# Table 4 == S13
print("== (2) Table 4 == S13 == JSON ==")
t4seg=MS.split("**Table 4.")[1].split("### ")[0]; t4={}
for line in t4seg.splitlines():
    cells=[c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells)==5 and re.match(r"^\d?\.\d", cells[1]): t4[cells[0]]={"slope":num(cells[3]),"oe":num(cells[4])}
jt={r["Model"]:r for r in v12["table4_with_calibration"]}
ok(len(t4)==6, f"Table 4 parsed 6 models (got {len(t4)})")
for mn,sn in [("Frozen 21-variable (transported)","Frozen 21-var"),("Two-parameter recalibration","Two-parameter recalibration"),("SOFA only (refit on update half)","SOFA (refit)"),("APS-III only (refit on update half)","APS-III (refit)"),("SOFA + APS-III (refit on update half)","SOFA+APS-III (refit)"),("21-variable refit (update half)","21-var refit")]:
    j=jt[sn]; ok(abs(t4[mn]["slope"]-round(j["slope"],2))<0.005 and abs(t4[mn]["oe"]-round(j["OE"],2))<0.005, f"T4 {sn} slope/OE == JSON")

# model-spec betas == sklearn L2
print("== (3) model_specification betas == sklearn L2 ==")
COMMON=["Age","male","HeartRate","RespiratoryRate","Temperature","SBP","AG","BUN","Creatinine","Chloride","MCH","MCHC","MCV","Platelet","Potassium","RBC","RDW","Sodium","WBC","COPD","Diabetes"]
spec=pd.read_csv("outputs/model_specification.csv")
b=pd.read_csv("data/Derivation cohorts--MIMIC-IV database.csv"); b["male"]=(b["Gender"]==1).astype(int); b["outcome"]=b["Hstatus"].astype(int)
X=b[COMMON].astype(float); imp=SimpleImputer(strategy="median").fit(X); sc=StandardScaler().fit(imp.transform(X))
lr=LogisticRegression(max_iter=2000).fit(sc.transform(imp.transform(X)), b["outcome"].values)
skmap=dict(zip(COMMON,lr.coef_[0])); sd=pd.Series(dict(zip(spec["variable"],spec["beta_standardised"])))
ok(max(abs(skmap[v]-sd[v]) for v in COMMON)<1e-4, "spec betas match sklearn L2")
ok("−2.339" in MS or "-2.339" in MS, "intercept −2.339 cited")

# ---------- (4) references: appearance order + SR style + verification ----------
print("== (4) references (order of appearance, SR/Nature style, verified) ==")
ok("[[" not in MS and "@@REFERENCES@@" not in MS, "no leftover [[tokens]] or @@REFERENCES@@ marker")
refs_sec=MS.split("## References")[1].split("## Acknowledgements")[0]
refnums=[int(m.group(1)) for m in re.finditer(r"(?m)^(\d+)\.\s", refs_sec)]
ok(refnums==list(range(1,len(refnums)+1)), f"reference list is sequential 1..N (got {refnums})")
ok(len(refnums)==19, f"exactly 19 references (got {len(refnums)})")
# first-appearance of «..» in body must be ascending 1..19
body=MS[:MS.find("## References")]
seen=[]
for m in re.finditer(r"\u00ab([^\u00bb]+)\u00bb", body):
    for tok in m.group(1).split(","):
        if "–" in tok:
            a,b2=tok.split("–")
            for x in range(int(a),int(b2)+1):
                if x not in seen: seen.append(x)
        else:
            x=int(tok)
            if x not in seen: seen.append(x)
ok(seen==list(range(1,len(seen)+1)), f"citations first appear in ascending order 1..N (got {seen})")
ok("Updating methods for improving clinical prediction models: a systematic review" not in MS, "unverifiable old Steyerberg 'updating systematic review' dropped")
ok("nonparametric updating method to correct clinical prediction model drift" in MS, "Davis nonparametric drift-correction ref present (verified)")
ok("medRxiv" in MS and "10.64898/2026.05.03.26352335" in MS, "verified medRxiv preprint (Patel & Beedala) present")
ok("TRIPOD+AI statement" in MS and "Calibration: the Achilles heel" in MS, "core method refs present")

# ---------- (5) figures/tables by appearance + embedded ----------
print("== (5) figures/tables ordered by appearance; figures embedded ==")
def first_pos(pat): 
    m=re.search(pat, body); return m.start() if m else 10**9
figpos=[(k,first_pos(rf"Fig\. {k}(?![0-9])")) for k in range(1,6)]
ok(all(figpos[i][1]<figpos[i+1][1] for i in range(4)), f"Figs 1..5 first-appear ascending {[(k,p) for k,p in figpos]}")
tabpos=[(k,first_pos(rf"\*\*Table {k}\.")) for k in range(1,5)]
ok(all(tabpos[i][1]<tabpos[i+1][1] for i in range(3)), f"Tables 1..4 first-appear ascending {[(k,p) for k,p in tabpos]}")
for f in ["v17_fig1_roadmap","v15_fig2_transport","v15_fig3_stability","v15_fig4_strategies","v15_fig5_reweighting"]:
    ok(os.path.exists(f"figures/{f}.png"), f"embedded figure exists: figures/{f}.png")
# PNG submission folder
for f in ["Figure1","Figure2","Figure3","Figure4","Figure5","FigureS1","FigureS2","FigureS3"]:
    ok(os.path.exists(f"PNG/{f}.png"), f"PNG folder has {f}.png")
ok(os.path.exists("PNG/图例.txt") and "Figure 1" in open("PNG/图例.txt",encoding="utf-8").read() and "Supplementary Figure S3" in open("PNG/图例.txt",encoding="utf-8").read(), "PNG/图例.txt has all figure legends")
# docx
ok(os.path.exists("outputs/SCI_manuscript_v17.docx"), "Word .docx produced")
if os.path.exists("outputs/SCI_manuscript_v17.docx"):
    z=zipfile.ZipFile("outputs/SCI_manuscript_v17.docx"); x=z.read("word/document.xml").decode("utf-8")
    nmedia=len([n for n in z.namelist() if n.startswith("word/media/") and n.endswith(".png")])
    ok(nmedia==5, f"docx embeds 5 figures (got {nmedia})")
    ok("superscript" in x, "docx uses superscript citations")
    ok("[missing image" not in x, "docx has no missing-image placeholders")
    ok(x.count("<w:tbl>")>=4, f"docx has >=4 tables (got {x.count('<w:tbl>')})")

# ---------- (6) reduced defensive language but honesty retained ----------
print("== (6) prose: defensive language reduced, essential honesty kept ==")
ok("**Scope statement.**" not in MS, "oversized standalone 'Scope statement.' block folded into concise prose")
for keep in ["could not be verified","at-admission tool","within-eICU","a new external validation","cannot determine the contribution of case-mix"]:
    ok(keep in MS, f"honesty marker retained: {keep!r}")
main=MS[MS.find("## Introduction"):MS.find("## Methods")]
ok(len(main.split())<=4200, f"main text tightened (<=4200 words; got {len(main.split())})")

# ---------- (7) version/build + caps ----------
print("== (7) version/build tags + caps ==")
ok("# Supplementary Material (analysis build 2026-10-12.r17)" in SUPP, "supplement header r17")
ok("analysis build 2026-10-12.r17" in TRI, "TRIPOD header r17")
ok("25_finalize_v17.py" in MS, "v17 finalize script declared in Code availability")
ok("audit_consistency_v16.py" in MS or "audit_consistency_v17.py" in MS, "audit declared")
ab=re.search(r"## Abstract\n\n(.+?)\n\n---",MS,re.S).group(1); ws=len(ab.split())
ok(ws<=200, f"abstract <=200 words (={ws})")
ok(MS.count("[AUTHOR-SPECIFIC")>=5, f"AUTHOR-SPECIFIC >=5 (={MS.count('[AUTHOR-SPECIFIC')})")
ntab=len(re.findall(r"\*\*Table \d",MS)); nfig=len(re.findall(r"\*\*Fig\. \d \|",MS))
ok(ntab==4, f"4 main tables (got {ntab})")
ok(nfig==5, f"5 main figure legends (got {nfig})")

print("\nRESULT:", "PASS" if not fails else f"FAIL ({len(fails)})")
for f in fails: print("   -",f)
sys.exit(0 if not fails else 1)
