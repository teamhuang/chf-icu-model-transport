"""audit_consistency_v11.py — round-11 audit. Locks main Table 2 == supplement S7 (single source),
verifies near-separation wording, abstract length, and re-checks the v10 corrections.
Exit 0 = PASS."""
import os, re, json, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; os.chdir(ROOT)
MS=open("outputs/SCI_manuscript_v11.md",encoding="utf-8").read()   # v11 edits applied in-place to v10 file
SUPP=open("outputs/SUPPLEMENT_v11.md",encoding="utf-8").read()
diag=json.load(open("outputs/sr_diag_v10.json",encoding="utf-8"))["slope_and_events"]
fails=[]
def ok(c,m):
    print(("  OK  " if c else "  FAIL ")+m)
    if not c: fails.append(m)

def num(s): return float(s.replace("−","-"))
# parse main Table 2 rows: | size | events (a–b) | median [q1–q2] (min to max) | ...
t2={}
for line in MS.splitlines():
    mm=re.match(r"\|\s*([\d,]+)\s*\|\s*([\d.]+)\s*\((\d+)[–-](\d+)\)\s*\|\s*(-?[\d.]+)\s*\[", line)
    if mm:
        size=int(mm.group(1).replace(",","")); t2[size]={"events_median":num(mm.group(2)),"slope_median":num(mm.group(5))}
# parse S7 markdown table
s7={}
s7seg=SUPP.split("## Table S7")[1].split("## Table S8")[0]
for line in s7seg.splitlines():
    mm=re.match(r"\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|", line)
    if mm:
        s7[int(mm.group(1))]={"events_median":num(mm.group(2)),"slope_median":num(re.search(r"\|\s*([\d.]+)\s*\|\s*[-\d.]+–[-\d.]+\s*\|", line).group(1)) if re.search(r"\|\s*([\d.]+)\s*\|\s*[-\d.]+–[-\d.]+\s*\|", line) else None}
# more robust S7 slope median: columns are Update_size|events_median|events_min|events_max|epp|slope_median|slope_IQR|...
s7={}
for line in s7seg.splitlines():
    cells=[c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells)>=6 and re.match(r"^\d+$", cells[0]):
        s7[int(cells[0])]={"events_median":num(cells[1]),"epp":num(cells[4]),"slope_median":num(cells[5])}

print("== Table 2 vs S7 vs JSON consistency (single source) ==")
for n in [100,200,500,1000,2000,5000]:
    jm=round(diag[str(n)]["slope_median"],2); je=round(diag[str(n)]["events_median"],1)
    tm=t2.get(n,{}).get("slope_median"); sm=s7.get(n,{}).get("slope_median")
    te=t2.get(n,{}).get("events_median"); se=s7.get(n,{}).get("events_median")
    ok(tm is not None and abs(tm-jm)<0.005, f"n={n} Table2 slope {tm} == JSON {jm}")
    ok(sm is not None and abs(sm-jm)<0.005, f"n={n} S7 slope {sm} == JSON {jm}")
    ok(tm==sm, f"n={n} Table2 slope == S7 slope")
    ok(te is not None and abs(te-je)<0.06 and se is not None and abs(se-je)<0.06, f"n={n} events median Table2 {te}/S7 {se} == JSON {je}")
ok(abs(s7[500]["epp"]-15.9)<0.2 or abs(s7[500]["epp"]-31.8)<0.2, "S7 n=500 events/parameter consistent with events median 63.5 (31.8)")
ok(abs(s7[500]["events_median"]/2 - s7[500]["epp"])<0.06, "S7 n=500 epp == events_median/2 (self-consistent)")

print("== near-separation wording softened ==")
ok("potentially reflecting sparse-data instability or near-separation" in MS, "softened near-separation wording present")
ok("reflects a handful of near-separated fits" not in MS, "definitive 'handful of near-separated fits' removed")
ok("did not run a dedicated separation diagnostic" in MS or "a dedicated separation diagnostic was not run" in SUPP, "no-separation-diagnostic caveat present")

print("== carryover v10 corrections still present ==")
for t in ["events-per-variable below 1" ,"all scores fitted on the same eICU update half","case-mix alone does not explain","paired patient bootstrap","only re-estimation restores ranking","Obermeyer"]:
    ok(t not in MS, f"absent: {t!r}")
for t in ["retrospective cross-database assessment","within-database held-out validation","outer-split resampling interval","stay-level paired bootstrap","Models evaluated on the same held-out eICU sample","cannot determine the contribution of case-mix","illustrative"]:
    ok(t in MS, f"present: {t}")
ok(MS.count("42056904")==1 and "12(7):e059761" in MS and "chronic critical illness" in MS, "reference state retained")

print("== caps ==")
ab=re.search(r"## Abstract\n\n(.+?)\n\n---",MS,re.S).group(1)
ws=len(ab.split()); naive=len(re.split(r"[\s\-/–,;.()]+", ab))
ok(ws<=185, f"abstract whitespace-words <=185 (={ws})")
ok(naive<=200, f"abstract naive-token count <=200 (={naive})")
ntab=len(re.findall(r"\*\*Table \d",MS)); nfig=len(re.findall(r"\*\*Fig\. \d",MS.split("## Figure legends")[1]))
ok(ntab+nfig<=8, f"display items <=8 ({ntab}+{nfig})")
main=MS[MS.find("## Introduction"):MS.find("## Methods")]; ok(len(main.split())<=4500, f"main<=4500 (={len(main.split())})")
ok(MS.count("[AUTHOR-SPECIFIC")>=5, f"AUTHOR-SPECIFIC >=5 (={MS.count('[AUTHOR-SPECIFIC')})")

print("\nRESULT:", "PASS" if not fails else f"FAIL ({len(fails)})")
sys.exit(0 if not fails else 1)
