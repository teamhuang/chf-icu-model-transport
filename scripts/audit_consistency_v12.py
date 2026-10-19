"""audit_consistency_v12.py — round-12 audit.
Locks: (1) main Table 2 == S7 == sr_diag_v10.json (single source, carried over);
(2) NEW S11 unpenalised column == Table 2/S7 slope median & max (same update-half fit);
(3) NEW S11 events median == Table 2 events median;
(4) NEW main Table 4 slope/OE == S13 == sr_v12.json table4_with_calibration;
(5) NEW S12 subgroup == sr_v12.json subgroup_frozen_eICU;
(6) version-consistency: supplement header build tag, no stale "(v10)" title, audit script self-reference v12;
(7) new title, abstract/Discussion de-promotional wording, ethics/COI placeholders,
    registration + PPI statements, Fig 3a strategy file, mapping tables S9/S10 present;
(8) banned-phrase sweep + caps. Exit 0 = PASS.
"""
import os, re, json, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; os.chdir(ROOT)
MS=open("outputs/SCI_manuscript_v12.md",encoding="utf-8").read()
SUPP=open("outputs/SUPPLEMENT_v12.md",encoding="utf-8").read()
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
        t2[size]={"events_median":num(mm.group(2)),"slope_median":num(mm.group(5)),
                  "slope_min":num(mm.group(8)),"slope_max":num(mm.group(9))}
print("== parse Table 2 rows ==")
ok(len(t2)==6, f"Table 2 parsed 6 sizes (got {len(t2)})")

# ---------- parse S7 ----------
s7={}
s7seg=SUPP.split("## Table S7")[1].split("## Table S8")[0]
for line in s7seg.splitlines():
    cells=[c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells)>=6 and re.match(r"^\d+$", cells[0]):
        s7[int(cells[0])]={"events_median":num(cells[1]),"epp":num(cells[4]),"slope_median":num(cells[5])}

# ---------- parse S11 ----------
s11={}
s11seg=SUPP.split("## Table S11")[1].split("## Table S12")[0]
for line in s11seg.splitlines():
    cells=[c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells)>=10 and re.match(r"^\d+$", cells[0]):
        s11[int(cells[0])]={"unpen_med":num(cells[1]),"unpen_max":num(cells[2]),"unpen_neg":int(cells[3]),
                            "rob_max":num(cells[5]),"rob_neg":int(cells[6]),
                            "n_ext":int(cells[7]),"ev_all":num(cells[9])}

# ---------- (1) Table 2 == S7 == JSON ----------
print("== (1) Table 2 vs S7 vs sr_diag_v10.json (single source) ==")
for n in [100,200,500,1000,2000,5000]:
    jm=round(diag[str(n)]["slope_median"],2); je=round(diag[str(n)]["events_median"],1)
    ok(abs(t2[n]["slope_median"]-jm)<0.005, f"n={n} Table2 slope {t2[n]['slope_median']} == JSON {jm}")
    ok(abs(s7[n]["slope_median"]-jm)<0.005, f"n={n} S7 slope {s7[n]['slope_median']} == JSON {jm}")
    ok(abs(t2[n]["events_median"]-je)<0.06 and abs(s7[n]["events_median"]-je)<0.06, f"n={n} events median Table2/S7 == JSON {je}")

# ---------- (2) S11 unpenalised == Table 2/S7 ----------
print("== (2) S11 unpenalised column reproduces Table 2/S7 (same update-half fit) ==")
for n in [100,200,500,1000,2000,5000]:
    jm=round(diag[str(n)]["slope_median"],2); jmax=round(diag[str(n)]["slope_max"],2)
    ok(abs(s11[n]["unpen_med"]-jm)<0.005, f"n={n} S11 unpen median {s11[n]['unpen_med']} == JSON {jm}")
    ok(abs(s11[n]["unpen_max"]-jmax)<0.02, f"n={n} S11 unpen max {s11[n]['unpen_max']} == JSON {jmax}")
    ok(s11[n]["unpen_neg"]==diag[str(n)]["n_negative_slope"], f"n={n} S11 unpen neg {s11[n]['unpen_neg']} == JSON {diag[str(n)]['n_negative_slope']}")
    ok(s11[n]["rob_max"] < s11[n]["unpen_max"] or n>=1000, f"n={n} robust max {s11[n]['rob_max']} damps unpen max {s11[n]['unpen_max']}")

# ---------- (3) S11 events median == Table 2 ----------
print("== (3) S11 events median == Table 2 events median ==")
for n in [100,200,500,1000,2000,5000]:
    je=round(diag[str(n)]["events_median"],1)
    ok(abs(s11[n]["ev_all"]-je)<0.06, f"n={n} S11 events {s11[n]['ev_all']} == Table2/JSON {je}")

# ---------- (4) Table 4 slope/OE == S13 == JSON ----------
print("== (4) main Table 4 slope/OE == S13 == sr_v12.json ==")
t4map={}
t4seg=MS.split("**Table 4.")[1].split("### ")[0]
for line in t4seg.splitlines():
    cells=[c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells)==5 and re.match(r"^\d?\.\d", cells[1]):
        t4map[cells[0]]={"auc":num(cells[1]),"brier":num(cells[2]),"slope":num(cells[3]),"oe":num(cells[4])}
ok(len(t4map)==6, f"Table 4 parsed 6 models (got {len(t4map)})")
s13={}
s13seg=SUPP.split("## Table S13")[1]
for line in s13seg.splitlines():
    cells=[c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells)==5 and re.match(r"^\d", cells[1]):
        s13[cells[0]]={"auc":num(cells[1]),"brier":num(cells[2]),"slope":num(cells[3]),"oe":num(cells[4])}
jt={r["Model"]:r for r in v12["table4_with_calibration"]}
pairs=[("Frozen 21-variable (transported)","Frozen 21-var"),
       ("Two-parameter recalibration","Two-parameter recalibration"),
       ("SOFA only (refit on update half)","SOFA (refit)"),
       ("APS-III only (refit on update half)","APS-III (refit)"),
       ("SOFA + APS-III (refit on update half)","SOFA+APS-III (refit)"),
       ("21-variable refit (update half)","21-var refit")]
for mname,sname in pairs:
    j=jt[sname]
    ok(abs(t4map[mname]["slope"]-round(j["slope"],2))<0.005, f"T4 {sname} slope {t4map[mname]['slope']} == JSON {round(j['slope'],2)}")
    ok(abs(t4map[mname]["oe"]-round(j["OE"],2))<0.005, f"T4 {sname} O/E {t4map[mname]['oe']} == JSON {round(j['OE'],2)}")
    ok(abs(s13[sname]["slope"]-j["slope"])<0.0005 and abs(s13[sname]["oe"]-j["OE"])<0.0005, f"S13 {sname} slope/OE == JSON (unrounded)")

# ---------- (5) S12 subgroup == JSON ----------
print("== (5) S12 subgroup == sr_v12.json ==")
sg={r["subgroup"]:r for r in v12["subgroup_frozen_eICU"]}
s12seg=SUPP.split("## Table S12")[1].split("## Table S13")[0]
s12rows={}
for line in s12seg.splitlines():
    cells=[c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells)==6 and cells[0].startswith(("Sex","Age")):
        s12rows[cells[0]]={"n":int(cells[1]),"deaths":int(cells[2]),"auc":num(cells[3]),"slope":num(cells[4]),"oe":num(cells[5])}
ok(len(s12rows)==5, f"S12 parsed 5 subgroups (got {len(s12rows)})")
for k,r in sg.items():
    ok(k in s12rows and abs(s12rows[k]["auc"]-r["AUC"])<0.0005 and abs(s12rows[k]["slope"]-r["slope"])<0.0005, f"S12 {k} AUC/slope == JSON")

# ---------- (6) version consistency ----------
print("== (6) version / build-tag consistency ==")
ok("# Supplementary Material (analysis build 2026-10-12.r12)" in SUPP, "supplement header uses build tag (not '(v10)')")
ok("(v10)" not in SUPP.split("\n")[0] and "Supplementary Material (v10)" not in SUPP, "no stale '(v10)' supplement title")
ok("audit_consistency_v12.py" in MS, "manuscript references audit_consistency_v12.py")
ok("audit_consistency_v10.py" not in MS, "no stale audit_consistency_v10.py reference in manuscript")
ok("17_v12_diagnostics.py" in MS and "18_v12_supp_patch.py" in MS, "v12 scripts declared in Code availability")

# ---------- (7) content requirements ----------
print("== (7) round-12 content requirements ==")
ok(MS.startswith("# Cross-database evaluation and recalibration of a refitted mortality prediction model"), "new title in use")
ok("viable statistical repair" not in MS, "abstract phrase 'viable statistical repair' removed")
ok("cheap, effective *statistical repair" not in MS, "Discussion promotional 'statistical repair' removed")
ok("not constrained to be positive" in MS, "slope-not-constrained-positive note present")
ok("not pre-registered" in MS, "no-pre-registration statement present")
ok("Patient and public involvement" in MS and "No patients or members of the public" in MS, "PPI-none statement present")
ok("AUTHOR-SPECIFIC: insert the institutional-review" in MS, "ethics determination is author-specific (not self-declared 'not required')")
ok("confirmed by **all** named authors" in MS, "COI requires all-author confirmation")
ok("sr_fig_strategies.png" in MS and "fig13_strategy_dca" not in MS, "Fig 3a points to strategy figure, not DCA")
ok("## Table S9" in SUPP and "## Table S10" in SUPP, "provenance + estimate-mapping tables present")
ok("## Table S11" in SUPP and "## Table S12" in SUPP and "## Table S13" in SUPP, "S11-S13 present")
ok("within-eICU stay-level held-out evaluation" in MS, "stay-level held-out scoping retained")
ok("empirical" in MS.lower() and "target-attainment" in MS, "empirical target-attainment curve naming present")
ok("fixed illustrative split" in MS, "fixed illustrative split wording present")

# ---------- (8) banned phrases ----------
print("== (8) banned-phrase sweep ==")
banned=["before any result was inspected","chosen before any result","events-per-variable below 1",
        "all scores fitted on the same eICU update half","case-mix alone does not explain",
        "paired patient bootstrap","only re-estimation restores ranking","The honest conclusion",
        "load-bearing","heterogeneity survived","was protective","EJE","Running head",
        "Significance statement","reviewer requests","v11-SR","v10-SR","(14 words)"]
for b in banned:
    ok(b not in MS, f"absent banned: {b!r}")

# ---------- (9) caps ----------
print("== (9) caps ==")
ab=re.search(r"## Abstract\n\n(.+?)\n\n---",MS,re.S).group(1)
ws=len(ab.split()); naive=len(re.split(r"[\s\-/–,;.()]+", ab))
ok(ws<=200, f"abstract whitespace-words <=200 (={ws})  [SR word count]")
ok(naive<=235, f"abstract naive-token loose bound <=235 (={naive})  [splits hyphens/slashes; not the journal metric]")
ntab=len(re.findall(r"\*\*Table \d",MS)); nfig=len(re.findall(r"\*\*Fig\. \d",MS.split("## Figure legends")[1]))
ok(ntab+nfig<=8, f"display items <=8 ({ntab}+{nfig})")
main=MS[MS.find("## Introduction"):MS.find("## Methods")]; ok(len(main.split())<=4800, f"main<=4800 (={len(main.split())})")
ok(MS.count("[AUTHOR-SPECIFIC")>=5, f"AUTHOR-SPECIFIC >=5 (={MS.count('[AUTHOR-SPECIFIC')})")

print("\nRESULT:", "PASS" if not fails else f"FAIL ({len(fails)})")
for f in fails: print("   -",f)
sys.exit(0 if not fails else 1)
