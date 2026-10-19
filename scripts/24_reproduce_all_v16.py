"""24_reproduce_all_v16.py — round-16 clean-room reproducibility driver.

Runs the FULL analysis pipeline from the raw data/ files in dependency order, then
regenerates every Nature-styled figure (Fig. 1 roadmap + Fig. 2-5 + Fig. S2/S3),
snapshots each artefact's SHA-256 before and after a second pass to prove determinism
(seed 2026), then runs audit_consistency_v16.py to confirm the manuscript's Tables 1-4 /
S1-S13 / figure numbering / abstract numbers match the freshly regenerated outputs.

Figure determinism note: PNG outputs are byte-stable and are hashed; the editable PDF
outputs embed a matplotlib /CreationDate timestamp and therefore vary by design, so they
are checked for presence and clean regeneration but NOT byte-hashed.

Writes outputs/reproduction_report.json. Exit 0 = reproducible AND consistent.
"""
import os, sys, json, hashlib, subprocess, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; os.chdir(ROOT)
PY=sys.executable

GEN=["analysis_v5.py","08_v6.py","09_v7_inference.py","11_v8_heldout_bench.py",
     "13_v9_sensitivity.py","15_v10_diagnostics.py","17_v12_diagnostics.py","19_v13_model_spec.py",
     "22_v15_fig1_roadmap.py","23_v15_data_figs.py"]
ART=["outputs/v5_results.json","outputs/v5_repeated_curve.csv","outputs/v5_strategy.csv",
     "outputs/v6_results.json","outputs/v7_inference.json","outputs/sr_sens_v9.json",
     "outputs/sr_diag_v10.json","outputs/sr_v12.json","outputs/model_specification.csv",
     "outputs/model_specification.json","outputs/multivar_logit.csv"]
FIGPNG=["figures/v15_fig1_roadmap.png","figures/v15_fig2_transport.png","figures/v15_fig3_stability.png",
        "figures/v15_fig4_strategies.png","figures/v15_fig5_reweighting.png",
        "figures/v15_figS2_slope_reg.png","figures/v15_figS3_tails_trunc.png"]
FIGPDF=[p.replace(".png",".pdf") for p in FIGPNG]

def sha(p):
    if not os.path.exists(p): return None
    return hashlib.sha256(open(p,"rb").read()).hexdigest()[:16]

def run(script):
    t=time.time()
    r=subprocess.run([PY,os.path.join("scripts",script)],capture_output=True,text=True)
    return r.returncode, round(time.time()-t,1), (r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "")

report={"seed":2026,"python":os.path.basename(PY),"pass1":{},"pass2":{},"hash_stable":{},"figures":{},"audit":None}

print("=== PASS 1: run generators + figures from raw data ===")
for s in GEN:
    code,secs,err=run(s)
    report["pass1"][s]={"exit":code,"secs":secs,"err":err[:160]}
    print(f"  {'OK ' if code==0 else 'ERR'} {s} (exit {code}, {secs}s) {('  '+err[:80]) if code!=0 else ''}")
snap1={a:sha(a) for a in ART}; fig1={p:sha(p) for p in FIGPNG}

print("=== PASS 2: re-run to test determinism ===")
for s in GEN:
    code,secs,err=run(s)
    report["pass2"][s]={"exit":code,"secs":secs}
snap2={a:sha(a) for a in ART}; fig2={p:sha(p) for p in FIGPNG}
for a in ART:
    stable=(snap1[a]==snap2[a]) and snap1[a] is not None
    report["hash_stable"][a]={"hash":snap1[a],"stable":stable}
    print(f"  {'STABLE' if stable else 'CHANGED/MISSING':16s} {a}  {snap1[a]}")
print("--- figure PNGs (byte-hashed) + PDFs (presence only) ---")
for p in FIGPNG:
    stable=(fig1[p]==fig2[p]) and fig1[p] is not None
    pdf_ok=os.path.exists(p.replace(".png",".pdf"))
    report["figures"][p]={"png_hash":fig1[p],"png_stable":stable,"pdf_present":pdf_ok}
    print(f"  {'STABLE' if stable else 'CHANGED':8s} png {os.path.basename(p)}  pdf={'yes' if pdf_ok else 'MISSING'}")

print("=== AUDIT: manuscript vs regenerated outputs ===")
r=subprocess.run([PY,os.path.join("scripts","audit_consistency_v16.py")],capture_output=True,text=True)
tail=r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
report["audit"]={"exit":r.returncode,"last_line":tail}
print("  ",tail,f"(exit {r.returncode})")

all_gen_ok=all(report["pass1"][s]["exit"]==0 for s in GEN)
all_stable=all(v["stable"] for v in report["hash_stable"].values())
figs_ok=all(v["png_stable"] and v["pdf_present"] for v in report["figures"].values())
audit_ok=report["audit"]["exit"]==0
report["verdict"]="REPRODUCIBLE-AND-CONSISTENT" if (all_gen_ok and all_stable and figs_ok and audit_ok) else "PROBLEM"
json.dump(report,open("outputs/reproduction_report.json","w",encoding="utf-8"),indent=2,ensure_ascii=False)
print("\nVERDICT:",report["verdict"],"| generators_ok:",all_gen_ok,"| data deterministic:",all_stable,"| figures deterministic:",figs_ok,"| audit_pass:",audit_ok)
sys.exit(0 if report["verdict"]=="REPRODUCIBLE-AND-CONSISTENT" else 1)
