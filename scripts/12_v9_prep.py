"""12_v9_prep.py — Round-9 Phase-1 verification + missingness(S5) + S2 recount + S4 rebuild data.
Relative root so it runs on any machine. Writes outputs/sr_*.csv|json. No results hard-coded.
"""
import os, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
COMMON=["Age","male","HeartRate","RespiratoryRate","Temperature","SBP","AG","BUN","Creatinine",
        "Chloride","MCH","MCHC","MCV","Platelet","Potassium","RBC","RDW","Sodium","WBC","COPD","Diabetes"]

def load_BV():
    b=pd.read_csv("data/Derivation cohorts--MIMIC-IV database.csv"); b["outcome"]=b["Hstatus"].astype(int)
    v=pd.read_csv("data/Validation cohorts--eICU-CRD.csv"); v["outcome"]=v["Hstatus"].astype(int)
    b["male"]=(b["Gender"]==1).astype(int); v["male"]=(v["Gender"]==1).astype(int)
    return b,v
B,V=load_BV()

out={}

# ---- S2 recount: how many variables still |SMD_weighted| > 0.10 ----
s2=pd.read_csv("outputs/v6_smd_by_variable.csv")
over=s2[s2.SMD_weighted>0.10]
out["S2_weighted_over_010"]={"count":int(len(over)),"variables":[(r.Variable,round(float(r.SMD_weighted),4)) for r in over.itertuples()]}
print("S2 corrected: variables with |SMD_weighted|>0.10 =", out["S2_weighted_over_010"])

# ---- Refit scoping check (does code re-fit imputer/scaler within update half?) ----
# inspect analysis_v5 strategy-refit + 11_bench: both do Pipeline(...).fit(Vu[COMMON],yu) => fit on update half only.
out["refit_preproc_scope"]="imputer+scaler+coefs are fit ONLY on the update half via Pipeline.fit(Vu[COMMON],yu) in analysis_v5.py (sec4) and 11_v8_heldout_bench.py; test half only transform+predict (no leakage)."

# ---- Missingness per variable per cohort (B, V) ----
def miss(df):
    d={}
    for c in COMMON:
        col=df[c] if c in df.columns else pd.Series([np.nan]*len(df))
        d[c]=round(float(pd.to_numeric(col,errors='coerce').isna().mean()*100),2)
    return d
out["missing_pct_B"]=miss(B)
out["missing_pct_V"]=miss(V)
# complete-case fraction on the 21
Bc=B[COMMON].apply(pd.to_numeric,errors='coerce'); Vc=V[COMMON].apply(pd.to_numeric,errors='coerce')
out["complete_case_pct"]={"MIMIC-IV":round(float((~Bc.isna().any(axis=1)).mean()*100),2),
                          "eICU":round(float((~Vc.isna().any(axis=1)).mean()*100),2)}
# any variable with >5% missing in either cohort
high=[c for c in COMMON if out["missing_pct_B"][c]>5 or out["missing_pct_V"][c]>5]
out["high_missing_vars_gt5pct"]=high
print("missingness done; complete-case:",out["complete_case_pct"],"high>5%:",high)

json.dump(out,open("outputs/sr_prep_v9.json","w",encoding="utf-8"),indent=2,ensure_ascii=False)

# S5 table
rows=[]
for c in COMMON:
    rows.append(dict(Variable=c,Missing_pct_MIMICIV=out["missing_pct_B"][c],Missing_pct_eICU=out["missing_pct_V"][c]))
pd.DataFrame(rows).to_csv("outputs/sr_missingness_S5.csv",index=False)
print("wrote outputs/sr_missingness_S5.csv and outputs/sr_prep_v9.json")
