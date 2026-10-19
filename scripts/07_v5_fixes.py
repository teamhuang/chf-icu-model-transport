"""DEPRECATED — do not use for reported numbers. Retained for provenance only; superseded by the active pipeline in requirements.txt. See scripts/analysis_v5.py for the current implementation.

v5 correction probe: fix the two P0 coding errors flagged in the SR audit.
(1) INVERTED importance weights in 05/06 -> recompute CORRECT-direction IPW and,
    decisively, whether the frozen model's calibration SLOPE recovers toward 1.0
    after re-weighting eICU to the MIMIC-IV covariate distribution.
(2) min-stable-size double-counting -> recompute one value per repetition.
Unpenalised (statsmodels-style) calibration slope/intercept via IRLS to avoid
sklearn L2 shrinkage. Prints a clean summary; does NOT overwrite v4 outputs.
"""
import numpy as np, pandas as pd, collections, os
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
RNG=np.random.RandomState(2026)
COMMON=["Age","male","HeartRate","RespiratoryRate","Temperature","SBP","AG","BUN","Creatinine","Chloride","MCH","MCHC","MCV","Platelet","Potassium","RBC","RDW","Sodium","WBC","COPD","Diabetes"]
def load():
    a=pd.read_csv("data/data01.csv").rename(columns={"heart rate":"HeartRate","Respiratory rate":"RespiratoryRate","temperature":"Temperature","Systolic blood pressure":"SBP","Anion gap":"AG","Urea nitrogen":"BUN","Blood potassium":"Potassium","Blood sodium":"Sodium","Leucocyte":"WBC","Platelets":"Platelet","age":"Age"})
    a["male"]=(a["gendera"]==1).astype(int); a["Diabetes"]=a["diabetes"].astype(int)
    a=a[a["outcome"].notna()].copy(); a["outcome"]=a["outcome"].astype(int)
    b=pd.read_csv("data/Derivation cohorts--MIMIC-IV database.csv"); b["male"]=(b["Gender"]==1).astype(int); b["outcome"]=b["Hstatus"].astype(int)
    v=pd.read_csv("data/Validation cohorts--eICU-CRD.csv"); v["male"]=(v["Gender"]==1).astype(int); v["outcome"]=v["Hstatus"].astype(int)
    return a,b,v
A,B,V=load()
pipe=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("lr",LogisticRegression(max_iter=2000))]).fit(B[COMMON].astype(float),B["outcome"].values)
def lp(df):
    p=pipe.predict_proba(df[COMMON].astype(float))[:,1]
    return np.log(np.clip(p,1e-9,1-1e-9)/np.clip(1-p,1e-9,1-1e-9)), p

def unpen_calib(y,l,w=None):
    # IRLS logistic with only an intercept+slope on l, no L2 penalty; optional weights
    X=np.column_stack([np.ones(len(l)),l]); y=y.astype(float)
    if w is None: w=np.ones(len(l))
    beta=np.zeros(2)
    for _ in range(50):
        eta=X@beta; p=1/(1+np.exp(-eta)); Wd=w*p*(1-p)
        g=X.T@(w*(y-p)); H=X.T@(X*Wd[:,None])
        step=np.linalg.solve(H+1e-8*np.eye(2),g); beta+=step
        if np.max(np.abs(step))<1e-9: break
    return beta[1],beta[0]   # slope, intercept

yV=V["outcome"].values; lpV,prV=lp(V)
print("== Frozen eICU (unweighted) ==")
s0,i0=unpen_calib(yV,lpV)
print("  slope=%.3f intercept=%+.3f Brier=%.3f OE=%.3f"%(s0,i0,brier_score_loss(yV,prV),yV.sum()/prV.sum()))

X=pd.concat([B[COMMON].astype(float),V[COMMON].astype(float)],ignore_index=True)
isV=np.r_[np.zeros(len(B)),np.ones(len(V))]
clf=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("lr",LogisticRegression(max_iter=2000))]).fit(X,isV)
ps=clf.predict_proba(V[COMMON].astype(float))[:,1]
piV=len(V)/(len(V)+len(B)); piB=1-piV
w_inv=np.clip((ps/(1-ps))*(piB/piV),0.1,10)      # v4 (inverted) 
w_cor=np.clip(((1-ps)/ps)*(piV/piB),0.1,10)       # CORRECT: reweight eICU toward MIMIC-IV
def smd_mean(w):
    xV=V[COMMON].astype(float).values; xB=B[COMMON].astype(float).values
    out=[]
    for j in range(xV.shape[1]):
        v1,v0=xV[:,j],xB[:,j]
        m1=np.average(v1,weights=w); var1=np.average((v1-m1)**2,weights=w); m0=np.mean(v0); var0=np.var(v0,ddof=1)
        out.append(abs(m1-m0)/np.sqrt((var1+var0)/2))
    return np.mean(out)
for tag,w in [("v4-inverted",w_inv),("v5-correct",w_cor)]:
    ess=w.sum()**2/(w**2).sum(); s,i=unpen_calib(yV,lpV,w)
    wb=np.average((prV-yV)**2,weights=w); oe=(w*yV).sum()/(w*prV).sum()
    print("== weights %s == ESS=%.1f%% mean|SMD|->MIMICIV=%.3f  weighted slope=%.3f int=%+.3f  weighted OE=%.3f Brier=%.3f"
          %(tag,100*ess/len(w),smd_mean(w),s,i,oe,wb))
print("\nINTERPRETATION: correct weights reach balance (<0.1). Does slope recover to ~1?")
print("  If weighted slope (correct) still << 1  -> drift NOT purely case-mix (claim survives, correctly).")
print("  If weighted slope -> ~1                  -> drift IS case-mix driven (claim must be retracted).")

# ---- (2) min-stable-size correct ----
r=pd.read_csv("outputs/v4_repeated_splits.csv")
def first_all_stable(df):
    for n in [100,200,500,1000,2000,5000]:
        sub=df[df.n_pat==n]
        if len(sub)>0 and sub.stable.all(): return n
    return None
cnt=collections.Counter(first_all_stable(r[r.rep==rep]) for rep in sorted(r.rep.unique()))
print("\nCorrect min-stable per rep (sums to 10):",dict(cnt))
