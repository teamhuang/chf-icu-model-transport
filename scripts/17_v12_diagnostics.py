"""17_v12_diagnostics.py — Round-12 computations (real data, seed 2026, relative root).
(a) penalised (ridge) recalibration sensitivity at small n + extreme-slope vs low-events association [P1-3]
(b) Table 4 held-out calibration slope and O/E for every model [P1-5]
(c) subgroup (sex, age tertile) discrimination/calibration of the frozen model on eICU [TRIPOD fairness]
(d) Fig 3a strategy-comparison figure (no DCA)
Writes outputs/sr_v12.json + figures/sr_fig_strategies.png.
"""
import os, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer; from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss
ROOT=Path(__file__).resolve().parents[1]; os.chdir(ROOT)
COMMON=["Age","male","HeartRate","RespiratoryRate","Temperature","SBP","AG","BUN","Creatinine","Chloride","MCH","MCHC","MCV","Platelet","Potassium","RBC","RDW","Sodium","WBC","COPD","Diabetes"]
SIZES=[100,200,500,1000,2000,5000]
def lg(p): return np.log(np.clip(p,1e-9,1-1e-9)/np.clip(1-p,1e-9,1-1e-9))
def unpen(y,l,w=None):
    y=np.asarray(y,float); X=np.column_stack([np.ones(len(l)),np.asarray(l,float)])
    w=np.ones(len(l)) if w is None else np.asarray(w,float); beta=np.zeros(2)
    for _ in range(100):
        eta=X@beta;p=1/(1+np.exp(-eta));Wd=w*p*(1-p)+1e-12
        g=X.T@(w*(y-p));H=X.T@(X*Wd[:,None]);beta=beta+np.linalg.solve(H+1e-8*np.eye(2),g)
    return beta[1],beta[0]
def load():
    b=pd.read_csv("data/Derivation cohorts--MIMIC-IV database.csv"); b["male"]=(b["Gender"]==1).astype(int); b["outcome"]=b["Hstatus"].astype(int)
    v=pd.read_csv("data/Validation cohorts--eICU-CRD.csv"); v["male"]=(v["Gender"]==1).astype(int); v["outcome"]=v["Hstatus"].astype(int)
    return b,v
B,V=load()
pipe=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("lr",LogisticRegression(max_iter=2000))]).fit(B[COMMON].astype(float),B["outcome"].values)
def probs(df): return pipe.predict_proba(df[COMMON].astype(float))[:,1]
R={}

# ---- (a) penalised recalibration sensitivity + extreme-slope vs events (HELD-OUT validation slope, matches S7) ----
def pen_recal_fit(lp,y,lam=2.0):
    # fit logit(p)=alpha+beta*lp with a Gaussian prior centred at identity (beta=1, alpha=0): minimise deviance + 0.5*lam*((beta-1)^2+alpha^2)
    X=np.column_stack([np.ones(len(lp)),lp]); y=np.asarray(y,float); beta=np.array([0.0,1.0])
    for _ in range(200):
        eta=X@beta; p=1/(1+np.exp(-eta)); W=p*(1-p)+1e-9
        g=X.T@(y-p)-lam*np.array([beta[0],beta[1]-1.0]); H=X.T@(X*W[:,None])+lam*np.eye(2)
        step=np.linalg.solve(H+1e-9*np.eye(2),g); beta=beta+step
        if np.max(np.abs(step))<1e-9: break
    return beta[1],beta[0]   # slope, intercept
half=len(V)//2; cr=np.random.RandomState(2026)
held={n:{"unpen":[],"robust":[]} for n in SIZES}; extreme_events={n:[] for n in SIZES}; all_events={n:[] for n in SIZES}
for rep in range(10):
    p_=cr.permutation(len(V)); Vu=V.iloc[p_[:half]].reset_index(drop=True); Vt=V.iloc[p_[half:]].reset_index(drop=True)
    yu,yt=Vu.outcome.values,Vt.outcome.values; lp_u=lg(probs(Vu)); lp_t=lg(probs(Vt))
    sr=np.random.RandomState(1000+rep)
    for n in SIZES:
        for _ in range(6):
            idx=sr.choice(len(Vu),n,replace=False); ev=int(yu[idx].sum()); all_events[n].append(ev)
            _m=LogisticRegression(max_iter=2000).fit(lp_u[idx][:,None],yu[idx]); b0,a0=_m.coef_[0][0],_m.intercept_[0]
            s_un=unpen(yt,lg(1/(1+np.exp(-(a0+b0*lp_t)))))[0]
            br,ar=pen_recal_fit(lp_u[idx],yu[idx]); s_rb=unpen(yt,lg(1/(1+np.exp(-(ar+br*lp_t)))))[0]
            held[n]["unpen"].append(float(s_un)); held[n]["robust"].append(float(s_rb))
            if abs(s_un-1)>1 or s_un<0: extreme_events[n].append(ev)
sens={}
for n in SIZES:
    u=np.array(held[n]["unpen"]); r=np.array(held[n]["robust"])
    sens[str(n)]={"heldout_median_unpen":round(float(np.median(u)),3),"heldout_max_unpen":round(float(u.max()),2),"heldout_neg_unpen":int((u<0).sum()),
        "heldout_median_robust":round(float(np.median(r)),3),"heldout_max_robust":round(float(r.max()),2),"heldout_neg_robust":int((r<0).sum())}
R["penalised_recalibration_sensitivity"]=sens
R["extreme_slope_events"]={str(n):{"n_extreme_or_neg":len(extreme_events[n]),
    "median_events_extreme":(round(float(np.median(extreme_events[n])),1) if extreme_events[n] else None),
    "median_events_all":round(float(np.median(all_events[n])),1)} for n in SIZES}

# ---- (b) Table 4 held-out calibration slope + O/E (representative split rep0) ----
cr2=np.random.RandomState(2026); p0=cr2.permutation(len(V)); Vu=V.iloc[p0[:half]].reset_index(drop=True); Vt=V.iloc[p0[half:]].reset_index(drop=True)
yu,yt=Vu.outcome.values,Vt.outcome.values
def fitpred(cols,dfu,dfte):
    m=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("lr",LogisticRegression(max_iter=2000))]).fit(dfu[cols].astype(float),dfu.outcome.values)
    return m.predict_proba(dfte[cols].astype(float))[:,1]
lrf=LogisticRegression(max_iter=2000).fit(lg(probs(Vu))[:,None],yu)
pr_recal=1/(1+np.exp(-(lrf.coef_[0][0]*lg(probs(Vt))+lrf.intercept_[0])))
models={"Frozen 21-var":probs(Vt),"Two-parameter recalibration":pr_recal,
        "SOFA (refit)":fitpred(["SOFA"],Vu,Vt),"APS-III (refit)":fitpred(["APSIII"],Vu,Vt),
        "SOFA+APS-III (refit)":fitpred(["SOFA","APSIII"],Vu,Vt),"21-var refit":fitpred(COMMON,Vu,Vt)}
t4=[]
for name,p in models.items():
    s,i=unpen(yt,lg(p)); t4.append(dict(Model=name,AUC=round(float(roc_auc_score(yt,p)),4),Brier=round(float(brier_score_loss(yt,p)),4),
        slope=round(float(s),3),OE=round(float(yt.sum()/p.sum()),3)))
R["table4_with_calibration"]=t4

# ---- (c) subgroup performance of frozen model on full eICU ----
pf=probs(V); yv=V.outcome.values; lpv=lg(pf)
def sub(mask,label):
    if mask.sum()<50 or len(np.unique(yv[mask]))<2: return dict(subgroup=label,n=int(mask.sum()),note="insufficient")
    s,_=unpen(yv[mask],lpv[mask])
    return dict(subgroup=label,n=int(mask.sum()),deaths=int(yv[mask].sum()),AUC=round(float(roc_auc_score(yv[mask],pf[mask])),3),
                slope=round(float(s),2),OE=round(float(yv[mask].sum()/pf[mask].sum()),2))
age=V["Age"].astype(float); t1,t2=np.nanpercentile(age,[33,66])
subs=[sub(V["male"].values==1,"Sex: male"),sub(V["male"].values==0,"Sex: female"),
      sub((age<=t1).values,"Age tertile 1 (youngest)"),sub(((age>t1)&(age<=t2)).values,"Age tertile 2"),
      sub((age>t2).values,"Age tertile 3 (oldest)")]
R["subgroup_frozen_eICU"]=subs
json.dump(R,open("outputs/sr_v12.json","w",encoding="utf-8"),indent=2,ensure_ascii=False)

# ---- (d) Fig 3a strategies (no DCA) ----
sc=pd.read_csv("outputs/v5_strategy.csv") if os.path.exists("outputs/v5_strategy.csv") else None
labels=["Frozen","Intercept-only","Two-param","Refit"]; keys=["frozen","intercept_only","recalibrate_2par","full_refit"]
v5=json.load(open("outputs/v5_results.json",encoding="utf-8"))["strategy_comparison"]
auc=[v5[k]["AUC"][0] for k in keys]; brier=[v5[k]["Brier"][0] for k in keys]; slope=[v5[k]["slope"][0] for k in keys]; oe=[v5[k]["OE"] for k in keys]
fig,ax=plt.subplots(1,4,figsize=(13,3.4))
for a,val,ttl,ref in [(ax[0],auc,"AUC",None),(ax[1],brier,"Brier",None),(ax[2],slope,"Calibration slope",1.0),(ax[3],oe,"O/E",1.0)]:
    a.bar(labels,val,color=["#D55E00","#0072B2","#009E73","#CC79A7"]); a.set_title(ttl)
    if ref is not None: a.axhline(ref,ls=":",color="k",lw=1)
    a.tick_params(axis="x",rotation=20,labelsize=8)
fig.suptitle("Updating strategies on held-out eICU stays (10-split means); recalibration fixes calibration, only refit raises AUC",fontsize=10)
plt.tight_layout(); plt.savefig("figures/sr_fig_strategies.png",dpi=300); plt.close()
print(json.dumps(R,indent=2,ensure_ascii=False))
