"""13_v9_sensitivity.py — Round-9 core new computations (all real, seed 2026, relative root).
A) Stability-curve threshold sensitivity + per-size event counts + fit-failure counts.
B) Reweighting upgrades: propensity separability (AUC), weight quantiles/extreme %, absolute ESS,
   bootstrap WITH propensity re-fit (honest interval), quantile/non-linear balance, COPD/RR diagnosis, overlap figure.
C) SOFA/APSIII vs recalibrated/refit: across-10-split paired delta-AUC + paired patient bootstrap + calibration curves.
Writes outputs/sr_sens_v9.json, sr_threshold_sensitivity.csv, sr_reweight_v9.json, sr_sofa_compare.csv, figures.
"""
import os, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_auc_score
ROOT=Path(__file__).resolve().parents[1]; os.chdir(ROOT)
COMMON=["Age","male","HeartRate","RespiratoryRate","Temperature","SBP","AG","BUN","Creatinine",
        "Chloride","MCH","MCHC","MCV","Platelet","Potassium","RBC","RDW","Sodium","WBC","COPD","Diabetes"]
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

# ============ A) threshold sensitivity + events + failures ============
rd=pd.read_csv("outputs/v5_repeated_curve.csv")  # stored slope, OE, stable, rep, n_pat
th_slo=[0.10,0.20,0.30]; th_oe=[0.10,0.15,0.20]
thr=[]
for ts in th_slo:
    for toe in th_oe:
        row={"slope_tol":ts,"oe_tol":toe}
        for n in SIZES:
            sub=rd[rd.n_pat==n]; hit=((sub.slope-1).abs()<ts)&((sub.OE-1).abs()<toe)
            row[f"rate@{n}"]=round(float(hit.mean()*100),1)
        thr.append(row)
    thr_pd=pd.DataFrame(thr); 
thr_pd.to_csv("outputs/sr_threshold_sensitivity.csv",index=False)
R["threshold_sensitivity"]=thr_pd.to_dict("records")
# primary config check (0.2/0.15) reproduces Table 2
prim=rd  # stable column already = |slope-1|<0.2 & |OE-1|<0.15
R["primary_reproduced"]={str(n):round(float(prim[prim.n_pat==n].stable.mean()*100),1) for n in SIZES}

# events + fit-failure: redraw identical to analysis_v5 (sub_seed=1000+rep, same outer 50/50 seed 2026 sequence)
# We must match the exact loop. In analysis_v5: curve_rng=RandomState(2026); for rep: p_=curve_rng.permutation(len(V)); Vu=first half.
curve_rng=np.random.RandomState(2026); half=len(V)//2
ev_fail={}
for n in SIZES: ev_fail[n]={"update_events_min":None,"events_list":[],"neg_slope_update_fit":0,"extreme_update_fit_slope_gt3":0,"nan_or_nonconv":0,"total":0}
for rep in range(10):
    p_=curve_rng.permutation(len(V)); Vu=V.iloc[p_[:half]].reset_index(drop=True)
    yu=Vu.outcome.values; lp_u=lg(probs(Vu))
    sub_rng=np.random.RandomState(1000+rep)
    for n in SIZES:
        for _ in range(6):
            idx=sub_rng.choice(len(Vu),n,replace=False)
            ev=int(yu[idx].sum()); ev_fail[n]["events_list"].append(ev)
            try:
                lrn=LogisticRegression(max_iter=2000).fit(lp_u[idx][:,None],yu[idx])
                sl=lrn.coef_[0][0]
                if sl<0: ev_fail[n]["neg_slope_update_fit"]+=1
                if abs(sl)>3: ev_fail[n]["extreme_update_fit_slope_gt3"]+=1
            except Exception:
                ev_fail[n]["nan_or_nonconv"]+=1
            ev_fail[n]["total"]+=1
R["update_event_counts"]={str(n):{"min":int(min(d["events_list"])),"median":int(np.median(d["events_list"])),
    "max":int(max(d["events_list"])),"neg_slope_update_fit":d["neg_slope_update_fit"],"extreme_update_fit_slope_gt3":d["extreme_update_fit_slope_gt3"],
    "nonconv":d["nan_or_nonconv"],"total":d["total"]} for n,d in ev_fail.items()}

# ============ B) reweighting upgrades ============
Xpool=pd.concat([B[COMMON].astype(float),V[COMMON].astype(float)],ignore_index=True)
isV=np.r_[np.zeros(len(B)),np.ones(len(V))]
clf=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("lr",LogisticRegression(max_iter=2000))]).fit(Xpool,isV)
ps=clf.predict_proba(V[COMMON].astype(float))[:,1]
piV=len(V)/(len(V)+len(B)); piB=1-piV
# separability: AUC of distinguishing eICU vs MIMIC-IV from the 21 covariates
pB=clf.predict_proba(B[COMMON].astype(float))[:,1]
sep_auc=roc_auc_score(np.r_[np.zeros(len(B)),np.ones(len(V))], np.r_[pB,ps])
w=np.clip(((1-ps)/ps)*(piV/piB),0.1,10)
yv=V.outcome.values; lpv=lg(probs(V)); pv=probs(V)
ess=float(w.sum()**2/(w**2).sum())
q=np.percentile(w,[1,5,25,50,75,95,99])
extreme=float(((w<=0.1000001)|(w>=9.9999999)).mean())  # share pinned at either clip boundary
# quantile/non-linear balance: compare eICU (weighted) vs MIMIC-IV at 10/50/90 pct
def qbal(c):
    x1=np.asarray(V[c],float); x0=np.asarray(B[c],float)
    ok=~np.isnan(x1)
    sw=w[ok]; v1=x1[ok]
    def wq(v,s,qq):
        o=np.argsort(v); v=v[o]; s=s[o]; cw=np.cumsum(s)/s.sum(); return float(v[np.searchsorted(cw,qq)])
    return {"p10_dev":round(wq(v1,sw,0.10)-np.nanpercentile(x0,10),3),
            "p50_dev":round(wq(v1,sw,0.50)-np.nanpercentile(x0,50),3),
            "p90_dev":round(wq(v1,sw,0.90)-np.nanpercentile(x0,90),3)}
R["weight_diagnostics"]=dict(propensity_separability_AUC=round(float(sep_auc),3),
    weight_min=float(q[0]),weight_p25=float(q[2]),weight_median=float(q[3]),weight_p75=float(q[4]),weight_p99=float(q[6]),
    weight_max=float(w.max()),extreme_weight_fraction=round(extreme,3),
    ESS_absolute=int(round(ess)),ESS_pct=round(100*ess/len(V),1),
    unweighted_propensity_median=round(float(np.median(ps)),3),
    nonlinear_quantile_deviation={c:qbal(c) for c in ["RespiratoryRate","COPD","Age","RBC","AG","HeartRate"]})
# bootstrap WITH propensity re-fit (honest weighted-slope interval)
rng=np.random.RandomState(5); bs=[]
Xb=B[COMMON].astype(float).values; Xv=V[COMMON].astype(float).values
nb=len(V); nbB=len(B)
for _ in range(200):
    iv=rng.choice(nb,nb,replace=True); ib=rng.choice(nbB,nbB,replace=True)
    ybi=yv[iv]
    if len(np.unique(ybi))<2: continue
    Xp=pd.concat([pd.DataFrame(Xb[ib],columns=COMMON),pd.DataFrame(Xv[iv],columns=COMMON)],ignore_index=True)
    isVb=np.r_[np.zeros(len(ib)),np.ones(len(iv))]
    try:
        clfb=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("lr",LogisticRegression(max_iter=1000))]).fit(Xp,isVb)
        psb=clfb.predict_proba(Xv[iv])[:,1]; piv=len(iv)/(len(iv)+len(ib))
        wb=np.clip(((1-psb)/psb)*(piv/(1-piv)),0.1,10)
        s,_=unpen(ybi,lpv[iv],wb); 
        if np.isfinite(s): bs.append(s)
    except Exception: pass
bs=np.array(bs)
R["weighted_slope_bootstrap_refit"]={"n_boot":int(len(bs)),"mean":round(float(bs.mean()),3),
    "ci95":[round(float(np.percentile(bs,2.5)),3),round(float(np.percentile(bs,97.5)),3)],
    "note":"propensity/weight model re-fit inside each bootstrap replicate; compare with the fixed-weight 0.383-0.520 interval."}
# old fixed-weight interval for contrast
try:
    import json as _j; v6=_j.load(open("outputs/v6_results.json",encoding="utf-8"))
    R["weighted_slope_bootstrap_fixedweight_FOR_CONTRAST"]=v6["weighted_slope_CI01_10"]
except Exception as e: R["weighted_slope_bootstrap_fixedweight_FOR_CONTRAST"]="err:"+str(e)
json.dump(R,open("outputs/sr_sens_v9.json","w",encoding="utf-8"),indent=2,ensure_ascii=False)

# overlap figure: propensity distribution both cohorts
pb_all=clf.predict_proba(Xpool)[:,1]
fig,ax=plt.subplots(1,2,figsize=(10,4))
ax[0].hist(clf.predict_proba(B[COMMON].astype(float))[:,1],bins=40,alpha=.6,label="MIMIC-IV",color="tab:blue",density=True)
ax[0].hist(ps,bins=40,alpha=.6,label="eICU",color="tab:red",density=True)
ax[0].set_xlabel("P(eICU | X)"); ax[0].set_ylabel("density"); ax[0].set_title("Propensity overlap (separability AUC=%.2f)"%sep_auc); ax[0].legend()
ax[1].hist(np.log10(w),bins=40,color="tab:green"); ax[1].set_xlabel("log10 weight (0.1-10 clip)"); ax[1].set_title("Weight distribution (ESS=%.0f, %.1f%%)"%(ess,100*ess/nb))
plt.tight_layout(); plt.savefig("figures/sr_fig_overlap.png",dpi=300); plt.close()

# ============ C) SOFA / APSIII vs recalibrated / refit: across-10-split paired ΔAUC ============
srng=np.random.RandomState(2026)
rows=[]; paired={k:[] for k in ["recal","sofa","aps3","sofa_aps3","refit"]}
per_split={}
for rep in range(10):
    p_=srng.permutation(len(V)); Vu=V.iloc[p_[:half]].reset_index(drop=True); Vt=V.iloc[p_[half:]].reset_index(drop=True)
    yu,yt=Vu.outcome.values,Vt.outcome.values; lp_u,lp_t=lg(probs(Vu)),lg(probs(Vt))
    pr_frozen=probs(Vt)
    lrf=LogisticRegression(max_iter=2000).fit(lp_u[:,None],yu); pr_recal=1/(1+np.exp(-(lrf.coef_[0][0]*lp_t+lrf.intercept_[0])))
    def fitpred(cols):
        m=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("lr",LogisticRegression(max_iter=2000))]).fit(Vu[cols].astype(float),yu)
        return m.predict_proba(Vt[cols].astype(float))[:,1]
    pr_sofa=fitpred(["SOFA"]); pr_aps=fitpred(["APSIII"]); pr_sa=fitpred(["SOFA","APSIII"]); pr_ref=fitpred(COMMON)
    a_f=roc_auc_score(yt,pr_frozen)
    named={"recal":pr_recal,"sofa":pr_sofa,"aps3":pr_aps,"sofa_aps3":pr_sa,"refit":pr_ref}
    per_split[rep]={k:round(float(roc_auc_score(yt,v)),4) for k,v in named.items()} | {"frozen":round(float(a_f),4)}
    for k,v in named.items(): paired[k].append(roc_auc_score(yt,v)-a_f)
    # store representative split (rep0) predictions for calibration-curve fig + paired patient bootstrap
    if rep==0:
        np.save("outputs/sr_rep0_test.npy",np.column_stack([yt,pr_frozen,pr_recal,pr_sofa,pr_ref]))
def tci(x): x=np.array(x); m=x.mean(); se=x.std(ddof=1)/np.sqrt(len(x)); return [round(float(m-1.96*se),4),round(float(m+1.96*se),4)]
R["delta_auc_vs_frozen_across10"]={k:{"mean":round(float(np.mean(v)),4),"ci95":tci(v)} for k,v in paired.items()}
R["per_split_auc"]=per_split
# paired patient bootstrap on rep0 test: sofa vs recal
d=0.0
yt,pf,prc,psf,pref=np.load("outputs/sr_rep0_test.npy",allow_pickle=True).T if False else (None,)*5
arr=np.load("outputs/sr_rep0_test.npy"); yt=arr[:,0]; pf=arr[:,1]; prc=arr[:,2]; psf=arr[:,3]; pref=arr[:,4]
r2=np.random.RandomState(11); nn=len(yt); d_so_re=[]; d_ref_so=[]
for _ in range(1000):
    i=r2.choice(nn,nn,replace=True); yy=yt[i]
    if yy.sum() in (0,len(yy)): continue
    d_so_re.append(roc_auc_score(yy,psf[i])-roc_auc_score(yy,prc[i]))
    d_ref_so.append(roc_auc_score(yy,pref[i])-roc_auc_score(yy,psf[i]))
def pci(x): x=np.array(x); return [round(float(np.percentile(x,2.5)),4),round(float(np.percentile(x,97.5)),4)]
R["paired_patient_bootstrap_rep0"]={"dAUC_sofa_minus_recalibrated":{"mean":round(float(np.mean(d_so_re)),4),"ci95":pci(d_so_re)},
    "dAUC_refit_minus_sofa":{"mean":round(float(np.mean(d_ref_so)),4),"ci95":pci(d_ref_so)},
    "n_boot":len(d_so_re)}
pd.DataFrame([{"rep":k,**v} for k,v in per_split.items()]).to_csv("outputs/sr_sofa_compare.csv",index=False)

# calibration curves figure (rep0 held-out): frozen, recalibrated, SOFA, refit
fig,ax=plt.subplots(figsize=(6,5))
for p,lab,c,ls in [(pf,"Frozen","tab:red",":"),(prc,"Recalibrated","tab:blue","-"),(psf,"SOFA (refit)","tab:green","-."),(pref,"21-var refit","tab:purple","--")]:
    df=pd.DataFrame({"y":yt,"p":p}); df["b"]=pd.qcut(df.p,10,duplicates="drop")
    g=df.groupby("b",observed=True).agg(pred=("p","mean"),obs=("y","mean"))
    ax.plot(g.pred,g.obs,color=c,ls=ls,label=lab,marker="o",ms=4)
ax.plot([0,1],[0,1],"k:",lw=.8,label="Ideal")
ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed rate"); ax.set_title("Held-out calibration curves (representative split)")
ax.legend(fontsize=8); plt.tight_layout(); plt.savefig("figures/sr_fig_calib_curves.png",dpi=300); plt.close()

json.dump(R,open("outputs/sr_sens_v9.json","w",encoding="utf-8"),indent=2,ensure_ascii=False)
print(json.dumps(R,indent=2,ensure_ascii=False))
