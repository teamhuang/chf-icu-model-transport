"""08_v6.py — additional real analyses demanded by the v5 SR pre-submission review.
Adds (all written to outputs/v6_results.json + supplementary CSVs):
 MC4: patient-level PAIRED bootstrap of recalibrated-vs-retrain differences (dAUC, dBrier, dNetBenefit).
 MC1: per-variable SMD before/after weighting, weight diagnostics, weight-TRUNCATION sensitivity,
      and uncertainty (bootstrap CI) on the weighted calibration slope.
 MC2: monotonicity check of the 'all-six-sub-stable-by-n' rule; Wilson CI on the per-update success rate.
 Flex calibration: decile table + Expected Calibration Error (ECE) and average calibrated deviation for frozen vs recalibrated.
"""
import os, json, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss
warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMMON = ["Age","male","HeartRate","RespiratoryRate","Temperature","SBP","AG","BUN","Creatinine",
          "Chloride","MCH","MCHC","MCV","Platelet","Potassium","RBC","RDW","Sodium","WBC","COPD","Diabetes"]
def load():
    a=pd.read_csv("data/data01.csv").rename(columns={"heart rate":"HeartRate","Respiratory rate":"RespiratoryRate","temperature":"Temperature","Systolic blood pressure":"SBP","Anion gap":"AG","Urea nitrogen":"BUN","Blood potassium":"Potassium","Blood sodium":"Sodium","Leucocyte":"WBC","Platelets":"Platelet","age":"Age"})
    a["male"]=(a["gendera"]==1).astype(int); a["Diabetes"]=a["diabetes"].astype(int)
    a=a[a["outcome"].notna()].copy(); a["outcome"]=a["outcome"].astype(int)
    b=pd.read_csv("data/Derivation cohorts--MIMIC-IV database.csv"); b["male"]=(b["Gender"]==1).astype(int); b["outcome"]=b["Hstatus"].astype(int)
    v=pd.read_csv("data/Validation cohorts--eICU-CRD.csv"); v["male"]=(v["Gender"]==1).astype(int); v["outcome"]=v["Hstatus"].astype(int)
    return a,b,v
A,B,V=load()
pipe=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("lr",LogisticRegression(max_iter=2000))]).fit(B[COMMON].astype(float),B["outcome"].values)
def probs(df): return pipe.predict_proba(df[COMMON].astype(float))[:,1]
def lg(p): return np.log(np.clip(p,1e-9,1-1e-9)/np.clip(1-p,1e-9,1-1e-9))
def unpen(y,l,w=None):
    y=np.asarray(y,float); X=np.column_stack([np.ones(len(l)),np.asarray(l,float)])
    w=np.ones(len(l)) if w is None else np.asarray(w,float); beta=np.zeros(2)
    for _ in range(100):
        eta=X@beta;p=1/(1+np.exp(-eta));Wd=w*p*(1-p)+1e-12
        g=X.T@(w*(y-p));H=X.T@(X*Wd[:,None]);beta=beta+np.linalg.solve(H+1e-8*np.eye(2),g)
    return beta[1],beta[0]
R={}

# ---- representative eICU 50/50 split (same seed as v5) ----
perm=np.random.RandomState(2026).permutation(len(V)); half=len(V)//2
Vupd,Vtest=V.iloc[perm[:half]].reset_index(drop=True),V.iloc[perm[half:]].reset_index(drop=True)
yupd,ytest=Vupd.outcome.values,Vtest.outcome.values
lp_u,lp_t=lg(probs(Vupd)),lg(probs(Vtest))
lrf=LogisticRegression(max_iter=2000).fit(lp_u[:,None],yupd)
pr_recal=lrf.predict_proba(lp_t[:,None])[:,1]
pr_frozen=probs(Vtest)
lrfit=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("lr",LogisticRegression(max_iter=2000))]).fit(Vupd[COMMON].astype(float),yupd)
pr_refit=lrfit.predict_proba(Vtest[COMMON].astype(float))[:,1]

# ---- MC4: paired bootstrap (resample test patients; predictions fixed per patient) ----
rng=np.random.RandomState(11); n=len(ytest); Bn=1000
def nb(y,p,t): nn=len(y); return ((p>=t)&(y==1)).sum()/nn - (((p>=t)&(y==0)).sum()/nn)*(t/(1-t))
ths=[0.05,0.10,0.20,0.30]
dbA=[];dB=[];dNB={t:[] for t in ths}
rec={a:[] for a in []}
for _ in range(Bn):
    i=rng.choice(n,n,replace=True); yy=ytest[i]
    if yy.sum()==0 or yy.sum()==len(yy): continue
    ar=roc_auc_score(yy,pr_recal[i]);af=roc_auc_score(yy,pr_refit[i])
    dbA.append(af-ar); dB.append(brier_score_loss(yy,pr_refit[i])-brier_score_loss(yy,pr_recal[i]))
    for t in ths: dNB[t].append(nb(yy,pr_refit[i],t)-nb(yy,pr_recal[i],t))
def ci(x): x=np.array(x); return [round(float(np.percentile(x,2.5)),4),round(float(np.percentile(x,97.5)),4)]
R["paired_recal_vs_refit"]=dict(
    n_boot=int(len(dbA)),
    dAUC_refit_minus_recal=dict(mean=round(float(np.mean(dbA)),4),ci95=ci(dbA)),
    dBrier_refit_minus_recal=dict(mean=round(float(np.mean(dB)),4),ci95=ci(dB)),
    dNetBenefit_refit_minus_recal={str(t):dict(mean=round(float(np.mean(dNB[t])),5),ci95=ci(dNB[t])) for t in ths},
    point_AUC=dict(recal=round(float(roc_auc_score(ytest,pr_recal)),4),refit=round(float(roc_auc_score(ytest,pr_refit)),4)),
    point_Brier=dict(recal=round(float(brier_score_loss(ytest,pr_recal)),4),refit=round(float(brier_score_loss(ytest,pr_refit)),4)))

# ---- flexible calibration + ECE for frozen & recalibrated ----
def decile_ece(y,p,groups=10):
    df=pd.DataFrame({"y":y,"p":p}); df["b"]=pd.qcut(df.p,groups,duplicates="drop")
    g=df.groupby("b",observed=True).agg(n=("y","size"),obs=("y","mean"),pred=("p","mean"))
    ece=float((np.abs(g["obs"]-g["pred"])*g["n"]).sum()/g["n"].sum())
    acd=float(np.abs(g["obs"]-g["pred"]).mean())
    return g.reset_index(drop=True).round(4), ece, acd
gd_f,ece_f,acd_f=decile_ece(ytest,pr_frozen); gd_r,ece_r,acd_r=decile_ece(ytest,pr_recal)
gd_f.to_csv("outputs/v6_calibration_deciles_frozen.csv",index=False); gd_r.to_csv("outputs/v6_calibration_deciles_recalibrated.csv",index=False)
R["flexible_calibration"]=dict(frozen=dict(ECE=round(ece_f,4),avg_calibrated_deviation=round(acd_f,4)),
    recalibrated=dict(ECE=round(ece_r,4),avg_calibrated_deviation=round(acd_r,4)))

# ---- MC1: per-variable SMD before/after correct weighting + truncation sensitivity + weighted-slope CI ----
Xp=pd.concat([B[COMMON].astype(float),V[COMMON].astype(float)],ignore_index=True)
isV=np.r_[np.zeros(len(B)),np.ones(len(V))]
clf=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("lr",LogisticRegression(max_iter=2000))]).fit(Xp,isV)
ps=clf.predict_proba(V[COMMON].astype(float))[:,1]; piV=len(V)/(len(V)+len(B)); piB=1-piV
def wdir(clip): 
    w=((1-ps)/ps)*(piV/piB); 
    if clip=="none": return w
    if clip=="0.1-10": return np.clip(w,0.1,10)
    if clip=="1-5": return np.clip(w,1,5)
    if clip=="p1p99": lo,hi=np.percentile(w,[1,99]); return np.clip(w,lo,hi)
def smd_one(x1,x0,w):
    m1=np.average(x1,weights=w); v1=np.average((x1-m1)**2,weights=w); m0=np.mean(x0); v0=np.var(x0,ddof=1)
    return abs(m1-m0)/np.sqrt((v1+v0)/2)
yv=V.outcome.values; lpv=lg(probs(V)); pv=probs(V)
rows=[]
w01=wdir("0.1-10")
for c in COMMON:
    x1=V[c].astype(float).values; x0=B[c].astype(float).values
    rows.append(dict(Variable=c,SMD_unweighted=round(smd_one(x1,x0,np.ones(len(V))),4),SMD_weighted=round(smd_one(x1,x0,w01),4)))
smdtab=pd.DataFrame(rows); smdtab["improved"]=smdtab.SMD_unweighted>smdtab.SMD_weighted
smdtab.to_csv("outputs/v6_smd_by_variable.csv",index=False)
def diagnostics(w):
    ess=float(w.sum()**2/(w**2).sum())
    msmd=float(np.mean([smd_one(V[c].astype(float).values,B[c].astype(float).values,w) for c in COMMON]))
    mx=float(np.max([smd_one(V[c].astype(float).values,B[c].astype(float).values,w) for c in COMMON]))
    sl,it=unpen(yv,lpv,w); oe=float((w*yv).sum()/(w*pv).sum())
    return dict(mean_abs_SMD=round(msmd,4),max_abs_SMD=round(mx,4),ESS=round(ess,0),ESS_pct=round(100*ess/len(w),1),
        weighted_slope=round(float(sl),4),weighted_intercept=round(float(it),4),weighted_OE=round(oe,4),
        weight_min=round(float(w.min()),3),weight_median=round(float(np.median(w)),3),weight_max=round(float(w.max()),3))
R["weight_sensitivity"]={lbl:diagnostics(wdir(lbl)) for lbl in ["0.1-10","1-5","p1p99"]}
# bootstrap CI on weighted slope (corrected, 0.1-10)
rng2=np.random.RandomState(5); bs=[]
for _ in range(300):
    i=rng2.choice(len(V),len(V),replace=True)
    if len(np.unique(yv[i]))<2: continue
    try: s,_=unpen(yv[i],lpv[i],w01[i]); bs.append(s)
    except Exception: pass
R["weighted_slope_CI01_10"]=dict(mean=round(float(np.mean(bs)),4),ci95=[round(float(np.percentile(bs,2.5)),4),round(float(np.percentile(bs,97.5)),4)],n_boot=len(bs))

# ---- MC2: monotonicity of 'all-six-sub-stable-by-n' + Wilson CI for per-update success rate ----
rd=pd.read_csv("outputs/v5_repeated_curve.csv"); SIZES=[100,200,500,1000,2000,5000]
viol=0; mono_all=[]
for rep,grp in rd.groupby("rep"):
    prev=False; mono=True
    for n in SIZES:
        sub=grp[grp.n_pat==n]; allst=bool(sub.stable.all())
        if prev and not allst: mono=False  # stable at smaller size but not at larger -> non-monotone
        prev=allst
    mono_all.append(mono)
viol=len([m for m in mono_all if not m])
def wilson(k,n,z=1.96):
    if n==0: return [None,None]
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [round(max(0,c-h),3),round(min(1,c+h),3)]
per_success=[]
for n in SIZES:
    sub=rd[rd.n_pat==n]; k=int(sub.stable.sum()); m=len(sub)
    per_success.append(dict(n_pat=n,success_frac=round(100*k/m,1),wilson_lo=round(100*wilson(k,m)[0],1),wilson_hi=round(100*wilson(k,m)[1],1),trials=m))
R["learning_curve_monotonicity"]=dict(n_splits=len(mono_all),non_monotone_splits=viol,
    per_update_success_with_CI=per_success,
    note="success_frac = share of the (10x6) held-out recalibrations judged stable at each size (PRIMARY per-update metric); the 'all-six-per-split' minimum-size distribution is the SECONDARY metric reported separately.")

json.dump(R,open("outputs/v6_results.json","w"),indent=2,ensure_ascii=False)
print(json.dumps(R,indent=2,ensure_ascii=False))
