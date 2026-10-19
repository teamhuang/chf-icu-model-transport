"""11_v8_heldout_bench.py — fair held-out comparison demanded by the SR review:
Severity scores (SOFA, APS-III) must be evaluated on the SAME held-out stays as the
transported model, and fitted only on the update half (never in-sample on the test half).
Also records honest in-sample (apparent) values for contrast. Writes outputs/v8_heldout.json.
Deterministic seed 2026.
"""
import os, json, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss
warnings.filterwarnings("ignore")
os.chdir(r"H:\ICU\心衰苏闽东")
COMMON=["Age","male","HeartRate","RespiratoryRate","Temperature","SBP","AG","BUN","Creatinine",
        "Chloride","MCH","MCHC","MCV","Platelet","Potassium","RBC","RDW","Sodium","WBC","COPD","Diabetes"]
B=pd.read_csv("data/Derivation cohorts--MIMIC-IV database.csv"); B["outcome"]=B["Hstatus"].astype(int)
V=pd.read_csv("data/Validation cohorts--eICU-CRD.csv"); V["outcome"]=V["Hstatus"].astype(int)
B["male"]=(B["Gender"]==1).astype(int); V["male"]=(V["Gender"]==1).astype(int)
pipe=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
              ("lr",LogisticRegression(max_iter=2000))]).fit(B[COMMON].astype(float),B["outcome"].values)
def probs(df): return pipe.predict_proba(df[COMMON].astype(float))[:,1]
def lg(p): return np.log(np.clip(p,1e-9,1-1e-9)/np.clip(1-p,1e-9,1-1e-9))

perm=np.random.RandomState(2026).permutation(len(V)); half=len(V)//2
Vu=V.iloc[perm[:half]].reset_index(drop=True); Vt=V.iloc[perm[half:]].reset_index(drop=True)
yu,yt=Vu.outcome.values,Vt.outcome.values

def fit_eval(cols, fitdf, testdf):
    Xf=fitdf[cols].astype(float); Xt=testdf[cols].astype(float)
    m=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                ("lr",LogisticRegression(max_iter=2000))]).fit(Xf,fitdf.outcome.values)
    p=m.predict_proba(Xt)[:,1]
    return round(float(roc_auc_score(testdf.outcome.values,p)),4), round(float(brier_score_loss(testdf.outcome.values,p)),4)

def insample(cols, df):
    X=df[cols].astype(float)
    m=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                ("lr",LogisticRegression(max_iter=2000))]).fit(X,df.outcome.values)
    return round(float(roc_auc_score(df.outcome.values,m.predict_proba(X)[:,1])),4)

# frozen transported model on held-out
pf=probs(Vt); frozen_auc=round(float(roc_auc_score(yt,pf)),4); frozen_brier=round(float(brier_score_loss(yt,pf)),4)
# recalibration (2par) held-out
lrf=LogisticRegression(max_iter=2000).fit(lg(probs(Vu))[:,None],yu)
pr=lrf.predict_proba(lg(probs(Vt))[:,None])[:,1]
recal_auc=round(float(roc_auc_score(yt,pr)),4); recal_brier=round(float(brier_score_loss(yt,pr)),4)

R={}
R["heldout_same_split"]=dict(
    note="All target-database scores fitted ONLY on the update half and evaluated on the identical held-out half (fair, no in-sample).",
    frozen_21var=dict(AUC=frozen_auc,Brier=frozen_brier),
    recalibrated_2par=dict(AUC=recal_auc,Brier=recal_brier),
    SOFA_only=dict(zip(["AUC","Brier"],fit_eval(["SOFA"],Vu,Vt))),
    APSIII_only=dict(zip(["AUC","Brier"],fit_eval(["APSIII"],Vu,Vt))),
    SOFA_APSIII=dict(zip(["AUC","Brier"],fit_eval(["SOFA","APSIII"],Vu,Vt))),
    refit_21var=dict(zip(["AUC","Brier"],fit_eval(COMMON,Vu,Vt))),
)
R["in_sample_apparent_CONTRAST"]=dict(
    note="These are the values the manuscript previously quoted as 'cohort-fitted'; they are in-sample (apparent) and NOT held-out-fair.",
    B_SOFA=insample(["SOFA"],B), B_APSIII=insample(["APSIII"],B),
    V_SOFA=insample(["SOFA"],V), V_APSIII=insample(["APSIII"],V),
    V_SOFA_APSIII=insample(["SOFA","APSIII"],V))
json.dump(R,open("outputs/v8_heldout.json","w"),indent=2,ensure_ascii=False)
print(json.dumps(R,indent=2,ensure_ascii=False))
