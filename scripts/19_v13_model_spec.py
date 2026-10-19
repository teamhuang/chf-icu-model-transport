"""19_v13_model_spec.py — emit a COMPLETE, verifiable model specification so the
TRIPOD+AI 'model accessibility' claim is literally true: intercept + 21 betas +
StandardScaler mean_/scale_ + SimpleImputer medians, from the same fitted pipeline."""
import os, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
ROOT=Path(__file__).resolve().parents[1]; os.chdir(ROOT)
COMMON=["Age","male","HeartRate","RespiratoryRate","Temperature","SBP","AG","BUN","Creatinine","Chloride","MCH","MCHC","MCV","Platelet","Potassium","RBC","RDW","Sodium","WBC","COPD","Diabetes"]
b=pd.read_csv("data/Derivation cohorts--MIMIC-IV database.csv"); b["male"]=(b["Gender"]==1).astype(int); b["outcome"]=b["Hstatus"].astype(int)
X=b[COMMON].astype(float)
imp=SimpleImputer(strategy="median").fit(X)
sc=StandardScaler().fit(imp.transform(X))
lr=LogisticRegression(max_iter=2000).fit(sc.transform(imp.transform(X)), b["outcome"].values)
rows=[]
for i,v in enumerate(COMMON):
    rows.append(dict(variable=v,
                     impute_median=float(imp.statistics_[i]),
                     scaler_mean=float(sc.mean_[i]),
                     scaler_scale=float(sc.scale_[i]),
                     beta_standardised=float(lr.coef_[0][i])))
spec=pd.DataFrame(rows)
spec.to_csv("outputs/model_specification.csv", index=False)
meta=dict(model="L2 logistic (sklearn LogisticRegression, C=1.0, intercept unpenalised), linear terms only",
          fit_on="MIMIC-IV derivation file (15,983 stays; 1,988 deaths); seed 2026",
          intercept_standardised=float(lr.intercept_[0]),
          prediction_rule="eta = intercept + sum_i beta_i * z_i ; z_i = (x_i - scaler_mean_i)/scaler_scale_i with x_i imputed to impute_median_i if missing ; P = sigmoid(eta)",
          n_variables=len(COMMON))
json.dump(dict(meta=meta, coefficients=rows), open("outputs/model_specification.json","w",encoding="utf-8"), indent=2, ensure_ascii=False)
print("intercept(standardised)=", round(meta["intercept_standardised"],4))
print("wrote outputs/model_specification.csv and .json")
print(spec.to_string(index=False))
