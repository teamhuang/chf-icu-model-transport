"""DEPRECATED — do not use for reported numbers. Retained for provenance only; superseded by the active pipeline in requirements.txt. See scripts/analysis_v5.py for the current implementation.

Harmonize and merge two independent CHF-in-ICU cohorts (Dryad CC0).

Cohort A (MIMIC-III, n=1177):  Zhou J, et al. BMJ Open 2021. doi:10.5061/dryad.0p2ngf1zd
Cohort B (MIMIC-IV, n=9555):  Han D, et al. BMJ Open 2022. doi:10.5061/dryad.tx95x6b18

MIMIC-IV is the successor of MIMIC-III (same hospital, later period), so the two
cohorts are plausibly non-overlapping in time and there is no shared patient id.
"""
import pandas as pd
import numpy as np
import os

DATA = "data"
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# ---- Cohort A: MIMIC-III -------------------------------------------------
a = pd.read_csv(os.path.join(DATA, "data01.csv"))
outcome_a = None
for col in ["outcome", "Outcome", "OUTCOME"]:
    if col in a.columns:
        outcome_a = col
        break
assert outcome_a is not None, "no outcome col in cohort A"
group_col = "group" if "group" in a.columns else None

print("Cohort A raw: rows=%d, outcome=%d (pct=%.2f%%) unique ids=%d" % (
    len(a), (a[outcome_a] == 1).sum(), 100 * (a[outcome_a] == 1).mean(), a["ID"].nunique()))
print("Cohort A group counts:")
print(a[group_col].value_counts() if group_col else "(none)")

rename_a = {
    "gendera": "male",  # 1 = male, 2 = female
    "hypertensive": "Hypertensive",
    "atrialfibrillation": "AF",
    "CHD with no MI": "CHD_no_MI",
    "diabetes": "Diabetes",
    "deficiencyanemias": "DeficiencyAnemia",
    "depression": "Depression",
    "Hyperlipemia": "Hyperlipidemia",
    "Renal failure": "RenalFailure",
    "COPD": "COPD",
    "heart rate": "HeartRate",
    "Systolic blood pressure": "SBP",
    "Diastolic blood pressure": "DBP",
    "Respiratory rate": "RespiratoryRate",
    "temperature": "Temperature",
    "SP O2": "SPO2",
    "Urine output": "UrineOutput",
    "hematocrit": "Hematocrit",
    "RBC": "RBC",
    "MCH": "MCH", "MCHC": "MCHC", "MCV": "MCV", "RDW": "RDW",
    "Leucocyte": "WBC",
    "Platelets": "Platelet",
    "Neutrophils": "Neutrophils",
    "Basophils": "Basophils",
    "Lymphocyte": "Lymphocyte",
    "PT": "PT",
    "INR": "INR",
    "NT-proBNP": "NTproBNP",
    "Creatine kinase": "CK",
    "Creatinine": "Creatinine",
    "Urea nitrogen": "BUN",
    "glucose": "Glucose",
    "Blood potassium": "Potassium",
    "Blood sodium": "Sodium",
    "Blood calcium": "Calcium",
    "Chloride": "Chloride",
    "Anion gap": "AG",
    "Magnesium ion": "Magnesium",
    "PH": "PH",
    "Bicarbonate": "Bicarbonate",
    "Lactic acid": "Lactate",
    "PCO2": "PCO2",
    "EF": "EF",
    "group": "Group",
    "age": "Age",
    "BMI": "BMI",
}
a = a.rename(columns=rename_a)
a["male"] = (a["male"] == 1).astype(int)          # 1=male 2=female -> 1/0
a["Cohort"] = "A_MIMIC-III"
a["ID"] = "A_" + a["ID"].astype(str)

# Derivation/validation split used by the source paper (group 1 vs 2)
a["Group"] = a["Group"].map({1: "Derivation", 2: "Validation"})

# ---- Cohort B: MIMIC-IV --------------------------------------------------
b = pd.read_csv(os.path.join(DATA, "Derivation cohorts--MIMIC-IV database.csv"))
print("\nCohort B raw: rows=%d, outcome(Hstatus=1)=%d" % (len(b), (b["Hstatus"] == 1).sum()))
print("Group counts:")
print(b["Group"].value_counts(dropna=False))
print("Hstatus values:", b["Hstatus"].unique())
print("Gender values:", b["Gender"].unique())
print("Race values:", b["Race"].unique())

# Gender encoding: inspect numerically
for g in sorted(b["Gender"].dropna().unique()):
    gm = b.loc[b["Gender"] == g, "Age"]
    print("  Gender=%s -> n=%d, mean age=%.1f" % (g, len(gm), gm.mean()))

b = b.rename(columns={
    "Gender": "Gender_raw",
    "Hstatus": "outcome",
})
# Gender coding 1/2 mirrors cohort A (gendera 1=male, 2=female): verify by age pattern
b["male"] = (b["Gender_raw"] == 1).astype(int)
# keep a subset of scientifically defensible variables common/compat with A where possible,
# but B has its own richer set -> we keep B's variables under original names.
b["Cohort"] = "B_MIMIC-IV"
b["ID"] = "B_" + b.index.astype(str).str.zfill(6)

# ---- harmonized merged table --------------------------------------------
# Keep union of variables; overlapping variables share the same name.
common = sorted(set(a.columns) & set(b.columns))
print("\nOverlapping harmonized columns:", common)
merged = pd.concat([a, b], axis=0, ignore_index=True, sort=False)
merged.to_csv(os.path.join(OUT, "merged_cohorts.csv"), index=False)
print("Merged table: %d rows x %d cols -> outputs/merged_cohorts.csv" % (merged.shape[0], merged.shape[1]))
print(merged["Cohort"].value_counts())
print(merged.groupby("Cohort")["outcome"].agg(["size", "sum", "mean"]))
