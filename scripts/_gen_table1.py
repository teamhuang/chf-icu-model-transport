import json, sys
d = json.load(open("outputs/sr_baseline_v18.json", encoding="utf-8"))
cs = d["cohort_sizes"]; V = d["variables"]; U = d["units"]
def f(v):
    if v["kind"] == "binary":
        return "%d (%.1f)" % (v["n_pos"], v["pct_pos"])
    return "%.1f (%.1f)" % (v["mean"], v["sd"])
rows = []
rows.append("| Characteristic | MIMIC-IV (n=15,983) | MIMIC-III (n=1,176) | eICU-CRD (n=14,428) | SMD (eICU vs MIMIC-IV) |")
rows.append("|---|---|---|---|---|")
rows.append("| **In-hospital deaths, n (pct)** | %s (%.1f) | %s (%.1f) | %s (%.1f) | — |" % (
    format(cs["MIMIC_IV"]["deaths"], ","), 100*cs["MIMIC_IV"]["prevalence"],
    format(cs["MIMIC_III"]["deaths"], ","), 100*cs["MIMIC_III"]["prevalence"],
    format(cs["eICU"]["deaths"], ","), 100*cs["eICU"]["prevalence"]))
groups = [("Demographics", ["Age", "male"]),
          ("Vital signs", ["HeartRate", "RespiratoryRate", "Temperature", "SBP"]),
          ("Laboratory values", ["AG", "BUN", "Creatinine", "Chloride", "Sodium", "Potassium",
                                 "MCH", "MCHC", "MCV", "Platelet", "RBC", "RDW", "WBC"]),
          ("Comorbidities", ["COPD", "Diabetes"])]
for gname, cols in groups:
    rows.append("| *%s* | | | | |" % gname)
    for c in cols:
        r = V[c]; s = r["SMD_eICU_vs_MIMICIV"]
        ss = ("%.3f" % s) if s is not None else "—"
        if s is not None and abs(s) > 0.10:
            ss += "**"
        lbl = c if c in ("male", "COPD", "Diabetes") else "%s (%s)" % (c, U.get(c, U.get(c.lower(), "")))
        rows.append("| %s | %s | %s | %s | %s |" % (lbl, f(r["MIMIC_IV"]), f(r["MIMIC_III"]), f(r["eICU"]), ss))
txt = "\n".join(rows)
open("outputs/_table1_chars.md", "w", encoding="utf-8").write(txt)
print(txt)
