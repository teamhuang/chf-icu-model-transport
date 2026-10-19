"""28_finalize_v18.py — turn the v18 body (with [[oldnum]] citation tokens and a
@@REFERENCES@@ marker) into the final submission manuscript:
  1. compute the order of first appearance of every cited reference,
  2. renumber strictly by that order (Nature/SR sequential style),
  3. replace tokens with superscript markers «k» / «k-m» / «k,m,l»,
  4. emit the reference list in the new order in Nature/SR format,
  5. write outputs/SCI_manuscript_v18.md and a mapping JSON for the audit.

Changes vs 25_finalize_v17.py: DOIs added to every reference; the all-ICU Zhu
reference (old 5) is no longer bundled into the heart-failure citation range and
is cited on its own instead.
"""
import os, re, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; os.chdir(ROOT)
body = open("outputs/SCI_manuscript_v18_body.md", encoding="utf-8").read()

REF = {
 1: "Li, F. et al. Prediction model of in-hospital mortality in intensive care unit patients with heart failure: machine learning-based, retrospective analysis of the MIMIC-III database. *BMJ Open* **11**, e044779 (2021). https://doi.org/10.1136/bmjopen-2020-044779",
 2: "Han, D. et al. Early prediction of in-hospital mortality in patients with congestive heart failure in intensive care unit: a retrospective observational cohort study. *BMJ Open* **12**, e059761 (2022). https://doi.org/10.1136/bmjopen-2021-059761",
 3: "Chen, Z. et al. Machine learning-based in-hospital mortality risk prediction tool for intensive care unit patients with heart failure. *Front Cardiovasc Med* **10**, 1119699 (2023). https://doi.org/10.3389/fcvm.2023.1119699",
 4: "He, M. et al. Interpretable machine learning models for predicting in-hospital mortality in patients with chronic critical illness and heart failure: a multicenter study. *Digit Health* **11**, 20552076251347785 (2025). https://doi.org/10.1177/20552076251347785",
 5: "Zhu, J. et al. Development and external validation of an ICU mortality prediction model using routinely collected clinical data. *Medicine (Baltimore)* **105**, e49009 (2026). https://doi.org/10.1097/MD.0000000000049009",
 6: "Sun, M. et al. Early prediction of in-hospital mortality in ICU patients with heart failure using an interpretable machine learning model: a retrospective study based on MIMIC-IV. *BMC Cardiovasc Disord* **26**, 187 (2026). https://doi.org/10.1186/s12872-026-02187-1",
 7: "Xu, L. et al. Development and external validation of an explainable machine learning model for in-hospital mortality risk stratification in intensive care unit patients with heart failure. *Front Cardiovasc Med* **13**, 1942393 (2026). https://doi.org/10.3389/fcvm.2026.1942393",
 8: "Xiong, Y.-J. et al. Real-time monitoring of in-hospital mortality risk in intensive care units heart failure patients using an extreme gradient boosting model. *J Geriatr Cardiol* **23**, 184–205 (2026). https://doi.org/10.1016/j.jgc.2026.02.001",
 9: "Van Calster, B. et al. Calibration: the Achilles heel of predictive analytics. *BMC Med* **17**, 230 (2019). https://doi.org/10.1186/s12916-019-1466-7",
 11: "Vergouwe, Y. et al. A closed testing procedure to select an appropriate method for updating prediction models. *Stat Med* **36**, 4529–4539 (2017). https://doi.org/10.1002/sim.7292",
 12: "Schnellinger, E. M., Yang, W. & Kimmel, S. E. Comparison of dynamic updating strategies for clinical prediction models. *Diagn Progn Res* **5**, 20 (2021). https://doi.org/10.1186/s41698-021-00112-5",
 13: "Riley, R. D., Whittle, R., Sadatsafavi, M., Martin, G. P., Pate, A., Collins, G. S. & Ensor, J. A general sample size framework for developing or updating a predictive algorithm: with application to clinical prediction models. *BMC Med Res Methodol* **26**, 202 (2026). https://doi.org/10.1186/s12874-026-02856-7",
 14: "Legha, A. et al. Sequential sample size calculations and learning curves safeguard the robust development of a clinical prediction model for individuals. *J Clin Epidemiol* **191**, 112117 (2026). https://doi.org/10.1016/j.jclinepi.2026.112117",
 15: "Elayan, H. et al. Correcting for case-mix shift when developing clinical prediction models. *BMC Med Res Methodol* **25**, 186 (2025). https://doi.org/10.1186/s12874-025-04031-4",
 16: "Barlow, C. J. & Pilcher, D. V. Severity scoring systems in intensive care: new models, or continuous recalibration? *Curr Opin Crit Care* **32**, 484–490 (2026). https://doi.org/10.1097/MCC.0000000000001416",
 17: "Patel, K. & Beedala, P. Calibration drift under cross-institutional deployment: an external validation framework for ICU mortality prediction across MIMIC-IV and eICU. Preprint at *medRxiv* https://doi.org/10.64898/2026.05.03.26352335 (2026).",
 18: "Collins, G. S. et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ* **385**, e078378 (2024). https://doi.org/10.1136/bmj-2023-078378",
 19: "la Roi-Teeuw, H. M. et al. Don't be misled: 3 misconceptions about external validation of clinical prediction models. *J Clin Epidemiol* **172**, 111387 (2024). https://doi.org/10.1016/j.jclinepi.2024.111387",
 20: "Davis, S. E. et al. A nonparametric updating method to correct clinical prediction model drift. *J Am Med Inform Assoc* **26**, 1448–1457 (2019). https://doi.org/10.1093/jamia/ocz058",
}

# ---- 1. order of first appearance ----
order = []
for m in re.finditer(r"\[\[([0-9,\-\s]+)\]\]", body):
    for tok in m.group(1).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-"); a, b = int(a), int(b)
            for x in range(a, b + 1):
                if x not in order:
                    order.append(x)
        else:
            x = int(tok)
            if x not in order:
                order.append(x)
missing = [o for o in order if o not in REF]
assert not missing, f"cited but no reference string: {missing}"
newnum = {old: i + 1 for i, old in enumerate(order)}
uncited = [o for o in REF if o not in order]
print("order of first appearance (old):", order)
print("new numbering:", newnum)
print("defined-but-uncited (should be empty):", uncited)

# ---- 2/3. replace tokens with superscript guillemets ----
def collapse(nums):
    nums = sorted(set(nums)); out = []; i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        if j - i >= 1:
            out.append(f"{nums[i]}–{nums[j]}")
        else:
            out.append(str(nums[i]))
        i = j + 1
    return ",".join(out)


def repl(m):
    olds = []
    for tok in m.group(1).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-"); olds += list(range(int(a), int(b) + 1))
        else:
            olds.append(int(tok))
    news = [newnum[o] for o in olds]
    return "\u00ab" + collapse(news) + "\u00bb"


final = re.sub(r"\[\[([0-9,\-\s]+)\]\]", repl, body)

# ---- 4. references list in new order ----
lines = ["## References", ""]
for old in order:
    lines.append(f"{newnum[old]}. {REF[old]}")
final = final.replace("@@REFERENCES@@", "\n".join(lines))

open("outputs/SCI_manuscript_v18.md", "w", encoding="utf-8").write(final)
json.dump({"order_old": order, "old_to_new": {str(k): v for k, v in newnum.items()}, "n_refs": len(order)},
          open("outputs/v18_reference_order.json", "w"), indent=2)
print("wrote outputs/SCI_manuscript_v18.md with", len(order), "references")
print("remaining [[ tokens:", len(re.findall(r"\[\[", final)))
print("remaining @@ markers:", final.count("@@REFERENCES@@"))