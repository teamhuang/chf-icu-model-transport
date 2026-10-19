"""30_v18_tripod.py — build SUPPLEMENT_TRIPOD_AI_v18.md from the v17 map with the
round-18 corrections applied. Kept as a script so the corrections are reproducible
and auditable rather than hand-edited.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; os.chdir(ROOT)
t = open("outputs/SUPPLEMENT_TRIPOD_AI_v17.md", encoding="utf-8").read()

SUBS = [
    ("(analysis build 2026-10-12.r17)", "(analysis build 2026-10-12.r18)"),
    ("Unstructured summary (176 words)", "Unstructured summary (161 words)"),
    ("| Participants — characteristics & outcome | Table 1; Suppl. Table S4 | Per-layer stay counts, deaths, prevalence; per-variable dictionary. | ✅ |",
     "| Participants — characteristics & outcome | **Table 1**; Suppl. Table S15 | **New this round.** Main-text Table 1 now reports stays, deaths, mortality, age, sex and the predictors with the largest between-cohort imbalance, each with its standardised mean difference; Supplementary Table S15 gives all 21 predictors for all three cohorts with per-cohort row counts. | ✅ |"),
    ("| Participants — flow | Table 1; Suppl. Table S9 |",
     "| Participants — flow | Table 1; Suppl. Table S11 |"),
    ("| Model performance | Tables 1, 3, 4; Suppl. Table S13; Fig. 2b, 4 |",
     "| Model performance | Tables 2, 3; Suppl. Tables S16, S17, S18; Fig. 2, 4 |"),
    ("the transport ROC/calibration panels are Fig. 2.", "the transport ROC/calibration panels are Fig. 2."),
    ("| Model performance — subgroups | Suppl. Table S12; Discussion | Sex and age-tertile AUC/slope/O/E for the frozen model; **descriptive only — no per-stratum CIs or interaction test**. | ⚠️ |",
     "| Model performance — subgroups | Suppl. Table S12; Discussion | Sex, age-tertile **and coded race/ethnicity** AUC/slope/O/E for the frozen model (race added this round); **descriptive only — no per-stratum CIs or interaction test**. | ⚠️ |"),
    ("| Statistical analysis | Methods: Statistical analysis | Preprocessing (median impute, standardise, fit only on update half); L2 logistic C=1.0; unpenalised IRLS for calibration; slope not constrained positive; penalised sensitivity; bootstrap definitions; software = Python/scikit-learn (requirements.txt). | ✅ |",
     "| Statistical analysis | Methods: Statistical analysis | Preprocessing (median impute, standardise, fit only on update half); L2 logistic C=1.0. **Corrected this round:** the two-parameter *update* is fitted with L2 logistic regression (C = 1.0) and the Methods now say so, separately from the estimator used to *report* calibration (unpenalised IRLS on the linear predictor); the penalty's numerical effect on the fitted slope is quantified in the Methods. Slope not constrained positive; penalised sensitivity; 500-replicate bootstrap intervals for every calibration quantity; software = Python/scikit-learn (requirements.txt). | ✅ |"),
    ("| Outcomes | Methods: Outcome and predictors |",
     "| Outcomes | Methods: Outcome and predictors |"),
    ("| Data availability | Declarations |", "| Data availability | Additional Information |"),
    ("| Funding | Declarations | Placeholder. | ▢ [AUTHOR-SPECIFIC] |",
     "| Funding | Additional Information | Placeholder. | ▢ [AUTHOR-SPECIFIC] |"),
    ("| Code availability | Declarations |", "| Code availability | Additional Information |"),
    ("| Authorship / COI | Declarations |", "| Authorship / COI | Additional Information |"),
    ("| Ethics | Methods: Ethics |", "| Ethics | Methods: Ethics; Additional Information |"),
]
for a, b in SUBS:
    n = t.count(a)
    print(("OK  " if n else "MISS"), n, a[:72])
    t = t.replace(a, b)

FOOTER_OLD = ("**Author actions before submission (▢):** transcribe this map onto the official numbered "
              "TRIPOD+AI checklist PDF with page/line numbers; mint the versioned code DOI; complete funding, "
              "ethics determination, AI-tool version/dates, author list/ORCID/CRediT and all-author COI "
              "confirmation. These are intentionally left as placeholders and not fabricated.")
FOOTER_NEW = (
 "**New in round 18.** (i) Participant characteristics are now a real main-text Table 1 with a full "
 "Supplementary Table S15, replacing an overclaimed ✅. (ii) The Methods now state the estimator actually "
 "used to *fit* the recalibration (L2, C = 1.0) separately from the estimator used to *report* calibration "
 "(unpenalised IRLS), with the penalty's numerical effect given. (iii) Stay-level bootstrap intervals added "
 "for every headline calibration quantity (Supplementary Table S17). (iv) Missingness is now reported for all "
 "three cohorts: MIMIC-IV and eICU are complete, MIMIC-III has 21 of 1,176 rows (1.8%) with missing vital "
 "signs that are median-imputed (Supplementary Table S5) — v17 generalised \"0% missing\" to all three files, "
 "which was wrong. (v) Race/ethnicity subgroups added (Supplementary Table S12). (vi) A new transport "
 "diagnostic — the linear-predictor dispersion ratio as an expectation for the calibration slope — added and "
 "reported in full (Supplementary Table S18). (vii) The stability section now reports three explicitly named "
 "attainment statistics plus the per-repetition minimum vector (Supplementary Table S14a/S14b). (viii) "
 "Declarations relocated to the Additional Information block in the publisher's required order. (ix) Display "
 "items reduced to 8 (5 figures + 3 tables) to meet the publisher's hard cap.\n\n" + FOOTER_OLD)
assert FOOTER_OLD in t
t = t.replace(FOOTER_OLD, FOOTER_NEW)

open("outputs/SUPPLEMENT_TRIPOD_AI_v18.md", "w", encoding="utf-8").write(t)
print("wrote outputs/SUPPLEMENT_TRIPOD_AI_v18.md")