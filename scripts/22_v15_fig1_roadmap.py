"""22_v15_fig1_roadmap.py — Nature-style Figure 1: study design & analytical workflow.

six-module (a-f) roadmap built from the REAL cohort counts. Reviewer's key rule:
the within-eICU 50/50 update (E) uses DASHED connectors and is labelled "within-
database, NOT hospital-level validation"; the frozen model's cross-database
assessment (D) uses SOLID connectors = true external validation.
Boxes auto-size to their text so nothing overflows. Emits PNG (300 dpi) + PDF.
"""
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm

ROOT = Path(__file__).resolve().parents[1]; os.chdir(ROOT)
for fam in ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]:
    if any(fam.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = fam; break
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8})

OK = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73",
      "orange": "#E69F00", "sky": "#56B4E9", "purple": "#CC79A7",
      "grey": "#5A5A5A", "black": "#222222"}

W, H = 100.0, 168.0
fig = plt.figure(figsize=(7.0, 7.0 * H / W), dpi=300)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

TITLE_FS, LINE_FS, PAD, LH = 8.4, 7.4, 2.2, 3.7

def box(xc, ytop, w, lines, title=None, ec=OK["grey"], fc="#FFFFFF", lw=1.2, ls="-"):
    """Draw a box whose TOP is at ytop; auto-height from content. Returns bottom y."""
    n = len(lines)
    h = PAD*2 + (LH + 1.2 if title else 0) + LH*n
    y = ytop - h
    ax.add_patch(FancyBboxPatch((xc - w/2, y), w, h,
                 boxstyle="round,pad=0.4,rounding_size=1.0",
                 linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls, zorder=2))
    cy = ytop - PAD
    if title:
        ax.text(xc, cy, title, ha="center", va="top", fontsize=TITLE_FS,
                fontweight="bold", color=ec, zorder=3); cy -= (LH + 1.2)
    for ln in lines:
        ax.text(xc, cy, ln, ha="center", va="top", fontsize=LINE_FS,
                color=OK["black"], zorder=3); cy -= LH
    return y

def arrow(p1, p2, color=OK["grey"], ls="-", lw=1.5):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=11,
                 linewidth=lw, color=color, linestyle=ls, zorder=1, shrinkA=1, shrinkB=1))

def panel(x, y, ch):
    ax.text(x, y, ch, ha="left", va="top", fontsize=11.5, fontweight="bold", color=OK["black"], zorder=5)

# ---------- A: data sources ----------
panel(1.5, 167, "a")
ax.text(6, 166.2, "Retrospective data sources — publicly released, model-ready ICU heart-failure datasets",
        fontsize=8.6, fontweight="bold", va="top", color=OK["black"])
yA = box(19, 160, 33, ["n = 15,983 stays · 1,988 deaths", "role: model DEVELOPMENT"],
         "MIMIC-IV", OK["blue"])
box(50, 160, 33, ["n = 14,428 stays · 1,852 deaths", "role: EXTERNAL assessment"],
    "eICU-CRD", OK["vermillion"])
box(81, 160, 33, ["n = 1,176 stays · 159 deaths", "role: exploratory cross-version"],
    "MIMIC-III", OK["green"])

# ---------- B: predictor set ----------
panel(1.5, yA-3, "b")
yB = box(50, yA-4, 92,
         ["21 harmonised predictors  ·  outcome: in-hospital mortality",
          "age · sex · 4 vital signs · 13 laboratory values · COPD · diabetes",
          "NO organ-support, ventilation, vasoactive or HF-marker variables",
          "source-derived covariates WITHOUT verifiable timestamps → retrospective estimand"],
         None, OK["grey"], fc="#F4F4F4")
arrow((50, yA), (50, yB + (yA-yB)*0 + 0.1))  # short connector A->B
arrow((50, yB + 0.1), (50, yB))  # noop guard

# ---------- C: development ----------
panel(1.5, yB-3, "c")
yC = box(50, yB-5, 74,
         ["L2-penalised logistic regression (C = 1.0), linear terms",
          "→ FREEZE 21 coefficients + preprocessing (imputer, scaler)"],
         "Model development (MIMIC-IV)", OK["blue"])
arrow((50, yB), (50, yC + 12))

# ---------- D (external, solid) & E (within-database, dashed) ----------
panel(1.5, yC-3, "d")
yD = box(26, yC-6, 46,
         ["apply the FROZEN model to eICU-CRD", "(and exploratory MIMIC-III)",
          "AUC · Brier · calibration slope · O/E", "→ TRUE external validation"],
         "Cross-database assessment", OK["vermillion"], ls="-")
panel(53.5, yC-3, "e")
yE = box(76, yC-6, 46,
         ["update half (50%) → fit strategies", "held-out half (50%) → evaluate",
          "intercept-only · 2-param recalib · refit", "10 repeated splits × 6 sub-updates",
          "→ WITHIN-database, NOT hospital-level"],
         "Within-eICU updating + held-out", OK["sky"], ls="--")
arrow((38, yC), (30, yC-6), color=OK["vermillion"], ls="-")
arrow((62, yC), (70, yC-6), color=OK["sky"], ls="--")

# ---------- f: outputs ----------
panel(1.5, min(yD, yE)-3, "f")
ax.text(6, min(yD, yE)-3.8, "Analytical outputs", fontsize=8.6, fontweight="bold", va="top", color=OK["black"])
yF0 = min(yD, yE) - 6
box(13, yF0, 23, ["(Fig. 4; Table 3)"], "Update strategies", OK["purple"])
box(37, yF0, 23, ["(Fig. 3; Suppl. S1)"], "Sample-size stability", OK["purple"])
box(61, yF0, 23, ["(Fig. 4; Table 3)"], "Severity benchmarks", OK["purple"])
box(85, yF0, 23, ["(Fig. 5)"], "Reweighting", OK["purple"])
for xc in (13, 37, 61, 85):
    src = 26 if xc < 50 else 76
    arrow((src, min(yD, yE)), (xc, yF0 + 6), color=OK["purple"], lw=1.1)

# ---------- interpretation boundary ----------
yG = box(50, yF0 - 17, 96,
         ["retrospective predictor timing (no timestamp) ·",
          "no independent hospital-level or prospective updating validation ·",
          "possible correlated repeat stays (no patient/hospital id in MIMIC-IV/eICU files)"],
         "Interpretation boundary (kept explicit throughout):", OK["grey"], fc="#FBF3EC", lw=1.0)

# ---------- line-style legend ----------
yl = yG - 6
ax.plot([6, 13], [yl, yl], color=OK["vermillion"], lw=1.8)
ax.text(15, yl, "solid = cross-database external validation (frozen model)", fontsize=7, va="center")
ax.plot([6, 13], [yl-5, yl-5], color=OK["sky"], lw=1.8, ls="--")
ax.text(15, yl-5, "dashed = within-eICU stay-level held-out update (NOT external validation)", fontsize=7, va="center")

fig.savefig("figures/v15_fig1_roadmap.png", dpi=300, bbox_inches="tight", pad_inches=0.15)
fig.savefig("figures/v15_fig1_roadmap.pdf", bbox_inches="tight", pad_inches=0.15)
plt.close(fig)
print("wrote figures/v15_fig1_roadmap.png and .pdf")
