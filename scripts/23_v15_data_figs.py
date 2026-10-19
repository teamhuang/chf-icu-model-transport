"""23_v15_data_figs.py — Nature-styled evidence-chain figures (Fig 2-5) + S2/S3,
all generated from the REAL data/ and outputs/ artefacts (no invented numbers).

Fig 2  Cross-database transport & calibration drift (A ROC, B calibration, C metrics)
Fig 3  Recalibration stability by update size (A attainment+CI, B tolerance sensitivity, C slope distribution)
Fig 4  Updating strategies (A AUC, B Brier, C slope, D held-out calibration curves)
Fig 5  Covariate shift & reweighting (A SMD love plot, B propensity overlap, C weight dist + ESS)
Fig S2 Held-out slope distribution + regularisation before/after (S7/S11)
Fig S3 Reweighting tail deviations + truncation sensitivity (S2/S3)

Nature style: Arial, Okabe-Ito colour-blind-safe palette, thin spines, bold panel letters,
PNG 300 dpi + editable PDF. Each figure is independent (one failure won't block the rest).
"""
import os, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]; os.chdir(ROOT)
for fam in ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]:
    import matplotlib.font_manager as fm
    if any(fam.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = fam; break
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "xtick.major.width": 0.8, "ytick.major.width": 0.8})
OK = {"blue":"#0072B2","vermillion":"#D55E00","green":"#009E73","orange":"#E69F00",
      "sky":"#56B4E9","purple":"#CC79A7","yellow":"#F0E442","grey":"#5A5A5A","black":"#222222"}
COMMON=["Age","male","HeartRate","RespiratoryRate","Temperature","SBP","AG","BUN","Creatinine",
        "Chloride","MCH","MCHC","MCV","Platelet","Potassium","RBC","RDW","Sodium","WBC","COPD","Diabetes"]

def panel(ax, ch):
    ax.text(-0.16, 1.06, ch, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="left")

def save(fig, name):
    fig.savefig(f"figures/{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(f"figures/{name}.pdf", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig); print("  wrote", name)

def load():
    """Canonical loader identical to analysis_v5.py: rename MIMIC-III cols to COMMON
    scheme, drop rows with missing outcome BEFORE casting to int."""
    a=pd.read_csv("data/data01.csv").rename(columns={
        "heart rate":"HeartRate","Respiratory rate":"RespiratoryRate","temperature":"Temperature",
        "Systolic blood pressure":"SBP","Anion gap":"AG","Urea nitrogen":"BUN","Blood potassium":"Potassium",
        "Blood sodium":"Sodium","Leucocyte":"WBC","Platelets":"Platelet","age":"Age"})
    a["male"]=(a["gendera"]==1).astype(int); a["Diabetes"]=a["diabetes"].astype(int)
    a=a[a["outcome"].notna()].copy(); a["outcome"]=a["outcome"].astype(int)
    b=pd.read_csv("data/Derivation cohorts--MIMIC-IV database.csv")
    b["male"]=(b["Gender"]==1).astype(int); b["outcome"]=b["Hstatus"].astype(int)
    v=pd.read_csv("data/Validation cohorts--eICU-CRD.csv")
    v["male"]=(v["Gender"]==1).astype(int); v["outcome"]=v["Hstatus"].astype(int)
    return b,v,a

def frozen_probs(b,v):
    """Refit the identical frozen MIMIC-IV pipeline (matches analysis_v5.py) so every
    curve is produced by the same estimator that generated the reported metrics."""
    pipe=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                   ("lr",LogisticRegression(max_iter=2000))]).fit(b[COMMON].astype(float), b["outcome"].values)
    return lambda df: pipe.predict_proba(df[COMMON].astype(float))[:,1]

def calib_bins(y,p,nb=10):
    q=pd.qcut(pd.Series(p), nb, labels=False, duplicates="drop")
    d=pd.DataFrame({"y":np.asarray(y),"p":np.asarray(p),"q":q}).groupby("q")
    return d["p"].mean().values, d["y"].mean().values, d.size().values

# ================= FIG 2 : transport =================
def fig2():
    b,v,m3=load(); pr=frozen_probs(b,v)
    layers=[("MIMIC-IV (development)",b,OK["blue"]),("eICU-CRD (external)",v,OK["vermillion"]),
            ("MIMIC-III (exploratory)",m3,OK["green"])]
    fig=plt.figure(figsize=(7.0,2.6)); gs=gridspec.GridSpec(1,3,width_ratios=[1,1,1.05],wspace=0.34)
    axA=fig.add_subplot(gs[0]); axB=fig.add_subplot(gs[1]); axC=fig.add_subplot(gs[2])
    for nm,df,c in layers:
        y=df["outcome"].values; p=pr(df)
        fpr,tpr,_=roc_curve(y,p); axA.plot(fpr,tpr,color=c,lw=1.3,label=f"{nm.split()[0]} {roc_auc_score(y,p):.3f}")
    axA.plot([0,1],[0,1],ls=":",color=OK["grey"],lw=0.8); axA.set_xlabel("1 − specificity"); axA.set_ylabel("Sensitivity")
    axA.legend(fontsize=6,frameon=False,loc="lower right"); axA.set_title("ROC",fontsize=8.5); panel(axA,"A")
    for nm,df,c in layers:
        y=df["outcome"].values; p=pr(df); mp,my,sz=calib_bins(y,p)
        axB.plot(mp,my,"o-",color=c,ms=4,lw=1.1,label=nm.split()[0])
    axB.plot([0,1],[0,1],ls=":",color=OK["grey"],lw=0.8); axB.set_xlabel("Predicted risk"); axB.set_ylabel("Observed risk")
    axB.set_title("Calibration",fontsize=8.5); axB.legend(fontsize=6,frameon=False); panel(axB,"B")
    tr=json.load(open("outputs/v5_results.json",encoding="utf-8"))["transport"]
    names=["MIMIC-IV","MIMIC-III","eICU"]; cols=[OK["blue"],OK["green"],OK["vermillion"]]
    keys=["A_MIMIC-III_temporal_exploratory","V_eICU_geographic_primary","B_MIMIC-IV_insample"]
    order=[("B_MIMIC-IV_insample","MIMIC-IV",OK["blue"]),("V_eICU_geographic_primary","eICU",OK["vermillion"]),("A_MIMIC-III_temporal_exploratory","MIMIC-III",OK["green"])]
    slope=[tr[k]["calib_slope"] for k,_,_ in order]; oe=[tr[k]["OE"] for k,_,_ in order]
    x=np.arange(3); w=0.38
    axC.bar(x-w/2,slope,w,color=[c for _,_,c in order],label="slope")
    axC.bar(x+w/2,oe,w,color=[c for _,_,c in order],alpha=0.45,hatch="//",label="O/E")
    axC.axhline(1,ls=":",color=OK["grey"],lw=0.8); axC.set_xticks(x); axC.set_xticklabels([n for _,n,_ in order],fontsize=7)
    axC.set_ylabel("value"); axC.set_title("Calibration metrics",fontsize=8.5); axC.legend(fontsize=6,frameon=False); panel(axC,"C")
    save(fig,"v15_fig2_transport")

# ================= FIG 3 : stability =================
def fig3():
    rc=pd.read_csv("outputs/v5_repeated_curve.csv"); diag=json.load(open("outputs/sr_diag_v10.json",encoding="utf-8"))["slope_and_events"]
    sens=json.load(open("outputs/sr_sens_v9.json",encoding="utf-8"))["threshold_sensitivity"]
    SIZES=[100,200,500,1000,2000,5000]
    fig=plt.figure(figsize=(7.0,2.6)); gs=gridspec.GridSpec(1,3,width_ratios=[1,1.1,1.1],wspace=0.36)
    axA=fig.add_subplot(gs[0]); axB=fig.add_subplot(gs[1]); axC=fig.add_subplot(gs[2])
    hit=[100*np.mean(rc[rc.n_pat==n].stable.values) for n in SIZES]
    axA.plot(SIZES,hit,"o-",color=OK["blue"],ms=5,lw=1.4)
    axA.set_xscale("log"); axA.set_xticks(SIZES); axA.set_xticklabels([f"{n:,}" for n in SIZES],fontsize=6,rotation=30)
    axA.set_ylabel("target-hit rate (%)"); axA.set_xlabel("update size (stays)"); axA.set_ylim(0,105)
    axA.set_title("Attainment vs size",fontsize=8.5); panel(axA,"A")
    for st,oe,c in [(0.1,0.1,OK["vermillion"]),(0.2,0.15,OK["blue"]),(0.3,0.2,OK["green"])]:
        row=[r for r in sens if abs(r["slope_tol"]-st)<1e-6 and abs(r["oe_tol"]-oe)<1e-6]
        if row: r=row[0]; axB.plot(SIZES,[r[f"rate@{n}"] for n in SIZES],"o-",ms=4,lw=1.2,color=c,label=f"±{st}/{oe}")
    axB.set_xscale("log"); axB.set_xticks(SIZES); axB.set_xticklabels([f"{n:,}" for n in SIZES],fontsize=6,rotation=30)
    axB.set_ylabel("hit rate (%)"); axB.set_xlabel("update size"); axB.legend(fontsize=6,frameon=False,title="slope/OE tolerance",title_fontsize=6)
    axB.set_title("Tolerance sensitivity",fontsize=8.5); panel(axB,"B")
    data=[rc[rc.n_pat==n].slope.values for n in SIZES]
    parts=axC.violinplot(data,showmedians=True); 
    for pc in parts["bodies"]: pc.set_facecolor(OK["sky"]); pc.set_alpha(0.6)
    for k in ("cbars","cmins","cmaxes","cmeans","cwhiskers"):
        if k in parts: parts[k].set_color(OK["grey"])
    axC.axhline(1,ls=":",color=OK["grey"],lw=0.8); axC.axhline(0,ls=":",color=OK["vermillion"],lw=0.8)
    axC.set_yscale("symlog",linthresh=1); axC.set_xticks(range(1,7)); axC.set_xticklabels([f"{n:,}" for n in SIZES],fontsize=6,rotation=30)
    axC.set_ylabel("held-out slope"); axC.set_xlabel("update size"); axC.set_title("Slope distribution",fontsize=8.5); panel(axC,"C")
    save(fig,"v15_fig3_stability")

# ================= FIG 4 : strategies =================
def fig4():
    sc=json.load(open("outputs/v5_results.json",encoding="utf-8"))["strategy_comparison"]
    order=[("frozen","Frozen",OK["grey"]),("intercept_only","Intercept-only",OK["orange"]),
           ("recalibrate_2par","2-param recal",OK["sky"]),("full_refit","Refit",OK["blue"])]
    fig=plt.figure(figsize=(7.0,2.5)); gs=gridspec.GridSpec(1,4,width_ratios=[1,1,1,1.15],wspace=0.42)
    axs=[fig.add_subplot(gs[i]) for i in range(4)]
    for ax,metric,ttl,fmt in [(axs[0],"AUC","AUC",".3f"),(axs[1],"Brier","Brier",".3f"),(axs[2],"slope","Calib. slope",".2f")]:
        vals=[sc[k][metric][0] for k,_,_ in order]; errs=[sc[k][metric][1] for k,_,_ in order]
        ax.bar(range(4),vals,0.62,yerr=errs,capsize=2,color=[c for _,_,c in order],error_kw=dict(lw=0.8))
        ax.set_xticks(range(4)); ax.set_xticklabels([n for _,n,_ in order],fontsize=5.5,rotation=40,ha="right")
        ax.set_title(ttl,fontsize=8); panel(ax,"ABCD"[axs.index(ax)])
    # panel D: held-out calibration curves (recompute representative split)
    b,v,m3=load(); pr=frozen_probs(b,v); half=len(v)//2; cr=np.random.RandomState(2026); p_=cr.permutation(len(v))
    Vu=v.iloc[p_[:half]].reset_index(drop=True); Vt=v.iloc[p_[half:]].reset_index(drop=True)
    yu,yt=Vu.outcome.values,Vt.outcome.values; lg=lambda p: np.log(np.clip(p,1e-9,1-1e-9)/np.clip(1-p,1e-9,1-1e-9))
    pf_f=pr(Vt); lp_u=lg(pr(Vu))
    m=LogisticRegression(max_iter=2000).fit(lp_u[:,None],yu); pf_r=1/(1+np.exp(-(m.intercept_[0]+m.coef_[0][0]*lg(pf_f))))
    lr=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("lr",LogisticRegression(max_iter=2000))]).fit(Vu[COMMON].astype(float),yu)
    pf_ref=lr.predict_proba(Vt[COMMON].astype(float))[:,1]
    ms=LogisticRegression(max_iter=2000).fit(Vu[["SOFA"]].astype(float),yu); pf_sofa=ms.predict_proba(Vt[["SOFA"]].astype(float))[:,1]
    axD=axs[3]
    for nm,p,c in [("frozen",pf_f,OK["grey"]),("recal",pf_r,OK["sky"]),("refit",pf_ref,OK["blue"]),("SOFA",pf_sofa,OK["vermillion"])]:
        mp,my,sz=calib_bins(yt,p); axD.plot(mp,my,"o-",ms=3,lw=1.0,color=c,label=nm)
    axD.plot([0,1],[0,1],ls=":",color=OK["grey"],lw=0.8); axD.set_xlabel("Predicted risk",fontsize=7)
    axD.set_ylabel("Observed risk",fontsize=7); axD.set_title("Held-out calibration",fontsize=8)
    axD.legend(fontsize=5.5,frameon=False); panel(axD,"D")
    save(fig,"v15_fig4_strategies")

# ================= FIG 5 : reweighting =================
def fig5():
    b,v,m3=load(); wd=json.load(open("outputs/sr_sens_v9.json",encoding="utf-8"))["weight_diagnostics"]
    smd=pd.read_csv("outputs/v6_smd_by_variable.csv")
    fig=plt.figure(figsize=(7.0,2.7)); gs=gridspec.GridSpec(1,3,width_ratios=[1.1,1,1],wspace=0.40)
    axA=fig.add_subplot(gs[0]); axB=fig.add_subplot(gs[1]); axC=fig.add_subplot(gs[2])
    # A: love plot
    smd=smd.reindex(smd["SMD_unweighted"].abs().sort_values(ascending=True).index)
    yy=np.arange(len(smd))
    axA.hlines(yy,0,smd["SMD_unweighted"],color=OK["grey"],lw=1.2)
    axA.plot(smd["SMD_unweighted"],yy,"o",ms=4,color=OK["vermillion"],label="unweighted")
    axA.plot(smd["SMD_weighted"],yy,"o",ms=4,color=OK["blue"],label="weighted")
    axA.axvline(0.1,ls=":",color=OK["grey"],lw=0.8); axA.set_yticks(yy); axA.set_yticklabels(smd["Variable"],fontsize=5)
    axA.set_xlabel("absolute SMD"); axA.set_title("Covariate balance",fontsize=8.5); axA.legend(fontsize=6,frameon=False); panel(axA,"A")
    # B: propensity overlap (recompute)
    X=pd.concat([b[COMMON].assign(_d=0)[COMMON], v[COMMON].assign(_d=1)[COMMON]])
    lab=np.r_[np.zeros(len(b)),np.ones(len(v))]
    sc=StandardScaler().fit_transform(X.astype(float).values)
    ps=LogisticRegression(max_iter=2000).fit(sc,lab).predict_proba(sc)[:,1]
    bins=np.linspace(0,1,41)
    axB.hist(ps[lab==0],bins=bins,alpha=0.6,color=OK["blue"],label="MIMIC-IV",density=True)
    axB.hist(ps[lab==1],bins=bins,alpha=0.6,color=OK["vermillion"],label="eICU",density=True)
    axB.set_xlabel("P(eICU | X)",fontsize=7); axB.set_ylabel("density",fontsize=7)
    axB.set_title(f"Overlap (sep. AUC {wd['propensity_separability_AUC']:.3f})",fontsize=8)
    axB.legend(fontsize=6,frameon=False); panel(axB,"B")
    # C: weight distribution + ESS
    pi_e=ps[lab==1]; wgt=((1-pi_e)/pi_e)*(len(v)/len(b)); wgt=np.clip(wgt,0.1,10)
    axC.hist(wgt,bins=np.linspace(0,10,41),color=OK["purple"],alpha=0.8)
    axC.axvline(0.1,color=OK["grey"],ls=":",lw=0.8); axC.axvline(10,color=OK["grey"],ls=":",lw=0.8)
    axC.set_xlabel("truncated weight",fontsize=7); axC.set_ylabel("eICU stays",fontsize=7)
    axC.set_title(f"ESS {wd['ESS_pct']:.1f}% · extreme {wd['extreme_weight_fraction']*100:.0f}%",fontsize=8); panel(axC,"C")
    save(fig,"v15_fig5_reweighting")

# ================= FIG S2 : slope dist + regularisation =================
def figS2():
    rc=pd.read_csv("outputs/v5_repeated_curve.csv"); v12=json.load(open("outputs/sr_v12.json",encoding="utf-8"))["penalised_recalibration_sensitivity"]
    SIZES=[100,200,500,1000,2000,5000]
    fig=plt.figure(figsize=(6.6,2.6)); gs=gridspec.GridSpec(1,2,width_ratios=[1.2,1],wspace=0.32)
    axA=fig.add_subplot(gs[0]); axB=fig.add_subplot(gs[1])
    data=[rc[rc.n_pat==n].slope.values for n in SIZES]
    parts=axA.violinplot(data,showmedians=True)
    for pc in parts["bodies"]: pc.set_facecolor(OK["sky"]); pc.set_alpha(0.6)
    for k in ("cbars","cmins","cmaxes"): parts[k].set_color(OK["grey"])
    axA.axhline(1,ls=":",color=OK["grey"],lw=0.8); axA.axhline(0,ls=":",color=OK["vermillion"],lw=0.8)
    axA.set_yscale("symlog",linthresh=1); axA.set_xticks(range(1,7)); axA.set_xticklabels([f"{n:,}" for n in SIZES],fontsize=6,rotation=30)
    axA.set_ylabel("held-out slope"); axA.set_xlabel("update size"); axA.set_title("Unpenalised slope distribution (S7)",fontsize=8); panel(axA,"A")
    x=np.arange(len(SIZES)); w=0.38
    un=[v12[str(n)]["heldout_median_unpen"] for n in SIZES]; rb=[v12[str(n)]["heldout_median_robust"] for n in SIZES]
    umx=[v12[str(n)]["heldout_max_unpen"] for n in SIZES]; rmx=[v12[str(n)]["heldout_max_robust"] for n in SIZES]
    axB.plot(x,un,"o-",color=OK["vermillion"],ms=4,label="unpenalised median")
    axB.plot(x,rb,"s-",color=OK["green"],ms=4,label="penalised median")
    axB.plot(x,umx,"o--",color=OK["vermillion"],ms=3,alpha=0.5,label="unpenalised max")
    axB.plot(x,rmx,"s--",color=OK["green"],ms=3,alpha=0.5,label="penalised max")
    axB.axhline(1,ls=":",color=OK["grey"],lw=0.8); axB.set_yscale("log")
    axB.set_xticks(x); axB.set_xticklabels([f"{n:,}" for n in SIZES],fontsize=6,rotation=30)
    axB.set_ylabel("held-out slope (log)"); axB.set_xlabel("update size")
    axB.set_title("Regularisation trade-off (S11)",fontsize=8); axB.legend(fontsize=5.5,frameon=False); panel(axB,"B")
    save(fig,"v15_figS2_slope_reg")

# ================= FIG S3 : tails + truncation =================
def figS3():
    wd=json.load(open("outputs/sr_sens_v9.json",encoding="utf-8"))["weight_diagnostics"]
    b,v,m3=load()
    # recompute the correct (0.1-10 clipped) IPW weights exactly as analysis_v5.py
    Xp=pd.concat([b[COMMON].astype(float),v[COMMON].astype(float)],ignore_index=True)
    isV=np.r_[np.zeros(len(b)),np.ones(len(v))]
    clf=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                  ("lr",LogisticRegression(max_iter=2000))]).fit(Xp,isV)
    ps=clf.predict_proba(v[COMMON].astype(float))[:,1]
    piV=len(v)/(len(v)+len(b)); piB=1-piV
    w=np.clip(((1-ps)/ps)*(piV/piB),0.1,10)
    def wq(x,weights,q):
        o=np.argsort(x); x=x[o]; w=weights[o]; cw=np.cumsum(w)/np.sum(w)
        return float(np.interp(q,cw,x))
    def sd(x): return float(np.std(x,ddof=1)) if np.std(x,ddof=1)>0 else 1.0
    KEYVARS=["HeartRate","RespiratoryRate","Age","AG","RBC","COPD"]
    p10=[];p50=[];p90=[]
    for c in KEYVARS:
        xb=b[c].astype(float).values; xv=v[c].astype(float).values; s=sd(xb)
        p10.append((wq(xv,w,0.10)-np.percentile(xb,10))/s)
        p50.append((wq(xv,w,0.50)-np.percentile(xb,50))/s)
        p90.append((wq(xv,w,0.90)-np.percentile(xb,90))/s)
    fig=plt.figure(figsize=(6.6,2.6)); gs=gridspec.GridSpec(1,2,width_ratios=[1,1.1],wspace=0.34)
    axA=fig.add_subplot(gs[0]); axB=fig.add_subplot(gs[1])
    x=np.arange(len(KEYVARS)); ww=0.26
    axA.bar(x-ww,p10,ww,color=OK["sky"],label="p10")
    axA.bar(x,p50,ww,color=OK["blue"],label="p50")
    axA.bar(x+ww,p90,ww,color=OK["vermillion"],label="p90")
    axA.axhline(0,color=OK["grey"],lw=0.8); axA.set_xticks(x); axA.set_xticklabels(KEYVARS,fontsize=6,rotation=30)
    axA.set_ylabel("weighted − MIMIC-IV\nquantile gap (SD units)",fontsize=7)
    axA.set_title("Residual tail imbalance after reweighting",fontsize=8)
    axA.legend(fontsize=6,frameon=False); panel(axA,"A")
    # truncation sensitivity: sourced directly from v6_results weight_sensitivity (table-consistent)
    ws=json.load(open("outputs/v6_results.json",encoding="utf-8"))["weight_sensitivity"]
    keys=["0.1-10","1-5","p1p99"]; lab=["0.1–10","1–5","p1/p99"]
    slope=[ws[k]["weighted_slope"] for k in keys]; ess=[ws[k]["ESS_pct"] for k in keys]
    axB.bar(lab,slope,0.55,color=[OK["blue"],OK["sky"],OK["purple"]])
    axB.axhline(1,ls=":",color=OK["grey"],lw=0.8); axB.set_ylim(0,0.6)
    axB.set_ylabel("weighted slope"); axB.set_xlabel("weight truncation")
    for i,e in enumerate(ess): axB.text(i,slope[i]+0.01,f"ESS {e:.0f}%",ha="center",fontsize=6)
    axB.set_title("Truncation sensitivity (S3)",fontsize=8); panel(axB,"B")
    save(fig,"v15_figS3_tails_trunc")

if __name__=="__main__":
    for fn in [fig2,fig3,fig4,fig5,figS2,figS3]:
        try: fn()
        except Exception as e:
            import traceback; print("  FAIL",fn.__name__,repr(e)); traceback.print_exc()
    print("done")
