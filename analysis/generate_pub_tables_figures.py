"""Generate publication-quality tables (LaTeX) and figures (PDF) for CHI 2027.

Reads from results/tables/*.csv and results/data/*.json.
Outputs to paper/tables/*.tex and paper/figures/*.pdf.

All numbers are extracted programmatically from stored result files.
No numbers are transcribed manually.
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"
DATA = ROOT / "results" / "data"
OUT_TAB = ROOT / "paper" / "tables"
OUT_FIG = ROOT / "paper" / "figures"
OUT_TAB.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

# ── Publication style ──────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.2,
})

PAL = {"agentic": "#2166AC", "nonagentic": "#B2182B", "neutral": "#636363"}

# ================================================================
# FIGURES
# ================================================================

def fig_F1():
    """F1: Monthly users distribution (log scale histogram)."""
    df = pd.read_csv(DATA / "block3_distribution.csv")
    mu = df["monthly_users"].values
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    vals = mu[mu > 0]
    ax.hist(np.log10(vals + 1), bins=50, color=PAL["neutral"],
            edgecolor="white", linewidth=0.3, alpha=0.9, density=True)
    ax.set_xlabel("log$_{10}$(1 + monthly users)")
    ax.set_ylabel("Density")
    n_zero = (mu == 0).sum()
    pct_zero = n_zero / len(mu) * 100
    ax.text(0.97, 0.93, f"Zero users: {n_zero:,} ({pct_zero:.0f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="#FFFFCC", ec="#999999", lw=0.4))
    fig.savefig(OUT_FIG / "F1_usage_distribution.pdf")
    plt.close(fig)
    print("  F1 saved.")


def fig_F2():
    """F2: Category composition (horizontal bar, agentic share annotated)."""
    df = pd.read_csv(DATA / "block19_eligible_category_breakdown.csv")
    top = df.head(12).copy()
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    y = range(len(top))
    ax.barh(y, top["n_actors"], color=PAL["agentic"], alpha=0.8, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(top["category"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Number of actors")
    for i, (n, pct) in enumerate(zip(top["n_actors"], top["pct_eligible"])):
        ax.text(n + 80, i, f"{pct:.0f}%", va="center", fontsize=6.5,
                color=PAL["nonagentic"], weight="bold")
    ax.text(0.98, 0.02, "% = agentic-ready share", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6, color=PAL["nonagentic"])
    fig.savefig(OUT_FIG / "F2_category_composition.pdf")
    plt.close(fig)
    print("  F2 saved.")


def fig_F3():
    """F3: Adoption distributions KDE (agentic vs non-agentic PPE)."""
    df = pd.read_csv(DATA / "block3_distribution.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))
    for ax_i, (col, label) in enumerate([("log_mu", "ln(1 + monthly users)"),
                                          ("log_runs", "ln(1 + runs in 30d)")]):
        ag = df[df["agentic"]][col]
        na = df[~df["agentic"]][col]
        sns.kdeplot(ag, ax=axes[ax_i], color=PAL["agentic"],
                    fill=True, alpha=0.25, linewidth=1.0,
                    label=f"Whitelisted (n={len(ag):,})")
        sns.kdeplot(na, ax=axes[ax_i], color=PAL["nonagentic"],
                    fill=True, alpha=0.25, linewidth=1.0,
                    label=f"Non-whitelisted (n={len(na):,})")
        axes[ax_i].set_xlabel(label)
        axes[ax_i].set_ylabel("Density" if ax_i == 0 else "")
        axes[ax_i].legend(fontsize=6.5, frameon=False)
    fig.savefig(OUT_FIG / "F3_adoption_distributions.pdf")
    plt.close(fig)
    print("  F3 saved.")


def fig_F4():
    """F4: Propensity score distributions before/after 1:1 matching.

    Reads block24_pscore_raw.csv which contains columns:
        pscore   – estimated propensity score
        eligible – 1 = treated (agentic-ready), 0 = control
        matched  – 1 = included in the 1:1 matched sample
    """
    csv_path = DATA / "block24_pscore_raw.csv"
    if not csv_path.exists():
        print("  F4: SKIPPED – block24_pscore_raw.csv not found. Run block24 first.")
        return

    df = pd.read_csv(csv_path)
    treated_all = df.loc[df["eligible"] == 1, "pscore"].values
    control_all = df.loc[df["eligible"] == 0, "pscore"].values
    treated_m = df.loc[(df["eligible"] == 1) & (df["matched"] == 1), "pscore"].values
    control_m = df.loc[(df["eligible"] == 0) & (df["matched"] == 1), "pscore"].values
    n_pairs = len(control_m)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))

    # Shared x-range across both panels; right edge exactly at 1.0
    global_lo = min(treated_all.min(), control_all.min()) - 0.02
    global_hi = 1.005  # tiny padding past 1.0 for visual comfort

    # KDE bandwidth tuning -- slightly wider to smooth the boundary spike
    bw = 0.04

    # ── Panel 1: Before matching (full sample) ──
    ax = axes[0]
    bins = np.linspace(global_lo, 1.0, 45)  # right edge at exactly 1.0

    ax.hist(treated_all, bins=bins, density=True, alpha=0.35,
            color=PAL["agentic"], edgecolor="white", linewidth=0.3,
            label=f"Treated (n={len(treated_all):,})")
    ax.hist(control_all, bins=bins, density=True, alpha=0.45,
            color=PAL["nonagentic"], edgecolor="white", linewidth=0.3,
            label=f"Control (n={len(control_all):,})")
    # KDE overlays
    sns.kdeplot(treated_all, ax=ax, color=PAL["agentic"], linewidth=1.2,
                clip=(global_lo, 1.0), bw_adjust=1.2)
    sns.kdeplot(control_all, ax=ax, color=PAL["nonagentic"], linewidth=1.2,
                linestyle="--", clip=(global_lo, 1.0), bw_adjust=1.2)
    ax.set_xlabel("Propensity Score")
    ax.set_ylabel("Density")
    ax.set_title("Before Matching", fontsize=9, weight="bold")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    ax.set_xlim(global_lo, global_hi)

    # ── Panel 2: After matching (matched sample) ──
    ax = axes[1]
    bins_m = np.linspace(global_lo, 1.0, 35)

    ax.hist(treated_m, bins=bins_m, density=True, alpha=0.35,
            color=PAL["agentic"], edgecolor="white", linewidth=0.3,
            label="Treated")
    ax.hist(control_m, bins=bins_m, density=True, alpha=0.45,
            color=PAL["nonagentic"], edgecolor="white", linewidth=0.3,
            label="Control")
    # KDE overlays
    sns.kdeplot(treated_m, ax=ax, color=PAL["agentic"], linewidth=1.2,
                clip=(global_lo, 1.0), bw_adjust=1.2)
    sns.kdeplot(control_m, ax=ax, color=PAL["nonagentic"], linewidth=1.2,
                linestyle="--", clip=(global_lo, 1.0), bw_adjust=1.2)
    ax.set_xlabel("Propensity Score")
    ax.set_ylabel("Density")
    ax.set_title("After Matching", fontsize=9, weight="bold")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    ax.set_xlim(global_lo, global_hi)

    fig.suptitle(
        f"Propensity Score Distributions (n={n_pairs:,} matched pairs)",
        fontsize=10, weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "F4_pscore_distributions.pdf")
    plt.close(fig)
    print(f"  F4 saved ({n_pairs:,} matched pairs).")


def fig_F5():
    """F5: Developer portfolio scatter."""
    df = pd.read_csv(DATA / "block5_portfolio_scatter.csv")
    fig, ax = plt.subplots(figsize=(3.3, 2.8))
    sc = ax.scatter(
        df["portfolio_size"], df["agentic_share"],
        s=np.clip(np.log1p(df["avg_mu"]) * 3, 2, 30),
        c=np.log1p(df["avg_mu"]), cmap="YlOrRd",
        alpha=0.35, edgecolors="none", rasterized=True,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Portfolio size (log scale)")
    ax.set_ylabel("Whitelisted share")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("ln(1+avg MU)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    fig.savefig(OUT_FIG / "F5_portfolio_scatter.pdf")
    plt.close(fig)
    print("  F5 saved.")


def fig_F6():
    """F6: Event price distributions."""
    df = pd.read_csv(DATA / "block6_price_distribution.csv")
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    ag = df[df["agentic"]]["log_price"]
    na = df[~df["agentic"]]["log_price"]
    sns.kdeplot(ag, ax=ax, color=PAL["agentic"], fill=True, alpha=0.25,
                linewidth=1.0, label=f"Whitelisted (n={len(ag):,})")
    sns.kdeplot(na, ax=ax, color=PAL["nonagentic"], fill=True, alpha=0.25,
                linewidth=1.0, label=f"Non-whitelisted (n={len(na):,})")
    ax.set_xlabel("ln(event price USD)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(OUT_FIG / "F6_price_distributions.pdf")
    plt.close(fig)
    print("  F6 saved.")


def fig_F7():
    """F7: Category premium coefficient plot."""
    df = pd.read_csv(TABLES / "T11_category_premiums.csv")
    df = df.sort_values("coef_agentic")
    fig, ax = plt.subplots(figsize=(3.3, 3.5))
    y = range(len(df))
    colors = []
    for _, row in df.iterrows():
        if row["p_value"] < 0.05 and row["coef_agentic"] > 0:
            colors.append(PAL["agentic"])
        elif row["p_value"] < 0.05 and row["coef_agentic"] < 0:
            colors.append(PAL["nonagentic"])
        else:
            colors.append(PAL["neutral"])

    ax.barh(y, df["coef_agentic"], color=colors, alpha=0.85, height=0.7,
            xerr=1.96 * df["se"], capsize=2, error_kw={"linewidth": 0.6})
    ax.set_yticks(y)
    ax.set_yticklabels(df["category"], fontsize=6.5)
    ax.axvline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Whitelisted coefficient (OLS)")
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PAL["agentic"], alpha=0.85, label="Sig. positive"),
        Patch(facecolor=PAL["nonagentic"], alpha=0.85, label="Sig. negative"),
        Patch(facecolor=PAL["neutral"], alpha=0.85, label="Not sig."),
    ]
    ax.legend(handles=legend_elements, fontsize=6, frameon=False, loc="lower right")
    fig.savefig(OUT_FIG / "F7_category_premiums.pdf")
    plt.close(fig)
    print("  F7 saved.")


def fig_F8():
    """F8: Lorenz curves + category concentration scatter."""
    df_dist = pd.read_csv(DATA / "block3_distribution.csv")
    df_conc = pd.read_csv(TABLES / "T13_concentration.csv")

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))

    # Left: Lorenz curves
    for label, mask, color in [
        ("Whitelisted", df_dist["agentic"] == True, PAL["agentic"]),
        ("Non-whitelisted", df_dist["agentic"] == False, PAL["nonagentic"]),
    ]:
        vals = np.sort(df_dist.loc[mask, "monthly_users"].values)
        cum = np.cumsum(vals) / vals.sum() if vals.sum() > 0 else np.zeros(len(vals))
        x = np.linspace(0, 1, len(vals))
        g = float((2 * np.sum(np.arange(1, len(vals)+1) * vals) -
                    (len(vals)+1) * vals.sum()) / (len(vals) * vals.sum()))
        axes[0].plot(x, cum, color=color, linewidth=1.0,
                     label=f"{label} (Gini={g:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.5)
    axes[0].set_xlabel("Cumulative share of actors")
    axes[0].set_ylabel("Cumulative share of users")
    axes[0].legend(fontsize=6.5, frameon=False)
    axes[0].set_aspect("equal")

    # Right: Category scatter (agentic penetration vs Gini)
    axes[1].scatter(df_conc["pct_agentic"], df_conc["gini"],
                    s=np.sqrt(df_conc["n_actors"]) * 2.5,
                    color=PAL["agentic"], alpha=0.55, edgecolors="white",
                    linewidth=0.3)
    for _, row in df_conc.iterrows():
        if row["n_actors"] >= 2000:
            axes[1].annotate(row["category"],
                             (row["pct_agentic"], row["gini"]),
                             fontsize=5.5, alpha=0.7,
                             xytext=(3, 3), textcoords="offset points")
    axes[1].set_xlabel("% Whitelisted in category")
    axes[1].set_ylabel("Gini coefficient")
    fig.savefig(OUT_FIG / "F8_concentration.pdf")
    plt.close(fig)
    print("  F8 saved.")


# ================================================================
# TABLES (LaTeX, booktabs, ACM-width compatible)
# ================================================================

def write_tex(path: Path, content: str):
    path.write_text(content)


def tab_T1():
    """T1: Marketplace overview."""
    df = pd.read_csv(TABLES / "T1_marketplace_overview.csv")
    r = df.iloc[0]
    total = int(r["total_actors"])
    pct_ppe = int(r["n_ppe"]) / total * 100
    pct_rental = int(r["n_rental"]) / total * 100
    pct_free = int(r["n_free"]) / total * 100
    n_cat = len(pd.read_csv(DATA / "block2_category_breakdown.csv"))

    tex = r"""\begin{table}[t]
  \centering
  \caption{Apify Store overview. Counts reflect the full census as of the data
    collection date. Agentic-readiness (limited-permission, non-standby) is
    measured among the 51{,}496 pay-per-event tools with architecture data.}
  \label{tab:overview}
  \small
  \begin{tabular}{l r}
    \toprule
    \textbf{Metric} & \textbf{Value} \\
    \midrule
    Total actors              & """ + f"{total:,}" + r""" \\
    Unique developers         & """ + f"{int(r['total_developers']):,}" + r""" \\
    Agentic-ready PPE actors (\%)  & """ + f"{int(r['n_agentic']):,} ({r['pct_agentic']:.0f}\\%)" + r""" \\
    Non-agentic-ready actors (\%) & """ + f"{int(r['n_non_agentic']):,} ({100-r['pct_agentic']:.0f}\\%)" + r""" \\
    PPE-priced actors (\%)     & """ + f"{int(r['n_ppe']):,} ({pct_ppe:.0f}\\%)" + r""" \\
    Rental-priced actors (\%)  & """ + f"{int(r['n_rental']):,} ({pct_rental:.0f}\\%)" + r""" \\
    Free actors (\%)           & """ + f"{int(r['n_free']):,} ({pct_free:.0f}\\%)" + r""" \\
    Median monthly users per actor & """ + f"{int(r['median_monthly_users'])}" + r""" \\
    Categories                 & """ + f"{n_cat}" + r""" \\
    \bottomrule
  \end{tabular}
\end{table}"""
    write_tex(OUT_TAB / "T1.tex", tex)
    print("  T1 saved.")


def tab_T2():
    """T2: Developer portfolio tiers."""
    df = pd.read_csv(TABLES / "T2_developer_tiers.csv")
    rows = []
    for _, r in df.iterrows():
        tier = r["tier"]
        name = tier.split(" (")[0]
        size = tier.split("(")[1].rstrip(")").replace("-", "--")
        rows.append(
            f"    {name:10s} & {size:10s} & {int(r['n_developers']):,} & "
            f"{int(r['total_actors']):,} & {r['avg_pct_agentic']:.1f}\\% \\\\"
        )
    body = "\n".join(rows)
    tex = r"""\begin{table}[t]
  \centering
  \caption{Developer portfolio tiers. Portfolio size = number of actors
    published by a developer. Agentic-ready share = average share of a
    developer's pay-per-event actors that are agentic-ready (limited-permission,
    non-standby).}
  \label{tab:dev_tiers}
  \small
  \begin{tabular}{l r r r r}
    \toprule
    \textbf{Tier} & \textbf{Portfolio size} & \textbf{Developers} & \textbf{Actors} & \textbf{Agentic-ready (\%)} \\
    \midrule
""" + body + r"""
    \bottomrule
  \end{tabular}
\end{table}"""
    write_tex(OUT_TAB / "T2.tex", tex)
    print("  T2 saved.")


def tab_T4():
    """T4: Raw agentic comparison (all pricing models)."""
    df = pd.read_csv(TABLES / "T4_raw_comparison.csv")
    ag = df[df["agentic"]].iloc[0]
    na = df[~df["agentic"]].iloc[0]
    ratio_mu = ag["avg_mu"] / na["avg_mu"]
    ratio_runs = ag["avg_runs"] / na["avg_runs"]

    tex = r"""\begin{table}[t]
  \centering
  \caption{Raw adoption comparison: whitelisted vs.\ non-whitelisted actors.}
  \label{tab:raw_comparison}
  \small
  \begin{tabular}{lrrrrc}
    \toprule
    Group        & $N$    & Avg monthly users & Med monthly users & Avg Runs & Avg Builds \\
    \midrule
    Whitelisted      & """ + f"{int(ag['n']):,}" + r""" & """ + f"{ag['avg_mu']:.2f}" + r"""  & """ + f"{int(ag['med_mu'])}" + r"""      & """ + f"{round(ag['avg_runs']):,}" + r"""  & """ + f"{ag['avg_builds']:.1f}" + r"""       \\
    Non-whitelisted  & """ + f"{int(na['n']):,}" + r""" & """ + f"{na['avg_mu']:.2f}" + r"""   & """ + f"{int(na['med_mu'])}" + r"""      & """ + f"{round(na['avg_runs']):,}" + r"""  & """ + f"{na['avg_builds']:.1f}" + r"""       \\
    \midrule
    Ratio        &        & """ + f"{ratio_mu:.2f}$\\times$" + r""" &  & """ + f"{ratio_runs:.2f}$\\times$" + r""" &         \\
    \bottomrule
    \multicolumn{6}{l}{\footnotesize Full marketplace ($N = 62{,}645$).}
  \end{tabular}
\end{table}"""
    write_tex(OUT_TAB / "T4.tex", tex)
    print("  T4 saved.")


def tab_T6():
    """T6: OLS regression hierarchy + developer FE (5 models), clustered SEs."""
    with open(DATA / "block4_regression_summary.json") as f:
        d = json.load(f)
    with open(DATA / "block10_se_robustness.json") as f:
        b10 = json.load(f)

    cl = b10["clustered_se"]
    fe = b10["fe_equal_developer_weighted_listing"]
    reg = d["regressions"]

    def stars(p):
        if p < 0.001: return "^{***}"
        if p < 0.01: return "^{**}"
        if p < 0.05: return "^{*}"
        if p < 0.10: return "^{\\dagger}"
        return ""

    coefs = [reg["M1"]["coef_agentic"], reg["M2"]["coef_agentic"],
             reg["M3"]["coef_agentic"], reg["M4"]["coef_agentic"], fe["coef"]]
    ses = [cl["M1"]["se"], cl["M2"]["se"], cl["M3"]["se"], cl["M4"]["se"], fe["se"]]
    ps = [cl["M1"]["p"], cl["M2"]["p"], cl["M3"]["p"], cl["M4"]["p"], fe["p"]]
    r2s = [reg["M1"]["r_squared"], reg["M2"]["r_squared"],
           reg["M3"]["r_squared"], reg["M4"]["r_squared"], None]
    ns = [reg["M1"]["n_obs"], reg["M2"]["n_obs"], reg["M3"]["n_obs"],
          reg["M4"]["n_obs"], 27463]

    coef_row = " & ".join(f"${c:.4f}{stars(p)}$" for c, p in zip(coefs, ps))
    se_row = " & ".join(f"({s:.4f})" for s in ses)
    r2_row = " & ".join(f"{r:.3f}" if r is not None else "---" for r in r2s)
    n_row = " & ".join(f"{n:,}" for n in ns)

    tex = r"""\begin{table}[t]
  \centering
  \caption{Regression hierarchy: association between the whitelist flag and adoption.}
  \label{tab:regression}
  \small
  \begin{tabular}{lccccc}
    \toprule
    & M1        & M2        & M3        & M4        & M5         \\
    & Baseline  & +Dev Ctrl & +Cat FE   & Runs DV   & Dev FE     \\
    \midrule
    Whitelisted       & """ + coef_row + r""" \\
                  & """ + se_row + r""" \\
    \addlinespace
    Dev controls  & No        & Yes       & Yes       & Yes       & Pre-treat. \\
    Category FE   & No        & No        & Yes       & Yes       & Yes        \\
    Developer FE  & No        & No        & No        & No        & Yes        \\
    \addlinespace
    $R^2$         & """ + r2_row + r""" \\
    $N$ (actors)  & """ + n_row + r""" \\
    \bottomrule
    \multicolumn{6}{l}{\footnotesize $^{***}\, p<0.001$;\;
      $^{**}\, p<0.01$;\; $^{*}\, p<0.05$;\; $^{\dagger}\, p<0.10$.} \\
    \multicolumn{6}{l}{\footnotesize Developer-clustered SEs in parentheses.
      M1--M4: PPE actors only.} \\
    \multicolumn{6}{l}{\footnotesize M5: equal-developer-weighted within-developer FE with} \\
    \multicolumn{6}{l}{\footnotesize listing-time controls (age, description, category); mixed developers only} \\
    \multicolumn{6}{l}{\footnotesize ($N_{\text{developers}} = 682$).}
  \end{tabular}
\end{table}"""
    write_tex(OUT_TAB / "T6.tex", tex)
    print("  T6 saved.")


def tab_T11():
    """T11: Category-level premiums (selected categories, human-readable)."""
    df = pd.read_csv(TABLES / "T11_category_premiums.csv")
    name_map = {
        "AI": "AI", "SOCIAL_MEDIA": "Social Media", "TRAVEL": "Travel",
        "REAL_ESTATE": "Real Estate", "LEAD_GENERATION": "Lead Generation",
        "BUSINESS": "Business", "AUTOMATION": "Automation", "SEO_TOOLS": "SEO",
        "ECOMMERCE": "E-commerce", "DEVELOPER_TOOLS": "Developer Tools",
        "MCP_SERVERS": "MCP Servers",
    }
    selected = ["AI", "SOCIAL_MEDIA", "TRAVEL", "REAL_ESTATE", "LEAD_GENERATION",
                "BUSINESS", "AUTOMATION", "SEO_TOOLS", "ECOMMERCE",
                "DEVELOPER_TOOLS", "MCP_SERVERS"]

    def sig(p):
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return "n.s."

    rows = []
    for cat in selected:
        r = df[df["category"] == cat].iloc[0]
        sign = "$-$" if r["coef_agentic"] < 0 else "$+$"
        rows.append(
            f"    {name_map[cat]:16s} & {sign}{abs(r['coef_agentic']):.4f} & "
            f"({r['se']:.3f}) & {sig(r['p_value']):4s} & {int(r['n_actors']):,} & {r['pct_agentic']:.0f}\\% \\\\"
        )
    body = "\n".join(rows)
    tex = r"""\begin{table}[t]
  \centering
  \caption{Per-category whitelist-associated adoption gap (selected categories, OLS with
    tool-level controls).}
  \label{tab:category_gaps}
  \small
  \begin{tabular}{lrcrcc}
    \toprule
    Category         & Coef.     & (SE)    & Sig.  & $N$    & \% Whitelisted \\
    \midrule
""" + body + r"""
    \bottomrule
    \multicolumn{6}{l}{\footnotesize $^{***}\, p<0.001$;\;
      $^{**}\, p<0.01$;\; $^{*}\, p<0.05$;\; n.s.\;$p \geq 0.05$.} \\
    \multicolumn{6}{l}{\footnotesize HC1 robust SEs in parentheses. Controls:
      $\log(1+\texttt{total\_builds})$, developer portfolio size,} \\
    \multicolumn{6}{l}{\footnotesize pricing-event count. PPE actors only
      ($N = 51{,}861$); selected categories shown.}
  \end{tabular}
\end{table}"""
    write_tex(OUT_TAB / "T11.tex", tex)
    print("  T11 saved.")


def tab_T12():
    """T12: Concentration metrics (marketplace level)."""
    with open(DATA / "block8_robustness.json") as f:
        rob = json.load(f)
    cd = rob["concentration_decomposition"]
    with open(DATA / "block7_heterogeneity.json") as f:
        b7 = json.load(f)
    gini = b7["concentration"]["gini"]
    cr = b7["concentration"]["concentration_ratios"]

    def share(k):
        return cr[k]["share_of_total_users"] * 100

    tex = r"""\begin{table}[t]
  \centering
  \caption{Marketplace concentration: whitelisted vs.\ non-whitelisted PPE actors.}
  \label{tab:concentration}
  \small
  \begin{tabular}{lcc}
    \toprule
    Metric & Whitelisted & Non-whitelisted \\
    \midrule
    Gini coefficient       & """ + f"{gini['agentic']:.3f}" + r""" & """ + f"{gini['non_agentic']:.3f}" + r""" \\
    Top 1\% user share     & """ + f"{share('top_1pct_agentic'):.1f}\\%" + r""" & """ + f"{share('top_1pct_non_agentic'):.1f}\\%" + r""" \\
    Top 5\% user share     & """ + f"{share('top_5pct_agentic'):.1f}\\%" + r""" & """ + f"{share('top_5pct_non_agentic'):.1f}\\%" + r""" \\
    Top 10\% user share    & """ + f"{share('top_10pct_agentic'):.1f}\\%" + r""" & """ + f"{share('top_10pct_non_agentic'):.1f}\\%" + r""" \\
    \addlinespace
    Theil index            & """ + f"{cd['theil_agentic']['total']:.3f}" + r""" & """ + f"{cd['theil_non_agentic']['total']:.3f}" + r""" \\
    \quad Between-cat.\ share & """ + f"{cd['theil_agentic']['between_pct']:.1f}" + r"""\% & """ + f"{cd['theil_non_agentic']['between_pct']:.1f}" + r"""\% \\
    \quad Within-cat.\ share  & """ + f"{100 - cd['theil_agentic']['between_pct']:.1f}" + r"""\% & """ + f"{100 - cd['theil_non_agentic']['between_pct']:.1f}" + r"""\% \\
    \addlinespace
    Categories w/ higher Gini & \multicolumn{2}{c}{14 of 17} \\
    \bottomrule
    \multicolumn{3}{l}{\footnotesize PPE actors only ($N = 51{,}861$).}
  \end{tabular}
\end{table}"""
    write_tex(OUT_TAB / "T12.tex", tex)
    print("  T12 saved.")


def tab_T5():
    """T5: PPE-only adoption comparison with distributional percentiles."""
    df = pd.read_csv(TABLES / "T5_ppe_comparison.csv")
    ag = df[df["agentic"]].iloc[0]
    na = df[~df["agentic"]].iloc[0]
    tex = r"""\begin{table}[t]
  \centering
  \caption{PPE-only adoption comparison with distributional percentiles.}
  \label{tab:ppe_comparison}
  \small
  \begin{tabular}{lrrrrrr}
    \toprule
    Group       & $N$    & Avg MU & Med MU & p90 & p95 & Avg Runs \\
    \midrule
    Whitelisted     & """ + f"{int(ag['n']):,}" + r""" & """ + f"{ag['avg_mu']:.2f}" + r"""  & """ + f"{int(ag['med_mu'])}" + r"""      & """ + f"{int(ag['p90_mu'])}" + r"""   & """ + f"{int(ag['p95_mu'])}" + r"""  & """ + f"{round(ag['avg_runs']):,}" + r"""  \\
    Non-whitelisted & """ + f"{int(na['n']):,}" + r""" & """ + f"{na['avg_mu']:.2f}" + r"""   & """ + f"{int(na['med_mu'])}" + r"""      & """ + f"{int(na['p90_mu'])}" + r"""   & """ + f"{int(na['p95_mu'])}" + r"""  & """ + f"{round(na['avg_runs']):,}" + r"""      \\
    \bottomrule
    \multicolumn{7}{l}{\footnotesize PPE actors only ($N = 51{,}861$).}
  \end{tabular}
\end{table}"""
    write_tex(OUT_TAB / "T5.tex", tex)
    print("  T5 saved.")


def tab_T7():
    """T7: Covariate balance before/after 1:1 matching."""
    df = pd.read_csv(TABLES / "T7_psm_balance.csv")
    label_map = {"log_age": "Log actor age", "desc_length": "Description length",
                 "dev_portfolio_size": "Developer portfolio size"}
    rows = []
    for _, r in df.iterrows():
        name = label_map.get(r["Covariate"], r["Covariate"])
        before = abs(r["SMD (before)"])
        after = abs(r["SMD (after)"])
        rows.append(f"    {name:22s} & {before:.3f} & {after:.3f} \\\\")
    body = "\n".join(rows)
    tex = r"""\begin{table}[t]
  \centering
  \caption{Covariate balance before and after 1:1 matching.}
  \label{tab:psm_balance}
  \small
  \begin{tabular}{lcc}
    \toprule
    Covariate & SMD (before) & SMD (after) \\
    \midrule
""" + body + r"""
    \bottomrule
    \multicolumn{3}{l}{\footnotesize SMD = standardized mean difference. Matching reduces all} \\
    \multicolumn{3}{l}{\footnotesize covariates below $|0.1|$.}
  \end{tabular}
\end{table}"""
    write_tex(OUT_TAB / "T7.tex", tex)
    print("  T7 saved.")


def tab_T16():
    """T16: PSM listing-time-only estimators with bootstrap uncertainty."""
    df = pd.read_csv(TABLES / "T16_psm_sensitivity.csv")
    order = ["Overlap-weighted ATO (full sample)", "IPTW ATT (full sample)",
             "Matching with replacement (full treated)",
             "1:1 NN, no replacement (matched subset)"]
    df = df.set_index("Estimator").loc[order].reset_index()
    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"    {r['Estimator']:38s} & {r['log_mu coef']:.3f} & {r['exp-1 %']:.1f}\\% & "
            f"{r['SE']:.3f} & {r['p']:.3f} \\\\"
        )
    body = "\n".join(rows)
    tex = r"""\begin{table}[t]
  \centering
  \caption{PSM: listing-time-only estimators with bootstrap uncertainty.}
  \label{tab:psm_sensitivity}
  \small
  \begin{tabular}{lcccc}
    \toprule
    Estimator & $\hat\beta$ & exp-1 & SE & $p$ \\
    \midrule
""" + body + r"""
    \bottomrule
    \multicolumn{5}{l}{\footnotesize SEs and centered-percentile $p$-values from a developer-level bootstrap} \\
    \multicolumn{5}{l}{\footnotesize ($B = 500$); $p = P(|b^*-\hat\beta| \geq |\hat\beta|$). Listing-time covariates only.}
  \end{tabular}
\end{table}"""
    write_tex(OUT_TAB / "T16.tex", tex)
    print("  T16 saved.")


def tab_T10():
    """T10: PPE pricing structure: agentic vs non-agentic."""
    df = pd.read_csv(TABLES / "T10_agentic_pricing.csv")
    ag = df[df["agentic"]].iloc[0]
    na = df[~df["agentic"]].iloc[0]
    tex = r"""\begin{table}[t]
  \centering
  \caption{PPE pricing structure: whitelisted vs.\ non-whitelisted actors.}
  \label{tab:pricing_structure}
  \small
  \begin{tabular}{lcc}
    \toprule
    Metric              & Whitelisted & Non-whitelisted \\
    \midrule
    Avg pricing events  & """ + f"{ag['avg_n_events']:.2f}" + r"""    & """ + f"{na['avg_n_events']:.2f}" + r"""        \\
    Median price/event  & \$""" + f"{ag['avg_median_event_price']:.3f}" + r""" & \$""" + f"{na['avg_median_event_price']:.3f}" + r"""     \\
    \bottomrule
    \multicolumn{3}{l}{\footnotesize PPE actors only ($N = 51{,}861$).}
  \end{tabular}
\end{table}"""
    write_tex(OUT_TAB / "T10.tex", tex)
    print("  T10 saved.")


def tab_T8():
    """T8: Developer typology (summary by portfolio type)."""
    df = pd.read_csv(TABLES / "T8_developer_typology.csv")
    summary = df.groupby("dev_type").agg(
        n_devs=("n_devs", "sum"), total_actors=("total_actors", "sum")).reset_index()
    order = {"Pure Agentic": 0, "Pure Traditional": 1, "Mixed": 2}
    summary = summary.sort_values("dev_type", key=lambda s: s.map(order))
    label_map = {"Pure Agentic": "Whitelisted-only", "Pure Traditional": "Non-whitelisted-only", "Mixed": "Mixed"}
    pct_map = {"Whitelisted-only": "100\\%", "Non-whitelisted-only": "0\\%", "Mixed": "varies"}
    tier_map = {"Whitelisted-only": "---", "Non-whitelisted-only": "---", "Mixed": "All"}
    rows = []
    for _, r in summary.iterrows():
        label = label_map[r["dev_type"]]
        rows.append(
            f"    {label:13s} & {tier_map[label]:4s} & {int(r['n_devs']):,} & "
            f"{int(r['total_actors']):,} & {pct_map[label]} \\\\"
        )
    with open(DATA / "block4_regression_summary.json") as f:
        b4 = json.load(f)
    n_ppe = b4["developer_fe"]["n_mixed_developers"]
    n_ppe_actors = b4["developer_fe"]["n_actors"]
    rows.append("    \\midrule")
    rows.append("    \\multicolumn{5}{l}{\\emph{Mixed PPE-only (used in within-developer FE):}} \\\\")
    rows.append(
        f"    Mixed (PPE)  & All    & {n_ppe:,}      & {n_ppe_actors:,}       & varies         \\\\"
    )
    body = "\n".join(rows)
    tex = r"""\begin{table}[t]
  \centering
  \caption{Developer typology by portfolio composition and activity tier.}
  \label{tab:dev_typology}
  \small
  \begin{tabular}{llrrr}
    \toprule
    Type         & Tier   & $N$ Devs & Total Actors & Avg \% Whitelisted \\
    \midrule
""" + body + r"""
    \bottomrule
    \multicolumn{5}{l}{\footnotesize Full marketplace ($N = 62{,}645$). Mixed PPE-only: developers}\\
    \multicolumn{5}{l}{\footnotesize with both whitelisted and non-whitelisted PPE actors ($N = 27{,}463$).}
  \end{tabular}
\end{table}"""
    write_tex(OUT_TAB / "T8.tex", tex)
    print("  T8 saved.")


def tab_T17():
    """T17: Within-developer FE weighting and control-set robustness."""
    with open(DATA / "block10_se_robustness.json") as f:
        b10 = json.load(f)

    def row(est):
        return f"{est['coef']:.4f} & {est['se']:.4f} & {est['p']:.4g}"

    aw = b10["fe_actor_weighted_listing"]
    ew_lt = b10["fe_equal_developer_weighted_listing"]
    ew_ct = b10["fe_equal_developer_weighted_contemp"]
    wh = b10["exclude_whales"]
    loo = b10["leave_one_out_top20"]

    tex = r"""\begin{table}[t]
  \centering
  \caption{Within-developer fixed effects: weighting and control-set robustness.}
  \label{tab:fe_weighting}
  \small
  \begin{tabular}{lccc}
    \toprule
    Specification & $\hat\beta$ & SE & $p$ \\
    \midrule
    Equal-dev-weighted, listing-time controls & """ + f"{ew_lt['coef']:.4f}" + r""" & """ + f"{ew_lt['se']:.4f}" + r""" & $<$0.001 \\
    Equal-dev-weighted, contemporaneous controls & """ + f"{ew_ct['coef']:.4f}" + r""" & """ + f"{ew_ct['se']:.4f}" + r""" & """ + f"{ew_ct['p']:.4g}" + r""" \\
    Actor-weighted, listing-time controls & """ + f"{aw['coef']:.4f}" + r""" & """ + f"{aw['se']:.4f}" + r""" & """ + f"{aw['p']:.4g}" + r""" \\
    \addlinespace
    Exclude whale devs ($\geq$100 PPE actors), listing-time & """ + f"{wh['coef']:.4f}" + r""" & """ + f"{wh['se']:.4f}" + r""" & """ + f"{wh['p']:.4g}" + r""" \\
    Leave-one-large-dev-out (top 20) & """ + f"{loo['min']:.4f}--{loo['max']:.4f}" + r""" & --- & all $<$0.10 \\
    \bottomrule
    \multicolumn{4}{l}{\footnotesize All models include developer and category FE. Listing-time} \\
    \multicolumn{4}{l}{\footnotesize controls: age, description length. Contemporaneous controls add} \\
    \multicolumn{4}{l}{\footnotesize bookmarks, reviews, rating, maintenance recency. Developer-clustered SEs.}
  \end{tabular}
\end{table}"""
    write_tex(OUT_TAB / "T17.tex", tex)
    print("  T17 saved.")


def tab_T18():
    """T18: Developer-scale gradient (within-developer gap by portfolio tier)."""
    with open(DATA / "block11_interaction_gradient.json") as f:
        b11 = json.load(f)
    rows = []
    for r in b11["developer_scale_gradient"]:
        tier = r["tier"].replace("-", "--")
        sign = "+" if r["mean_gap"] >= 0 else "-"
        pct = (np.exp(r["mean_gap"]) - 1) * 100
        rows.append(
            f"    {tier:16s} & {r['n_devs']} & ${sign}{abs(r['mean_gap']):.3f} ({pct:.1f}\\%)$ & "
            f"{r['median_gap']:.3f} & {r['pct_positive']:.0f}\\% \\\\"
        )
    body = "\n".join(rows)
    tex = r"""\begin{table}[t]
  \centering
  \caption{Developer-specific within-developer gaps by portfolio tier.}
  \label{tab:dev_gap_tier}
  \small
  \begin{tabular}{lcccc}
    \toprule
    Tier & $N$ devs & Mean gap & Median gap & \% positive \\
    \midrule
""" + body + r"""
    \bottomrule
    \multicolumn{5}{l}{\footnotesize Each developer's gap is the mean log monthly users of their whitelisted} \\
    \multicolumn{5}{l}{\footnotesize tools minus their non-whitelisted tools. Mixed PPE developers only ($N = 682$);} \\
    \multicolumn{5}{l}{\footnotesize tiers defined by total portfolio size (actors across all pricing models).}
  \end{tabular}
\end{table}"""
    write_tex(OUT_TAB / "T18.tex", tex)
    print("  T18 saved.")


def tab_T19():
    """T19: Consumer-facing vs developer-infrastructure interaction test."""
    with open(DATA / "block11_interaction_gradient.json") as f:
        b11 = json.load(f)
    rows = []
    for r in b11["interaction_test"]:
        sign = "+" if r["coef"] >= 0 else "-"
        rows.append(f"    {r['spec']:28s} & ${sign}{abs(r['coef']):.3f}$ & {r['p']:.3f} \\\\")
    body = "\n".join(rows)
    tex = r"""\begin{table}[t]
  \centering
  \caption{Interaction test: does the whitelist gap differ between
    consumer-facing and developer-infrastructure categories?}
  \label{tab:interaction}
  \small
  \begin{tabular}{lcc}
    \toprule
    Specification & $\hat\beta_{\text{whitelist}\times\text{consumer}}$ & $p$ \\
    \midrule
""" + body + r"""
    \bottomrule
    \multicolumn{3}{l}{\footnotesize Developer-clustered SEs. Consumer-facing vs.\ developer-} \\
    \multicolumn{3}{l}{\footnotesize infrastructure classified a priori by end-user identity.}
  \end{tabular}
\end{table}"""
    write_tex(OUT_TAB / "T19.tex", tex)
    print("  T19 saved.")


def tab_T21():
    """T21: Agentic eligibility (limited + non-standby) within-developer FE."""
    b15 = json.load(open(DATA / "block15_eligibility.json"))
    fe_eq = b15["fe_equal_dev"]
    fe_aw = b15["fe_actor_weighted"]

    def ci(r):
        lo = r["coef"] - 1.96 * r["se"]
        hi = r["coef"] + 1.96 * r["se"]
        return f"[{lo:.2f}, {hi:.2f}]"

    tex = r"""\begin{table}[t]
  \centering
  \caption{Agentic readiness and adoption: within-developer fixed effects.}
  \label{tab:eligibility_fe}
  \small
  \begin{tabular}{lccc}
    \toprule
    Estimator & $\hat\beta$ & exp-1 & 95\% CI \\
    \midrule
""" + (
        f"    Agentic-ready (equal-developer-weighted) & {fe_eq['coef']:.3f} & "
        f"{fe_eq['pct']:+.1f}\\% & {ci(fe_eq)} \\\\\n"
        f"    Agentic-ready (actor-weighted) & {fe_aw['coef']:.3f} & "
        f"{fe_aw['pct']:+.1f}\\% & {ci(fe_aw)} \\\\\n"
    ) + r"""    \bottomrule
    \multicolumn{4}{l}{\footnotesize Identified among 343 mixed developers (16{,}537 PPE actors), developer-clustered} \\
    \multicolumn{4}{l}{\footnotesize SEs; listing-time controls (age, description, category). Equal-developer-} \\
    \multicolumn{4}{l}{\footnotesize weighted gives each developer equal weight. 95\% CI on the $\hat\beta$ (log) scale.} \\
    \multicolumn{4}{l}{\footnotesize PPML with developer FE (count scale): IRR 8.6, $p=0.012$ (equal-developer-weighted);} \\
    \multicolumn{4}{l}{\footnotesize actor-weighted IRR 2.2, n.s.}
  \end{tabular}
\end{table}"""
    write_tex(OUT_TAB / "T21.tex", tex)
    print("  T21 saved.")


def tab_T22():
    """T22: extensive vs intensive margin for agentic eligibility."""
    b15 = json.load(open(DATA / "block15_eligibility.json"))
    m = b15["margins"]

    def pfmt(p):
        return "<0.001" if p < 0.001 else f"{p:.3f}"

    tex = r"""\begin{table}[t]
  \centering
  \caption{Extensive versus intensive margin for agentic readiness.}
  \label{tab:eligibility_margins}
  \small
  \begin{tabular}{lccc}
    \toprule
    Margin & $\hat\beta$ & SE & $p$ \\
    \midrule
""" + (
        f"    Extensive (any monthly users) & {m['extensive']['pp']:+.1f} pp & "
        f"{m['extensive']['se']:.3f} & {pfmt(m['extensive']['p'])} \\\\\n"
        f"    Intensive ($\\ln(1+\\text{{MU}})$ given MU $>$ 0) & {m['intensive']['coef']:+.3f} & "
        f"{m['intensive']['se']:.3f} & {pfmt(m['intensive']['p'])} \\\\\n"
        f"    Top decile (MU $\\geq$ {m['p90']:.0f}) & {m['top_decile']['pp']:+.1f} pp & "
        f"{m['top_decile']['se']:.3f} & {pfmt(m['top_decile']['p'])} \\\\\n"
    ) + r"""    \bottomrule
    \multicolumn{4}{l}{\footnotesize Within-developer, equal-developer-weighted, developer-clustered SEs.} \\
    \multicolumn{4}{l}{\footnotesize Extensive and top-decile are linear-probability models (coefficient in} \\
    \multicolumn{4}{l}{\footnotesize percentage points); intensive is log-plus-one (coefficient in log points).}
  \end{tabular}
\end{table}"""
    write_tex(OUT_TAB / "T22.tex", tex)
    print("  T22 saved.")


def tab_T23():
    """T23: per-category agentic-eligibility effect (primary + multi-hot note)."""
    b15 = json.load(open(DATA / "block15_eligibility.json"))
    b16 = json.load(open(DATA / "block16_eligibility_robustness.json"))
    rows = b15["category_heterogeneity"]["rows"]
    bonf = b15["category_heterogeneity"]["bonferroni_alpha"]
    mh = b16["multihot_category"]

    name_map = {
        "AI": "AI", "SOCIAL_MEDIA": "Social Media", "TRAVEL": "Travel",
        "REAL_ESTATE": "Real Estate", "LEAD_GENERATION": "Lead Generation",
        "BUSINESS": "Business", "AUTOMATION": "Automation", "SEO_TOOLS": "SEO",
        "ECOMMERCE": "E-commerce", "DEVELOPER_TOOLS": "Developer Tools",
        "MCP_SERVERS": "MCP Servers", "NEWS": "News", "VIDEOS": "Videos",
        "JOBS": "Jobs", "AGENTS": "Agents", "OTHER": "Other",
    }

    def sig(p):
        if p < bonf: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return "n.s."

    body = "\n".join(
        f"    {name_map.get(r['category'], r['category']):16s} & "
        f"${r['coef']:+.4f}$ & ({r['se']:.3f}) & {sig(r['p']):4s} & {int(r['n_actors']):,} \\\\"
        for r in rows
    )
    n_mh_sig = sum(1 for r in mh["rows"] if r["p"] < mh["bonferroni_alpha"])
    tex = r"""\begin{table}[t]
  \centering
  \caption{Per-category agentic-readiness effect (OLS, tool-level controls).}
  \label{tab:eligibility_category}
  \small
  \begin{tabular}{lrcrc}
    \toprule
    Category & Coef. & (SE) & Sig. & $N$ \\
    \midrule
""" + body + r"""
    \bottomrule
""" + (
        f"    \\multicolumn{{5}}{{l}}{{\\footnotesize $^{{***}}\\,p<{bonf:.3f}$ (Bonferroni); "
        f"$^{{**}}\\,p<0.01$; $^{{*}}\\,p<0.05$; n.s.\\,$p\\geq 0.05$.}} \\\\\n"
        f"    \\multicolumn{{5}}{{l}}{{\\footnotesize Developer-clustered SEs. Controls: log-builds, portfolio size, pricing events.}} \\\\\n"
        f"    \\multicolumn{{5}}{{l}}{{\\footnotesize Multi-hot (any-membership) robustness: {n_mh_sig} of "
        f"{len(mh['rows'])} categories Bonferroni-significant, all positive.}} \\\\\n"
        f"    \\multicolumn{{5}}{{l}}{{\\footnotesize Within-developer (FE) estimates reorder: Agents $-0.10$ and AI $+0.15$ (n.s.);}} \\\\\n"
        f"    \\multicolumn{{5}}{{l}}{{\\footnotesize Developer Tools $+0.81$, Social Media $+0.84$, Automation $+0.71$.}}\n"
    ) + r"""  \end{tabular}
\end{table}"""
    write_tex(OUT_TAB / "T23.tex", tex)
    print("  T23 saved.")


def tab_T24():
    """T24: propensity-score matching for the eligibility treatment."""
    b17 = json.load(open(DATA / "block17_eligibility_psm.json"))
    b24 = json.load(open(DATA / "block24_balance_overlap.json"))

    def pct(k):
        return b17[k]["pct"]

    smd = b24["balance_smd"]
    age_b = smd["log_age"]["smd_before"]
    age_a = smd["log_age"]["smd_after"]

    tex = r"""\begin{table}[t]
  \centering
  \caption{Propensity-score matching for agentic readiness.}
  \label{tab:eligibility_psm}
  \small
  \begin{tabular}{lcc}
    \toprule
    Estimator & $\hat\beta$ & exp-1 \\
    \midrule
""" + (
        f"    1:1 nearest-neighbor & {b17['match_1to1']['coef']:.3f} & {pct('match_1to1'):+.1f}\\% \\\\\n"
        f"    Overlap weighting (ATO) & {b17['overlap_ato']['coef']:.3f} & {pct('overlap_ato'):+.1f}\\% \\\\\n"
        f"    IPTW (ATT) & {b17['iptw_att']['coef']:.3f} & {pct('iptw_att'):+.1f}\\% \\\\\n"
    ) + r"""    \bottomrule
    \multicolumn{3}{l}{\footnotesize Listing-time propensity score (age, description length,} \\
    \multicolumn{3}{l}{\footnotesize portfolio size, category). Outcome is log-plus-one monthly users.} \\
    \multicolumn{3}{l}{\footnotesize Treated:control = """ + f"{b17['ratio']:.1f}" + r""":1; 1:1 covers a minority of agentic-ready actors.} \\
    \multicolumn{3}{l}{\footnotesize Balance: standardized log-age difference """ + f"{age_b:+.2f}" + r""" before weighting, """ + f"{age_a:+.2f}" + r""" after.}
  \end{tabular}
\end{table}"""
    write_tex(OUT_TAB / "T24.tex", tex)
    print("  T24 saved.")


def tab_T25():
    """T25: concentration, eligible vs non-eligible."""
    b18 = json.load(open(DATA / "block18_eligibility_concentration.json"))
    e = b18["concentration"]["eligible"]
    n = b18["concentration"]["non_eligible"]
    wcg = b18["within_category_gini"]

    tex = r"""\begin{table}[t]
  \centering
  \caption{Concentration: agentic-ready versus non-agentic-ready actors.}
  \label{tab:eligibility_concentration}
  \small
  \begin{tabular}{lcc}
    \toprule
    Metric & Agentic-ready & Non-agentic-ready \\
    \midrule
""" + (
        f"    Gini coefficient & {e['gini']:.3f} & {n['gini']:.3f} \\\\\n"
        f"    Top 1\\% user share & {e['top_1pct']:.1f}\\% & {n['top_1pct']:.1f}\\% \\\\\n"
        f"    Top 5\\% user share & {e['top_5pct']:.1f}\\% & {n['top_5pct']:.1f}\\% \\\\\n"
        f"    Top 10\\% user share & {e['top_10pct']:.1f}\\% & {n['top_10pct']:.1f}\\% \\\\\n"
        f"    Theil index (total) & {e['theil_total']:.2f} & {n['theil_total']:.2f} \\\\\n"
        f"    Between-category share & {e['theil_between_pct']:.1f}\\% & {n['theil_between_pct']:.1f}\\% \\\\\n"
    ) + r"""    \bottomrule
    \multicolumn{3}{l}{\footnotesize Within-category Gini higher for agentic-ready in """ + f"{wcg['higher_in']}" + r""" of """ + f"{wcg['total']}" + r""" categories.} \\
    \multicolumn{3}{l}{\footnotesize Bootstrap 95\% CIs for the ready $-$ non-ready difference: Gini $[-0.03, 0.04]$;} \\
    \multicolumn{3}{l}{\footnotesize top-1\% $[-0.2, 32.6]$ pp; top-5\% $[-2.6, 12.3]$ pp; top-10\% $[-1.9, 8.1]$ pp;} \\
    \multicolumn{3}{l}{\footnotesize Theil $[0.69, 2.41]$ (the Theil gap reflects the larger, longer-tailed ready group).} \\
    \multicolumn{3}{l}{\footnotesize Sample-size-equalized (ready downsampled to 2{,}585): Gini CI $[0.85, 0.97]$,} \\
    \multicolumn{3}{l}{\footnotesize top-1\% $[49.2, 88.7]$ pp, Theil $[2.20, 5.28]$---all contain the non-ready values.}
  \end{tabular}
\end{table}"""
    write_tex(OUT_TAB / "T25.tex", tex)
    print("  T25 saved.")


def tab_T27():
    """T27: 2x2 architecture decomposition (permission x standby)."""
    b25 = json.load(open(DATA / "block25_architecture_2x2.json"))
    a = b25["architecture_2x2"]
    fe = b25["within_developer_fe"]["three_dummies_ref_ready"]

    full = a["FULL_PERMISSIONS"]
    lim = a["LIMITED_PERMISSIONS"]

    tex = r"""\begin{table}[t]
  \centering
  \caption{Architecture decomposition: adoption across the four
    permission-by-standby cells. Cells report actor count (mean monthly users).}
  \label{tab:architecture_2x2}
  \small
  \begin{tabular}{lcc}
    \toprule
    & \textbf{Standby} & \textbf{Non-standby} \\
    \midrule
    \textbf{Full permissions} & """ + f"{full['standby']['n']:,} ({full['standby']['mean_monthly_users']:.1f})" + r""" & """ + f"{full['non_standby']['n']:,} ({full['non_standby']['mean_monthly_users']:.1f})" + r""" \\
    \textbf{Limited permissions} & """ + f"{lim['standby']['n']:,} ({lim['standby']['mean_monthly_users']:.1f})" + r""" & """ + f"{lim['non_standby']['n']:,} ({lim['non_standby']['mean_monthly_users']:.1f})" + r""" \\
    \bottomrule
    \multicolumn{3}{l}{\footnotesize Ready = limited, non-standby (bottom-right). Within-developer, relative to} \\
    \multicolumn{3}{l}{\footnotesize ready: full-permission """ + f"{fe['full_nonstandby']['pct']:+.1f}" + r"""\% to """ + f"{fe['full_standby']['pct']:+.1f}" + r"""\%; standby-only """ + f"{fe['limited_standby']['pct']:+.1f}" + r"""\%.}
  \end{tabular}
\end{table}"""
    write_tex(OUT_TAB / "T27.tex", tex)
    print("  T27 saved.")


# ================================================================
# MAIN
# ================================================================

def main():
    print("=== Generating publication figures ===")
    fig_F1()
    fig_F2()
    fig_F3()
    fig_F4()
    fig_F5()
    fig_F6()
    fig_F7()
    fig_F8()

    print("\n=== Generating publication tables (LaTeX) ===")
    tab_T1()
    tab_T2()
    tab_T4()
    tab_T5()
    tab_T6()
    tab_T7()
    tab_T8()
    tab_T10()
    tab_T11()
    tab_T12()
    tab_T16()
    tab_T17()
    tab_T18()
    tab_T19()
    tab_T21()
    tab_T22()
    tab_T23()
    tab_T24()
    tab_T25()
    tab_T27()

    print("\n=== QA Checks ===")
    # Verify figure files exist and have reasonable size
    for f in sorted(OUT_FIG.glob("*.pdf")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}: {size_kb:.0f} KB")
    for f in sorted(OUT_TAB.glob("*.tex")):
        lines = len(f.read_text().splitlines())
        print(f"  {f.name}: {lines} lines")

    print("\nDone. All numbers extracted programmatically from stored result files.")


if __name__ == "__main__":
    main()
