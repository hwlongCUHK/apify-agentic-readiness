"""Block 7 (NEW): Premium Heterogeneity and Competitive Concentration (RQ3).

RQ3: How does the agentic premium vary across marketplace segments,
     and does the transition amplify competitive concentration?

Outputs:
  - results/tables/T11_category_premiums.csv
  - results/tables/T12_scale_premiums.csv
  - results/tables/T13_concentration.csv
  - results/data/block7_category_regressions.json
  - results/data/block7_gini.json
  - results/figures/F8_category_premium_heatmap.pdf
  - results/figures/F9_concentration_lorenz.pdf
  - results/figures/F10_scale_interaction.pdf
"""

import json
import logging
import warnings
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "apify_panel.duckdb"
BASE = Path(__file__).resolve().parent.parent / "results"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
})


def gini(values: np.ndarray) -> float:
    """Compute Gini coefficient."""
    v = np.sort(values.astype(float))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * v) - (n + 1) * np.sum(v)) / (n * np.sum(v)))


def prepare_data(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load PPE analysis dataset with category info."""
    df = con.execute("""
        SELECT
            a.actor_id,
            a.username,
            a.is_agentic_payments_whitelisted AS agentic,
            a.monthly_users,
            COALESCE(a.runs_30d_total, 0) AS runs_30d_total,
            a.total_builds,
            a.bookmarks,
            a.rating,
            dev.n_actors AS dev_portfolio_size,
            COALESCE(pe.n_events, 0) AS n_pricing_events,
            LENGTH(COALESCE(a.description, '')) AS desc_length,
            json_extract_string(a.categories, '$[0]') AS primary_category,
            LN(1 + a.monthly_users) AS log_mu,
            LN(1 + COALESCE(a.runs_30d_total, 0)) AS log_runs,
            LN(1 + a.total_builds) AS log_builds,
            CASE
                WHEN dev.n_actors >= 100 THEN 'Whale'
                WHEN dev.n_actors >= 10 THEN 'Mid'
                WHEN dev.n_actors >= 2 THEN 'Boutique'
                ELSE 'Solo'
            END AS dev_tier
        FROM actor_day a
        LEFT JOIN (
            SELECT username, COUNT(*) AS n_actors
            FROM actor_day
            WHERE crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
            GROUP BY username
        ) dev ON a.username = dev.username
        LEFT JOIN (
            SELECT actor_id, COUNT(*) AS n_events
            FROM pricing_event_detail
            WHERE crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
            GROUP BY actor_id
        ) pe ON a.actor_id = pe.actor_id
        WHERE a.crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
          AND a.pricing_model = 'PAY_PER_EVENT'
        ORDER BY a.actor_id
    """).df()

    df["agentic_int"] = df["agentic"].astype(int)
    df["primary_category"] = df["primary_category"].fillna("UNKNOWN").str.strip('"')
    df["n_pricing_events"] = df["n_pricing_events"].fillna(0)
    df["desc_length"] = df["desc_length"].fillna(0)
    df["log_builds"] = df["log_builds"].fillna(0)
    df["dev_portfolio_size"] = df["dev_portfolio_size"].fillna(1)

    return df


# ============================================================
# Part 1: Category-level premium heterogeneity
# ============================================================

def category_premiums(df: pd.DataFrame) -> dict:
    """Run per-category regressions to show premium heterogeneity."""
    results = []
    cats = df["primary_category"].value_counts()
    # Only categories with >= 200 actors and both agentic/non-agentic
    valid_cats = []
    for cat in cats.index:
        sub = df[df["primary_category"] == cat]
        if len(sub) >= 200 and sub["agentic_int"].nunique() == 2:
            n_ag = sub["agentic_int"].sum()
            n_na = len(sub) - n_ag
            if n_ag >= 30 and n_na >= 30:
                valid_cats.append(cat)

    logger.info("Categories with sufficient data for per-category regression: %d", len(valid_cats))

    for cat in valid_cats:
        sub = df[df["primary_category"] == cat]
        try:
            m = smf.ols(
                "log_mu ~ agentic_int + log_builds + dev_portfolio_size + n_pricing_events",
                data=sub,
            ).fit(cov_type="HC1")
            coef = m.params.get("agentic_int", np.nan)
            se = m.bse.get("agentic_int", np.nan)
            p = m.pvalues.get("agentic_int", np.nan)

            results.append({
                "category": cat,
                "n_actors": len(sub),
                "n_agentic": int(sub["agentic_int"].sum()),
                "n_nonagentic": int(len(sub) - sub["agentic_int"].sum()),
                "pct_agentic": round(sub["agentic_int"].mean() * 100, 1),
                "coef_agentic": round(float(coef), 4),
                "se": round(float(se), 4),
                "p_value": float(p),
                "significant": p < 0.05,
                "direction": "positive" if coef > 0 else "negative" if coef < 0 else "zero",
            })
        except Exception as e:
            logger.warning("Failed for category %s: %s", cat, e)

    results.sort(key=lambda x: x["coef_agentic"], reverse=True)

    # Save T11
    t11 = pd.DataFrame(results)
    t11.to_csv(BASE / "tables" / "T11_category_premiums.csv", index=False)
    logger.info("T11 saved. %d categories analyzed.", len(results))

    for r in results:
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "ns"
        logger.info("  %-25s coef=%.4f %s  (n=%d, %d%% agentic)",
                     r["category"], r["coef_agentic"], sig, r["n_actors"], r["pct_agentic"])

    # F8: Heatmap of category premiums
    if len(results) > 0:
        plot_df = t11[["category", "coef_agentic", "pct_agentic", "n_actors"]].copy()
        plot_df = plot_df.sort_values("coef_agentic", ascending=True)

        fig, ax = plt.subplots(figsize=(9, max(5, len(plot_df) * 0.35)))
        colors = ["#C44E52" if c < 0 else "#4C72B0" if p < 0.05 else "#8C8C8C"
                   for c, p in zip(plot_df["coef_agentic"], t11.set_index("category").loc[plot_df["category"], "p_value"])]

        bars = ax.barh(range(len(plot_df)), plot_df["coef_agentic"], color=colors, alpha=0.85)
        ax.set_yticks(range(len(plot_df)))
        ax.set_yticklabels(plot_df["category"], fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Agentic Premium (OLS coefficient on log monthly users)")
        ax.set_title("Agentic Adoption Premium by Category\n(blue = sig. positive, red = sig. negative, gray = n.s.)")

        # Annotate with sample size
        for i, (_, row) in enumerate(plot_df.iterrows()):
            ax.text(row["coef_agentic"] + 0.005 if row["coef_agentic"] >= 0 else row["coef_agentic"] - 0.005,
                    i, f"n={row['n_actors']:,}", va="center", fontsize=7,
                    ha="left" if row["coef_agentic"] >= 0 else "right")

        fig.tight_layout()
        fig.savefig(BASE / "figures" / "F8_category_premium_heatmap.pdf")
        plt.close(fig)
        logger.info("F8 saved.")

    return {"per_category": results}


# ============================================================
# Part 2: Scale-dependent premium (developer tier interaction)
# ============================================================

def scale_premiums(df: pd.DataFrame) -> dict:
    """Analyze how the premium varies by developer scale."""
    results = []

    tier_order = ["Solo", "Boutique", "Mid", "Whale"]
    for tier in tier_order:
        sub = df[df["dev_tier"] == tier]
        if sub["agentic_int"].nunique() < 2 or len(sub) < 100:
            continue
        n_ag = sub["agentic_int"].sum()
        n_na = len(sub) - n_ag
        if n_ag < 20 or n_na < 20:
            continue

        m = smf.ols(
            "log_mu ~ agentic_int + log_builds + n_pricing_events",
            data=sub,
        ).fit(cov_type="HC1")
        coef = m.params.get("agentic_int", np.nan)
        se = m.bse.get("agentic_int", np.nan)
        p = m.pvalues.get("agentic_int", np.nan)

        results.append({
            "tier": tier,
            "n_actors": len(sub),
            "n_agentic": int(n_ag),
            "n_nonagentic": int(n_na),
            "coef_agentic": round(float(coef), 4),
            "se": round(float(se), 4),
            "p_value": float(p),
        })
        logger.info("  %s: coef=%.4f (SE=%.4f, p=%.2e), n=%d",
                     tier, coef, se, p, len(sub))

    t12 = pd.DataFrame(results)
    t12.to_csv(BASE / "tables" / "T12_scale_premiums.csv", index=False)
    logger.info("T12 saved.")

    # Interaction model: agentic × tier
    df_interaction = df.copy()
    df_interaction["dev_tier"] = pd.Categorical(df_interaction["dev_tier"],
                                                 categories=tier_order, ordered=True)
    m_interact = smf.ols(
        "log_mu ~ agentic_int * C(dev_tier) + log_builds + n_pricing_events",
        data=df_interaction,
    ).fit(cov_type="HC1")

    interaction_terms = {k: {"coef": round(float(v), 4),
                             "p": round(float(m_interact.pvalues[k]), 4)}
                         for k, v in m_interact.params.items()
                         if "agentic_int" in k}
    logger.info("Interaction terms: %s", interaction_terms)

    # F10: Scale interaction plot
    if len(results) > 0:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        tiers = [r["tier"] for r in results]
        coefs = [r["coef_agentic"] for r in results]
        ses = [r["se"] for r in results]
        colors = ["#4C72B0" if r["p_value"] < 0.05 else "#8C8C8C" for r in results]

        ax.bar(tiers, coefs, yerr=[1.96 * s for s in ses], capsize=5,
               color=colors, alpha=0.85, edgecolor="white")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Developer Tier")
        ax.set_ylabel("Agentic Premium (OLS coefficient)")
        ax.set_title("Agentic Premium by Developer Scale\n(with 95% CI; blue = p < 0.05)")

        for i, r in enumerate(results):
            ax.text(i, coefs[i] + 1.96 * ses[i] + 0.01,
                    f"n={r['n_actors']:,}", ha="center", fontsize=8)

        fig.tight_layout()
        fig.savefig(BASE / "figures" / "F10_scale_interaction.pdf")
        plt.close(fig)
        logger.info("F10 saved.")

    return {"per_tier": results, "interaction_terms": interaction_terms}


# ============================================================
# Part 3: Competitive concentration analysis
# ============================================================

def concentration_analysis(df: pd.DataFrame) -> dict:
    """Compare competitive concentration between agentic and non-agentic segments."""
    results = {}

    # Gini coefficients: overall, agentic, non-agentic
    ag = df[df["agentic"]]["monthly_users"].values
    na = df[~df["agentic"]]["monthly_users"].values

    gini_all = gini(df["monthly_users"].values)
    gini_ag = gini(ag)
    gini_na = gini(na)

    results["gini"] = {
        "all_ppe": round(gini_all, 4),
        "agentic": round(gini_ag, 4),
        "non_agentic": round(gini_na, 4),
        "agentic_more_concentrated": gini_ag > gini_na,
    }
    logger.info("Gini coefficients: all=%.4f, agentic=%.4f, non-agentic=%.4f",
                gini_all, gini_ag, gini_na)

    # Top-N concentration ratios
    for top_n_pct in [1, 5, 10]:
        for label, subset in [("agentic", ag), ("non_agentic", na)]:
            sorted_vals = np.sort(subset)[::-1]
            top_n = max(1, int(len(sorted_vals) * top_n_pct / 100))
            total = sorted_vals.sum()
            top_share = sorted_vals[:top_n].sum() / total if total > 0 else 0
            key = f"top_{top_n_pct}pct_{label}"
            results.setdefault("concentration_ratios", {})[key] = {
                "top_n_actors": top_n,
                "share_of_total_users": round(float(top_share), 4),
            }

    logger.info("Top 1%% agentic: %.1f%% of users; non-agentic: %.1f%%",
                results["concentration_ratios"]["top_1pct_agentic"]["share_of_total_users"] * 100,
                results["concentration_ratios"]["top_1pct_non_agentic"]["share_of_total_users"] * 100)

    # Per-category concentration: agentic penetration vs HHI
    cat_conc = []
    for cat in df["primary_category"].unique():
        sub = df[df["primary_category"] == cat]
        if len(sub) < 50:
            continue
        total_mu = sub["monthly_users"].sum()
        if total_mu == 0:
            continue
        shares = sub["monthly_users"] / total_mu
        hhi = float((shares ** 2).sum() * 10000)
        pct_ag = sub["agentic"].mean() * 100
        cat_conc.append({
            "category": cat,
            "n_actors": len(sub),
            "pct_agentic": round(pct_ag, 1),
            "hhi": round(hhi, 1),
            "gini": round(gini(sub["monthly_users"].values), 4),
        })

    cat_conc_df = pd.DataFrame(cat_conc)
    cat_conc_df.to_csv(BASE / "tables" / "T13_concentration.csv", index=False)
    results["category_concentration"] = cat_conc
    logger.info("T13 saved. %d categories.", len(cat_conc))

    # Correlation: agentic penetration vs concentration
    if len(cat_conc_df) > 5:
        from scipy.stats import spearmanr
        rho, p = spearmanr(cat_conc_df["pct_agentic"], cat_conc_df["gini"])
        results["agentic_concentration_correlation"] = {
            "spearman_rho": round(float(rho), 4),
            "p_value": float(p),
        }
        logger.info("Correlation (agentic%% vs Gini): rho=%.4f, p=%.4f", rho, p)

    # F9: Lorenz curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_i, (label, subset, color) in enumerate([
        ("Agentic PPE", ag, "#4C72B0"),
        ("Non-agentic PPE", na, "#C44E52"),
    ]):
        sorted_vals = np.sort(subset)
        cumulative = np.cumsum(sorted_vals) / sorted_vals.sum() if sorted_vals.sum() > 0 else np.zeros(len(sorted_vals))
        x = np.linspace(0, 1, len(sorted_vals))

        axes[0].plot(x, cumulative, label=f"{label} (Gini={gini(subset):.3f})",
                     color=color, linewidth=2)

    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect equality")
    axes[0].set_xlabel("Cumulative share of actors")
    axes[0].set_ylabel("Cumulative share of monthly users")
    axes[0].set_title("Lorenz Curves: Adoption Concentration")
    axes[0].legend(fontsize=9)
    axes[0].set_aspect("equal")

    # Right panel: category-level agentic penetration vs Gini
    if len(cat_conc_df) > 3:
        axes[1].scatter(cat_conc_df["pct_agentic"], cat_conc_df["gini"],
                        s=np.sqrt(cat_conc_df["n_actors"]) * 3,
                        color="#4C72B0", alpha=0.6, edgecolors="white")
        for _, row in cat_conc_df.iterrows():
            if row["n_actors"] >= 1000:
                axes[1].annotate(row["category"], (row["pct_agentic"], row["gini"]),
                                 fontsize=7, alpha=0.8)
        axes[1].set_xlabel("% Agentic in Category")
        axes[1].set_ylabel("Gini Coefficient (adoption concentration)")
        axes[1].set_title("Category Agentic Penetration vs. Concentration")

    fig.tight_layout()
    fig.savefig(BASE / "figures" / "F9_concentration_lorenz.pdf")
    plt.close(fig)
    logger.info("F9 saved.")

    return results


def main() -> None:
    for subdir in ["tables", "figures", "data"]:
        (BASE / subdir).mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        df = prepare_data(con)

        logger.info("=" * 60)
        logger.info("Part 1: Category-level premium heterogeneity")
        logger.info("=" * 60)
        cat_results = category_premiums(df)

        logger.info("=" * 60)
        logger.info("Part 2: Scale-dependent premium")
        logger.info("=" * 60)
        scale_results = scale_premiums(df)

        logger.info("=" * 60)
        logger.info("Part 3: Competitive concentration")
        logger.info("=" * 60)
        conc_results = concentration_analysis(df)

        all_results = {
            "category_premiums": cat_results,
            "scale_premiums": scale_results,
            "concentration": conc_results,
        }

        with open(BASE / "data" / "block7_heterogeneity.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        logger.info("=" * 60)
        logger.info("Block 7 (NEW) complete.")
    finally:
        con.close()


if __name__ == "__main__":
    main()
