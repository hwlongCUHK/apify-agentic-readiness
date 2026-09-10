"""Block 3: The Agentic Divide -- Raw Premium (RQ2, Step 1).

Outputs:
  - results/tables/T4_raw_comparison.csv
  - results/tables/T5_ppe_comparison.csv
  - results/data/block3_distribution.csv     (for F3)
  - results/data/block3_stats.json
  - results/figures/F3_adoption_distributions.pdf
"""

import json
import logging
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "apify_panel.duckdb"
BASE = Path(__file__).resolve().parent.parent / "results"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
})


def run(con: duckdb.DuckDBPyConnection) -> dict:
    results: dict = {}

    # --- Step 3.1: Raw comparison (T4) ---
    t4 = con.execute("""
        SELECT
            is_agentic_payments_whitelisted AS agentic,
            COUNT(*) AS n,
            ROUND(AVG(monthly_users), 2) AS avg_mu,
            MEDIAN(monthly_users) AS med_mu,
            ROUND(AVG(COALESCE(runs_30d_total, 0)), 2) AS avg_runs,
            MEDIAN(COALESCE(runs_30d_total, 0)) AS med_runs,
            ROUND(AVG(bookmarks), 2) AS avg_bookmarks,
            MEDIAN(bookmarks) AS med_bookmarks,
            ROUND(AVG(rating), 3) AS avg_rating,
            ROUND(AVG(total_builds), 1) AS avg_builds,
            ROUND(AVG(review_count), 2) AS avg_reviews
        FROM actor_day
        WHERE crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
        GROUP BY agentic
        ORDER BY agentic
    """).df()
    t4.to_csv(BASE / "tables" / "T4_raw_comparison.csv", index=False)
    results["T4"] = t4.to_dict(orient="records")
    logger.info("T4: Raw comparison across all pricing models")
    for _, row in t4.iterrows():
        logger.info("  agentic=%s: n=%d, avg_mu=%.2f, med_mu=%d",
                     row["agentic"], row["n"], row["avg_mu"], row["med_mu"])

    # --- Step 3.2: Within-PPE comparison (T5) ---
    t5 = con.execute("""
        SELECT
            is_agentic_payments_whitelisted AS agentic,
            COUNT(*) AS n,
            ROUND(AVG(monthly_users), 2) AS avg_mu,
            MEDIAN(monthly_users) AS med_mu,
            percentile_cont(0.75) WITHIN GROUP (ORDER BY monthly_users) AS p75_mu,
            percentile_cont(0.90) WITHIN GROUP (ORDER BY monthly_users) AS p90_mu,
            percentile_cont(0.95) WITHIN GROUP (ORDER BY monthly_users) AS p95_mu,
            ROUND(AVG(COALESCE(runs_30d_total, 0)), 2) AS avg_runs,
            MEDIAN(COALESCE(runs_30d_total, 0)) AS med_runs,
            ROUND(AVG(total_builds), 1) AS avg_builds,
            ROUND(AVG(bookmarks), 2) AS avg_bookmarks
        FROM actor_day
        WHERE crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
          AND pricing_model = 'PAY_PER_EVENT'
        GROUP BY agentic
        ORDER BY agentic
    """).df()
    t5.to_csv(BASE / "tables" / "T5_ppe_comparison.csv", index=False)
    results["T5"] = t5.to_dict(orient="records")
    logger.info("T5: Within-PPE comparison")
    for _, row in t5.iterrows():
        logger.info("  agentic=%s: n=%d, avg_mu=%.2f, med_mu=%d, p90_mu=%.0f",
                     row["agentic"], row["n"], row["avg_mu"], row["med_mu"], row["p90_mu"])

    # --- Step 3.3: Distribution data for F3 ---
    dist_df = con.execute("""
        SELECT
            is_agentic_payments_whitelisted AS agentic,
            monthly_users,
            LN(1 + monthly_users) AS log_mu,
            COALESCE(runs_30d_total, 0) AS runs_30d_total,
            LN(1 + COALESCE(runs_30d_total, 0)) AS log_runs
        FROM actor_day
        WHERE crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
          AND pricing_model = 'PAY_PER_EVENT'
    """).df()
    dist_df.to_csv(BASE / "data" / "block3_distribution.csv", index=False)

    # Figure F3: Dual KDE - agentic vs non-agentic PPE
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    agentic_mu = dist_df[dist_df["agentic"]]["log_mu"]
    nonagentic_mu = dist_df[~dist_df["agentic"]]["log_mu"]

    sns.kdeplot(agentic_mu, ax=axes[0], label=f"Agentic (n={len(agentic_mu):,})",
                color="#4C72B0", fill=True, alpha=0.3)
    sns.kdeplot(nonagentic_mu, ax=axes[0], label=f"Non-agentic (n={len(nonagentic_mu):,})",
                color="#C44E52", fill=True, alpha=0.3)
    axes[0].set_xlabel("ln(1 + monthly users)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Monthly Users (PPE Only)")
    axes[0].legend(fontsize=8)

    agentic_runs = dist_df[dist_df["agentic"]]["log_runs"]
    nonagentic_runs = dist_df[~dist_df["agentic"]]["log_runs"]

    sns.kdeplot(agentic_runs, ax=axes[1], label="Agentic",
                color="#4C72B0", fill=True, alpha=0.3)
    sns.kdeplot(nonagentic_runs, ax=axes[1], label="Non-agentic",
                color="#C44E52", fill=True, alpha=0.3)
    axes[1].set_xlabel("ln(1 + runs in 30d)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("30-Day Runs (PPE Only)")
    axes[1].legend(fontsize=8)

    fig.suptitle("Agentic vs. Non-Agentic Adoption Distributions (PPE Actors)", y=1.02)
    fig.tight_layout()
    fig.savefig(BASE / "figures" / "F3_adoption_distributions.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("F3 saved.")

    # --- Step 3.4: Non-parametric tests ---
    ag_mu = dist_df[dist_df["agentic"]]["monthly_users"].values
    na_mu = dist_df[~dist_df["agentic"]]["monthly_users"].values
    ag_runs = dist_df[dist_df["agentic"]]["runs_30d_total"].values
    na_runs = dist_df[~dist_df["agentic"]]["runs_30d_total"].values

    stat_mu, p_mu = mannwhitneyu(ag_mu, na_mu, alternative="two-sided")
    stat_runs, p_runs = mannwhitneyu(ag_runs, na_runs, alternative="two-sided")

    # Effect size: rank-biserial correlation r = 1 - 2U/(n1*n2)
    r_mu = 1 - 2 * stat_mu / (len(ag_mu) * len(na_mu))
    r_runs = 1 - 2 * stat_runs / (len(ag_runs) * len(na_runs))

    stats = {
        "mann_whitney_monthly_users": {
            "U": float(stat_mu), "p": float(p_mu),
            "rank_biserial_r": round(float(r_mu), 4),
            "n_agentic": len(ag_mu), "n_nonagentic": len(na_mu),
        },
        "mann_whitney_runs_30d": {
            "U": float(stat_runs), "p": float(p_runs),
            "rank_biserial_r": round(float(r_runs), 4),
            "n_agentic": len(ag_runs), "n_nonagentic": len(na_runs),
        },
    }
    results["statistics"] = stats
    logger.info("Mann-Whitney U (monthly_users): U=%.0f, p=%.2e, r=%.4f",
                stat_mu, p_mu, r_mu)
    logger.info("Mann-Whitney U (runs_30d):      U=%.0f, p=%.2e, r=%.4f",
                stat_runs, p_runs, r_runs)

    with open(BASE / "data" / "block3_stats.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def main() -> None:
    for subdir in ["tables", "figures", "data"]:
        (BASE / subdir).mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        run(con)
        logger.info("Block 3 complete.")
    finally:
        con.close()


if __name__ == "__main__":
    main()
