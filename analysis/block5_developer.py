"""Block 5: Developer Portfolio Transition Analysis (RQ3).

Outputs:
  - results/tables/T8_developer_typology.csv
  - results/data/block5_portfolio_scatter.csv  (for F5)
  - results/data/block5_within_dev_pairs.json
  - results/figures/F5_portfolio_scatter.pdf
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
from scipy.stats import wilcoxon

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "apify_panel.duckdb"
BASE = Path(__file__).resolve().parent.parent / "results"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
})


def run(con: duckdb.DuckDBPyConnection) -> dict:
    results: dict = {}

    # --- Step 5.1: Developer typology cross-tab (T8) ---
    t8 = con.execute("""
        WITH dev_profile AS (
            SELECT username,
                   COUNT(*) AS n_actors,
                   SUM(CASE WHEN is_agentic_payments_whitelisted THEN 1 ELSE 0 END) AS n_agentic,
                   SUM(CASE WHEN pricing_model = 'PAY_PER_EVENT' THEN 1 ELSE 0 END) AS n_ppe,
                   SUM(CASE WHEN pricing_model = 'FLAT_PRICE_PER_MONTH' THEN 1 ELSE 0 END) AS n_rental,
                   SUM(CASE WHEN pricing_model = 'FREE' THEN 1 ELSE 0 END) AS n_free,
                   AVG(monthly_users) AS avg_mu,
                   SUM(monthly_users) AS total_mu,
                   AVG(total_builds) AS avg_builds
            FROM actor_day
            WHERE crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
            GROUP BY username
        )
        SELECT
            CASE
                WHEN n_agentic = n_actors THEN 'Pure Agentic'
                WHEN n_agentic = 0 THEN 'Pure Traditional'
                ELSE 'Mixed'
            END AS dev_type,
            CASE
                WHEN n_actors >= 100 THEN 'Whale (100+)'
                WHEN n_actors >= 10 THEN 'Mid (10-99)'
                WHEN n_actors >= 2 THEN 'Boutique (2-9)'
                ELSE 'Solo (1)'
            END AS tier,
            COUNT(*) AS n_devs,
            SUM(n_actors) AS total_actors,
            ROUND(AVG(n_agentic * 100.0 / n_actors), 1) AS avg_pct_agentic,
            ROUND(AVG(avg_mu), 1) AS avg_mu_per_actor,
            ROUND(AVG(avg_builds), 1) AS avg_builds_per_actor
        FROM dev_profile
        GROUP BY dev_type, tier
        ORDER BY dev_type, MIN(n_actors)
    """).df()
    t8.to_csv(BASE / "tables" / "T8_developer_typology.csv", index=False)
    results["T8"] = t8.to_dict(orient="records")
    logger.info("T8 saved. Developer typology: %d rows", len(t8))
    for _, row in t8.iterrows():
        logger.info("  %s / %s: %d devs, %d actors, %.1f%% agentic",
                     row["dev_type"], row["tier"], row["n_devs"],
                     row["total_actors"], row["avg_pct_agentic"])

    # --- Step 5.2: Portfolio scatter data (F5) ---
    scatter_df = con.execute("""
        SELECT username,
               COUNT(*) AS portfolio_size,
               ROUND(SUM(CASE WHEN is_agentic_payments_whitelisted THEN 1.0 ELSE 0 END)
                     / COUNT(*), 3) AS agentic_share,
               AVG(monthly_users) AS avg_mu,
               AVG(total_builds) AS avg_builds
        FROM actor_day
        WHERE crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
        GROUP BY username
        HAVING portfolio_size >= 2
    """).df()
    scatter_df.to_csv(BASE / "data" / "block5_portfolio_scatter.csv", index=False)

    # Figure F5
    fig, ax = plt.subplots(figsize=(8, 5))
    sizes = np.clip(scatter_df["avg_mu"].values, 1, 500)  # cap for viz
    scatter = ax.scatter(
        scatter_df["portfolio_size"],
        scatter_df["agentic_share"],
        s=np.log1p(sizes) * 5,
        c=np.log1p(scatter_df["avg_mu"]),
        cmap="YlOrRd",
        alpha=0.4,
        edgecolors="none",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Portfolio Size (number of actors, log scale)")
    ax.set_ylabel("Agentic Share")
    ax.set_title("Developer Portfolio Size vs. Agentic Share")
    cbar = fig.colorbar(scatter, ax=ax, label="ln(1 + avg monthly users)")
    fig.tight_layout()
    fig.savefig(BASE / "figures" / "F5_portfolio_scatter.pdf")
    plt.close(fig)
    logger.info("F5 saved. Developers with 2+ actors: %d", len(scatter_df))

    # --- Step 5.3: Within-developer pair analysis ---
    pairs = con.execute("""
        WITH mixed_devs AS (
            SELECT username
            FROM actor_day
            WHERE crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
              AND pricing_model = 'PAY_PER_EVENT'
            GROUP BY username
            HAVING SUM(CASE WHEN is_agentic_payments_whitelisted THEN 1 ELSE 0 END) > 0
               AND SUM(CASE WHEN NOT is_agentic_payments_whitelisted THEN 1 ELSE 0 END) > 0
        )
        SELECT
            is_agentic_payments_whitelisted AS agentic,
            COUNT(*) AS n,
            ROUND(AVG(monthly_users), 2) AS avg_mu,
            MEDIAN(monthly_users) AS med_mu,
            ROUND(AVG(COALESCE(runs_30d_total, 0)), 2) AS avg_runs,
            MEDIAN(COALESCE(runs_30d_total, 0)) AS med_runs,
            ROUND(AVG(total_builds), 1) AS avg_builds,
            ROUND(AVG(bookmarks), 2) AS avg_bookmarks
        FROM actor_day
        WHERE crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
          AND pricing_model = 'PAY_PER_EVENT'
          AND username IN (SELECT username FROM mixed_devs)
        GROUP BY agentic
        ORDER BY agentic
    """).df()

    n_mixed = con.execute("""
        SELECT COUNT(DISTINCT username)
        FROM actor_day
        WHERE crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
          AND pricing_model = 'PAY_PER_EVENT'
          AND username IN (
              SELECT username
              FROM actor_day
              WHERE crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
                AND pricing_model = 'PAY_PER_EVENT'
              GROUP BY username
              HAVING SUM(CASE WHEN is_agentic_payments_whitelisted THEN 1 ELSE 0 END) > 0
                 AND SUM(CASE WHEN NOT is_agentic_payments_whitelisted THEN 1 ELSE 0 END) > 0
          )
    """).fetchone()[0]

    pair_result = {
        "n_mixed_developers": n_mixed,
        "comparison": pairs.to_dict(orient="records"),
    }
    results["within_dev_pairs"] = pair_result
    logger.info("Within-developer pairs: %d mixed devs", n_mixed)
    for _, row in pairs.iterrows():
        logger.info("  agentic=%s: n=%d, avg_mu=%.2f, med_mu=%d",
                     row["agentic"], row["n"], row["avg_mu"], row["med_mu"])

    with open(BASE / "data" / "block5_within_dev_pairs.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def main() -> None:
    for subdir in ["tables", "figures", "data"]:
        (BASE / subdir).mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        run(con)
        logger.info("Block 5 complete.")
    finally:
        con.close()


if __name__ == "__main__":
    main()
