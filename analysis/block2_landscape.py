"""Block 2: Marketplace Landscape Characterization (RQ1).

Outputs:
  - results/tables/T1_marketplace_overview.csv
  - results/tables/T2_developer_tiers.csv
  - results/data/block2_usage_distribution.csv  (for F1)
  - results/data/block2_category_breakdown.csv  (for F2)
  - results/figures/F1_usage_distribution.pdf
  - results/figures/F2_category_composition.pdf
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "apify_panel.duckdb"
DETAIL_PATH = (Path(__file__).resolve().parent.parent / "data" / "raw"
               / "2026-08-27.actor_detail.jsonl")
BASE = Path(__file__).resolve().parent.parent / "results"

# Publication style
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.figsize": (7, 4.5),
})


def load_detail() -> pd.DataFrame:
    """Actor detail (permission level + standby) from the archived JSONL.

    agentic-ready = limited-permissions AND non-standby, measured within the
    pay-per-event sample (the agentic-payments universe).
    """
    recs = [json.loads(line) for line in open(DETAIL_PATH)]
    d = pd.DataFrame(recs)
    d = d[d["status"] == 200][["actorId", "actorPermissionLevel", "standbyEnabled"]]
    d["limited"] = (d["actorPermissionLevel"] == "LIMITED_PERMISSIONS").astype(int)
    d["standby"] = d["standbyEnabled"].fillna(False).astype(int)
    d["eligible"] = ((d["limited"] == 1) & (d["standby"] == 0)).astype(int)
    return d


def run(con: duckdb.DuckDBPyConnection) -> dict:
    results: dict = {}

    # --- Step 2.1: Marketplace overview table (T1) ---
    t1 = con.execute("""
        WITH latest AS (
            SELECT * FROM actor_day
            WHERE crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
        )
        SELECT
            COUNT(*) AS total_actors,
            COUNT(DISTINCT username) AS total_developers,
            SUM(CASE WHEN pricing_model = 'PAY_PER_EVENT' THEN 1 ELSE 0 END) AS n_ppe,
            SUM(CASE WHEN pricing_model = 'FLAT_PRICE_PER_MONTH' THEN 1 ELSE 0 END) AS n_rental,
            SUM(CASE WHEN pricing_model = 'FREE' THEN 1 ELSE 0 END) AS n_free,
            ROUND(AVG(monthly_users), 2) AS avg_monthly_users,
            MEDIAN(monthly_users) AS median_monthly_users,
            ROUND(AVG(COALESCE(runs_30d_total, 0)), 2) AS avg_runs_30d,
            MEDIAN(COALESCE(runs_30d_total, 0)) AS median_runs_30d,
            ROUND(AVG(rating), 3) AS avg_rating,
            ROUND(AVG(bookmarks), 2) AS avg_bookmarks,
            ROUND(AVG(total_builds), 2) AS avg_builds,
            ROUND(AVG(review_count), 2) AS avg_reviews
        FROM latest
    """).df()

    # Agentic-readiness comes from the detail fields, not the whitelist flag.
    detail = load_detail()
    ppe_ids = con.execute("""
        SELECT actor_id FROM actor_day
        WHERE crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
          AND pricing_model = 'PAY_PER_EVENT'
    """).df()
    ppe_detail = ppe_ids.merge(detail, left_on="actor_id", right_on="actorId",
                               how="inner")
    n_eligible = int(ppe_detail["eligible"].sum())
    n_non_eligible = int(len(ppe_detail) - n_eligible)
    pct_eligible = round(n_eligible / len(ppe_detail) * 100, 1)
    t1["n_agentic"] = n_eligible
    t1["n_non_agentic"] = n_non_eligible
    t1["pct_agentic"] = pct_eligible

    t1.to_csv(BASE / "tables" / "T1_marketplace_overview.csv", index=False)
    results["T1"] = t1.to_dict(orient="records")[0]
    logger.info("T1 saved. Total actors: %s, Developers: %s, agentic-ready: %s",
                results["T1"]["total_actors"], results["T1"]["total_developers"],
                n_eligible)

    # --- Step 2.2: Usage distribution (F1) ---
    usage_df = con.execute("""
        SELECT monthly_users
        FROM actor_day
        WHERE crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
    """).df()

    # Bucket distribution
    buckets = con.execute("""
        SELECT
            CASE
                WHEN monthly_users = 0 THEN '0'
                WHEN monthly_users BETWEEN 1 AND 5 THEN '1-5'
                WHEN monthly_users BETWEEN 6 AND 20 THEN '6-20'
                WHEN monthly_users BETWEEN 21 AND 50 THEN '21-50'
                WHEN monthly_users BETWEEN 51 AND 100 THEN '51-100'
                WHEN monthly_users BETWEEN 101 AND 500 THEN '101-500'
                WHEN monthly_users BETWEEN 501 AND 1000 THEN '501-1K'
                ELSE '1K+'
            END AS bucket,
            COUNT(*) AS n_actors,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct
        FROM actor_day
        WHERE crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
        GROUP BY bucket
        ORDER BY MIN(monthly_users)
    """).df()
    buckets.to_csv(BASE / "data" / "block2_usage_distribution.csv", index=False)

    # Figure F1: Log-scale histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = usage_df["monthly_users"].values
    vals_pos = vals[vals > 0]
    ax.hist(np.log10(vals_pos + 1), bins=50, color="#4C72B0", edgecolor="white",
            alpha=0.85, density=True)
    ax.set_xlabel("log10(1 + monthly users)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Monthly Users Across All Actors (Log Scale)")
    ax.annotate(f"Zero users: {(vals == 0).sum():,} ({(vals == 0).mean()*100:.1f}%)",
                xy=(0.65, 0.9), xycoords="axes fraction", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    fig.tight_layout()
    fig.savefig(BASE / "figures" / "F1_usage_distribution.pdf")
    plt.close(fig)
    logger.info("F1 saved.")

    # --- Step 2.3: Category breakdown (F2) ---
    cat_df = con.execute("""
        WITH latest AS (
            SELECT a.*, UNNEST(
                CAST(json_extract(a.categories, '$[*]') AS VARCHAR[])
            ) AS cat
            FROM actor_day a
            WHERE a.crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
        )
        SELECT
            REPLACE(cat, '"', '') AS category,
            COUNT(*) AS n_actors,
            ROUND(SUM(CASE WHEN is_agentic_payments_whitelisted THEN 1.0 ELSE 0 END)
                  / COUNT(*) * 100, 1) AS pct_agentic,
            ROUND(AVG(monthly_users), 1) AS avg_monthly_users,
            MEDIAN(monthly_users) AS med_monthly_users,
            ROUND(AVG(COALESCE(runs_30d_total, 0)), 1) AS avg_runs_30d
        FROM latest
        GROUP BY category
        ORDER BY n_actors DESC
    """).df()
    cat_df.to_csv(BASE / "data" / "block2_category_breakdown.csv", index=False)

    # Figure F2: Category composition (horizontal bar)
    top_cats = cat_df.head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = range(len(top_cats))
    bars = ax.barh(y_pos, top_cats["n_actors"], color="#4C72B0", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_cats["category"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Actors")
    ax.set_title("Top 15 Categories by Size (with Agentic Share %)")

    for i, (n, pct) in enumerate(zip(top_cats["n_actors"], top_cats["pct_agentic"])):
        ax.text(n + 100, i, f"{pct:.0f}%", va="center", fontsize=8, color="#C44E52")

    fig.tight_layout()
    fig.savefig(BASE / "figures" / "F2_category_composition.pdf")
    plt.close(fig)
    logger.info("F2 saved. Top categories: %d", len(cat_df))

    # --- Step 2.4: Developer portfolio tiers (T2) ---
    # Tiers and developer/actor counts are over the full store; agentic-ready
    # share is computed among each developer's pay-per-event actors.
    ad = con.execute("""
        SELECT actor_id, username, pricing_model, monthly_users
        FROM actor_day
        WHERE crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
    """).df()
    m = ad.merge(detail, left_on="actor_id", right_on="actorId", how="left")
    port = ad.groupby("username")["actor_id"].count().rename("portfolio")
    m = m.merge(port, left_on="username", right_index=True)

    def _tier(n: int) -> str:
        if n >= 100:
            return "Whale (100+)"
        if n >= 10:
            return "Mid (10-99)"
        if n >= 2:
            return "Boutique (2-9)"
        return "Solo (1)"

    m["tier"] = m["portfolio"].map(_tier)

    # Full-store developer and actor counts per tier.
    counts = (m.groupby("tier")
              .agg(n_developers=("username", "nunique"),
                   total_actors=("actor_id", "count")))

    # Agentic-ready share over each developer's pay-per-event actors.
    ppe = m[m["pricing_model"] == "PAY_PER_EVENT"]
    share = (ppe.groupby("tier")["eligible"].mean() * 100).rename("avg_pct_agentic")

    t2 = counts.join(share).reset_index()
    t2["avg_pct_agentic"] = t2["avg_pct_agentic"].round(1)
    tier_order = ["Solo (1)", "Boutique (2-9)", "Mid (10-99)", "Whale (100+)"]
    t2["tier"] = pd.Categorical(t2["tier"], categories=tier_order, ordered=True)
    t2 = t2.sort_values("tier")
    t2.to_csv(BASE / "tables" / "T2_developer_tiers.csv", index=False)
    results["T2"] = t2.to_dict(orient="records")
    logger.info("T2 saved. Tiers: %s", t2["tier"].tolist())

    return results


def main() -> None:
    for subdir in ["tables", "figures", "data"]:
        (BASE / subdir).mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        results = run(con)
        with open(BASE / "data" / "block2_summary.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Block 2 complete.")
    finally:
        con.close()


if __name__ == "__main__":
    main()
