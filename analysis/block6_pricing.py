"""Block 6: Pricing Microstructure Analysis (RQ4).

Outputs:
  - results/tables/T9_pricing_complexity.csv
  - results/tables/T10_agentic_pricing.csv
  - results/data/block6_price_distribution.csv   (for F6)
  - results/data/block6_complexity_adoption.csv   (for F7)
  - results/data/block6_rental_profile.json
  - results/figures/F6_price_distributions.pdf
  - results/figures/F7_complexity_adoption.pdf
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
BASE = Path(__file__).resolve().parent.parent / "results"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
})


def run(con: duckdb.DuckDBPyConnection) -> dict:
    results: dict = {}

    # --- Step 6.1: PPE pricing anatomy (T9) ---
    t9 = con.execute("""
        SELECT
            CASE
                WHEN n_events = 1 THEN '1 event'
                WHEN n_events BETWEEN 2 AND 3 THEN '2-3 events'
                WHEN n_events BETWEEN 4 AND 5 THEN '4-5 events'
                ELSE '6+ events'
            END AS complexity_tier,
            COUNT(*) AS n_actors,
            ROUND(AVG(med_price), 6) AS avg_median_price,
            ROUND(AVG(max_price), 4) AS avg_max_price,
            ROUND(AVG(mean_price), 6) AS avg_mean_price
        FROM (
            SELECT actor_id,
                   COUNT(*) AS n_events,
                   MEDIAN(event_price_usd) AS med_price,
                   MAX(event_price_usd) AS max_price,
                   AVG(event_price_usd) AS mean_price
            FROM pricing_event_detail
            WHERE crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
            GROUP BY actor_id
        ) events
        GROUP BY complexity_tier
        ORDER BY MIN(n_events)
    """).df()
    t9.to_csv(BASE / "tables" / "T9_pricing_complexity.csv", index=False)
    results["T9"] = t9.to_dict(orient="records")
    logger.info("T9 saved. Pricing complexity tiers:")
    for _, row in t9.iterrows():
        logger.info("  %s: %d actors, avg median price=$%.6f",
                     row["complexity_tier"], row["n_actors"], row["avg_median_price"])

    # --- Step 6.2: Agentic vs non-agentic pricing (T10) ---
    t10 = con.execute("""
        SELECT
            a.is_agentic_payments_whitelisted AS agentic,
            COUNT(DISTINCT pe.actor_id) AS n_actors,
            ROUND(AVG(pe.n_events), 2) AS avg_n_events,
            ROUND(AVG(pe.median_price), 6) AS avg_median_event_price,
            ROUND(AVG(pe.mean_price), 6) AS avg_mean_event_price,
            ROUND(AVG(pe.max_price), 4) AS avg_max_price,
            ROUND(AVG(pe.price_range), 4) AS avg_price_range
        FROM actor_day a
        JOIN (
            SELECT actor_id, crawl_date,
                   COUNT(*) AS n_events,
                   MEDIAN(event_price_usd) AS median_price,
                   AVG(event_price_usd) AS mean_price,
                   MAX(event_price_usd) AS max_price,
                   MAX(event_price_usd) - MIN(event_price_usd) AS price_range
            FROM pricing_event_detail
            WHERE crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
            GROUP BY actor_id, crawl_date
        ) pe ON a.actor_id = pe.actor_id AND a.crawl_date = pe.crawl_date
        WHERE a.crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
          AND a.pricing_model = 'PAY_PER_EVENT'
        GROUP BY agentic
        ORDER BY agentic
    """).df()
    t10.to_csv(BASE / "tables" / "T10_agentic_pricing.csv", index=False)
    results["T10"] = t10.to_dict(orient="records")
    logger.info("T10 saved. Agentic pricing comparison:")
    for _, row in t10.iterrows():
        logger.info("  agentic=%s: %d actors, avg events=%.2f, median price=$%.6f",
                     row["agentic"], row["n_actors"], row["avg_n_events"],
                     row["avg_median_event_price"])

    # --- Step 6.3: Event price distribution (F6) ---
    price_df = con.execute("""
        SELECT
            a.is_agentic_payments_whitelisted AS agentic,
            pe.event_price_usd,
            LN(pe.event_price_usd) AS log_price
        FROM pricing_event_detail pe
        JOIN actor_day a ON pe.actor_id = a.actor_id AND pe.crawl_date = a.crawl_date
        WHERE pe.crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
          AND pe.event_price_usd > 0
    """).df()
    price_df.to_csv(BASE / "data" / "block6_price_distribution.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    agentic_prices = price_df[price_df["agentic"]]["log_price"]
    nonagentic_prices = price_df[~price_df["agentic"]]["log_price"]

    sns.kdeplot(agentic_prices, ax=ax, label=f"Agentic (n={len(agentic_prices):,})",
                color="#4C72B0", fill=True, alpha=0.3)
    sns.kdeplot(nonagentic_prices, ax=ax, label=f"Non-agentic (n={len(nonagentic_prices):,})",
                color="#C44E52", fill=True, alpha=0.3)
    ax.set_xlabel("ln(event price USD)")
    ax.set_ylabel("Density")
    ax.set_title("Event Price Distributions: Agentic vs. Non-Agentic PPE Actors")
    ax.legend()
    fig.tight_layout()
    fig.savefig(BASE / "figures" / "F6_price_distributions.pdf")
    plt.close(fig)
    logger.info("F6 saved.")

    # --- Step 6.4: Pricing complexity vs adoption (F7) ---
    complexity_df = con.execute("""
        SELECT
            pe.actor_id,
            pe.n_events,
            pe.cv_price,
            a.monthly_users,
            a.is_agentic_payments_whitelisted AS agentic,
            LN(1 + a.monthly_users) AS log_mu
        FROM (
            SELECT actor_id,
                   COUNT(*) AS n_events,
                   CASE WHEN AVG(event_price_usd) > 0
                        THEN STDDEV(event_price_usd) / AVG(event_price_usd)
                        ELSE 0 END AS cv_price
            FROM pricing_event_detail
            WHERE crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
            GROUP BY actor_id
        ) pe
        JOIN actor_day a ON pe.actor_id = a.actor_id
        WHERE a.crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
          AND a.pricing_model = 'PAY_PER_EVENT'
        ORDER BY a.actor_id
    """).df()
    complexity_df.to_csv(BASE / "data" / "block6_complexity_adoption.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: n_events vs log_mu, by agentic
    for agentic_val, color, label in [(True, "#4C72B0", "Agentic"), (False, "#C44E52", "Non-agentic")]:
        sub = complexity_df[complexity_df["agentic"] == agentic_val]
        # Binned means
        bins = sub.groupby("n_events")["log_mu"].agg(["mean", "count"]).reset_index()
        bins = bins[bins["count"] >= 10]
        axes[0].plot(bins["n_events"], bins["mean"], "o-", color=color, label=label, markersize=5)

    axes[0].set_xlabel("Number of Pricing Events")
    axes[0].set_ylabel("Mean ln(1 + monthly users)")
    axes[0].set_title("Pricing Complexity vs. Adoption")
    axes[0].legend()

    # Right: scatter with jitter (seeded for reproducibility)
    np.random.seed(42)
    for agentic_val, color, label in [(True, "#4C72B0", "Agentic"), (False, "#C44E52", "Non-agentic")]:
        sub = complexity_df[complexity_df["agentic"] == agentic_val].sample(
            min(2000, len(complexity_df[complexity_df["agentic"] == agentic_val])),
            random_state=42,
        )
        axes[1].scatter(
            sub["n_events"] + np.random.normal(0, 0.1, len(sub)),
            sub["log_mu"],
            alpha=0.1, s=5, color=color, label=label,
        )
    axes[1].set_xlabel("Number of Pricing Events")
    axes[1].set_ylabel("ln(1 + monthly users)")
    axes[1].set_title("Individual Actors (Subsampled)")
    axes[1].legend()

    fig.suptitle("Pricing Complexity and Tool Adoption", y=1.02)
    fig.tight_layout()
    fig.savefig(BASE / "figures" / "F7_complexity_adoption.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("F7 saved.")

    # --- Step 6.5: Rental actor profile ---
    rental = con.execute("""
        SELECT
            COUNT(*) AS n_rental,
            ROUND(AVG(a.monthly_users), 2) AS avg_mu,
            MEDIAN(a.monthly_users) AS med_mu,
            ROUND(AVG(COALESCE(a.runs_30d_total, 0)), 2) AS avg_runs,
            ROUND(AVG(a.total_builds), 1) AS avg_builds,
            ROUND(AVG(a.bookmarks), 2) AS avg_bookmarks,
            ROUND(AVG(p.price_per_unit_usd), 2) AS avg_monthly_fee,
            MEDIAN(p.price_per_unit_usd) AS med_monthly_fee
        FROM actor_day a
        JOIN pricing_day p ON a.actor_id = p.actor_id AND a.crawl_date = p.crawl_date
        WHERE a.crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
          AND a.pricing_model = 'FLAT_PRICE_PER_MONTH'
    """).df().to_dict(orient="records")[0]
    results["rental_profile"] = rental
    logger.info("Rental profile: %d actors, avg fee=$%.2f, avg MU=%.2f",
                rental["n_rental"], rental["avg_monthly_fee"], rental["avg_mu"])

    with open(BASE / "data" / "block6_rental_profile.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def main() -> None:
    for subdir in ["tables", "figures", "data"]:
        (BASE / subdir).mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        run(con)
        logger.info("Block 6 complete.")
    finally:
        con.close()


if __name__ == "__main__":
    main()
