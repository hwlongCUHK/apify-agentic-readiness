"""Block 14: Multi-hot category robustness (RQ3).

Addresses a substantive concern about the category-heterogeneity finding: the
primary analysis assigns each actor its *primary* category (the first element of
the API-returned ``categories`` array), yet 84% of actors are multi-category, so
the Bonferroni-surviving categories (AI, Social Media, Lead Generation) could in
principle be an artifact of label ordering rather than genuine category-level
heterogeneity.

This block re-estimates per-category whitelist coefficients under a *multi-hot*
"any membership" assignment: an actor contributes to every category it is tagged
with, so the estimate no longer depends on which tag appears first. The three
Bonferroni-surviving categories are then compared against the primary-category
estimates.

Outputs:
  - results/tables/T20_category_multihot.csv
  - results/data/block14_category_multihot.json
"""

import json
import logging
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "apify_panel.duckdb"
BASE = Path(__file__).resolve().parent.parent / "results"

# Controls match block7 (primary-category specification) exactly.
FORMULA = "log_mu ~ agentic_int + log_builds + dev_portfolio_size + n_pricing_events"


def prepare_data(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load PPE actors with full multi-hot category membership."""
    df = con.execute("""
        SELECT
            a.actor_id,
            a.username,
            a.is_agentic_payments_whitelisted AS agentic,
            a.monthly_users,
            a.total_builds,
            dev.n_actors AS dev_portfolio_size,
            COALESCE(pe.n_events, 0) AS n_pricing_events,
            CAST(json_extract(a.categories, '$[*]') AS VARCHAR[]) AS categories_array,
            json_extract_string(a.categories, '$[0]') AS primary_category,
            LN(1 + a.monthly_users) AS log_mu,
            LN(1 + a.total_builds) AS log_builds
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
    df["log_builds"] = df["log_builds"].fillna(0)
    df["dev_portfolio_size"] = df["dev_portfolio_size"].fillna(1)
    df["categories_list"] = df["categories_array"].apply(
        lambda arr: [c.strip('"') for c in (arr if arr is not None else [])]
    )
    return df


def multihot_category_premiums(df: pd.DataFrame) -> list:
    """Per-category regression under multi-hot (any-membership) assignment."""
    all_cats = sorted({c for cats in df["categories_list"] for c in cats})
    results = []
    for cat in all_cats:
        sub = df[df["categories_list"].apply(lambda x: cat in x)]
        n_ag = int(sub["agentic_int"].sum())
        n_na = int(len(sub) - n_ag)
        if len(sub) < 200 or n_ag < 30 or n_na < 30:
            continue
        m = smf.ols(FORMULA, data=sub).fit(cov_type="HC1")
        results.append({
            "category": cat,
            "n_actors": int(len(sub)),
            "n_agentic": n_ag,
            "n_nonagentic": n_na,
            "pct_agentic": round(sub["agentic_int"].mean() * 100, 1),
            "coef_agentic": round(float(m.params["agentic_int"]), 4),
            "se": round(float(m.bse["agentic_int"]), 4),
            "p_value": float(m.pvalues["agentic_int"]),
        })
    results.sort(key=lambda r: -r["coef_agentic"])
    return results


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = prepare_data(con)
    multi = multihot_category_premiums(df)

    # Primary-category estimates (block7) for comparison.
    primary_df = pd.read_csv(BASE / "tables" / "T11_category_premiums.csv")
    primary = {r["category"]: r for _, r in primary_df.iterrows()}

    n_cats = len(multi)
    bonf_alpha = 0.05 / n_cats
    targets = ["AI", "SOCIAL_MEDIA", "LEAD_GENERATION"]

    # Comparison summary for the three Bonferroni-surviving categories.
    comparison = []
    for cat in targets:
        pm = primary.get(cat)
        mm = next((r for r in multi if r["category"] == cat), None)
        if pm is None or mm is None:
            continue
        comparison.append({
            "category": cat,
            "primary_coef": float(pm["coef_agentic"]),
            "primary_p": float(pm["p_value"]),
            "multihot_coef": mm["coef_agentic"],
            "multihot_p": mm["p_value"],
            "multihot_n_actors": mm["n_actors"],
            "bonferroni_survives": mm["p_value"] < bonf_alpha,
        })

    # Save.
    pd.DataFrame(multi).to_csv(BASE / "tables" / "T20_category_multihot.csv", index=False)
    payload = {
        "n_ppe_actors": int(len(df)),
        "multi_category_share_pct": round(
            100 * df["categories_list"].apply(len).gt(1).mean(), 1),
        "n_categories_threshold": n_cats,
        "bonferroni_alpha": round(bonf_alpha, 5),
        "targets": comparison,
    }
    (BASE / "data" / "block14_category_multihot.json").write_text(
        json.dumps(payload, indent=2))

    logger.info("T20 saved: %d categories (multi-hot).", n_cats)
    logger.info("Bonferroni alpha (multi-hot) = %.5f", bonf_alpha)
    for c in comparison:
        logger.info(
            "  %s: primary %.4f (p=%.3g) -> multi-hot %.4f (p=%.3g) [%s]",
            c["category"], c["primary_coef"], c["primary_p"],
            c["multihot_coef"], c["multihot_p"],
            "SURVIVES" if c["bonferroni_survives"] else "FAILS")


if __name__ == "__main__":
    main()
