"""Block 11: Developer-scale gradient and consumer-facing interaction test.

Computes two results that feed tab:dev_gap_tier and tab:interaction:
  (1) developer-scale gradient: within-developer adoption gap stratified by
      total portfolio tier (mixed PPE developers only).
  (2) consumer-facing vs developer-infrastructure interaction test across three
      specifications (pooled / pooled+controls / within-developer FE).

Outputs to results/data/block11_interaction_gradient.json.
"""

import json
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "apify_panel.duckdb"
BASE = Path(__file__).resolve().parent.parent / "results"

CONSUMER = {"AI", "SOCIAL_MEDIA", "TRAVEL", "REAL_ESTATE", "LEAD_GENERATION",
            "VIDEOS", "NEWS", "JOBS", "ECOMMERCE", "MARKETING", "EDUCATION",
            "FOR_CREATORS", "GAMES", "SPORTS"}
DEV_INFRA = {"DEVELOPER_TOOLS", "MCP_SERVERS", "INTEGRATIONS", "OPEN_SOURCE",
             "AGENTS"}


def prepare(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute("""
        SELECT a.username,
               a.is_agentic_payments_whitelisted AS agentic,
               a.monthly_users,
               dev.n_actors AS dev_portfolio_size,
               LENGTH(COALESCE(a.description, '')) AS desc_length,
               json_extract_string(a.categories, '$[0]') AS primary_category,
               EXTRACT(EPOCH FROM ((SELECT MAX(crawl_date)::TIMESTAMP FROM actor_day) - p.pricing_created_at)) / 86400.0 AS age_days
        FROM actor_day a
        JOIN pricing_day p ON a.actor_id = p.actor_id AND a.crawl_date = p.crawl_date
        LEFT JOIN (
            SELECT username, COUNT(*) AS n_actors
            FROM actor_day WHERE crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
            GROUP BY username
        ) dev ON a.username = dev.username
        WHERE a.crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
          AND a.pricing_model = 'PAY_PER_EVENT'
        ORDER BY a.actor_id
    """).df()
    con.close()
    df["agentic_int"] = df["agentic"].astype(int)
    df["primary_category"] = df["primary_category"].fillna("UNKNOWN").str.strip('"')
    df["desc_length"] = df["desc_length"].fillna(0)
    df["log_age"] = np.log1p(df["age_days"].fillna(df["age_days"].median()))
    df["log_mu"] = np.log1p(df["monthly_users"])
    return df


def main():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = prepare(con)

    # --- Developer-scale gradient (mixed PPE developers, by total portfolio) ---
    mixed = df.groupby("username")["agentic_int"].nunique()
    mixed_users = mixed[mixed == 2].index.tolist()
    mdf = df[df["username"].isin(mixed_users)].copy()
    dev_gap = mdf.groupby("username").apply(
        lambda g: g.loc[g["agentic_int"] == 1, "log_mu"].mean()
                  - g.loc[g["agentic_int"] == 0, "log_mu"].mean()
    )
    dev_tier = mdf.groupby("username")["dev_portfolio_size"].first().apply(
        lambda n: "Whale (100+)" if n >= 100 else ("Mid (10-99)" if n >= 10 else "Boutique (2-9)")
    )
    gap_df = pd.DataFrame({"gap": dev_gap, "tier": dev_tier})
    gradient = []
    for tier in ["Boutique (2-9)", "Mid (10-99)", "Whale (100+)"]:
        sub = gap_df[gap_df["tier"] == tier]["gap"]
        gradient.append({
            "tier": tier, "n_devs": int(len(sub)),
            "mean_gap": round(float(sub.mean()), 4),
            "median_gap": round(float(sub.median()), 4),
            "pct_positive": round(float((sub > 0).mean() * 100), 1),
        })

    # --- Consumer-facing interaction test ---
    df["consumer"] = df["primary_category"].isin(CONSUMER).astype(int)
    df_it = df[df["primary_category"].isin(CONSUMER | DEV_INFRA)].copy()

    specs = {
        "Pooled, no controls": "log_mu ~ agentic_int + consumer + agentic_int:consumer",
        "Pooled, listing-time ctrls": ("log_mu ~ agentic_int + consumer + agentic_int:consumer"
                                       " + log_age + desc_length"),
        "Within-developer FE": ("log_mu ~ agentic_int + consumer + agentic_int:consumer"
                                " + log_age + desc_length + C(username)"),
    }
    interaction = []
    for name, formula in specs.items():
        m = smf.ols(formula, data=df_it).fit(
            cov_type="cluster", cov_kwds={"groups": df_it["username"]})
        interaction.append({
            "spec": name,
            "coef": round(float(m.params["agentic_int:consumer"]), 4),
            "p": round(float(m.pvalues["agentic_int:consumer"]), 4),
        })

    out = {"developer_scale_gradient": gradient, "interaction_test": interaction}
    with open(BASE / "data" / "block11_interaction_gradient.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
