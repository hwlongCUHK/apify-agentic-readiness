"""Block 12: Extensive/intensive margin decomposition and runs gap.

Computes four within-developer, equal-developer-weighted estimates that feed
the RQ2 "Extensive versus intensive margin" subsection and the RQ1
"runs-versus-users contrast":
  (1) extensive margin  -- LPM of any-adoption indicator
  (2) intensive margin  -- OLS of log monthly users, conditional on adoption>0
  (3) top-decile margin -- LPM of a top-decile indicator
  (4) runs gap          -- OLS of log 30-day runs

All use listing-time controls (age, description, category), developer fixed
effects, and equal-developer weighting (WLS with weight 1/N_j).

Outputs to results/data/block12_margins.json.
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

FE_FORMULA = ("agentic_int + log_age + desc_length + C(primary_category) + C(username)")


def prepare(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute("""
        SELECT a.username,
               a.is_agentic_payments_whitelisted AS agentic,
               a.monthly_users,
               COALESCE(a.runs_30d_total, 0) AS runs_30d_total,
               LENGTH(COALESCE(a.description, '')) AS desc_length,
               json_extract_string(a.categories, '$[0]') AS primary_category,
               EXTRACT(EPOCH FROM ((SELECT MAX(crawl_date)::TIMESTAMP FROM actor_day) - p.pricing_created_at)) / 86400.0 AS age_days
        FROM actor_day a
        JOIN pricing_day p ON a.actor_id = p.actor_id AND a.crawl_date = p.crawl_date
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
    df["log_runs"] = np.log1p(df["runs_30d_total"])
    df["any_adoption"] = (df["monthly_users"] > 0).astype(int)
    return df


def fit_wls(df, outcome):
    # Equal-developer weight recomputed on the *passed* dataframe, so the
    # intensive margin (a subsample) weights each developer by its count there.
    w = 1.0 / df.groupby("username")["username"].transform("count")
    m = smf.wls(f"{outcome} ~ {FE_FORMULA}", data=df, weights=w).fit(
        cov_type="cluster", cov_kwds={"groups": df["username"]})
    return {
        "coef": round(float(m.params["agentic_int"]), 4),
        "se": round(float(m.bse["agentic_int"]), 4),
        "p": float(m.pvalues["agentic_int"]),
    }


def main():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = prepare(con)

    # Mixed developers (both agentic and non-agentic PPE actors)
    mixed = df.groupby("username")["agentic_int"].nunique()
    mixed_users = mixed[mixed == 2].index.tolist()
    mdf = df[df["username"].isin(mixed_users)].copy()

    # Top-decile indicator: top 10% of tools with at least one user, over the
    # full PPE sample (p90 of the nonzero monthly-users distribution = 9).
    p90 = df[df["monthly_users"] > 0]["monthly_users"].quantile(0.90)
    mdf["top_decile"] = (mdf["monthly_users"] >= p90).astype(int)

    out = {
        "extensive_margin": fit_wls(mdf, "any_adoption"),
        "intensive_margin": fit_wls(mdf[mdf["monthly_users"] > 0], "log_mu"),
        "top_decile": fit_wls(mdf, "top_decile"),
        "runs_gap": fit_wls(mdf, "log_runs"),
    }

    with open(BASE / "data" / "block12_margins.json", "w") as f:
        json.dump(out, f, indent=2)
    for k, v in out.items():
        print(f"{k}: coef={v['coef']} se={v['se']} p={v['p']:.4g}")


if __name__ == "__main__":
    main()
