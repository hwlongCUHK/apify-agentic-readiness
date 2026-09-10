"""Block 16: robustness for the eligibility analysis.

Complements block15 (treatment = eligible = limited-permissions AND non-standby):

1. Whale sensitivity: re-run the equal-developer-weighted within-developer FE
   excluding whale developers, to confirm the eligibility effect is not driven
   by a small set of large developers.

2. Multi-hot category: re-estimate the per-category eligibility effect using
   "any-membership" assignment (an actor contributes to every category it is
   tagged with), so the category heterogeneity does not depend on the
   primary-category (first-tag) operationalization.

Outputs:
  - results/data/block16_eligibility_robustness.json
"""

import json
import math
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from block15_eligibility import (
    DB_PATH, DETAIL_PATH, BASE, CTRLS, load_ppe, load_detail, mixed_developers,
)

warnings.filterwarnings("ignore")


def fe_coef(mdf: pd.DataFrame, formula: str) -> dict:
    m = smf.wls(formula, data=mdf, weights=mdf["w_dev"]).fit(
        cov_type="cluster", cov_kwds={"groups": mdf["username"]})
    c = float(m.params["eligible"]); se = float(m.bse["eligible"]); p = float(m.pvalues["eligible"])
    return {"coef": round(c, 4), "se": round(se, 4), "p": p,
            "pct": round((math.exp(c) - 1) * 100, 1),
            "n_developers": int(mdf["username"].nunique()),
            "n_actors": int(len(mdf))}


def whale_sensitivity(df: pd.DataFrame) -> dict:
    """Equal-dev FE excluding whale developers (>=100 PPE actors)."""
    whales = set(df[df["dev_portfolio_size"] >= 100]["username"].unique())
    sub = df[~df["username"].isin(whales)]
    mdf = mixed_developers(sub, "eligible")
    return fe_coef(mdf, f"log_mu ~ eligible + {CTRLS}")


def multihot_category(df: pd.DataFrame, con: duckdb.DuckDBPyConnection) -> list:
    """Per-category eligible effect under multi-hot (any-membership) assignment."""
    # Re-load categories array (multi-hot) plus the eligibility columns.
    cats_df = con.execute("""
        SELECT a.actor_id,
               CAST(json_extract(a.categories, '$[*]') AS VARCHAR[]) AS cats_array
        FROM actor_day a
        WHERE a.crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
          AND a.pricing_model = 'PAY_PER_EVENT'
        ORDER BY a.actor_id
    """).df()
    def _parse_cats(arr):
        if arr is None or (isinstance(arr, float) and arr != arr) or str(arr) == "<NA>":
            return []
        return [c.strip('"') for c in arr]
    cats_df["cats_list"] = cats_df["cats_array"].apply(_parse_cats)

    base = df[["actor_id", "username", "eligible", "log_mu", "log_builds",
               "dev_portfolio_size", "n_pricing_events"]]
    all_cats = sorted({c for cats in cats_df["cats_list"] for c in cats})

    rows = []
    for cat in all_cats:
        members = set(cats_df[cats_df["cats_list"].apply(lambda x: cat in x)]["actor_id"])
        s = base[base["actor_id"].isin(members)]
        na = int(s["eligible"].sum()); nn = len(s) - na
        if len(s) < 200 or na < 30 or nn < 30:
            continue
        m = smf.ols(
            "log_mu ~ eligible + log_builds + dev_portfolio_size + n_pricing_events",
            data=s).fit(cov_type="cluster", cov_kwds={"groups": s["username"]})
        rows.append({
            "category": cat, "n_actors": int(len(s)),
            "coef": round(float(m.params["eligible"]), 4),
            "se": round(float(m.bse["eligible"]), 4),
            "p": float(m.pvalues["eligible"]),
        })
    rows.sort(key=lambda r: -r["coef"])
    return rows


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = load_ppe(con)
    detail = load_detail()
    con.close()
    df = df.merge(detail, left_on="actor_id", right_on="actorId", how="inner")
    df["eligible"] = ((df["limited"] == 1) & (df["standby"] == 0)).astype(int)

    con = duckdb.connect(str(DB_PATH), read_only=True)
    whales = whale_sensitivity(df)
    cats = multihot_category(df, con)
    con.close()
    bonf = 0.05 / len(cats) if cats else None

    out = {
        "whale_sensitivity": whales,
        "multihot_category": {
            "n_categories": len(cats),
            "bonferroni_alpha": round(bonf, 5) if bonf else None,
            "rows": cats,
        },
    }
    (BASE / "data" / "block16_eligibility_robustness.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
