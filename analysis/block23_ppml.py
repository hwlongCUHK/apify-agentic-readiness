"""Block 23: Poisson pseudo-MLE (PPML) with developer fixed effects.

The primary outcome is log(1 + monthly_users), so exp(beta)-1 refers to the
(1 + monthly_users) scale, not monthly users itself. As a robustness check that
estimates the effect on the count scale directly, we fit a Poisson pseudo-MLE
with developer fixed effects. exp(beta) is the incidence-rate ratio on monthly
users.

Outputs:
  - results/data/block23_ppml.json
"""

import json
import math
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "apify_panel.duckdb"
DETAIL_PATH = (Path(__file__).resolve().parent.parent / "data" / "raw"
               / "2026-08-27.actor_detail.jsonl")
BASE = Path(__file__).resolve().parent.parent / "results"


def load_ppe(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute("""
        SELECT a.actor_id, a.username, a.monthly_users, a.total_builds,
               LENGTH(COALESCE(a.description, '')) AS desc_length,
               json_extract_string(a.categories, '$[0]') AS primary_category,
               EXTRACT(EPOCH FROM ((SELECT MAX(crawl_date)::TIMESTAMP FROM actor_day) - p.pricing_created_at))
                 / 86400.0 AS age_days
        FROM actor_day a
        JOIN pricing_day p ON a.actor_id = p.actor_id AND a.crawl_date = p.crawl_date
        WHERE a.crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
          AND a.pricing_model = 'PAY_PER_EVENT'
        ORDER BY a.actor_id
    """).df()

    df["primary_category"] = df["primary_category"].fillna("UNKNOWN").str.strip('"')
    df["log_age"] = np.log1p(df["age_days"].fillna(df["age_days"].median()))
    return df


def load_detail() -> pd.DataFrame:
    recs = [json.loads(line) for line in open(DETAIL_PATH)]
    d = pd.DataFrame(recs)
    d = d[d["status"] == 200][["actorId", "actorPermissionLevel", "standbyEnabled"]]
    d["limited"] = (d["actorPermissionLevel"] == "LIMITED_PERMISSIONS").astype(int)
    d["standby"] = d["standbyEnabled"].fillna(False).astype(int)
    d["eligible"] = ((d["limited"] == 1) & (d["standby"] == 0)).astype(int)
    return d


def mixed_developers(df: pd.DataFrame, col: str) -> pd.DataFrame:
    mixed = df.groupby("username").apply(lambda g: g[col].nunique() == 2)
    out = df[df["username"].isin(mixed[mixed].index)].copy()
    out["w_dev"] = 1.0 / out.groupby("username")["username"].transform("count")
    return out


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = load_ppe(con)
    con.close()
    detail = load_detail()
    df = df.merge(detail, left_on="actor_id", right_on="actorId", how="inner")
    df["eligible"] = ((df["limited"] == 1) & (df["standby"] == 0)).astype(int)

    mdf = mixed_developers(df, "eligible")
    mdf["w_dev"] = 1.0 / mdf.groupby("username")["username"].transform("count")

    formula = ("monthly_users ~ eligible + log_age + desc_length "
               "+ C(primary_category) + C(username)")

    out = {}
    # Actor-weighted PPML-FE (each tool weighted equally).
    m = smf.glm(formula, data=mdf, family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": mdf["username"]})
    c = float(m.params["eligible"]); se = float(m.bse["eligible"])
    p = float(m.pvalues["eligible"]); irr = math.exp(c)
    out["ppml_fe_actor_weighted"] = {
        "coef": round(c, 4), "se": round(se, 4), "p": p,
        "irr": round(irr, 3), "pct": round((irr - 1) * 100, 1),
        "n_developers": int(mdf["username"].nunique()),
        "n_actors": int(len(mdf)),
    }

    # Equal-developer-weighted PPML (frequency weights proportional to 1/n_j).
    # Scale w_dev so the weights sum to the number of developers.
    w = mdf["w_dev"] * mdf["username"].nunique()
    m2 = smf.glm(formula, data=mdf, family=sm.families.Poisson(),
                 freq_weights=w).fit(
        cov_type="cluster", cov_kwds={"groups": mdf["username"]})
    c2 = float(m2.params["eligible"]); se2 = float(m2.bse["eligible"])
    p2 = float(m2.pvalues["eligible"]); irr2 = math.exp(c2)
    out["ppml_fe_equal_dev"] = {
        "coef": round(c2, 4), "se": round(se2, 4), "p": p2,
        "irr": round(irr2, 3), "pct": round((irr2 - 1) * 100, 1),
    }

    (BASE / "data" / "block23_ppml.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
