"""Block 28: Secondary outcome — runs_30d (usage intensity).

Re-estimates the within-developer fixed-effects association of agentic
readiness with log(1 + runs_30d_total), mirroring block15's core FE but with
the 30-day run count (rather than monthly users) as the outcome.

Outputs:
  - results/data/block28_runs_30d.json
"""

import json
import math
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "apify_panel.duckdb"
DETAIL_PATH = (Path(__file__).resolve().parent.parent / "data" / "raw"
               / "2026-08-27.actor_detail.jsonl")
BASE = Path(__file__).resolve().parent.parent / "results"

CTRLS = "log_age + desc_length + C(primary_category) + C(username)"


def load_ppe(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """PPE actors (latest crawl) with listing-time covariates + runs_30d."""
    df = con.execute("""
        SELECT a.actor_id, a.username, a.monthly_users, a.total_builds,
               a.runs_30d_total,
               dev.n_actors AS dev_portfolio_size,
               COALESCE(pe.n_events, 0) AS n_pricing_events,
               LENGTH(COALESCE(a.description, '')) AS desc_length,
               json_extract_string(a.categories, '$[0]') AS primary_category,
               EXTRACT(EPOCH FROM ((SELECT MAX(crawl_date)::TIMESTAMP FROM actor_day) - p.pricing_created_at))
                 / 86400.0 AS age_days
        FROM actor_day a
        JOIN pricing_day p ON a.actor_id = p.actor_id AND a.crawl_date = p.crawl_date
        LEFT JOIN (
            SELECT username, COUNT(*) AS n_actors
            FROM actor_day WHERE crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
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

    df["primary_category"] = df["primary_category"].fillna("UNKNOWN").str.strip('"')
    for c in ["n_pricing_events", "desc_length", "dev_portfolio_size"]:
        df[c] = df[c].fillna(0)
    df["log_builds"] = np.log1p(df["total_builds"].fillna(0))
    df["log_age"] = np.log1p(df["age_days"].fillna(df["age_days"].median()))
    df["log_runs"] = np.log1p(df["runs_30d_total"].fillna(0))
    return df


def load_detail() -> pd.DataFrame:
    recs = [json.loads(line) for line in open(DETAIL_PATH)]
    d = pd.DataFrame(recs)
    d = d[d["status"] == 200][["actorId", "actorPermissionLevel", "standbyEnabled"]]
    d["limited"] = (d["actorPermissionLevel"] == "LIMITED_PERMISSIONS").astype(int)
    d["standby"] = d["standbyEnabled"].fillna(False).astype(int)
    return d


def mixed_developers(df: pd.DataFrame, col: str) -> pd.DataFrame:
    mixed = df.groupby("username").apply(lambda g: g[col].nunique() == 2)
    out = df[df["username"].isin(mixed[mixed].index)].copy()
    out["w_dev"] = 1.0 / out.groupby("username")["username"].transform("count")
    return out


def fe_effect(df: pd.DataFrame, formula: str, weighted: bool) -> dict:
    mdf = mixed_developers(df, "eligible")
    if weighted:
        m = smf.wls(formula, data=mdf, weights=mdf["w_dev"]).fit(
            cov_type="cluster", cov_kwds={"groups": mdf["username"]})
    else:
        m = smf.ols(formula, data=mdf).fit(
            cov_type="cluster", cov_kwds={"groups": mdf["username"]})
    c = float(m.params["eligible"]); se = float(m.bse["eligible"]); p = float(m.pvalues["eligible"])
    return {
        "coef": round(c, 4), "se": round(se, 4), "p": p,
        "pct": round((math.exp(c) - 1) * 100, 1),
        "n_developers": int(mdf["username"].nunique()),
        "n_actors": int(len(mdf)),
    }


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = load_ppe(con)
    con.close()
    detail = load_detail()
    df = df.merge(detail, left_on="actor_id", right_on="actorId", how="inner")
    df["eligible"] = ((df["limited"] == 1) & (df["standby"] == 0)).astype(int)

    equal_dev = fe_effect(df, f"log_runs ~ eligible + {CTRLS}", weighted=True)
    actor_w = fe_effect(df, f"log_runs ~ eligible + {CTRLS}", weighted=False)

    out = {
        "outcome": "log_runs_30d_total",
        "n_ppe": int(len(df)),
        "n_eligible": int(df["eligible"].sum()),
        "fe_equal_dev": equal_dev,
        "fe_actor_weighted": actor_w,
    }
    (BASE / "data" / "block28_runs_30d.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
