"""Block 21: Ready x category interaction within the within-developer FE.

Addresses whether the category gradient survives developer fixed effects. Two
analyses:

  1. A global Ready x Category interaction in the mixed-developer FE model,
     with a Wald test for joint significance of the interaction terms.
  2. Per-category within-developer FE estimates, restricted to mixed
     developers within each category, so the gradient is estimated holding the
     developer constant (not merely with tool-level controls).

Outputs:
  - results/data/block21_interaction.json
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
    df = con.execute("""
        SELECT a.actor_id, a.username, a.monthly_users, a.total_builds,
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
    df["log_builds"] = np.log1p(df["total_builds"].fillna(0))
    df["log_age"] = np.log1p(df["age_days"].fillna(df["age_days"].median()))
    df["log_mu"] = np.log1p(df["monthly_users"])
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

    # --- 1. Global interaction test ---
    f = f"log_mu ~ eligible + eligible:C(primary_category) + {CTRLS}"
    m = smf.wls(f, data=mdf, weights=mdf["w_dev"]).fit(
        cov_type="cluster", cov_kwds={"groups": mdf["username"]})

    inter_terms = [t for t in m.params.index
                   if "eligible:C(primary_category)" in t]
    r_mat = np.zeros((len(inter_terms), len(m.params)))
    for i, t in enumerate(inter_terms):
        r_mat[i, list(m.params.index).index(t)] = 1
    wald = m.wald_test(r_mat, use_f=True)
    interaction_test = {
        "n_interaction_terms": len(inter_terms),
        "statistic": round(float(wald.statistic), 3),
        "df": int(wald.df_num),
        "p": float(wald.pvalue),
    }

    # --- 2. Per-category within-developer FE ---
    rows = []
    for cat in sorted(df["primary_category"].unique()):
        s = df[df["primary_category"] == cat]
        if len(s) < 200 or s["eligible"].nunique() != 2:
            continue
        na = int(s["eligible"].sum()); nn = len(s) - na
        if na < 30 or nn < 30:
            continue
        # restrict to developers with within-category variation
        sc = mixed_developers(s, "eligible")
        n_devs = int(sc["username"].nunique())
        if n_devs < 5:
            continue
        sc["w_dev"] = 1.0 / sc.groupby("username")["username"].transform("count")
        m2 = smf.wls(
            "log_mu ~ eligible + log_age + desc_length + C(username)",
            data=sc, weights=sc["w_dev"]).fit(
            cov_type="cluster", cov_kwds={"groups": sc["username"]})
        c = float(m2.params["eligible"]); se = float(m2.bse["eligible"])
        p = float(m2.pvalues["eligible"])
        rows.append({
            "category": cat, "n_actors": int(len(sc)), "n_developers": n_devs,
            "coef": round(c, 4), "se": round(se, 4), "p": p,
            "pct": round((math.exp(c) - 1) * 100, 1),
        })
    rows.sort(key=lambda r: -r["coef"])

    out = {
        "interaction_test": interaction_test,
        "per_category_fe": rows,
        "n_mixed_developers": int(mdf["username"].nunique()),
        "n_mixed_actors": int(len(mdf)),
    }
    (BASE / "data" / "block21_interaction.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
