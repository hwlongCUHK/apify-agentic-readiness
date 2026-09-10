"""Block 13: Overlap-weighted (reweighted) concentration.

Addresses the concern that the concentration gap between whitelisted and
non-whitelisted segments could reflect observable compositional differences
rather than whitelist status itself. Re-estimates Gini and Theil using overlap
weights that balance the two segments on listing-time covariates (age,
description, developer portfolio size, category).

Overlap weights (ATO): whitelisted -> 1 - p(x), non-whitelisted -> p(x),
where p(x) is the propensity score from a logistic regression.

Outputs to results/data/block13_reweighted_concentration.json.
"""

import json
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "apify_panel.duckdb"
BASE = Path(__file__).resolve().parent.parent / "results"


def weighted_gini(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    order = np.argsort(x)
    x, w = x[order], w[order]
    cum_w = np.cumsum(w)
    cum_wx = np.cumsum(w * x)
    W, S = cum_w[-1], cum_wx[-1]
    if W <= 0 or S <= 0:
        return 0.0
    p = np.concatenate([[0.0], cum_w / W])
    L = np.concatenate([[0.0], cum_wx / S])
    area = np.sum((p[1:] - p[:-1]) * (L[1:] + L[:-1]) / 2.0)
    return float(1.0 - 2.0 * area)


def weighted_theil(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    W = w.sum()
    S = (w * x).sum()
    if W <= 0 or S <= 0:
        return 0.0
    mu = S / W
    mask = x > 0
    return float(np.sum(w[mask] * (x[mask] / mu) * np.log(x[mask] / mu)) / W)


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
    df["dev_portfolio_size"] = df["dev_portfolio_size"].fillna(1)
    df["log_age"] = np.log1p(df["age_days"].fillna(df["age_days"].median()))
    return df


def main():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = prepare(con)

    # Propensity score with listing-time covariates
    num = df[["log_age", "desc_length", "dev_portfolio_size"]].values
    cat = pd.get_dummies(df["primary_category"], prefix="cat").astype(float).values
    X = np.hstack([num, cat])
    y = df["agentic_int"].values
    lr = LogisticRegression(solver="liblinear", max_iter=10000, C=1.0, random_state=42)
    lr.fit(X, y)
    p = lr.predict_proba(X)[:, 1]

    # Overlap (ATO) weights
    w = np.where(df["agentic_int"] == 1, 1.0 - p, p)

    out = {}
    for label, mask in [("whitelisted", df["agentic_int"] == 1),
                        ("non_whitelisted", df["agentic_int"] == 0)]:
        x = df.loc[mask, "monthly_users"].values
        ww = w[mask]
        out[label] = {
            "gini": round(weighted_gini(x, ww), 4),
            "theil": round(weighted_theil(x, ww), 4),
        }

    gini_gap = out["whitelisted"]["gini"] - out["non_whitelisted"]["gini"]
    theil_gap = out["whitelisted"]["theil"] - out["non_whitelisted"]["theil"]
    out["gini_gap"] = round(gini_gap, 4)
    out["theil_gap"] = round(theil_gap, 4)

    with open(BASE / "data" / "block13_reweighted_concentration.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
