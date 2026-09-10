"""Block 27: within-developer 'similar-tool' matching.

Directly attacks the alternative explanation that agentic-ready and
non-agentic-ready tools are simply different tool types. Within each mixed
developer, we match every ready tool to its most similar non-ready tool
(1:1 nearest-neighbor, with replacement, no caliper) under two similarity
notions---description TF-IDF cosine similarity and a covariate distance
(category + price + age + description length)---then report the mean adoption
gap, covariate balance before/after matching, and a bootstrap 95% CI.

Outputs:
  - results/data/block27_similar_tool_matching.json
"""

import json
import math
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from block15_eligibility import DB_PATH, DETAIL_PATH, BASE, load_detail

warnings.filterwarnings("ignore")


def load() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("""
        SELECT a.actor_id, a.username, a.monthly_users, a.description,
               json_extract_string(a.categories, '$[0]') AS primary_category,
               EXTRACT(EPOCH FROM ((SELECT MAX(crawl_date)::TIMESTAMP FROM actor_day) - p.pricing_created_at))
                 / 86400.0 AS age_days,
               p.price_per_unit_usd
        FROM actor_day a
        JOIN pricing_day p ON a.actor_id = p.actor_id AND a.crawl_date = p.crawl_date
        WHERE a.crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
          AND a.pricing_model = 'PAY_PER_EVENT'
        ORDER BY a.actor_id
    """).df()
    con.close()

    df["primary_category"] = df["primary_category"].fillna("UNKNOWN").str.strip('"')
    df["description"] = df["description"].fillna("")
    df["log_age"] = np.log1p(df["age_days"].clip(lower=0))
    df["log_price"] = np.log1p(df["price_per_unit_usd"].fillna(0))
    df["desc_length"] = df["description"].str.len()
    df["log_mu"] = np.log1p(df["monthly_users"])
    return df


def desc_sim(ready: pd.DataFrame, non: pd.DataFrame) -> np.ndarray:
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english",
                            ngram_range=(1, 2))
    tfidf.fit(pd.concat([ready["description"], non["description"]]))
    return cosine_similarity(tfidf.transform(ready["description"]),
                             tfidf.transform(non["description"]))


def cov_sim(ready: pd.DataFrame, non: pd.DataFrame) -> np.ndarray:
    cols = ["log_price", "log_age", "desc_length"]
    r = ready[cols].values.astype(float)
    n = non[cols].values.astype(float)
    std = np.nanstd(np.vstack([r, n]), axis=0) + 1e-6
    dist = np.linalg.norm((r / std)[:, None, :] - (n / std)[None, :, :], axis=2)
    cat_bonus = (ready["primary_category"].values[:, None]
                 == non["primary_category"].values[None, :]).astype(float)
    return -(dist - 2.0 * cat_bonus)


def match_pairs(df: pd.DataFrame, sim_fn) -> list:
    """(ready_idx, non_idx, username) pairs; with replacement, no caliper."""
    pairs = []
    for username, g in df.groupby("username"):
        ready = g[g["eligible"] == 1]
        non = g[g["eligible"] == 0]
        if len(ready) == 0 or len(non) == 0:
            continue
        S = sim_fn(ready, non)
        best = S.argmax(axis=1)
        for i, j in enumerate(best):
            pairs.append((ready.index[i], non.index[j], username))
    return pairs


def smd(a, b):
    sp = np.sqrt((a.var() + b.var()) / 2.0)
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def analyze(df: pd.DataFrame, sim_fn) -> dict:
    pairs = match_pairs(df, sim_fn)
    ready_idx = [r for r, _, _ in pairs]
    non_idx = [n for _, n, _ in pairs]
    gap = df.loc[ready_idx, "log_mu"].values - df.loc[non_idx, "log_mu"].values

    # Developer-clustered bootstrap 95% CI of the mean gap. Matched pairs are
    # nested within developers, so we resample developers (with replacement) and
    # recompute the mean gap over all pairs of the sampled developers, rather
    # than resampling individual gaps as independent draws (which would ignore
    # the within-developer correlation and understate uncertainty).
    by_dev = {}
    for (r, n, u) in pairs:
        by_dev.setdefault(u, []).append((r, n))
    devs = list(by_dev.keys())
    rng = np.random.default_rng(42)
    B = 1000
    means = []
    for _ in range(B):
        sampled = rng.choice(devs, size=len(devs), replace=True)
        gaps = [df.loc[r, "log_mu"] - df.loc[n, "log_mu"]
                for u in sampled for (r, n) in by_dev[u]]
        means.append(float(np.mean(gaps)))
    lo, hi = np.percentile(means, [2.5, 97.5])

    # Covariate balance before (all ready vs all non-ready) and after (matched).
    cols = ["log_price", "log_age", "desc_length"]
    all_ready = df[df["eligible"] == 1]
    all_non = df[df["eligible"] == 0]
    balance = {}
    for c in cols:
        balance[c] = {
            "smd_before": round(smd(all_ready[c], all_non[c]), 3),
            "smd_after": round(smd(df.loc[ready_idx, c], df.loc[non_idx, c]), 3),
        }

    return {
        "n_matched_pairs": int(len(gap)),
        "n_non_ready_tools": int(all_non.shape[0]),
        "mean_gap": round(float(gap.mean()), 4),
        "median_gap": round(float(np.median(gap)), 4),
        "pct_gap": round((math.exp(gap.mean()) - 1) * 100, 1),
        "frac_positive": round(float((gap > 0).mean()), 3),
        "bootstrap_ci95": [round(float(lo), 4), round(float(hi), 4)],
        "balance": balance,
    }


def main() -> None:
    df = load()
    detail = load_detail()
    df = df.merge(detail, left_on="actor_id", right_on="actorId", how="inner")
    df["eligible"] = ((df["limited"] == 1) & (df["standby"] == 0)).astype(int)

    out = {
        "description_match": analyze(df, desc_sim),
        "covariate_match": analyze(df, cov_sim),
    }
    (BASE / "data" / "block27_similar_tool_matching.json").write_text(
        json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
