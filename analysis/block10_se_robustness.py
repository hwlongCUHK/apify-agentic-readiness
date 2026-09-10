"""Block 10: Inference robustness — clustered SEs, equal-developer weighting,
exclude-whales, leave-one-large-developer-out.

Addresses: (1) all actor-level regressions should cluster SE by developer given
extreme developer concentration; (2) within-developer FE is actor-weighted, so
whale developers dominate; test equal-developer weighting and whale-robustness.
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
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "apify_panel.duckdb"
BASE = Path(__file__).resolve().parent.parent / "results"


def prepare(con):
    df = con.execute("""
        SELECT a.actor_id, a.username,
               a.is_agentic_payments_whitelisted AS agentic,
               a.monthly_users, COALESCE(a.runs_30d_total,0) AS runs_30d_total,
               a.total_builds, a.bookmarks, a.rating, a.review_count,
               a.last_run_started_at,
               dev.n_actors AS dev_portfolio_size,
               COALESCE(pe.n_events,0) AS n_pricing_events,
               LENGTH(COALESCE(a.description,'')) AS desc_length,
               json_extract_string(a.categories,'$[0]') AS primary_category,
               EXTRACT(EPOCH FROM ((SELECT MAX(crawl_date)::TIMESTAMP FROM actor_day) - p.pricing_created_at))/86400.0 AS age_days
        FROM actor_day a
        JOIN pricing_day p ON a.actor_id=p.actor_id AND a.crawl_date=p.crawl_date
        LEFT JOIN (SELECT username, COUNT(*) n_actors FROM actor_day WHERE crawl_date=(SELECT MAX(crawl_date) FROM actor_day) GROUP BY username) dev ON a.username=dev.username
        LEFT JOIN (SELECT actor_id, COUNT(*) n_events FROM pricing_event_detail WHERE crawl_date=(SELECT MAX(crawl_date) FROM actor_day) GROUP BY actor_id) pe ON a.actor_id=pe.actor_id
        WHERE a.crawl_date=(SELECT MAX(crawl_date) FROM actor_day) AND a.pricing_model='PAY_PER_EVENT'
        ORDER BY a.actor_id
    """).df()
    con.close()

    df["agentic_int"] = df["agentic"].astype(int)
    df["primary_category"] = df["primary_category"].fillna("UNKNOWN").str.strip('"')
    for c in ["n_pricing_events","desc_length","dev_portfolio_size"]:
        df[c] = df[c].fillna(0)
    df["log_builds"] = np.log1p(df["total_builds"].fillna(0))
    df["log_bookmarks"] = np.log1p(df["bookmarks"].fillna(0))
    df["log_reviews"] = np.log1p(df["review_count"].fillna(0))
    df["rating"] = df["rating"].fillna(0)
    df["log_age"] = np.log1p(df["age_days"].fillna(df["age_days"].median()))
    df["days_since_run"] = pd.to_numeric(
        (pd.Timestamp(df['crawl_date_ref'].iloc[0] if 'crawl_date_ref' in df else '2026-09-04') - df["last_run_started_at"]).dt.total_seconds()/86400,
        errors='coerce').fillna(30)
    df["log_days_since_run"] = np.log1p(df["days_since_run"].clip(lower=0))
    df["log_mu"] = np.log1p(df["monthly_users"])
    df["log_runs"] = np.log1p(df["runs_30d_total"])
    return df


def fmt(coef, se, p):
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    pct = (math.exp(coef)-1)*100
    return f"{coef:.4f}{stars} ({se:.4f}), exp-1={pct:.1f}%"


def main():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = prepare(con)

    logger.info("=" * 70)
    logger.info("1. HC1 vs developer-clustered SEs for OLS models M1-M4")
    logger.info("=" * 70)

    specs = {
        "M1": "log_mu ~ agentic_int",
        "M2": "log_mu ~ agentic_int + log_builds + dev_portfolio_size + desc_length",
        "M3": "log_mu ~ agentic_int + log_builds + dev_portfolio_size + desc_length + n_pricing_events + C(primary_category)",
        "M4": "log_runs ~ agentic_int + log_builds + dev_portfolio_size + desc_length + n_pricing_events + C(primary_category)",
    }

    clustered = {}
    for name, formula in specs.items():
        m_hc1 = smf.ols(formula, data=df).fit(cov_type="HC1")
        m_cl = smf.ols(formula, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["username"]})
        c_hc1, se_hc1, p_hc1 = m_hc1.params["agentic_int"], m_hc1.bse["agentic_int"], m_hc1.pvalues["agentic_int"]
        c_cl, se_cl, p_cl = m_cl.params["agentic_int"], m_cl.bse["agentic_int"], m_cl.pvalues["agentic_int"]
        logger.info("%s: HC1 %s", name, fmt(c_hc1, se_hc1, p_hc1))
        logger.info("      CL  %s", fmt(c_cl, se_cl, p_cl))
        clustered[name] = {"coef": round(float(c_cl),4), "se": round(float(se_cl),4),
                           "p": float(p_cl), "se_hc1": round(float(se_hc1),4)}

    logger.info("")
    logger.info("=" * 70)
    logger.info("2. Within-developer FE: actor-weighted vs equal-developer-weighted")
    logger.info("=" * 70)

    # Mixed developers
    mixed = df.groupby("username").apply(lambda g: g["agentic_int"].nunique() == 2)
    mixed_users = mixed[mixed].index.tolist()
    mdf = df[df["username"].isin(mixed_users)].copy()

    # Listing-time controls (primary) vs contemporaneous controls (sensitivity)
    fe_listing = "log_mu ~ agentic_int + log_age + desc_length + C(primary_category) + C(username)"
    fe_contemp = ("log_mu ~ agentic_int + log_age + desc_length + C(primary_category)"
                  " + log_bookmarks + log_reviews + rating + log_days_since_run + C(username)")

    # Actor-weighted (standard FE, clustered at developer), listing-time
    m_aw = smf.ols(fe_listing, data=mdf).fit(
        cov_type="cluster", cov_kwds={"groups": mdf["username"]})
    c_aw, se_aw, p_aw = m_aw.params["agentic_int"], m_aw.bse["agentic_int"], m_aw.pvalues["agentic_int"]
    logger.info("Actor-weighted FE (listing-time): %s", fmt(c_aw, se_aw, p_aw))
    logger.info("  N_actors=%d, N_devs=%d", len(mdf), mdf["username"].nunique())

    # Actor-weighted, contemporaneous (sensitivity)
    m_aw_ct = smf.ols(fe_contemp, data=mdf).fit(
        cov_type="cluster", cov_kwds={"groups": mdf["username"]})
    c_aw_ct, se_aw_ct, p_aw_ct = m_aw_ct.params["agentic_int"], m_aw_ct.bse["agentic_int"], m_aw_ct.pvalues["agentic_int"]
    logger.info("Actor-weighted FE (contemporaneous): %s", fmt(c_aw_ct, se_aw_ct, p_aw_ct))

    # Equal-developer-weighted: WLS with weight 1/N_j
    mdf = mdf.copy()
    mdf["n_actors_dev"] = mdf.groupby("username")["username"].transform("count")
    mdf["w_dev"] = 1.0 / mdf["n_actors_dev"]

    # Equal-dev, listing-time (primary)
    m_ew_lt = smf.wls(fe_listing, data=mdf, weights=mdf["w_dev"]).fit(
        cov_type="cluster", cov_kwds={"groups": mdf["username"]})
    c_ew_lt, se_ew_lt, p_ew_lt = m_ew_lt.params["agentic_int"], m_ew_lt.bse["agentic_int"], m_ew_lt.pvalues["agentic_int"]
    logger.info("Equal-dev FE (listing-time):     %s", fmt(c_ew_lt, se_ew_lt, p_ew_lt))

    # Equal-dev, contemporaneous (sensitivity)
    m_ew_ct = smf.wls(fe_contemp, data=mdf, weights=mdf["w_dev"]).fit(
        cov_type="cluster", cov_kwds={"groups": mdf["username"]})
    c_ew_ct, se_ew_ct, p_ew_ct = m_ew_ct.params["agentic_int"], m_ew_ct.bse["agentic_int"], m_ew_ct.pvalues["agentic_int"]
    logger.info("Equal-dev FE (contemporaneous): %s", fmt(c_ew_ct, se_ew_ct, p_ew_ct))

    # Developer-level collapse (intuition): mean within-dev gap
    dev_gap = mdf.groupby("username").apply(
        lambda g: g.loc[g["agentic_int"]==1,"log_mu"].mean() - g.loc[g["agentic_int"]==0,"log_mu"].mean()
    )
    logger.info("Developer-collapse mean gap: %.4f (n=%d devs), median %.4f",
                dev_gap.mean(), len(dev_gap), dev_gap.median())

    logger.info("")
    logger.info("=" * 70)
    logger.info("3. Exclude whale developers (>=100 actors)")
    logger.info("=" * 70)

    whale_users = mdf[mdf["n_actors_dev"] >= 100]["username"].unique()
    n_whales = len(whale_users)
    n_whale_actors = int(mdf[mdf["username"].isin(whale_users)].shape[0])
    logger.info("Whale mixed developers: %d (contributing %d actors)", n_whales, n_whale_actors)

    mdf_nowhale = mdf[~mdf["username"].isin(whale_users)]
    m_nw = smf.ols(fe_listing, data=mdf_nowhale).fit(
        cov_type="cluster", cov_kwds={"groups": mdf_nowhale["username"]})
    c_nw, se_nw, p_nw = m_nw.params["agentic_int"], m_nw.bse["agentic_int"], m_nw.pvalues["agentic_int"]
    logger.info("Exclude-whales FE: %s", fmt(c_nw, se_nw, p_nw))
    logger.info("  N_actors=%d, N_devs=%d", len(mdf_nowhale), mdf_nowhale["username"].nunique())

    logger.info("")
    logger.info("=" * 70)
    logger.info("4. Leave-one-large-developer-out (top 20 by actor count)")
    logger.info("=" * 70)

    top_devs = mdf.groupby("username")["username"].count().sort_values(ascending=False).head(20).index.tolist()
    loo_coefs = []
    for dev in top_devs:
        sub = mdf[mdf["username"] != dev]
        try:
            m = smf.ols(fe_listing, data=sub).fit(
                cov_type="cluster", cov_kwds={"groups": sub["username"]})
            loo_coefs.append((dev, float(m.params["agentic_int"]), float(m.bse["agentic_int"])))
        except Exception as e:
            logger.info("  (skip %s: %s)", dev, e)
    coefs = [c for _, c, _ in loo_coefs]
    logger.info("Leave-one-out (top 20 devs): min=%.4f, max=%.4f, mean=%.4f",
                min(coefs), max(coefs), np.mean(coefs))
    for dev, c, se in loo_coefs[:5]:
        logger.info("  drop %-30s -> %.4f (%.4f)", dev, c, se)

    results = {
        "clustered_se": clustered,
        "fe_actor_weighted_listing": {"coef": round(float(c_aw),4), "se": round(float(se_aw),4), "p": float(p_aw)},
        "fe_actor_weighted_contemp": {"coef": round(float(c_aw_ct),4), "se": round(float(se_aw_ct),4), "p": float(p_aw_ct)},
        "fe_equal_developer_weighted_listing": {"coef": round(float(c_ew_lt),4), "se": round(float(se_ew_lt),4), "p": float(p_ew_lt)},
        "fe_equal_developer_weighted_contemp": {"coef": round(float(c_ew_ct),4), "se": round(float(se_ew_ct),4), "p": float(p_ew_ct)},
        "developer_collapse": {"mean_gap": round(float(dev_gap.mean()),4), "median_gap": round(float(dev_gap.median()),4), "n_devs": len(dev_gap)},
        "exclude_whales": {"n_whales": int(n_whales), "n_whale_actors": n_whale_actors,
                           "coef": round(float(c_nw),4), "se": round(float(se_nw),4), "p": float(p_nw),
                           "n_actors": len(mdf_nowhale), "n_devs": mdf_nowhale["username"].nunique()},
        "leave_one_out_top20": {"min": round(float(min(coefs)),4), "max": round(float(max(coefs)),4),
                                "mean": round(float(np.mean(coefs)),4), "details": loo_coefs[:5]},
    }
    with open(BASE / "data" / "block10_se_robustness.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("")
    logger.info("Saved block10_se_robustness.json")


if __name__ == "__main__":
    main()
