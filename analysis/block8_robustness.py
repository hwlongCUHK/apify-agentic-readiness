"""Block 8: Robustness — Additional Controls + Concentration Decomposition.

Addresses reviewer concerns:
1. Add rating, bookmarks, review_count, last_run recency as controls
2. Decompose concentration paradox to rule out compositional artifact

Outputs:
  - results/tables/T14_extended_regression.csv
  - results/tables/T15_concentration_decomposition.csv
  - results/data/block8_robustness.json
"""

import json
import logging
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "apify_panel.duckdb"
BASE = Path(__file__).resolve().parent.parent / "results"


def gini(values: np.ndarray) -> float:
    v = np.sort(values.astype(float))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * v) - (n + 1) * np.sum(v)) / (n * np.sum(v)))


def prepare_extended_data(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """PPE dataset with ALL available controls."""
    df = con.execute("""
        SELECT
            a.actor_id,
            a.username,
            a.is_agentic_payments_whitelisted AS agentic,
            a.monthly_users,
            COALESCE(a.runs_30d_total, 0) AS runs_30d_total,
            a.total_builds,
            a.bookmarks,
            a.rating,
            a.review_count,
            a.last_run_started_at,
            dev.n_actors AS dev_portfolio_size,
            COALESCE(pe.n_events, 0) AS n_pricing_events,
            LENGTH(COALESCE(a.description, '')) AS desc_length,
            json_extract_string(a.categories, '$[0]') AS primary_category,
            -- Derived
            LN(1 + a.monthly_users) AS log_mu,
            LN(1 + COALESCE(a.runs_30d_total, 0)) AS log_runs,
            LN(1 + a.total_builds) AS log_builds,
            LN(1 + a.bookmarks) AS log_bookmarks,
            LN(1 + COALESCE(a.review_count, 0)) AS log_reviews,
            -- Recency: days since last run (proxy for maintenance/activity)
            CASE WHEN a.last_run_started_at IS NOT NULL
                 THEN EXTRACT(EPOCH FROM ((SELECT MAX(crawl_date)::TIMESTAMP FROM actor_day) - a.last_run_started_at)) / 86400.0
                 ELSE NULL END AS days_since_last_run,
            -- Has rating flag
            CASE WHEN a.rating IS NOT NULL AND a.rating > 0 THEN 1 ELSE 0 END AS has_rating
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
    df["desc_length"] = df["desc_length"].fillna(0)
    df["log_builds"] = df["log_builds"].fillna(0)
    df["log_bookmarks"] = df["log_bookmarks"].fillna(0)
    df["log_reviews"] = df["log_reviews"].fillna(0)
    df["dev_portfolio_size"] = df["dev_portfolio_size"].fillna(1)
    df["rating"] = df["rating"].fillna(0)
    df["days_since_last_run"] = df["days_since_last_run"].fillna(
        df["days_since_last_run"].median() if df["days_since_last_run"].notna().any() else 30
    )
    df["log_days_since_run"] = np.log1p(df["days_since_last_run"].clip(lower=0))

    logger.info("Extended dataset: %d PPE actors, %d with rating>0, %d with bookmarks>0",
                len(df), (df["rating"] > 0).sum(), (df["bookmarks"] > 0).sum())
    return df


def extended_regressions(df: pd.DataFrame) -> dict:
    """Run OLS with progressively more controls."""
    results = {}

    specs = {
        "M3_original": "log_mu ~ agentic_int + log_builds + dev_portfolio_size + desc_length"
                        " + n_pricing_events + C(primary_category)",
        "M3a_plus_quality": "log_mu ~ agentic_int + log_builds + dev_portfolio_size + desc_length"
                            " + n_pricing_events + log_bookmarks + log_reviews + has_rating"
                            " + C(primary_category)",
        "M3b_plus_recency": "log_mu ~ agentic_int + log_builds + dev_portfolio_size + desc_length"
                            " + n_pricing_events + log_bookmarks + log_reviews + has_rating"
                            " + log_days_since_run + C(primary_category)",
        "M3c_kitchen_sink": "log_mu ~ agentic_int + log_builds + dev_portfolio_size + desc_length"
                            " + n_pricing_events + log_bookmarks + log_reviews + has_rating"
                            " + log_days_since_run + rating + C(primary_category)",
    }

    for name, formula in specs.items():
        m = smf.ols(formula, data=df).fit(cov_type="HC1")
        coef = m.params.get("agentic_int", np.nan)
        se = m.bse.get("agentic_int", np.nan)
        p = m.pvalues.get("agentic_int", np.nan)
        ci = m.conf_int().loc["agentic_int"].values

        results[name] = {
            "coef": round(float(coef), 4),
            "se": round(float(se), 4),
            "p": float(p),
            "ci_95": [round(float(ci[0]), 4), round(float(ci[1]), 4)],
            "r2": round(float(m.rsquared), 4),
            "r2_adj": round(float(m.rsquared_adj), 4),
            "n": int(m.nobs),
        }
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        logger.info("  %s: agentic=%.4f%s (SE=%.4f), R2=%.4f, N=%d",
                     name, coef, stars, se, m.rsquared, m.nobs)

    # Also run developer FE + category FE with extended controls
    mixed_devs = df.groupby("username").apply(
        lambda g: g["agentic_int"].nunique() == 2
    )
    mixed_usernames = mixed_devs[mixed_devs].index.tolist()
    mixed_df = df[df["username"].isin(mixed_usernames)].copy()

    if len(mixed_df) > 100:
        m_fe_ext = smf.ols(
            "log_mu ~ agentic_int + log_builds + n_pricing_events"
            " + log_bookmarks + log_reviews + log_days_since_run"
            " + C(primary_category) + C(username)",
            data=mixed_df,
        ).fit(cov_type="HC1")

        coef = m_fe_ext.params.get("agentic_int", np.nan)
        se = m_fe_ext.bse.get("agentic_int", np.nan)
        p = m_fe_ext.pvalues.get("agentic_int", np.nan)
        ci = m_fe_ext.conf_int().loc["agentic_int"].values

        results["M5_fe_extended"] = {
            "coef": round(float(coef), 4),
            "se": round(float(se), 4),
            "p": float(p),
            "ci_95": [round(float(ci[0]), 4), round(float(ci[1]), 4)],
            "r2": round(float(m_fe_ext.rsquared), 4),
            "n_actors": len(mixed_df),
            "n_devs": mixed_df["username"].nunique(),
        }
        logger.info("  M5_fe_extended: agentic=%.4f (SE=%.4f, p=%.2e), N=%d devs=%d",
                     coef, se, p, len(mixed_df), mixed_df["username"].nunique())

    # Save table
    rows = []
    for name, r in results.items():
        stars = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else ""
        rows.append({
            "Model": name,
            "Agentic Coef": f"{r['coef']:.4f}{stars}",
            "SE": f"({r['se']:.4f})",
            "95% CI": f"[{r['ci_95'][0]:.4f}, {r['ci_95'][1]:.4f}]",
            "R-squared": f"{r['r2']:.4f}",
            "N": r.get("n", r.get("n_actors", "")),
        })
    pd.DataFrame(rows).to_csv(BASE / "tables" / "T14_extended_regression.csv", index=False)
    logger.info("T14 saved.")

    return results


def concentration_decomposition(df: pd.DataFrame) -> dict:
    """Decompose concentration to test whether the paradox is a compositional artifact."""
    results = {}

    # 1. Within-category Gini by agentic status (paired comparison)
    cat_gini_pairs = []
    for cat in df["primary_category"].unique():
        sub = df[df["primary_category"] == cat]
        ag = sub[sub["agentic"]]["monthly_users"].values
        na = sub[~sub["agentic"]]["monthly_users"].values
        if len(ag) >= 30 and len(na) >= 30:
            cat_gini_pairs.append({
                "category": cat,
                "n_agentic": len(ag),
                "n_nonagentic": len(na),
                "gini_agentic": round(gini(ag), 4),
                "gini_nonagentic": round(gini(na), 4),
                "gini_diff": round(gini(ag) - gini(na), 4),
                "pct_agentic": round(len(ag) / (len(ag) + len(na)) * 100, 1),
                "mean_mu_agentic": round(ag.mean(), 2),
                "mean_mu_nonagentic": round(na.mean(), 2),
            })

    pairs_df = pd.DataFrame(cat_gini_pairs)
    logger.info("Within-category Gini pairs: %d categories", len(pairs_df))

    # How many categories have agentic Gini > non-agentic Gini?
    n_ag_more_concentrated = (pairs_df["gini_diff"] > 0).sum()
    n_na_more_concentrated = (pairs_df["gini_diff"] < 0).sum()
    logger.info("  Agentic more concentrated in %d/%d categories",
                n_ag_more_concentrated, len(pairs_df))

    results["within_category_gini_pairs"] = cat_gini_pairs
    results["agentic_more_concentrated_count"] = int(n_ag_more_concentrated)
    results["nonagentic_more_concentrated_count"] = int(n_na_more_concentrated)

    # 2. Size-weighted Spearman correlation
    cat_conc = []
    for cat in df["primary_category"].unique():
        sub = df[df["primary_category"] == cat]
        if len(sub) < 50:
            continue
        total_mu = sub["monthly_users"].sum()
        if total_mu == 0:
            continue
        cat_conc.append({
            "category": cat,
            "n": len(sub),
            "pct_agentic": round(sub["agentic"].mean() * 100, 1),
            "gini": round(gini(sub["monthly_users"].values), 4),
        })

    conc_df = pd.DataFrame(cat_conc)

    # Unweighted (original)
    rho_unw, p_unw = spearmanr(conc_df["pct_agentic"], conc_df["gini"])
    logger.info("  Unweighted Spearman: rho=%.4f, p=%.4f, n=%d", rho_unw, p_unw, len(conc_df))

    # Exclude tiny categories (< 500)
    big = conc_df[conc_df["n"] >= 500]
    rho_big, p_big = spearmanr(big["pct_agentic"], big["gini"])
    logger.info("  Excluding <500: rho=%.4f, p=%.4f, n=%d", rho_big, p_big, len(big))

    # Exclude extreme categories (top/bottom 2 by agentic %)
    mid = conc_df.sort_values("pct_agentic").iloc[2:-2]
    if len(mid) >= 5:
        rho_mid, p_mid = spearmanr(mid["pct_agentic"], mid["gini"])
        logger.info("  Excluding extremes: rho=%.4f, p=%.4f, n=%d", rho_mid, p_mid, len(mid))
    else:
        rho_mid, p_mid = np.nan, np.nan

    results["spearman_robustness"] = {
        "unweighted": {"rho": round(float(rho_unw), 4), "p": float(p_unw), "n": len(conc_df)},
        "excluding_small": {"rho": round(float(rho_big), 4), "p": float(p_big), "n": len(big)},
        "excluding_extremes": {"rho": round(float(rho_mid), 4) if not np.isnan(rho_mid) else None,
                               "p": float(p_mid) if not np.isnan(p_mid) else None,
                               "n": len(mid)},
    }

    # 3. Theil index decomposition: between-category vs within-category
    # Theil T = between + within
    total_mu = df["monthly_users"].sum()
    N = len(df)
    overall_mean = df["monthly_users"].mean()

    between = 0
    within = 0
    decomp_rows = []

    for cat in df["primary_category"].unique():
        sub = df[df["primary_category"] == cat]
        if len(sub) == 0:
            continue
        n_k = len(sub)
        mu_k = sub["monthly_users"].mean()
        s_k = sub["monthly_users"].sum()

        # Between component
        if overall_mean > 0 and mu_k > 0:
            between += (s_k / total_mu) * np.log(mu_k / overall_mean)

        # Within component (Theil T within category k)
        theil_k = 0
        if mu_k > 0:
            for val in sub["monthly_users"].values:
                if val > 0:
                    theil_k += (val / s_k) * np.log(val / mu_k) if s_k > 0 else 0
            within += (s_k / total_mu) * theil_k

        decomp_rows.append({
            "category": cat,
            "n": n_k,
            "mean_mu": round(mu_k, 2),
            "share_total_users": round(s_k / total_mu * 100, 2) if total_mu > 0 else 0,
            "within_theil": round(theil_k, 4),
        })

    total_theil = between + within
    results["theil_decomposition"] = {
        "total": round(float(total_theil), 4),
        "between_categories": round(float(between), 4),
        "within_categories": round(float(within), 4),
        "between_share_pct": round(float(between / total_theil * 100), 1) if total_theil > 0 else 0,
        "within_share_pct": round(float(within / total_theil * 100), 1) if total_theil > 0 else 0,
    }
    logger.info("  Theil decomposition: total=%.4f, between=%.4f (%.1f%%), within=%.4f (%.1f%%)",
                total_theil, between,
                between / total_theil * 100 if total_theil > 0 else 0,
                within,
                within / total_theil * 100 if total_theil > 0 else 0)

    # Do the same for agentic and non-agentic separately
    for label, subset in [("agentic", df[df["agentic"]]), ("non_agentic", df[~df["agentic"]])]:
        t_mu = subset["monthly_users"].sum()
        o_mean = subset["monthly_users"].mean()
        b, w = 0, 0
        for cat in subset["primary_category"].unique():
            s = subset[subset["primary_category"] == cat]
            if len(s) == 0:
                continue
            mu_k = s["monthly_users"].mean()
            s_k = s["monthly_users"].sum()
            if o_mean > 0 and mu_k > 0 and t_mu > 0:
                b += (s_k / t_mu) * np.log(mu_k / o_mean)
            if mu_k > 0 and s_k > 0:
                for val in s["monthly_users"].values:
                    if val > 0:
                        w += (s_k / t_mu) * (val / s_k) * np.log(val / mu_k)
        t = b + w
        results[f"theil_{label}"] = {
            "total": round(float(t), 4),
            "between": round(float(b), 4),
            "within": round(float(w), 4),
            "between_pct": round(float(b / t * 100), 1) if t > 0 else 0,
        }
        logger.info("  Theil %s: total=%.4f, between=%.4f (%.1f%%), within=%.4f",
                     label, t, b, b / t * 100 if t > 0 else 0, w)

    # Save decomposition table
    pd.DataFrame(decomp_rows).to_csv(BASE / "tables" / "T15_concentration_decomposition.csv", index=False)

    return results


def main() -> None:
    for subdir in ["tables", "data"]:
        (BASE / subdir).mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        df = prepare_extended_data(con)

        logger.info("=" * 60)
        logger.info("Part 1: Extended Regressions")
        logger.info("=" * 60)
        reg_results = extended_regressions(df)

        logger.info("=" * 60)
        logger.info("Part 2: Concentration Decomposition")
        logger.info("=" * 60)
        conc_results = concentration_decomposition(df)

        all_results = {
            "extended_regressions": reg_results,
            "concentration_decomposition": conc_results,
        }
        with open(BASE / "data" / "block8_robustness.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        logger.info("=" * 60)
        logger.info("Block 8 complete.")
    finally:
        con.close()


if __name__ == "__main__":
    main()
