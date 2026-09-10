"""Block 4: Adjusted Association Analysis (RQ2, Steps 2-3).

Outputs:
  - results/tables/T6_regression_table.csv
  - results/tables/T7_psm_balance.csv
  - results/data/block4_regression_summary.json
  - results/data/block4_psm_results.json
  - results/figures/F4_pscore_distributions.pdf
"""

import json
import logging
import warnings
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "apify_panel.duckdb"
BASE = Path(__file__).resolve().parent.parent / "results"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
})


def prepare_data(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Create analysis dataset: PPE actors with covariates."""
    df = con.execute("""
        SELECT
            a.actor_id,
            a.username,
            a.is_agentic_payments_whitelisted AS agentic,
            a.monthly_users,
            COALESCE(a.runs_30d_total, 0) AS runs_30d_total,
            a.bookmarks,
            a.rating,
            a.total_builds,
            a.review_count,
            dev.n_actors AS dev_portfolio_size,
            COALESCE(pe.n_events, 0) AS n_pricing_events,
            pe.median_event_price,
            pe.cv_event_price,
            json_extract_string(a.categories, '$[0]') AS primary_category,
            LENGTH(COALESCE(a.description, '')) AS desc_length,
            LN(1 + a.monthly_users) AS log_mu,
            LN(1 + COALESCE(a.runs_30d_total, 0)) AS log_runs,
            LN(1 + a.total_builds) AS log_builds,
            LN(1 + a.bookmarks) AS log_bookmarks,
            EXTRACT(EPOCH FROM ((SELECT MAX(crawl_date)::TIMESTAMP FROM actor_day) - p.pricing_created_at)) / 86400.0 AS age_days
        FROM actor_day a
        JOIN pricing_day p ON a.actor_id = p.actor_id AND a.crawl_date = p.crawl_date
        LEFT JOIN (
            SELECT username, COUNT(*) AS n_actors
            FROM actor_day
            WHERE crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
            GROUP BY username
        ) dev ON a.username = dev.username
        LEFT JOIN (
            SELECT actor_id,
                   COUNT(*) AS n_events,
                   MEDIAN(event_price_usd) AS median_event_price,
                   CASE WHEN AVG(event_price_usd) > 0
                        THEN STDDEV(event_price_usd) / AVG(event_price_usd)
                        ELSE 0 END AS cv_event_price
            FROM pricing_event_detail
            WHERE crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
            GROUP BY actor_id
        ) pe ON a.actor_id = pe.actor_id
        WHERE a.crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
          AND a.pricing_model = 'PAY_PER_EVENT'
        ORDER BY a.actor_id
    """).df()

    # Clean up
    df["agentic"] = df["agentic"].astype(int)
    df["primary_category"] = df["primary_category"].fillna("UNKNOWN").str.strip('"')
    df["n_pricing_events"] = df["n_pricing_events"].fillna(0)
    df["median_event_price"] = df["median_event_price"].fillna(0)
    df["cv_event_price"] = df["cv_event_price"].fillna(0)
    df["desc_length"] = df["desc_length"].fillna(0)
    df["log_builds"] = df["log_builds"].fillna(0)
    df["dev_portfolio_size"] = df["dev_portfolio_size"].fillna(1)
    df["log_age"] = np.log1p(df["age_days"].fillna(df["age_days"].median()))

    logger.info("Analysis dataset: %d PPE actors (%d agentic, %d non-agentic)",
                len(df), df["agentic"].sum(), (1 - df["agentic"]).sum())
    return df


def run_regressions(df: pd.DataFrame) -> dict:
    """OLS regression hierarchy (Models 1-4)."""
    results = {}

    # Model 1: Baseline
    m1 = smf.ols("log_mu ~ agentic", data=df).fit(cov_type="HC1")

    # Model 2: + developer quality controls
    m2 = smf.ols(
        "log_mu ~ agentic + log_builds + dev_portfolio_size + desc_length",
        data=df,
    ).fit(cov_type="HC1")

    # Model 3: + pricing + category FE
    m3 = smf.ols(
        "log_mu ~ agentic + log_builds + dev_portfolio_size + desc_length"
        " + n_pricing_events + C(primary_category)",
        data=df,
    ).fit(cov_type="HC1")

    # Model 4: Same covariates, outcome = log_runs
    m4 = smf.ols(
        "log_runs ~ agentic + log_builds + dev_portfolio_size + desc_length"
        " + n_pricing_events + C(primary_category)",
        data=df,
    ).fit(cov_type="HC1")

    models = {"M1": m1, "M2": m2, "M3": m3, "M4": m4}
    for name, m in models.items():
        coef = m.params.get("agentic", np.nan)
        se = m.bse.get("agentic", np.nan)
        p = m.pvalues.get("agentic", np.nan)
        ci = m.conf_int().loc["agentic"].values if "agentic" in m.params else [np.nan, np.nan]
        results[name] = {
            "outcome": "log_mu" if name != "M4" else "log_runs",
            "coef_agentic": round(float(coef), 4),
            "se": round(float(se), 4),
            "p_value": float(p),
            "ci_95": [round(float(ci[0]), 4), round(float(ci[1]), 4)],
            "r_squared": round(float(m.rsquared), 4),
            "adj_r_squared": round(float(m.rsquared_adj), 4),
            "n_obs": int(m.nobs),
            "controls": list(m.params.index[:10]),
        }
        logger.info("%s: agentic=%.4f (SE=%.4f, p=%.2e), R2=%.4f",
                     name, coef, se, p, m.rsquared)

    # Save regression table as CSV
    reg_rows = []
    for name in ["M1", "M2", "M3", "M4"]:
        r = results[name]
        stars = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else ""
        reg_rows.append({
            "Model": name,
            "Outcome": r["outcome"],
            "Agentic Coef": f"{r['coef_agentic']:.4f}{stars}",
            "SE": f"({r['se']:.4f})",
            "95% CI": f"[{r['ci_95'][0]:.4f}, {r['ci_95'][1]:.4f}]",
            "R-squared": f"{r['r_squared']:.4f}",
            "Adj R-squared": f"{r['adj_r_squared']:.4f}",
            "N": r["n_obs"],
        })
    pd.DataFrame(reg_rows).to_csv(BASE / "tables" / "T6_regression_table.csv", index=False)
    logger.info("T6 saved.")

    return results


def run_psm(df: pd.DataFrame) -> dict:
    """Propensity score matching: 1:1 nearest-neighbor with caliper."""
    results = {}

    # Features for propensity model: listing-time covariates only
    feature_cols = ["log_age", "desc_length", "dev_portfolio_size"]
    cat_dummies = pd.get_dummies(df["primary_category"], prefix="cat").astype(float)
    cat_cols = list(cat_dummies.columns)
    X = np.hstack([df[feature_cols].fillna(0).values, cat_dummies.values])
    y = df["agentic"].values

    # Fit logistic regression (liblinear: fast, deterministic, converges)
    lr = LogisticRegression(solver="liblinear", max_iter=10000, C=1.0, random_state=42)
    lr.fit(X, y)
    df = df.copy()
    df["pscore"] = lr.predict_proba(X)[:, 1]
    for c in cat_cols:
        df[c] = cat_dummies[c].values

    logger.info("Propensity scores: treated mean=%.4f, control mean=%.4f",
                df.loc[df["agentic"] == 1, "pscore"].mean(),
                df.loc[df["agentic"] == 0, "pscore"].mean())

    # 1:1 nearest-neighbor matching with caliper
    treated = df[df["agentic"] == 1].reset_index(drop=True)
    control = df[df["agentic"] == 0].reset_index(drop=True)

    caliper = 0.05 * df["pscore"].std()
    matched_t_idx = []
    matched_c_idx = []
    used_control = set()

    # Compute all pairwise distances
    dist_matrix = cdist(
        treated["pscore"].values.reshape(-1, 1),
        control["pscore"].values.reshape(-1, 1),
        metric="euclidean",
    )

    # Greedy match in ascending propensity-score order (deterministic, matches block9)
    order_t = np.argsort(treated["pscore"].values, kind="stable")

    for i in order_t:
        dists = dist_matrix[i]
        sorted_idx = np.argsort(dists, kind="stable")
        for j in sorted_idx:
            if j not in used_control and dists[j] <= caliper:
                matched_t_idx.append(i)
                matched_c_idx.append(j)
                used_control.add(j)
                break

    n_matched = len(matched_t_idx)
    logger.info("PSM: %d pairs matched (caliper=%.5f)", n_matched, caliper)

    if n_matched < 100:
        logger.warning("Too few matches. Relaxing caliper to 0.1*SD")
        caliper = 0.1 * df["pscore"].std()
        matched_t_idx = []
        matched_c_idx = []
        used_control = set()
        for i in order_t:
            dists = dist_matrix[i]
            sorted_idx = np.argsort(dists, kind="stable")
            for j in sorted_idx:
                if j not in used_control and dists[j] <= caliper:
                    matched_t_idx.append(i)
                    matched_c_idx.append(j)
                    used_control.add(j)
                    break
        n_matched = len(matched_t_idx)
        logger.info("PSM (relaxed): %d pairs matched", n_matched)

    matched_treated = treated.iloc[matched_t_idx]
    matched_control = control.iloc[matched_c_idx]

    # Balance diagnostics: SMD before/after matching for listing-time covariates
    full_treated = df[df["agentic"] == 1]
    full_control = df[df["agentic"] == 0]

    def cont_smd(a, b, col):
        pooled_sd = np.sqrt((a[col].var() + b[col].var()) / 2)
        return (a[col].mean() - b[col].mean()) / pooled_sd if pooled_sd > 0 else 0.0

    def cat_max_smd(a, b, cols):
        smds = []
        for c in cols:
            p_t, p_c = a[c].mean(), b[c].mean()
            pooled = np.sqrt((p_t * (1 - p_t) + p_c * (1 - p_c)) / 2)
            smds.append((p_t - p_c) / pooled if pooled > 0 else 0.0)
        return max(smds, key=abs)

    balance_rows = []
    for col in feature_cols:
        balance_rows.append({
            "Covariate": col,
            "SMD (before)": round(cont_smd(full_treated, full_control, col), 4),
            "SMD (after)": round(cont_smd(matched_treated, matched_control, col), 4),
        })
    balance_rows.append({
        "Covariate": "Category (max)",
        "SMD (before)": round(cat_max_smd(full_treated, full_control, cat_cols), 4),
        "SMD (after)": round(cat_max_smd(matched_treated, matched_control, cat_cols), 4),
    })
    balance_df = pd.DataFrame(balance_rows)
    balance_df.to_csv(BASE / "tables" / "T7_psm_balance.csv", index=False)
    logger.info("T7 (balance diagnostics) saved.")

    # ATT on matched sample
    att_mu = matched_treated["log_mu"].mean() - matched_control["log_mu"].mean()
    att_runs = matched_treated["log_runs"].mean() - matched_control["log_runs"].mean()

    results = {
        "n_matched_pairs": n_matched,
        "caliper": float(caliper),
        "att_log_mu": round(float(att_mu), 4),
        "att_log_runs": round(float(att_runs), 4),
        "balance": balance_rows,
    }
    logger.info("ATT (matched): log_mu=%.4f, log_runs=%.4f", att_mu, att_runs)

    # Figure F4: Propensity score distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Before matching
    axes[0].hist(treated["pscore"], bins=40, alpha=0.5, label="Treated", color="#4C72B0", density=True)
    axes[0].hist(control["pscore"], bins=40, alpha=0.5, label="Control", color="#C44E52", density=True)
    axes[0].set_title("Before Matching")
    axes[0].set_xlabel("Propensity Score")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # After matching
    axes[1].hist(matched_treated["pscore"], bins=40, alpha=0.5, label="Treated", color="#4C72B0", density=True)
    axes[1].hist(matched_control["pscore"], bins=40, alpha=0.5, label="Control", color="#C44E52", density=True)
    axes[1].set_title("After Matching")
    axes[1].set_xlabel("Propensity Score")
    axes[1].legend()

    fig.suptitle(f"Propensity Score Distributions (n={n_matched:,} matched pairs)")
    fig.tight_layout()
    fig.savefig(BASE / "figures" / "F4_pscore_distributions.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("F4 saved.")

    return results


def run_developer_fe(df: pd.DataFrame, con: duckdb.DuckDBPyConnection) -> dict:
    """Within-developer fixed effects analysis."""
    # Find mixed developers
    mixed_devs = con.execute("""
        SELECT username
        FROM actor_day
        WHERE crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
          AND pricing_model = 'PAY_PER_EVENT'
        GROUP BY username
        HAVING SUM(CASE WHEN is_agentic_payments_whitelisted THEN 1 ELSE 0 END) > 0
           AND SUM(CASE WHEN NOT is_agentic_payments_whitelisted THEN 1 ELSE 0 END) > 0
    """).df()["username"].tolist()

    mixed_df = df[df["username"].isin(mixed_devs)].copy()
    n_devs = mixed_df["username"].nunique()
    logger.info("Within-developer FE: %d mixed developers, %d actors",
                n_devs, len(mixed_df))

    if len(mixed_df) < 100:
        logger.warning("Insufficient mixed-developer sample for FE analysis")
        return {"error": "insufficient sample", "n_mixed_devs": n_devs}

    # Developer FE regression
    m_fe = smf.ols(
        "log_mu ~ agentic + log_builds + n_pricing_events + C(username)",
        data=mixed_df,
    ).fit(cov_type="HC1")

    coef = m_fe.params.get("agentic", np.nan)
    se = m_fe.bse.get("agentic", np.nan)
    p = m_fe.pvalues.get("agentic", np.nan)
    ci = m_fe.conf_int().loc["agentic"].values if "agentic" in m_fe.params else [np.nan, np.nan]

    result = {
        "n_mixed_developers": n_devs,
        "n_actors": len(mixed_df),
        "n_agentic": int(mixed_df["agentic"].sum()),
        "n_nonagentic": int((1 - mixed_df["agentic"]).sum()),
        "coef_agentic": round(float(coef), 4),
        "se": round(float(se), 4),
        "p_value": float(p),
        "ci_95": [round(float(ci[0]), 4), round(float(ci[1]), 4)],
        "r_squared": round(float(m_fe.rsquared), 4),
    }
    logger.info("Developer FE: agentic=%.4f (SE=%.4f, p=%.2e), R2=%.4f",
                coef, se, p, m_fe.rsquared)

    return result


def main() -> None:
    for subdir in ["tables", "figures", "data"]:
        (BASE / subdir).mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        df = prepare_data(con)
        reg_results = run_regressions(df)
        psm_results = run_psm(df)
        fe_results = run_developer_fe(df, con)

        all_results = {
            "regressions": reg_results,
            "psm": psm_results,
            "developer_fe": fe_results,
        }
        with open(BASE / "data" / "block4_regression_summary.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        logger.info("Block 4 complete.")
    finally:
        con.close()


if __name__ == "__main__":
    main()
