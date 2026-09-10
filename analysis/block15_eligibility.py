"""Block 15: Agentic eligibility analysis.

Treatment = agentic eligibility *substance*, not the whitelist label.

The whitelist flag is a platform label for machine-consumer eligibility; its
substance is the architecture the platform requires. Within the pay-per-event
(PPE) sample, eligibility is:

    eligible = limited-permissions AND non-standby

(permissions come from ``actorPermissionLevel``, standby from ``actorStandby`` /
``standbyUrl``, both fetched from GET /v2/acts/{actorId} and archived in
``data/raw/2026-08-27.actor_detail.jsonl``).

This block re-estimates the core adoption analyses with ``eligible`` as the
binary treatment: within-developer fixed effects (equal-developer- and
actor-weighted), the extensive/intensive margin, and per-category
heterogeneity. The whitelist flag is never used.

Outputs:
  - results/data/block15_eligibility.json
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
    """PPE actors (latest crawl) with listing-time covariates."""
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
    for c in ["n_pricing_events", "desc_length", "dev_portfolio_size"]:
        df[c] = df[c].fillna(0)
    df["log_builds"] = np.log1p(df["total_builds"].fillna(0))
    df["log_age"] = np.log1p(df["age_days"].fillna(df["age_days"].median()))
    df["log_mu"] = np.log1p(df["monthly_users"])
    return df


def load_detail() -> pd.DataFrame:
    """Actor detail (permission level + standby) from the archived JSONL."""
    recs = [json.loads(line) for line in open(DETAIL_PATH)]
    d = pd.DataFrame(recs)
    d = d[d["status"] == 200][["actorId", "actorPermissionLevel", "standbyEnabled"]]
    d["limited"] = (d["actorPermissionLevel"] == "LIMITED_PERMISSIONS").astype(int)
    d["standby"] = d["standbyEnabled"].fillna(False).astype(int)
    return d


def mixed_developers(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Restrict to developers whose `col` varies within their portfolio."""
    mixed = df.groupby("username").apply(lambda g: g[col].nunique() == 2)
    out = df[df["username"].isin(mixed[mixed].index)].copy()
    out["w_dev"] = 1.0 / out.groupby("username")["username"].transform("count")
    return out


def fe_effect(df: pd.DataFrame, formula: str, weighted: bool) -> dict:
    """Within-developer FE on the mixed-developer sample."""
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


def decompose(df: pd.DataFrame) -> dict:
    """limited and standby as two independent binary regressors."""
    mdf = mixed_developers(df, "eligible")
    f = f"log_mu ~ limited + standby + {CTRLS}"
    m = smf.wls(f, data=mdf, weights=mdf["w_dev"]).fit(
        cov_type="cluster", cov_kwds={"groups": mdf["username"]})
    out = {}
    for k in ["limited", "standby"]:
        c = float(m.params[k]); se = float(m.bse[k]); p = float(m.pvalues[k])
        out[k] = {"coef": round(c, 4), "se": round(se, 4), "p": p,
                  "pct": round((math.exp(c) - 1) * 100, 1)}
    return out


def margins(df: pd.DataFrame) -> dict:
    """Extensive / intensive / top-decile margin for the eligible treatment."""
    mdf = mixed_developers(df, "eligible")
    mdf["any_mu"] = (mdf["monthly_users"] > 0).astype(int)
    p90 = float(np.quantile(df[df["monthly_users"] > 0]["monthly_users"], 0.90))
    mdf["top_decile"] = (mdf["monthly_users"] >= p90).astype(int)

    def lpm(y):
        m = smf.wls(f"{y} ~ eligible + {CTRLS}", data=mdf, weights=mdf["w_dev"]).fit(
            cov_type="cluster", cov_kwds={"groups": mdf["username"]})
        c = float(m.params["eligible"]); se = float(m.bse["eligible"]); p = float(m.pvalues["eligible"])
        return {"coef": round(c, 4), "se": round(se, 4), "p": p, "pp": round(c * 100, 1)}

    ext = lpm("any_mu")
    top = lpm("top_decile")

    sub = mdf[mdf["monthly_users"] > 0].copy()
    sub["w_dev"] = 1.0 / sub.groupby("username")["username"].transform("count")
    sub_mixed = sub.groupby("username").apply(lambda g: g["eligible"].nunique() == 2)
    sub = sub[sub["username"].isin(sub_mixed[sub_mixed].index)].copy()
    sub["w_dev"] = 1.0 / sub.groupby("username")["username"].transform("count")
    m = smf.wls(f"log_mu ~ eligible + {CTRLS}", data=sub, weights=sub["w_dev"]).fit(
        cov_type="cluster", cov_kwds={"groups": sub["username"]})
    c = float(m.params["eligible"]); se = float(m.bse["eligible"]); p = float(m.pvalues["eligible"])
    intens = {"coef": round(c, 4), "se": round(se, 4), "p": p,
              "pct": round((math.exp(c) - 1) * 100, 1)}

    return {"extensive": ext, "intensive": intens, "top_decile": top, "p90": round(p90, 1)}


def category_heterogeneity(df: pd.DataFrame) -> list:
    """Per-category eligible effect (M2-style controls, developer-clustered SEs)."""
    rows = []
    for cat in sorted(df["primary_category"].unique()):
        s = df[df["primary_category"] == cat]
        if len(s) < 200 or s["eligible"].nunique() != 2:
            continue
        na = int(s["eligible"].sum()); nn = len(s) - na
        if na < 30 or nn < 30:
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
    con.close()
    detail = load_detail()
    df = df.merge(detail, left_on="actor_id", right_on="actorId", how="inner")
    df["eligible"] = ((df["limited"] == 1) & (df["standby"] == 0)).astype(int)

    equal_dev = fe_effect(df, f"log_mu ~ eligible + {CTRLS}", weighted=True)
    actor_w = fe_effect(df, f"log_mu ~ eligible + {CTRLS}", weighted=False)
    dec = decompose(df)
    marg = margins(df)
    cats = category_heterogeneity(df)
    bonf = 0.05 / len(cats) if cats else None

    out = {
        "n_ppe": int(len(df)),
        "n_eligible": int(df["eligible"].sum()),
        "pct_eligible": round(df["eligible"].mean() * 100, 1),
        "fe_equal_dev": equal_dev,
        "fe_actor_weighted": actor_w,
        "decompose": dec,
        "margins": marg,
        "category_heterogeneity": {
            "n_categories": len(cats),
            "bonferroni_alpha": round(bonf, 5) if bonf else None,
            "rows": cats,
        },
    }
    (BASE / "data" / "block15_eligibility.json").write_text(json.dumps(out, indent=2))

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
