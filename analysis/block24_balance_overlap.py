"""Block 24: covariate balance and propensity-score overlap diagnostics.

Because 95% of PPE tools are agentic-ready, the comparison group is small and
potentially selected. This block reports standardized mean differences (SMD)
for the listing-time covariates before and after inverse-probability weighting,
plus the propensity-score overlap between the treated and control groups.

Outputs:
  - results/data/block24_balance_overlap.json
"""

import json
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from block15_eligibility import DB_PATH, DETAIL_PATH, BASE, load_ppe, load_detail

warnings.filterwarnings("ignore")


def fit_ps(df: pd.DataFrame) -> np.ndarray:
    X = pd.get_dummies(
        df[["log_age", "desc_length", "dev_portfolio_size", "primary_category"]],
        columns=["primary_category"], drop_first=True).astype(float)
    lr = LogisticRegression(solver="liblinear", max_iter=10000, C=1.0, random_state=42)
    lr.fit(X, df["eligible"])
    return lr.predict_proba(X)[:, 1]


def smd(t: np.ndarray, c: np.ndarray, wt=None, wc=None) -> float:
    """Standardized mean difference (treated - control)."""
    mt = np.average(t, weights=wt) if wt is not None else t.mean()
    mc = np.average(c, weights=wc) if wc is not None else c.mean()
    nt, nc = len(t), len(c)
    vt = np.average((t - t.mean()) ** 2, weights=wt) if wt is not None else t.var()
    vc = np.average((c - c.mean()) ** 2, weights=wc) if wc is not None else c.var()
    sp = np.sqrt((vt + vc) / 2.0)
    return float((mt - mc) / sp) if sp > 0 else 0.0


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = load_ppe(con)
    con.close()
    detail = load_detail()
    df = df.merge(detail, left_on="actor_id", right_on="actorId", how="inner")
    df["eligible"] = ((df["limited"] == 1) & (df["standby"] == 0)).astype(int)

    ps = fit_ps(df)
    t = df["eligible"].values == 1

    # IPTW(ATT) weights: control reweighted toward the treated distribution.
    ps_clip = np.clip(ps, 1e-6, 1 - 1e-6)
    w_t = np.ones(len(df))
    w_c = ps_clip[~t] / (1 - ps_clip[~t])

    balance = {}
    for cov in ["log_age", "desc_length", "dev_portfolio_size"]:
        x = df[cov].values
        before = smd(x[t], x[~t])
        after = smd(x[t], x[~t], wt=np.ones(t.sum()), wc=w_c)
        balance[cov] = {
            "smd_before": round(before, 3),
            "smd_after": round(after, 3),
        }

    # Propensity-score overlap: distribution of PS in each group.
    def ps_summary(arr):
        return {
            "mean": round(float(arr.mean()), 3),
            "min": round(float(arr.min()), 3),
            "p10": round(float(np.percentile(arr, 10)), 3),
            "median": round(float(np.median(arr)), 3),
            "p90": round(float(np.percentile(arr, 90)), 3),
            "max": round(float(arr.max()), 3),
        }

    out = {
        "n_eligible": int(t.sum()),
        "n_non_eligible": int((~t).sum()),
        "ratio": round(t.sum() / max((~t).sum(), 1), 2),
        "balance_smd": balance,
        "ps_treated": ps_summary(ps[t]),
        "ps_control": ps_summary(ps[~t]),
    }
    (BASE / "data" / "block24_balance_overlap.json").write_text(
        json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
