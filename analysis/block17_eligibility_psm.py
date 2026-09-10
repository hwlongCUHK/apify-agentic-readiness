"""Block 17: propensity-score matching for the eligibility treatment.

Treatment = eligible (limited-permissions AND non-standby). Propensity score is
estimated from listing-time covariates only (actor age, description length,
developer portfolio size, primary category), matching block15's controls.
Because eligible is ~19:1 over non-eligible, 1:1 matching covers only a small
fraction of eligible actors, so we report full-sample weighting estimators
(overlap/ATO and IPTW/ATT) as the primary PSM output.

Outputs:
  - results/data/block17_eligibility_psm.json
"""

import json
import math
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


def overlap_ato(df: pd.DataFrame, ps: np.ndarray) -> float:
    w_t = 1 - ps
    w_c = ps
    y = df["log_mu"].values
    t = df["eligible"].values == 1
    return float((y[t] * w_t[t]).sum() / w_t[t].sum()
                 - (y[~t] * w_c[~t]).sum() / w_c[~t].sum())


def iptw_att(df: pd.DataFrame, ps: np.ndarray) -> float:
    ps = np.clip(ps, 1e-6, 1 - 1e-6)
    t = df["eligible"].values == 1
    w = np.ones(len(df))
    w[~t] = ps[~t] / (1 - ps[~t])
    y = df["log_mu"].values
    return float(y[t].mean() - (y[~t] * w[~t]).sum() / w[~t].sum())


def match_1to1(df: pd.DataFrame, ps: np.ndarray) -> float:
    t = df["eligible"].values == 1
    pt, pc = ps[t], ps[~t]
    yt, yc = df["log_mu"].values[t], df["log_mu"].values[~t]
    order = np.argsort(pc)
    pc_sorted = pc[order]
    yc_sorted = yc[order]
    idx = np.searchsorted(pc_sorted, pt)
    idx = np.clip(idx, 0, len(pc_sorted) - 1)
    left = pc_sorted[np.clip(idx - 1, 0, len(pc_sorted) - 1)]
    right = pc_sorted[np.clip(idx, 0, len(pc_sorted) - 1)]
    use_left = np.abs(pt - left) <= np.abs(pt - right)
    match = np.where(use_left, np.clip(idx - 1, 0, len(pc_sorted) - 1), idx)
    matched_yc = yc_sorted[match]
    return float(yt.mean() - matched_yc.mean())


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = load_ppe(con)
    con.close()
    detail = load_detail()
    df = df.merge(detail, left_on="actor_id", right_on="actorId", how="inner")
    df["eligible"] = ((df["limited"] == 1) & (df["standby"] == 0)).astype(int)

    ps = fit_ps(df)
    n_elig = int(df["eligible"].sum())
    n_non = int(len(df) - n_elig)

    out = {
        "n_eligible": n_elig, "n_non_eligible": n_non,
        "ratio": round(n_elig / max(n_non, 1), 2),
        "overlap_ato": {"coef": round(overlap_ato(df, ps), 4),
                        "pct": round((math.exp(overlap_ato(df, ps)) - 1) * 100, 1)},
        "iptw_att": {"coef": round(iptw_att(df, ps), 4),
                     "pct": round((math.exp(iptw_att(df, ps)) - 1) * 100, 1)},
        "match_1to1": {"coef": round(match_1to1(df, ps), 4),
                       "pct": round((math.exp(match_1to1(df, ps)) - 1) * 100, 1)},
    }
    (BASE / "data" / "block17_eligibility_psm.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
