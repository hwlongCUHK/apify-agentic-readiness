"""Block 26: age robustness for the eligibility analysis.

Complements block15 (treatment = eligible = limited-permissions AND non-standby,
within pay-per-event actors) by checking sensitivity of the within-developer
fixed-effects estimate to actor age:

1. Age subsamples: re-run the equal-developer-weighted FE on mixed developers
   restricted to actors aged >=30, >=90, and >=180 days.

2. Age-bin FE: replace the continuous ``log_age`` control with 4 age-bin fixed
   effects, so the eligible effect is identified off within-bin variation.

3. Developer-specific gap distribution: for each mixed developer, the
   ready-minus-non-ready gap in mean ``log_mu``, summarized across developers.

Outputs:
  - results/data/block26_age_robustness.json
"""

import json
import math
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from block15_eligibility import (
    BASE,
    CTRLS,
    DB_PATH,
    DETAIL_PATH,
    load_detail,
    load_ppe,
    mixed_developers,
)

warnings.filterwarnings("ignore")


def fe_coef(mdf: pd.DataFrame, formula: str) -> dict:
    """Equal-developer-weighted FE coefficient for ``eligible``."""
    m = smf.wls(formula, data=mdf, weights=mdf["w_dev"]).fit(
        cov_type="cluster", cov_kwds={"groups": mdf["username"]})
    c = float(m.params["eligible"])
    se = float(m.bse["eligible"])
    p = float(m.pvalues["eligible"])
    return {"coef": round(c, 4), "se": round(se, 4), "p": p,
            "pct": round((math.exp(c) - 1) * 100, 1),
            "n_developers": int(mdf["username"].nunique()),
            "n_actors": int(len(mdf))}


def age_subsamples(df: pd.DataFrame) -> dict:
    """FE on actors aged >=30 / >=90 / >=180 days (mixed developers only)."""
    out = {}
    for thresh in [30, 90, 180]:
        sub = df[df["age_days"] >= thresh]
        mdf = mixed_developers(sub, "eligible")
        out[f">={thresh}"] = fe_coef(mdf, f"log_mu ~ eligible + {CTRLS}")
    return out


def age_bin_fe(df: pd.DataFrame) -> dict:
    """FE with age-bin fixed effects replacing the continuous log_age control."""
    mdf = mixed_developers(df, "eligible")
    # Clip the small number of negative ages (created-after-crawl anomalies) to
    # zero so they fall into the youngest bin and the sample matches the base FE.
    mdf["age_bin"] = pd.cut(
        mdf["age_days"].clip(lower=0), bins=[0, 30, 90, 180, np.inf],
        right=False, labels=["0-30", "30-90", "90-180", "180+"])
    formula = ("log_mu ~ eligible + C(age_bin) + desc_length "
               "+ C(primary_category) + C(username)")
    m = smf.wls(formula, data=mdf, weights=mdf["w_dev"]).fit(
        cov_type="cluster", cov_kwds={"groups": mdf["username"]})
    c = float(m.params["eligible"])
    se = float(m.bse["eligible"])
    p = float(m.pvalues["eligible"])
    return {"coef": round(c, 4), "se": round(se, 4), "p": p,
            "pct": round((math.exp(c) - 1) * 100, 1),
            "n_developers": int(mdf["username"].nunique()),
            "n_actors": int(len(mdf))}


def gap_distribution(df: pd.DataFrame) -> dict:
    """Per-developer ready-minus-non-ready gap in mean log_mu."""
    mdf = mixed_developers(df, "eligible")
    gaps = []
    for _, g in mdf.groupby("username"):
        ready = float(g.loc[g["eligible"] == 1, "log_mu"].mean())
        nonready = float(g.loc[g["eligible"] == 0, "log_mu"].mean())
        gaps.append(ready - nonready)
    gaps = pd.Series(gaps, dtype=float)
    return {
        "n": int(len(gaps)),
        "mean": round(float(gaps.mean()), 4),
        "median": round(float(gaps.median()), 4),
        "sd": round(float(gaps.std()), 4),
        "p25": round(float(gaps.quantile(0.25)), 4),
        "p75": round(float(gaps.quantile(0.75)), 4),
        "frac_positive": round(float((gaps > 0).mean()), 4),
    }


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = load_ppe(con)
    con.close()
    detail = load_detail()
    df = df.merge(detail, left_on="actor_id", right_on="actorId", how="inner")
    df["eligible"] = ((df["limited"] == 1) & (df["standby"] == 0)).astype(int)

    out = {
        "age_subsamples": age_subsamples(df),
        "age_bin_fe": age_bin_fe(df),
        "gap_distribution": gap_distribution(df),
    }
    (BASE / "data" / "block26_age_robustness.json").write_text(
        json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
