"""Block 25: 2x2 architecture decomposition of the eligibility treatment.

Treatment = agentic-ready = limited-permissions AND non-standby, within the
pay-per-event (PPE) sample. This block unpacks the binary treatment into its
two binary architecture dimensions (permission level x standby) and reports:

1. A 2x2 descriptive table over the full PPE-with-detail sample (rows =
   permission level, columns = standby), with n / mean / median monthly_users
   per cell. The ready group is limited & non-standby; the other three cells
   are the non-ready group.

2. Within-developer fixed-effects estimates (equal-developer-weighted WLS,
   developer-clustered SEs) restricted to mixed developers:
   (a) limited vs full, (b) standby vs non-standby, (c) three cell dummies
   with limited & non-standby (= ready) as the omitted reference.

Outputs:
  - results/data/block25_architecture_2x2.json
"""

import json
import math
import warnings

import duckdb
import statsmodels.formula.api as smf

from block15_eligibility import (
    DB_PATH, DETAIL_PATH, BASE, CTRLS, load_ppe, load_detail, mixed_developers,
)

warnings.filterwarnings("ignore")


def fe_term(m, name: str) -> dict:
    """Extract coef/se/p/pct for one coefficient from a fitted model."""
    c = float(m.params[name])
    se = float(m.bse[name])
    p = float(m.pvalues[name])
    return {
        "coef": round(c, 4),
        "se": round(se, 4),
        "p": p,
        "pct": round((math.exp(c) - 1) * 100, 1),
    }


def architecture_2x2(df) -> dict:
    """2x2 descriptive table: permission x standby -> n/mean/median MU."""
    table = {}
    for perm_val, perm_label in [(0, "FULL_PERMISSIONS"),
                                 (1, "LIMITED_PERMISSIONS")]:
        row = {}
        for sb_val, sb_label in [(1, "standby"), (0, "non_standby")]:
            s = df[(df["limited"] == perm_val) & (df["standby"] == sb_val)]
            row[sb_label] = {
                "n": int(len(s)),
                "mean_monthly_users": round(float(s["monthly_users"].mean()), 2),
                "median_monthly_users": round(float(s["monthly_users"].median()), 1),
            }
        table[perm_label] = row
    table["ready_cell"] = "LIMITED_PERMISSIONS x non_standby"
    return table


def within_developer_fe(df) -> dict:
    """Within-developer FE on the mixed-developer sample (3 specifications)."""
    mdf = mixed_developers(df, "eligible")
    out = {
        "n_developers": int(mdf["username"].nunique()),
        "n_actors": int(len(mdf)),
    }

    # (a) limited vs full
    ma = smf.wls(f"log_mu ~ limited + {CTRLS}", data=mdf,
                 weights=mdf["w_dev"]).fit(
        cov_type="cluster", cov_kwds={"groups": mdf["username"]})
    out["limited_vs_full"] = fe_term(ma, "limited")

    # (b) standby vs non-standby
    mb = smf.wls(f"log_mu ~ standby + {CTRLS}", data=mdf,
                 weights=mdf["w_dev"]).fit(
        cov_type="cluster", cov_kwds={"groups": mdf["username"]})
    out["standby_vs_nonstandby"] = fe_term(mb, "standby")

    # (c) three cell dummies, reference = ready (limited & non-standby)
    mc = smf.wls(
        f"log_mu ~ full_nonstandby + full_standby + limited_standby + {CTRLS}",
        data=mdf, weights=mdf["w_dev"]).fit(
        cov_type="cluster", cov_kwds={"groups": mdf["username"]})
    out["three_dummies_ref_ready"] = {
        "full_nonstandby": fe_term(mc, "full_nonstandby"),
        "full_standby": fe_term(mc, "full_standby"),
        "limited_standby": fe_term(mc, "limited_standby"),
    }
    return out


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = load_ppe(con)
    con.close()
    detail = load_detail()
    df = df.merge(detail, left_on="actor_id", right_on="actorId", how="inner")
    df["eligible"] = ((df["limited"] == 1) & (df["standby"] == 0)).astype(int)

    # Cell dummies for the 2x2 within-developer specification (c).
    df["full_nonstandby"] = ((df["limited"] == 0) & (df["standby"] == 0)).astype(int)
    df["full_standby"] = ((df["limited"] == 0) & (df["standby"] == 1)).astype(int)
    df["limited_standby"] = ((df["limited"] == 1) & (df["standby"] == 1)).astype(int)

    out = {
        "architecture_2x2": architecture_2x2(df),
        "within_developer_fe": within_developer_fe(df),
    }
    (BASE / "data" / "block25_architecture_2x2.json").write_text(
        json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
