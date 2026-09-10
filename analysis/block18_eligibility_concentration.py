"""Block 18: concentration for the eligibility treatment.

Replaces the whitelisted/non-whitelisted split with the eligibility substance
(eligible = limited-permissions AND non-standby), and re-computes the
concentration statistics: Gini, top-N shares, and a Theil decomposition into
within- and between-category components, for eligible vs non-eligible PPE
actors.

Outputs:
  - results/data/block18_eligibility_concentration.json
"""

import json
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from block15_eligibility import DB_PATH, DETAIL_PATH, BASE, load_ppe, load_detail

warnings.filterwarnings("ignore")


def gini(values: np.ndarray) -> float:
    v = np.sort(values.astype(float))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * v) - (n + 1) * np.sum(v)) / (n * np.sum(v)))


def top_share(values: np.ndarray, q: float) -> float:
    v = np.sort(values.astype(float))[::-1]
    if v.sum() == 0:
        return 0.0
    k = max(1, int(np.ceil(len(v) * q)))
    return float(v[:k].sum() / v.sum() * 100)


def theil(values: np.ndarray) -> float:
    v = values.astype(float)
    v = v[v > 0]
    if len(v) == 0 or v.sum() == 0:
        return 0.0
    s = v / v.sum()
    return float(np.sum(s * np.log(s * len(v))))


def between_theil(groups: pd.Series, values: pd.Series) -> float:
    df = pd.DataFrame({"g": groups, "v": values.astype(float)})
    X = df["v"].sum()
    N = len(df)
    bt = 0.0
    for _, grp in df.groupby("g"):
        Xc = grp["v"].sum()
        Nc = len(grp)
        if Xc > 0:
            bt += (Xc / X) * np.log((Xc / X) / (Nc / N))
    return float(bt)


def concentration(df: pd.DataFrame, col: str) -> dict:
    """Gini / top-N / Theil for a binary group column."""
    out = {}
    for label, mask in [(1, df[col] == 1), (0, df[col] == 0)]:
        v = df.loc[mask, "monthly_users"].values
        t = theil(v)
        bt = between_theil(df.loc[mask, "primary_category"], df.loc[mask, "monthly_users"])
        key = "eligible" if label == 1 else "non_eligible"
        out[key] = {
            "n": int(mask.sum()),
            "gini": round(gini(v), 3),
            "top_1pct": round(top_share(v, 0.01), 1),
            "top_5pct": round(top_share(v, 0.05), 1),
            "top_10pct": round(top_share(v, 0.10), 1),
            "theil_total": round(t, 3),
            "theil_between_pct": round(bt / t * 100, 1) if t > 0 else 0.0,
        }
    return out


def within_category_gini(df: pd.DataFrame, col: str) -> dict:
    """Count categories where eligible Gini > non-eligible Gini."""
    n_higher = 0
    n_eligible_cats = 0
    details = []
    for cat, s in df.groupby("primary_category"):
        e = s[s[col] == 1]["monthly_users"].values
        ne = s[s[col] == 0]["monthly_users"].values
        if len(e) < 30 or len(ne) < 30:
            continue
        ge, gne = gini(e), gini(ne)
        n_eligible_cats += 1
        if ge > gne:
            n_higher += 1
        details.append({"category": cat, "eligible_gini": round(ge, 3),
                        "non_eligible_gini": round(gne, 3)})
    return {"higher_in": n_higher, "total": n_eligible_cats, "details": details}


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = load_ppe(con)
    con.close()
    detail = load_detail()
    df = df.merge(detail, left_on="actor_id", right_on="actorId", how="inner")
    df["eligible"] = ((df["limited"] == 1) & (df["standby"] == 0)).astype(int)

    conc = concentration(df, "eligible")
    wcg = within_category_gini(df, "eligible")

    out = {"concentration": conc, "within_category_gini": wcg}
    (BASE / "data" / "block18_eligibility_concentration.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
