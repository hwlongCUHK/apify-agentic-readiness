"""Block 22: Concentration bootstrap / downsampling.

The non-agentic-ready group is small (n = 2,585), so its top-1% share is a
single top-26-tool quantity while the agentic-ready top-1% is a 489-tool
quantity. To test whether the upper-tail difference survives equal sample size,
we downsample the agentic-ready group to the non-agentic-ready size and
recompute Gini and top-N shares across many draws.

Outputs:
  - results/data/block22_concentration_bootstrap.json
"""

import json
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "apify_panel.duckdb"
DETAIL_PATH = (Path(__file__).resolve().parent.parent / "data" / "raw"
               / "2026-08-27.actor_detail.jsonl")
BASE = Path(__file__).resolve().parent.parent / "results"


def load_detail() -> pd.DataFrame:
    recs = [json.loads(line) for line in open(DETAIL_PATH)]
    d = pd.DataFrame(recs)
    d = d[d["status"] == 200][["actorId", "actorPermissionLevel", "standbyEnabled"]]
    d["limited"] = (d["actorPermissionLevel"] == "LIMITED_PERMISSIONS").astype(int)
    d["standby"] = d["standbyEnabled"].fillna(False).astype(int)
    d["eligible"] = ((d["limited"] == 1) & (d["standby"] == 0)).astype(int)
    return d


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * idx - n - 1) @ x / (n * x.sum()))


def top_share(x: np.ndarray, p: float) -> float:
    x = np.sort(np.asarray(x, dtype=float))[::-1]
    if len(x) == 0 or x.sum() == 0:
        return 0.0
    k = max(1, int(np.ceil(len(x) * p)))
    return float(x[:k].sum() / x.sum())


def theil(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    if len(x) == 0:
        return 0.0
    mu = x.mean()
    return float((x / mu * np.log(x / mu)).mean())


def summarize(x: np.ndarray) -> dict:
    return {
        "n": int(len(x)),
        "gini": round(gini(x), 3),
        "top_1pct": round(top_share(x, 0.01) * 100, 1),
        "top_5pct": round(top_share(x, 0.05) * 100, 1),
        "top_10pct": round(top_share(x, 0.10) * 100, 1),
        "theil": round(theil(x), 2),
    }


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    ad = con.execute("""
        SELECT actor_id, monthly_users FROM actor_day
        WHERE crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
          AND pricing_model = 'PAY_PER_EVENT'
    """).df()
    con.close()

    detail = load_detail()
    m = ad.merge(detail, left_on="actor_id", right_on="actorId", how="inner")
    elig = m[m["eligible"] == 1]["monthly_users"].values
    non = m[m["eligible"] == 0]["monthly_users"].values

    obs = {"eligible": summarize(elig), "non_eligible": summarize(non)}

    # Downsample eligible to the non-eligible size.
    rng = np.random.default_rng(42)
    B = 2000
    n_non = len(non)
    draws = {"gini": [], "top_1pct": [], "top_5pct": [], "top_10pct": [], "theil": []}
    for _ in range(B):
        s = rng.choice(elig, size=n_non, replace=False)
        draws["gini"].append(gini(s))
        draws["top_1pct"].append(top_share(s, 0.01) * 100)
        draws["top_5pct"].append(top_share(s, 0.05) * 100)
        draws["top_10pct"].append(top_share(s, 0.10) * 100)
        draws["theil"].append(theil(s))

    ds = {}
    for k, arr in draws.items():
        arr = np.asarray(arr)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        ds[k] = {
            "mean": round(float(arr.mean()), 3),
            "ci95_low": round(float(lo), 3),
            "ci95_high": round(float(hi), 3),
        }

    # Bootstrap CIs for the difference (eligible - non-eligible) per metric.
    stat_fn = {
        "gini": gini,
        "top_1pct": lambda x: top_share(x, 0.01) * 100,
        "top_5pct": lambda x: top_share(x, 0.05) * 100,
        "top_10pct": lambda x: top_share(x, 0.10) * 100,
        "theil": theil,
    }
    diff_ci = {}
    for k, fn in stat_fn.items():
        d = []
        for _ in range(B):
            s_elig = rng.choice(elig, size=len(elig), replace=True)
            s_non = rng.choice(non, size=len(non), replace=True)
            d.append(fn(s_elig) - fn(s_non))
        d = np.asarray(d)
        lo, hi = np.percentile(d, [2.5, 97.5])
        diff_ci[k] = {
            "mean_diff": round(float(d.mean()), 3),
            "ci95_low": round(float(lo), 3),
            "ci95_high": round(float(hi), 3),
            "zero_in_ci": bool(lo <= 0 <= hi),
        }

    out = {
        "observed": obs,
        "downsampled_eligible_to_non_eligible_size": {
            "n_non_eligible": n_non,
            "n_draws": B,
            "metrics": ds,
        },
        "difference_bootstrap_ci": diff_ci,
    }
    (BASE / "data" / "block22_concentration_bootstrap.json").write_text(
        json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
