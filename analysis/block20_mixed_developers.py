"""Block 20: Characterize mixed versus non-mixed developers.

The within-developer fixed-effects design runs on the 335 "mixed" developers
who publish both agentic-ready and non-agentic-ready pay-per-event tools. This
block characterizes that subsample against the developers it excludes, so the
reader can judge how special the identification sample is.

Outputs:
  - results/data/block20_mixed_developers.json
"""

import json
import warnings
from pathlib import Path

import duckdb
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


def _tier(n: int) -> str:
    if n >= 100:
        return "Whale"
    if n >= 10:
        return "Mid"
    if n >= 2:
        return "Boutique"
    return "Solo"


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)

    # Full-store portfolio size (all pricing models, no pricing_day join).
    full_ad = con.execute("""
        SELECT actor_id, username, monthly_users
        FROM actor_day
        WHERE crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
    """).df()

    # PPE actors with detail (pricing_day join for age; inner detail merge).
    ppe_ad = con.execute("""
        SELECT a.actor_id, a.username, a.monthly_users,
               EXTRACT(EPOCH FROM ((SELECT MAX(crawl_date)::TIMESTAMP FROM actor_day) - p.pricing_created_at))
                 / 86400.0 AS age_days
        FROM actor_day a
        JOIN pricing_day p ON a.actor_id = p.actor_id AND a.crawl_date = p.crawl_date
        WHERE a.crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
          AND a.pricing_model = 'PAY_PER_EVENT'
    """).df()
    con.close()

    detail = load_detail()
    ppe = ppe_ad.merge(detail, left_on="actor_id", right_on="actorId",
                       how="inner").copy()

    full_port = full_ad.groupby("username")["actor_id"].count().rename("full_portfolio")
    ppe_port = ppe.groupby("username")["actor_id"].count().rename("ppe_portfolio")

    dev = (ppe.groupby("username")
           .agg(n_ppe=("actor_id", "count"),
                n_eligible=("eligible", "sum"),
                med_age=("age_days", "median"),
                med_mu=("monthly_users", "median"),
                mean_mu=("monthly_users", "mean"))
           .reset_index())
    dev["n_non_eligible"] = dev["n_ppe"] - dev["n_eligible"]
    dev["mixed"] = ((dev["n_eligible"] > 0) & (dev["n_non_eligible"] > 0)).astype(int)
    dev = dev.merge(full_port, on="username", how="left")
    dev = dev.merge(ppe_port, on="username", how="left")
    dev["tier"] = dev["full_portfolio"].map(_tier)

    out: dict = {}
    for grp, name in [(1, "mixed"), (0, "non_mixed")]:
        s = dev[dev["mixed"] == grp]
        out[name] = {
            "n_developers": int(len(s)),
            "pct_of_ppe_developers": round(len(s) / len(dev) * 100, 1),
            "median_full_portfolio": float(s["full_portfolio"].median()),
            "median_ppe_portfolio": float(s["ppe_portfolio"].median()),
            "median_tool_age_days": round(float(s["med_age"].median()), 1),
            "median_monthly_users": round(float(s["med_mu"].median()), 1),
            "mean_monthly_users": round(float(s["mean_mu"].mean()), 2),
            "pct_eligible": round(float(s["n_eligible"].sum() / s["n_ppe"].sum() * 100), 1),
            "tier_pct": (s["tier"].value_counts(normalize=True)
                         .reindex(["Solo", "Boutique", "Mid", "Whale"])
                         .fillna(0) * 100).round(1).to_dict(),
        }

    out["n_ppe_developers_total"] = int(len(dev))
    out["n_mixed"] = int(dev["mixed"].sum())

    (BASE / "data" / "block20_mixed_developers.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
