"""Block 19: eligible category breakdown for F2.

Regenerates the category-composition data with agentic-ready (eligible) share
instead of whitelist share, using multi-hot category membership.

Outputs:
  - results/data/block19_eligible_category_breakdown.csv
"""

import warnings
from pathlib import Path

import duckdb
import pandas as pd

from block15_eligibility import DB_PATH, DETAIL_PATH, BASE, load_detail

warnings.filterwarnings("ignore")


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    cats = con.execute("""
        SELECT a.actor_id,
               CAST(json_extract(a.categories, '$[*]') AS VARCHAR[]) AS cats_array
        FROM actor_day a
        WHERE a.crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
          AND a.pricing_model = 'PAY_PER_EVENT'
        ORDER BY a.actor_id
    """).df()
    con.close()

    d = load_detail()[["actorId", "limited", "standby"]]
    d["eligible"] = ((d["limited"] == 1) & (d["standby"] == 0)).astype(int)

    cats["cats_list"] = cats["cats_array"].apply(
        lambda arr: [c.strip(chr(34)) for c in arr] if arr is not None and not (isinstance(arr, float) and arr != arr) and str(arr) != "<NA>" else [])
    merged = cats[["actor_id", "cats_list"]].merge(
        d[["actorId", "eligible"]], left_on="actor_id", right_on="actorId", how="inner")
    exploded = merged.explode("cats_list")
    agg = (exploded.groupby("cats_list")
           .agg(n_actors=("eligible", "size"),
                pct_eligible=("eligible", lambda s: round(s.mean() * 100, 1)))
           .reset_index()
           .rename(columns={"cats_list": "category"})
           .sort_values("n_actors", ascending=False))

    agg.to_csv(BASE / "data" / "block19_eligible_category_breakdown.csv", index=False)
    print(agg.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
