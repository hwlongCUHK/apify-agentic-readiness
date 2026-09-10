"""Block 1: Data Quality and Panel Validation.

Sanity-check the DuckDB panel before any analysis.
Outputs: results/data/block1_quality_report.json
"""

import json
import logging
from pathlib import Path

import duckdb

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "apify_panel.duckdb"
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "data"


def run(con: duckdb.DuckDBPyConnection) -> dict:
    report: dict = {}

    # 1.1 Panel integrity: duplicate (crawl_date, actor_id) check
    dupes = con.execute("""
        SELECT crawl_date, actor_id, COUNT(*) AS n
        FROM actor_day
        GROUP BY crawl_date, actor_id
        HAVING n > 1
    """).fetchall()
    report["duplicate_rows"] = len(dupes)
    logger.info("Duplicate (crawl_date, actor_id) rows: %d", len(dupes))

    # Coverage: actors present across all days
    coverage = con.execute("""
        SELECT
            COUNT(DISTINCT actor_id) AS total_actors,
            COUNT(DISTINCT actor_id) FILTER (
                WHERE actor_id IN (
                    SELECT actor_id FROM actor_day
                    GROUP BY actor_id
                    HAVING COUNT(DISTINCT crawl_date) = (
                        SELECT COUNT(DISTINCT crawl_date) FROM actor_day
                    )
                )
            ) AS actors_all_days
        FROM actor_day
    """).fetchone()
    report["total_actors_panel"] = coverage[0]
    report["actors_present_all_days"] = coverage[1]
    report["coverage_pct"] = round(coverage[1] / coverage[0] * 100, 2) if coverage[0] else 0
    logger.info(
        "Coverage: %d/%d actors present all days (%.1f%%)",
        coverage[1], coverage[0], report["coverage_pct"],
    )

    # Per-day actor counts
    daily = con.execute("""
        SELECT crawl_date, COUNT(*) AS n_actors
        FROM actor_day
        GROUP BY crawl_date
        ORDER BY crawl_date
    """).fetchall()
    report["daily_counts"] = {str(r[0]): r[1] for r in daily}
    for d in daily:
        logger.info("  %s: %d actors", d[0], d[1])

    # 1.2 Agentic flag stability
    flippers = con.execute("""
        SELECT actor_id, COUNT(DISTINCT is_agentic_payments_whitelisted) AS n_states
        FROM actor_day
        GROUP BY actor_id
        HAVING n_states > 1
    """).fetchall()
    report["agentic_flag_flippers"] = len(flippers)
    logger.info("Actors with unstable agentic flag: %d", len(flippers))

    # Pricing model stability
    pricing_flips = con.execute("""
        SELECT actor_id, COUNT(DISTINCT pricing_model) AS n_models
        FROM actor_day
        GROUP BY actor_id
        HAVING n_models > 1
    """).fetchall()
    report["pricing_model_changers"] = len(pricing_flips)
    logger.info("Actors with pricing model changes: %d", len(pricing_flips))

    # 1.3 Missing data audit
    missing = con.execute("""
        SELECT crawl_date,
               COUNT(*) AS n,
               SUM(CASE WHEN monthly_users IS NULL THEN 1 ELSE 0 END) AS null_monthly_users,
               SUM(CASE WHEN runs_30d_total IS NULL THEN 1 ELSE 0 END) AS null_runs_30d,
               SUM(CASE WHEN rating IS NULL THEN 1 ELSE 0 END) AS null_rating,
               SUM(CASE WHEN total_builds IS NULL THEN 1 ELSE 0 END) AS null_builds,
               SUM(CASE WHEN categories IS NULL THEN 1 ELSE 0 END) AS null_categories
        FROM actor_day
        GROUP BY crawl_date
        ORDER BY crawl_date
    """).fetchall()

    missing_report = []
    for row in missing:
        entry = {
            "date": str(row[0]),
            "n": row[1],
            "null_monthly_users": row[2],
            "null_runs_30d": row[3],
            "null_rating": row[4],
            "null_builds": row[5],
            "null_categories": row[6],
        }
        missing_report.append(entry)
        nulls = {k: v for k, v in entry.items() if k.startswith("null_") and v > 0}
        if nulls:
            logger.info("  %s: %s", row[0], nulls)
    report["missing_data"] = missing_report

    # Pricing event detail coverage
    ppe_no_events = con.execute("""
        SELECT COUNT(*) FROM actor_day a
        WHERE a.crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
          AND a.pricing_model = 'PAY_PER_EVENT'
          AND a.actor_id NOT IN (
              SELECT DISTINCT actor_id FROM pricing_event_detail
              WHERE crawl_date = a.crawl_date
          )
    """).fetchone()[0]
    report["ppe_actors_missing_event_detail"] = ppe_no_events
    logger.info("PPE actors with no event pricing detail: %d", ppe_no_events)

    # Summary verdict
    issues = []
    if report["duplicate_rows"] > 0:
        issues.append(f"{report['duplicate_rows']} duplicate rows")
    if report["agentic_flag_flippers"] > 0:
        issues.append(f"{report['agentic_flag_flippers']} actors with unstable agentic flag")
    if report["coverage_pct"] < 95:
        issues.append(f"Low panel coverage: {report['coverage_pct']}%")

    report["verdict"] = "PASS" if not issues else "ISSUES"
    report["issues"] = issues
    logger.info("Verdict: %s %s", report["verdict"], issues if issues else "")

    return report


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        report = run(con)
        out_path = OUT_DIR / "block1_quality_report.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Report saved to %s", out_path)
    finally:
        con.close()


if __name__ == "__main__":
    main()
