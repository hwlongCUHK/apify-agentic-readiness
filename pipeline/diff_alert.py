#!/usr/bin/env python3
"""Compare today's snapshot vs the previous one; detect events and emit alerts.

Detects (and persists into `pricing_events` where applicable):
  1. pricing regime changes (rental -> PPE / pay-per-usage / PPR -> PPE, ...)
  2. actor entries (new today) and exits (gone today)
  3. large monthly-user jumps (default: |Δ|/prev > 50%)
  4. description changes (hash mismatch)

Alerts are written to data/alerts/{cur}.json.

Usage:
    python3 diff_alert.py --cur 2026-08-18
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import DB_PATH, DATA_DIR

JUMP_THRESHOLD = 0.5        # relative: flag |delta|/prev > 50%
MIN_PREV_USERS = 10         # absolute: ignore actors with tiny denominators
MIN_ABS_DELTA = 10          # absolute: ignore trivial month-over-month changes


def classify_switch(old_model: str | None, new_model: str | None) -> str:
    o = (old_model or "").upper()
    n = (new_model or "").upper()
    if o == "FLAT_PRICE_PER_MONTH" and n == "PAY_PER_EVENT":
        return "platform_to_ppe"
    if o == "FLAT_PRICE_PER_MONTH" and n == "FREE":
        return "platform_to_usage"
    if o == "PRICE_PER_DATASET_ITEM" and n == "PAY_PER_EVENT":
        return "ppr_to_ppe"
    if o == "FREE" and n == "PAY_PER_EVENT":
        return "voluntary_to_ppe"
    if o == "":
        return f"born_{n.lower() or 'unknown'}"
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description="Diff two crawls and emit alerts")
    ap.add_argument("--cur", default=None, help="current crawl date (default: latest in db)")
    ap.add_argument("--prev", default=None, help="previous crawl date (default: latest < cur)")
    ap.add_argument("--db", default=str(DB_PATH))
    args = ap.parse_args()

    conn = __import__("duckdb").connect(str(Path(args.db)))
    conn.execute((Path(__file__).parent / "schema.sql").read_text())

    cur = args.cur
    if cur is None:
        r = conn.execute("SELECT MAX(crawl_date) FROM actor_day").fetchone()
        cur = str(r[0]) if r and r[0] else None
    if cur is None:
        print("[diff_alert] no data in db yet", file=__import__("sys").stderr)
        return

    prev = args.prev
    if prev is None:
        r = conn.execute(
            "SELECT MAX(crawl_date) FROM actor_day WHERE crawl_date < ?", [cur]).fetchone()
        prev = str(r[0]) if r and r[0] else None

    alerts: dict = {"cur": cur, "prev": prev, "regime_changes": [], "entries": [],
                    "exits": [], "user_jumps": [], "description_changes": []}

    # idempotent re-diff: recompute this date's events from scratch
    conn.execute("DELETE FROM pricing_events WHERE detected_date = ?", [cur])

    # 1) regime changes
    changes = conn.execute(
        """
        SELECT a.actor_id, a.username, a.name,
               b.pricing_regime, a.pricing_regime, b.pricing_model, a.pricing_model
        FROM actor_day a
        JOIN actor_day b ON a.actor_id = b.actor_id
        WHERE a.crawl_date = ? AND b.crawl_date = ?
          AND a.pricing_regime IS DISTINCT FROM b.pricing_regime
        """, [cur, prev]).fetchall() if prev else []

    for actor_id, username, name, old_r, new_r, old_m, new_m in changes:
        reason = classify_switch(old_m, new_m)
        conn.execute(
            """
            INSERT INTO pricing_events
              (event_id, actor_id, username, name, detected_date, prev_date,
               old_regime, new_regime, old_model, new_model, switch_date, switch_reason, evidence)
            VALUES (nextval('pricing_event_seq'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [actor_id, username, name, cur, prev, old_r, new_r, old_m, new_m,
             cur, reason, f"regime {old_m} -> {new_m} observed between {prev} and {cur}"])
        alerts["regime_changes"].append(
            {"actor_id": actor_id, "username": username, "name": name,
             "old_model": old_m, "new_model": new_m, "switch_reason": reason})

    # 2) entries / exits
    if prev:
        entries = conn.execute(
            "SELECT actor_id, username, name FROM actor_day WHERE crawl_date = ? "
            "AND actor_id NOT IN (SELECT actor_id FROM actor_day WHERE crawl_date = ?)",
            [cur, prev]).fetchall()
        exits = conn.execute(
            "SELECT actor_id, username, name FROM actor_day WHERE crawl_date = ? "
            "AND actor_id NOT IN (SELECT actor_id FROM actor_day WHERE crawl_date = ?)",
            [prev, cur]).fetchall()
        alerts["entries"] = [{"actor_id": a, "username": u, "name": n} for a, u, n in entries]
        alerts["exits"] = [{"actor_id": a, "username": u, "name": n} for a, u, n in exits]

        # 3) large monthly-user jumps
        jumps = conn.execute(
            """
            SELECT a.actor_id, a.username, a.name, b.monthly_users, a.monthly_users
            FROM actor_day a JOIN actor_day b ON a.actor_id = b.actor_id
            WHERE a.crawl_date = ? AND b.crawl_date = ?
              AND a.monthly_users IS NOT NULL AND b.monthly_users IS NOT NULL
              AND b.monthly_users >= ?
              AND ABS(a.monthly_users - b.monthly_users) >= ?
              AND ABS(a.monthly_users - b.monthly_users) / b.monthly_users > ?
            """, [cur, prev, MIN_PREV_USERS, MIN_ABS_DELTA, JUMP_THRESHOLD]).fetchall()
        alerts["user_jumps"] = [
            {"actor_id": a, "username": u, "name": n, "prev": p, "cur": c}
            for a, u, n, p, c in jumps]

        # 4) description changes
        desc_changes = conn.execute(
            """
            SELECT a.actor_id, a.username, a.name
            FROM actor_day a JOIN actor_day b ON a.actor_id = b.actor_id
            WHERE a.crawl_date = ? AND b.crawl_date = ?
              AND a.description_hash IS DISTINCT FROM b.description_hash
            """, [cur, prev]).fetchall()
        alerts["description_changes"] = [
            {"actor_id": a, "username": u, "name": n} for a, u, n in desc_changes]

    conn.close()

    alerts_dir = Path(DATA_DIR) / "alerts"
    alerts_dir.mkdir(parents=True, exist_ok=True)
    (alerts_dir / f"{cur}.json").write_text(json.dumps(alerts, indent=2))

    print(f"[diff_alert] {cur} vs {prev}: "
          f"{len(alerts['regime_changes'])} regime changes, "
          f"{len(alerts['entries'])} entries, {len(alerts['exits'])} exits, "
          f"{len(alerts['user_jumps'])} user jumps, "
          f"{len(alerts['description_changes'])} desc changes -> {alerts_dir / f'{cur}.json'}")


if __name__ == "__main__":
    main()
