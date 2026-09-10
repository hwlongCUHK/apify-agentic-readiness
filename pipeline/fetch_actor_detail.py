#!/usr/bin/env python3
"""Fetch actor detail (permission level + standby) for all PPE actors.

Detail endpoint GET /v2/acts/{actorId} returns `actorPermissionLevel` and
`actorStandby`/`standbyUrl`, which are the agentic-payment eligibility
determinants that the store listing API omits. This backfills them for every
PPE actor so we can (a) validate the eligibility rule and (b) add them to the
PSM propensity model and FE controls.

Rate limit ~1 req/s (server caps 60/min). Full 51,861-actor sweep ~14-15h.
Resumable: appends to the day's JSONL, skipping actorIds already present.

Usage:
    python3 fetch_actor_detail.py --date 2026-08-23
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb
import requests

from config import (
    USER_AGENT, REQUEST_TIMEOUT, REQUEST_INTERVAL, MAX_RETRIES,
    RETRY_BACKOFF_BASE,
)

# The analysis DB and raw archive live under CHI/ (the paper project), not the
# pipeline root (which holds an older, pre-08-23 DB). Point at the live copy.
DB_PATH = Path(__file__).resolve().parent / "CHI" / "data" / "apify_panel.duckdb"
RAW_DIR = Path(__file__).resolve().parent / "CHI" / "data" / "raw"

DETAIL_API = "https://api.apify.com/v2/acts"


class RateLimiter:
    """Steady-state rate limiter (same as fetch.py)."""

    def __init__(self, interval: float = REQUEST_INTERVAL):
        self.interval = interval
        self._last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        delay = self._last + self.interval - now
        if delay > 0:
            time.sleep(delay)
        self._last = time.monotonic()


def get_actor_detail(session: requests.Session, actor_id: str, limiter: RateLimiter) -> dict | None:
    """Return a compact record for one actor, or None on permanent failure."""
    last_err = None
    for attempt in range(MAX_RETRIES):
        limiter.wait()
        try:
            r = session.get(f"{DETAIL_API}/{actor_id}",
                            headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
            if r.status_code == 429:
                try:
                    retry_after = float(r.headers.get("Retry-After", RETRY_BACKOFF_BASE ** attempt))
                except (TypeError, ValueError):
                    retry_after = float(RETRY_BACKOFF_BASE ** attempt)
                time.sleep(max(retry_after, 1.0))
                last_err = "429"
                continue
            if r.status_code == 404:
                return {"actorId": actor_id, "status": 404,
                        "actorPermissionLevel": None, "standbyEnabled": None}
            r.raise_for_status()
            d = r.json().get("data", r.json())
            standby = d.get("actorStandby")
            standby_on = (standby.get("isEnabled") if isinstance(standby, dict)
                          else bool(d.get("standbyUrl")))
            return {
                "actorId": actor_id,
                "status": 200,
                "actorPermissionLevel": d.get("actorPermissionLevel"),
                "standbyEnabled": standby_on,
            }
        except requests.RequestException as e:
            last_err = repr(e)
            time.sleep(RETRY_BACKOFF_BASE ** attempt)
    return {"actorId": actor_id, "status": "ERROR", "error": last_err,
            "actorPermissionLevel": None, "standbyEnabled": None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="crawl date YYYY-MM-DD (matches snapshot)")
    args = ap.parse_args()

    con = duckdb.connect(str(DB_PATH), read_only=True)
    rows = con.execute("""
        SELECT DISTINCT actor_id
        FROM actor_day
        WHERE crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
          AND pricing_model = 'PAY_PER_EVENT'
        ORDER BY actor_id
    """).fetchall()
    ids = [r[0] for r in rows]
    print(f"[fetch_actor_detail] PPE actors: {len(ids)}")

    out_path = RAW_DIR / f"{args.date}.actor_detail.jsonl"
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["actorId"])
                except Exception:
                    pass
    print(f"[fetch_actor_detail] already fetched: {len(done)}; remaining: {len(ids) - len(done)}")

    session = requests.Session()
    limiter = RateLimiter()
    n_fetched = 0
    t0 = time.time()
    with open(out_path, "a") as f:
        for i, aid in enumerate(ids):
            if aid in done:
                continue
            rec = get_actor_detail(session, aid, limiter)
            if rec is not None:
                f.write(json.dumps(rec) + "\n")
                f.flush()
            n_fetched += 1
            if n_fetched % 200 == 0:
                el = time.time() - t0
                rate = n_fetched / el if el > 0 else 0
                print(f"  fetched {n_fetched}/{len(ids) - len(done)} "
                      f"({rate:.2f} act/s, elapsed {el/60:.1f} min)", flush=True)

    print(f"[fetch_actor_detail] done. total fetched this run: {n_fetched}")
    print(f"[fetch_actor_detail] output: {out_path}")


if __name__ == "__main__":
    main()
