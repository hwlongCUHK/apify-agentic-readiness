#!/usr/bin/env python3
"""Fetch the Apify Store listing API (anonymous) and archive raw responses.

Strategy: read the username list produced by enumerate.py, then for each
developer issue paginated
    GET /v2/store?username=...&limit=1000&offset=...&includeUnrunnableActors=true
until the envelope `total` is reached (or a short page is returned). This avoids
the silent truncation that occurs when a developer has >1000 actors.

Each actor's full JSON (stats + currentPricingInfo) is appended as one line to
the day's gzipped JSONL archive: data/raw/{date}.jsonl.gz. The archive is
written to a temp file and atomically renamed on success.

Rate limit: the server caps at 60 req/min (x-ratelimit-limit). We default to
~1 req/s (REQUEST_INTERVAL=1.0), so a full 5,334-username sweep takes ~1.5-2.5h.

Usage:
    python3 fetch.py --date 2026-08-18 --usernames data/usernames.json
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import requests

from config import (
    STORE_API, USER_AGENT, REQUEST_TIMEOUT, RATE_LIMIT_PER_MINUTE,
    REQUEST_INTERVAL, MAX_RETRIES, RETRY_BACKOFF_BASE, FETCH_LIMIT,
    MAX_PAGES_PER_USER, INCLUDE_UNRUNNABLE, USERNAMES_FILE, RAW_DIR,
)


class RateLimiter:
    """Steady-state rate limiter: sleep so consecutive calls are >= `interval` apart."""

    def __init__(self, interval: float = REQUEST_INTERVAL,
                 max_per_min: int = RATE_LIMIT_PER_MINUTE):
        self.interval = interval
        self.max_per_min = max_per_min
        self._last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        delay = self._last + self.interval - now
        if delay > 0:
            time.sleep(delay)
        self._last = time.monotonic()


def _get_json(session: requests.Session, params: dict, limiter: RateLimiter) -> dict:
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        limiter.wait()
        try:
            r = session.get(STORE_API, params=params,
                            headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
            if r.status_code == 429:
                try:
                    retry_after = float(r.headers.get("Retry-After", RETRY_BACKOFF_BASE ** attempt))
                except (TypeError, ValueError):
                    retry_after = float(RETRY_BACKOFF_BASE ** attempt)
                time.sleep(max(retry_after, 1.0))
                last_err = RuntimeError("429 rate limited")
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last_err = e
            time.sleep(RETRY_BACKOFF_BASE ** attempt)
    raise RuntimeError(f"failed after {MAX_RETRIES} retries: {last_err}")


def _envelope(payload) -> dict:
    """Return the data envelope (payload['data'] if it is a dict, else payload)."""
    if not isinstance(payload, dict):
        return {}
    data = payload.get("data", payload)
    return data if isinstance(data, dict) else payload


def fetch_user_actors(session: requests.Session, username: str, limiter: RateLimiter):
    """Fetch ALL actors for a username, paginating past the 1000-item page cap.

    Returns (items, total, n_pages). `total` is the envelope-reported total; a
    shortfall (len(items) < total) signals truncation to the caller.
    """
    all_items: list[dict] = []
    offset = 0
    pages = 0
    total = None
    while pages < MAX_PAGES_PER_USER:
        params = {
            "username": username,
            "limit": FETCH_LIMIT,
            "offset": offset,
            "includeUnrunnableActors": str(INCLUDE_UNRUNNABLE).lower(),
        }
        payload = _get_json(session, params, limiter)
        env = _envelope(payload)
        items = env.get("items")
        items = items if isinstance(items, list) else []
        total_raw = env.get("total")
        try:
            total = int(total_raw) if total_raw is not None else None
        except (TypeError, ValueError):
            total = None
        all_items.extend(items)
        pages += 1
        if total is not None and len(all_items) >= total:
            break
        if len(items) < FETCH_LIMIT:
            break
        offset += len(items)
    return all_items, total, pages


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch Apify Store actors into a raw JSONL.gz archive")
    ap.add_argument("--usernames", default=str(USERNAMES_FILE))
    ap.add_argument("--raw-dir", default=str(RAW_DIR))
    ap.add_argument("--date", default=None, help="crawl date YYYY-MM-DD (default: today)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N usernames (testing)")
    args = ap.parse_args()

    crawl_date = args.date or date.today().isoformat()
    usernames: list[str] = json.loads(Path(args.usernames).read_text())
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / f"{crawl_date}.jsonl.gz"

    # archive the exact username list used, plus its hash, for provenance
    usernames_json = json.dumps(sorted(usernames), indent=2)
    usernames_hash = hashlib.sha256(usernames_json.encode("utf-8")).hexdigest()
    (raw_dir / f"{crawl_date}.usernames.json").write_text(usernames_json)

    session = requests.Session()
    limiter = RateLimiter()

    started = time.time()
    started_at = datetime.now(timezone.utc).isoformat()

    n_actors = 0
    n_errors = 0
    n_processed = 0
    n_truncated = 0          # usernames where fetched < reported total
    n_empty = 0              # usernames returning 0 actors
    audit_rows: list[dict] = []

    # write to a temp file, atomically rename on success (no partial archive)
    tmp_path = raw_dir / f"{crawl_date}.jsonl.gz.tmp"
    with gzip.open(tmp_path, "wt", encoding="utf-8") as fh:
        for i, username in enumerate(usernames):
            if args.limit and i >= args.limit:
                break
            try:
                items, total, pages = fetch_user_actors(session, username, limiter)
            except Exception as e:  # noqa: BLE001 - log and continue on one bad dev
                n_errors += 1
                print(f"[fetch] ERROR username={username}: {e}", file=sys.stderr)
                audit_rows.append({"username": username, "status": "error", "error": str(e)})
                continue
            for item in items:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
            n_actors += len(items)
            n_processed += 1
            if total is not None and len(items) < total:
                n_truncated += 1
                print(f"[fetch] WARNING username={username}: got {len(items)} < total {total}",
                      file=sys.stderr)
            if len(items) == 0:
                n_empty += 1
            audit_rows.append({
                "username": username, "status": "ok", "items": len(items),
                "total": total, "pages": pages,
            })
            if n_processed % 200 == 0:
                print(f"[fetch] {n_processed}/{len(usernames)} usernames, "
                      f"{n_actors} actors, {n_errors} errors")

    os.replace(tmp_path, out_path)  # atomic: only publish a complete archive

    finished_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "crawl_date": crawl_date,
        "raw_file": str(out_path),
        "actor_count": n_actors,
        "n_usernames_processed": n_processed,
        "n_errors": n_errors,
        "n_truncated": n_truncated,
        "n_empty": n_empty,
        "usernames_hash": usernames_hash,
        "usernames_file": str(raw_dir / f"{crawl_date}.usernames.json"),
        "started_at": started_at,
        "finished_at": finished_at,
        "seconds": round(time.time() - started, 1),
    }
    (raw_dir / f"{crawl_date}.manifest.json").write_text(json.dumps(manifest, indent=2))

    # per-username audit (status / item count / total / pages) for coverage checks
    audit_path = raw_dir / f"{crawl_date}.audit.jsonl"
    with open(audit_path, "w", encoding="utf-8") as af:
        for row in audit_rows:
            af.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[fetch] DONE: {n_actors} actors -> {out_path} "
          f"({manifest['seconds']}s, {n_errors} errors, {n_truncated} truncated, "
          f"{n_empty} empty)")


if __name__ == "__main__":
    main()
