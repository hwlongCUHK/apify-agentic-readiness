#!/usr/bin/env python3
"""Normalize a raw daily snapshot (data/raw/{date}.jsonl.gz) into DuckDB tables.

Reads the raw archive line-by-line (each line = one full actor JSON), flattens
the nested `stats` + `currentPricingInfo` objects into the normalized tables,
and recomputes the per-category daily aggregates.

Usage:
    python3 normalize.py --date 2026-08-18
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import duckdb

from config import DB_PATH, RAW_DIR, regime_of, KNOWN_PRICING_MODELS

# Column orders must match schema.sql.
ACTOR_DAY_COLS = [
    "crawl_date", "actor_id", "username", "name", "title", "url",
    "pricing_regime", "pricing_model", "total_users", "monthly_users",
    "users_7d", "users_90d", "total_runs", "runs_30d_total",
    "runs_30d_succeeded", "runs_30d_failed", "runs_30d_aborted",
    "runs_30d_timedout", "rating", "review_count", "bookmarks",
    "total_builds", "last_run_started_at", "categories",
    "is_agentic_payments_whitelisted", "notice", "description", "description_hash",
]
PRICING_DAY_COLS = [
    "crawl_date", "actor_id", "pricing_regime", "pricing_model",
    "apify_margin_pct", "price_per_unit_usd", "trial_minutes",
    "minimal_max_total_charge_usd", "pricing_created_at", "pricing_raw",
]
EVENT_COLS = ["crawl_date", "actor_id", "event_key", "event_name", "event_price_usd",
              "is_primary", "is_one_time", "event_description", "event_raw"]
STATIC_COLS = [
    "actor_id", "username", "name", "title", "url", "developer_full_name",
    "picture_url", "categories", "is_agentic_payments_whitelisted", "ai_native",
    "agent_native", "mcp_compatible", "vertical", "requires_llm", "llm_provider",
    "input_complexity", "output_type", "classification_version", "first_seen_date",
    "is_runnable",
]
CATEGORY_COLS = [
    "crawl_date", "category", "num_products", "share_ppe", "num_rental",
    "num_usage_free", "num_ppr", "entries", "exits", "avg_price_per_unit_usd",
    "avg_rating", "avg_monthly_users", "hhi_users",
]


# --- small helpers ----------------------------------------------------------

def _g(obj, *path, default=None):
    """Deep get with a default; returns default if any hop is not a dict."""
    cur = obj
    for k in path:
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            return default
    return cur if cur is not None else default


def _num(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _ts(v):
    """Pass through ISO timestamp strings, else None (let DuckDB cast to TIMESTAMP)."""
    return v if isinstance(v, str) and v[:1].isdigit() else None


def _bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "t")
    return None


def _categories_names(cats) -> list[str]:
    names = []
    for c in cats or []:
        if isinstance(c, dict):
            names.append(str(c.get("name") or c.get("id") or c))
        else:
            names.append(str(c))
    return names


def _runs_get(runs30, key: str):
    """Look up a publicActorRunStats30Days key, normalizing key casing/punctuation."""
    if not isinstance(runs30, dict):
        return None
    norm = {k.lower().replace("-", "").replace("_", ""): v for k, v in runs30.items()}
    return norm.get(key)


def parse_pricing_per_event(ppe) -> list[tuple]:
    """Flatten currentPricingInfo.pricingPerEvent into event-detail rows.

    Real shape (verified 2026-08-18):
        {"actorChargeEvents": {"<eventKey>": {
            "eventTitle": ..., "eventDescription": ...,
            "isOneTimeEvent": bool, "eventPriceUsd": float, "isPrimaryEvent": bool?}}}

    Returns list of (event_key, event_name, price_usd, is_primary, is_one_time, description, raw).
    """
    rows: list[tuple] = []
    if not isinstance(ppe, dict):
        return rows
    charge_events = ppe.get("actorChargeEvents") or {}
    if not isinstance(charge_events, dict):
        return rows
    for key, obj in charge_events.items():
        if not isinstance(obj, dict):
            continue
        rows.append((
            str(key),
            obj.get("eventTitle"),
            _num(obj.get("eventPriceUsd")),
            _bool(obj.get("isPrimaryEvent")),
            _bool(obj.get("isOneTimeEvent")),
            obj.get("eventDescription"),
            json.dumps(obj),
        ))
    return rows


# --- flatten ----------------------------------------------------------------

def flatten_actor(item: dict, crawl_date: str) -> dict:
    stats = item.get("stats") or {}
    runs30 = stats.get("publicActorRunStats30Days") or {}
    cpi = item.get("currentPricingInfo") or {}

    username = item.get("username")
    name = item.get("name")
    actor_id = item.get("id") or item.get("actorId")
    if not actor_id:
        actor_id = hashlib.sha256(
            f"{username}|{name}|{item.get('url') or ''}".encode("utf-8")
        ).hexdigest()[:16]
    model = _g(cpi, "pricingModel")
    regime = regime_of(model)
    categories_json = json.dumps(_categories_names(item.get("categories"))) if item.get("categories") else None
    description = item.get("description") or ""
    desc_hash = hashlib.md5(description.encode("utf-8")).hexdigest()

    # rating / bookmarks: fall back to the top-level field only when the stats
    # value is genuinely missing (None), NOT when it is a falsy 0.
    rating_val = _g(stats, "actorReviewRating")
    if rating_val is None:
        rating_val = _g(item, "actorReviewRating")
    bookmark_val = _g(stats, "bookmarkCount")
    if bookmark_val is None:
        bookmark_val = _g(item, "bookmarkCount")

    actor_day = (
        crawl_date, actor_id, username, name, item.get("title"), item.get("url"),
        regime, model,
        _num(_g(stats, "totalUsers")),
        _num(_g(stats, "totalUsers30Days")),
        _num(_g(stats, "totalUsers7Days")),
        _num(_g(stats, "totalUsers90Days")),
        _num(_g(stats, "totalRuns")),
        _num(_runs_get(runs30, "total")),
        _num(_runs_get(runs30, "succeeded")),
        _num(_runs_get(runs30, "failed")),
        _num(_runs_get(runs30, "aborted")),
        _num(_runs_get(runs30, "timedout")),
        _num(rating_val),
        _num(_g(stats, "actorReviewCount")),
        _num(bookmark_val),
        _num(_g(stats, "totalBuilds")),
        _ts(_g(stats, "lastRunStartedAt")),
        categories_json,
        _bool(item.get("isWhiteListedForAgenticPayments")),
        item.get("notice"),
        description,
        desc_hash,
    )

    pricing_day = (
        crawl_date, actor_id, regime, model,
        _num(_g(cpi, "apifyMarginPercentage")),
        _num(_g(cpi, "pricePerUnitUsd")),
        _num(_g(cpi, "trialMinutes")),
        _num(_g(cpi, "minimalMaxTotalChargeUsd")),
        _ts(_g(cpi, "createdAt")),
        json.dumps(cpi) if cpi else None,
    )

    events = []
    for ev_key, ev_name, ev_price, ev_primary, ev_one_time, ev_desc, ev_raw in \
            parse_pricing_per_event(_g(cpi, "pricingPerEvent")):
        events.append((crawl_date, actor_id, ev_key, ev_name, ev_price,
                       ev_primary, ev_one_time, ev_desc, ev_raw))

    static = (
        actor_id, username, name, item.get("title"), item.get("url"),
        item.get("userFullName"), item.get("pictureUrl"),
        categories_json,
        _bool(item.get("isWhiteListedForAgenticPayments")),
        None, None, None, None, None, None, None, None, None,  # classification (filled later)
        crawl_date,
        None,  # is_runnable: not exposed in the list payload
    )

    return {"actor_day": actor_day, "pricing_day": pricing_day, "events": events, "static": static}


# --- db ---------------------------------------------------------------------

def connect(db_path: Path = DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    conn.execute((Path(__file__).parent / "schema.sql").read_text())
    return conn


# actor_static fields that may legitimately change over time and should be
# refreshed on re-normalize (title/categories/url etc.). first_seen_date and the
# researcher classification columns are intentionally left untouched on conflict.
STATIC_MUTABLE_COLS = [
    "username", "name", "title", "url", "developer_full_name",
    "picture_url", "categories", "is_agentic_payments_whitelisted",
]


def _upsert_static(conn, rows: list[tuple]) -> None:
    """Upsert actor_static: insert new actors, refresh mutable metadata on conflict."""
    if not rows:
        return
    ph = ",".join(["?"] * len(STATIC_COLS))
    updates = ",".join(f"{c}=excluded.{c}" for c in STATIC_MUTABLE_COLS)
    sql = (f"INSERT INTO actor_static ({','.join(STATIC_COLS)}) VALUES ({ph}) "
           f"ON CONFLICT (actor_id) DO UPDATE SET {updates}")
    conn.executemany(sql, rows)


def _insert(conn, table: str, cols: list[str], rows: list[tuple]) -> None:
    if not rows:
        return
    ph = ",".join(["?"] * len(cols))
    sql = f"INSERT INTO {table} ({','.join(cols)}) VALUES ({ph})"
    conn.executemany(sql, rows)


def _latest_prev_date(conn, crawl_date: str):
    r = conn.execute(
        "SELECT MAX(crawl_date) FROM actor_day WHERE crawl_date < ?", [crawl_date]
    ).fetchone()
    return r[0] if r else None


# --- category_day aggregation ------------------------------------------------

def compute_category_day(conn, crawl_date: str, prev_date: str | None) -> list[tuple]:
    today = conn.execute(
        "SELECT actor_id, categories, pricing_regime, monthly_users, rating "
        "FROM actor_day WHERE crawl_date = ?", [crawl_date]).fetchall()
    prices = {r[0]: r[1] for r in conn.execute(
        "SELECT actor_id, price_per_unit_usd FROM pricing_day WHERE crawl_date = ?", [crawl_date]).fetchall()}

    def expand(rows):
        cat_actors: dict[str, set] = defaultdict(set)
        for actor_id, cats_json, *_ in rows:
            names = json.loads(cats_json) if cats_json else ["(uncategorized)"]
            for c in names:
                cat_actors[c].add(actor_id)
        return cat_actors

    today_cat = expand(today)
    prev_cat: dict[str, set] = {}
    if prev_date:
        prev_rows = conn.execute(
            "SELECT actor_id, categories FROM actor_day WHERE crawl_date = ?", [prev_date]).fetchall()
        prev_cat = expand(prev_rows)

    agg = defaultdict(lambda: {
        "count": 0, "ppe": 0, "rental": 0, "usage_free": 0, "ppr": 0,
        "sum_price": 0.0, "n_price": 0, "sum_rating": 0.0, "n_rating": 0,
        "sum_users": 0.0, "n_users": 0, "sq_users": 0.0, "tot_users": 0.0,
    })
    for actor_id, cats_json, regime, monthly_users, rating in today:
        names = json.loads(cats_json) if cats_json else ["(uncategorized)"]
        for c in names:
            a = agg[c]
            a["count"] += 1
            if regime == 1:
                a["ppe"] += 1
            elif regime == 0:
                a["rental"] += 1
            elif regime == 2:
                a["ppr"] += 1
            elif regime == 3:
                a["usage_free"] += 1
            # avg_price_per_unit_usd only makes sense for rental (regime 0):
            # it is the monthly fee. PPE/PPR per-event prices live in
            # pricing_event_detail and must not be mixed in here.
            if regime == 0:
                p = prices.get(actor_id)
                if p is not None:
                    a["sum_price"] += p
                    a["n_price"] += 1
            if rating is not None:
                a["sum_rating"] += rating
                a["n_rating"] += 1
            if monthly_users is not None:
                a["sum_users"] += monthly_users
                a["n_users"] += 1
                a["sq_users"] += monthly_users ** 2
                a["tot_users"] += monthly_users

    rows = []
    for c, a in sorted(agg.items()):
        n = a["count"] or 1
        share_ppe = a["ppe"] / n
        avg_price = (a["sum_price"] / a["n_price"]) if a["n_price"] else None
        avg_rating = (a["sum_rating"] / a["n_rating"]) if a["n_rating"] else None
        avg_users = (a["sum_users"] / a["n_users"]) if a["n_users"] else None
        hhi = (a["sq_users"] / (a["tot_users"] ** 2) * 10000) if a["tot_users"] > 0 else None
        entries = len(today_cat.get(c, set()) - prev_cat.get(c, set())) if prev_date else None
        exits = len(prev_cat.get(c, set()) - today_cat.get(c, set())) if prev_date else None
        rows.append((crawl_date, c, a["count"], share_ppe, a["rental"], a["usage_free"],
                     a["ppr"], entries, exits, avg_price, avg_rating, avg_users, hhi))
    return rows


# --- main -------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize a raw snapshot into DuckDB")
    ap.add_argument("--date", default=None, help="crawl date YYYY-MM-DD (default: today)")
    ap.add_argument("--db", default=str(DB_PATH))
    args = ap.parse_args()

    crawl_date = args.date or date.today().isoformat()
    raw_path = RAW_DIR / f"{crawl_date}.jsonl.gz"
    if not raw_path.exists():
        sys.exit(f"[normalize] raw snapshot not found: {raw_path}")

    conn = connect(Path(args.db))

    actor_day_rows, pricing_day_rows, event_rows, static_rows = [], [], [], []
    unknown_models: set[str] = set()
    bad_lines = 0
    with gzip.open(raw_path, "rt", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                bad_lines += 1
                print(f"[normalize] WARNING: bad JSON at {raw_path}:{lineno}: {e}",
                      file=sys.stderr)
                continue
            model = _g(item.get("currentPricingInfo") or {}, "pricingModel")
            if model and str(model).upper() not in KNOWN_PRICING_MODELS:
                unknown_models.add(str(model))
            r = flatten_actor(item, crawl_date)
            actor_day_rows.append(r["actor_day"])
            pricing_day_rows.append(r["pricing_day"])
            event_rows.extend(r["events"])
            static_rows.append(r["static"])

    if unknown_models:
        print(f"[normalize] WARNING: {len(unknown_models)} unknown pricing model(s): "
              f"{sorted(unknown_models)}", file=sys.stderr)
    if bad_lines:
        print(f"[normalize] WARNING: {bad_lines} line(s) skipped due to bad JSON", file=sys.stderr)

    # delete-reload daily tables so re-normalizing a date is idempotent,
    # wrapped in a single transaction (all-or-nothing on failure).
    conn.execute("BEGIN TRANSACTION")
    try:
        for tbl in ("actor_day", "pricing_day", "pricing_event_detail"):
            conn.execute(f"DELETE FROM {tbl} WHERE crawl_date = ?", [crawl_date])
        _insert(conn, "actor_day", ACTOR_DAY_COLS, actor_day_rows)
        _insert(conn, "pricing_day", PRICING_DAY_COLS, pricing_day_rows)
        _insert(conn, "pricing_event_detail", EVENT_COLS, event_rows)
        _upsert_static(conn, static_rows)

        # snapshot_manifest (from fetch.py's manifest if present, else minimal)
        manifest_path = RAW_DIR / f"{crawl_date}.manifest.json"
        conn.execute("DELETE FROM snapshot_manifest WHERE crawl_date = ?", [crawl_date])
        if manifest_path.exists():
            m = json.loads(manifest_path.read_text())
            conn.execute(
                "INSERT INTO snapshot_manifest VALUES (?,?,?,?,?,?,?)",
                [crawl_date, m.get("raw_file"), m.get("actor_count"), m.get("n_usernames_processed"),
                 m.get("n_errors"), m.get("started_at"), m.get("finished_at")])
        else:
            conn.execute(
                "INSERT INTO snapshot_manifest VALUES (?,?,?,?,?,?,?)",
                [crawl_date, str(raw_path), len(actor_day_rows), None, None, None, None])

        # category_day (full recompute for this date)
        prev_date = _latest_prev_date(conn, crawl_date)
        conn.execute("DELETE FROM category_day WHERE crawl_date = ?", [crawl_date])
        _insert(conn, "category_day", CATEGORY_COLS, compute_category_day(conn, crawl_date, prev_date))

        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    conn.close()
    print(f"[normalize] DONE: {len(actor_day_rows)} actors -> {args.db} "
          f"(prev_date={prev_date}, {len(event_rows)} event rows)")


if __name__ == "__main__":
    main()
