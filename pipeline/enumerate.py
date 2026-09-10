#!/usr/bin/env python3
"""Discover the full set of developer usernames from Apify's public sitemaps.

Why username-partitioning (see DESIGN.md §6):
    The /v2/store list endpoint silently returns 0 items past offset ~15-16k,
    so you cannot page the whole ~61k catalog linearly. Instead we enumerate the
    ~5,334 unique developer usernames from the sitemaps, then issue one
    `?username=...&limit=1000` call per developer in fetch.py — provably complete
    and well under the rate limit.

Usage:
    python3 enumerate.py --out data/usernames.json
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlsplit

import requests

from config import SITEMAP_INDEX, USER_AGENT, REQUEST_TIMEOUT

NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

# First path segments that never name a developer.
RESERVED_FIRST = {
    "api", "store", "users", "docs", "blog", "change-log", "help",
    "academy", "pricing", "templates", "actors", "platform", "about",
}


def _get(url: str) -> str:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.text


def _locs(xml_text: str) -> list[str]:
    """Return all <loc> texts (recursively), for both index (<sitemap><loc>) and leaf (<url><loc>)."""
    root = ET.fromstring(xml_text)
    return [e.text.strip() for e in root.findall(".//sm:loc", NS) if e.text and e.text.strip()]


def _usernames_from_paths(locs: list[str]) -> set[str]:
    """Extract developer usernames from actor URLs and /users/{name} URLs."""
    out: set[str] = set()
    for loc in locs:
        parts = [p for p in urlsplit(loc).path.split("/") if p]
        if len(parts) != 2:
            continue
        first, second = parts
        if first == "users":
            out.add(second)
        elif first.lower() not in RESERVED_FIRST and second != "api":
            out.add(first)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Enumerate Apify developer usernames from sitemaps")
    ap.add_argument("--out", default="data/usernames.json")
    ap.add_argument("--max-sitemaps", type=int, default=0,
                    help="limit child sitemaps fetched (0 = all); for testing")
    args = ap.parse_args()

    index_xml = _get(SITEMAP_INDEX)
    children = _locs(index_xml)
    actor_children = [u for u in children if "actors" in urlsplit(u).path.lower()]
    users_children = [u for u in children if "users" in urlsplit(u).path.lower()]
    todo = actor_children + users_children
    if args.max_sitemaps:
        todo = todo[: args.max_sitemaps]

    usernames: set[str] = set()
    for u in todo:
        try:
            locs = _locs(_get(u))
        except Exception as e:  # noqa: BLE001 - keep going on one bad child
            print(f"[warn] {u}: {e}", file=sys.stderr)
            continue
        usernames |= _usernames_from_paths(locs)
        print(f"[enumerate] {u}: {len(usernames)} usernames so far")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sorted(usernames), indent=2))
    print(f"[enumerate] DONE: {len(usernames)} unique usernames -> {out}")


if __name__ == "__main__":
    main()
