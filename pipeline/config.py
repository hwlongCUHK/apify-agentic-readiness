"""Central configuration for the Apify panel pipeline.

All paths, API settings, rate limits and the pricing-regime mapping live here.
Override paths via environment variables if you move the data directory.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("APIFY_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"           # raw API snapshots: {date}.jsonl.gz
PARQUET_DIR = DATA_DIR / "parquet"   # optional columnar export
DB_PATH = Path(os.environ.get("APIFY_DB_PATH", DATA_DIR / "apify_panel.duckdb"))
USERNAMES_FILE = Path(os.environ.get("APIFY_USERNAMES_FILE", DATA_DIR / "usernames.json"))

# --- HTTP / API -------------------------------------------------------------
STORE_API = "https://api.apify.com/v2/store"
SITEMAP_INDEX = "https://apify.com/sitemap.xml"
USER_AGENT = "apify-panel-research/0.1 (academic; non-commercial)"
REQUEST_TIMEOUT = 30            # seconds per request
RATE_LIMIT_PER_MINUTE = 60      # hard server limit (x-ratelimit-limit)
REQUEST_INTERVAL = 1.0          # seconds between requests; ~1/s keeps us safe
MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 2.0        # seconds; exponential backoff base
FETCH_LIMIT = 1000              # max items per username call (server cap)
MAX_PAGES_PER_USER = 20         # defensive pagination ceiling (20 * 1000 = 20k actors/username)
INCLUDE_UNRUNNABLE = True       # include un-KYC'd / low-usage actors

# --- pricing model enum -> normalized regime code ---------------------------
# regime codes (see DESIGN.md §3.1):
#   0 = rental/subscription (fixed monthly fee + usage)
#   1 = pay-per-event (PPE)
#   2 = pay-per-result (PPR, legacy)
#   3 = pay-per-usage / free (platform usage only)
PRICING_MODEL_TO_REGIME: dict[str, int] = {
    "FLAT_PRICE_PER_MONTH": 0,
    "PAY_PER_EVENT": 1,
    "PRICE_PER_DATASET_ITEM": 2,
    "FREE": 3,
    # defensive aliases (in case the enum string varies)
    "PAY_PER_USAGE": 3,
    "MONTHLY_RENTAL": 0,
}
REGIME_LABEL: dict[int, str] = {0: "rental", 1: "ppe", 2: "ppr", 3: "usage_free"}

# All pricingModel enum strings we can map. Anything outside this set is
# flagged (warned) by normalize.py, not silently dropped.
KNOWN_PRICING_MODELS: frozenset[str] = frozenset(PRICING_MODEL_TO_REGIME.keys())


def regime_of(pricing_model: str | None) -> int | None:
    """Map an Apify pricingModel string to a normalized regime code."""
    if not pricing_model:
        return None
    return PRICING_MODEL_TO_REGIME.get(str(pricing_model).upper())
