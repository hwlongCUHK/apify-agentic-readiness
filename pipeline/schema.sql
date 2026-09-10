-- =============================================================================
-- Apify panel schema (DuckDB)
-- 5 core tables + 2 auxiliary. Raw API responses live in data/raw/{date}.jsonl.gz
-- (the raw archive IS the source of truth; these tables are normalized views).
-- Regime codes: 0 rental / 1 pay-per-event / 2 pay-per-result / 3 usage_free.
-- =============================================================================

-- (1) actor_day — one row per (actor, crawl_date). Core demand/usage panel.
CREATE TABLE IF NOT EXISTS actor_day (
  crawl_date           DATE    NOT NULL,
  actor_id             VARCHAR NOT NULL,
  username             VARCHAR,
  name                 VARCHAR,
  title                VARCHAR,
  url                  VARCHAR,
  pricing_regime       INTEGER,
  pricing_model        VARCHAR,
  total_users          BIGINT,
  monthly_users        BIGINT,      -- stats.totalUsers30Days
  users_7d             BIGINT,
  users_90d            BIGINT,
  total_runs           BIGINT,
  runs_30d_total       BIGINT,
  runs_30d_succeeded   BIGINT,
  runs_30d_failed      BIGINT,
  runs_30d_aborted     BIGINT,
  runs_30d_timedout    BIGINT,
  rating               DOUBLE,
  review_count         BIGINT,
  bookmarks            BIGINT,
  total_builds         BIGINT,      -- seller-effort proxy (update frequency)
  last_run_started_at  TIMESTAMP,
  categories           VARCHAR,     -- JSON-encoded list of category names
  is_agentic_payments_whitelisted BOOLEAN,  -- item.isWhiteListedForAgenticPayments
  notice               VARCHAR,     -- 'NONE' | 'UNDER_MAINTENANCE'
  description          VARCHAR,
  description_hash     VARCHAR,     -- md5(description), for change detection
  PRIMARY KEY (crawl_date, actor_id)
);

-- (2) pricing_day — one row per (actor, crawl_date). Pricing regime details.
CREATE TABLE IF NOT EXISTS pricing_day (
  crawl_date                 DATE    NOT NULL,
  actor_id                   VARCHAR NOT NULL,
  pricing_regime             INTEGER,
  pricing_model              VARCHAR,
  apify_margin_pct           DOUBLE,
  price_per_unit_usd         DOUBLE,   -- rental: monthly fee; PPR: per-item price
  trial_minutes              BIGINT,   -- rental free-trial period (minutes)
  minimal_max_total_charge_usd DOUBLE, -- PPE: max charge cap (NULL for rental)
  pricing_created_at         TIMESTAMP, -- currentPricingInfo.createdAt -> T_i anchor
  pricing_raw                VARCHAR,   -- full currentPricingInfo JSON
  PRIMARY KEY (crawl_date, actor_id)
);

-- (2b) pricing_event_detail — long-format per-event pricing (one row per event).
CREATE TABLE IF NOT EXISTS pricing_event_detail (
  crawl_date        DATE    NOT NULL,
  actor_id          VARCHAR NOT NULL,
  event_key         VARCHAR NOT NULL,   -- machine key, e.g. 'task-search'
  event_name        VARCHAR,            -- eventTitle, e.g. 'Task search'
  event_price_usd   DOUBLE,             -- eventPriceUsd
  is_primary        BOOLEAN,            -- isPrimaryEvent (main metered event)
  is_one_time       BOOLEAN,            -- isOneTimeEvent
  event_description VARCHAR,
  event_raw         VARCHAR,
  PRIMARY KEY (crawl_date, actor_id, event_key)
);

-- (3) actor_static — time-invariant-ish product/developer characteristics.
CREATE TABLE IF NOT EXISTS actor_static (
  actor_id             VARCHAR PRIMARY KEY,
  username             VARCHAR,
  name                 VARCHAR,
  title                VARCHAR,
  url                  VARCHAR,
  developer_full_name  VARCHAR,
  picture_url          VARCHAR,
  categories           VARCHAR,    -- JSON-encoded list of category names
  is_agentic_payments_whitelisted BOOLEAN,
  -- researcher classification (filled later, versioned; see DESIGN.md §5)
  ai_native            BOOLEAN,
  agent_native         BOOLEAN,
  mcp_compatible       BOOLEAN,
  vertical             VARCHAR,
  requires_llm         BOOLEAN,
  llm_provider         VARCHAR,
  input_complexity     VARCHAR,
  output_type          VARCHAR,
  classification_version INTEGER,
  -- provenance
  first_seen_date      DATE,
  is_runnable          BOOLEAN
);

-- (4) pricing_events — detected regime transitions (the treatment events).
CREATE SEQUENCE IF NOT EXISTS pricing_event_seq START 1;
CREATE TABLE IF NOT EXISTS pricing_events (
  event_id       BIGINT PRIMARY KEY,
  actor_id       VARCHAR,
  username       VARCHAR,
  name           VARCHAR,
  detected_date  DATE,     -- crawl date we first observed the change
  prev_date      DATE,     -- last crawl date before the change
  old_regime     INTEGER,
  new_regime     INTEGER,
  old_model      VARCHAR,
  new_model      VARCHAR,
  switch_date    DATE,     -- detection date (= detected_date); actual switch lies in (prev_date, detected_date]
  switch_reason  VARCHAR,  -- platform_to_ppe / platform_to_usage / ppr_to_ppe / voluntary_to_ppe / born_* / unknown
  evidence       VARCHAR,
  UNIQUE (actor_id, detected_date, old_model, new_model)  -- prevent dup events on re-diff
);

-- (5) category_day — per-category daily aggregates (competition outcomes).
CREATE TABLE IF NOT EXISTS category_day (
  crawl_date          DATE    NOT NULL,
  category            VARCHAR NOT NULL,
  num_products        BIGINT,
  share_ppe           DOUBLE,
  num_rental          BIGINT,
  num_usage_free      BIGINT,
  num_ppr             BIGINT,
  entries             BIGINT,   -- new actors vs previous crawl
  exits               BIGINT,
  avg_price_per_unit_usd DOUBLE,
  avg_rating          DOUBLE,
  avg_monthly_users   DOUBLE,
  hhi_users           DOUBLE,   -- sum(s_i^2)*10000 (adoption concentration)
  PRIMARY KEY (crawl_date, category)
);

-- (aux) snapshot_manifest — provenance for each daily crawl.
CREATE TABLE IF NOT EXISTS snapshot_manifest (
  crawl_date       DATE PRIMARY KEY,
  raw_file         VARCHAR,
  actor_count      BIGINT,
  n_requests       BIGINT,
  n_errors         BIGINT,
  started_at       TIMESTAMP,
  finished_at      TIMESTAMP
);

-- Optional convenience: query the raw archive directly without an ETL step.
-- Uncomment and adjust the glob if you want a live view over data/raw/*.jsonl.gz:
-- CREATE OR REPLACE VIEW raw_snapshot AS
--   SELECT * FROM read_json_auto('data/raw/*.jsonl.gz', format='newline_delimited');
