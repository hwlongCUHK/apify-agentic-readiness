# Apify Agentic Readiness: Adoption in an Automation Marketplace

Code, data, and analysis for the CHI 2027 paper **"Admitting AI Agents: Agentic Readiness and Adoption in an Automation Marketplace."**

We study the Apify Store's agentic-readiness boundary — the platform's architectural gate on which tools AI agents may discover, invoke, and pay for — and its association with tool adoption and within-group adoption inequality.

## Repository structure

```
.
├── pipeline/          # Data-collection pipeline (enumerate → fetch → normalize)
│   ├── config.py      #   paths, pricing-model mapping, API settings
│   ├── enumerate.py   #   developer enumeration via the public sitemap
│   ├── fetch.py       #   per-developer store listing snapshots
│   ├── fetch_actor_detail.py  # actor-detail (permissions + standby)
│   ├── normalize.py   #   raw JSONL → DuckDB tables
│   ├── diff_alert.py  #   regime-transition detection
│   └── schema.sql     #   DuckDB schema
├── analysis/          # Analysis blocks (one file per result)
│   ├── block2_landscape.py          # T1/T2, marketplace census
│   ├── block15_eligibility.py       # core within-developer FE
│   ├── block16_eligibility_robustness.py
│   ├── block17_eligibility_psm.py
│   ├── block18_eligibility_concentration.py
│   ├── block20_mixed_developers.py
│   ├── block21_interaction.py
│   ├── block22_concentration_bootstrap.py
│   ├── block23_ppml.py
│   ├── block24_balance_overlap.py
│   ├── block25_architecture_2x2.py
│   ├── block26_age_robustness.py
│   ├── block27_similar_tool_matching.py
│   ├── block28_runs_30d.py
│   └── generate_pub_tables_figures.py  # publication tables/figures
├── run_pipeline.py    # Runs all analysis blocks + generates tables/figures
├── paper/             # LaTeX source (main.tex, sections/, tables/, figures/, references.bib)
├── results/           # Analysis outputs (JSON/CSV data, tables, figures)
└── data/raw/          # Raw store snapshots (17 daily snapshots, 2026-08-19 to 2026-09-04)
                       #   + actor-detail fields (2026-08-27)
```

## Data

The raw data consist of **17 daily snapshots** of the Apify Store (2026-08-19 to 2026-09-04), each a newline-delimited JSON dump of the store listing for every developer, plus an **actor-detail** snapshot (2026-08-27) carrying the two architectural fields that define agentic readiness:

- `actorPermissionLevel` — `LIMITED_PERMISSIONS` vs `FULL_PERMISSIONS`
- `standbyEnabled` — whether the tool uses standby mode

**Agentic readiness** is operationalized as `limited permissions ∧ non-standby`, restricted to pay-per-event (PPE) actors. All data are public and require no authentication.

### Rebuilding the DuckDB panel

The normalized DuckDB database (~2.8 GB) is *not* committed. Rebuild it from the raw snapshots:

```bash
export APIFY_DATA_DIR=./data
for d in 2026-08-19 2026-08-20 2026-08-21 2026-08-22 2026-08-23 \
         2026-08-24 2026-08-25 2026-08-26 2026-08-27 2026-08-28 \
         2026-08-29 2026-08-30 2026-08-31 2026-09-01 2026-09-02 \
         2026-09-03 2026-09-04; do
  python pipeline/normalize.py --date "$d"
done
```

The analysis blocks read the resulting `data/apify_panel.duckdb` and
`data/raw/2026-08-27.actor_detail.jsonl`.

## Reproducing the analysis

```bash
pip install -r pipeline/requirements.txt   # duckdb, pandas, numpy
pip install statsmodels scikit-learn matplotlib seaborn scipy

python run_pipeline.py --skip-tex   # run all blocks + tables/figures (skip LaTeX)
```

All stochastic steps (PSM logistic, concentration bootstrap, matching
bootstrap) use fixed seeds (`random_state=42` / `default_rng(42)`).

## Key results

- **Adoption gap.** Within the same developer, agentic-ready tools attract 71.5% more monthly users (equal-developer-weighted FE, `log(1+MU)` scale; 39.6% actor-weighted), spanning both the extensive (16.5 pp) and intensive (60.2%) margins.
- **No robust concentration increase.** Gini and top-share bootstrap CIs for the ready vs non-ready difference contain zero; only the Theil gap is distinguishable.
- **Architectural boundary.** The limited-permissions / non-standby boundary is a *necessary* architectural condition for agentic-payment eligibility (developer-level KYC and approval-level curation apply on top), and doubles as a discovery and governance boundary.

## Citation

```bibtex
@misc{apify2026agenticreadiness,
  title = {Admitting {AI} Agents: Agentic Readiness and Adoption in an Automation Marketplace},
  year = {2026},
  howpublished = {CHI 2027 submission}
}
```
