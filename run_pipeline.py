"""Reproducible analysis pipeline for the CHI paper.

Runs the paper-relevant analysis blocks in dependency order, then generates the
publication tables/figures and recompiles the paper. Deterministic given the
input data (data/apify_panel.duckdb + data/raw/2026-08-27.actor_detail.jsonl);
the only stochastic steps (PSM logistic, concentration bootstrap) use fixed
seeds (random_state=42 / default_rng(42)).

Usage:
    python3 run_pipeline.py          # run all blocks + tables/figures
    python3 run_pipeline.py --skip-tex  # skip the latexmk recompile
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis"
PAPER = ROOT / "paper"

# Blocks that produce inputs to the paper (in a safe dependency order).
BLOCKS = [
    "block2_landscape.py",                # T1/T2 CSVs, F1, category breakdown
    "block15_eligibility.py",             # core FE / margins / category
    "block16_eligibility_robustness.py",  # whale sensitivity, multi-hot
    "block17_eligibility_psm.py",         # PSM (imports block15)
    "block18_eligibility_concentration.py",
    "block19_eligible_category_breakdown.py",  # F2 input
    "block20_mixed_developers.py",
    "block21_interaction.py",
    "block22_concentration_bootstrap.py",
    "block23_ppml.py",
    "block24_balance_overlap.py",         # balance/overlap (imports block15)
    "block25_architecture_2x2.py",        # 2x2 permission x standby decomposition
    "block26_age_robustness.py",          # age subsamples, age-bin FE, gap distribution
    "block27_similar_tool_matching.py",   # within-developer description/covariate matching
    "block28_runs_30d.py",                # secondary outcome: runs_30d usage intensity
]


def run_block(name: str) -> None:
    print(f"\n=== {name} ===", flush=True)
    proc = subprocess.run(
        [sys.executable, str(ANALYSIS / name)],
        cwd=ROOT, capture_output=True, text=True,
    )
    # Print the tail of stdout (the JSON result) and any stderr errors.
    out = proc.stdout.strip().splitlines()
    if out:
        print("\n".join(out[-15:]))
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"{name} failed with code {proc.returncode}")


def gen_tables() -> None:
    print("\n=== generate_pub_tables_figures (paper tables/figures) ===", flush=True)
    sys.path.insert(0, str(ANALYSIS))
    import generate_pub_tables_figures as g
    for fn in ["tab_T1", "tab_T2", "tab_T21", "tab_T22", "tab_T23",
               "tab_T24", "tab_T25", "tab_T27", "fig_F1", "fig_F2"]:
        getattr(g, fn)()
    # T26 is a static descriptive table (numbers from block20); verify it exists.
    if not (PAPER / "tables" / "T26.tex").exists():
        raise SystemExit("paper/tables/T26.tex missing (static table)")


def compile_tex() -> None:
    print("\n=== latexmk ===", flush=True)
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"],
        cwd=PAPER, check=True,
    )


def main() -> None:
    for name in BLOCKS:
        run_block(name)
    gen_tables()
    if "--skip-tex" not in sys.argv:
        compile_tex()
    print("\nDone. All numbers extracted programmatically from stored result files.")


if __name__ == "__main__":
    main()
