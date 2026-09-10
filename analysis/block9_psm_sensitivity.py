"""Block 9: PSM robustness — common support, matched-vs-unmatched, IPTW, matching with replacement.

Addresses the critique that 1:1 NN matching without replacement (10,785 pairs
out of 39,490 treated) reports a local overlap effect, not the population ATT.

Adds a developer-level (cluster) bootstrap (B = 500) for all four estimators.
Bootstrap standard errors are the std of the B resample estimates; bootstrap
p-values are centered-percentile: p = P(|b* - point| >= |point|) with a
+1/(B+1) continuity correction (Efron & Tibshirani). These differ from Wald
p-values (beta / SE -> normal) because the bootstrap distribution need not be
normal.

Outputs:
  - results/tables/T16_psm_sensitivity.csv  (now with SE and p columns)
  - results/data/block9_psm_sensitivity.json
"""

import json
import logging
import math
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "apify_panel.duckdb"
BASE = Path(__file__).resolve().parent.parent / "results"

RNG = np.random.default_rng(42)
N_BOOT = 500


def prepare(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute("""
        SELECT
            a.actor_id, a.username,
            a.is_agentic_payments_whitelisted AS agentic,
            a.monthly_users,
            COALESCE(a.runs_30d_total, 0) AS runs_30d_total,
            a.total_builds, a.bookmarks, a.rating, a.review_count,
            dev.n_actors AS dev_portfolio_size,
            COALESCE(pe.n_events, 0) AS n_pricing_events,
            LENGTH(COALESCE(a.description, '')) AS desc_length,
            json_extract_string(a.categories, '$[0]') AS primary_category,
            EXTRACT(EPOCH FROM ((SELECT MAX(crawl_date)::TIMESTAMP FROM actor_day) - p.pricing_created_at)) / 86400.0 AS age_days
        FROM actor_day a
        JOIN pricing_day p ON a.actor_id = p.actor_id AND a.crawl_date = p.crawl_date
        LEFT JOIN (
            SELECT username, COUNT(*) AS n_actors
            FROM actor_day WHERE crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
            GROUP BY username
        ) dev ON a.username = dev.username
        LEFT JOIN (
            SELECT actor_id, COUNT(*) AS n_events
            FROM pricing_event_detail
            WHERE crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
            GROUP BY actor_id
        ) pe ON a.actor_id = pe.actor_id
        WHERE a.crawl_date = (SELECT MAX(crawl_date) FROM actor_day)
          AND a.pricing_model = 'PAY_PER_EVENT'
        ORDER BY a.actor_id
    """).df()
    con.close()

    df["agentic_int"] = df["agentic"].astype(int)
    df["primary_category"] = df["primary_category"].fillna("UNKNOWN").str.strip('"')
    for c in ["n_pricing_events", "desc_length", "dev_portfolio_size"]:
        df[c] = df[c].fillna(0)
    df["log_builds"] = np.log1p(df["total_builds"].fillna(0))
    df["log_bookmarks"] = np.log1p(df["bookmarks"].fillna(0))
    df["log_reviews"] = np.log1p(df["review_count"].fillna(0))
    df["rating"] = df["rating"].fillna(0)
    df["log_age"] = np.log1p(df["age_days"].fillna(df["age_days"].median()))
    df["log_mu"] = np.log1p(df["monthly_users"])
    return df


def fit_ps(df: pd.DataFrame, cat_cols: list) -> np.ndarray:
    """Propensity score with listing-time covariates only: actor age,
    description length, developer portfolio size, and category (one-hot).

    Excludes post-treatment bookmarks, reviews, and rating, consistent with the
    paper's predetermined-versus-contemporaneous distinction. liblinear solver
    (fast, deterministic, converges).
    """
    num = df[["log_age", "desc_length", "dev_portfolio_size"]].values
    cat = (pd.get_dummies(df["primary_category"], prefix="cat")
             .reindex(columns=cat_cols, fill_value=0)
             .values.astype(float))
    X = np.hstack([num, cat])
    y = df["agentic_int"].values
    lr = LogisticRegression(solver="liblinear", max_iter=10000, C=1.0, random_state=42)
    lr.fit(X, y)
    return lr.predict_proba(X)[:, 1]


def overlap_weight_att(df: pd.DataFrame, ps: np.ndarray) -> float:
    """Overlap-weighted ATO: treated weight = (1-p), control weight = p."""
    df = df.copy()
    df["ps"] = ps
    w_t = 1 - ps  # treated
    w_c = ps      # control
    y_t = df.loc[df["agentic_int"] == 1, "log_mu"]
    y_c = df.loc[df["agentic_int"] == 0, "log_mu"]
    w_t = w_t[df["agentic_int"] == 1]
    w_c = w_c[df["agentic_int"] == 0]
    att = (y_t * w_t).sum() / w_t.sum() - (y_c * w_c).sum() / w_c.sum()
    return float(att)


def iptw_att(df: pd.DataFrame, ps: np.ndarray) -> float:
    """ATT via IPTW: treated weight = 1, control weight = p/(1-p)."""
    df = df.copy()
    df["ps"] = np.clip(ps, 1e-6, 1 - 1e-6)
    treated = df["agentic_int"] == 1
    w = np.ones(len(df))
    w[~treated] = df.loc[~treated, "ps"] / (1 - df.loc[~treated, "ps"])
    y = df["log_mu"].values
    att = y[treated].mean() - (y[~treated] * w[~treated]).sum() / w[~treated].sum()
    return float(att)


def _nearest_neighbor_1d(pt: np.ndarray, pc: np.ndarray) -> np.ndarray:
    """For each treated propensity score, index of nearest control (replacement ok).

    Exact O(n log n) nearest-neighbor on 1D via argsort + searchsorted.
    """
    order = np.argsort(pc)
    sorted_pc = pc[order]
    pos = np.searchsorted(sorted_pc, pt)
    pos = np.clip(pos, 1, len(sorted_pc) - 1)
    left = sorted_pc[pos - 1]
    right = sorted_pc[pos]
    use_left = np.abs(pt - left) <= np.abs(right - pt)
    nearest_sorted = np.where(use_left, pos - 1, pos)
    return order[nearest_sorted]


def matching_with_replacement(df: pd.DataFrame, ps: np.ndarray, caliper: float) -> dict:
    """1:1 NN matching WITH replacement, full treated sample, caliper."""
    treated = df[df["agentic_int"] == 1].reset_index(drop=True)
    control = df[df["agentic_int"] == 0].reset_index(drop=True)
    pt = ps[df["agentic_int"] == 1]
    pc = ps[df["agentic_int"] == 0]
    nearest = _nearest_neighbor_1d(pt, pc)
    dist = np.abs(pt - pc[nearest])
    within = dist <= caliper
    matched_t = np.where(within)[0]
    matched_c = nearest[within]
    att = treated.iloc[matched_t]["log_mu"].mean() - control.iloc[matched_c]["log_mu"].mean()
    return {"n_matched": int(len(matched_t)), "att": float(att),
            "n_treated": int(len(treated))}


class _Fenwick:
    """Fenwick tree for order-statistic queries (which sorted-control positions
    are still available). Enables O(log n) nearest-available lookups."""
    __slots__ = ("n", "bit")

    def __init__(self, n: int):
        self.n = n
        self.bit = [0] * (n + 1)
        for i in range(1, n + 1):
            self.bit[i] += 1
            j = i + (i & -i)
            if j <= n:
                self.bit[j] += self.bit[i]

    def add(self, i: int, delta: int) -> None:
        i += 1
        while i <= self.n:
            self.bit[i] += delta
            i += i & -i

    def sum(self, i: int) -> int:
        s = 0
        while i > 0:
            s += self.bit[i]
            i -= i & -i
        return s

    def kth(self, k: int) -> int:
        """0-indexed position of the k-th (1-indexed) available element."""
        idx = 0
        m = 1 << (self.n.bit_length() - 1)
        while m:
            t = idx + m
            if t <= self.n and self.bit[t] < k:
                idx = t
                k -= self.bit[t]
            m >>= 1
        return idx


def matching_1to1_noreplacement(df: pd.DataFrame, ps: np.ndarray, caliper: float) -> dict:
    """Greedy 1:1 NN matching WITHOUT replacement, with caliper.

    Matches treated actors (in ascending propensity-score order) to their
    nearest unused control within the caliper, using a Fenwick tree for O(n
    log n) nearest-available lookups. Returns matched indices plus the
    matched-treated frame used by the balance table.
    """
    treated = df[df["agentic_int"] == 1].reset_index(drop=True)
    control = df[df["agentic_int"] == 0].reset_index(drop=True)
    pt = ps[df["agentic_int"] == 1]
    pc = ps[df["agentic_int"] == 0]

    order_c = np.argsort(pc)
    pc_sorted = pc[order_c]
    n_c = len(pc_sorted)

    ft = _Fenwick(n_c)
    total = n_c

    matched_t, matched_c = [], []
    for i in np.argsort(pt, kind="stable"):   # treated in ascending ps order (stable ties)
        p = pt[i]
        pos = int(np.searchsorted(pc_sorted, p))
        cnt_left = ft.sum(pos)        # available controls with sorted index < pos
        best, best_d = -1, caliper + 1e-12
        # nearest available to the right (smallest available index >= pos)
        if cnt_left < total:
            r = ft.kth(cnt_left + 1)
            dr = pc_sorted[r] - p
            if dr < best_d:
                best, best_d = r, dr
        # nearest available to the left (largest available index < pos)
        if cnt_left > 0:
            l = ft.kth(cnt_left)
            dl = p - pc_sorted[l]
            if dl < best_d:
                best, best_d = l, dl
        if best >= 0:
            ft.add(best, -1)
            total -= 1
            matched_t.append(i)
            matched_c.append(order_c[best])

    matched_t = np.array(matched_t, dtype=int)
    matched_c = np.array(matched_c, dtype=int)
    att = float(treated.iloc[matched_t]["log_mu"].mean() - control.iloc[matched_c]["log_mu"].mean())
    return {"n_pairs": int(len(matched_t)), "att": att,
            "matched_treated": treated.iloc[matched_t]}


def compute_estimates(df: pd.DataFrame, ps: np.ndarray, caliper: float) -> dict:
    """All four ATT point estimates on one dataframe with given ps."""
    m1 = matching_1to1_noreplacement(df, ps, caliper)
    return {
        "1to1": m1["att"],
        "overlap": overlap_weight_att(df, ps),
        "iptw": iptw_att(df, ps),
        "replacement": matching_with_replacement(df, ps, caliper)["att"],
    }


def percentile_p(boot: np.ndarray, point: float) -> float:
    """Two-sided centered-percentile p-value (Efron & Tibshirani).

    p = P(|b* - point| >= |point|), with a +1/(B+1) continuity correction so
    the value is never exactly 0. Tests H0: theta = 0 against a two-sided
    alternative. Unlike the naive 2*min(P(b*<=0), P(b*>=0)) rule, it does not
    collapse to 0 when every resample lands on the same side of 0.
    """
    B = len(boot)
    extreme = int(np.sum(np.abs(boot - point) >= np.abs(point)))
    return float((1 + extreme) / (B + 1))


def bootstrap_estimates(df: pd.DataFrame, cat_cols: list, rng, B: int) -> dict:
    """Developer-level (cluster) bootstrap.

    Resamples developers with replacement (keeping all actors of each sampled
    developer), re-fits the propensity score, and recomputes the four ATT
    estimators on each resample. Returns a dict mapping estimator name to a
    length-B array of estimates.
    """
    pos_map = {}
    for pos, u in enumerate(df["username"].values):
        pos_map.setdefault(u, []).append(pos)
    pos_map = {u: np.array(v, dtype=int) for u, v in pos_map.items()}
    users = np.array(list(pos_map.keys()))

    boots = {"1to1": np.empty(B), "overlap": np.empty(B),
             "iptw": np.empty(B), "replacement": np.empty(B)}
    for b in range(B):
        sampled = rng.choice(users, size=len(users), replace=True)
        idx = np.concatenate([pos_map[u] for u in sampled])
        boot_df = df.iloc[idx].reset_index(drop=True)
        ps_b = fit_ps(boot_df, cat_cols)
        caliper_b = 0.05 * float(np.std(ps_b))
        est = compute_estimates(boot_df, ps_b, caliper_b)
        for k in boots:
            boots[k][b] = est[k]
        if (b + 1) % 100 == 0:
            logger.info("  bootstrap %d/%d", b + 1, B)
    return boots


def main():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = prepare(con)
    cat_cols = ["cat_" + c for c in sorted(df["primary_category"].unique())]
    ps = fit_ps(df, cat_cols)

    n_treated = int(df["agentic_int"].sum())
    n_control = int((1 - df["agentic_int"]).sum())
    logger.info("Treated=%d, Control=%d (ratio %.2f:1)", n_treated, n_control, n_treated / n_control)

    # Common support: min/max ps in each group
    ps_t = ps[df["agentic_int"] == 1]
    ps_c = ps[df["agentic_int"] == 0]
    support_lo = max(ps_t.min(), ps_c.min())
    support_hi = min(ps_t.max(), ps_c.max())
    n_support = int(((ps >= support_lo) & (ps <= support_hi)).sum())
    n_treated_in_support = int(((df["agentic_int"] == 1) & (ps >= support_lo) & (ps <= support_hi)).sum())
    logger.info("Common support [%.4f, %.4f]: %d/%d treated (%.1f%%) inside",
                support_lo, support_hi, n_treated_in_support, n_treated,
                n_treated_in_support / n_treated * 100)

    caliper = 0.05 * ps.std()

    # 1:1 no-replacement (with matched-treated frame for balance table)
    m1 = matching_1to1_noreplacement(df, ps, caliper)
    n_pairs = m1["n_pairs"]
    matched_treated = m1["matched_treated"]
    logger.info("1:1 no-replacement: %d pairs (%.1f%% of treated matched)",
                n_pairs, n_pairs / n_treated * 100)

    # Balance table: matched vs unmatched treated on key observables
    cols = ["log_builds", "dev_portfolio_size", "desc_length", "n_pricing_events",
            "log_bookmarks", "log_reviews", "rating", "log_age", "monthly_users"]
    bal_rows = []
    treated_all = df[df["agentic_int"] == 1].reset_index(drop=True)
    unmatched_treated = treated_all.drop(matched_treated.index)
    for c in cols:
        m = matched_treated[c].mean()
        u = unmatched_treated[c].mean()
        bal_rows.append({
            "variable": c,
            "matched_treated_mean": round(float(m), 3),
            "unmatched_treated_mean": round(float(u), 3),
            "diff": round(float(m - u), 3),
        })
    logger.info("Matched vs unmatched treated (key variable diffs):")
    for r in bal_rows:
        if abs(r["diff"]) / max(1, abs(r["matched_treated_mean"])) > 0.05:
            logger.info("  %-22s matched=%.3f unmatched=%.3f diff=%.3f",
                        r["variable"], r["matched_treated_mean"], r["unmatched_treated_mean"], r["diff"])

    # Point estimates
    point = compute_estimates(df, ps, caliper)
    logger.info("Estimates (log_mu):")
    logger.info("  1:1 no-replacement ATT: %.4f (exp-1=%.1f%%)", point["1to1"], (math.exp(point["1to1"])-1)*100)
    logger.info("  Overlap-weighted ATT:   %.4f (exp-1=%.1f%%)", point["overlap"], (math.exp(point["overlap"])-1)*100)
    logger.info("  IPTW ATT:               %.4f (exp-1=%.1f%%)", point["iptw"], (math.exp(point["iptw"])-1)*100)
    logger.info("  Match w/ replacement:   %.4f (exp-1=%.1f%%)", point["replacement"], (math.exp(point["replacement"])-1)*100)

    # Bootstrap (developer-level, B = 500)
    logger.info("Running developer-level bootstrap (B=%d)...", N_BOOT)
    boots = bootstrap_estimates(df, cat_cols, RNG, N_BOOT)

    se = {k: float(np.std(v)) for k, v in boots.items()}
    p = {k: percentile_p(v, point[k]) for k, v in boots.items()}
    logger.info("Bootstrap SE / percentile p:")
    for k in boots:
        logger.info("  %-15s SE=%.4f  p=%.3f", k, se[k], p[k])

    results = {
        "n_treated": n_treated, "n_control": n_control,
        "common_support": {"lo": support_lo, "hi": support_hi,
                            "n_treated_in_support": n_treated_in_support,
                            "pct_treated_in_support": round(n_treated_in_support/n_treated*100, 1)},
        "matching_1to1": {"n_pairs": n_pairs, "pct_treated_matched": round(n_pairs/n_treated*100, 1),
                          "att": round(point["1to1"], 4)},
        "overlap_weighted_att": round(point["overlap"], 4),
        "iptw_att": round(point["iptw"], 4),
        "matching_with_replacement": {"att": round(point["replacement"], 4)},
        "bootstrap": {
            "B": N_BOOT,
            "estimators": {
                "1to1": {"se": round(se["1to1"], 4), "p": round(p["1to1"], 3)},
                "overlap": {"se": round(se["overlap"], 4), "p": round(p["overlap"], 3)},
                "iptw": {"se": round(se["iptw"], 4), "p": round(p["iptw"], 3)},
                "replacement": {"se": round(se["replacement"], 4), "p": round(p["replacement"], 3)},
            },
        },
        "matched_vs_unmatched_treated": bal_rows,
    }

    with open(BASE / "data" / "block9_psm_sensitivity.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save table
    rows = [
        {"Estimator": "1:1 NN, no replacement (matched subset)",
         "log_mu coef": round(point["1to1"], 4),
         "exp-1 %": round((math.exp(point["1to1"])-1)*100, 1),
         "SE": round(se["1to1"], 4), "p": round(p["1to1"], 3), "N treated": n_pairs},
        {"Estimator": "Matching with replacement (full treated)",
         "log_mu coef": round(point["replacement"], 4),
         "exp-1 %": round((math.exp(point["replacement"])-1)*100, 1),
         "SE": round(se["replacement"], 4), "p": round(p["replacement"], 3), "N treated": n_treated},
        {"Estimator": "Overlap-weighted ATO (full sample)",
         "log_mu coef": round(point["overlap"], 4),
         "exp-1 %": round((math.exp(point["overlap"])-1)*100, 1),
         "SE": round(se["overlap"], 4), "p": round(p["overlap"], 3), "N treated": n_treated},
        {"Estimator": "IPTW ATT (full sample)",
         "log_mu coef": round(point["iptw"], 4),
         "exp-1 %": round((math.exp(point["iptw"])-1)*100, 1),
         "SE": round(se["iptw"], 4), "p": round(p["iptw"], 3), "N treated": n_treated},
    ]
    pd.DataFrame(rows).to_csv(BASE / "tables" / "T16_psm_sensitivity.csv", index=False)
    logger.info("Saved T16 + block9 json (with bootstrap SE/p).")


if __name__ == "__main__":
    main()
