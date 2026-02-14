"""
PPT Version A (baseline): "Test allocations"

Implements:
1) Generate many candidate portfolio weight vectors (5,000)
2) For each portfolio, run retirement Monte Carlo and compute:
      success% = fraction of paths that never drop below $0
3) Identify the weights with the highest success%
4) Summarize characteristics of successful vs failed portfolios (top vs bottom deciles)
5) Strategy tests:
      Step 2: equity-only protective put (early years)
      Step 3: equity-only collar (early years)
      Step 4: spending guardrail (skip inflation after negative return; per path)
      Step 5: guardrail + protective put
      Step 6: grid search over guardrail+put parameters
      Step 7: WR sweep to find max WR meeting success targets (90% and 95%)
      Sanity A: multi-seed stability check (10k paths)
      Sanity B: higher-sample (50k paths) stability check
      Step 8A: Determinism check (same seed, run twice)
      Step 8B: Stress tests (lower mean, higher vol)

Inputs (CSV files in same folder as this script):
- balances_and_target_allocations.csv
- returns_std_devs.csv
- index_correlations.csv

Assumptions (per user):
- Starting wealth = $1,000,000 (not necessarily equal to sum of balances file)
- Withdrawal rate WR = 4% of initial wealth, inflated by 3% per year (baseline)
- Time horizon = 50 years
- Monte Carlo paths = n_samples
- Seed = 1,000,000

NOTE ON TICKERS:
- If returns_std_devs.csv uses "v" or "vmo" for VWO, we map it to "vwo" at load time.

Notes:
- All strategy comparisons in Steps 2–7 use the SAME underlying market paths (market_cube) so results
  are apples-to-apples.
- Sanity checks re-generate market paths using independent seeds / higher n_samples, without re-optimizing weights.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------
# User-specified settings
# -----------------------
SEED = 1_000_000
np.random.seed(SEED)

INITIAL_WEALTH = 1_000_000.0
WR_FIXED = 0.04
inflation_rate = 0.03
time_horizon = 50

N_PORTFOLIOS = 5_000
n_samples = 10_000  # can reduce if runtime is slow

current_year = datetime.now().year


# -----------------------
# Utilities
# -----------------------
def make_positive_semi_definite(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, a_min=1e-10, a_max=None)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def build_market_cube_from_seed(
    *,
    seed: int,
    time_horizon: int,
    n_samples: int,
    n_assets: int,
    L: np.ndarray,
    v_mu: np.ndarray,
    start_year: int,
) -> tuple[dict, list[np.ndarray]]:
    """
    Build a market cube (gross returns) and the corresponding base arithmetic returns by year,
    using a local RNG so results are reproducible per seed and independent of global np.random state.
    """
    rng = np.random.default_rng(seed)
    Z_by_year = [rng.normal(size=(n_samples, n_assets)) for _ in range(time_horizon)]
    base_returns_by_year = [(Z @ L.T + v_mu) for Z in Z_by_year]

    cube = {}
    for i in range(time_horizon):
        cube[f"Market_Returns_{start_year + i}"] = (base_returns_by_year[i] + 1.0).T

    return cube, base_returns_by_year


def max_wr_at_or_above(df: pd.DataFrame, col: str, target: float) -> float | None:
    ok = df[df[col] >= target]
    if ok.empty:
        return None
    return float(ok["wr"].max())


def compute_wr_thresholds(
    *,
    weights: np.ndarray,
    market_cube: dict,
    base_returns_by_year: list[np.ndarray],
    start_year: int,
    time_horizon: int,
    n_samples: int,
    inflation_rate: float,
    wr_grid: np.ndarray,
    targets: list[float],
    equity_mask: np.ndarray,
    hedge_alloc: float,
    best_hy: int,
    best_ps: float,
    best_pp: float,
) -> tuple[pd.DataFrame, dict]:
    """
    Computes success curves vs WR for:
      - baseline spending
      - guardrail spending
      - guardrail spending + best put overlay
    Returns:
      - df: full WR sweep table
      - thresholds: dict with max WR meeting each target for each strategy
    """
    # Build best guardrail+put cube for this market path set
    cube_gp = {}
    for i in range(time_horizon):
        r = base_returns_by_year[i]
        on = hedge_alloc if i < best_hy else 0.0
        r2 = apply_equity_put_overlay(
            r,
            weights=weights,
            equity_mask=equity_mask,
            hedge_allocation=on,
            put_strike_return=best_ps,
            put_premium_rate=best_pp,
        )
        cube_gp[f"Market_Returns_{start_year + i}"] = (r2 + 1.0).T

    rows = []
    for wr in wr_grid:
        sr_b = simulate_success_rate_fixed_weights(
            weights=weights,
            withdrawal_rate=float(wr),
            beg_net_worth=INITIAL_WEALTH,
            market_cube_returns_plus1=market_cube,
            start_year=start_year,
            time_horizon=time_horizon,
            n_samples=n_samples,
            inflation_rate=inflation_rate,
        )
        sr_g = simulate_success_rate_guardrail_skip_inflation(
            weights=weights,
            withdrawal_rate=float(wr),
            beg_net_worth=INITIAL_WEALTH,
            market_cube_returns_plus1=market_cube,
            start_year=start_year,
            time_horizon=time_horizon,
            n_samples=n_samples,
            inflation_rate=inflation_rate,
            debug=False,
        )
        sr_gp = simulate_success_rate_guardrail_skip_inflation(
            weights=weights,
            withdrawal_rate=float(wr),
            beg_net_worth=INITIAL_WEALTH,
            market_cube_returns_plus1=cube_gp,
            start_year=start_year,
            time_horizon=time_horizon,
            n_samples=n_samples,
            inflation_rate=inflation_rate,
            debug=False,
        )
        rows.append(
            {
                "wr": float(wr),
                "baseline_success": float(sr_b),
                "guardrail_success": float(sr_g),
                "guardrail_put_success": float(sr_gp),
            }
        )

    df = pd.DataFrame(rows)

    thresholds = {}
    for t in targets:
        thresholds[f"max_wr_baseline_{int(t*100)}"] = max_wr_at_or_above(df, "baseline_success", t)
        thresholds[f"max_wr_guardrail_{int(t*100)}"] = max_wr_at_or_above(df, "guardrail_success", t)
        thresholds[f"max_wr_guardrail_put_{int(t*100)}"] = max_wr_at_or_above(df, "guardrail_put_success", t)

    return df, thresholds


def simulate_success_rate_fixed_weights(
    *,
    weights: np.ndarray,
    withdrawal_rate: float,
    beg_net_worth: float,
    market_cube_returns_plus1: dict,
    start_year: int,
    time_horizon: int,
    n_samples: int,
    inflation_rate: float,
) -> float:
    """
    Baseline spending rule:
      - withdraw a single spending amount each year (same for all paths)
      - increase spending by inflation every year
    """
    spending = withdrawal_rate * beg_net_worth
    alive = np.ones(n_samples, dtype=bool)
    net_worth = np.full(n_samples, beg_net_worth, dtype=float)

    for j in range(time_horizon):
        net_worth[alive] -= spending
        alive &= (net_worth > 0)

        if not np.any(alive):
            return 0.0

        key = f"Market_Returns_{start_year + j}"
        if key not in market_cube_returns_plus1:
            available = sorted(market_cube_returns_plus1.keys())
            raise KeyError(
                f"{key} not found in market_cube_returns_plus1. "
                f"Available range: {available[0]} .. {available[-1]}"
            )

        gross = market_cube_returns_plus1[key].T  # (n_samples, n_assets)
        net_worth = (np.outer(net_worth, weights) * gross).sum(axis=1)

        spending *= (1 + inflation_rate)

    return float(alive.mean())


def simulate_success_rate_guardrail_skip_inflation(
    *,
    weights: np.ndarray,
    withdrawal_rate: float,
    beg_net_worth: float,
    market_cube_returns_plus1: dict,
    start_year: int,
    time_horizon: int,
    n_samples: int,
    inflation_rate: float,
    debug: bool = False,
) -> float:
    """
    Per-path guardrail:
      - spending is tracked per path
      - after NEGATIVE portfolio return in a year, that path does NOT receive an inflation increase next year
      - after NON-NEGATIVE return, that path DOES receive the inflation increase
    """
    spending = np.full(n_samples, withdrawal_rate * beg_net_worth, dtype=float)  # per path
    alive = np.ones(n_samples, dtype=bool)
    net_worth = np.full(n_samples, beg_net_worth, dtype=float)

    skipped_inflation_count = 0
    alive_path_years = 0

    for j in range(time_horizon):
        net_worth[alive] -= spending[alive]
        alive &= (net_worth > 0)

        if not np.any(alive):
            if debug and alive_path_years > 0:
                print(
                    "Guardrail debug: share inflation-skipped (alive path-years) =",
                    skipped_inflation_count / alive_path_years,
                )
            return 0.0

        key = f"Market_Returns_{start_year + j}"
        if key not in market_cube_returns_plus1:
            available = sorted(market_cube_returns_plus1.keys())
            raise KeyError(
                f"{key} not found in market_cube_returns_plus1. "
                f"Available range: {available[0]} .. {available[-1]}"
            )

        gross = market_cube_returns_plus1[key].T  # (n_samples, n_assets)
        net_worth_next = (np.outer(net_worth, weights) * gross).sum(axis=1)

        port_ret = np.zeros(n_samples, dtype=float)
        port_ret[alive] = (net_worth_next[alive] / net_worth[alive]) - 1.0

        net_worth = net_worth_next

        apply_infl = alive & (port_ret >= 0)
        skip_infl = alive & (port_ret < 0)

        spending[apply_infl] *= (1.0 + inflation_rate)

        if debug:
            skipped_inflation_count += int(skip_infl.sum())
            alive_path_years += int(alive.sum())

    if debug and alive_path_years > 0:
        print(
            "Guardrail debug: share inflation-skipped (alive path-years) =",
            skipped_inflation_count / alive_path_years,
        )

    return float(alive.mean())


def sample_dirichlet_weights(n_portfolios: int, n_assets: int) -> np.ndarray:
    """Long-only weights summing to 1; uniform over simplex."""
    alpha = np.ones(n_assets)
    return np.random.dirichlet(alpha, size=n_portfolios)


def compute_group_summaries(results_df: pd.DataFrame, assets: list[str], top_frac: float = 0.10) -> pd.DataFrame:
    """Compare average weights for top vs bottom fraction of portfolios by success."""
    n = len(results_df)
    k = max(1, int(round(n * top_frac)))

    top = results_df.sort_values("success_rate", ascending=False).head(k)
    bottom = results_df.sort_values("success_rate", ascending=True).head(k)

    rows = []
    for a in assets:
        tm = float(top[f"w_{a}"].mean())
        bm = float(bottom[f"w_{a}"].mean())
        rows.append((a, tm, bm, tm - bm))

    out = pd.DataFrame(rows, columns=["asset", "top_mean", "bottom_mean", "top_minus_bottom"])
    out = out.sort_values("top_minus_bottom", ascending=False).reset_index(drop=True)
    return out


def apply_equity_put_overlay(
    returns: np.ndarray,
    *,
    weights: np.ndarray,
    equity_mask: np.ndarray,
    hedge_allocation: float,
    put_strike_return: float,
    put_premium_rate: float,
) -> np.ndarray:
    """Equity-only protective put overlay in return space."""
    if hedge_allocation <= 0.0:
        return returns

    w_eq = weights[equity_mask]
    w_eq_sum = float(w_eq.sum())
    if w_eq_sum <= 0:
        return returns

    w_eq_norm = w_eq / w_eq_sum
    r_eq = returns[:, equity_mask] @ w_eq_norm

    put_payoff = np.maximum(put_strike_return - r_eq, 0.0)
    r_eq_hedged = r_eq + put_payoff - put_premium_rate
    r_eq_final = hedge_allocation * r_eq_hedged + (1 - hedge_allocation) * r_eq

    out = returns.copy()
    out[:, equity_mask] = r_eq_final[:, None]
    return out


def apply_equity_collar_overlay(
    returns: np.ndarray,
    *,
    weights: np.ndarray,
    equity_mask: np.ndarray,
    hedge_allocation: float,
    put_strike_return: float,
    call_strike_return: float,
    put_premium_rate: float,
    call_premium_rate: float,
) -> np.ndarray:
    """Equity-only collar overlay (buy put + sell call) in return space."""
    if hedge_allocation <= 0.0:
        return returns

    w_eq = weights[equity_mask]
    w_eq_sum = float(w_eq.sum())
    if w_eq_sum <= 0:
        return returns

    w_eq_norm = w_eq / w_eq_sum
    r_eq = returns[:, equity_mask] @ w_eq_norm

    put_payoff = np.maximum(put_strike_return - r_eq, 0.0)
    call_payoff = -np.maximum(r_eq - call_strike_return, 0.0)

    r_eq_hedged = r_eq + put_payoff + call_payoff - put_premium_rate + call_premium_rate
    r_eq_final = hedge_allocation * r_eq_hedged + (1 - hedge_allocation) * r_eq

    out = returns.copy()
    out[:, equity_mask] = r_eq_final[:, None]
    return out


def main():
    BASE_DIR = Path(__file__).resolve().parent
    print("BASE_DIR =", BASE_DIR)
    print("Seed =", SEED)
    print("WR_FIXED =", WR_FIXED, "Inflation =", inflation_rate, "Horizon =", time_horizon)
    print("N_PORTFOLIOS =", N_PORTFOLIOS, "n_samples =", n_samples)

    # -----------------------
    # Load CSV inputs
    # -----------------------
    df_returns = pd.read_csv(BASE_DIR / "returns_std_devs.csv")
    df_corr = pd.read_csv(BASE_DIR / "index_correlations.csv")
    df_alloc = pd.read_csv(BASE_DIR / "balances_and_target_allocations.csv")

    for df in (df_returns, df_corr, df_alloc):
        df["Index_Fund"] = df["Index_Fund"].astype(str).str.strip().str.lower()

    df_returns["Index_Fund"] = df_returns["Index_Fund"].replace({"v": "vwo", "vmo": "vwo"})

    assets = df_alloc["Index_Fund"].to_list()
    n_assets = len(assets)

    df_returns = df_returns.set_index("Index_Fund").reindex(assets)
    if df_returns.isna().any().any():
        missing = df_returns[df_returns.isna().any(axis=1)].index.to_list()
        raise ValueError(f"Missing Mean/StdDev for: {missing}. Fix returns_std_devs.csv tickers.")

    v_mu = df_returns["Mean"].astype(float).to_numpy()
    v_sigma = df_returns["Standard_Deviation"].astype(float).to_numpy()

    df_corr = df_corr.set_index("Index_Fund")
    corr = df_corr.reindex(index=assets, columns=assets).astype(float).to_numpy()
    corr = make_positive_semi_definite(corr)

    cov = np.diag(v_sigma) @ corr @ np.diag(v_sigma)
    L = np.linalg.cholesky(cov)

    # -----------------------
    # Generate ONE market cube (same for all portfolios and strategies in Steps 1–7)
    # -----------------------
    start_year = current_year
    market_cube, base_returns_by_year = build_market_cube_from_seed(
        seed=SEED,
        time_horizon=time_horizon,
        n_samples=n_samples,
        n_assets=n_assets,
        L=L,
        v_mu=v_mu,
        start_year=start_year,
    )

    # -----------------------
    # Generate candidate portfolios
    # -----------------------
    W = sample_dirichlet_weights(N_PORTFOLIOS, n_assets)

    equity_funds = {"voo", "vb", "vo", "vv", "vtv", "vti", "vwo"}
    equity_mask = np.array([a in equity_funds for a in assets], dtype=bool)

    # -----------------------
    # Evaluate each portfolio (baseline)
    # -----------------------
    rows = []
    for i in range(N_PORTFOLIOS):
        w = W[i]
        sr = simulate_success_rate_fixed_weights(
            weights=w,
            withdrawal_rate=WR_FIXED,
            beg_net_worth=INITIAL_WEALTH,
            market_cube_returns_plus1=market_cube,
            start_year=start_year,
            time_horizon=time_horizon,
            n_samples=n_samples,
            inflation_rate=inflation_rate,
        )

        eq_w = float(w[equity_mask].sum()) if np.any(equity_mask) else float("nan")
        row = {"portfolio_id": i, "success_rate": sr, "equity_weight": eq_w}
        for a, wa in zip(assets, w):
            row[f"w_{a}"] = float(wa)
        rows.append(row)

        if (i + 1) % 500 == 0:
            print(f"  evaluated {i+1}/{N_PORTFOLIOS} portfolios...")

    results = pd.DataFrame(rows).sort_values("success_rate", ascending=False).reset_index(drop=True)

    # -----------------------
    # STEP 1: Best baseline portfolio re-run + save weights
    # -----------------------
    best = results.iloc[0]
    best_id = int(best["portfolio_id"])

    best_weights = np.array([best[f"w_{a}"] for a in assets], dtype=float)
    best_weights = best_weights / best_weights.sum()

    baseline_best_success = simulate_success_rate_fixed_weights(
        weights=best_weights,
        withdrawal_rate=WR_FIXED,
        beg_net_worth=INITIAL_WEALTH,
        market_cube_returns_plus1=market_cube,
        start_year=start_year,
        time_horizon=time_horizon,
        n_samples=n_samples,
        inflation_rate=inflation_rate,
    )

    print("\n===== STEP 1: BASELINE BEST PORTFOLIO RE-RUN =====")
    print(f"Best portfolio_id={best_id}")
    print(f"Baseline success at WR={WR_FIXED:.2%}: {baseline_best_success:.2%}")

    best_weights_df = pd.DataFrame({"Index_Fund": assets, "Weight": best_weights}).sort_values("Weight", ascending=False)
    best_weights_path = BASE_DIR / "best_portfolio_weights_versionA.csv"
    best_weights_df.to_csv(best_weights_path, index=False)
    print("Saved best weights to:", best_weights_path)

    # -----------------------
    # STEP 2: Protective put (equity-only) early years
    # -----------------------
    hedge_years = 15
    put_strike_return = -0.15
    put_premium_rate = 0.01
    hedge_alloc = 1.0

    market_cube_put = {}
    for i in range(time_horizon):
        r = base_returns_by_year[i]
        on = hedge_alloc if i < hedge_years else 0.0
        r2 = apply_equity_put_overlay(
            r,
            weights=best_weights,
            equity_mask=equity_mask,
            hedge_allocation=on,
            put_strike_return=put_strike_return,
            put_premium_rate=put_premium_rate,
        )
        market_cube_put[f"Market_Returns_{start_year + i}"] = (r2 + 1.0).T

    put_success = simulate_success_rate_fixed_weights(
        weights=best_weights,
        withdrawal_rate=WR_FIXED,
        beg_net_worth=INITIAL_WEALTH,
        market_cube_returns_plus1=market_cube_put,
        start_year=start_year,
        time_horizon=time_horizon,
        n_samples=n_samples,
        inflation_rate=inflation_rate,
    )

    print("\n===== STEP 2: STRATEGY TEST (Protective Put, equity-only) =====")
    print(f"Baseline success @ WR={WR_FIXED:.2%}: {baseline_best_success:.2%}")
    print(f"Put strategy success @ WR={WR_FIXED:.2%}: {put_success:.2%}")
    print(
        "Params: "
        f"hedge_years={hedge_years}, put_strike={put_strike_return:.0%}, "
        f"put_premium={put_premium_rate:.2%}/yr, hedge_alloc={hedge_alloc:.0%}"
    )

    # -----------------------
    # STEP 3: Collar (equity-only) early years
    # -----------------------
    call_strike_return = 0.15
    call_premium_rate = 0.01

    market_cube_collar = {}
    for i in range(time_horizon):
        r = base_returns_by_year[i]
        on = hedge_alloc if i < hedge_years else 0.0
        r3 = apply_equity_collar_overlay(
            r,
            weights=best_weights,
            equity_mask=equity_mask,
            hedge_allocation=on,
            put_strike_return=put_strike_return,
            call_strike_return=call_strike_return,
            put_premium_rate=put_premium_rate,
            call_premium_rate=call_premium_rate,
        )
        market_cube_collar[f"Market_Returns_{start_year + i}"] = (r3 + 1.0).T

    collar_success = simulate_success_rate_fixed_weights(
        weights=best_weights,
        withdrawal_rate=WR_FIXED,
        beg_net_worth=INITIAL_WEALTH,
        market_cube_returns_plus1=market_cube_collar,
        start_year=start_year,
        time_horizon=time_horizon,
        n_samples=n_samples,
        inflation_rate=inflation_rate,
    )

    print("\n===== STEP 3: STRATEGY TEST (Collar, equity-only) =====")
    print(f"Baseline success @ WR={WR_FIXED:.2%}: {baseline_best_success:.2%}")
    print(f"Put strategy success @ WR={WR_FIXED:.2%}: {put_success:.2%}")
    print(f"Collar strategy success @ WR={WR_FIXED:.2%}: {collar_success:.2%}")
    print(
        "Params: "
        f"hedge_years={hedge_years}, put_strike={put_strike_return:.0%}, put_premium={put_premium_rate:.2%}/yr, "
        f"call_strike={call_strike_return:.0%}, call_premium={call_premium_rate:.2%}/yr, "
        f"hedge_alloc={hedge_alloc:.0%}"
    )

    # -----------------------
    # STEP 4: Guardrail (skip inflation after negative return)
    # -----------------------
    guardrail_success = simulate_success_rate_guardrail_skip_inflation(
        weights=best_weights,
        withdrawal_rate=WR_FIXED,
        beg_net_worth=INITIAL_WEALTH,
        market_cube_returns_plus1=market_cube,
        start_year=start_year,
        time_horizon=time_horizon,
        n_samples=n_samples,
        inflation_rate=inflation_rate,
        debug=True,
    )

    print("\n===== STEP 4: STRATEGY TEST (Guardrail: skip inflation after negative return) =====")
    print(f"Baseline success @ WR={WR_FIXED:.2%}: {baseline_best_success:.2%}")
    print(f"Put success @ WR={WR_FIXED:.2%}: {put_success:.2%}")
    print(f"Collar success @ WR={WR_FIXED:.2%}: {collar_success:.2%}")
    print(f"Guardrail success @ WR={WR_FIXED:.2%}: {guardrail_success:.2%}")

    # -----------------------
    # STEP 5: Guardrail + Protective Put (using Step 2 put cube)
    # -----------------------
    guardrail_put_success = simulate_success_rate_guardrail_skip_inflation(
        weights=best_weights,
        withdrawal_rate=WR_FIXED,
        beg_net_worth=INITIAL_WEALTH,
        market_cube_returns_plus1=market_cube_put,
        start_year=start_year,
        time_horizon=time_horizon,
        n_samples=n_samples,
        inflation_rate=inflation_rate,
        debug=False,
    )

    print("\n===== STEP 5: STRATEGY TEST (Guardrail + Protective Put) =====")
    print(f"Guardrail success @ WR={WR_FIXED:.2%}: {guardrail_success:.2%}")
    print(f"Guardrail+Put success @ WR={WR_FIXED:.2%}: {guardrail_put_success:.2%}")

    # -----------------------
    # STEP 6: Grid search (Guardrail + Put)
    # -----------------------
    grid_hedge_years = [10, 15, 20]
    grid_put_strikes = [-0.10, -0.15, -0.20]
    grid_put_premiums = [0.005, 0.010, 0.015]

    grid_rows = []
    for hy in grid_hedge_years:
        for ps in grid_put_strikes:
            for pp in grid_put_premiums:
                cube = {}
                for i in range(time_horizon):
                    r = base_returns_by_year[i]
                    on = hedge_alloc if i < hy else 0.0
                    r2 = apply_equity_put_overlay(
                        r,
                        weights=best_weights,
                        equity_mask=equity_mask,
                        hedge_allocation=on,
                        put_strike_return=ps,
                        put_premium_rate=pp,
                    )
                    cube[f"Market_Returns_{start_year + i}"] = (r2 + 1.0).T

                sr = simulate_success_rate_guardrail_skip_inflation(
                    weights=best_weights,
                    withdrawal_rate=WR_FIXED,
                    beg_net_worth=INITIAL_WEALTH,
                    market_cube_returns_plus1=cube,
                    start_year=start_year,
                    time_horizon=time_horizon,
                    n_samples=n_samples,
                    inflation_rate=inflation_rate,
                    debug=False,
                )
                grid_rows.append({"hedge_years": hy, "put_strike": ps, "put_premium": pp, "success_rate": sr})

    grid_df = pd.DataFrame(grid_rows).sort_values("success_rate", ascending=False).reset_index(drop=True)

    print("\n===== STEP 6: GRID SEARCH RESULTS (Guardrail + Put) =====")
    print("Top 10 parameter sets:")
    print(grid_df.head(10).to_string(index=False))

    grid_out = BASE_DIR / "step6_guardrail_put_grid.csv"
    grid_df.to_csv(grid_out, index=False)
    print("Saved:", grid_out)

    winner = grid_df.iloc[0].to_dict()
    BEST_HY = int(winner["hedge_years"])
    BEST_PS = float(winner["put_strike"])
    BEST_PP = float(winner["put_premium"])
    BEST_SR = float(winner["success_rate"])

    # -----------------------
    # STEP 7: WR sweep for 90% and 95% success targets
    # -----------------------
    targets = [0.90, 0.95]
    wr_grid = np.arange(0.03, 0.0701, 0.0025)  # 3%..7% in 0.25% steps

    wr_df, _ = compute_wr_thresholds(
        weights=best_weights,
        market_cube=market_cube,
        base_returns_by_year=base_returns_by_year,
        start_year=start_year,
        time_horizon=time_horizon,
        n_samples=n_samples,
        inflation_rate=inflation_rate,
        wr_grid=wr_grid,
        targets=targets,
        equity_mask=equity_mask,
        hedge_alloc=hedge_alloc,
        best_hy=BEST_HY,
        best_ps=BEST_PS,
        best_pp=BEST_PP,
    )

    print("\n===== STEP 7: MAX WR MEETING SUCCESS TARGETS =====")
    for t in targets:
        b = max_wr_at_or_above(wr_df, "baseline_success", t)
        g = max_wr_at_or_above(wr_df, "guardrail_success", t)
        gp = max_wr_at_or_above(wr_df, "guardrail_put_success", t)

        def fmt(x):
            return "None" if x is None else f"{x:.2%}"

        print(f"Target success >= {t:.0%}")
        print(f"  Baseline:           max WR = {fmt(b)}")
        print(f"  Guardrail:          max WR = {fmt(g)}")
        print(f"  Guardrail + Put:    max WR = {fmt(gp)}")

    wr_out = BASE_DIR / "step7_wr_sweep_90_95.csv"
    wr_df.to_csv(wr_out, index=False)
    print("Saved:", wr_out)

    # -----------------------
    # FINAL SUMMARY (PPT-ready) + CSV
    # -----------------------
    final_rows = [
        {"metric": "Baseline success @ WR=4%", "value": float(baseline_best_success)},
        {"metric": "Put success @ WR=4%", "value": float(put_success)},
        {"metric": "Collar success @ WR=4%", "value": float(collar_success)},
        {"metric": "Guardrail success @ WR=4%", "value": float(guardrail_success)},
        {"metric": "Guardrail+Put success @ WR=4%", "value": float(guardrail_put_success)},
        {"metric": "Best Grid (Guardrail+Put) success", "value": float(BEST_SR)},
        {"metric": "Best Grid params", "value": f"hy={BEST_HY}, strike={BEST_PS}, prem={BEST_PP}"},
        {"metric": "Max WR @ >=90% success (Baseline)", "value": max_wr_at_or_above(wr_df, "baseline_success", 0.90)},
        {"metric": "Max WR @ >=90% success (Guardrail)", "value": max_wr_at_or_above(wr_df, "guardrail_success", 0.90)},
        {"metric": "Max WR @ >=90% success (Guardrail+Put)", "value": max_wr_at_or_above(wr_df, "guardrail_put_success", 0.90)},
        {"metric": "Max WR @ >=95% success (Baseline)", "value": max_wr_at_or_above(wr_df, "baseline_success", 0.95)},
        {"metric": "Max WR @ >=95% success (Guardrail)", "value": max_wr_at_or_above(wr_df, "guardrail_success", 0.95)},
        {"metric": "Max WR @ >=95% success (Guardrail+Put)", "value": max_wr_at_or_above(wr_df, "guardrail_put_success", 0.95)},
    ]
    final_df = pd.DataFrame(final_rows)

    print("\n===== FINAL SUMMARY (for PPT) =====")
    for _, r in final_df.iterrows():
        metric = r["metric"]
        v = r["value"]
        if isinstance(v, (float, np.floating)) and not np.isnan(v):
            if 0 <= float(v) <= 1:
                print(f"{metric}: {float(v):.2%}")
            else:
                print(f"{metric}: {float(v)}")
        else:
            print(f"{metric}: {v}")

    final_out = BASE_DIR / "final_summary_for_ppt.csv"
    final_df.to_csv(final_out, index=False)
    print("Saved:", final_out)

    # -----------------------
    # SANITY CHECKS A + B
    # -----------------------
    # Sanity A: multi-seed, 10k paths (fixed best_weights)
    sanityA_seeds = [1_000_000, 2_000_000, 3_000_000]
    sanityA_rows = []
    for s in sanityA_seeds:
        cube_s, base_s = build_market_cube_from_seed(
            seed=s,
            time_horizon=time_horizon,
            n_samples=n_samples,
            n_assets=n_assets,
            L=L,
            v_mu=v_mu,
            start_year=start_year,
        )
        _, thr = compute_wr_thresholds(
            weights=best_weights,
            market_cube=cube_s,
            base_returns_by_year=base_s,
            start_year=start_year,
            time_horizon=time_horizon,
            n_samples=n_samples,
            inflation_rate=inflation_rate,
            wr_grid=wr_grid,
            targets=targets,
            equity_mask=equity_mask,
            hedge_alloc=hedge_alloc,
            best_hy=BEST_HY,
            best_ps=BEST_PS,
            best_pp=BEST_PP,
        )
        for t in targets:
            sanityA_rows.append(
                {
                    "seed": s,
                    "n_samples": n_samples,
                    "target": t,
                    "max_wr_baseline": thr[f"max_wr_baseline_{int(t*100)}"],
                    "max_wr_guardrail": thr[f"max_wr_guardrail_{int(t*100)}"],
                    "max_wr_guardrail_put": thr[f"max_wr_guardrail_put_{int(t*100)}"],
                }
            )

    sanityA_df = pd.DataFrame(sanityA_rows)
    print("\n===== SANITY A: Multi-seed max WR (10k paths; fixed best_weights) =====")
    print(sanityA_df.to_string(index=False))
    sanityA_out = BASE_DIR / "sanityA_multiseed_maxwr_10k.csv"
    sanityA_df.to_csv(sanityA_out, index=False)
    print("Saved:", sanityA_out)

    # Sanity B: higher n_samples, 50k paths (single seed)
    n_samples_hi = 50_000
    seed_hi = 1_000_000
    cube_hi, base_hi = build_market_cube_from_seed(
        seed=seed_hi,
        time_horizon=time_horizon,
        n_samples=n_samples_hi,
        n_assets=n_assets,
        L=L,
        v_mu=v_mu,
        start_year=start_year,
    )
    wr_hi_df, _ = compute_wr_thresholds(
        weights=best_weights,
        market_cube=cube_hi,
        base_returns_by_year=base_hi,
        start_year=start_year,
        time_horizon=time_horizon,
        n_samples=n_samples_hi,
        inflation_rate=inflation_rate,
        wr_grid=wr_grid,
        targets=targets,
        equity_mask=equity_mask,
        hedge_alloc=hedge_alloc,
        best_hy=BEST_HY,
        best_ps=BEST_PS,
        best_pp=BEST_PP,
    )

    print("\n===== SANITY B: Max WR with 50k paths (seed=1,000,000; fixed best_weights) =====")
    for t in targets:
        b = max_wr_at_or_above(wr_hi_df, "baseline_success", t)
        g = max_wr_at_or_above(wr_hi_df, "guardrail_success", t)
        gp = max_wr_at_or_above(wr_hi_df, "guardrail_put_success", t)

        def fmt(x):
            return "None" if x is None else f"{x:.2%}"

        print(f"Target success >= {t:.0%}")
        print(f"  Baseline:        max WR = {fmt(b)}")
        print(f"  Guardrail:       max WR = {fmt(g)}")
        print(f"  Guardrail+Put:   max WR = {fmt(gp)}")

    sanityB_out = BASE_DIR / "sanityB_wr_sweep_50k.csv"
    wr_hi_df.to_csv(sanityB_out, index=False)
    print("Saved:", sanityB_out)

    # -----------------------
    # STEP 8A: Determinism check (same seed, run twice)
    # -----------------------
    cube1, base1 = build_market_cube_from_seed(
        seed=SEED,
        time_horizon=time_horizon,
        n_samples=n_samples,
        n_assets=n_assets,
        L=L,
        v_mu=v_mu,
        start_year=start_year,
    )
    cube2, base2 = build_market_cube_from_seed(
        seed=SEED,
        time_horizon=time_horizon,
        n_samples=n_samples,
        n_assets=n_assets,
        L=L,
        v_mu=v_mu,
        start_year=start_year,
    )

    _, thr1 = compute_wr_thresholds(
        weights=best_weights,
        market_cube=cube1,
        base_returns_by_year=base1,
        start_year=start_year,
        time_horizon=time_horizon,
        n_samples=n_samples,
        inflation_rate=inflation_rate,
        wr_grid=wr_grid,
        targets=targets,
        equity_mask=equity_mask,
        hedge_alloc=hedge_alloc,
        best_hy=BEST_HY,
        best_ps=BEST_PS,
        best_pp=BEST_PP,
    )
    _, thr2 = compute_wr_thresholds(
        weights=best_weights,
        market_cube=cube2,
        base_returns_by_year=base2,
        start_year=start_year,
        time_horizon=time_horizon,
        n_samples=n_samples,
        inflation_rate=inflation_rate,
        wr_grid=wr_grid,
        targets=targets,
        equity_mask=equity_mask,
        hedge_alloc=hedge_alloc,
        best_hy=BEST_HY,
        best_ps=BEST_PS,
        best_pp=BEST_PP,
    )

    print("\n===== STEP 8A: DETERMINISM CHECK (same seed twice) =====")
    print("Run 1 thresholds:", thr1)
    print("Run 2 thresholds:", thr2)
    print("Match:", thr1 == thr2)

    # -----------------------
    # STEP 8B: Stress tests (lower mean, higher vol)
    # -----------------------
    stress_cases = [
        {"name": "baseline_assumptions", "mu_shift": 0.0, "sigma_mult": 1.0, "seed": SEED},
        {"name": "lower_mean_-1pct", "mu_shift": -0.01, "sigma_mult": 1.0, "seed": SEED},
        {"name": "higher_vol_1p25x", "mu_shift": 0.0, "sigma_mult": 1.25, "seed": SEED},
    ]

    stress_rows = []
    for case in stress_cases:
        v_mu_case = v_mu + case["mu_shift"]
        v_sigma_case = v_sigma * case["sigma_mult"]

        cov_case = np.diag(v_sigma_case) @ corr @ np.diag(v_sigma_case)
        cov_case = make_positive_semi_definite(cov_case)
        L_case = np.linalg.cholesky(cov_case)

        cube_s, base_s = build_market_cube_from_seed(
            seed=case["seed"],
            time_horizon=time_horizon,
            n_samples=n_samples,
            n_assets=n_assets,
            L=L_case,
            v_mu=v_mu_case,
            start_year=start_year,
        )

        _, thr = compute_wr_thresholds(
            weights=best_weights,
            market_cube=cube_s,
            base_returns_by_year=base_s,
            start_year=start_year,
            time_horizon=time_horizon,
            n_samples=n_samples,
            inflation_rate=inflation_rate,
            wr_grid=wr_grid,
            targets=targets,
            equity_mask=equity_mask,
            hedge_alloc=hedge_alloc,
            best_hy=BEST_HY,
            best_ps=BEST_PS,
            best_pp=BEST_PP,
        )

        stress_rows.append(
            {
                "case": case["name"],
                "mu_shift": case["mu_shift"],
                "sigma_mult": case["sigma_mult"],
                **thr,
            }
        )

    stress_df = pd.DataFrame(stress_rows)
    print("\n===== STEP 8B: STRESS TEST WR THRESHOLDS =====")
    print(stress_df.to_string(index=False))

    stress_out = BASE_DIR / "step8_stress_test_thresholds.csv"
    stress_df.to_csv(stress_out, index=False)
    print("Saved:", stress_out)

    # -----------------------
    # Weight characteristics + outputs
    # -----------------------
    summary = compute_group_summaries(results, assets, top_frac=0.10)
    print("\n===== WEIGHT CHARACTERISTICS: TOP 10% vs BOTTOM 10% =====")
    print(summary.to_string(index=False))

    results.to_csv(BASE_DIR / "ppt_versionA_portfolio_results.csv", index=False)
    summary.to_csv(BASE_DIR / "ppt_versionA_top_vs_bottom_summary.csv", index=False)
    print("\nSaved:")
    print(" - ppt_versionA_portfolio_results.csv")
    print(" - ppt_versionA_top_vs_bottom_summary.csv")


if __name__ == "__main__":
    main()