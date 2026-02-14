"""
PPT Version A (baseline): "Test allocations"

Implements:
1) Generate many candidate portfolio weight vectors (5,000)
2) For each portfolio, run retirement Monte Carlo and compute:
      success% = fraction of paths that never drop below $0
3) Identify the weights with the highest success%
4) Summarize characteristics of successful vs failed portfolios (top vs bottom deciles)

Inputs (CSV files in same folder as this script):
- balances_and_target_allocations.csv
- returns_std_devs.csv
- index_correlations.csv

Assumptions (per user):
- Starting wealth = $1,000,000 (not necessarily equal to sum of balances file)
- Withdrawal rate WR = 4% of initial wealth, inflated by 3% per year
- Time horizon = 50 years
- Monte Carlo paths = n_samples
- Seed = 1,000,000

NOTE ON TICKERS:
- If returns_std_devs.csv uses "v" or "vmo" for VWO, we map it to "vwo" at load time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path


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
n_samples = 10_000  # you can change to 5_000 if runtime is slow

current_year = datetime.now().year


# -----------------------
# Utilities
# -----------------------
def make_positive_semi_definite(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, a_min=1e-10, a_max=None)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


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
    Success = portfolio never hits 0 during horizon.
    Withdraw at start of year; spending inflates each year.
    Rebalance to target weights annually (implicit via outer-product).
    """
    spending = withdrawal_rate * beg_net_worth
    alive = np.ones(n_samples, dtype=bool)
    net_worth = np.full(n_samples, beg_net_worth, dtype=float)

    for j in range(time_horizon):
        net_worth[alive] -= spending
        alive &= (net_worth > 0)

        if not np.any(alive):
            return 0.0

        beg_balances = np.outer(net_worth, weights)  # (n_samples, n_assets)
        gross = market_cube_returns_plus1[f"Market_Returns_{start_year + j}"].T  # (n_samples, n_assets)
        end_balances = beg_balances * gross
        net_worth = end_balances.sum(axis=1)

        spending *= (1 + inflation_rate)

    return float(alive.mean())


def sample_dirichlet_weights(n_portfolios: int, n_assets: int) -> np.ndarray:
    """
    Simple long-only weight generator: each portfolio weights sum to 1.
    Dirichlet(alpha=1) gives uniform random points on the simplex.
    """
    alpha = np.ones(n_assets)
    return np.random.dirichlet(alpha, size=n_portfolios)


def compute_group_summaries(results_df: pd.DataFrame, assets: list[str], top_frac: float = 0.10) -> pd.DataFrame:
    """
    Compare average weights for top vs bottom fraction of portfolios by success.
    Returns a summary dataframe with columns:
      top_mean, bottom_mean, top_minus_bottom
    """
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

    # Normalize tickers
    for df in (df_returns, df_corr, df_alloc):
        df["Index_Fund"] = df["Index_Fund"].astype(str).str.strip().str.lower()

    # Map common naming mismatch to vwo (supports earlier "v" or "vmo")
    df_returns["Index_Fund"] = df_returns["Index_Fund"].replace({"v": "vwo", "vmo": "vwo"})

    # Use allocation file as "asset universe"
    assets = df_alloc["Index_Fund"].to_list()
    n_assets = len(assets)

    # Align returns and correlations to assets
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
    # Generate ONE market cube (same for all portfolios)
    # -----------------------
    Z_by_year = [np.random.normal(size=(n_samples, n_assets)) for _ in range(time_horizon)]
    base_returns_by_year = [(Z @ L.T + v_mu) for Z in Z_by_year]

    market_cube = {}
    for i in range(time_horizon):
        market_cube[f"Market_Returns_{current_year + i}"] = (base_returns_by_year[i] + 1.0).T

    # -----------------------
    # Generate candidate portfolios
    # -----------------------
    W = sample_dirichlet_weights(N_PORTFOLIOS, n_assets)  # (N_PORTFOLIOS, n_assets)

    # (Optional) define equity set for summary metrics
    equity_funds = {"voo", "vb", "vo", "vv", "vtv", "vti", "vwo"}
    equity_mask = np.array([a in equity_funds for a in assets], dtype=bool)

    # -----------------------
    # Evaluate each portfolio
    # -----------------------
    rows = []
    for i in range(N_PORTFOLIOS):
        w = W[i]
        sr = simulate_success_rate_fixed_weights(
            weights=w,
            withdrawal_rate=WR_FIXED,
            beg_net_worth=INITIAL_WEALTH,
            market_cube_returns_plus1=market_cube,
            start_year=current_year,
            time_horizon=time_horizon,
            n_samples=n_samples,
            inflation_rate=inflation_rate,
        )

        eq_w = float(w[equity_mask].sum()) if np.any(equity_mask) else float("nan")

        row = {
            "portfolio_id": i,
            "success_rate": sr,
            "equity_weight": eq_w,
        }
        # store weights for later summaries
        for a, wa in zip(assets, w):
            row[f"w_{a}"] = float(wa)
        rows.append(row)

        # light progress
        if (i + 1) % 500 == 0:
            print(f"  evaluated {i+1}/{N_PORTFOLIOS} portfolios...")

    results = pd.DataFrame(rows).sort_values("success_rate", ascending=False).reset_index(drop=True)

    # -----------------------
    # Report best portfolio and top 10
    # -----------------------
    best = results.iloc[0]
    print("\n===== BEST PORTFOLIO (by success rate) =====")
    print(f"portfolio_id={int(best['portfolio_id'])}  success={best['success_rate']:.2%}  equity_weight={best['equity_weight']:.2%}")
    for a in assets:
        print(f"  {a}: {best[f'w_{a}']:.4f}")

    print("\n===== TOP 10 PORTFOLIOS =====")
    top10 = results.head(10)[["portfolio_id", "success_rate", "equity_weight"]]
    print(top10.to_string(index=False))

    # -----------------------
    # Summarize characteristics: top 10% vs bottom 10%
    # -----------------------
    summary = compute_group_summaries(results, assets, top_frac=0.10)
    print("\n===== WEIGHT CHARACTERISTICS: TOP 10% vs BOTTOM 10% =====")
    print(summary.to_string(index=False))

    # Save outputs for your presentation
    results.to_csv(BASE_DIR / "ppt_versionA_portfolio_results.csv", index=False)
    summary.to_csv(BASE_DIR / "ppt_versionA_top_vs_bottom_summary.csv", index=False)
    print("\nSaved:")
    print(" - ppt_versionA_portfolio_results.csv")
    print(" - ppt_versionA_top_vs_bottom_summary.csv")


if __name__ == "__main__":
    main()

# --- STEP 1: Save best baseline portfolio weights + re-evaluate baseline at WR_FIXED ---

best = results.iloc[0]
best_id = int(best["portfolio_id"])

best_weights = np.array([best[f"w_{a}"] for a in assets], dtype=float)
best_weights = best_weights / best_weights.sum()

baseline_best_success = simulate_success_rate_fixed_weights(
    weights=best_weights,
    withdrawal_rate=WR_FIXED,
    beg_net_worth=INITIAL_WEALTH,
    market_cube_returns_plus1=market_cube,
    start_year=current_year,
    time_horizon=time_horizon,
    n_samples=n_samples,
    inflation_rate=inflation_rate,
)

print("\n===== STEP 1: BASELINE BEST PORTFOLIO RE-RUN =====")
print(f"Best portfolio_id={best_id}")
print(f"Baseline success at WR={WR_FIXED:.2%}: {baseline_best_success:.2%}")

best_weights_df = pd.DataFrame(
    {"Index_Fund": assets, "Weight": best_weights}
).sort_values("Weight", ascending=False)

best_weights_path = BASE_DIR / "best_portfolio_weights_versionA.csv"
best_weights_df.to_csv(best_weights_path, index=False)
print("Saved best weights to:", best_weights_path)