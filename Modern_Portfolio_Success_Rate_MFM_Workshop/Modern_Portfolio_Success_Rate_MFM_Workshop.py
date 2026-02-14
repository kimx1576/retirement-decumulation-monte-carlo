# Fixed-weights withdrawal-rate sweep Monte Carlo
#
# Updates:
# - Uses your CSVs with tickers: voo, bsv, biv, blv, vb, vo, vv, vtv, vti, vwo
# - Fixes the returns file mismatch by mapping "vmo" -> "vwo" (per your note: you changed v -> vmo)
# - Applies an annual-reset collar ONLY to the equity sleeve (more realistic)
# - Leaves bonds (bsv, biv, blv) unhedged
#
# IMPORTANT:
# - Your correlation matrix and allocations contain "vwo".
# - Your returns_std_devs.csv currently ends with "vmo".
#   This script maps "vmo" to "vwo" at load time so all three datasets align.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# -----------------------
# Parameters
# -----------------------
SEED = 1_000_000
np.random.seed(SEED)

time_horizon = 50
n_samples = 10_000
inflation_rate = 0.03
current_year = datetime.now().year
TARGET_SUCCESS = 0.95

# -----------------------
# Equity-only collar settings (annual reset)
# -----------------------
hedge_years = 15  # collar only in first N years (sequence-of-returns risk)

# strike levels expressed in return space (proxy)
put_strike_return = -0.10     # 90% put
call_strike_return = 0.15     # 115% call

# annual premium assumptions (as fraction of hedged equity sleeve)
put_premium_rate = 0.01
call_premium_rate = 0.03

# hedge fraction of equity sleeve (0..1)
equity_hedge_allocation_when_on = 1.0


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
    Success = portfolio never hits 0 during horizon (withdraw at start of year).
    Withdrawal rule = constant real spending:
      spending_0 = WR * initial_wealth, then inflate by inflation_rate each year.
    """
    spending = withdrawal_rate * beg_net_worth
    alive = np.ones(n_samples, dtype=bool)
    net_worth = np.full(n_samples, beg_net_worth, dtype=float)

    for j in range(time_horizon):
        # Withdraw at start of year
        net_worth[alive] -= spending
        alive &= (net_worth > 0)

        if not np.any(alive):
            return 0.0

        # Rebalance and apply returns
        beg_balances = np.outer(net_worth, weights)  # (n_samples, n_assets)
        gross = market_cube_returns_plus1[f"Market_Returns_{start_year + j}"].T  # (n_samples, n_assets)
        end_balances = beg_balances * gross
        net_worth = end_balances.sum(axis=1)

        # Inflate next year's spending
        spending *= (1 + inflation_rate)

    return float(alive.mean())


def run_withdrawal_rate_sweep(
    *,
    label: str,
    weights: np.ndarray,
    beg_net_worth: float,
    market_cube_returns_plus1: dict,
    wr_grid: np.ndarray,
    time_horizon: int,
    n_samples: int,
    inflation_rate: float,
    current_year: int,
):
    success_rates = []

    print(f"\nWithdrawal Rate Sweep (fixed weights) | {label}:")
    for wr in wr_grid:
        sr = simulate_success_rate_fixed_weights(
            weights=weights,
            withdrawal_rate=float(wr),
            beg_net_worth=beg_net_worth,
            market_cube_returns_plus1=market_cube_returns_plus1,
            start_year=current_year,
            time_horizon=time_horizon,
            n_samples=n_samples,
            inflation_rate=inflation_rate,
        )
        success_rates.append(sr)
        print(f"  WR={wr:.2%}  success={sr:.2%}")

    return np.array(success_rates, dtype=float)


def best_wr_meeting_target(wr_grid, success_rates, target=0.95):
    candidates = [(wr, sr) for wr, sr in zip(wr_grid, success_rates) if sr >= target]
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])  # (best_wr, success)


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
    """
    Apply an annual-reset collar to the EQUITY sleeve only.

    returns: (n_samples, n_assets) simple returns for the year
    weights: (n_assets,) target weights (used to compute equity sleeve return)
    equity_mask: boolean mask of equity assets
    hedge_allocation: 0..1 how much of equity sleeve is hedged

    Implementation detail:
    - Compute equity sleeve return as weighted return of equity assets (normalized weights).
    - Apply collar payoff in return space:
        put_payoff  = max(put_strike - R_eq, 0)
        call_payoff = -max(R_eq - call_strike, 0)  (short call)
      then add/subtract premia.
    - Map the resulting equity sleeve return back onto each equity asset
      (simple but consistent for an index-like hedge).
    """
    if hedge_allocation <= 0.0:
        return returns

    w_eq = weights[equity_mask]
    w_eq_sum = float(w_eq.sum())
    if w_eq_sum <= 0:
        return returns

    w_eq_norm = w_eq / w_eq_sum
    r_eq = returns[:, equity_mask] @ w_eq_norm  # (n_samples,)

    put_payoff = np.maximum(put_strike_return - r_eq, 0.0)
    call_payoff = -np.maximum(r_eq - call_strike_return, 0.0)

    r_eq_hedged = (
        r_eq
        + put_payoff
        + call_payoff
        - put_premium_rate
        + call_premium_rate
    )

    r_eq_final = hedge_allocation * r_eq_hedged + (1 - hedge_allocation) * r_eq

    out = returns.copy()
    out[:, equity_mask] = r_eq_final[:, None]
    return out


def main():
    BASE_DIR = Path(__file__).resolve().parent

    # Load CSVs (use your exact file names)
    df_returns = pd.read_csv(BASE_DIR / "Returns_Std_Devs.csv")
    df_corr = pd.read_csv(BASE_DIR / "Index_Correlations.csv")
    df_alloc = pd.read_csv(BASE_DIR / "Balances_and_Target_Allocations.csv")

    # ---- FIX: map returns ticker "vmo" -> "vwo" so all files align
    # (You said you updated v to vmo; correlations/allocations still use vwo.)
    df_returns["Index_Fund"] = df_returns["Index_Fund"].astype(str).str.lower().replace({"vmo": "vwo"})

    # Asset universe/order: take from allocation file (source of truth for weights)
    assets = df_alloc["Index_Fund"].astype(str).str.lower().to_list()
    df_alloc["Index_Fund"] = df_alloc["Index_Fund"].astype(str).str.lower()

    # Align returns to assets
    df_returns["Index_Fund"] = df_returns["Index_Fund"].astype(str).str.lower()
    df_returns = df_returns.set_index("Index_Fund").reindex(assets)

    if df_returns.isna().any().any():
        missing = df_returns[df_returns.isna().any(axis=1)].index.to_list()
        raise ValueError(
            f"Missing Mean/Standard_Deviation for: {missing}. "
            f"Your returns_std_devs.csv must contain these tickers."
        )

    v_mu = df_returns["Mean"].astype(float).to_numpy()
    v_sigma = df_returns["Standard_Deviation"].astype(float).to_numpy()

    # Align correlation matrix to assets
    df_corr["Index_Fund"] = df_corr["Index_Fund"].astype(str).str.lower()
    df_corr = df_corr.set_index("Index_Fund")
    corr_matrix = df_corr.reindex(index=assets, columns=assets).astype(float).to_numpy()
    corr_matrix = make_positive_semi_definite(corr_matrix)

    cov_matrix = np.diag(v_sigma) @ corr_matrix @ np.diag(v_sigma)
    L = np.linalg.cholesky(cov_matrix)

    # Weights / starting wealth
    weights = df_alloc["Target_Allocation"].astype(float).to_numpy()
    weights = weights / weights.sum()
    beg_net_worth = float(df_alloc["Balance"].sum())

    # Equity definition (options applied ONLY to these)
    equity_funds = {"voo", "vb", "vo", "vv", "vtv", "vti", "vwo"}
    equity_mask = np.array([a in equity_funds for a in assets], dtype=bool)

    if not np.any(equity_mask):
        raise ValueError("Equity mask is empty. Check equity fund tickers vs assets list.")

    # Pre-generate the same random shocks each run for fair comparison
    Z_by_year = [np.random.normal(size=(n_samples, len(assets))) for _ in range(time_horizon)]
    base_returns_by_year = [(Z @ L.T + v_mu) for Z in Z_by_year]

    market_cube_baseline = {}
    market_cube_collar = {}

    for i in range(time_horizon):
        r = base_returns_by_year[i]

        # Baseline: no options
        market_cube_baseline[f"Market_Returns_{current_year + i}"] = (r + 1.0).T

        # Collar: equity-only, early years only
        hedge_alloc = equity_hedge_allocation_when_on if i < hedge_years else 0.0
        r2 = apply_equity_collar_overlay(
            r,
            weights=weights,
            equity_mask=equity_mask,
            hedge_allocation=hedge_alloc,
            put_strike_return=put_strike_return,
            call_strike_return=call_strike_return,
            put_premium_rate=put_premium_rate,
            call_premium_rate=call_premium_rate,
        )
        market_cube_collar[f"Market_Returns_{current_year + i}"] = (r2 + 1.0).T

    wr_grid = np.arange(0.02, 0.061, 0.0025)

    baseline_success = run_withdrawal_rate_sweep(
        label=f"Baseline (seed={SEED})",
        weights=weights,
        beg_net_worth=beg_net_worth,
        market_cube_returns_plus1=market_cube_baseline,
        wr_grid=wr_grid,
        time_horizon=time_horizon,
        n_samples=n_samples,
        inflation_rate=inflation_rate,
        current_year=current_year,
    )

    collar_success = run_withdrawal_rate_sweep(
        label=f"Equity-only collar (first {hedge_years}y, seed={SEED})",
        weights=weights,
        beg_net_worth=beg_net_worth,
        market_cube_returns_plus1=market_cube_collar,
        wr_grid=wr_grid,
        time_horizon=time_horizon,
        n_samples=n_samples,
        inflation_rate=inflation_rate,
        current_year=current_year,
    )

    b_best = best_wr_meeting_target(wr_grid, baseline_success, TARGET_SUCCESS)
    c_best = best_wr_meeting_target(wr_grid, collar_success, TARGET_SUCCESS)

    print("\n===== SUMMARY =====")
    if b_best:
        print(f"Baseline best WR >= {TARGET_SUCCESS:.0%}: {b_best[0]:.2%} (success={b_best[1]:.2%})")
    else:
        print("Baseline: no WR met target in grid")

    if c_best:
        print(f"Collar best WR >= {TARGET_SUCCESS:.0%}: {c_best[0]:.2%} (success={c_best[1]:.2%})")
    else:
        print("Collar: no WR met target in grid")

    # Plot
    plt.plot(wr_grid * 100, baseline_success * 100, marker="o", label="Baseline")
    plt.plot(wr_grid * 100, collar_success * 100, marker="o", label=f"Equity-only collar (first {hedge_years}y)")
    plt.axhline(TARGET_SUCCESS * 100, color="red", linestyle="--", label=f"{int(TARGET_SUCCESS*100)}% target")
    plt.xlabel("Withdrawal Rate (%)")
    plt.ylabel("Success Rate (%)")
    plt.title("Withdrawal Rate vs Success Rate (Fixed Weights)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()