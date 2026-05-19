"""
Regret computation and visualization for the auction simulation.

Regret Definition
-----------------
At each round t the agent pays ``cost_t``.  The **oracle** (clairvoyant
offline baseline) solves the per-product LP relaxation of the ILP in
Section IV-A, knowing all future prices and quantities in advance, and
finds the minimum-cost purchasing schedule that keeps inventory >= alpha
throughout the horizon.

    Regret(T) = sum_{t=1}^{T} cost_agent(t) - cost_oracle(T)

Because we use the LP relaxation the oracle cost is a *lower bound* on
the true ILP optimum, making our measured regret an **upper bound** on
true regret -- the standard conservative convention in online learning.

Public API
----------
    compute_oracle_cost(results, alpha, initial_inventory) -> np.ndarray
    compute_regret(results, oracle_round_costs, alpha)     -> np.ndarray
    plot_regret_vs_theory(runs, alpha, ...)                -> Figure
    compare_and_plot_regret(...)                           -> Figure
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.figure import Figure

from agents.base import PRODUCTS as DEFAULT_PRODUCTS


_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


# ════════════════════════════════════════════════════════════════════════
#  Oracle (clairvoyant offline baseline)
# ════════════════════════════════════════════════════════════════════════

def compute_oracle_cost(
    results: list,
    alpha: np.ndarray,
    initial_inventory: np.ndarray,
    our_agent_id: int = 0,
) -> np.ndarray:
    """Compute the per-round cost of the clairvoyant oracle.

    The oracle knows all future prices and quantities.  For each product
    it solves the LP relaxation of the offline ILP (Section IV-A):

        min  sum_t  p_t * x_t
        s.t. s_0 + sum_{s<t} q_s * x_s - sum_{s<t} d_s  >= alpha   for all t
             0 <= x_t <= 1

    Products are separable so we solve independently per product.

    Returns
    -------
    oracle_round_costs : np.ndarray, shape (T,)
    """
    try:
        from scipy.optimize import linprog
    except ImportError as e:
        raise ImportError(
            "scipy is required for regret computation. "
            "Install with: pip install scipy"
        ) from e

    T = len(results)
    N = len(alpha)
    oracle_x = np.zeros((T, N))

    for i in range(N):
        prices     = np.array([r.winning_bids[i] for r in results])
        quantities = np.array([r.quantities[i]   for r in results])
        depletions = np.array([r.depletions[i]   for r in results])

        prices = np.where(prices <= 0, 1e-6, prices)
        active = quantities > 0

        cum_dep = np.concatenate([[0.0], np.cumsum(depletions[:-1])])

        A_ub = np.zeros((T, T))
        b_ub = np.zeros(T)
        for t in range(T):
            A_ub[t, :t] = -quantities[:t]
            b_ub[t]     = -(alpha[i] - initial_inventory[i] + cum_dep[t])

        A_ub_full = np.vstack([
            A_ub,
            -quantities.reshape(1, -1),
        ])
        total_dep = depletions.sum()
        b_ub_full = np.append(
            b_ub,
            -(alpha[i] - initial_inventory[i] + total_dep),
        )

        bounds = [(0.0, 1.0 if active[t] else 0.0) for t in range(T)]

        res = linprog(
            c=prices,
            A_ub=A_ub_full,
            b_ub=b_ub_full,
            bounds=bounds,
            method="highs",
        )

        if res.success:
            oracle_x[:, i] = np.clip(res.x, 0.0, 1.0)
        else:
            oracle_x[:, i] = active.astype(float)

    prices_matrix = np.array([
        [r.winning_bids[i] for i in range(N)]
        for r in results
    ])
    oracle_round_costs = (prices_matrix * oracle_x).sum(axis=1)
    return oracle_round_costs


# ════════════════════════════════════════════════════════════════════════
#  Regret computation
# ════════════════════════════════════════════════════════════════════════

def compute_regret(
    results: list,
    oracle_round_costs: np.ndarray,
    alpha: np.ndarray,
    violation_penalty: float | None = None,
    our_agent_id: int = 0,
) -> np.ndarray:
    """Compute the cumulative penalized regret series.

    penalised_cost(t) = agent_cost(t) + lambda * sum_i max(0, alpha_i - s_i(t))

    lambda defaults to the maximum market price seen across the episode.

    Returns
    -------
    (cumulative_regret, violation_penalty) : (np.ndarray shape (T,), float)
    """
    N = len(alpha)

    if violation_penalty is None:
        all_prices = np.array([
            [r.winning_bids[i] for i in range(N)]
            for r in results
        ])
        violation_penalty = float(np.max(all_prices)) if all_prices.size > 0 else 1.0
        violation_penalty = max(violation_penalty, 1.0)

    agent_costs    = np.array([r.our_cost for r in results], dtype=float)
    violation_cost = np.zeros(len(results), dtype=float)

    for t, r in enumerate(results):
        inv = r.inventories_after[our_agent_id]
        deficit = np.maximum(0.0, alpha - inv)
        violation_cost[t] = violation_penalty * deficit.sum()

    per_round = (agent_costs + violation_cost) - oracle_round_costs
    return np.cumsum(per_round), violation_penalty


# ════════════════════════════════════════════════════════════════════════
#  Main visualization
# ════════════════════════════════════════════════════════════════════════

def plot_regret_vs_theory(
    runs: list[tuple[str, list, np.ndarray]],
    alpha: np.ndarray,
    V: float = 2.0,
    n_products: int = 4,
    n_competitors: int = 3,
    context_dim: int | None = None,
    save_path: str | None = None,
    show: bool = False,
) -> Figure:
    """Two-panel regret plot (standard in online learning literature).

    Left  — Cumulative regret Reg(T): all agents on the same axes.
             Sublinear growth => no-regret algorithm.

    Right — Average regret Reg(T)/T: should converge to 0 for any
             no-regret algorithm.  No scaling or bound fitting needed;
             the convergence to zero is self-evident.

    Parameters
    ----------
    runs : list of (label, results, cum_regret)
        label      : str
        results    : list[RoundResult]
        cum_regret : np.ndarray (T,) from compute_regret()
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for idx, (label, _results, cum_regret) in enumerate(runs):
        T = len(cum_regret)
        t_ax = np.arange(1, T + 1)
        color = _PALETTE[idx % len(_PALETTE)]
        avg_regret = cum_regret / t_ax

        ax.plot(t_ax, avg_regret, color=color, linewidth=2.2, label=label)
        ax.fill_between(t_ax, avg_regret, alpha=0.08, color=color)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Round $t$")
    ax.set_ylabel(r"Average regret  $\mathrm{Reg}(T)/T$")
    ax.set_title(r"Average Regret per Round  (no-regret $\Rightarrow\ \to 0$)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [viz] Regret plot saved -> {save_path}")

    if show:
        plt.show()
    return fig


# ════════════════════════════════════════════════════════════════════════
#  Convenience: run both agents and produce the comparison figure
# ════════════════════════════════════════════════════════════════════════

def compare_and_plot_regret(
    results_all_seeing: list,
    results_partially_blind: list,
    alpha: np.ndarray,
    initial_inventory: np.ndarray,
    products: Sequence[str] = DEFAULT_PRODUCTS,
    violation_penalty: float | None = None,
    save_path: str | None = None,
    show: bool = True,
    V: float = 2.0,
    n_competitors: int = 3,
) -> Figure:
    """Compute oracle baseline once, then plot regret for both agents."""
    print("  [regret] Computing clairvoyant oracle via LP ...")
    oracle_costs = compute_oracle_cost(
        results_all_seeing, alpha, initial_inventory
    )
    print(f"  [regret] Oracle total cost      : {oracle_costs.sum():,.2f}")

    if violation_penalty is None:
        all_prices = [
            r.winning_bids
            for run in (results_all_seeing, results_partially_blind)
            for r in run
        ]
        violation_penalty = float(np.max(all_prices)) if all_prices else 1.0
        violation_penalty = max(violation_penalty, 1.0)
    print(f"  [regret] Violation penalty (lambda)  : {violation_penalty:,.2f}")

    regret_as, _ = compute_regret(
        results_all_seeing, oracle_costs, alpha,
        violation_penalty=violation_penalty,
    )
    regret_pb, _ = compute_regret(
        results_partially_blind, oracle_costs, alpha,
        violation_penalty=violation_penalty,
    )

    print(f"  [regret] All-Seeing final regret     : {regret_as[-1]:,.2f}")
    print(f"  [regret] Partially-Blind final regret: {regret_pb[-1]:,.2f}")

    runs = [
        ("All-Seeing",       results_all_seeing,       regret_as),
        ("Partially-Blind",  results_partially_blind,  regret_pb),
    ]
    return plot_regret_vs_theory(
        runs=runs,
        alpha=alpha,
        V=V,
        n_products=len(alpha),
        n_competitors=n_competitors,
        context_dim=len(alpha),
        save_path=save_path,
        show=show,
    )
