from __future__ import annotations

from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.figure import Figure

from agents.base import PRODUCTS as DEFAULT_PRODUCTS


_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def compute_oracle_cost(
    results: list,
    alpha: np.ndarray,
    initial_inventory: np.ndarray,
    our_agent_id: int = 0,
) -> np.ndarray:
    """LP relaxation of the offline ILP — clairvoyant lower bound on cost."""
    from scipy.optimize import linprog

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

        A_ub_full = np.vstack([A_ub, -quantities.reshape(1, -1)])
        b_ub_full = np.append(
            b_ub,
            -(alpha[i] - initial_inventory[i] + depletions.sum()),
        )

        bounds = [(0.0, 1.0 if active[t] else 0.0) for t in range(T)]
        res = linprog(c=prices, A_ub=A_ub_full, b_ub=b_ub_full,
                      bounds=bounds, method="highs")

        oracle_x[:, i] = np.clip(res.x, 0.0, 1.0) if res.success else active.astype(float)

    prices_matrix = np.array([[r.winning_bids[i] for i in range(N)] for r in results])
    return (prices_matrix * oracle_x).sum(axis=1)


def compute_regret(
    results: list,
    oracle_round_costs: np.ndarray,
    alpha: np.ndarray,
    violation_penalty: float | None = None,
    our_agent_id: int = 0,
) -> tuple[np.ndarray, float]:
    """Cumulative penalised regret: agent cost + inventory deficit penalty - oracle cost."""
    N = len(alpha)

    if violation_penalty is None:
        all_prices = np.array([[r.winning_bids[i] for i in range(N)] for r in results])
        violation_penalty = max(float(np.max(all_prices)) if all_prices.size > 0 else 1.0, 1.0)

    agent_costs = np.array([r.our_cost for r in results], dtype=float)
    violation_cost = np.array([
        violation_penalty * np.maximum(0.0, alpha - r.inventories_after[our_agent_id]).sum()
        for r in results
    ])

    return np.cumsum(agent_costs + violation_cost - oracle_round_costs), violation_penalty


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
    """Average regret Reg(T)/T per round — converges to 0 for no-regret algorithms."""
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

    if save_path:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [viz] Regret plot saved -> {save_path}")

    if show:
        plt.show()
    return fig


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
    print("  [regret] Computing clairvoyant oracle via LP ...")
    oracle_costs = compute_oracle_cost(results_all_seeing, alpha, initial_inventory)
    print(f"  [regret] Oracle total cost      : {oracle_costs.sum():,.2f}")

    if violation_penalty is None:
        all_prices = [r.winning_bids for run in (results_all_seeing, results_partially_blind) for r in run]
        violation_penalty = max(float(np.max(all_prices)) if all_prices else 1.0, 1.0)
    print(f"  [regret] Violation penalty (lambda)  : {violation_penalty:,.2f}")

    regret_as, _ = compute_regret(results_all_seeing,    oracle_costs, alpha, violation_penalty)
    regret_pb, _ = compute_regret(results_partially_blind, oracle_costs, alpha, violation_penalty)

    print(f"  [regret] All-Seeing final regret     : {regret_as[-1]:,.2f}")
    print(f"  [regret] Partially-Blind final regret: {regret_pb[-1]:,.2f}")

    return plot_regret_vs_theory(
        runs=[
            ("All-Seeing",      results_all_seeing,      regret_as),
            ("Partially-Blind", results_partially_blind, regret_pb),
        ],
        alpha=alpha,
        V=V,
        n_products=len(alpha),
        n_competitors=n_competitors,
        context_dim=len(alpha),
        save_path=save_path,
        show=show,
    )
