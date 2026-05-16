"""
Regret computation and visualization for the auction simulation.

Regret Definition
-----------------
At each round t the agent pays ``cost_t``.  The **oracle** (clairvoyant
offline baseline) solves the per-product LP relaxation of the ILP in
Section IV-A, knowing all future prices and quantities in advance, and
finds the minimum-cost purchasing schedule that keeps inventory ≥ α
throughout the horizon.

    Regret(T) = Σ_{t=1}^{T} cost_agent(t) − cost_oracle(T)

Because we use the LP relaxation the oracle cost is a *lower bound* on
the true ILP optimum, making our measured regret an **upper bound** on
true regret — the standard conservative convention in online learning.

For the **Partially-Blind** agent the regret is strictly ≥ the
All-Seeing regret because it carries both Lyapunov approximation error
AND OLS estimation error.  Plotting both curves on the same axes makes
the "cost of blindness" directly visible.

Public API
----------
    compute_oracle_cost(results, alpha, initial_inventory) -> np.ndarray
    compute_regret(results, oracle_round_costs)            -> np.ndarray
    plot_regret(runs, oracle_costs, ...)                   -> Figure
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from agents.base import PRODUCTS as DEFAULT_PRODUCTS


# ── colour palette (same as plots.py) ───────────────────────────────────
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

        min  Σ_t  p_t · x_t
        s.t. s_0 + Σ_{s<t} q_s · x_s − Σ_{s<t} d_s  ≥ α   ∀ t = 1…T
             0 ≤ x_t ≤ 1

    Products are separable so we solve independently per product.

    Parameters
    ----------
    results : list[RoundResult]
    alpha : np.ndarray, shape (N,)
        Hard inventory minimum per product.
    initial_inventory : np.ndarray, shape (N,)
        Inventory at t = 0 (before any round).
    our_agent_id : int

    Returns
    -------
    oracle_round_costs : np.ndarray, shape (T,)
        Minimum cost the oracle would spend at each round.
        ``oracle_round_costs.sum()`` is the total oracle cost.
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
    oracle_x = np.zeros((T, N))     # buying decisions (LP relaxation)

    for i in range(N):
        # Price to win product i at round t  (market price = winning bid)
        prices     = np.array([r.winning_bids[i] for r in results])
        quantities = np.array([r.quantities[i]   for r in results])
        depletions = np.array([r.depletions[i]   for r in results])

        # If no one bid the price is 0; oracle pays a nominal ε to win
        prices = np.where(prices <= 0, 1e-6, prices)
        # If quantity is 0 there is nothing to buy — skip this round
        active = quantities > 0

        # Cumulative depletion up to (but not including) round t
        cum_dep = np.concatenate([[0.0], np.cumsum(depletions[:-1])])

        # Inventory must satisfy:
        #   init_inv + Σ_{s<t} q_s x_s - cum_dep[t] >= alpha
        # Rearranged to linprog's A_ub @ x <= b_ub convention:
        #   -Σ_{s<t} q_s x_s <= alpha - init_inv[i] + cum_dep[t]
        # (one row per time step)
        A_ub = np.zeros((T, T))
        b_ub = np.zeros(T)
        for t in range(T):
            A_ub[t, :t] = -quantities[:t]
            b_ub[t]     = -(alpha[i] - initial_inventory[i] + cum_dep[t])

        # Also need to satisfy at horizon end
        A_ub_full = np.vstack([
            A_ub,
            -quantities.reshape(1, -1),   # final constraint: all rounds considered
        ])
        total_dep = depletions.sum()
        b_ub_full = np.append(
            b_ub,
            -(alpha[i] - initial_inventory[i] + total_dep),
        )

        # Bounds: x_t in [0, 1], but 0 if quantity == 0
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
            # If infeasible (e.g. insufficient total supply), always buy
            oracle_x[:, i] = active.astype(float)

    # Oracle round cost = Σ_i  price_i(t) * oracle_x_i(t)
    prices_matrix = np.array([
        [r.winning_bids[i] for i in range(N)]
        for r in results
    ])                                          # shape (T, N)
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

    To make the comparison fair, constraint violations are penalised:

        penalised_cost(t) = agent_cost(t)
                          + λ × Σ_i max(0, α_i − s_i(t))

    where λ (``violation_penalty``) defaults to the maximum market price
    observed across the whole episode, so that failing to maintain a unit
    of inventory is at least as costly as buying it at the most expensive
    available round.

    This prevents agents from appearing to beat the oracle simply by
    skipping purchases and letting inventory deplete.

    Parameters
    ----------
    results : list[RoundResult]
    oracle_round_costs : np.ndarray, shape (T,)
        Per-round oracle cost from ``compute_oracle_cost``.
    alpha : np.ndarray, shape (N,)
        Hard inventory minimum per product.
    violation_penalty : float | None
        λ per unit of inventory deficit.  Defaults to the max winning-bid
        price seen in ``results``.
    our_agent_id : int

    Returns
    -------
    cumulative_regret : np.ndarray, shape (T,)
        Σ_{s≤t} (penalised_agent_cost[s] − oracle_cost[s])
    """
    N = len(alpha)

    # Default λ = max market price seen (conservative upper-bound on unit cost)
    if violation_penalty is None:
        all_prices = np.array([
            [r.winning_bids[i] for i in range(N)]
            for r in results
        ])
        violation_penalty = float(np.max(all_prices)) if all_prices.size > 0 else 1.0
        violation_penalty = max(violation_penalty, 1.0)   # never zero

    agent_costs    = np.array([r.our_cost for r in results], dtype=float)
    violation_cost = np.zeros(len(results), dtype=float)

    for t, r in enumerate(results):
        inv = r.inventories_after[our_agent_id]          # shape (N,)
        deficit = np.maximum(0.0, alpha - inv)           # units below α
        violation_cost[t] = violation_penalty * deficit.sum()

    per_round = (agent_costs + violation_cost) - oracle_round_costs
    return np.cumsum(per_round), violation_penalty


# ════════════════════════════════════════════════════════════════════════
#  Plot
# ════════════════════════════════════════════════════════════════════════

def plot_regret(
    runs: list[tuple[str, list, np.ndarray]],
    products: Sequence[str] = DEFAULT_PRODUCTS,
    theoretical_bound: bool = True,
    ax: Axes | None = None,
    show: bool = False,
    save_path: str | None = None,
) -> Figure:
    """Plot cumulative regret curves for one or more algorithm runs.

    Parameters
    ----------
    runs : list of (label, cumulative_regret, results)
        Each entry is a named run to plot:
        - ``label``              : str, e.g. "All-Seeing" / "Partially-Blind"
        - ``cumulative_regret``  : np.ndarray shape (T,) from ``compute_regret``
        - ``results``            : list[RoundResult] (used for per-round per-product breakdown)
    products : sequence of str
    theoretical_bound : bool
        Overlay O(√T) reference curve (scale-matched to the first run).
    ax : Axes | None
        Single axes for the main regret plot.  If None a new figure is created.
    show : bool
    save_path : str | None
        Save figure to this path if provided.

    Returns
    -------
    Figure
    """
    # ── layout ────────────────────────────────────────────────────────
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax_main, ax_per_round = axes
    else:
        fig = ax.get_figure()
        ax_main = ax
        ax_per_round = None

    # ── main: cumulative regret ───────────────────────────────────────
    for idx, (label, cum_regret, _results) in enumerate(runs):
        T   = len(cum_regret)
        t_ax = np.arange(1, T + 1)
        color = _PALETTE[idx % len(_PALETTE)]
        ax_main.plot(t_ax, cum_regret, label=label, color=color, linewidth=2.2)
        ax_main.fill_between(t_ax, cum_regret, alpha=0.08, color=color)

    # Optional O(√T) reference
    if theoretical_bound and runs:
        _, first_regret, _ = runs[0]
        T = len(first_regret)
        t_ax = np.arange(1, T + 1)
        # Scale to match the first run at T/2
        mid = max(first_regret[T // 2], 1.0)
        scale = mid / np.sqrt(T // 2)
        ref = scale * np.sqrt(t_ax)
        ax_main.plot(t_ax, ref, "k--", linewidth=1.2, alpha=0.5, label=r"$O(\sqrt{T})$ reference")

    ax_main.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax_main.set_xlabel("Round (t)")
    ax_main.set_ylabel("Cumulative regret (tonnes·price)")
    ax_main.set_title("Cumulative penalised regret vs clairvoyant oracle\n"
                      r"(agent cost + λ·deficit − oracle cost, λ = max market price)")
    ax_main.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax_main.legend(fontsize=9)
    ax_main.grid(True, alpha=0.3)

    # ── right: per-round incremental regret (stacked bars) ────────────
    if ax_per_round is not None:
        for idx, (label, cum_regret, _results) in enumerate(runs):
            T = len(cum_regret)
            t_ax = np.arange(1, T + 1)
            per_round = np.diff(np.concatenate([[0.0], cum_regret]))
            color = _PALETTE[idx % len(_PALETTE)]
            ax_per_round.bar(
                t_ax + idx * 0.35,
                per_round,
                width=0.35,
                label=label,
                color=color,
                alpha=0.7,
                edgecolor="none",
            )

        ax_per_round.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax_per_round.set_xlabel("Round (t)")
        ax_per_round.set_ylabel("Per-round regret")
        ax_per_round.set_title("Per-round regret  (positive = overpaid vs oracle)")
        ax_per_round.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax_per_round.legend(fontsize=9)
        ax_per_round.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [viz] Regret plot saved → {save_path}")

    if show:
        plt.show()
    return fig


# ════════════════════════════════════════════════════════════════════════
#  Convenience: run both agents, compute and plot regret
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
) -> Figure:
    """Compute oracle baseline once, then plot penalised regret for both algorithms.

    The oracle is computed from the All-Seeing results (same price/quantity
    trajectory) and used as a shared baseline for both agents.
    Violation penalties are shared (λ = max market price across both runs)
    so the comparison is fair.
    """
    print("  [regret] Computing clairvoyant oracle via LP …")
    oracle_costs = compute_oracle_cost(
        results_all_seeing, alpha, initial_inventory
    )
    print(f"  [regret] Oracle total cost      : {oracle_costs.sum():,.2f}")

    # Compute shared λ from both runs so the penalty is identical
    if violation_penalty is None:
        all_prices = [
            r.winning_bids for run in (results_all_seeing, results_partially_blind)
            for r in run
        ]
        violation_penalty = float(np.max(all_prices)) if all_prices else 1.0
        violation_penalty = max(violation_penalty, 1.0)
    print(f"  [regret] Violation penalty (λ)  : {violation_penalty:,.2f}")

    regret_as, _ = compute_regret(results_all_seeing,     oracle_costs, alpha,
                                   violation_penalty=violation_penalty)
    regret_pb, _ = compute_regret(results_partially_blind, oracle_costs, alpha,
                                   violation_penalty=violation_penalty)

    print(f"  [regret] All-Seeing final regret     : {regret_as[-1]:,.2f}")
    print(f"  [regret] Partially-Blind final regret: {regret_pb[-1]:,.2f}")

    T = len(regret_as)
    t_ax = np.arange(1, T + 1)
    
    # Extract dimensions for PB bound
    N = len(alpha)
    M = len(results_partially_blind[0].all_bids) - 1 if len(results_partially_blind) > 0 else 3
    d = N
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax_as, ax_pb, ax_loss = axes
    
    # ── 1) All-Seeing ───────────────────────────────────────────────
    ax_as.plot(t_ax, regret_as, label="All-Seeing", color=_PALETTE[0], linewidth=2.2)
    ax_as.fill_between(t_ax, regret_as, alpha=0.08, color=_PALETTE[0])
    
    mid_as = max(regret_as[T // 2], 1.0)
    as_shape = (t_ax / V) + V
    scale_as = mid_as / as_shape[T // 2]
    ref_as = scale_as * as_shape
    ax_as.plot(t_ax, ref_as, "k--", linewidth=1.2, alpha=0.5, label=r"$O(T/V + V)$ reference")
    
    ax_as.set_xlabel("Round (t)")
    ax_as.set_ylabel("Cumulative regret")
    ax_as.set_title("All-Seeing Regret vs Oracle")
    ax_as.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax_as.legend(fontsize=9)
    ax_as.grid(True, alpha=0.3)
    
    # ── 2) Partially-Blind ──────────────────────────────────────────
    ax_pb.plot(t_ax, regret_pb, label="Partially-Blind", color=_PALETTE[1], linewidth=2.2)
    ax_pb.fill_between(t_ax, regret_pb, alpha=0.08, color=_PALETTE[1])
    
    log_factor = d + np.log(max(N * M, 1))
    pb_shape = np.sqrt(t_ax * log_factor)
    mid_pb = max(regret_pb[T // 2], 1.0)
    scale_pb = mid_pb / max(pb_shape[T // 2], 1e-9)
    ref_pb = scale_pb * pb_shape
    
    pb_lbl = (
        r"$O\!\left(N K_{{\rm max}}\sqrt{{T(d+\log NM)}}\right)$"
        "\n"
        r"[Partially-Blind §12, $N$=" + str(N) + r"$,M$=" + str(M) + r"$,d$=" + str(d) + "]"
    )
    
    ax_pb.plot(t_ax, ref_pb, color="#2ca02c", linestyle="-.", linewidth=1.2, alpha=0.6, label=pb_lbl)
    
    ax_pb.set_xlabel("Round (t)")
    ax_pb.set_ylabel("Cumulative regret")
    ax_pb.set_title("Partially-Blind Regret vs Oracle")
    ax_pb.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax_pb.legend(fontsize=9)
    ax_pb.grid(True, alpha=0.3)
    
    # ── 3) Bid Loss (Partially Blind) ──────────────────────────────────
    # bid loss = our_bid - market_price (only plotted when we bid > 0)
    our_agent_id = 0
    bid_losses = np.zeros((T, N))
    for t_idx, r in enumerate(results_partially_blind):
        our_bid = r.all_bids.get(our_agent_id, np.zeros(N))
        market_price = r.winning_bids
        loss = our_bid - market_price
        bid_losses[t_idx, :] = np.where(our_bid > 0, loss, np.nan)
        
    for i, prod in enumerate(products):
        # We add marker='.' so isolated rounds where we bid are still visible
        ax_loss.plot(t_ax, bid_losses[:, i], label=prod, color=_PALETTE[i % len(_PALETTE)], 
                     alpha=0.8, marker=".", markersize=4, linestyle="-")
        
    ax_loss.axhline(0, color="k", linestyle=":", linewidth=1.2)
    ax_loss.set_xlabel("Round (t)")
    ax_loss.set_ylabel("Bid Loss (Bid - Market Price)")
    ax_loss.set_title("Partially-Blind Bid Loss Per Round")
    ax_loss.legend(fontsize=9)
    ax_loss.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [viz] Regret plot saved → {save_path}")

    if show:
        plt.show()
    return fig


# ════════════════════════════════════════════════════════════════════════
#  4-Panel Convergence Dashboard (from regret-bounds doc §5)
# ════════════════════════════════════════════════════════════════════════

def _violation_series(
    results: list,
    alpha: "np.ndarray",
    our_agent_id: int = 0,
) -> "np.ndarray":
    """Cumulative constraint violation V_viol(t) = Σ_{s≤t} Σ_i max(0, α_i − s_i(s)).

    A well-behaved Lyapunov policy should have V_viol(t) = o(t).
    """
    N = len(alpha)
    per_round = np.array([
        sum(max(0.0, alpha[i] - r.inventories_after[our_agent_id][i]) for i in range(N))
        for r in results
    ])
    return np.cumsum(per_round)


def plot_convergence_dashboard(
    runs: list,
    alpha: "np.ndarray",
    our_agent_id: int = 0,
    V: float | None = None,
    n_products: int = 4,
    n_competitors: int = 3,
    context_dim: int | None = None,
    save_path: str | None = None,
    show: bool = False,
) -> Figure:
    """4-panel convergence dashboard aligned with §5 of the regret-bounds doc.

    Plot 1  Cumulative regret   Reg(t)          should be o(t)
    Plot 2  Average regret      Reg(t)/t        should → 0
    Plot 3  Cumulative violation V_viol(t)      should be o(t)
    Plot 4  Average violation    V_viol(t)/t    should → 0

    Reference lines drawn on Plots 1 & 2
    ─────────────────────────────────────
    Black dashed : O((T/V) + V) — All-Seeing Lyapunov bound (§5.3)

    Green dashed : O(N·K_max·√(T·(d+log(NM)))) — Partially-Blind OLS bound (§12)
        This is the dominant term for the PB agent under online OLS:
            Reg_T = O_p( N · K_max · √(T · (d + log(N·M))) )
        where N = n_products, M = n_competitors, d = context_dim.

    Both reference curves are **scaled** so they pass through the respective
    empirical curve at t = T//2, making the growth-rate comparison clear
    without hiding either curve off-screen.

    Parameters
    ----------
    runs : list of (label, results, cum_regret)
        label       : str
        results     : list[RoundResult]
        cum_regret  : np.ndarray (T,) from compute_regret()
    alpha : np.ndarray (N,)
    V : float | None        — annotates Plot 2 if given
    n_products : int        — N in the PB bound (default 4)
    n_competitors : int     — M in the PB bound (default 3)
    context_dim : int|None  — d in the PB bound; defaults to n_products
    save_path, show
    """
    N = n_products
    M = n_competitors
    d = context_dim if context_dim is not None else N   # context = competitor inventories ∈ ℝ^N

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax1, ax2, ax3, ax4 = axes.flat

    fig.suptitle(
        "Convergence Dashboard — Regret & Constraint Violation",
        fontsize=13, fontweight="bold",
    )

    V_val = V if V is not None else 2.0
    ref_as = ref_cv = scale_cr = scale_cv = t_ax = None   # set on first pass
    # Find the PB run index (label contains "Blind") to scale the PB reference
    pb_regret_mid = None

    for idx, (label, results, cum_regret) in enumerate(runs):
        T = len(cum_regret)
        t_ax = np.arange(1, T + 1)
        color = _PALETTE[idx % len(_PALETTE)]

        cum_viol   = _violation_series(results, alpha, our_agent_id)
        avg_viol   = cum_viol   / t_ax
        avg_regret = cum_regret / t_ax

        # Scale O(T/V + V) reference to ALL-SEEING curve midpoint (first run)
        if idx == 0:
            mid = max(T // 2, 1)
            as_shape = (t_ax / V_val) + V_val
            scale_cr = max(abs(float(cum_regret[mid])), 1.0) / as_shape[mid]
            scale_cv = max(float(cum_viol[mid]), 1.0)        / as_shape[mid]
            ref_as = scale_cr * as_shape
            ref_cv   = scale_cv * as_shape

        # Capture PB curve midpoint for PB reference scaling
        if "blind" in label.lower() or (len(runs) == 1):
            mid = max(T // 2, 1)
            pb_regret_mid = abs(float(cum_regret[mid]))

        kw  = dict(linewidth=2.0, label=label, color=color)
        fkw = dict(alpha=0.08, color=color)

        ax1.plot(t_ax, cum_regret, **kw); ax1.fill_between(t_ax, cum_regret, **fkw)
        ax2.plot(t_ax, avg_regret, **kw); ax2.fill_between(t_ax, avg_regret, **fkw)
        ax3.plot(t_ax, cum_viol,   **kw); ax3.fill_between(t_ax, cum_viol,   **fkw)
        ax4.plot(t_ax, avg_viol,   **kw); ax4.fill_between(t_ax, avg_viol,   **fkw)

    # ── Reference lines on Plots 1 & 2 ─────────────────────────────────
    # O((T/V) + V) — All-Seeing Lyapunov (§5.3)
    as_kw = dict(color="black", linestyle="--", linewidth=1.2, alpha=0.5)
    ax1.plot(t_ax, ref_as,
             **as_kw, label=r"$O(T/V + V)$  [All-Seeing §5.3]")
    
    # Average regret bound is (T/V + V)/t = 1/V + V/t
    avg_as_shape = (1.0 / V_val) + (V_val / t_ax)
    ax2.plot(t_ax, scale_cr * avg_as_shape,
             **as_kw, label=r"$O((1/V) + V/t)$  [All-Seeing §5.3]")

    # O(N·Kmax·√(T·(d+log(NM)))) — Partially-Blind OLS (§12)
    #   Shape: √(T·(d + log(NM)))  =  √(d+log(NM)) · √T  (still O(√T) but larger constant)
    log_factor = d + np.log(N * M)                     # d + log(NM) scalar
    pb_shape   = np.sqrt(t_ax * log_factor)            # shape of PB bound

    # Scale so the PB reference passes through pb_regret_mid at T//2
    if pb_regret_mid is not None and pb_regret_mid > 0:
        mid = max(T // 2, 1)
        scale_pb = max(pb_regret_mid, 1.0) / pb_shape[mid]
    else:
        scale_pb = scale_cr * np.sqrt(log_factor)      # fallback: ratio to AS bound

    pb_ref     = scale_pb * pb_shape
    pb_avg_ref = scale_pb * np.sqrt(log_factor / t_ax)

    pb_kw = dict(color="#2ca02c", linestyle="-.", linewidth=1.2, alpha=0.6)
    pb_label_cum = (
        r"$O\!\left(N K_{{\rm max}}\sqrt{{T(d+\log NM)}}\right)$"
        "\n"
        r"[Partially-Blind §12,  $N$=" + str(N) +
        r"$,M$=" + str(M) + r"$,d$=" + str(d) + "]"
    )
    pb_label_avg = (
        r"$O\!\left(NK_{{\rm max}}\sqrt{{(d+\log NM)/T}}\right)$"
        "\n"
        r"[Partially-Blind §12]"
    )
    ax1.plot(t_ax, pb_ref,     **pb_kw, label=pb_label_cum)
    ax2.plot(t_ax, pb_avg_ref, **pb_kw, label=pb_label_avg)

    # O(√T) / O(1/√T) for violation axes (no PB variant — same as AS)
    cv_kw = dict(color="black", linestyle="--", linewidth=1.2, alpha=0.5)
    ax3.plot(t_ax, ref_cv,                   **cv_kw, label=r"$O(\sqrt{T})$")
    ax4.plot(t_ax, scale_cv / np.sqrt(t_ax), **cv_kw, label=r"$O(1/\sqrt{T})$")

    # ── axis formatting ──────────────────────────────────────────────────
    fmt   = mticker.FuncFormatter(lambda x, _: f"{x:,.1f}")
    V_str = f"  (V={V:.1f})" if V is not None else ""

    _cfg = [
        (ax1,
         "Plot 1 — Cumulative regret  $\\mathrm{Reg}(t)$\n"
         r"sublinear $\Rightarrow$ $o(t)$",
         "Cumulative regret"),
        (ax2,
         f"Plot 2 — Average regret  $\\mathrm{{Reg}}(t)/t${V_str}\n"
         r"no-regret $\Rightarrow$ $\to 0$",
         "Avg regret per round"),
        (ax3,
         r"Plot 3 — Cumulative violation  $V_{\rm viol}(t)$"
         "\n"
         r"$=\sum_{s\leq t}\sum_i\max(0,\alpha_i-s_i(s))$",
         "Cumulative inventory deficit"),
        (ax4,
         r"Plot 4 — Average violation  $V_{\rm viol}(t)/t$"
         "\n"
         r"feasible policy $\Rightarrow$ $\to 0$",
         "Avg deficit per round"),
    ]
    for ax, title, ylabel in _cfg:
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("Round $t$")
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_formatter(fmt)
        ax.axhline(0, color="gray", lw=0.8, ls=":")
        ax.legend(fontsize=7.5)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [viz] Convergence dashboard saved → {save_path}")

    if show:
        plt.show()

    return fig

