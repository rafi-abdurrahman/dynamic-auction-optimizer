"""Visualization functions for the auction simulation."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from agents.base import PRODUCTS as DEFAULT_PRODUCTS, N_PRODUCTS


_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def plot_inventory_over_time(
    results: list,
    our_agent_id: int = 0,
    alpha: np.ndarray | None = None,
    products: Sequence[str] = DEFAULT_PRODUCTS,
    ax: Axes | None = None,
    show: bool = False,
) -> Figure:
    """Plot our agent's inventory trajectory for each product."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4))
    else:
        fig = ax.get_figure()

    T = [r.t for r in results]
    for i, prod in enumerate(products):
        inv = [r.inventories_after[our_agent_id][i] for r in results]
        ax.plot(T, inv, label=prod, color=_PALETTE[i % len(_PALETTE)], linewidth=1.8)
        if alpha is not None:
            ax.axhline(
                alpha[i], color=_PALETTE[i % len(_PALETTE)],
                linestyle="--", linewidth=0.9, alpha=0.6,
            )

    ax.set_xlabel("Round (t)")
    ax.set_ylabel("Inventory (tonnes)")
    ax.set_title("Inventory over time  [— inventory  - - alpha]")
    ax.legend(loc="upper right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_cumulative_cost(
    results: list,
    ax: Axes | None = None,
    show: bool = False,
) -> Figure:
    """Plot the cumulative budget spent by our agent over all rounds."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4))
    else:
        fig = ax.get_figure()

    T = [r.t for r in results]
    cum_cost = np.cumsum([r.our_cost for r in results])
    per_round = [r.our_cost for r in results]

    ax.fill_between(T, cum_cost, alpha=0.15, color=_PALETTE[0])
    ax.plot(T, cum_cost, color=_PALETTE[0], linewidth=2, label="Cumulative cost")

    ax2 = ax.twinx()
    ax2.bar(T, per_round, color=_PALETTE[1], alpha=0.4, width=0.8, label="Per-round cost")
    ax2.set_ylabel("Per-round cost (tonnes)")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    ax.set_xlabel("Round (t)")
    ax.set_ylabel("Cumulative cost (tonnes)")
    ax.set_title("Cumulative cost spent by our agent")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(True, alpha=0.3)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_bid_vs_market_price(
    results: list,
    our_agent_id: int = 0,
    products: Sequence[str] = DEFAULT_PRODUCTS,
    show: bool = False,
) -> Figure:
    """For each product, plot our bid vs the winning market price (rounds where bid > 0)."""
    n = len(products)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for i, (prod, ax) in enumerate(zip(products, axes)):
        T, our_bids, market = [], [], []
        for r in results:
            ob = r.all_bids[our_agent_id][i]
            mp = r.winning_bids[i]
            if ob > 0:
                T.append(r.t)
                our_bids.append(ob)
                market.append(mp)

        if T:
            ax.plot(T, market,   label="Market price", color=_PALETTE[2],
                    linewidth=1.4, alpha=0.8)
            ax.scatter(T, our_bids, label="Our bid", color=_PALETTE[0],
                       s=18, zorder=3)

        ax.set_title(prod, fontsize=10)
        ax.set_xlabel("Round (t)")
        ax.set_ylabel("Bid (tonnes)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Our bid vs market price per product", y=1.02)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_win_rate(
    results: list,
    our_agent_id: int = 0,
    products: Sequence[str] = DEFAULT_PRODUCTS,
    ax: Axes | None = None,
    show: bool = False,
) -> Figure:
    """Bar chart of bid win-rate per product for our agent."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    win_counts = np.zeros(len(products))
    bid_counts = np.zeros(len(products))
    for r in results:
        win_counts += r.our_won.astype(float)
        bid_counts += (r.all_bids[our_agent_id] > 0).astype(float)

    safe = np.where(bid_counts > 0, bid_counts, 1.0)
    win_rates = np.where(bid_counts > 0, win_counts / safe, 0.0) * 100

    bars = ax.bar(products, win_rates, color=_PALETTE, edgecolor="white", linewidth=0.8)
    for bar, wr in zip(bars, win_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{wr:.1f}%", ha="center", va="bottom", fontsize=9,
        )

    ax.set_ylim(0, 115)
    ax.set_ylabel("Win rate (%)")
    ax.set_title("Win rate per product (our agent)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_lyapunov_deficit(
    agent_history: list[dict],
    products: Sequence[str] = DEFAULT_PRODUCTS,
    ax: Axes | None = None,
    show: bool = False,
) -> Figure:
    """Plot H_i(t) = β_i − s_i(t) per product. Each history entry needs 't' and 'H' keys."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4))
    else:
        fig = ax.get_figure()

    T = [entry["t"] for entry in agent_history]
    for i, prod in enumerate(products):
        H = [entry["H"][i] for entry in agent_history]
        ax.plot(T, H, label=prod, color=_PALETTE[i % len(_PALETTE)], linewidth=1.8)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5,
               label="H=0 (at β)")
    ax.set_xlabel("Round (t)")
    ax.set_ylabel("Deficit H_i(t) = β_i − s_i(t)")
    ax.set_title("Lyapunov deficit over time  (positive → below soft target β)")
    ax.legend(loc="upper right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_all(
    results: list,
    our_agent,
    products: Sequence[str] = DEFAULT_PRODUCTS,
    save_path: str | None = None,
    show: bool = True,
) -> Figure:
    """Render a 2×3 dashboard of all plots in a single figure."""
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("Auction Simulation Dashboard", fontsize=14, fontweight="bold", y=0.98)

    ax_inv  = fig.add_subplot(2, 3, 1)
    ax_cost = fig.add_subplot(2, 3, 2)
    ax_win  = fig.add_subplot(2, 3, 3)
    ax_def  = fig.add_subplot(2, 3, 4)
    ax_b0   = fig.add_subplot(2, 3, 5)
    ax_b1   = fig.add_subplot(2, 3, 6)

    plot_inventory_over_time(
        results, our_agent_id=our_agent.agent_id,
        alpha=our_agent.alpha, products=products, ax=ax_inv,
    )
    plot_cumulative_cost(results, ax=ax_cost)
    plot_win_rate(results, our_agent_id=our_agent.agent_id, products=products, ax=ax_win)
    plot_lyapunov_deficit(our_agent.history, products=products, ax=ax_def)

    for panel_ax, prod_idx in [(ax_b0, 0), (ax_b1, 1)]:
        T, our_bids, market = [], [], []
        prod = products[prod_idx]
        for r in results:
            ob = r.all_bids[our_agent.agent_id][prod_idx]
            mp = r.winning_bids[prod_idx]
            if ob > 0:
                T.append(r.t)
                our_bids.append(ob)
                market.append(mp)
        if T:
            panel_ax.plot(T, market, label="Market price", color=_PALETTE[2],
                          linewidth=1.4, alpha=0.8)
            panel_ax.scatter(T, our_bids, label="Our bid", color=_PALETTE[0],
                             s=18, zorder=3)
        panel_ax.set_title(f"Bid vs market — {prod}", fontsize=10)
        panel_ax.set_xlabel("Round (t)")
        panel_ax.set_ylabel("Bid (tonnes)")
        panel_ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
        )
        panel_ax.legend(fontsize=7)
        panel_ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [viz] Dashboard saved → {save_path}")

    if show:
        plt.show()
    return fig


def plot_ilp_schedule(
    x_opt: np.ndarray,
    results: list,
    alpha: np.ndarray,
    products: Sequence[str] = DEFAULT_PRODUCTS,
    our_agent_id: int = 0,
    save_path: str | None = None,
    show: bool = False,
) -> Figure:
    """Two-panel figure: binary purchase heatmap (top) and inventory trace (bottom)."""
    from matplotlib.colors import ListedColormap

    T, N = x_opt.shape
    fig, (ax_heat, ax_inv) = plt.subplots(
        2, 1, figsize=(14, 7),
        gridspec_kw={"height_ratios": [1, 2]},
        sharex=True,
    )
    fig.suptitle(
        "ILP Optimal Schedule (Section IV-A)",
        fontsize=13, fontweight="bold", y=0.98,
    )

    cmap = ListedColormap(["#f0f0f0", "#2b8cbe"])
    ax_heat.imshow(
        x_opt.T,
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        extent=[0, T, N - 0.5, -0.5],
    )
    ax_heat.set_yticks(range(N))
    ax_heat.set_yticklabels(products[:N], fontsize=9)
    ax_heat.set_title("Purchase decisions  (dark = buy, light = skip)", fontsize=10)

    # Annotate total buys per product on the right margin.
    for i in range(N):
        total_buys = int(x_opt[:, i].sum())
        ax_heat.text(
            T + 0.5, i, f"{total_buys}/{T}",
            va="center", ha="left", fontsize=8, color="#333333",
        )
    ax_heat.grid(False)

    t_ax = np.arange(T)
    for i, prod in enumerate(products[:N]):
        inv = [r.inventories_after[our_agent_id][i] for r in results]
        color = _PALETTE[i % len(_PALETTE)]
        ax_inv.plot(t_ax, inv, label=prod, color=color, linewidth=1.8)
        ax_inv.axhline(
            alpha[i], color=color,
            linestyle="--", linewidth=0.9, alpha=0.6,
        )

    ax_inv.set_xlabel("Round (t)")
    ax_inv.set_ylabel("Inventory")
    ax_inv.set_title(
        "Inventory trajectory under ILP schedule  [-- alpha constraint]",
        fontsize=10,
    )
    ax_inv.legend(loc="upper right", fontsize=8)
    ax_inv.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )
    ax_inv.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [viz] ILP schedule plot saved -> {save_path}")

    if show:
        plt.show()
    return fig
