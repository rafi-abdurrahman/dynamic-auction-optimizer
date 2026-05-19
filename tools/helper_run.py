"""Helper functions for scripts/run.py.

Extracted here to keep run.py focused on CLI definition and the game loop.
"""

from __future__ import annotations

from typing import Iterator
from pathlib import Path

import numpy as np

from agents.competitors.competitors import (
    CompetitorAgent,
    LinearCompetitorAgent,
    NaiveThresholdCompetitorAgent,
)
from agents.base import N_PRODUCTS


def build_competitors(
    spec: str,
    alpha: np.ndarray,
    init_inv: np.ndarray,
    seed: int,
) -> list:
    """Parse a comma-separated competitor spec and instantiate agents.

    Each token must be one of: stochastic | random | linear | naive.
    Agent ids start at 1 (our agent is always 0).
    """
    competitors = []
    for idx, name in enumerate(spec.split(",")):
        name = name.strip().lower()
        agent_id   = idx + 1
        agent_seed = seed + idx + 1

        if name in ("stochastic", "random"):
            competitors.append(
                CompetitorAgent(
                    agent_id=agent_id,
                    alpha=alpha,
                    initial_inventory=init_inv.copy(),
                    seed=agent_seed,
                )
            )
        elif name == "linear":
            competitors.append(
                LinearCompetitorAgent(
                    agent_id=agent_id,
                    alpha=alpha,
                    initial_inventory=init_inv.copy(),
                    seed=agent_seed,
                )
            )
        elif name == "naive":
            competitors.append(
                NaiveThresholdCompetitorAgent(
                    agent_id=agent_id,
                    alpha=alpha,
                    initial_inventory=init_inv.copy(),
                    seed=agent_seed,
                )
            )
        else:
            raise ValueError(
                f"Unknown competitor type '{name}'. "
                "Valid choices: stochastic | random | linear | naive"
            )
    return competitors


def quantity_iter_from_generator(gen) -> Iterator[np.ndarray]:
    """Yield per-auction quantity vectors from AuctionDataGenerator in canonical product order."""
    _ORDER = ("milk", "eggs", "poultry", "beef")
    for _date, _auction_idx, quantities_dict in gen.generate():
        vec = np.array(
            [quantities_dict.get(p, quantities_dict.get(p.split("_")[0], 0.0))
             for p in _ORDER],
            dtype=float,
        )
        yield vec


def depletion_iter(depletion_rates: np.ndarray, n: int) -> Iterator[np.ndarray]:
    """Yield the same depletion vector *n* times."""
    for _ in range(n):
        yield depletion_rates.copy()


def derive_data_stats(
    country: str,
    auctions_per_day: int,
    auction_fraction: float = 1.0 / 20,
    a: float = 0.5,
    d_rate: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load EUROSTAT data and compute per-auction quantity statistics.

    Parameters derive as follows (all values in normalised [0, 1000] units):
        mean_qty = normed_daily_mean ÷ auctions_per_day
        alpha    = a      × mean_qty
        d_max    = d_rate × alpha

    auction_fraction is unused but kept for API compatibility.
    Returns (mean_qty, d_max, alpha).
    """
    from datagen.generate_data import AuctionDataGenerator

    _ORDER = ["milk", "eggs", "poultry", "beef"]

    gen = AuctionDataGenerator(country=country, auctions_per_day=auctions_per_day)
    # normed_daily_mean is in [0, 1000] units; divide by auctions_per_day
    # to get the per-slot mean quantity.
    normed_mean = gen.normed_daily_mean

    def _get(series, key, fallback):
        return float(series[key]) if key in series.index else fallback

    mean_qty = np.array([_get(normed_mean, p, 0.0) for p in _ORDER]) / auctions_per_day
    alpha    = a * mean_qty
    d_max    = d_rate * alpha

    return mean_qty, d_max, alpha


def _rp(run_dir: Path, stem: str, suffix: str = ".txt") -> Path:
    return run_dir / f"{stem}{suffix}"


def write_summary(
    path: Path,
    ts: str,
    run_label: str,
    sim,
    args,
    alpha: "np.ndarray",
    d_max: "np.ndarray",
    init_inv: "np.ndarray",
    products,
) -> None:
    """Write a structured summary file built directly from sim.summary()."""
    stats = sim.summary()
    lines = []
    W = 60

    lines += [
        "=" * W,
        f"  AUCTION SIMULATION SUMMARY",
        f"  Timestamp  : {ts}",
        f"  Mode       : {run_label}",
        "=" * W,
        "",
        "── Configuration " + "─" * (W - 17),
        f"  n_rounds         : {args.n_rounds}",
        f"  V (Lyapunov)     : {args.V}",
        f"  a (scale)        : {args.a}",
        f"  d_rate           : {args.d_rate}",
        f"  competitors      : {args.competitors}",
        f"  seed             : {args.seed}",
        f"  mode             : {args.mode}",
        "",
        "── Resolved Parameters " + "─" * (W - 22),
    ]

    lines.append(
        f"  {'Product':15s}  {'alpha':>12}  {'d_max':>12}  {'init_inv':>12}"
    )
    lines.append("  " + "-" * 55)
    for i, prod in enumerate(products):
        lines.append(
            f"  {prod:15s}  {alpha[i]:12.2f}  {d_max[i]:12.2f}  {init_inv[i]:12.2f}"
        )

    lines += [
        "",
        "── Simulation Results " + "─" * (W - 21),
        f"  Rounds simulated     : {stats.get('n_rounds', 0)}",
        f"  Rounds won (any)     : {stats.get('rounds_won', 0)} ({stats.get('rounds_won_percentage', 0) * 100:.1f}%)",
        f"  Total cost           : {stats.get('total_cost', 0):,.2f}",
        f"  Avg cost / round     : {stats.get('avg_cost_per_round', 0):,.2f}",
        f"  Constraint violations: {stats.get('constraint_violations', 0)}"
        f"  ({stats.get('violation_rate', 0) * 100:.1f}%)",
        f"  Final inventory      : {[round(v, 2) for v in stats.get('final_inventory', [])]}",
        "",
        "── Win Rate per Product " + "─" * (W - 23),
    ]
    for prod, wr in zip(products, stats.get("win_rate_per_product", [])):
        lines.append(f"  {prod:15s}: {wr * 100:6.1f}%")

    lines += ["", "=" * W, ""]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [run] Summary saved → {path}")
