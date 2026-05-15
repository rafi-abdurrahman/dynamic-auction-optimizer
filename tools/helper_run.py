"""
Helper functions for scripts/run.py.

Extracted here to keep run.py focused on CLI definition and the game loop.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from agents.competitors.competitors import (
    CompetitorAgent,
    LinearCompetitorAgent,
    NaiveThresholdCompetitorAgent,
)
from agents.base import N_PRODUCTS


# ── competitor factory ──────────────────────────────────────────────────

def build_competitors(
    spec: str,
    alpha: np.ndarray,
    init_inv: np.ndarray,
    seed: int,
) -> list:
    """Parse a comma-separated competitor spec and instantiate agents.

    Parameters
    ----------
    spec : str
        Comma-separated competitor types, e.g. ``"stochastic,linear,naive"``.
        Each token must be one of: ``stochastic`` | ``random`` | ``linear`` | ``naive``.
    alpha : np.ndarray
        Hard minimum inventory constraint passed to each competitor.
    init_inv : np.ndarray
        Starting inventory for each competitor.
    seed : int
        Base random seed; each competitor receives ``seed + idx + 1``.

    Returns
    -------
    list[BaseAgent]
        Instantiated competitor agents with ids starting at 1 (our agent is 0).
    """
    competitors = []
    for idx, name in enumerate(spec.split(",")):
        name = name.strip().lower()
        agent_id  = idx + 1          # our agent is id=0
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


# ── data iterators ──────────────────────────────────────────────────────

def quantity_iter_from_generator(gen) -> Iterator[np.ndarray]:
    """Yield per-auction quantity vectors (shape N,) from AuctionDataGenerator.

    Maps the generator's product dict into the canonical product order
    ``('milk', 'eggs', 'poultry', 'beef')`` used by the agents.
    """
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


# ── EUROSTAT stats ──────────────────────────────────────────────────────

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
        alpha    = a      × mean_qty   (hard inventory safety stock)
        d_max    = d_rate × alpha      (max per-auction depletion)

    Quantities from AuctionDataGenerator are normalised per-product to
    [0, 1000] (1000 = historical max).  derive_data_stats reads
    ``gen.normed_daily_mean`` so alpha and d_max live in the same unit
    space as the auction lots.

    Parameters
    ----------
    country : str
        ISO-2 EUROSTAT country code (e.g. ``"DE"``).
    auctions_per_day : int
        Number of auction slots per calendar day.
    auction_fraction : float
        Unused (kept for API compatibility; generator uses auction_fraction=1.0
        internally and normalises to [0, 1000]).
    a : float
        Scale factor in (0, 1].  Sets ``alpha = a × mean_qty``.
    d_rate : float
        Depletion rate in (0, 1].  Sets ``d_max = d_rate × alpha``.

    Returns
    -------
    mean_qty : np.ndarray, shape (4,)   normalised tonnes per auction slot
    d_max    : np.ndarray, shape (4,)   = d_rate × alpha
    alpha    : np.ndarray, shape (4,)   = a × mean_qty
    """
    from datagen.generate_data import AuctionDataGenerator

    _ORDER = ["milk", "eggs", "poultry", "beef"]

    gen = AuctionDataGenerator(country=country, auctions_per_day=auctions_per_day)
    # normed_daily_mean is in [0, 1000] units; divide by auctions_per_day
    # to get the per-slot mean quantity.
    normed_mean = gen.normed_daily_mean  # pd.Series, index = products

    def _get(series, key, fallback):
        return float(series[key]) if key in series.index else fallback

    mean_qty = np.array([_get(normed_mean, p, 0.0) for p in _ORDER]) / auctions_per_day
    alpha    = a * mean_qty
    d_max    = d_rate * alpha

    return mean_qty, d_max, alpha
