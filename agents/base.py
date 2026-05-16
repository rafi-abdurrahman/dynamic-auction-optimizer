from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ── Product catalogue (matches the paper's 4 product types) ────────────
PRODUCTS = ["milk_dairy", "eggs", "poultry", "beef"]
N_PRODUCTS = len(PRODUCTS)


@dataclass
class AuctionState:
    """Snapshot of the auction environment at time *t*.

    Attributes
    ----------
    t : int
        Current time-step.
    quantities : np.ndarray, shape (N,)
        Quantity q_i^(t) of each product offered at this round.
    inventories : dict[int, np.ndarray]
        Mapping from buyer_id → inventory vector s_i^(t).
        Each vector has shape (N,).
    bids : dict[int, np.ndarray] | None
        Mapping from buyer_id → bid vector.  ``None`` when bids are
        hidden (Partially-Blind setting).
    depletions : np.ndarray, shape (N,)
        Depletion d_i^(t) for each product at this time-step.
    """

    t: int
    quantities: np.ndarray                          # (N,)
    inventories: dict[int, np.ndarray]              # buyer_id → (N,)
    bids: dict[int, np.ndarray] | None = None       # buyer_id → (N,) or None
    depletions: np.ndarray = field(default_factory=lambda: np.zeros(N_PRODUCTS))


class BaseAgent(ABC):
    """Abstract base class for every auction participant.

    Parameters
    ----------
    agent_id : int
        Unique identifier for this agent.
    n_products : int
        Number of distinct products in the auction.
    alpha : np.ndarray, shape (N,)
        Hard minimum inventory constraint per product.
    initial_inventory : np.ndarray, shape (N,)
        Starting inventory vector.
    """

    def __init__(
        self,
        agent_id: int,
        n_products: int = N_PRODUCTS,
        alpha: np.ndarray | None = None,
        initial_inventory: np.ndarray | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.n_products = n_products

        # Hard inventory constraint  α_i
        self.alpha = (
            alpha if alpha is not None
            else np.full(n_products, 10.0)
        )

        # Current inventory  s_i^(t)
        self.inventory = (
            initial_inventory.copy() if initial_inventory is not None
            else np.full(n_products, 20.0)
        )

        # Logging
        self.history: list[dict[str, Any]] = []

    # ── public interface ───────────────────────────────────────────────
    @abstractmethod
    def bid(self, state: AuctionState) -> np.ndarray:
        """Return a bid vector of shape (N,).

        ``bid[i] > 0`` means we place a bid of that amount for product *i*.
        ``bid[i] == 0`` means we do **not** bid for product *i*.
        """
        ...

    def update_inventory(
        self,
        won: np.ndarray,
        quantities: np.ndarray,
        depletions: np.ndarray,
    ) -> None:
        """Update inventory after auction resolution.

        Parameters
        ----------
        won : np.ndarray, shape (N,), dtype bool
            Whether we won the auction for each product.
        quantities : np.ndarray, shape (N,)
            Quantity q_i^(t) gained per product if won.
        depletions : np.ndarray, shape (N,)
            Depletion d_i^(t) consumed this round.
        """
        # s_i^(t+1) = s_i^(t) + q_i^(t) * x_i^(t) - d_i^(t)
        self.inventory = self.inventory + quantities * won - depletions
        # Inventory cannot be negative
        self.inventory = np.maximum(self.inventory, 0.0)

    def log(self, record: dict[str, Any]) -> None:
        """Append an arbitrary record to the agent's history."""
        self.history.append(record)

    def reset(self, initial_inventory: np.ndarray | None = None) -> None:
        """Reset the agent to its initial state."""
        self.inventory = (
            initial_inventory.copy() if initial_inventory is not None
            else np.full(self.n_products, 20.0)
        )
        self.history.clear()

    # ── helpers ────────────────────────────────────────────────────────
    def inventory_deficit(self, beta: np.ndarray | None = None) -> np.ndarray:
        """Compute deficit H_i(t) = β_i − s_i(t).

        A positive value means we are *below* the soft target.
        """
        if beta is None:
            beta = self.alpha
        
        return beta - self.inventory

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(id={self.agent_id}, "
            f"inv={np.round(self.inventory, 1)})"
        )
