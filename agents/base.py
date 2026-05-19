from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# Product catalogue 
PRODUCTS = ["milk_dairy", "eggs", "poultry", "beef"]
N_PRODUCTS = len(PRODUCTS)


@dataclass
class AuctionState:
    """Snapshot of the auction environment at time t."""

    t: int
    quantities: np.ndarray                          # (N,)
    inventories: dict[int, np.ndarray]              # buyer_id → (N,)
    bids: dict[int, np.ndarray] | None = None       # buyer_id → (N,) or None
    depletions: np.ndarray = field(default_factory=lambda: np.zeros(N_PRODUCTS))


class BaseAgent(ABC):
    """Abstract base class for every auction participant."""

    def __init__(
        self,
        agent_id: int,
        n_products: int = N_PRODUCTS,
        alpha: np.ndarray | None = None,
        initial_inventory: np.ndarray | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.n_products = n_products

        # Hard inventory constraint α_i
        self.alpha = (
            alpha if alpha is not None
            else np.full(n_products, 10.0)
        )

        # Current inventory s_i^(t)
        self.inventory = (
            initial_inventory.copy() if initial_inventory is not None
            else np.full(n_products, 20.0)
        )

        self.history: list[dict[str, Any]] = []

    @abstractmethod
    def bid(self, state: AuctionState) -> np.ndarray:
        """Return a bid vector of shape (N,).

        bid[i] > 0 means we place a bid for product i; 0 means no bid.
        """
        ...

    def update_inventory(
        self,
        won: np.ndarray,
        quantities: np.ndarray,
        depletions: np.ndarray,
    ) -> None:
        """Update inventory after auction resolution."""
        # s_i^(t+1) = s_i^(t) + q_i^(t) * x_i^(t) - d_i^(t)
        self.inventory = self.inventory + quantities * won - depletions
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

    def inventory_deficit(self, beta: np.ndarray | None = None) -> np.ndarray:
        """Compute deficit H_i(t) = β_i − s_i(t); positive means below soft target."""
        if beta is None:
            beta = self.alpha
        return beta - self.inventory

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(id={self.agent_id}, "
            f"inv={np.round(self.inventory, 1)})"
        )
