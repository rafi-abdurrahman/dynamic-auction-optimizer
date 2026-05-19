from __future__ import annotations

import numpy as np

from agents.base import AuctionState, BaseAgent, N_PRODUCTS


class CompetitorAgent(BaseAgent):
    """Stochastic competitor whose bids scale with inventory deficit."""

    def __init__(
        self,
        agent_id: int,
        n_products: int = N_PRODUCTS,
        alpha: np.ndarray | None = None,
        initial_inventory: np.ndarray | None = None,
        base_bid: float = 30.0,
        noise_std: float = 0.2,
        sensitivity: float = 1.5,
        seed: int | None = None,
    ) -> None:
        super().__init__(agent_id, n_products, alpha, initial_inventory)
        self.base_bid = base_bid
        self.noise_std = noise_std
        self.sensitivity = sensitivity
        self.rng = np.random.default_rng(seed)

    def bid(self, state: AuctionState) -> np.ndarray:
        """Generate stochastic bids inversely proportional to inventory."""
        bids = np.zeros(self.n_products)

        for i in range(self.n_products):
            if state.quantities[i] <= 0:
                continue

            # Higher deficit → higher willingness to pay
            deficit = max(self.alpha[i] - self.inventory[i], 0.0)
            mean_bid = self.base_bid * (
                1.0 + self.sensitivity * deficit / (self.alpha[i] + 1e-8)
            )
            noise = self.rng.lognormal(mean=0.0, sigma=self.noise_std)
            bids[i] = mean_bid * noise

        return bids


class LinearCompetitorAgent(BaseAgent):
    """Competitor that bids linearly w.r.t. its own inventory: bid_i = θ_i^T · K."""

    def __init__(
        self,
        agent_id: int,
        n_products: int = N_PRODUCTS,
        alpha: np.ndarray | None = None,
        initial_inventory: np.ndarray | None = None,
        theta: np.ndarray | None = None,
        noise_std: float = 2.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(agent_id, n_products, alpha, initial_inventory)
        self.rng = np.random.default_rng(seed)
        self.noise_std = noise_std

        if theta is not None:
            self.theta = theta.copy()
        else:
            # Negative diagonal → low inventory produces high bid;
            # small random cross-product terms add realism.
            self.theta = np.zeros((n_products, n_products))
            for i in range(n_products):
                self.theta[i, i] = -self.rng.uniform(0.5, 2.0)
                for j in range(n_products):
                    if j != i:
                        self.theta[i, j] = self.rng.uniform(-0.1, 0.1)

    def bid(self, state: AuctionState) -> np.ndarray:
        """Bid linearly: bid_i = θ_i^T · inventory + base + noise."""
        bids = np.zeros(self.n_products)

        for i in range(self.n_products):
            if state.quantities[i] <= 0:
                continue

            linear_part = self.theta[i] @ self.inventory
            # Shift so bids stay positive at high inventory
            base = -self.theta[i, i] * self.alpha[i] * 2.0
            noise = self.rng.normal(0.0, self.noise_std)
            bids[i] = max(linear_part + base + noise, 0.01)

        return bids


class NaiveThresholdCompetitorAgent(BaseAgent):
    """Competitor that bids a fixed price whenever inventory is below α."""

    def __init__(
        self,
        agent_id: int,
        n_products: int = N_PRODUCTS,
        alpha: np.ndarray | None = None,
        initial_inventory: np.ndarray | None = None,
        fixed_bid: float = 40.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(agent_id, n_products, alpha, initial_inventory)
        self.fixed_bid = fixed_bid
        self.rng = np.random.default_rng(seed)

    def bid(self, state: AuctionState) -> np.ndarray:
        """Bid the fixed amount for any product below inventory threshold."""
        bids = np.zeros(self.n_products)

        for i in range(self.n_products):
            if state.quantities[i] <= 0:
                continue
            if self.inventory[i] < self.alpha[i]:
                # Small noise so ties are broken randomly
                bids[i] = self.fixed_bid + self.rng.uniform(0, 2.0)

        return bids
