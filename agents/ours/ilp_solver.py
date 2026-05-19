"""
Offline ILP solver for the deterministic bidding setup from Section IV-A.

Given complete knowledge of future prices, lot sizes, and depletions, solves:

    min  sum_t sum_i  b_i(t) * x_i(t)
    s.t. s_i(t) >= alpha_i       for all i, t
         s_i(t+1) = s_i(t) + q_i(t) * x_i(t) - d_i(t)
         x_i(t)  in {0, 1}

Products are separable, so the problem decomposes into N independent
single-product ILPs solved via scipy.optimize.milp (HiGHS backend).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

from agents.base import AuctionState, BaseAgent, N_PRODUCTS


def _solve_single_product_ilp(
    prices: np.ndarray,
    quantities: np.ndarray,
    depletions: np.ndarray,
    alpha_i: float,
    s0_i: float,
) -> tuple[np.ndarray, float, bool]:
    """Solve the single-product ILP; returns (x_opt, cost, feasible)."""
    T = len(prices)

    active = quantities > 0

    # cum_dep[t] = sum_{s=0}^{t-1} d_i(s)
    cum_dep = np.concatenate([[0.0], np.cumsum(depletions[:-1])])

    # Inventory constraint at each t:
    #   sum_{s<t} q(s)*x(s) >= alpha_i - s0_i + cum_dep[t]
    A = np.zeros((T, T))
    lower_bounds = np.zeros(T)

    for t in range(T):
        A[t, :t] = quantities[:t]
        lower_bounds[t] = alpha_i - s0_i + cum_dep[t]

    # Also enforce feasibility after the final round
    total_dep = depletions.sum()
    A_final = quantities.reshape(1, -1)
    lb_final = np.array([alpha_i - s0_i + total_dep])

    A_full = np.vstack([A, A_final])
    lb_full = np.concatenate([lower_bounds, lb_final])
    ub_full = np.full(len(lb_full), np.inf)

    constraints = LinearConstraint(A_full, lb_full, ub_full)

    var_lb = np.zeros(T)
    var_ub = np.where(active, 1.0, 0.0)
    integrality = np.ones(T, dtype=int)

    result = milp(
        c=prices,
        constraints=constraints,
        integrality=integrality,
        bounds=Bounds(lb=var_lb, ub=var_ub),
    )

    if result.success:
        x_opt = np.round(result.x).astype(int)
        cost = float(prices @ x_opt)
        return x_opt, cost, True
    else:
        # Infeasible: fall back to buying every active round
        x_fallback = active.astype(int)
        cost = float(prices @ x_fallback)
        return x_fallback, cost, False


def solve_ilp_offline(
    prices: np.ndarray,
    quantities: np.ndarray,
    depletions: np.ndarray,
    alpha: np.ndarray,
    initial_inventory: np.ndarray,
) -> tuple[np.ndarray, float, list[bool]]:
    """Solve the offline ILP from Section IV-A decomposed over N products.

    Returns (x_opt, total_cost, feasibility) where x_opt is the (T, N)
    binary decision matrix and feasibility[i] is False if the ILP was
    infeasible for product i (fallback to always-buy).
    """
    T, N = prices.shape
    x_opt = np.zeros((T, N), dtype=int)
    total_cost = 0.0
    feasibility = []

    for i in range(N):
        x_i, cost_i, feasible_i = _solve_single_product_ilp(
            prices=prices[:, i],
            quantities=quantities[:, i],
            depletions=depletions[:, i],
            alpha_i=float(alpha[i]),
            s0_i=float(initial_inventory[i]),
        )
        x_opt[:, i] = x_i
        total_cost += cost_i
        feasibility.append(feasible_i)

    return x_opt, total_cost, feasibility


class ILPOracleAgent(BaseAgent):
    """Clairvoyant offline oracle that replays a pre-computed ILP schedule."""

    def __init__(
        self,
        agent_id: int,
        n_products: int = N_PRODUCTS,
        alpha: np.ndarray | None = None,
        initial_inventory: np.ndarray | None = None,
        x_opt: np.ndarray | None = None,
        prices: np.ndarray | None = None,
        ilp_total_cost: float = 0.0,
        feasibility: list[bool] | None = None,
    ) -> None:
        super().__init__(agent_id, n_products, alpha, initial_inventory)

        if x_opt is None:
            raise ValueError("ILPOracleAgent requires a pre-computed x_opt matrix.")
        if prices is None:
            raise ValueError("ILPOracleAgent requires the prices matrix.")

        self._x_opt = x_opt
        self._prices = prices
        self._T = x_opt.shape[0]
        self._current_t = 0

        self.ilp_total_cost = ilp_total_cost
        self.feasibility = feasibility if feasibility is not None else [True] * n_products

    @property
    def schedule(self) -> np.ndarray:
        """Return the full (T, N) binary decision matrix."""
        return self._x_opt

    def bid(self, state: AuctionState) -> np.ndarray:
        """Replay the pre-computed ILP decision for the current round."""
        t = self._current_t
        bids_out = np.zeros(self.n_products)

        if t < self._T:
            for i in range(self.n_products):
                if self._x_opt[t, i] == 1:
                    # Bid recorded price + small epsilon to guarantee winning
                    price = self._prices[t, i]
                    epsilon = price * 0.01 + 0.01
                    bids_out[i] = price + epsilon

        self.log({
            "t": t,
            "x_opt": self._x_opt[t].copy() if t < self._T else np.zeros(self.n_products),
            "bids": bids_out.copy(),
            "inventory": self.inventory.copy(),
        })

        self._current_t += 1
        return bids_out

    def reset(self, initial_inventory: np.ndarray | None = None) -> None:
        """Reset the agent replay counter."""
        super().reset(initial_inventory)
        self._current_t = 0
