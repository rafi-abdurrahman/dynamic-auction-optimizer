"""
Offline Integer Linear Programming (ILP) solver for the deterministic bidding
setup described in Section IV-A of the proposal.

Given complete knowledge of future prices, lot sizes, and depletions, this
module solves the exact binary optimisation problem:

    min  sum_t sum_i  b_i(t) * x_i(t)
    s.t. s_i(t) >= alpha_i       for all i, t
         s_i(t+1) = s_i(t) + q_i(t) * x_i(t) - d_i(t)
         x_i(t)  in {0, 1}

Products are separable (no cross-product constraints), so we decompose into
N independent single-product ILPs and solve each with scipy.optimize.milp.

Public API
----------
    solve_ilp_offline(prices, quantities, depletions, alpha, initial_inventory)
        -> (x_opt, total_cost)

    ILPOracleAgent(BaseAgent)
        Agent wrapper that replays a pre-computed ILP schedule through the
        simulation framework.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

from agents.base import AuctionState, BaseAgent, N_PRODUCTS


# ════════════════════════════════════════════════════════════════════════
#  Core ILP Solver
# ════════════════════════════════════════════════════════════════════════

def _solve_single_product_ilp(
    prices: np.ndarray,
    quantities: np.ndarray,
    depletions: np.ndarray,
    alpha_i: float,
    s0_i: float,
) -> tuple[np.ndarray, float, bool]:
    """Solve the ILP for a single product.

    Parameters
    ----------
    prices : np.ndarray, shape (T,)
        Cost to win the product at each round.
    quantities : np.ndarray, shape (T,)
        Lot size offered at each round.
    depletions : np.ndarray, shape (T,)
        Depletion consumed at each round.
    alpha_i : float
        Hard minimum inventory constraint.
    s0_i : float
        Initial inventory level.

    Returns
    -------
    x_opt : np.ndarray, shape (T,), dtype int
        Binary decisions (1 = buy, 0 = skip).
    cost : float
        Optimal cost for this product.
    feasible : bool
        Whether a feasible integer solution was found.
    """
    T = len(prices)

    # Identify rounds with zero quantity (nothing to buy).
    active = quantities > 0

    # Cumulative depletion up to (but not including) round t.
    #   cum_dep[t] = sum_{s=0}^{t-1} d_i(s)
    cum_dep = np.concatenate([[0.0], np.cumsum(depletions[:-1])])

    # ── Inventory constraint at each time-step ──
    #
    # s_i(t) = s0_i + sum_{s=0}^{t-1} q_i(s)*x_i(s) - cum_dep[t] >= alpha_i
    #
    # Rearranged:
    #   sum_{s=0}^{t-1} q_i(s)*x_i(s) >= alpha_i - s0_i + cum_dep[t]
    #
    # In matrix form A_lb @ x >= b_lb, where:
    #   A_lb[t, s] = q_i(s)  for s < t,  0 otherwise
    #   b_lb[t]    = alpha_i - s0_i + cum_dep[t]

    A = np.zeros((T, T))
    lower_bounds = np.zeros(T)

    for t in range(T):
        A[t, :t] = quantities[:t]
        lower_bounds[t] = alpha_i - s0_i + cum_dep[t]

    # Enforce the constraint *after* the final round:
    # s_i(T+1) = s0_i + sum_{s=0}^{T-1} q(s)*x(s) - sum_{s=0}^{T-1} d(s) >= alpha_i
    total_dep = depletions.sum()
    A_final = quantities.reshape(1, -1)
    lb_final = np.array([alpha_i - s0_i + total_dep])

    A_full = np.vstack([A, A_final])
    lb_full = np.concatenate([lower_bounds, lb_final])

    # Upper bound on each constraint row is +inf (no upper limit on inventory).
    ub_full = np.full(len(lb_full), np.inf)

    constraints = LinearConstraint(A_full, lb_full, ub_full)

    # Variable bounds: x_t in {0, 1}, but fix to 0 if quantity == 0.
    var_lb = np.zeros(T)
    var_ub = np.where(active, 1.0, 0.0)

    # Integrality: 1 = integer (binary since bounds are [0, 1]).
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
        # Infeasible: fall back to buying every active round.
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
    """Solve the offline ILP from Section IV-A of the proposal.

    Decomposes the problem into N independent single-product ILPs and
    solves each with the HiGHS MILP solver via ``scipy.optimize.milp``.

    Parameters
    ----------
    prices : np.ndarray, shape (T, N)
        Cost to win product i at round t.
        (max competing bid + epsilon).
    quantities : np.ndarray, shape (T, N)
        Lot size q_i(t) offered at each round.
    depletions : np.ndarray, shape (T, N)
        Depletion d_i(t) consumed at each round.
    alpha : np.ndarray, shape (N,)
        Hard minimum inventory constraint per product.
    initial_inventory : np.ndarray, shape (N,)
        Starting inventory s_i(0).

    Returns
    -------
    x_opt : np.ndarray, shape (T, N), dtype int
        Binary decision matrix.  x_opt[t, i] = 1 iff the ILP decides to
        bid for product i at round t.
    total_cost : float
        Optimal total cost across all products and rounds.
    feasibility : list[bool]
        Per-product feasibility flags.  False indicates the ILP was
        infeasible for that product and the fallback (always-buy) was used.
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


# ════════════════════════════════════════════════════════════════════════
#  ILP Oracle Agent
# ════════════════════════════════════════════════════════════════════════

class ILPOracleAgent(BaseAgent):
    """Clairvoyant offline oracle that replays a pre-computed ILP schedule.

    This agent does **not** make decisions online.  It requires the full
    future trajectory to be solved ahead of time via ``solve_ilp_offline``,
    then simply replays the optimal binary schedule ``x_opt[t, :]`` at
    each round.

    Parameters
    ----------
    agent_id : int
        Unique agent identifier.
    n_products : int
        Number of products.
    alpha : np.ndarray, shape (N,)
        Hard minimum inventory constraint per product.
    initial_inventory : np.ndarray, shape (N,)
        Starting inventory.
    x_opt : np.ndarray, shape (T, N), dtype int
        Pre-computed binary decision matrix from ``solve_ilp_offline``.
    prices : np.ndarray, shape (T, N)
        Market prices used during the ILP solve.  The agent bids
        ``prices[t, i] + epsilon`` for products where ``x_opt[t, i] == 1``.
    ilp_total_cost : float
        Total ILP-optimal cost (for metadata / reporting).
    feasibility : list[bool]
        Per-product feasibility flags from the ILP solver.
    """

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

        # Metadata for reporting.
        self.ilp_total_cost = ilp_total_cost
        self.feasibility = feasibility if feasibility is not None else [True] * n_products

    @property
    def schedule(self) -> np.ndarray:
        """Return the full (T, N) binary decision matrix."""
        return self._x_opt

    def bid(self, state: AuctionState) -> np.ndarray:
        """Replay the pre-computed ILP decision for the current round.

        For each product i where x_opt[t, i] == 1, bid the recorded
        market price plus a small epsilon to guarantee winning.

        Parameters
        ----------
        state : AuctionState
            Current auction state.

        Returns
        -------
        np.ndarray, shape (N,)
            Bid vector.  Positive entries are bids; zeros mean no bid.
        """
        t = self._current_t
        bids_out = np.zeros(self.n_products)

        if t < self._T:
            for i in range(self.n_products):
                if self._x_opt[t, i] == 1:
                    # Bid the pre-roll market price + epsilon.
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
