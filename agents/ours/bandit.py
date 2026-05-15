from __future__ import annotations

from typing import Any

import numpy as np

from agents.base import AuctionState, BaseAgent, N_PRODUCTS


# ════════════════════════════════════════════════════════════════════════
#  IV-B  All-Seeing Bandit Agent
# ════════════════════════════════════════════════════════════════════════

class AllSeeingBanditAgent(BaseAgent):
    """Lyapunov drift-plus-penalty bidding with full bid visibility.

    Bidding rule (Eq. 7 in the proposal):

        x_i(t) = 1   if  b_i(t) < H_i(t) / V · q_i(t)
                  0   otherwise

    where  H_i(t) = β_i − s_i(t)  is the inventory deficit,
    V > 0 trades off cost minimisation vs. inventory safety, and
    β_i = α_i + d_i^max  is the soft constraint buffer.

    Parameters
    ----------
    agent_id : int
        Unique agent identifier.
    n_products : int
        Number of products.
    alpha : np.ndarray
        Hard minimum inventory constraint per product.
    initial_inventory : np.ndarray
        Starting inventory.
    V : float
        Lyapunov trade-off hyperparameter.  Higher → more cost-conscious,
        lower → more inventory-safe.
    d_max : np.ndarray | None
        Upper bound on depletion per product.  Used to set
        β_i = α_i + d_i^max.  If *None*, defaults to 5.0 for every product.
    """

    def __init__(
        self,
        agent_id: int,
        n_products: int = N_PRODUCTS,
        alpha: np.ndarray | None = None,
        initial_inventory: np.ndarray | None = None,
        V: float = 1.0,
        d_max: np.ndarray | None = None,
    ) -> None:
        super().__init__(agent_id, n_products, alpha, initial_inventory)
        self.V = V

        # d_max per product (assumed bounded per the paper)
        self.d_max = (
            d_max if d_max is not None
            else np.full(n_products, 5.0)
        )

        # Soft constraint:  β_i = α_i + d_i^max   (Appendix)
        self.beta = self.alpha + self.d_max

    # ── core bidding logic ─────────────────────────────────────────────
    def bid(self, state: AuctionState) -> np.ndarray:
        """Decide bids using the Lyapunov bidding rule.

        In the All-Seeing setting we observe every competitor's bid, so
        we know the *market price* (highest competing bid) exactly.

        We bid ``market_price + ε`` for product *i* only when the
        Lyapunov rule says to buy (i.e. the product is cheap relative to
        our deficit).
        """
        assert state.bids is not None, (
            "AllSeeingBanditAgent requires full bid visibility."
        )

        bids_out = np.zeros(self.n_products)

        # H_i(t) = β_i − s_i(t)
        H = self.inventory_deficit(self.beta)

        for i in range(self.n_products):
            q_i = state.quantities[i]
            if q_i <= 0:
                # Nothing offered for this product
                continue

            # Market price = max bid among competitors for product i
            market_price = max(
                other_bids[i]
                for uid, other_bids in state.bids.items()
                if uid != self.agent_id
            )

            # Lyapunov bidding rule:  buy if  b_i < H_i · q_i / V
            # Equivalently:  H_i · q_i > V · b_i
            threshold = H[i] * q_i / self.V if self.V > 0 else np.inf

            if market_price < threshold:
                # Bid just above market price to win
                epsilon = market_price * 0.01 + 0.01
                bids_out[i] = market_price + epsilon

        # Log decision
        self.log({
            "t": state.t,
            "H": H.copy(),
            "bids": bids_out.copy(),
            "inventory": self.inventory.copy(),
        })

        return bids_out


# ════════════════════════════════════════════════════════════════════════
#  IV-C  Partially Blind Bandit Agent
# ════════════════════════════════════════════════════════════════════════

class IncrementalOLS:
    """Incremental (online) Ordinary Least Squares estimator.

    Maintains running sufficient statistics so that each update is O(d²)
    instead of re-fitting from scratch.

    Solves  θ = (X^T X)^{-1} X^T y  incrementally.

    Parameters
    ----------
    n_features : int
        Dimensionality of the feature vector K_o.
    reg : float
        Ridge regularisation term (λ·I added to X^T X) for numerical
        stability at the start when we have few samples.
    """

    def __init__(self, n_features: int, reg: float = 1.0) -> None:
        self.n_features = n_features
        self.reg = reg

        # Sufficient statistics
        self.XtX = reg * np.eye(n_features)  # (d, d)
        self.Xty = np.zeros(n_features)      # (d,)
        self.n_samples = 0

        # Cached parameter vector
        self._theta: np.ndarray | None = None
        self._dirty = True

    def update(self, x: np.ndarray, y: float) -> None:
        """Add one observation (x, y) and invalidate cached θ."""
        self.XtX += np.outer(x, x)
        self.Xty += x * y
        self.n_samples += 1
        self._dirty = True

    @property
    def theta(self) -> np.ndarray:
        """Current parameter estimate θ = (X^T X)^{-1} X^T y."""
        if self._dirty or self._theta is None:
            self._theta = np.linalg.solve(self.XtX, self.Xty)
            self._dirty = False
        return self._theta

    def predict(self, x: np.ndarray) -> float:
        """Predict  E[C | K] = K^T θ."""
        return float(x @ self.theta)

    def reset(self) -> None:
        """Reset to initial state."""
        self.XtX = self.reg * np.eye(self.n_features)
        self.Xty = np.zeros(self.n_features)
        self.n_samples = 0
        self._theta = None
        self._dirty = True


class PartiallyBlindBanditAgent(BaseAgent):
    """Lyapunov bidding with hidden bids — estimate market price via OLS.

    In the Partially Blind setting (Section IV-C) we can *see* every
    competitor's inventory but **not** their bids.  We model each
    competitor *o*'s bid for product *i* as a linear function of their
    inventory (context):

        E[C_{i,o} | K_o] = K_o^T  θ_{i,o}

    and learn θ_{i,o} online with incremental OLS.  The estimated market
    price is

        P_i(t) = max_o  E[C_{i,o} | K_o]

    and our bid is  P_i(t) + ε  whenever the Lyapunov rule triggers.

    Parameters
    ----------
    agent_id : int
        Unique agent identifier.
    n_products : int
        Number of products.
    alpha : np.ndarray
        Hard minimum inventory constraints.
    initial_inventory : np.ndarray
        Starting inventory.
    V : float
        Lyapunov trade-off hyperparameter.
    d_max : np.ndarray | None
        Upper-bound on depletion per product.
    competitor_ids : list[int]
        IDs of all competitor agents (needed to init OLS models).
    ols_reg : float
        Ridge regularisation for OLS warmup.
    """

    def __init__(
        self,
        agent_id: int,
        n_products: int = N_PRODUCTS,
        alpha: np.ndarray | None = None,
        initial_inventory: np.ndarray | None = None,
        V: float = 1.0,
        d_max: np.ndarray | None = None,
        competitor_ids: list[int] | None = None,
        ols_reg: float = 1.0,
    ) -> None:
        super().__init__(agent_id, n_products, alpha, initial_inventory)
        self.V = V
        self.d_max = (
            d_max if d_max is not None
            else np.full(n_products, 5.0)
        )
        self.beta = self.alpha + self.d_max

        self.competitor_ids = competitor_ids or []
        self.ols_reg = ols_reg

        # OLS models:  models[product_i][competitor_o]
        # Feature vector K_o = competitor o's inventory, shape (n_products,)
        self.models: dict[int, dict[int, IncrementalOLS]] = {
            i: {
                o: IncrementalOLS(n_features=n_products, reg=ols_reg)
                for o in self.competitor_ids
            }
            for i in range(n_products)
        }

    # ── online learning ────────────────────────────────────────────────
    def observe_outcome(
        self,
        product_idx: int,
        competitor_id: int,
        competitor_inventory: np.ndarray,
        actual_bid: float,
    ) -> None:
        """Feed a revealed (inventory, bid) pair into the OLS model.

        After an auction round resolves, the winning bid is typically
        revealed.  This method lets the agent learn from that feedback.
        """
        if competitor_id in self.models[product_idx]:
            self.models[product_idx][competitor_id].update(
                competitor_inventory, actual_bid,
            )

    def _estimate_market_price(
        self,
        product_idx: int,
        inventories: dict[int, np.ndarray],
    ) -> float:
        """Estimate P_i(t) = max_o E[C_{i,o} | K_o]  (Eq. 9).

        During warm-up (< 2 samples) returns ``-inf`` so the Lyapunov rule
        always triggers and the agent bids its threshold price.  This avoids
        the arbitrary hard-coded default that caused systematic under-bidding
        for low-price products like beef.
        """
        preds = []
        for o in self.competitor_ids:
            if o not in inventories:
                continue
            model = self.models[product_idx][o]
            if model.n_samples < 2:
                # Return -inf: Lyapunov rule will trigger and bid = threshold
                return -np.inf
            preds.append(model.predict(inventories[o]))
        return max(preds) if preds else -np.inf

    # ── core bidding logic ─────────────────────────────────────────────
    def bid(self, state: AuctionState) -> np.ndarray:
        """Bid using estimated market prices + Lyapunov rule.

        Eq. 10:  b_i = P_i(t) + ε   (if Lyapunov triggers)

        When OLS is in warm-up (_estimate_market_price returns -inf),
        estimated_price < threshold is always True so the agent bids
        its own Lyapunov threshold directly, guaranteeing participation
        while learning.
        """
        bids_out = np.zeros(self.n_products)
        H = self.inventory_deficit(self.beta)

        for i in range(self.n_products):
            q_i = state.quantities[i]
            if q_i <= 0:
                continue

            estimated_price = self._estimate_market_price(i, state.inventories)

            # Lyapunov rule:  buy if  H_i · q_i > V · estimated_price
            threshold = H[i] * q_i / self.V if self.V > 0 else np.inf

            if estimated_price < threshold:
                if np.isneginf(estimated_price):
                    # Warm-up: bid the threshold itself (agent's max willingness-to-pay)
                    bids_out[i] = max(threshold, 0.01)
                else:
                    epsilon = estimated_price * 0.01 + 0.01
                    bids_out[i] = estimated_price + epsilon

        self.log({
            "t": state.t,
            "H": H.copy(),
            "bids": bids_out.copy(),
            "inventory": self.inventory.copy(),
        })

        return bids_out

    def reset(self, initial_inventory: np.ndarray | None = None) -> None:
        """Reset agent state *and* OLS models."""
        super().reset(initial_inventory)
        for product_models in self.models.values():
            for ols in product_models.values():
                ols.reset()
