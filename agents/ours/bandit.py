from __future__ import annotations

import numpy as np

from agents.base import AuctionState, BaseAgent, N_PRODUCTS


class AllSeeingBanditAgent(BaseAgent):
    """Lyapunov drift-plus-penalty bidding with full competitor bid visibility.

    Buys product i when the market price is below H_i(t) * q_i / V,
    where H_i = beta_i - s_i is the inventory deficit and V trades off
    cost vs. safety.
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
        self.d_max = d_max if d_max is not None else np.full(n_products, 5.0)
        self.beta = self.alpha + self.d_max

    def bid(self, state: AuctionState) -> np.ndarray:
        assert state.bids is not None, "AllSeeingBanditAgent requires full bid visibility."

        bids_out = np.zeros(self.n_products)
        H = self.inventory_deficit(self.beta)

        for i in range(self.n_products):
            q_i = state.quantities[i]
            if q_i <= 0:
                continue

            market_price = max(
                other_bids[i]
                for uid, other_bids in state.bids.items()
                if uid != self.agent_id
            )
            threshold = H[i] * q_i / self.V if self.V > 0 else np.inf

            if market_price < threshold:
                bids_out[i] = market_price + market_price * 0.01 + 0.01

        self.log({"t": state.t, "H": H.copy(), "bids": bids_out.copy(),
                  "inventory": self.inventory.copy()})
        return bids_out


class IncrementalOLS:
    """Online OLS that maintains sufficient statistics instead of refitting each round."""

    def __init__(self, n_features: int, reg: float = 1.0) -> None:
        self.n_features = n_features
        self.reg = reg
        self.XtX = reg * np.eye(n_features)
        self.Xty = np.zeros(n_features)
        self.n_samples = 0
        self._theta: np.ndarray | None = None
        self._dirty = True

    def update(self, x: np.ndarray, y: float) -> None:
        self.XtX += np.outer(x, x)
        self.Xty += x * y
        self.n_samples += 1
        self._dirty = True

    @property
    def theta(self) -> np.ndarray:
        if self._dirty or self._theta is None:
            self._theta = np.linalg.solve(self.XtX, self.Xty)
            self._dirty = False
        return self._theta

    def predict(self, x: np.ndarray) -> float:
        return float(x @ self.theta)

    def reset(self) -> None:
        self.XtX = self.reg * np.eye(self.n_features)
        self.Xty = np.zeros(self.n_features)
        self.n_samples = 0
        self._theta = None
        self._dirty = True


class PartiallyBlindBanditAgent(BaseAgent):
    """Lyapunov bidding where competitor bids are hidden and estimated via OLS.

    Observes competitor inventories, fits a linear model per competitor per
    product, and uses the predicted max bid as the market price estimate.
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
        self.d_max = d_max if d_max is not None else np.full(n_products, 5.0)
        self.beta = self.alpha + self.d_max
        self.competitor_ids = competitor_ids or []
        self.ols_reg = ols_reg
        self.models: dict[int, dict[int, IncrementalOLS]] = {
            i: {o: IncrementalOLS(n_features=n_products, reg=ols_reg) for o in self.competitor_ids}
            for i in range(n_products)
        }

    def observe_outcome(
        self,
        product_idx: int,
        competitor_id: int,
        competitor_inventory: np.ndarray,
        actual_bid: float,
    ) -> None:
        if competitor_id in self.models[product_idx]:
            self.models[product_idx][competitor_id].update(competitor_inventory, actual_bid)

    def _estimate_market_price(self, product_idx: int, inventories: dict[int, np.ndarray]) -> float:
        preds = []
        for o in self.competitor_ids:
            if o not in inventories:
                continue
            model = self.models[product_idx][o]
            if model.n_samples < 2:
                # too few samples — return -inf so Lyapunov always triggers
                return -np.inf
            preds.append(model.predict(inventories[o]))
        return max(preds) if preds else -np.inf

    def bid(self, state: AuctionState) -> np.ndarray:
        bids_out = np.zeros(self.n_products)
        H = self.inventory_deficit(self.beta)

        for i in range(self.n_products):
            q_i = state.quantities[i]
            if q_i <= 0:
                continue

            estimated_price = self._estimate_market_price(i, state.inventories)
            threshold = H[i] * q_i / self.V if self.V > 0 else np.inf

            if estimated_price < threshold:
                if np.isneginf(estimated_price):
                    bids_out[i] = max(threshold, 0.01)
                else:
                    bids_out[i] = estimated_price + estimated_price * 0.01 + 0.01

        self.log({"t": state.t, "H": H.copy(), "bids": bids_out.copy(),
                  "inventory": self.inventory.copy()})
        return bids_out

    def reset(self, initial_inventory: np.ndarray | None = None) -> None:
        super().reset(initial_inventory)
        for product_models in self.models.values():
            for ols in product_models.values():
                ols.reset()
