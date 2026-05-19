"""Auction simulation environment.

Orchestrates a sequence of one-shot, sealed-bid auctions between one
*our* agent (All-Seeing or Partially-Blind) and a set of competitor agents.

Round lifecycle:
1. Build AuctionState visible to each agent type.
2. Collect bids from every participant.
3. Resolve each product auction independently (highest bid wins).
4. Update every agent's inventory via BaseAgent.update_inventory().
5. Feed revealed-bid feedback to PartiallyBlindBanditAgent (online OLS).
6. Record per-round metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional

import numpy as np

from agents.base import AuctionState, BaseAgent, N_PRODUCTS
from agents.ours.bandit import PartiallyBlindBanditAgent


@dataclass
class RoundResult:
    """Outcome of a single auction round."""

    t: int
    quantities: np.ndarray
    depletions: np.ndarray
    all_bids: dict[int, np.ndarray]
    winners: np.ndarray
    winning_bids: np.ndarray
    our_cost: float
    our_won: np.ndarray
    inventories_after: dict[int, np.ndarray] = field(default_factory=dict)


class AuctionSimulation:
    """One-shot sealed-bid auction simulation."""

    def __init__(
        self,
        our_agent: BaseAgent,
        competitors: list[BaseAgent],
        n_products: int = N_PRODUCTS,
        quantity_source: Optional[Iterator[np.ndarray]] = None,
        depletion_source: Optional[Iterator[np.ndarray]] = None,
        default_quantities: Optional[np.ndarray] = None,
        default_depletions: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.our_agent = our_agent
        self.competitors = competitors
        self.all_agents: list[BaseAgent] = [our_agent] + list(competitors)
        self.n_products = n_products

        self.quantity_source = quantity_source
        self.depletion_source = depletion_source

        self.rng = np.random.default_rng(seed)

        self.default_quantities = (
            default_quantities if default_quantities is not None
            else np.full(n_products, 5.0)
        )
        self.default_depletions = (
            default_depletions if default_depletions is not None
            else np.full(n_products, 1.0)
        )

        self.results: list[RoundResult] = []
        self._t = 0

    def _next_quantities(self) -> np.ndarray:
        if self.quantity_source is not None:
            try:
                return np.asarray(next(self.quantity_source), dtype=float)
            except StopIteration:
                self.quantity_source = None
        noise = self.rng.uniform(0.8, 1.2, size=self.n_products)
        return self.default_quantities * noise

    def _next_depletions(self) -> np.ndarray:
        if self.depletion_source is not None:
            try:
                return np.asarray(next(self.depletion_source), dtype=float)
            except StopIteration:
                self.depletion_source = None
        return self.default_depletions.copy()

    @staticmethod
    def _resolve(
        bids_matrix: np.ndarray,
        agent_ids: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Determine winner per product by highest bid; returns (winners, winning_bids)."""
        n_products = bids_matrix.shape[1]
        winners = np.full(n_products, -1, dtype=int)
        winning_bids = np.zeros(n_products)

        for i in range(n_products):
            col = bids_matrix[:, i]
            if col.max() <= 0:
                continue
            idx = int(np.argmax(col))
            winners[i] = agent_ids[idx]
            winning_bids[i] = col[idx]

        return winners, winning_bids

    def step(self) -> RoundResult:
        """Run one auction round and return its result."""
        t = self._t
        quantities = self._next_quantities()
        depletions = self._next_depletions()

        inventories: dict[int, np.ndarray] = {
            a.agent_id: a.inventory.copy() for a in self.all_agents
        }

        # Competitors always bid first (their bids become visible for the All-Seeing agent).
        competitor_bids: dict[int, np.ndarray] = {}
        for c in self.competitors:
            state_c = AuctionState(
                t=t,
                quantities=quantities,
                inventories=inventories,
                bids=None,          # competitors never see each other's bids
                depletions=depletions,
            )
            competitor_bids[c.agent_id] = c.bid(state_c)

        is_all_seeing = not isinstance(self.our_agent, PartiallyBlindBanditAgent)
        our_state = AuctionState(
            t=t,
            quantities=quantities,
            inventories=inventories,
            bids=competitor_bids if is_all_seeing else None,
            depletions=depletions,
        )
        our_bid_vec = self.our_agent.bid(our_state)

        all_bids: dict[int, np.ndarray] = {
            self.our_agent.agent_id: our_bid_vec,
            **competitor_bids,
        }
        agent_ids = list(all_bids.keys())
        bids_matrix = np.stack([all_bids[aid] for aid in agent_ids])

        winners, winning_bids = self._resolve(bids_matrix, agent_ids)

        our_id = self.our_agent.agent_id
        our_won = winners == our_id
        our_cost = float((winning_bids * our_won).sum())

        for agent in self.all_agents:
            won_mask = winners == agent.agent_id
            agent.update_inventory(won_mask, quantities, depletions)

        # In a sealed-bid auction only the WINNING bid is revealed.
        # We update the OLS model only when a competitor won product i,
        # feeding (their_inventory, winning_bid) as the training pair.
        if isinstance(self.our_agent, PartiallyBlindBanditAgent):
            for c in self.competitors:
                for i in range(self.n_products):
                    if winners[i] == c.agent_id:
                        self.our_agent.observe_outcome(
                            product_idx=i,
                            competitor_id=c.agent_id,
                            competitor_inventory=inventories[c.agent_id],
                            actual_bid=winning_bids[i],
                        )

        result = RoundResult(
            t=t,
            quantities=quantities,
            depletions=depletions,
            all_bids=all_bids,
            winners=winners,
            winning_bids=winning_bids,
            our_cost=our_cost,
            our_won=our_won,
            inventories_after={a.agent_id: a.inventory.copy() for a in self.all_agents},
        )
        self.results.append(result)
        self._t += 1
        return result

    def run(
        self,
        n_rounds: int,
        verbose: bool = False,
        callback: Optional[Callable[[RoundResult], None]] = None,
    ) -> list[RoundResult]:
        """Run *n_rounds* auction rounds sequentially and return all results."""
        for _ in range(n_rounds):
            result = self.step()
            if verbose:
                _print_round(result)
            if callback is not None:
                callback(result)
        return self.results

    def summary(self) -> dict:
        """Return aggregate metrics over all completed rounds."""
        if not self.results:
            return {}

        total_cost = sum(r.our_cost for r in self.results)
        n_rounds = len(self.results)

        win_counts = np.zeros(self.n_products)
        bid_counts = np.zeros(self.n_products)
        rounds_won = 0
        for r in self.results:
            win_counts += r.our_won.astype(float)
            bid_counts += (r.all_bids[self.our_agent.agent_id] > 0).astype(float)
            if np.any(r.our_won):
                rounds_won += 1

        safe_bid_counts = np.where(bid_counts > 0, bid_counts, 1.0)
        win_rate = np.where(bid_counts > 0, win_counts / safe_bid_counts, 0.0)

        violations = 0
        for r in self.results:
            inv = r.inventories_after[self.our_agent.agent_id]
            violations += int(np.any(inv < self.our_agent.alpha))

        return {
            "n_rounds": n_rounds,
            "total_cost": total_cost,
            "avg_cost_per_round": total_cost / n_rounds,
            "win_rate_per_product": win_rate.tolist(),
            "rounds_won": rounds_won,
            "rounds_won_percentage": rounds_won / n_rounds,
            "constraint_violations": violations,
            "violation_rate": violations / n_rounds,
            "final_inventory": self.our_agent.inventory.tolist(),
        }

    def reset(self) -> None:
        """Reset simulation state (but not agents)."""
        self.results.clear()
        self._t = 0


def _print_round(r: RoundResult) -> None:
    won_str = " ".join(
        f"P{i}({'✓' if r.our_won[i] else '✗'})"
        for i in range(len(r.quantities))
    )
    print(f"[t={r.t:4d}] cost={r.our_cost:8.2f} | {won_str}")
