"""
Entry point for the inventory-constrained dynamic bidding experiment.

Usage examples
--------------
# All-Seeing bandit, 150 rounds, verbose:
    python -m scripts.run --mode all_seeing --n_rounds 150 --verbose

# Partially-Blind bandit, custom V:
    python -m scripts.run --mode partially_blind --V 3.0

# With EUROSTAT data (requires data/raw/ TSVs):
    python -m scripts.run --mode all_seeing --use_real_data --country DE

# Override alpha and seed:
    python -m scripts.run --alpha 10 5 10 15 --seed 7

# Save visualisation dashboard:
    python -m scripts.run --n_rounds 150 --plot results/dashboard.png
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ── project root on sys.path ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.base import N_PRODUCTS, PRODUCTS
from agents.ours.bandit import AllSeeingBanditAgent, PartiallyBlindBanditAgent
from env.sim import AuctionSimulation
from tools.helper_run import (
    build_competitors,
    quantity_iter_from_generator,
    depletion_iter,
    derive_data_stats,
)


# ════════════════════════════════════════════════════════════════════════
#  Result-saving utilities
# ════════════════════════════════════════════════════════════════════════

RESULTS_DIR = Path("results")


def _rp(run_dir: Path, stem: str, suffix: str = ".txt") -> Path:
    """Return ``run_dir/stem+suffix``.  run_dir must already exist."""
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
    """Write a structured summary file built directly from sim.summary().

    Unlike capturing stdout, this pulls data from the simulation object so
    the file is always accurate regardless of console verbosity settings.

    Parameters
    ----------
    path     : destination Path
    ts       : run timestamp string
    run_label: e.g. "All-Seeing" or "Partially-Blind"
    sim      : completed AuctionSimulation instance
    args     : parsed argparse.Namespace
    alpha    : resolved hard inventory minimum (N,)
    d_max    : resolved per-auction depletion (N,)
    init_inv : resolved initial inventory (N,)
    products : sequence of product name strings
    """
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

    # Per-product table: alpha, d_max, init_inv
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


# ════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py",
        description="Run the inventory-constrained dynamic auction simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── scenario ──────────────────────────────────────────────────────
    p.add_argument(
        "--mode",
        choices=["all_seeing", "partially_blind"],
        default="all_seeing",
        help="Which of our agent types to use.",
    )
    p.add_argument(
        "--n_rounds",
        type=int,
        default=150,
        help="Number of auction rounds to simulate.",
    )

    # ── our agent ─────────────────────────────────────────────────────
    p.add_argument(
        "--V",
        type=float,
        default=2.0,
        help="Lyapunov trade-off hyperparameter (higher → more cost-conscious).",
    )
    p.add_argument(
        "--a",
        type=float,
        default=0.5,
        metavar="SCALE",
        help=(
            "Safety-stock scale factor a ∈ (0, 1]. "
            "Sets alpha = a × mean_qty_per_auction. "
            "Smaller a → looser safety-stock constraint."
        ),
    )
    p.add_argument(
        "--d_rate",
        type=float,
        default=0.5,
        metavar="RATE",
        help=(
            "Depletion rate d_rate ∈ (0, 1]. "
            "Sets d_max = d_rate × alpha. "
            "Controls how quickly inventory drains relative to α: "
            "d_rate=0.1 → ~10 rounds of runway; d_rate=1.0 → one round."
        ),
    )
    p.add_argument(
        "--alpha",
        type=float,
        nargs=N_PRODUCTS,
        default=None,
        metavar=("milk", "eggs", "poultry", "beef"),
        help=(
            "Hard minimum inventory constraint per product (tonnes). "
            "Defaults to 7× per-auction depletion derived from real data."
        ),
    )
    p.add_argument(
        "--d_max",
        type=float,
        nargs=N_PRODUCTS,
        default=None,
        metavar=("milk", "eggs", "poultry", "beef"),
        help=(
            "Max depletion per auction slot per product (tonnes). "
            "Defaults to mean daily depletion ÷ auctions_per_day from real data."
        ),
    )
    p.add_argument(
        "--init_inventory",
        type=float,
        nargs=N_PRODUCTS,
        default=None,
        metavar=("milk", "eggs", "poultry", "beef"),
        help="Initial inventory. Defaults to 2×alpha.",
    )

    # ── competitors ───────────────────────────────────────────────────
    p.add_argument(
        "--competitors",
        type=str,
        default="stochastic,linear,naive",
        help=(
            "Comma-separated competitor types. "
            "Choices per entry: stochastic | linear | naive."
        ),
    )

    # ── data source ───────────────────────────────────────────────────
    p.add_argument(
        "--use_real_data",
        action="store_true",
        help="Use EUROSTAT data as quantity source (requires data/raw/ TSVs).",
    )
    p.add_argument(
        "--country",
        type=str,
        default="DE",
        help="ISO-2 EUROSTAT country code.",
    )
    p.add_argument(
        "--auctions_per_day",
        type=int,
        default=1,
        help="Auction slots per calendar day (used with EUROSTAT data).",
    )

    # ── output ────────────────────────────────────────────────────────
    p.add_argument("--seed",    type=int,  default=42,    help="Global random seed.")
    p.add_argument("--verbose", action="store_true",      help="Print per-round output.")
    p.add_argument(
        "--plot",
        action="store_true",
        help=(
            "Save ALL visualisations to results/{timestamp}/. "
            "Produces: dashboard per agent, regret comparison (with --compare), "
            "and 4-panel convergence dashboard."
        ),
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help=(
            "Run BOTH All-Seeing and Partially-Blind on the same price sequence "
            "and produce a shared regret comparison plot."
        ),
    )

    return p


# ════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════

def main(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)

    # ── timestamp & run directory ──────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── save config JSON ─────────────────────────────────────────────
    config_path = _rp(run_dir, "config", ".json")
    config_dict = vars(args).copy()
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
    print(f"  [run] Results dir   → {run_dir}/")
    print(f"  [run] Timestamp     : {ts}")

    # ── derive alpha / d_max from real data (unless CLI override) ─────
    _FALLBACK_ALPHA = np.array([10.0,  5.0, 10.0, 15.0])
    _FALLBACK_DMAX  = np.array([ 3.0,  2.0,  3.0,  4.0])

    data_mean_qty = data_d_max = data_alpha = None
    try:
        data_mean_qty, data_d_max, data_alpha = derive_data_stats(
            args.country, args.auctions_per_day, a=args.a, d_rate=args.d_rate
        )
        print(f"  [data] Derived stats from EUROSTAT ({args.country}), a={args.a}, d_rate={args.d_rate}:")
        print(f"         mean qty/auction : {np.round(data_mean_qty, 2).tolist()}")
        print(f"         alpha (a×mean)   : {np.round(data_alpha, 2).tolist()}")
        print(f"         d_max (d_rate×α) : {np.round(data_d_max, 2).tolist()}")
    except Exception as exc:
        print(f"  [data] Could not load EUROSTAT data ({exc}). "
              "Using hardcoded fallback values.")

    # Priority: CLI flag > data-derived > hardcoded fallback
    d_max = (
        np.array(args.d_max, dtype=float) if args.d_max is not None
        else (data_d_max if data_d_max is not None else _FALLBACK_DMAX)
    )
    alpha = (
        np.array(args.alpha, dtype=float) if args.alpha is not None
        else (data_alpha if data_alpha is not None else _FALLBACK_ALPHA)
    )
    init_inv = (
        np.array(args.init_inventory, dtype=float)
        if args.init_inventory is not None
        else alpha + d_max          # = β (Lyapunov soft target), so H_i(0) = 0
    )

    # ── competitors ───────────────────────────────────────────────────
    competitors = build_competitors(args.competitors, alpha, init_inv, args.seed)
    competitor_ids = [c.agent_id for c in competitors]

    print("=" * 60)
    print(f"  Mode         : {args.mode}")
    print(f"  Rounds       : {args.n_rounds}")
    print(f"  V            : {args.V}")
    print(f"  a (scale)    : {args.a}")
    print(f"  d_rate       : {args.d_rate}")
    print(f"  Alpha        : {np.round(alpha, 2).tolist()}")
    print(f"  d_max        : {np.round(d_max, 2).tolist()}")
    print(f"  Competitors  : {args.competitors}")
    print(f"  Seed         : {args.seed}")
    print("=" * 60)

    # ── our agent(s) ──────────────────────────────────────────────────
    # In --compare mode we build both and run the sim twice.
    agent_as = AllSeeingBanditAgent(
        agent_id=0,
        alpha=alpha,
        initial_inventory=init_inv.copy(),
        V=args.V,
        d_max=d_max,
    )
    agent_pb = PartiallyBlindBanditAgent(
        agent_id=0,
        alpha=alpha,
        initial_inventory=init_inv.copy(),
        V=args.V,
        d_max=d_max,
        competitor_ids=competitor_ids,
    )

    if args.compare:
        agents_to_run = [("All-Seeing", agent_as), ("Partially-Blind", agent_pb)]
    elif args.mode == "all_seeing":
        agents_to_run = [("All-Seeing", agent_as)]
    else:
        agents_to_run = [("Partially-Blind", agent_pb)]

    # ── pre-generate the price/quantity sequence (shared across both agents) ─
    # We pre-roll the sim with a dummy agent to collect the environment
    # trajectory, then replay it deterministically for each agent.
    # Strategy: store quantities+depletions from a reference run,
    # then replay using list iterators — same sequence for both agents.

    def _make_env_sequences():
        """Build quantity and depletion lists for the full horizon.

        Quantities come from the AuctionDataGenerator (real EUROSTAT auction lots).
        Products with 0 quantity in the generator (missing EUROSTAT data, e.g. beef)
        fall back to d_max so the agent always has something to bid on.

        Depletions are set to d_max — our designed per-slot depletion cap — NOT
        gen.depletion_rates, which are raw EUROSTAT daily production totals in
        completely different units (orders of magnitude larger).
        """
        if args.use_real_data:
            try:
                from datagen.generate_data import AuctionDataGenerator
                gen = AuctionDataGenerator(
                    country=args.country,
                    auctions_per_day=args.auctions_per_day,
                    n_days=args.n_rounds // args.auctions_per_day + 1,
                    seed=args.seed,
                )
                raw_qtys = list(quantity_iter_from_generator(gen))[:args.n_rounds]

                # Replace 0-quantity entries (missing EUROSTAT products) with d_max
                # so the agent always has a positive lot to bid on each round.
                qtys = [
                    np.where(q > 0, q, d_max)
                    for q in raw_qtys
                ]

                # Depletions = d_max * 0.5 per slot.
                # d_max is the worst-case cap; expected demand is half that,
                # matching default_depletions used in the synthetic fallback.
                deps = [(d_max * 0.5).copy() for _ in range(args.n_rounds)]

                print(f"  [data] Using real EUROSTAT data — country={args.country}")
                print(f"         qty/slot sample  : {np.round(qtys[0], 1).tolist()}")
                print(f"         depletion/slot   : {np.round(d_max * 0.5, 2).tolist()}  (= d_max × 0.5)")
                return qtys, deps
            except Exception as exc:
                print(f"  [data] WARNING: Could not load real data ({exc}). "
                      "Falling back to synthetic data.")
        return None, None  # sim generates internally

    pre_qtys, pre_deps = _make_env_sequences()

    def _build_sim(agent):
        """Create a fresh AuctionSimulation for *agent*."""
        _competitors = build_competitors(args.competitors, alpha, init_inv, args.seed)
        # default_quantities must match the same scale as d_max/alpha so that
        # when no external quantity_source is provided, the agent can actually
        # replenish inventory each round.
        _default_qty = (
            data_mean_qty if data_mean_qty is not None
            else d_max / max(args.a, 1e-6)   # recover mean_qty = d_max / a
        )
        return AuctionSimulation(
            our_agent=agent,
            competitors=_competitors,
            n_products=N_PRODUCTS,
            quantity_source=iter(pre_qtys) if pre_qtys else None,
            depletion_source=iter(pre_deps) if pre_deps else None,
            default_quantities=_default_qty,
            default_depletions=d_max * 0.5,
            seed=args.seed,
        )

    # ════════════════════════════════════════════════════════════════════
    #  RUN LOOP  — iterate over agent(s)
    # ════════════════════════════════════════════════════════════════════
    all_run_results: list[tuple[str, list, object]] = []  # (label, results, agent)

    for run_label, our_agent in agents_to_run:
        sim = _build_sim(our_agent)

        total_cost = 0.0
        win_counts = np.zeros(N_PRODUCTS)
        bid_counts = np.zeros(N_PRODUCTS)
        violations = 0

        print()
        print(f"── {run_label} " + "─" * (57 - len(run_label)))
        print(f"{'t':>5} {'cost':>8}  {'cumcost':>9}  "
              f"{'inventory (our)':30}  won")
        print("-" * 75)

        # ════════════════════════════════════════════════════════════════
        #  GAME LOOP  — one iteration = one auction round
        # ════════════════════════════════════════════════════════════════
        for t in range(args.n_rounds):

            # ── advance simulation by one step ────────────────────────
            r = sim.step()

            # ── per-round: update running stats ───────────────────────
            total_cost += r.our_cost
            win_counts += r.our_won.astype(float)
            bid_counts += (r.all_bids[our_agent.agent_id] > 0).astype(float)

            inv_after = r.inventories_after[our_agent.agent_id]
            violated  = bool(np.any(inv_after < our_agent.alpha))
            if violated:
                violations += 1

            # ── per-round: console output ─────────────────────────────
            if args.verbose:
                won_str   = " ".join(
                    f"{PRODUCTS[i][0].upper()}{'✓' if r.our_won[i] else '✗'}"
                    for i in range(N_PRODUCTS)
                )
                inv_str   = " ".join(f"{v:6.1f}" for v in inv_after)
                viol_flag = " ⚠" if violated else ""
                print(f"{t:5d} {r.our_cost:8.2f}  {total_cost:9.2f}  "
                      f"{inv_str}  {won_str}{viol_flag}")

            # ── per-round: add any investigation logic here ───────────
            # Examples:
            #   print(r.all_bids)            — every agent's bid vector
            #   print(r.winners)             — who won each product
            #   print(r.winning_bids)        — at what price
            #   print(r.quantities)          — lot sizes this round
            #   print(our_agent.history[-1]) — our agent's internal log entry

        # ════════════════════════════════════════════════════════════════
        #  END OF GAME LOOP
        # ════════════════════════════════════════════════════════════════

        safe_bids = np.where(bid_counts > 0, bid_counts, 1.0)
        win_rate  = np.where(bid_counts > 0, win_counts / safe_bids, 0.0)
        rounds_won_any = sum(1 for r in sim.results if np.any(r.our_won))

        print()
        print("=" * 60)
        print(f"  SUMMARY — {run_label}")
        print("=" * 60)
        print(f"  Rounds simulated     : {args.n_rounds}")
        print(f"  Rounds won (any)     : {rounds_won_any} ({rounds_won_any / args.n_rounds * 100:.1f}%)")
        print(f"  Total cost           : {total_cost:.2f}")
        print(f"  Avg cost / round     : {total_cost / args.n_rounds:.2f}")
        print(f"  Constraint violations: {violations} "
              f"({violations / args.n_rounds * 100:.1f}%)")
        print(f"  Final inventory      : "
              f"{[round(v, 1) for v in our_agent.inventory.tolist()]}")
        print(f"  Win rate per product :")
        for prod, wr in zip(PRODUCTS, win_rate):
            print(f"    {prod:12s}: {wr * 100:5.1f}%")
        print("=" * 60)

        all_run_results.append((run_label, sim.results, our_agent))

        # ── per-agent dashboard (always when --plot) ─────────────────
        if args.plot:
            from visualization.plots import plot_all
            # In compare mode use agent-specific filename; single mode → dashboard.png
            dash_name = (
                f"dashboard_{run_label.lower().replace(' ', '-')}.png"
                if args.compare else "dashboard.png"
            )
            plot_all(
                sim.results,
                our_agent=our_agent,
                products=list(PRODUCTS),
                save_path=str(_rp(run_dir, dash_name.removesuffix(".png"), ".png")),
                show=False,
            )

        # ── structured summary from sim ──────────────────────────────
        summary_path = _rp(run_dir, f"summary_{run_label.lower().replace(' ', '_')}", ".txt")
        write_summary(
            path=summary_path,
            ts=ts,
            run_label=run_label,
            sim=sim,
            args=args,
            alpha=alpha,
            d_max=d_max,
            init_inv=init_inv,
            products=list(PRODUCTS),
        )

    # ── regret plot + convergence (--plot with --compare) ─────────────
    if args.plot and args.compare:
        from visualization.regret import (
            compare_and_plot_regret,
            compute_oracle_cost, compute_regret, plot_convergence_dashboard,
        )

        if len(all_run_results) == 2:
            _lbl_as, results_as, _agent_as = all_run_results[0]
            _lbl_pb, results_pb, _agent_pb = all_run_results[1]
        else:
            _lbl_as, results_as, _agent_as = all_run_results[0]
            _lbl_pb, results_pb, _agent_pb = all_run_results[0]

        # Regret comparison plot
        compare_and_plot_regret(
            results_all_seeing=results_as,
            results_partially_blind=results_pb,
            alpha=alpha,
            initial_inventory=init_inv,
            products=list(PRODUCTS),
            V=args.V,
            save_path=str(_rp(run_dir, "regret", ".png")),
            show=False,
        )

        # 4-panel convergence dashboard
        conv_runs = []
        for _lbl, _res, _agent in all_run_results:
            _oracle = compute_oracle_cost(_res, alpha, init_inv)
            _regret, _ = compute_regret(_res, _oracle, alpha)
            conv_runs.append((_lbl, _res, _regret))

        plot_convergence_dashboard(
            runs=conv_runs,
            alpha=alpha,
            V=args.V,
            n_products=N_PRODUCTS,
            n_competitors=len(competitors),
            context_dim=N_PRODUCTS,
            save_path=str(_rp(run_dir, "convergence", ".png")),
            show=False,
        )

    elif args.plot and not args.compare:
        # Single-mode convergence (no regret comparison, but still show conv)
        from visualization.regret import (
            compute_oracle_cost, compute_regret, plot_convergence_dashboard,
        )
        conv_runs = []
        for _lbl, _res, _agent in all_run_results:
            _oracle = compute_oracle_cost(_res, alpha, init_inv)
            _regret, _ = compute_regret(_res, _oracle, alpha)
            conv_runs.append((_lbl, _res, _regret))

        plot_convergence_dashboard(
            runs=conv_runs,
            alpha=alpha,
            V=args.V,
            n_products=N_PRODUCTS,
            n_competitors=len(competitors),
            context_dim=N_PRODUCTS,
            save_path=str(_rp(run_dir, "convergence", ".png")),
            show=False,
        )

    print(f"  [run] All results saved to {run_dir}/")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)