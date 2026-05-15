# Inventory-Constrained Dynamic Bidding via Lyapunov Optimization

> **202E Term Project** — Online auction simulation with inventory constraints,
> solved via Lyapunov drift-plus-penalty optimization and bandit learning.

---

## Overview

This project implements and compares two online bidding agents for a
multi-product inventory-replenishment auction. At each round *t*, a set of
perishable agricultural products (milk, eggs, poultry, beef) are put up for
sealed-bid auction. Our agent decides whether to bid for each product, subject
to:

- A **hard inventory minimum** α (must never let stock drop below this)
- A **cost objective** (minimize total spend over the horizon)

Two agent strategies are studied:

| Agent | Information | Algorithm |
|---|---|---|
| **All-Seeing Bandit** (§IV-B) | Observes all competitor bids | Lyapunov drift-plus-penalty |
| **Partially-Blind Bandit** (§IV-C) | Observes only competitor inventories | Lyapunov + incremental OLS |

Regret is measured against a **clairvoyant offline oracle** (LP relaxation of
the §IV-A ILP) that knows all future prices and quantities in advance.

---

## Project Structure

```
term-project/
│
├── agents/                     # Agent definitions
│   ├── base.py                 # AuctionState, BaseAgent ABC, Lyapunov deficit
│   ├── ours/
│   │   └── bandit.py           # AllSeeingBanditAgent, PartiallyBlindBanditAgent
│   └── competitors/
│       └── competitors.py      # Stochastic, Linear, NaiveThreshold competitors
│
├── env/
│   └── sim.py                  # AuctionSimulation — round lifecycle, RoundResult
│
├── data/
│   ├── preprocessor.py         # load_daily — EUROSTAT ingestion & scaling
│   └── raw/                    # ← EUROSTAT TSV files (see Data section)
│
├── datagen/
│   └── generate_data.py        # AuctionDataGenerator — stochastic lot sizes
│
├── tools/
│   └── helper_run.py           # build_competitors, derive_data_stats,
│                               #   quantity_iter_from_generator, depletion_iter
│
├── visualization/
│   ├── plots.py                # Dashboard: inventory, cost, bid vs price,
│   │                           #   win rate, Lyapunov deficit
│   └── regret.py               # LP oracle, penalised regret, comparison plot
│
├── scripts/
│   └── run.py                  # CLI entry point — game loop, result saving
│
└── results/                    # Auto-created; all outputs land here
    ├── {ts}_config.json
    ├── {ts}_summary_{mode}.txt
    ├── {ts}_dashboard.png
    └── {ts}_regret.png
```

---

## Hyperparameters

The simulation is controlled by three key hyperparameters derived from data:

```
mean_qty = mean daily EUROSTAT production × auction_fraction ÷ auctions_per_day

alpha    =  a      × mean_qty       (hard inventory safety-stock threshold)
d_max    =  d_rate × alpha          (per-auction depletion cap)
```

| Parameter | Flag | Default | Meaning |
|---|---|---|---|
| `a` | `--a` | `0.5` | Scales α relative to the typical lot size. Smaller → looser constraint. |
| `d_rate` | `--d_rate` | `0.5` | Sets d_max as a fraction of α. Controls how many rounds of runway exist before hitting α: `d_rate=0.1` → ~10 rounds; `d_rate=1.0` → 1 round. |
| `V` | `--V` | `2.0` | Lyapunov trade-off. Higher → more cost-conscious, fewer bids. Lower → more aggressive bidding to protect inventory. |

**Typical safe regime**: `a=0.2–0.5`, `d_rate=0.3–0.6`, `V=1.0–5.0`.

---

## Algorithm

### Lyapunov Drift-Plus-Penalty (§IV-B — All-Seeing)

Define the **deficit** for product *i* at time *t*:

$$H_i(t) = \beta_i - s_i(t), \qquad \beta_i = \alpha_i + d_{\max,i}$$

The agent bids for product *i* if and only if:

$$H_i(t) \cdot q_i(t) > V \cdot b_i(t)$$

where $q_i(t)$ is the lot size, $b_i(t)$ is the market price, and $V \geq 0$
controls the cost–safety trade-off.

### Incremental OLS (§IV-C — Partially-Blind)

The Partially-Blind agent cannot observe competitor bids. It estimates market
prices from a linear model of competitor inventories, updated online via
incremental OLS in $O(d^2)$ per round.

### Penalised Regret

$$R(T) = \sum_{t=1}^{T} \!\left[\, c_{\text{agent}}(t) + \lambda \cdot \textstyle\sum_i \max\!\left(0,\, \alpha_i - s_i(t)\right)\right] - c_{\text{oracle}}(T)$$

The oracle solves the per-product LP relaxation of the offline ILP via
`scipy.optimize.linprog` (lower bound → regret is a conservative upper bound).
$\lambda$ = max market price observed, ensuring violations are never cheaper
than purchasing.

---

## Installation

```bash
pip install numpy pandas matplotlib scipy
```

EUROSTAT data ingestion also uses `pandas` (already included).

---

## Data

Place EUROSTAT bulk-download TSV files in `data/raw/`:

| File | Contents |
|---|---|
| `apro_mk_cola.tsv` | Milk collection (monthly, by country) |
| `apro_ec_poula.tsv` | Eggs and poultry production |
| `apro_mt_pann.tsv` | Meat production (beef) |

Source: [EUROSTAT bulk download facility](https://ec.europa.eu/eurostat/web/main/data/database).

If absent, the simulation falls back to synthetic data automatically using
the EUROSTAT-derived `mean_qty` as the default lot size.

---

## Usage

### Basic run

```bash
# All-Seeing, 150 rounds (results auto-saved to results/)
python -m scripts.run --mode all_seeing --n_rounds 150

# Tune hyperparameters
python -m scripts.run --mode all_seeing --n_rounds 150 --a 0.3 --d_rate 0.4 --V 3.0

# Per-round console output
python -m scripts.run --mode all_seeing --n_rounds 50 --verbose
```

### Compare both algorithms + regret plot

```bash
python -m scripts.run --compare --n_rounds 150 --a 0.3 --d_rate 0.4
```

Runs both agents on the **same** price/quantity sequence and saves a
penalised regret comparison plot.

### With real EUROSTAT data

```bash
python -m scripts.run --mode all_seeing --use_real_data --country DE --n_rounds 150
```

### Full flag reference

```
Scenario
  --mode          all_seeing | partially_blind      (default: all_seeing)
  --compare       Run both modes, produce regret plot
  --n_rounds INT  Number of auction rounds           (default: 150)

Hyperparameters
  --V      FLOAT  Lyapunov trade-off                 (default: 2.0)
  --a      FLOAT  Safety-stock scale, alpha=a×mean   (default: 0.5)
  --d_rate FLOAT  Depletion rate, d_max=d_rate×alpha (default: 0.5)

Manual overrides (skip data derivation)
  --alpha     FLOAT×4  Hard inventory minimum per product
  --d_max     FLOAT×4  Per-auction depletion cap per product
  --init_inventory FLOAT×4  Starting inventory (default: 2×alpha)

Competitors
  --competitors STR  Comma-sep: stochastic|linear|naive
                     (default: stochastic,linear,naive)

Data source
  --use_real_data       Use EUROSTAT AuctionDataGenerator
  --country       STR   ISO-2 code                   (default: DE)
  --auctions_per_day INT                              (default: 1)

Output
  --seed       INT   Random seed                        (default: 42)
  --verbose          Print per-round table to terminal
  --plot       PATH  Override dashboard save path
  --regret     PATH  Override regret plot save path
  --convergence      4-panel convergence dashboard (regret + violation curves
                     with theoretical O(√T) and O(N·Kmax·√(T·(d+log NM))) bounds)
```

---

## Output Files

Every run creates timestamped files in `results/` automatically:

| File | Contents |
|---|---|
| `{ts}_config.json` | All CLI arguments as JSON |
| `{ts}_summary_{mode}.txt` | Structured results from `sim.summary()` |
| `{ts}_dashboard.png` | 3×2 visualisation panel |
| `{ts}_regret.png` | Regret comparison (with `--compare`) |
| `{ts}_convergence.png` | 4-panel convergence dashboard (with `--convergence`) |

### Summary file structure

```
── Configuration ─────────────────────────────
  n_rounds, V, a, d_rate, competitors, seed

── Resolved Parameters ───────────────────────
  Product          alpha     d_max   init_inv
  milk_dairy      888.19    444.09   1776.38
  eggs          18732.01   9366.00  37464.02
  ...

── Simulation Results ────────────────────────
  total_cost, avg_cost/round, violations, final_inventory

── Win Rate per Product ──────────────────────
  milk_dairy :  94.7%  ...
```

### Dashboard panels

| Position | Plot |
|---|---|
| Top-left | Inventory over time with α reference lines |
| Top-centre | Cumulative cost + per-round cost (dual axis) |
| Top-right | Win rate per product |
| Bottom-left | Lyapunov deficit $H_i(t) = \beta_i - s_i(t)$ |
| Bottom-centre/right | Bid vs market price (milk, eggs) |

### Convergence Dashboard (`--convergence`)

Produced from `visualization.regret.plot_convergence_dashboard`, these 4 plots
directly correspond to the theoretical convergence conditions:

| Plot | Quantity | Good-policy condition |
|---|---|---|
| 1 | $\text{Reg}(t)$ cumulative regret | $= o(t)$ — sublinear growth |
| 2 | $\text{Reg}(t)/t$ average regret | $\to 0$ — no-regret |
| 3 | $V_{\text{viol}}(t) = \sum_{s \leq t}\sum_i \max(0, \alpha_i - s_i(s))$ | $= o(t)$ — sublinear violation |
| 4 | $V_{\text{viol}}(t)/t$ average violation | $\to 0$ — feasibility |

Reference bounds shown on Plots 1 & 2:

- **Black dashed** — $O(\sqrt{T})$ : All-Seeing Lyapunov bound (§5.3, with $V = \Theta(\sqrt{T})$)
- **Green dash-dot** — $O\!\left(NK_{\max}\sqrt{T(d+\log NM)}\right)$ : Partially-Blind OLS bound (§12, online OLS)
  where $N$ = products, $M$ = competitors, $d$ = context dimension (competitor inventory size)


## Per-Round Investigation

The game loop in `scripts/run.py` exposes the full `RoundResult` at every step.
Uncomment any of these inside the `for t in range(args.n_rounds)` loop:

```python
print(r.all_bids)            # every agent's bid vector  (n_agents, N)
print(r.winners)             # which agent won each product
print(r.winning_bids)        # at what price
print(r.quantities)          # lot sizes this round
print(our_agent.history[-1]) # our agent's internal Lyapunov log
```

---

## Design Notes

### Why separate `a` and `d_rate`?

Previously both α and d_max were set to the same value (`a × mean_qty`),
which conflates the safety-stock level with the depletion rate. Decoupling
them lets you independently control:

- **How strict the inventory constraint is** (`a` — higher means harder to satisfy)
- **How fast inventory drains** (`d_rate` — lower means more rounds of runway)

### Why LP relaxation for the oracle?

The per-product ILP is NP-hard in general. The LP relaxation is a lower bound
on the optimal integer cost → measured regret is a valid **upper bound**.

### Why penalise violations in regret?

Without penalties an agent that never bids (spending 0) trivially beats the
oracle. The penalty $\lambda \cdot \text{deficit}$ makes each missing unit of
inventory at least as costly as purchasing at the highest market price.

### Why a shared price sequence for `--compare`?

Both agents see the same competitor bids and lot sizes, isolating the regret
gap to **algorithmic difference only** (OLS estimation error vs full
information).

---

## Module Reference

### `env.sim.AuctionSimulation`
```python
sim = AuctionSimulation(our_agent, competitors, n_products,
                        default_quantities=mean_qty,
                        default_depletions=d_max * 0.5)
r   = sim.step()        # → RoundResult (one auction round)
sim.run(n_rounds)       # run full episode
sim.summary()           # → dict with cost, win_rate, violations, …
```

### `visualization.plots`
```python
plot_all(results, our_agent, save_path="results/dashboard.png")
```

### `visualization.regret`
```python
oracle = compute_oracle_cost(results, alpha, init_inv)
regret, lam = compute_regret(results, oracle, alpha)
compare_and_plot_regret(results_as, results_pb, alpha, init_inv,
                        save_path="results/regret.png")

# 4-panel convergence dashboard
plot_convergence_dashboard(
    runs=[("All-Seeing", results, cum_regret), ("Partially-Blind", results_pb, cum_regret_pb)],
    alpha=alpha, V=2.0,
    n_products=4, n_competitors=3, context_dim=4,
    save_path="results/convergence.png",
)
```

### `tools.helper_run`
```python
mean_qty, d_max, alpha = derive_data_stats(country, auctions_per_day,
                                            a=0.3, d_rate=0.4)
competitors = build_competitors("stochastic,linear,naive", alpha, init_inv, seed)
```
