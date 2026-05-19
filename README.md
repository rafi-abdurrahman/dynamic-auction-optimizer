# Inventory-Constrained Dynamic Bidding via Lyapunov Optimization

> **202E Term Project** — Online auction simulation with inventory constraints,
> solved via Lyapunov drift-plus-penalty optimization and bandit learning.

---

## Setup

```bash
pip install -r requirements.txt
```

EUROSTAT data is optional — the simulation falls back to synthetic data automatically if the raw TSV files are absent.

---

## How to Run

### Interactive dashboard (recommended)

```bash
streamlit run app.py
```

Opens a web UI where you can switch agents, tune hyperparameters, and inspect results interactively.

### CLI

```bash
# All-Seeing agent, 150 rounds
python -m scripts.run --mode all_seeing --n_rounds 150

# Partially-Blind agent
python -m scripts.run --mode partially_blind --n_rounds 150

# Save per-agent dashboard PNGs
python -m scripts.run --mode all_seeing --n_rounds 150 --plot

# Verbose per-round output
python -m scripts.run --mode all_seeing --n_rounds 50 --verbose

# With real EUROSTAT data
python -m scripts.run --mode all_seeing --use_real_data --country DE --n_rounds 150
```

Results are saved automatically to `results/{timestamp}/`.

### Full flag reference

```
--mode          all_seeing | partially_blind | ilp_theoretical | ilp_static_plan
--compare       run both agents, produce shared regret plot
--n_rounds INT  auction rounds (default: 150)
--V      FLOAT  Lyapunov trade-off (default: 2.0)
--a      FLOAT  safety-stock scale, alpha = a × mean_qty (default: 0.5)
--d_rate FLOAT  depletion rate, d_max = d_rate × alpha (default: 0.5)
--plot          save visualisations to results/
--verbose       print per-round table
--seed   INT    random seed (default: 42)
--competitors   comma-sep: stochastic|linear|naive (default: all three)
```

---

## Overview

At each round, a set of perishable agricultural products (milk, eggs, poultry, beef) are put up for sealed-bid auction. Our agent decides whether to bid for each product, subject to a hard inventory minimum α and a cost objective.

Two strategies are compared:

| Agent | Information | Algorithm |
|---|---|---|
| **All-Seeing Bandit** | Observes all competitor bids | Lyapunov drift-plus-penalty |
| **Partially-Blind Bandit** | Observes only competitor inventories | Lyapunov + incremental OLS |

Regret is measured against a clairvoyant offline oracle (LP relaxation of the offline ILP) that knows all future prices and quantities in advance.

---

## Project Structure

```
├── agents/
│   ├── base.py                  # AuctionState, BaseAgent, Lyapunov deficit
│   ├── ours/
│   │   ├── bandit.py            # AllSeeingBanditAgent, PartiallyBlindBanditAgent
│   │   └── ilp_solver.py        # offline ILP solver + ILPOracleAgent
│   └── competitors/
│       └── competitors.py       # Stochastic, Linear, NaiveThreshold competitors
├── env/
│   └── sim.py                   # AuctionSimulation — round lifecycle, RoundResult
├── data/
│   ├── preprocessor.py          # EUROSTAT ingestion & scaling
│   └── raw/                     # EUROSTAT TSV files (optional)
├── datagen/
│   └── generate_data.py         # AuctionDataGenerator — stochastic lot sizes
├── tools/
│   └── helper_run.py            # build_competitors, derive_data_stats
├── visualization/
│   ├── plots.py                 # inventory, cost, win rate, Lyapunov deficit charts
│   └── regret.py                # oracle computation, penalised regret, regret plot
├── scripts/
│   └── run.py                   # CLI entry point
├── app.py                       # Streamlit dashboard
└── results/                     # auto-created; all outputs land here
```

---

## Hyperparameters

```
alpha  =  a      × mean_qty    (hard inventory safety-stock threshold)
d_max  =  d_rate × alpha       (per-auction depletion cap)
```

| Parameter | Flag | Default | Meaning |
|---|---|---|---|
| `a` | `--a` | `0.5` | Scales α relative to typical lot size. Smaller → looser constraint. |
| `d_rate` | `--d_rate` | `0.5` | d_max as fraction of α. `d_rate=0.1` → ~10 rounds of runway; `d_rate=1.0` → 1 round. |
| `V` | `--V` | `2.0` | Lyapunov trade-off. Higher → more cost-conscious, fewer bids. |

---

## Algorithm

### Lyapunov Drift-Plus-Penalty (All-Seeing)

Define the deficit $H_i(t) = \beta_i - s_i(t)$ where $\beta_i = \alpha_i + d_{\max,i}$.
Bid for product $i$ iff:

$$H_i(t) \cdot q_i(t) > V \cdot b_i(t)$$

### Incremental OLS (Partially-Blind)

Estimates market prices from a linear model of competitor inventories, updated online via incremental OLS in $O(d^2)$ per round.

### Penalised Regret

$$R(T) = \sum_{t=1}^{T} \left[ c_{\text{agent}}(t) + \lambda \cdot \sum_i \max(0,\, \alpha_i - s_i(t)) \right] - c_{\text{oracle}}(T)$$

$\lambda$ = max observed market price, so violations are never cheaper than purchasing.

---

## Data

Place EUROSTAT bulk-download TSV files in `data/raw/`:

| File | Contents |
|---|---|
| `apro_mk_cola.tsv` | Milk collection |
| `apro_ec_poula.tsv` | Eggs and poultry production |
| `apro_mt_pann.tsv` | Meat production (beef) |

Source: [EUROSTAT bulk download facility](https://ec.europa.eu/eurostat/web/main/data/database).

---

## Output Files

Each run creates a timestamped folder under `results/`:

| File | Contents |
|---|---|
| `config.json` | All CLI arguments |
| `summary_{mode}.txt` | Cost, violations, win rates |
| `dashboard_{mode}.png` | Inventory, cost, win rate, Lyapunov deficit charts |
| `regret.png` | Average regret plot (with `--compare` or `--plot`) |
