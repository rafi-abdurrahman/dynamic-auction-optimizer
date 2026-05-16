"""
Streamlit dashboard for the Inventory-Constrained Dynamic Bidding simulation.

Launch with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ── project root on sys.path ────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.base import N_PRODUCTS, PRODUCTS
from agents.ours.bandit import AllSeeingBanditAgent, PartiallyBlindBanditAgent
from agents.ours.ilp_solver import ILPOracleAgent, solve_ilp_offline
from env.sim import AuctionSimulation
from tools.helper_run import build_competitors, derive_data_stats
from visualization.regret import compute_oracle_cost, compute_regret, plot_convergence_dashboard

# ── constants ───────────────────────────────────────────────────────────
SEED = 42
PRODUCT_COLORS = {
    "milk_dairy": "#4C72B0",
    "eggs":       "#DD8452",
    "poultry":    "#55A868",
    "beef":       "#C44E52",
}
MODE_LABELS = {
    "All-Seeing":        "all_seeing",
    "Partially-Blind":   "partially_blind",
    "ILP (Theoretical)": "ilp_theoretical",
    "ILP (Static Plan)": "ilp_static_plan",
}


# ════════════════════════════════════════════════════════════════════════
#  Simulation Runner
# ════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Running simulation...")
def run_simulation(
    mode: str,
    n_rounds: int,
    V: float,
    a: float,
    d_rate: float,
    country: str,
    competitor_spec: str,
) -> dict:
    """Run the auction simulation and return all data needed by the dashboard.

    Parameters
    ----------
    mode : str
        One of ``"all_seeing"``, ``"partially_blind"``, ``"ilp_theoretical"``,
        ``"ilp_static_plan"``.
    n_rounds, V, a, d_rate : float
        Hyperparameters.
    country : str
        ISO-2 EUROSTAT country code.
    competitor_spec : str
        Comma-separated competitor types.

    Returns
    -------
    dict with keys:
        results    : list[RoundResult]
        summary    : dict from sim.summary()
        agent      : BaseAgent instance
        alpha      : np.ndarray (N,)
        d_max      : np.ndarray (N,)
        init_inv   : np.ndarray (N,)
        x_opt      : np.ndarray (T,N) or None  (ILP only)
        ilp_cost   : float or None              (ILP only)
        feasibility: list[bool] or None         (ILP only)
    """
    np.random.seed(SEED)

    # ── derive parameters ─────────────────────────────────────────────
    _FALLBACK_ALPHA = np.array([10.0, 5.0, 10.0, 15.0])
    _FALLBACK_DMAX = np.array([3.0, 2.0, 3.0, 4.0])

    data_mean_qty = None
    try:
        data_mean_qty, data_d_max, data_alpha = derive_data_stats(
            country, auctions_per_day=1, a=a, d_rate=d_rate,
        )
    except Exception:
        data_d_max = _FALLBACK_DMAX
        data_alpha = _FALLBACK_ALPHA

    d_max = data_d_max
    alpha = data_alpha
    init_inv = alpha + d_max  # = beta, so H_i(0) = 0

    # ── competitors ───────────────────────────────────────────────────
    competitors = build_competitors(competitor_spec, alpha, init_inv, SEED)
    competitor_ids = [c.agent_id for c in competitors]

    # ── simulation builder ────────────────────────────────────────────
    def _build_sim(agent):
        _competitors = build_competitors(competitor_spec, alpha, init_inv, SEED)
        _default_qty = (
            data_mean_qty if data_mean_qty is not None
            else d_max / max(a, 1e-6)
        )
        return AuctionSimulation(
            our_agent=agent,
            competitors=_competitors,
            n_products=N_PRODUCTS,
            default_quantities=_default_qty,
            default_depletions=d_max * 0.5,
            seed=SEED,
        )

    # ── ILP pre-roll (shared by both ILP modes) ────────────────────────
    x_opt_out = None
    ilp_cost_out = None
    feasibility_out = None
    prices_mat = None
    qtys_mat = None
    deps_mat = None

    is_ilp = mode in ("ilp_theoretical", "ilp_static_plan")

    if is_ilp:
        from agents.base import BaseAgent as _BaseAgent

        class _NullAgent(_BaseAgent):
            def bid(self, state):
                return np.zeros(self.n_products)

        null_agent = _NullAgent(agent_id=0, alpha=alpha,
                                initial_inventory=init_inv.copy())
        preroll_sim = _build_sim(null_agent)
        preroll_results = preroll_sim.run(n_rounds)

        prices_mat = np.array([
            [r.winning_bids[i] for i in range(N_PRODUCTS)]
            for r in preroll_results
        ])
        qtys_mat = np.array([r.quantities for r in preroll_results])
        deps_mat = np.array([r.depletions for r in preroll_results])
        prices_mat = np.where(prices_mat <= 0, 1e-6, prices_mat)

        x_opt, ilp_cost, feasibility = solve_ilp_offline(
            prices=prices_mat, quantities=qtys_mat, depletions=deps_mat,
            alpha=alpha, initial_inventory=init_inv,
        )
        x_opt_out = x_opt
        ilp_cost_out = ilp_cost
        feasibility_out = feasibility

    # ── ILP Theoretical: analytical computation, no simulation ────────
    if mode == "ilp_theoretical":
        # Compute inventory trajectory analytically.
        inv_trajectory = np.zeros((n_rounds + 1, N_PRODUCTS))
        inv_trajectory[0] = init_inv.copy()
        per_round_cost = np.zeros(n_rounds)

        for t in range(n_rounds):
            purchased = qtys_mat[t] * x_opt[t]  # q_i(t) * x_i(t)
            per_round_cost[t] = float(prices_mat[t] @ x_opt[t])
            inv_trajectory[t + 1] = inv_trajectory[t] + purchased - deps_mat[t]

        # Build a synthetic summary dict matching sim.summary() keys.
        total_cost = float(ilp_cost)
        summary = {
            "total_cost": total_cost,
            "avg_cost_per_round": total_cost / n_rounds,
            "constraint_violations": 0,  # guaranteed by solver
            "violation_rate": 0.0,
            "final_inventory": inv_trajectory[-1].tolist(),
            "win_rate_per_product": [
                float(x_opt[:, i].sum()) / max(1, int((x_opt[:, i] > -1).sum()))
                for i in range(N_PRODUCTS)
            ],
        }

        # Build synthetic RoundResult-like dicts for chart compatibility.
        from types import SimpleNamespace
        results = []
        for t in range(n_rounds):
            results.append(SimpleNamespace(
                t=t,
                quantities=qtys_mat[t],
                depletions=deps_mat[t],
                all_bids={0: prices_mat[t] * x_opt[t]},  # our "bid" = price if buying
                winners=np.where(x_opt[t] == 1, 0, -1).astype(int),
                winning_bids=prices_mat[t] * x_opt[t],
                our_cost=per_round_cost[t],
                our_won=x_opt[t].astype(bool),
                inventories_after={0: inv_trajectory[t + 1].copy()},
            ))

        return {
            "results": results,
            "summary": summary,
            "agent": None,
            "alpha": alpha,
            "d_max": d_max,
            "init_inv": init_inv,
            "x_opt": x_opt_out,
            "ilp_cost": ilp_cost_out,
            "feasibility": feasibility_out,
        }

    # ── ILP Static Plan: build agent and run through simulation ───────
    if mode == "ilp_static_plan":
        agent = ILPOracleAgent(
            agent_id=0, alpha=alpha, initial_inventory=init_inv.copy(),
            x_opt=x_opt, prices=prices_mat,
            ilp_total_cost=ilp_cost, feasibility=feasibility,
        )
    elif mode == "all_seeing":
        agent = AllSeeingBanditAgent(
            agent_id=0, alpha=alpha, initial_inventory=init_inv.copy(),
            V=V, d_max=d_max,
        )
    else:
        agent = PartiallyBlindBanditAgent(
            agent_id=0, alpha=alpha, initial_inventory=init_inv.copy(),
            V=V, d_max=d_max, competitor_ids=competitor_ids,
        )

    sim = _build_sim(agent)
    results = sim.run(n_rounds)
    summary = sim.summary()

    return {
        "results": results,
        "summary": summary,
        "agent": agent,
        "alpha": alpha,
        "d_max": d_max,
        "init_inv": init_inv,
        "x_opt": x_opt_out,
        "ilp_cost": ilp_cost_out,
        "feasibility": feasibility_out,
    }


# ════════════════════════════════════════════════════════════════════════
#  Chart Builders
# ════════════════════════════════════════════════════════════════════════

def _plotly_template() -> dict:
    """Shared Plotly layout defaults."""
    return dict(
        template="plotly_dark",
        margin=dict(l=50, r=30, t=50, b=40),
        font=dict(family="Inter, sans-serif", size=12),
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
    )


def chart_inventory(results: list, alpha: np.ndarray) -> go.Figure:
    """Inventory over time with alpha constraint lines."""
    fig = make_subplots(
        rows=len(PRODUCTS), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=list(PRODUCTS)
    )
    T = len(results)
    t_ax = list(range(T))

    for i, prod in enumerate(PRODUCTS):
        inv = [r.inventories_after[0][i] for r in results]
        color = PRODUCT_COLORS[prod]
        fig.add_trace(go.Scatter(
            x=t_ax, y=inv, name=prod, mode="lines",
            line=dict(color=color, width=2),
            showlegend=False
        ), row=i+1, col=1)
        
        fig.add_hline(
            y=float(alpha[i]), line_dash="dash",
            line_color=color, opacity=0.5,
            annotation_text=f"alpha ({prod})",
            annotation_position="bottom right",
            annotation_font_color=color,
            annotation_font_size=9,
            row=i+1, col=1
        )

    fig.update_layout(
        title="Inventory Over Time",
        height=650,
        **_plotly_template(),
    )
    fig.update_xaxes(title_text="Round", row=len(PRODUCTS), col=1)
    return fig


def chart_cumulative_cost(results: list) -> go.Figure:
    """Cumulative and per-round cost."""
    T = len(results)
    t_ax = list(range(T))
    per_round = [r.our_cost for r in results]
    cum_cost = np.cumsum(per_round).tolist()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=t_ax, y=cum_cost, name="Cumulative",
            mode="lines", line=dict(color="#4C72B0", width=2.5),
            fill="tozeroy", fillcolor="rgba(76,114,176,0.1)",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=t_ax, y=per_round, name="Per-round",
            marker_color="#DD8452", opacity=0.45,
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="Cost Over Time",
        xaxis_title="Round",
        **_plotly_template(),
    )
    fig.update_yaxes(title_text="Cumulative cost", secondary_y=False)
    fig.update_yaxes(title_text="Per-round cost", secondary_y=True)
    return fig


def chart_win_rate(results: list) -> go.Figure:
    """Win rate per product bar chart."""
    win_counts = np.zeros(N_PRODUCTS)
    bid_counts = np.zeros(N_PRODUCTS)
    for r in results:
        win_counts += r.our_won.astype(float)
        bid_counts += (r.all_bids[0] > 0).astype(float)

    safe = np.where(bid_counts > 0, bid_counts, 1.0)
    rates = np.where(bid_counts > 0, win_counts / safe * 100, 0.0)

    colors = [PRODUCT_COLORS[p] for p in PRODUCTS]

    fig = go.Figure(go.Bar(
        x=list(PRODUCTS), y=rates.tolist(),
        marker_color=colors,
        text=[f"{r:.1f}%" for r in rates],
        textposition="outside",
    ))
    fig.update_layout(
        title="Win Rate per Product",
        yaxis_title="Win Rate (%)",
        yaxis_range=[0, 115],
        **_plotly_template(),
    )
    return fig


def chart_lyapunov_deficit(agent) -> go.Figure | None:
    """Lyapunov deficit H_i(t) over time."""
    if not agent.history or "H" not in agent.history[0]:
        return None

    fig = go.Figure()
    for i, prod in enumerate(PRODUCTS):
        H = [entry["H"][i] for entry in agent.history]
        t_ax = [entry["t"] for entry in agent.history]
        fig.add_trace(go.Scatter(
            x=t_ax, y=H, name=prod, mode="lines",
            line=dict(color=PRODUCT_COLORS[prod], width=2),
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
    fig.update_layout(
        title="Lyapunov Deficit H_i(t) = beta_i - s_i(t)",
        xaxis_title="Round",
        yaxis_title="Deficit",
        **_plotly_template(),
    )
    return fig


def chart_bid_vs_market_price(results: list) -> go.Figure:
    """For each product, plot our bid amount vs the winning market price."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=PRODUCTS,
        shared_xaxes=True,
        vertical_spacing=0.15
    )
    
    for i, prod in enumerate(PRODUCTS):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        t_ax, our_bids, market = [], [], []
        for r in results:
            ob = float(r.all_bids[0][i])
            mp = float(r.winning_bids[i])
            if ob > 0:
                t_ax.append(r.t)
                our_bids.append(ob)
                market.append(mp)
                
        if t_ax:
            # Market price line
            fig.add_trace(go.Scatter(
                x=t_ax, y=market, name="Market price", 
                mode="lines", line=dict(color="#55A868", width=1.5),
                legendgroup="market", showlegend=(i==0)
            ), row=row, col=col)
            
            # Our bid scatter
            fig.add_trace(go.Scatter(
                x=t_ax, y=our_bids, name="Our bid", 
                mode="markers", marker=dict(color="#4C72B0", size=6),
                legendgroup="our", showlegend=(i==0)
            ), row=row, col=col)
            
    fig.update_layout(
        title="Our Bid vs Market Price (when bidding > 0)",
        height=500,
        **_plotly_template()
    )
    fig.update_xaxes(title_text="Round", row=2, col=1)
    fig.update_xaxes(title_text="Round", row=2, col=2)
    return fig


def chart_round_bids(results: list, t: int) -> go.Figure:
    """Grouped bar chart comparing all bidders at round t."""
    r = results[t]
    agent_ids = sorted(r.all_bids.keys())

    agent_labels = {}
    for aid in agent_ids:
        if aid == 0:
            agent_labels[aid] = "Our Agent"
        else:
            agent_labels[aid] = f"Competitor {aid}"

    fig = go.Figure()
    for aid in agent_ids:
        bids = r.all_bids[aid]
        fig.add_trace(go.Bar(
            name=agent_labels[aid],
            x=list(PRODUCTS),
            y=bids.tolist(),
        ))

    fig.update_layout(
        barmode="group",
        title=f"All Bids at Round {t}",
        xaxis_title="Product",
        yaxis_title="Bid Amount",
        **_plotly_template(),
    )
    return fig


# ════════════════════════════════════════════════════════════════════════
#  Streamlit App
# ════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.set_page_config(
        page_title="Auction Optimizer",
        page_icon="📊",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .metric-card {
            background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
            border: 1px solid #3d3d5c;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }
        .metric-card h3 {
            margin: 0;
            color: #a0a0b8;
            font-size: 0.85rem;
            font-weight: 500;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .metric-card p {
            margin: 8px 0 0 0;
            color: #e0e0f0;
            font-size: 1.7rem;
            font-weight: 700;
        }
        .metric-card .sub {
            color: #7a7a9a;
            font-size: 0.8rem;
            margin-top: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── sidebar ───────────────────────────────────────────────────────
    with st.sidebar:
        st.title("Simulation Parameters")

        mode_label = st.selectbox(
            "Agent Mode",
            options=list(MODE_LABELS.keys()),
            index=0,
        )
        mode = MODE_LABELS[mode_label]

        st.divider()
        st.subheader("Hyperparameters")

        n_rounds = st.slider("Rounds (T)", min_value=10, max_value=500,
                             value=150, step=10)

        if not mode.startswith("ilp"):
            V = st.slider("V (cost vs. safety trade-off)",
                          min_value=0.1, max_value=10.0, value=2.0, step=0.1,
                          help="Higher V = more cost-conscious, lower V = more inventory-safe.")
        else:
            V = 2.0  # unused for ILP modes

        a = st.slider("a (safety-stock scale)",
                      min_value=0.1, max_value=1.0, value=0.5, step=0.05,
                      help="alpha = a * mean_qty. Lower = looser constraint.")

        d_rate = st.slider("d_rate (depletion rate)",
                           min_value=0.1, max_value=1.0, value=0.5, step=0.05,
                           help="d_max = d_rate * alpha. Lower = more rounds of runway.")

        country = "DE"  # fixed to EUROSTAT Germany dataset

        competitor_opts = st.multiselect(
            "Competitors",
            options=["stochastic", "linear", "naive"],
            default=["stochastic", "linear", "naive"],
        )
        competitor_spec = ",".join(competitor_opts) if competitor_opts else "stochastic"

        st.divider()
        run_btn = st.button("Run Simulation", type="primary", width="stretch")

    # ── run simulation ────────────────────────────────────────────────
    if "sim_data" not in st.session_state:
        st.session_state.sim_data = None

    if run_btn:
        st.session_state.sim_data = run_simulation(
            mode=mode, n_rounds=n_rounds, V=V,
            a=a, d_rate=d_rate, country=country,
            competitor_spec=competitor_spec,
        )

    data = st.session_state.sim_data

    # ── title ─────────────────────────────────────────────────────────
    st.title("Inventory-Constrained Dynamic Bidding")
    st.caption("Simulation dashboard for Lyapunov-optimized auction strategies")

    if data is None:
        st.info("Configure parameters in the sidebar and click **Run Simulation** to begin.")
        return

    results = data["results"]
    summary = data["summary"]
    agent = data["agent"]
    alpha = data["alpha"]

    # ── tabs ──────────────────────────────────────────────────────────────────
    tab_names = ["Results Overview", "Round Inspector", "Convergence Analysis"]
    is_ilp = mode.startswith("ilp")

    tabs = st.tabs(tab_names)

    # ── Tab 1: Results Overview ───────────────────────────────────────
    with tabs[0]:
        # Metric cards
        col1, col2, col3, col4 = st.columns(4)

        # For ILP mode, the primary cost is the solver's analytical output
        # (the theoretical lower bound), not the static plan simulation cost.
        ilp_cost = data.get("ilp_cost")
        is_ilp = ilp_cost is not None

        with col1:
            if is_ilp:
                st.markdown(
                    f"""<div class="metric-card">
                    <h3>ILP Optimal Cost</h3>
                    <p>{ilp_cost:,.2f}</p>
                    <div class="sub">Simulated cost: {summary['total_cost']:,.2f}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""<div class="metric-card">
                    <h3>Total Cost</h3>
                    <p>{summary['total_cost']:,.2f}</p>
                    </div>""",
                    unsafe_allow_html=True,
                )

        with col2:
            if is_ilp:
                avg_ilp = ilp_cost / len(results)
                st.markdown(
                    f"""<div class="metric-card">
                    <h3>Avg Optimal / Round</h3>
                    <p>{avg_ilp:,.2f}</p>
                    <div class="sub">Simulated avg: {summary['avg_cost_per_round']:,.2f}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""<div class="metric-card">
                    <h3>Avg Cost / Round</h3>
                    <p>{summary['avg_cost_per_round']:,.2f}</p>
                    </div>""",
                    unsafe_allow_html=True,
                )

        with col3:
            violations = summary["constraint_violations"]
            viol_rate = summary["violation_rate"] * 100
            color = "#55A868" if violations == 0 else "#C44E52"
            sub_text = f"{viol_rate:.1f}% of rounds"
            if is_ilp and violations > 0:
                sub_text += " (simulation artifact)"
            st.markdown(
                f"""<div class="metric-card">
                <h3>Violations</h3>
                <p style="color: {color}">{violations}</p>
                <div class="sub">{sub_text}</div>
                </div>""",
                unsafe_allow_html=True,
            )

        with col4:
            final_inv = summary["final_inventory"]
            inv_str = ", ".join(f"{v:.0f}" for v in final_inv)
            st.markdown(
                f"""<div class="metric-card">
                <h3>Final Inventory</h3>
                <p style="font-size:1.1rem">{inv_str}</p>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Charts row 1
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(chart_inventory(results, alpha),
                           width="stretch", key="overview_inventory")
        with c2:
            st.plotly_chart(chart_cumulative_cost(results),
                           width="stretch", key="overview_cost")

        # Charts row 2
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(chart_win_rate(results),
                           width="stretch", key="overview_winrate")
        with c4:
            deficit_fig = chart_lyapunov_deficit(agent) if agent is not None else None
            if deficit_fig is not None:
                st.plotly_chart(deficit_fig, width="stretch", key="overview_deficit")
            else:
                st.info("Lyapunov deficit is not applicable for ILP mode.")

        # Charts row 3
        st.markdown("---")
        st.plotly_chart(chart_bid_vs_market_price(results), width="stretch", key="overview_bids_vs_market")

    # ── Tab 2: Round Inspector (all modes) ────────────────────────────
    with tabs[1]:
        T = len(results)
        selected_round = st.slider(
            "Select round", min_value=0, max_value=T - 1, value=0,
        )
        r = results[selected_round]

        # Build auction table
        rows = []
        agent_ids = sorted(r.all_bids.keys())
        for i, prod in enumerate(PRODUCTS):
            row = {
                "Product": prod,
                "Quantity": f"{r.quantities[i]:.1f}",
                "Our Bid": f"{r.all_bids[0][i]:.2f}",
            }
            for aid in agent_ids:
                if aid != 0:
                    row[f"Comp. {aid}"] = f"{r.all_bids[aid][i]:.2f}"

            winner_id = int(r.winners[i])
            if winner_id == 0:
                row["Winner"] = "Us"
            elif winner_id == -1:
                row["Winner"] = "None"
            else:
                row["Winner"] = f"Comp. {winner_id}"
            row["Winning Bid"] = f"{r.winning_bids[i]:.2f}"
            rows.append(row)

        df = pd.DataFrame(rows)

        def _highlight_wins(row):
            """Highlight rows where our agent won."""
            if row["Winner"] == "Us":
                return ["background-color: rgba(85,168,104,0.2)"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df.style.apply(_highlight_wins, axis=1),
            width="stretch",
            hide_index=True,
            key="round_table",
        )

        # Bid comparison chart
        st.plotly_chart(
            chart_round_bids(results, selected_round),
            width="stretch", key="round_bids",
        )

        # Inventory snapshot after this round
        inv_after = r.inventories_after[0]
        fig_snap = go.Figure()
        fig_snap.add_trace(go.Bar(
            y=list(PRODUCTS),
            x=inv_after.tolist(),
            orientation="h",
            marker_color=[PRODUCT_COLORS[p] for p in PRODUCTS],
            text=[f"{v:.0f}" for v in inv_after],
            textposition="outside",
        ))
        # Draw elegant discrete threshold lines over each bar
        for i, prod in enumerate(PRODUCTS):
            fig_snap.add_shape(
                type="line",
                x0=float(alpha[i]), x1=float(alpha[i]),
                y0=i - 0.35, y1=i + 0.35,
                line=dict(color="rgba(255, 255, 255, 0.8)", width=2, dash="dot"),
            )
        fig_snap.update_layout(
            title=f"Inventory After Round {selected_round}",
            xaxis_title="Inventory",
            **_plotly_template(),
        )
        st.plotly_chart(fig_snap, width="stretch", key="round_inventory")

    # ── Tab 3: Convergence Analysis ───────────────────────────────────
    with tabs[2]:
        st.markdown("### Convergence & Regret Analysis")
        st.markdown("This dashboard compares the agent's performance against the clairvoyant ILP oracle over time.")
        
        if mode == "ilp_theoretical":
            st.info("Convergence analysis is not applicable for the theoretical ILP baseline since it is the oracle itself.")
        else:
            with st.spinner("Computing clairvoyant oracle baseline (LP relaxation)..."):
                try:
                    oracle_costs = compute_oracle_cost(results, alpha, data["init_inv"])
                    # Use the same shared penalty logic as run.py
                    all_prices = [r.winning_bids for r in results]
                    violation_penalty = float(np.max(all_prices)) if all_prices else 1.0
                    violation_penalty = max(violation_penalty, 1.0)
                    
                    cum_regret, _ = compute_regret(
                        results, oracle_costs, alpha, violation_penalty=violation_penalty
                    )
                    
                    # Plotly doesn't natively support this dashboard, but we can render the matplotlib one
                    fig_conv = plot_convergence_dashboard(
                        runs=[(mode, results, cum_regret)],
                        alpha=alpha, V=V if not is_ilp else 2.0,
                        n_products=len(alpha),
                        n_competitors=len(competitor_opts),
                        context_dim=len(alpha),
                        show=False
                    )
                    st.pyplot(fig_conv)
                except Exception as e:
                    st.error(f"Error generating convergence dashboard: {e}")



if __name__ == "__main__":
    main()
