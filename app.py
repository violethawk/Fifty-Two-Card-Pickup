"""
Streamlit web app for 52 Card Pickup.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import base64
import math
import random
import time
from typing import Dict, List, Tuple
from urllib.parse import urlencode

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

from card_pickup import (
    AppState,
    Card,
    SUITS,
    RANKS,
    TRAVEL_COST_PER_UNIT,
    _card_key,
    _determine_region,
    _extract_elapsed,
    _greedy_nearest_card,
    _make_initial_state,
    build_graph,
    scatter_node,
    verify_node,
)
from benchmarks import PATTERNS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUIT_COLORS = {
    "hearts": "#e74c3c",
    "diamonds": "#e67e22",
    "clubs": "#2c3e50",
    "spades": "#555555",
}

SUIT_MARKERS = {
    "hearts": "h",      # hexagon ~ heart shape
    "diamonds": "D",     # thin diamond
    "clubs": "p",        # pentagon ~ club
    "spades": "^",       # triangle ~ spade
}

SUIT_LABELS = {
    "hearts": "\u2665 Hearts",
    "diamonds": "\u2666 Diamonds",
    "clubs": "\u2663 Clubs",
    "spades": "\u2660 Spades",
}

AGENT_COLORS = ["#3498db", "#e74c3c", "#27ae60", "#9b59b6"]

# Dark-mode-aware colors — detect Streamlit theme
def _is_dark_mode() -> bool:
    """Best-effort dark mode detection via Streamlit theme config."""
    try:
        theme = st.get_option("theme.base")
        return theme == "dark"
    except Exception:
        return False

DARK_MODE = _is_dark_mode()

# Matplotlib style for dark/light mode
def _apply_plot_theme(fig, ax):
    """Apply dark or light theme to a matplotlib figure."""
    if DARK_MODE:
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="#fafafa")
        ax.xaxis.label.set_color("#fafafa")
        ax.yaxis.label.set_color("#fafafa")
        for spine in ax.spines.values():
            spine.set_color("#333333")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_grid(
    cards: List[Card],
    agent_positions: Dict[str, Tuple[float, float]] | None = None,
    title: str = "Card Grid",
    show_regions: int = 0,
    show_verifier: bool = True,
    trails: Dict[str, List[Tuple[float, float]]] | None = None,
    num_agents: int = 0,
    scoreboard: Dict[str, int] | None = None,
) -> plt.Figure:
    """Render the 10x10 grid with cards and optional agent positions."""
    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=80)
    _apply_plot_theme(fig, ax)

    title_color = "#fafafa" if DARK_MODE else "#000000"
    grid_color = "#444444" if DARK_MODE else "#cccccc"
    picked_color = "#555555" if DARK_MODE else "#cccccc"

    # Region shading
    if show_regions == 2:
        ax.axvspan(0, 5, alpha=0.05, color=AGENT_COLORS[0])
        ax.axvspan(5, 10, alpha=0.05, color=AGENT_COLORS[1])
        ax.axvline(5, color=grid_color, linestyle="--", linewidth=1)
    elif show_regions == 4:
        for i, (x0, x1, y0, y1) in enumerate([
            (0, 5, 0, 5), (5, 10, 0, 5), (0, 5, 5, 10), (5, 10, 5, 10)
        ]):
            ax.fill_between([x0, x1], y0, y1, alpha=0.05, color=AGENT_COLORS[i])
        ax.axvline(5, color=grid_color, linestyle="--", linewidth=1)
        ax.axhline(5, color=grid_color, linestyle="--", linewidth=1)

    # Agent trails
    if trails:
        for aid, trail in trails.items():
            if len(trail) > 1:
                idx = int(aid.split("_")[1]) if "_" in aid else 0
                color = AGENT_COLORS[idx % len(AGENT_COLORS)]
                xs = [p[0] for p in trail]
                ys = [p[1] for p in trail]
                ax.plot(xs, ys, color=color, alpha=0.25, linewidth=1.2, zorder=1)

    # Cards
    unpicked = [c for c in cards if not c["picked_up"]]
    picked = [c for c in cards if c["picked_up"]]

    # Draw picked cards (faded)
    for c in picked:
        ax.scatter(c["x"], c["y"], c=picked_color, marker="o", s=30, alpha=0.3, zorder=1)

    # Draw unpicked cards (colored by suit)
    for suit, color in SUIT_COLORS.items():
        sc = [c for c in unpicked if c["suit"] == suit]
        if sc:
            ax.scatter(
                [c["x"] for c in sc], [c["y"] for c in sc],
                c=color, marker=SUIT_MARKERS[suit], s=70, alpha=0.85,
                edgecolors="white", linewidths=0.5, zorder=2,
            )

    # Agent positions
    if agent_positions:
        for aid, (ax_, ay) in agent_positions.items():
            idx = int(aid.split("_")[1]) if "_" in aid else 0
            color = AGENT_COLORS[idx % len(AGENT_COLORS)]
            ax.plot(ax_, ay, "o", color=color, markersize=14, markeredgecolor="white",
                    markeredgewidth=2, zorder=5)
            label = aid.split("_")[1]
            if scoreboard and aid in scoreboard:
                label = f"{scoreboard[aid]}"
            ax.annotate(label, (ax_, ay), ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white", zorder=6)

    # Verifier station
    if show_verifier:
        from card_pickup import VERIFIER_X, VERIFIER_Y
        ax.plot(VERIFIER_X, VERIFIER_Y, "*", color="#f1c40f", markersize=18,
                markeredgecolor="#d4ac0d", markeredgewidth=1.5, zorder=4)
        ax.annotate("V", (VERIFIER_X, VERIFIER_Y), ha="center", va="center",
                    fontsize=7, fontweight="bold", color="#7d6608", zorder=4)

    ax.set_xlim(-0.3, 10.3)
    ax.set_ylim(-0.3, 10.3)
    ax.set_aspect("equal")
    fig.suptitle(title, fontsize=12, fontweight="bold", color=title_color)
    ax.grid(True, alpha=0.15, linestyle="--")
    ax.set_xticks(range(11))
    ax.set_yticks(range(11))

    # Row 1: suit markers with Unicode labels
    card_handles = [
        plt.Line2D([0], [0], marker=SUIT_MARKERS[s], color="w",
                   markerfacecolor=c, markersize=8, label=SUIT_LABELS[s],
                   markeredgecolor="white", markeredgewidth=0.5)
        for s, c in SUIT_COLORS.items()
    ]
    leg1 = ax.legend(handles=card_handles, loc="upper center", bbox_to_anchor=(0.5, -0.06),
                     ncol=4, fontsize=7, frameon=False)
    # Color legend text for dark mode
    if DARK_MODE:
        for text in leg1.get_texts():
            text.set_color("#fafafa")

    # Row 2: verifier + agents (always 5 items for consistent layout)
    n = num_agents or (len(agent_positions) if agent_positions else 0)
    agent_handles = [
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#f1c40f",
                   markersize=10, label="Verifier", markeredgecolor="#d4ac0d",
                   markeredgewidth=0.5)
    ]
    for i in range(4):  # always show 4 agent slots
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        if i < n:
            agent_handles.append(
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                           markersize=8, label=f"Agent {i}",
                           markeredgecolor="white", markeredgewidth=1)
            )
        else:
            # Invisible placeholder to keep layout stable
            agent_handles.append(
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="w",
                           markersize=8, label=f"Agent {i}",
                           markeredgecolor="w", markeredgewidth=0, alpha=0)
            )
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=agent_handles, loc="upper center", bbox_to_anchor=(0.5, -0.14),
                     ncol=5, fontsize=7, frameon=False)
    # Hide placeholder labels
    for i, text in enumerate(leg2.get_texts()):
        if DARK_MODE:
            text.set_color("#fafafa")
        if i > n:  # after verifier + active agents
            text.set_alpha(0)
    for i, handle in enumerate(leg2.legend_handles):
        if i > n:
            handle.set_alpha(0)

    fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.22)
    return fig


def plot_benchmark_results(results: dict) -> plt.Figure:
    """Bar chart of benchmark timing results."""
    patterns = list(results.keys())
    configs = [1, 2, 4]

    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_plot_theme(fig, ax)
    x = range(len(patterns))
    width = 0.25

    for i, n in enumerate(configs):
        times = [results[p][n] for p in patterns]
        ax.bar([xi + i * width for xi in x], times, width,
               label=f"{n} agent{'s' if n > 1 else ''}",
               color=AGENT_COLORS[i], alpha=0.85)

    ax.set_xlabel("Scatter Pattern")
    ax.set_ylabel("Time (seconds)")
    title_color = "#fafafa" if DARK_MODE else "#000000"
    ax.set_title("Benchmark Results: Time by Pattern and Agent Count",
                 fontweight="bold", color=title_color)
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels([p.replace("_", "\n") for p in patterns], fontsize=9)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    return fig


def plot_compare(cards: List[Card], configs: List[int]) -> plt.Figure:
    """Side-by-side comparison of different agent counts on the same pattern."""
    n_configs = len(configs)
    fig, axes = plt.subplots(1, n_configs, figsize=(4 * n_configs, 4), dpi=80)
    if n_configs == 1:
        axes = [axes]

    for ax, n in zip(axes, configs):
        _apply_plot_theme(fig, ax)
        steps = simulate_pickup_steps([dict(c) for c in cards], n)
        final = steps[-1] if steps else None

        # Compute stats
        agent_stats = {}
        for s in steps:
            for evt in s.get("round_events", []):
                aid = evt["agent"]
                if aid not in agent_stats:
                    agent_stats[aid] = {"cards": 0, "total_dist": 0.0}
                agent_stats[aid]["total_dist"] += evt["distance"]
                if evt["phase"] == "pickup":
                    agent_stats[aid]["cards"] += 1

        total_dist = sum(d["total_dist"] for d in agent_stats.values())

        if final:
            for suit, color in SUIT_COLORS.items():
                sc = [c for c in final["cards"] if c["suit"] == suit]
                if sc:
                    ax.scatter([c["x"] for c in sc], [c["y"] for c in sc],
                               c=color, marker=SUIT_MARKERS[suit], s=40, alpha=0.3,
                               edgecolors="white", linewidths=0.3, zorder=1)

            # Draw trails
            trails = _build_trails(steps)
            for aid, trail in trails.items():
                if len(trail) > 1:
                    idx = int(aid.split("_")[1]) if "_" in aid else 0
                    c = AGENT_COLORS[idx % len(AGENT_COLORS)]
                    ax.plot([p[0] for p in trail], [p[1] for p in trail],
                            color=c, alpha=0.4, linewidth=1.5)

        from card_pickup import VERIFIER_X, VERIFIER_Y
        ax.plot(VERIFIER_X, VERIFIER_Y, "*", color="#f1c40f", markersize=14,
                markeredgecolor="#d4ac0d", markeredgewidth=1.2, zorder=4)

        ax.set_xlim(-0.3, 10.3)
        ax.set_ylim(-0.3, 10.3)
        ax.set_aspect("equal")
        title_color = "#fafafa" if DARK_MODE else "#000000"
        ax.set_title(f"{n} Agent{'s' if n > 1 else ''}\nDist: {total_dist:.1f} | Steps: {len(steps)}",
                      fontsize=11, fontweight="bold", color=title_color)
        ax.grid(True, alpha=0.15, linestyle="--")
        ax.set_xticks(range(11))
        ax.set_yticks(range(11))

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Simulation logic (no travel sleep for web app responsiveness)
# ---------------------------------------------------------------------------

def simulate_pickup_steps(
    cards: List[Card], num_agents: int
) -> List[Dict]:
    """Run greedy pickup step-by-step, returning a list of step snapshots.

    Agents start delivering to the verifier as soon as their territory is empty,
    even while other agents are still picking up. This interleaves pickup and
    delivery steps naturally.
    """
    from card_pickup import VERIFIER_X, VERIFIER_Y
    cards = [dict(c) for c in cards]  # deep copy
    steps = []

    # Partition cards into regions
    agent_ids = [f"agent_{i}" for i in range(num_agents)]
    region_cards: Dict[str, List[int]] = {aid: [] for aid in agent_ids}
    for idx, card in enumerate(cards):
        region_id = _determine_region(card, num_agents)
        region_id = min(region_id, num_agents - 1)
        region_cards[f"agent_{region_id}"].append(idx)

    positions = {aid: [0.0, 0.0] for aid in agent_ids}
    agent_remaining = {aid: list(idxs) for aid, idxs in region_cards.items()}

    # Track agent states: "picking", "delivering", "done"
    agent_state = {aid: "picking" for aid in agent_ids}
    agent_card_counts = {aid: 0 for aid in agent_ids}
    # Delivery interpolation state
    delivery_start_pos = {}  # aid -> (x, y) when delivery began
    delivery_progress = {}   # aid -> current step (0..delivery_steps)
    delivery_steps_per_agent = 8
    delivered = 0

    while any(s != "done" for s in agent_state.values()):
        # Process all agents for this round, then emit one snapshot
        round_events = []

        for aid in agent_ids:
            if agent_state[aid] == "picking":
                remaining = agent_remaining[aid]
                if not remaining:
                    # Territory empty — start delivering
                    agent_state[aid] = "delivering"
                    delivery_start_pos[aid] = tuple(positions[aid])
                    delivery_progress[aid] = 0
                    agent_card_counts[aid] = sum(
                        1 for c in cards if c["picked_up"] and c["picked_up_by"] == aid
                    )

                if agent_state[aid] == "picking" and remaining:
                    # Pick nearest card
                    px, py = positions[aid]
                    best_idx = None
                    best_dist = float("inf")
                    best_ri = None
                    for ri, cidx in enumerate(remaining):
                        c = cards[cidx]
                        d = math.hypot(c["x"] - px, c["y"] - py)
                        if d < best_dist:
                            best_dist = d
                            best_idx = cidx
                            best_ri = ri

                    if best_idx is not None:
                        card = cards[best_idx]
                        card["picked_up"] = True
                        card["picked_up_by"] = aid
                        positions[aid] = [card["x"], card["y"]]
                        remaining.pop(best_ri)
                        round_events.append({
                            "agent": aid,
                            "card": _card_key(card),
                            "distance": round(best_dist, 2),
                            "phase": "pickup",
                        })

            if agent_state[aid] == "delivering":
                delivery_progress[aid] += 1
                t = delivery_progress[aid]
                sx, sy = delivery_start_pos[aid]
                dist = math.hypot(VERIFIER_X - sx, VERIFIER_Y - sy)
                frac = min(t / delivery_steps_per_agent, 1.0)
                ix = sx + (VERIFIER_X - sx) * frac
                iy = sy + (VERIFIER_Y - sy) * frac
                positions[aid] = [ix, iy]

                arrived = t >= delivery_steps_per_agent
                if arrived:
                    delivered += agent_card_counts[aid]
                    agent_state[aid] = "done"

                round_events.append({
                    "agent": aid,
                    "card": f"delivered {agent_card_counts[aid]} cards" if arrived
                            else "traveling to verifier",
                    "distance": round(dist / delivery_steps_per_agent, 2),
                    "phase": "delivery",
                })

        # Emit one snapshot per round with all agents' combined positions
        if round_events:
            picked_count = sum(1 for c in cards if c["picked_up"])
            any_delivering = any(agent_state[a] in ("delivering", "done") for a in agent_ids)
            phase = "delivery" if any_delivering and picked_count == 52 else "pickup"
            primary = round_events[0]
            steps.append({
                "cards": [dict(c) for c in cards],
                "positions": {a: tuple(p) for a, p in positions.items()},
                "picked": picked_count,
                "agent": primary["agent"],
                "card": primary["card"],
                "distance": sum(e["distance"] for e in round_events),
                "phase": phase,
                "delivered": delivered,
                "round_events": round_events,
            })

    return steps


def _build_trails(steps: List[Dict]) -> Dict[str, List[Tuple[float, float]]]:
    """Extract agent movement trails from step history."""
    trails: Dict[str, List[Tuple[float, float]]] = {}
    for step in steps:
        active_agents = {evt["agent"] for evt in step.get("round_events", [])}
        for aid in active_agents:
            pos = step["positions"].get(aid)
            if pos is None:
                continue
            if aid not in trails:
                trails[aid] = []
            if not trails[aid] or trails[aid][-1] != pos:
                trails[aid].append(pos)
    return trails


def run_benchmark_fast() -> dict:
    """Run benchmarks without travel sleep for web responsiveness."""
    results = {}
    for name, pattern_fn in PATTERNS.items():
        cards = pattern_fn()
        results[name] = {}
        for n in [1, 2, 4]:
            state = _make_initial_state(n)
            state["cards"] = [dict(c) for c in cards]
            state["phase"] = "pickup"

            from langgraph.graph import END, START, StateGraph
            from card_pickup import (
                timer_start_node, pickup_node, delivery_node, timer_stop_node,
            )

            builder = StateGraph(AppState)
            builder.add_node("timer_start", timer_start_node)
            builder.add_node("pickup", pickup_node)
            builder.add_node("delivery", delivery_node)
            builder.add_node("timer_stop", timer_stop_node)
            builder.add_node("verify", verify_node)
            builder.add_edge(START, "timer_start")
            builder.add_edge("timer_start", "pickup")
            builder.add_edge("pickup", "delivery")
            builder.add_edge("delivery", "timer_stop")
            builder.add_edge("timer_stop", "verify")
            builder.add_edge("verify", END)
            mini = builder.compile()
            final = mini.invoke(state)
            elapsed = _extract_elapsed(final)
            results[name][n] = elapsed
    return results


# ---------------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="52 Card Pickup",
    page_icon="\U0001f0cf",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Read query params for shareable links
qp = st.query_params
if "seed" in qp and "seed_input" not in st.session_state:
    try:
        st.session_state["seed_input"] = int(qp["seed"])
    except (ValueError, TypeError):
        pass
if "agents" in qp and "qp_agents" not in st.session_state:
    try:
        st.session_state["qp_agents"] = int(qp["agents"])
    except (ValueError, TypeError):
        pass
if "pattern" in qp and "qp_pattern" not in st.session_state:
    st.session_state["qp_pattern"] = qp["pattern"]

st.title("\U0001f0cf 52 Card Pickup — Multi-Agent Simulation")
st.markdown("*The canonical hello world for multi-agent LLM systems*")

# How it works expander
with st.expander("How it works"):
    st.markdown("""
**52 playing cards** are scattered on a 10x10 grid. Agents fan out to pick them up, then converge on a central verifier to deliver their cards.

**Pipeline:** Scatter → Timer Start → Pickup (fan-out) → Delivery (converge) → Timer Stop → Verify

**The tradeoff:** More agents pick up faster (smaller territories), but spend more time traveling back to the verifier. This makes the optimal agent count depend on the scatter pattern — clustered cards favor fewer agents, spread-out cards favor more.

Built with [LangGraph](https://github.com/langchain-ai/langgraph) and [Claude](https://www.anthropic.com/claude). [Source code](https://github.com/violethawk/Fifty-Two-Card-Pickup).
""")

tab1, tab2, tab3, tab4 = st.tabs([
    "Interactive Simulation", "Compare Agents", "Benchmark Suite", "Pattern Gallery",
])

# ---- Tab 1: Interactive Simulation ----
with tab1:
    col_config, col_viz = st.columns([1, 2])

    with col_config:
        st.subheader("Configuration")
        source = st.radio("Card source", ["Random scatter", "Benchmark pattern"])

        if source == "Benchmark pattern":
            default_pattern = st.session_state.pop("qp_pattern", None)
            pattern_list = list(PATTERNS.keys())
            pattern_idx = pattern_list.index(default_pattern) if default_pattern in pattern_list else 0
            pattern_name = st.selectbox("Pattern", pattern_list, index=pattern_idx)
        else:
            pattern_name = None

        default_agents = st.session_state.pop("qp_agents", 2)
        agent_options = [1, 2, 4]
        default_agents = default_agents if default_agents in agent_options else 2
        num_agents = st.select_slider("Pickup agents", options=agent_options,
                                       value=default_agents)

        def _randomize_seed():
            st.session_state["seed_input"] = random.randint(0, 9999)

        seed_col1, seed_col2 = st.columns([3, 1])
        with seed_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.button("\U0001f3b2", help="Randomize seed", on_click=_randomize_seed)
        with seed_col1:
            if "seed_input" not in st.session_state:
                st.session_state["seed_input"] = 42
            seed = st.number_input("Random seed", min_value=0, max_value=9999,
                                   key="seed_input")

        speed = st.slider("Animation speed", min_value=1, max_value=4, value=2,
                          help="Cards picked per frame")
        show_trails = st.checkbox("Show agent trails", value=True)

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            run_btn = st.button("Run Simulation", type="primary")
        with btn_col2:
            replay_btn = st.button("Replay",
                                   disabled="sim_steps" not in st.session_state)

        # Shareable link
        link_params = {"seed": int(seed), "agents": num_agents}
        if pattern_name:
            link_params["pattern"] = pattern_name
        share_url = f"?{urlencode(link_params)}"
        st.caption(f"[Shareable link]({share_url}) for this configuration")

    with col_viz:
        viz_placeholder = st.empty()
        progress_placeholder = st.empty()
        # Live scoreboard placeholder
        scoreboard_placeholder = st.empty()
        stats_placeholder = st.empty()
        log_placeholder = st.empty()

        def _get_cards():
            if source == "Benchmark pattern" and pattern_name:
                return PATTERNS[pattern_name]()
            else:
                random.seed(seed)
                state = _make_initial_state(num_agents)
                state = scatter_node(state)
                return state["cards"]

        def _animate(steps, speed, num_agents, show_trails):
            progress_bar = progress_placeholder.progress(0, text="Picking up cards...")
            event_log_lines = []
            trails = {} if show_trails else None
            trail_cursor = 0  # tracks how far we've built trails
            # Per-agent card count for live scoreboard
            live_scores = {}
            score_cursor = 0

            for i in range(0, len(steps), speed):
                step = steps[min(i + speed - 1, len(steps) - 1)]
                phase = step.get("phase", "pickup")
                picked = step["picked"]
                delivered = step.get("delivered", 0)

                if phase == "pickup":
                    title = f"Fan Out \u2014 {picked}/52 cards picked"
                else:
                    title = f"Converge \u2014 {delivered}/52 cards delivered to verifier"

                # Update live scores (no double-counting)
                for j in range(score_cursor, min(i + speed, len(steps))):
                    s = steps[j]
                    for evt in s.get("round_events", []):
                        if evt["phase"] == "pickup":
                            aid = evt["agent"]
                            live_scores[aid] = live_scores.get(aid, 0) + 1
                score_cursor = min(i + speed, len(steps))

                # Build trails incrementally — only add positions for agents
                # that actually acted in each round
                if show_trails:
                    for j in range(trail_cursor, min(i + speed, len(steps))):
                        s = steps[j]
                        active_agents = {evt["agent"] for evt in s.get("round_events", [])}
                        for aid in active_agents:
                            pos = s["positions"].get(aid)
                            if pos is None:
                                continue
                            if aid not in trails:
                                trails[aid] = []
                            if not trails[aid] or trails[aid][-1] != pos:
                                trails[aid].append(pos)
                    trail_cursor = min(i + speed, len(steps))

                fig = plot_grid(
                    step["cards"],
                    agent_positions=step["positions"],
                    title=title,
                    show_regions=num_agents,
                    trails=trails,
                    num_agents=num_agents,
                    scoreboard=live_scores,
                )
                viz_placeholder.pyplot(fig, width="content")
                plt.close(fig)

                # Live scoreboard
                if live_scores:
                    score_parts = []
                    for idx, aid in enumerate(sorted(live_scores.keys())):
                        color = AGENT_COLORS[idx % len(AGENT_COLORS)]
                        count = live_scores.get(aid, 0)
                        score_parts.append(
                            f"<span style='color:{color};font-weight:bold'>"
                            f"Agent {idx}</span>: {count}"
                        )
                    scoreboard_placeholder.markdown(
                        " &nbsp;&nbsp; ".join(score_parts),
                        unsafe_allow_html=True,
                    )

                # Progress bar — track both pickup and delivery
                if picked < 52:
                    frac = picked / 52 * 0.7  # pickup is 0-70%
                    progress_bar.progress(max(frac, 0.01),
                                          text=f"Picking up: {picked}/52")
                else:
                    frac = 0.7 + (delivered / 52) * 0.3  # delivery is 70-100%
                    progress_bar.progress(min(frac, 1.0),
                                          text=f"Delivering: {delivered}/52 cards to verifier")

                # Accumulate and display log live (skip "traveling" noise)
                for j in range(i, min(i + speed, len(steps))):
                    s = steps[j]
                    for evt in s.get("round_events", []):
                        if evt["phase"] == "delivery":
                            if evt["card"].startswith("delivered"):
                                event_log_lines.append(
                                    f"`{evt['agent']}` \u2014 {evt['card']}"
                                )
                        else:
                            event_log_lines.append(
                                f"`{evt['agent']}` picked up **{evt['card']}** (dist: {evt['distance']})"
                            )

                # Update log live (show last 12 lines)
                with log_placeholder.expander("Event Log", expanded=False):
                    st.markdown("\n".join(event_log_lines[-12:]))

                time.sleep(0.05)

            progress_bar.progress(1.0, text="All 52 cards delivered and verified!")

            # Final stats
            agent_stats = {}
            for s in steps:
                for evt in s.get("round_events", []):
                    aid = evt["agent"]
                    if aid not in agent_stats:
                        agent_stats[aid] = {"cards": 0, "pickup_dist": 0.0, "delivery_dist": 0.0}
                    if evt["phase"] == "delivery":
                        agent_stats[aid]["delivery_dist"] += evt["distance"]
                    else:
                        agent_stats[aid]["cards"] += 1
                        agent_stats[aid]["pickup_dist"] += evt["distance"]

            stats_md = "### Results\n\n"
            stats_md += "| Agent | Cards | Pickup Dist | Delivery Dist | Total Dist |\n"
            stats_md += "|-------|-------|-------------|---------------|------------|\n"
            for aid in sorted(agent_stats.keys()):
                d = agent_stats[aid]
                total = d["pickup_dist"] + d["delivery_dist"]
                stats_md += (f"| {aid} | {d['cards']} | {d['pickup_dist']:.1f} "
                             f"| {d['delivery_dist']:.1f} | {total:.1f} |\n")
            stats_placeholder.markdown(stats_md)

            # Final log — show all events
            with log_placeholder.expander(f"Event Log ({len(event_log_lines)} events)", expanded=False):
                st.markdown("\n".join(event_log_lines))

        def _scatter_animation(cards, num_agents):
            """Animate cards being 'thrown' from center to final positions."""
            n_frames = 10
            cx, cy = 5.0, 5.0  # center of grid
            for frame in range(n_frames + 1):
                t = frame / n_frames
                # Ease-out: fast start, slow finish (like a throw)
                ease = 1 - (1 - t) ** 3
                frame_cards = []
                for c in cards:
                    fc = dict(c)
                    fc["x"] = cx + (c["x"] - cx) * ease
                    fc["y"] = cy + (c["y"] - cy) * ease
                    frame_cards.append(fc)
                title = "Scattering cards..." if frame < n_frames else "Ready to run"
                fig = plot_grid(frame_cards, title=title,
                                show_regions=num_agents, num_agents=num_agents)
                viz_placeholder.pyplot(fig, width="content")
                plt.close(fig)
                time.sleep(0.04)

        # Show initial grid
        if not run_btn and not replay_btn:
            display_cards = _get_cards()
            fig = plot_grid(display_cards, title="Ready to run", show_regions=num_agents,
                           num_agents=num_agents)
            viz_placeholder.pyplot(fig, width="content")
            plt.close(fig)

        # Run simulation
        if run_btn:
            cards = _get_cards()

            _scatter_animation(cards, num_agents)
            steps = simulate_pickup_steps(cards, num_agents)
            st.session_state["sim_steps"] = steps
            st.session_state["sim_agents"] = num_agents
            st.session_state["sim_show_trails"] = show_trails
            _animate(steps, speed, num_agents, show_trails)

        # Replay cached simulation
        if replay_btn and "sim_steps" in st.session_state:
            steps = st.session_state["sim_steps"]
            n = st.session_state.get("sim_agents", num_agents)
            trails_on = st.session_state.get("sim_show_trails", show_trails)
            _animate(steps, speed, n, trails_on)


# ---- Tab 2: Compare Agents ----
with tab2:
    st.subheader("Compare Agent Configurations")
    st.markdown("Run the same scatter pattern with different agent counts side-by-side.")

    cmp_col1, cmp_col2 = st.columns([1, 3])

    with cmp_col1:
        cmp_source = st.radio("Card source", ["Random scatter", "Benchmark pattern"],
                              key="cmp_source")
        if cmp_source == "Benchmark pattern":
            cmp_pattern = st.selectbox("Pattern", list(PATTERNS.keys()), key="cmp_pattern")
        else:
            cmp_pattern = None

        cmp_seed = st.number_input("Random seed", value=42, min_value=0, max_value=9999,
                                   key="cmp_seed")
        cmp_configs = st.multiselect("Agent counts to compare",
                                     options=[1, 2, 4], default=[1, 2, 4])

        cmp_run = st.button("Compare", type="primary", key="cmp_run")

    with cmp_col2:
        if cmp_run and cmp_configs:
            if cmp_source == "Benchmark pattern" and cmp_pattern:
                cmp_cards = PATTERNS[cmp_pattern]()
            else:
                random.seed(cmp_seed)
                state = _make_initial_state(max(cmp_configs))
                state = scatter_node(state)
                cmp_cards = state["cards"]

            with st.spinner("Running comparisons..."):
                fig = plot_compare(cmp_cards, sorted(cmp_configs))
            st.pyplot(fig, width="content")
            plt.close(fig)

            # Stats table
            st.markdown("### Comparison")
            table_md = "| Agents | Steps | Total Distance |\n"
            table_md += "|--------|-------|----------------|\n"
            for n in sorted(cmp_configs):
                steps = simulate_pickup_steps([dict(c) for c in cmp_cards], n)
                total_dist = 0.0
                for s in steps:
                    for evt in s.get("round_events", []):
                        total_dist += evt["distance"]
                table_md += f"| {n} | {len(steps)} | {total_dist:.1f} |\n"
            st.markdown(table_md)
        elif not cmp_run:
            st.info("Select agent counts and click 'Compare' to see side-by-side results.")


# ---- Tab 3: Benchmark Suite ----
with tab3:
    st.subheader("Benchmark Suite")
    st.markdown("Run all 6 scatter patterns with 1, 2, and 4 agents.")

    if st.button("Run Benchmarks", type="primary"):
        with st.spinner("Running benchmarks (this takes ~15 seconds)..."):
            results = run_benchmark_fast()

        # Results table
        st.markdown("### Results")
        table_md = "| Pattern | 1 Agent | 2 Agents | 4 Agents | Best |\n"
        table_md += "|---------|---------|----------|----------|------|\n"
        for name, times in results.items():
            best_n = min(times, key=times.get)
            row = f"| `{name}` "
            for n in [1, 2, 4]:
                marker = " **\\***" if n == best_n else ""
                row += f"| {times[n]:.4f}s{marker} "
            row += f"| {best_n} agent{'s' if best_n > 1 else ''} |"
            table_md += row + "\n"
        st.markdown(table_md)

        # Chart
        fig = plot_benchmark_results(results)
        st.pyplot(fig)
        plt.close(fig)

    else:
        st.info("Click 'Run Benchmarks' to start. No API key needed.")


# ---- Tab 4: Pattern Gallery ----
with tab4:
    st.subheader("Scatter Pattern Gallery")
    st.markdown("The 6 standardized benchmark patterns. Each is fully deterministic.")

    cols = st.columns(3)
    for idx, (name, pattern_fn) in enumerate(PATTERNS.items()):
        cards = pattern_fn()
        with cols[idx % 3]:
            fig = plot_grid(cards, title=name.replace("_", " ").title())
            st.pyplot(fig)
            plt.close(fig)
