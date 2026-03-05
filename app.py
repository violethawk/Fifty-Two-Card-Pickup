"""
Streamlit web app for 52 Card Pickup.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Tuple

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
    "hearts": "h",
    "diamonds": "D",
    "clubs": "s",
    "spades": "^",
}

AGENT_COLORS = ["#3498db", "#e74c3c", "#27ae60", "#9b59b6"]

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_grid(
    cards: List[Card],
    agent_positions: Dict[str, Tuple[float, float]] | None = None,
    title: str = "Card Grid",
    show_regions: int = 0,
) -> plt.Figure:
    """Render the 10x10 grid with cards and optional agent positions."""
    fig, ax = plt.subplots(figsize=(5, 5))

    # Region shading
    if show_regions == 2:
        ax.axvspan(0, 5, alpha=0.05, color=AGENT_COLORS[0])
        ax.axvspan(5, 10, alpha=0.05, color=AGENT_COLORS[1])
        ax.axvline(5, color="#cccccc", linestyle="--", linewidth=1)
    elif show_regions == 4:
        for i, (x0, x1, y0, y1) in enumerate([
            (0, 5, 0, 5), (5, 10, 0, 5), (0, 5, 5, 10), (5, 10, 5, 10)
        ]):
            ax.fill_between([x0, x1], y0, y1, alpha=0.05, color=AGENT_COLORS[i])
        ax.axvline(5, color="#cccccc", linestyle="--", linewidth=1)
        ax.axhline(5, color="#cccccc", linestyle="--", linewidth=1)

    # Cards
    unpicked = [c for c in cards if not c["picked_up"]]
    picked = [c for c in cards if c["picked_up"]]

    # Draw picked cards (faded)
    for c in picked:
        ax.scatter(c["x"], c["y"], c="#cccccc", marker="o", s=30, alpha=0.3, zorder=1)

    # Draw unpicked cards (colored by suit)
    for suit, color in SUIT_COLORS.items():
        sc = [c for c in unpicked if c["suit"] == suit]
        if sc:
            ax.scatter(
                [c["x"] for c in sc], [c["y"] for c in sc],
                c=color, marker=SUIT_MARKERS[suit], s=70, alpha=0.85,
                edgecolors="white", linewidths=0.5, label=suit.capitalize(), zorder=2,
            )

    # Agent positions
    if agent_positions:
        for aid, (ax_, ay) in agent_positions.items():
            idx = int(aid.split("_")[1]) if "_" in aid else 0
            color = AGENT_COLORS[idx % len(AGENT_COLORS)]
            ax.plot(ax_, ay, "o", color=color, markersize=14, markeredgecolor="white",
                    markeredgewidth=2, zorder=5)
            ax.annotate(aid.split("_")[1], (ax_, ay), ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white", zorder=6)

    ax.set_xlim(-0.3, 10.3)
    ax.set_ylim(-0.3, 10.3)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.15, linestyle="--")
    ax.set_xticks(range(11))
    ax.set_yticks(range(11))

    # Legend for suits
    if unpicked:
        handles = [mpatches.Patch(color=c, label=s.capitalize())
                   for s, c in SUIT_COLORS.items()]
        ax.legend(handles=handles, loc="upper right", fontsize=8)

    plt.tight_layout()
    return fig


def plot_benchmark_results(results: dict) -> plt.Figure:
    """Bar chart of benchmark timing results."""
    patterns = list(results.keys())
    configs = [1, 2, 4]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(patterns))
    width = 0.25

    for i, n in enumerate(configs):
        times = [results[p][n] for p in patterns]
        bars = ax.bar([xi + i * width for xi in x], times, width,
                      label=f"{n} agent{'s' if n > 1 else ''}",
                      color=AGENT_COLORS[i], alpha=0.85)

    ax.set_xlabel("Scatter Pattern")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Benchmark Results: Time by Pattern and Agent Count", fontweight="bold")
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels([p.replace("_", "\n") for p in patterns], fontsize=9)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Simulation logic (no travel sleep for web app responsiveness)
# ---------------------------------------------------------------------------

def simulate_pickup_steps(
    cards: List[Card], num_agents: int
) -> List[Dict]:
    """Run greedy pickup step-by-step, returning a list of step snapshots."""
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

    # Round-robin: each agent picks one card per step
    any_remaining = True
    while any_remaining:
        any_remaining = False
        for aid in agent_ids:
            remaining = agent_remaining[aid]
            if not remaining:
                continue
            any_remaining = True

            # Find nearest among this agent's remaining cards
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

                picked_count = sum(1 for c in cards if c["picked_up"])
                steps.append({
                    "cards": [dict(c) for c in cards],
                    "positions": {a: tuple(p) for a, p in positions.items()},
                    "picked": picked_count,
                    "agent": aid,
                    "card": _card_key(card),
                    "distance": round(best_dist, 2),
                })

    return steps


def run_benchmark_fast() -> dict:
    """Run benchmarks without travel sleep for web responsiveness."""
    results = {}
    for name, pattern_fn in PATTERNS.items():
        cards = pattern_fn()
        results[name] = {}
        for n in [1, 2, 4]:
            # Time the pickup (no sleep)
            work_cards = [dict(c) for c in cards]
            steps = simulate_pickup_steps(work_cards, n)
            # Measure by total distance as proxy
            total_dist = sum(s["distance"] for s in steps)
            # Actually time it
            start = time.perf_counter()
            graph = build_graph(with_supervisor=False, llm_pickup=False)
            state = _make_initial_state(n)
            state["cards"] = [dict(c) for c in cards]
            state["phase"] = "pickup"

            from langgraph.graph import END, START, StateGraph
            from card_pickup import timer_start_node, pickup_node, timer_stop_node

            builder = StateGraph(AppState)
            builder.add_node("timer_start", timer_start_node)
            builder.add_node("pickup", pickup_node)
            builder.add_node("timer_stop", timer_stop_node)
            builder.add_node("verify", verify_node)
            builder.add_edge(START, "timer_start")
            builder.add_edge("timer_start", "pickup")
            builder.add_edge("pickup", "timer_stop")
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
    page_icon="🃏",
    layout="wide",
)

st.title("🃏 52 Card Pickup — Multi-Agent Simulation")
st.markdown("*The canonical hello world for multi-agent LLM systems*")

tab1, tab2, tab3 = st.tabs(["Interactive Simulation", "Benchmark Suite", "Pattern Gallery"])

# ---- Tab 1: Interactive Simulation ----
with tab1:
    col_config, col_viz = st.columns([1, 2])

    with col_config:
        st.subheader("Configuration")
        source = st.radio("Card source", ["Random scatter", "Benchmark pattern"])

        if source == "Benchmark pattern":
            pattern_name = st.selectbox("Pattern", list(PATTERNS.keys()))
        else:
            pattern_name = None

        num_agents = st.select_slider("Pickup agents", options=[1, 2, 4], value=2)
        seed = st.number_input("Random seed", value=42, min_value=0, max_value=9999)
        speed = st.slider("Animation speed", min_value=1, max_value=20, value=8,
                          help="Cards picked per frame")

        run_btn = st.button("Run Simulation", type="primary", use_container_width=True)

    with col_viz:
        viz_placeholder = st.empty()
        stats_placeholder = st.empty()
        log_placeholder = st.empty()

        # Generate initial cards for display
        if "cards" not in st.session_state:
            random.seed(seed)
            state = _make_initial_state(num_agents)
            state = scatter_node(state)
            st.session_state.cards = state["cards"]
            st.session_state.steps = None

        # Show initial grid or pattern
        if not run_btn:
            if source == "Benchmark pattern" and pattern_name:
                display_cards = PATTERNS[pattern_name]()
            else:
                random.seed(seed)
                state = _make_initial_state(num_agents)
                state = scatter_node(state)
                display_cards = state["cards"]

            fig = plot_grid(display_cards, title="Ready to run", show_regions=num_agents)
            viz_placeholder.pyplot(fig)
            plt.close(fig)

        # Run simulation
        if run_btn:
            if source == "Benchmark pattern" and pattern_name:
                cards = PATTERNS[pattern_name]()
            else:
                random.seed(seed)
                state = _make_initial_state(num_agents)
                state = scatter_node(state)
                cards = state["cards"]

            steps = simulate_pickup_steps(cards, num_agents)

            # Animate
            progress_bar = st.progress(0, text="Picking up cards...")
            event_log_lines = []

            for i in range(0, len(steps), speed):
                step = steps[min(i + speed - 1, len(steps) - 1)]
                fig = plot_grid(
                    step["cards"],
                    agent_positions=step["positions"],
                    title=f"Round {i+1} — {step['picked']}/52 cards picked",
                    show_regions=num_agents,
                )
                viz_placeholder.pyplot(fig)
                plt.close(fig)

                progress_bar.progress(
                    step["picked"] / 52,
                    text=f"{step['picked']}/52 cards picked"
                )

                # Accumulate log
                for j in range(i, min(i + speed, len(steps))):
                    s = steps[j]
                    event_log_lines.append(
                        f"`{s['agent']}` picked up **{s['card']}** (dist: {s['distance']})"
                    )

                time.sleep(0.05)

            progress_bar.progress(1.0, text="All 52 cards picked up!")

            # Final stats
            final_step = steps[-1]
            agent_stats = {}
            for s in steps:
                aid = s["agent"]
                if aid not in agent_stats:
                    agent_stats[aid] = {"cards": 0, "distance": 0.0}
                agent_stats[aid]["cards"] += 1
                agent_stats[aid]["distance"] += s["distance"]

            stats_md = "### Results\n\n"
            stats_md += "| Agent | Cards | Total Distance |\n"
            stats_md += "|-------|-------|----------------|\n"
            for aid in sorted(agent_stats.keys()):
                d = agent_stats[aid]
                stats_md += f"| {aid} | {d['cards']} | {d['distance']:.1f} |\n"
            stats_placeholder.markdown(stats_md)

            # Show last N events
            with log_placeholder.expander("Event Log", expanded=False):
                st.markdown("\n".join(event_log_lines[-20:]))


# ---- Tab 2: Benchmark Suite ----
with tab2:
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


# ---- Tab 3: Pattern Gallery ----
with tab3:
    st.subheader("Scatter Pattern Gallery")
    st.markdown("The 6 standardized benchmark patterns. Each is fully deterministic.")

    cols = st.columns(3)
    for idx, (name, pattern_fn) in enumerate(PATTERNS.items()):
        cards = pattern_fn()
        with cols[idx % 3]:
            fig = plot_grid(cards, title=name.replace("_", " ").title())
            st.pyplot(fig)
            plt.close(fig)
